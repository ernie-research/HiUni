# Dense Pixel-DiT Recipes

Training, inference and checkpoint tooling for the dense Qwen3-VL-4B Pixel-DiT.
Run every command from the repository root.

## Checkpoints

| Directory | Contents | Used by |
| --- | --- | --- |
| `checkpoints/Qwen3-VL-4B-Instruct` | Base HF model (config, tokenizer, processor) | everything (`HF_PATH`) |
| `checkpoints/pixel_dit_megatron` | Release Megatron checkpoint (torch_dist, model weights only) | Megatron inference, hot start, export |
| `checkpoints/pixel_dit_hf` | Release HF checkpoint (safetensors, `trust_remote_code`) | `hf_inference/` |

The release weights are an EMA with decay 0.95 over the 81 model-only snapshots
taken every 50 steps from step 4000 to step 8000 of the SFT run. The Megatron
directory holds `iter_0008000/` (weights, `common.pt`, `metadata.json`,
`run_config.yaml`, `provenance.json`) and `latest_checkpointed_iteration.txt`.
It has no optimizer, RNG or dataloader state.

### Building a release checkpoint

```bash
# 1. EMA over the model-only snapshots (CPU only, streams one chunk at a time).
python recipe/transfusion/scripts/average_model_checkpoints.py \
    --input-root output/<run>-model-only --output-root output/<run>-averages \
    --start-step 4000 --end-step 8000 --interval 50 --ema-decays 0.95

# 2. Add common.pt, run_config.yaml and the iteration marker; strip local paths.
python recipe/transfusion/scripts/prepare_release_checkpoint.py \
    --averaged-dir output/<run>-averages/ema_0.95/iter_0008000 \
    --run-config output/<run>/iter_0008000/run_config.yaml \
    --output checkpoints/pixel_dit_megatron
```

## Megatron <-> HF conversion

```bash
# Megatron -> HF (writes config.json, safetensors shards, tokenizer and modeling code)
python hf_inference/export.py \
    --megatron-path checkpoints/pixel_dit_megatron \
    --hf-model checkpoints/Qwen3-VL-4B-Instruct \
    --hf-path checkpoints/pixel_dit_hf

# HF -> Megatron (parameter-only torch_dist checkpoint with run_config.yaml)
python recipe/transfusion/import_hf_to_megatron.py \
    --hf-path checkpoints/pixel_dit_hf \
    --megatron-path checkpoints/pixel_dit_megatron_from_hf
```

The HF model does not carry three groups of Megatron parameters that inference
never reads: the image input projection, the ViT LayerNorms and the ViT
statistics buffers. The import zero-fills them. Every other tensor survives a
Megatron -> HF -> Megatron round trip bit-exactly (741 of 741 compared tensors).
`export_transfusion_ckpt.py` is the lower-level Bridge export that
`hf_inference/export.py` builds on.

## Training

```bash
CONFIG_FILE=recipe/transfusion/conf/transfusion_qwen3_vl_4b_pixel_dit.yaml \
    bash recipe/transfusion/run_transfusion_pretrain.sh
```

The launcher reads `GPUS_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR` and
`MASTER_PORT`, then forwards extra arguments to `pretrain_transfusion_model.py`.

Configurations in `conf/`:

- `transfusion_qwen3_vl_4b_pixel_dit.yaml`: main 4B recipe.
- `transfusion_qwen3_vl_4b_pixel_dit_sft_hotstart.yaml`: continue training from the release checkpoint.
- `transfusion_qwen3_vl_{2b,4b}_pixel_dit_dense_ablation.yaml`: 2B / 4B ablations.
- `transfusion_qwen3_vl_4b_pixel_dit_overfit.yaml`, `..._overfit_i2i.yaml`: small T2I / I2I overfit runs.
- `transfusion_qwen3_vl_4b_pixel_dit_lora.yaml`: LoRA fine-tuning from the release checkpoint.

Data is declared per stream in `data_streams` (each entry points at a
`probs/*.yaml` blend of Energon datasets). `batch_schedule` gives the samples
per global batch for each stream; the values must sum to the global batch size
and each must be a multiple of the micro batch size. `data/convert_t2i.py` and
`data/convert_i2i.py` produce the Energon format from local image files or remote
URLs (see [Data](#data)); unreadable images are skipped.

### Hot start from the release checkpoint

```bash
CONFIG_FILE=recipe/transfusion/conf/transfusion_qwen3_vl_4b_pixel_dit_sft_hotstart.yaml \
    bash recipe/transfusion/run_transfusion_pretrain.sh
```

The config sets `checkpoint.pretrained_checkpoint: ./checkpoints/pixel_dit_megatron`
and uses `./output/pixel_dit_sft_hotstart` as both `load` and `save`. The first
launch finds no checkpoint under `load`, so it loads the release weights as a
fine-tune: iteration 0, fresh optimizer and scheduler. Later launches resume
from `load`, including optimizer state. Replace the `data_streams` entries
(currently the small `probs/test_*.yaml` blends) with your own data. Resuming
loads the full distributed optimizer state, so leave some GPU memory free: the
8-GPU resume was tested; a 4-GPU resume on GPUs that were already nearly full
failed inside the NCCL gather.

### LoRA

```bash
CONFIG_FILE=recipe/transfusion/conf/transfusion_qwen3_vl_4b_pixel_dit_lora.yaml \
    bash recipe/transfusion/run_transfusion_pretrain.sh
```

A top-level `lora:` section turns a run into LoRA training
(`model/transfusion/lora.py`). It has one entry per component, each configured
on its own:

| Component | Modules | Default `target_modules` |
| --- | --- | --- |
| `backbone` | `language_model.*`, `time_token` | `linear_qkv, linear_proj, linear_fc1, linear_fc2` |
| `pixel_dit` | `diffusion_head_a/b`, `pixel_proj`, gates, time embeddings | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| `vit` | `vision_model.*` | `attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2` |

Keys per component: `enabled`, `dim` (rank), `alpha` (scale `alpha / dim`),
`dropout`, `dropout_position` (`pre` / `post`), `target_modules` (leaf name,
dotted suffix or `*` wildcard over the module path), `layers` and
`trainable_modules`. `layers` selects layers of that component's own stack by
0-based index, as a list or a range string (`"0-17"`, `"0,4,8-9"`); `null` means
all layers (36 decoder layers, 10 Pixel-DiT layers, 24 ViT blocks).
`trainable_modules` are trained in full instead of adapted; the example
config trains the small Pixel-DiT output layer (`diffusion_head_b`) and
`pixel_proj` this way. Everything else is frozen, including modules the model
itself freezes. Override from the command line, e.g.
`lora.pixel_dit.dim=128 lora.backbone.layers=18-35 lora.vit.enabled=true`.

LoRA needs `checkpoint.pretrained_checkpoint`; the first launch loads it and
attaches zero-initialized adapters. Checkpoints under `checkpoint.save` contain
only the adapters, the `trainable_modules` and the optimizer state, and later
launches resume from them. Fold them into the base weights before inference or
HF export:

```bash
python recipe/transfusion/scripts/merge_lora_checkpoint.py \
    --base checkpoints/pixel_dit_megatron --lora output/pixel_dit_lora \
    --output output/pixel_dit_lora_merged
CKPT=output/pixel_dit_lora_merged PROMPT="A red fox in fresh snow." \
    bash recipe/transfusion/scripts/run_t2i_pixel_dit.sh
```

The merge runs on one GPU, reads the LoRA settings from the checkpoint's
`run_config.yaml` and writes a parameter-only checkpoint with the base model's
keys, so it works anywhere the release checkpoint does (inference wrappers,
`hf_inference/export.py`, hot start).

## Megatron inference

One wrapper per task in `scripts/`. All of them read `HF_PATH`
(default `./checkpoints/Qwen3-VL-4B-Instruct`), `CKPT`
(default `./checkpoints/pixel_dit_megatron`), `NPROC` and optional `ITERATION`,
take structural hyperparameters from the checkpoint's `run_config.yaml`, and
forward extra flags to the Python entrypoint in `eval/`.

| Task | Wrapper | Entrypoint |
| --- | --- | --- |
| Pure-text chat | `run_text_pixel_dit.sh` | `inference_transfusion_qwen3vl_pixel_dit_text.py` |
| Image understanding | `run_understand_pixel_dit.sh` | `inference_transfusion_qwen3vl_pixel_dit_understand.py` |
| Text-to-image | `run_t2i_pixel_dit.sh` | `inference_transfusion_qwen3vl_pixel_dit.py` |
| Image editing | `run_edit_pixel_dit.sh` | `inference_transfusion_qwen3vl_pixel_dit_i2i.py` |

```bash
PROMPT="Explain flow matching briefly." bash recipe/transfusion/scripts/run_text_pixel_dit.sh
IMAGE=cat.jpg PROMPT="Describe the image." bash recipe/transfusion/scripts/run_understand_pixel_dit.sh
PROMPT="A red fox in fresh snow." bash recipe/transfusion/scripts/run_t2i_pixel_dit.sh
IMAGE=input.jpg PROMPT="Make the sky purple." bash recipe/transfusion/scripts/run_edit_pixel_dit.sh
```

Batch input (`DATA_PATH`, with `NPROC` GPUs):

- text: jsonl `{id, prompt, system?}`, answers to `OUTPUT` (stdout if empty).
- understanding: jsonl `{id, prompt, image | images}`, optional `IMAGE_ROOT`, answers to `OUTPUT`.
- T2I: txt (one prompt per line), jsonl `{id?, prompt | text}` or csv; images under `OUT_DIR`.
- editing: a `datasets.load_from_disk` directory in the GEdit-Bench layout
  (`key`, `instruction`, `input_image`); images under `OUT_DIR`.

The defaults match `hf_inference/`: T2I at 2048 px (`RESOLUTION`, multiple of
32), 25 steps, cfg 4.0, shift 4.0, euler; editing with a 512x512-area reference,
25 steps, shift 4.0, `edit3` with `CFG_W_TEXT=4`, `CFG_W_IMG=1`; chat and
understanding in think mode with a 2048-token think budget. With the same seed,
Megatron and HF editing outputs agree at about 42.6 dB PSNR.

For benchmark sweeps over several checkpoints, use `scripts/batch_eval_pixel_dit.sh`
(OneIG / LongText / QwenImageBench T2I) and `scripts/batch_eval_pixel_dit_i2i.sh`
(GEdit-Bench).

## Data

`data/convert_t2i.py`, `data/convert_i2i.py` and `data/convert_und.py` turn
caption, editing and understanding records into the Energon format read by the
dense Pixel-DiT task encoder; `data/run_local_job_{t2i,i2i}.sh` show local runs.

Images are read from local files by default. A record names its image under
`image`, `image_path`, `path` or `url`; the value can be a local path
(absolute, relative, `~` or `file://`) or an `http(s)://` URL.

```jsonl
{"data_id": "0001", "image": "images/0001.jpg", "caption": "A red fox in fresh snow."}
{"data_id": "e001", "image_info": [{"image": "images/e001_src.jpg", "matched_text_index": 0}, {"image": "images/e001_dst.jpg", "matched_text_index": 1}], "text_list": [{"text": "Make the fox black."}], "order": {"type": ["image", "text", "image"], "index": [0, 0, 1], "mask": [1, 1, 0]}}
```

```bash
bash recipe/transfusion/data/run_local_job_t2i.sh --input /data/t2i/data.jsonl --output-dir ./output/my_t2i
bash recipe/transfusion/data/run_local_job_i2i.sh --input /data/edit/data.jsonl --output-dir ./output/my_i2i
```

Relative paths are resolved against `--image-root` (default: the directory of
the input file) and written into the shards as absolute paths, so every training
node must see the images at those paths. A record whose local image is missing is
dropped at conversion time, with a warning. To move the image directory later, convert with
`--keep-relative-paths` and set `TRANSFUSION_IMAGE_ROOT=/new/image/root` when
training; relative paths in the shards are joined with it.

Remote HTTP(S) URLs are fetched at training time. Purely local datasets need no
network credentials. Shards that embed decoded image arrays in `jpgs` are also
accepted. Any sample whose image cannot be read at training time is skipped.
