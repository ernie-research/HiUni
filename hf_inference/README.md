# HF / plain-PyTorch Pixel-DiT inference

A standalone `transformers` implementation of the dense Qwen3-VL Pixel-DiT. It needs no Megatron
and no FlashMask: `pip install -r hf_inference/requirements.txt` (torch, transformers 4.57.1,
safetensors, accelerate, Pillow, PyYAML). See the root `README.md` for environment notes.

## Model directory

`hf_inference/export.py` converts a Megatron checkpoint into a self-contained directory
(`model_type: pixel_dit_qwen3_vl`, safetensors shards, tokenizer, and the modeling code for
`trust_remote_code`). Architecture and sampling flags (`predict_x0`, dynamic noise scaling,
gate layer, ...) are read from the checkpoint's `run_config.yaml` and stored in `config.json`.

```bash
python hf_inference/export.py \
    --megatron-path checkpoints/pixel_dit_megatron \
    --hf-model checkpoints/Qwen3-VL-4B-Instruct \
    --hf-path checkpoints/pixel_dit_hf
```

The conversion runs inside this repository (it reuses `recipe/transfusion/export_transfusion_ckpt.py`).
The inverse direction is `recipe/transfusion/import_hf_to_megatron.py`.

## One script per task

| Task | CLI | Shell wrapper |
| --- | --- | --- |
| Pure-text chat | `chat.py` | `scripts/run_hf_chat.sh` |
| Image understanding | `understand.py` | `scripts/run_hf_understand.sh` |
| Text-to-image | `t2i.py` | `scripts/run_hf_t2i.sh` |
| Image editing | `edit.py` | `scripts/run_hf_edit.sh` |
| T2I benchmarks (OneIG / LongText / QwenImageBench) | `generate.py` | - |

The wrappers take environment variables: `MODEL` (default `./checkpoints/pixel_dit_hf`), `NPROC`,
and either a single sample (`PROMPT`, plus `IMAGE` for understanding/editing) or a jsonl file
(`DATA_PATH`, optional `IMAGE_ROOT`, `OUTPUT`). Extra flags are forwarded to the CLI.

```bash
PROMPT="Explain flow matching briefly." bash hf_inference/scripts/run_hf_chat.sh
IMAGE=cat.jpg PROMPT="Describe the image." bash hf_inference/scripts/run_hf_understand.sh
PROMPT="A red fox in fresh snow." bash hf_inference/scripts/run_hf_t2i.sh
IMAGE=input.jpg PROMPT="Make the sky purple." bash hf_inference/scripts/run_hf_edit.sh
DATA_PATH=prompts.jsonl OUTPUT=t2i/ NPROC=8 bash hf_inference/scripts/run_hf_t2i.sh --resume
```

jsonl records:

- chat: `{id?, prompt, system?}`
- understanding: `{id?, prompt, image | images}`
- T2I: `{id?, prompt | caption, height?, width?}` (pixels, multiples of 32)
- editing: `{id?, image | images, prompt | instruction}`

Defaults:

- T2I: 2048x2048, 25 steps, cfg 4.0, shift 4.0, euler.
- Editing: 512x512-area reference, 25 steps, shift 4.0, `edit3` with `w_text=4`, `w_img=1`.
- Understanding and chat: think mode with a 2048-token think budget.

## Python API

```python
from PIL import Image

from hf_inference.runtime import load_model, load_processor, load_tokenizer
from hf_inference.editing import edit_image
from hf_inference.text_decode import TextDecoder

model = load_model("checkpoints/pixel_dit_hf")          # bf16, rejects partial checkpoints
tokenizer = load_tokenizer("checkpoints/pixel_dit_hf")
processor = load_processor("checkpoints/pixel_dit_hf")

image = model.generate_image("A red fox in fresh snow.", tokenizer, height=1024, width=1024,
                             num_steps=25, guidance_scale=4.0, shift=4.0)
edited = edit_image(model, tokenizer, processor, "Make the sky purple.", Image.open("input.jpg"))
result = TextDecoder(model, tokenizer, processor).generate(
    "Describe the image.", images=[Image.open("cat.jpg").convert("RGB")])
print(result["answer"])                                 # result["raw"] keeps the thinking text
```

## Parity with Megatron

On the release checkpoint, outputs were compared against the Megatron entrypoints in
`recipe/transfusion/eval/`:

- Editing: the same seed and settings give images at 42-44 dB PSNR (bf16 noise).
- Chat and understanding: greedy decoding starts with the same tokens, but long outputs can
  diverge after a few dozen tokens where two candidates are nearly tied in bf16. From that
  point the two backends produce different, equally valid continuations.
