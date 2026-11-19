#!/usr/bin/env python3
"""Merge a LoRA training checkpoint into its base Pixel-DiT weights.

LoRA runs (a ``lora:`` section in the training YAML) save only the adapters, the
``trainable_modules`` and the optimizer state.  This tool folds them into the base
checkpoint and writes a full parameter-only Megatron checkpoint that the inference
entrypoints, ``hf_inference/export.py`` and hot start accept like the release:

    python recipe/transfusion/scripts/merge_lora_checkpoint.py \\
        --base checkpoints/pixel_dit_megatron \\
        --lora output/pixel_dit_lora \\
        --output output/pixel_dit_lora_merged

``--lora`` and ``--base`` may be checkpoint roots (the latest iteration is used,
or ``--iteration`` / ``--base-iteration``) or ``iter_XXXXXXX`` directories.  The
LoRA settings are read from the ``peft`` section of the LoRA checkpoint's
``run_config.yaml``; the merged checkpoint gets the base checkpoint's
``run_config.yaml`` (same architecture) and a ``lora_merge.json`` provenance file.
Runs on one GPU.
"""

import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BRIDGE = _PROJECT_ROOT / "3rdparty" / "Megatron-Bridge"
for _p in (_PROJECT_ROOT, _BRIDGE / "src", _BRIDGE / "3rdparty" / "Megatron-LM"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=None,
                        help="base checkpoint the LoRA run started from "
                             "(default: checkpoint.pretrained_checkpoint of the LoRA run)")
    parser.add_argument("--base-iteration", type=int, default=None)
    parser.add_argument("--lora", required=True, type=Path, help="LoRA training checkpoint (checkpoint.save)")
    parser.add_argument("--iteration", type=int, default=None, help="LoRA iteration (default: latest)")
    parser.add_argument("--output", required=True, type=Path, help="output checkpoint root (must be empty)")
    parser.add_argument("--hf-model", type=Path, default=Path("checkpoints/Qwen3-VL-4B-Instruct"),
                        help="base Qwen3-VL HF directory (config and tokenizer)")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float32",
                        help="dtype of the merged weights (the release checkpoint is float32)")
    return parser.parse_args(argv)


def load_lora_tensors(model, ckpt_dir: Path, names):
    """Load ``names`` (adapter and fully trained tensors) from an adapter-only checkpoint."""
    from megatron.core.dist_checkpointing import load as dist_ckpt_load
    from megatron.core.dist_checkpointing.validation import StrictHandling

    sharded = {k: v for k, v in model.sharded_state_dict(prefix="").items() if k in names}
    lost = sorted(set(names) - set(sharded))
    if lost:
        raise KeyError(f"{len(lost)} trainable tensors have no sharded-state-dict entry: {lost[:10]}")
    # Megatron's "unexpected" keys are requested tensors the checkpoint does not have.
    loaded, _, absent = dist_ckpt_load(sharded_state_dict=sharded, checkpoint_dir=str(ckpt_dir),
                                       strict=StrictHandling.RETURN_ALL)
    if absent:
        raise KeyError(f"LoRA checkpoint {ckpt_dir} lacks {len(absent)} tensors: {sorted(absent)[:10]}")
    # The result also carries the checkpoint's common state (iteration, optimizer, ...).
    loaded = {k: v for k, v in loaded.items() if k in names}
    _, unexpected = model.load_state_dict(loaded, strict=False)
    if unexpected:
        raise KeyError(f"Unexpected tensors: {unexpected[:10]}")
    return len(loaded)


def main(argv=None):
    args = parse_args(argv)
    for name, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                        ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "0")):
        os.environ.setdefault(name, value)

    from recipe.transfusion.export_transfusion_ckpt import read_run_config, resolve_iteration_dir

    lora_dir = resolve_iteration_dir(args.lora, args.iteration)
    lora_run_config = read_run_config(lora_dir)
    peft_cfg = lora_run_config.get("peft")
    if not peft_cfg:
        raise SystemExit(f"{lora_dir}/run_config.yaml has no peft section; not a LoRA checkpoint")
    base = args.base or Path(str((lora_run_config.get("checkpoint") or {}).get("pretrained_checkpoint") or ""))
    if not str(base) or not base.exists():
        raise SystemExit(f"Base checkpoint not found ({base!s}); pass --base")
    base_dir = resolve_iteration_dir(base, args.base_iteration)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output must be empty: {out}")

    import model.transfusion.transfusion_qwen3vl_bridge  # noqa: F401  (registers the bridge)
    from megatron.bridge.training.model_load_save import save_megatron_model

    from model.transfusion.lora import TransfusionLoRA, merge_lora_into_model
    from recipe.transfusion.eval.pixel_dit_runtime import cleanup_dist
    from recipe.transfusion.export_transfusion_ckpt import build_megatron_model

    peft = TransfusionLoRA.from_dict(peft_cfg)
    model, _ = build_megatron_model(args.hf_model, base_dir, None, args.dtype)
    base_keys = set(model.state_dict())
    # Reproduce the training-time trainable set: the run's freeze_* flags, then the LoRA selection.
    accepted = inspect.signature(model.freeze).parameters
    model.freeze(**{k: v for k, v in (lora_run_config.get("model") or {}).items() if k in accepted})
    peft(model, training=True)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    count = load_lora_tensors(model, lora_dir, names)
    merged = merge_lora_into_model(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    extra = sorted(set(model.state_dict()) ^ base_keys)
    if extra:
        raise RuntimeError(f"Merged model keys differ from the base model: {extra[:10]}")

    out.mkdir(parents=True, exist_ok=True)
    save_megatron_model([model], out)
    out_dir = resolve_iteration_dir(out)
    src = base_dir / "run_config.yaml"
    shutil.copy2(src if src.is_file() else base_dir.parent / "run_config.yaml", out_dir / "run_config.yaml")
    (out_dir / "lora_merge.json").write_text(json.dumps({
        "base": str(base_dir), "lora": str(lora_dir), "components": peft.enabled_components(),
        "merged_modules": merged, "loaded_tensors": count, "dtype": args.dtype,
    }, indent=2) + "\n")
    print(f"Merged {merged} LoRA modules ({count} tensors from {lora_dir}) into {base_dir}; "
          f"saved to {out_dir}", flush=True)
    cleanup_dist(True)


if __name__ == "__main__":
    main()
