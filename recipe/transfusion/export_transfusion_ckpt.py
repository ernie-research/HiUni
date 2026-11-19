#!/usr/bin/env python3
"""Export a dense Pixel-DiT Megatron checkpoint to Megatron-Bridge HF layout.

The output keeps standard Qwen3-VL keys (``model.*``) and stores Pixel-DiT
modules under ``transfusion.*``.  ``hf_inference/export.py`` calls this script
and then packages the result as a standalone ``PixelDiTForConditionalGeneration``
directory; run that wrapper unless you need the raw Bridge layout.

    python recipe/transfusion/export_transfusion_ckpt.py \\
        --megatron-path checkpoints/pixel_dit_megatron \\
        --hf-model checkpoints/Qwen3-VL-4B-Instruct \\
        --hf-path exports/pixel_dit_bridge

``--megatron-path`` may be a checkpoint root (``latest_checkpointed_iteration.txt``
selects the iteration) or an ``iter_XXXXXXX`` directory.  Structural
hyperparameters come from the checkpoint's ``run_config.yaml``.  Parameter-only
checkpoints (EMA / averaged releases) are supported: they lack unused ViT
statistics buffers and TransformerEngine extra state, which are not exported.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE = _PROJECT_ROOT / "3rdparty" / "Megatron-Bridge"
for _p in (_PROJECT_ROOT, _BRIDGE / "src", _BRIDGE / "3rdparty" / "Megatron-LM"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

BRIDGE_ARCHITECTURE = "TransfusionQwen3VLForConditionalGeneration"
DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def resolve_iteration_dir(path: Path, iteration=None) -> Path:
    """Return the iter_XXXXXXX directory holding ``.metadata``."""
    path = path.resolve()
    if (path / ".metadata").is_file():
        return path
    if iteration is None:
        tracker = path / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise FileNotFoundError(f"{path} has neither .metadata nor latest_checkpointed_iteration.txt")
        iteration = int(tracker.read_text().strip())
    target = path / f"iter_{int(iteration):07d}"
    if not (target / ".metadata").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint directory: {target}")
    return target


def read_run_config(iter_dir: Path) -> dict:
    """Load run_config.yaml from the iteration directory or the checkpoint root."""
    import yaml

    for candidate in (iter_dir / "run_config.yaml", iter_dir.parent / "run_config.yaml"):
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    raise FileNotFoundError(f"run_config.yaml not found next to {iter_dir}")


def hf_model_from_run_config(run_config: dict):
    """Best-effort lookup of the base HF model recorded by training."""
    model_cfg = run_config.get("model") or {}
    for value in ((run_config.get("checkpoint") or {}).get("pretrained_checkpoint"),
                  model_cfg.get("hf_model_id"), model_cfg.get("hf_model_path")):
        if value and Path(str(value), "config.json").is_file():
            return Path(str(value))
    return None


def prepare_bridge_hf_dir(hf_model: Path, workdir: Path, overrides=None) -> Path:
    """Mirror ``hf_model`` with a config that dispatches to the Transfusion bridge.

    Weight files are symlinked, everything else is copied, so the patched view
    costs only a few megabytes.  ``overrides`` updates config.json fields.
    """
    hf_model = hf_model.resolve()
    view = workdir / "_bridge_hf_config"
    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True)
    for item in hf_model.iterdir():
        if item.name.startswith(".") or item.is_dir():
            continue
        if item.suffix == ".safetensors" or item.name.endswith(".bin"):
            (view / item.name).symlink_to(item)
        else:
            shutil.copy2(item, view / item.name)
    config_path = view / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(overrides or {})
    if config.get("text_config", {}).get("num_experts"):
        raise ValueError("MoE Qwen3-VL checkpoints are not supported by this release")
    config["model_type"] = "qwen3_vl"
    config["architectures"] = [BRIDGE_ARCHITECTURE]
    config["auto_map"] = {"AutoModelForCausalLM": f"modeling_transfusion.{BRIDGE_ARCHITECTURE}"}
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # save_artifacts resolves auto_map, so the alias module must exist.
    (view / "modeling_transfusion.py").write_text(
        "from transformers import Qwen3VLForConditionalGeneration\n"
        f"{BRIDGE_ARCHITECTURE} = Qwen3VLForConditionalGeneration\n",
        encoding="utf-8",
    )
    return view


def build_megatron_model(hf_model: Path, ckpt, iteration, dtype, arch_overrides=None):
    """Build the dense Pixel-DiT Megatron model, optionally loading ``ckpt``.

    Uses the same builder as the inference entrypoints, so architecture flags
    are read from run_config.yaml when ``ckpt`` is given; otherwise from
    ``arch_overrides``.
    """
    import torch

    from recipe.transfusion.eval.inference_transfusion_qwen3vl_pixel_dit_i2i import (
        add_pixel_dit_arch_args,
        build_pixel_dit_from_args,
    )

    parser = argparse.ArgumentParser()
    add_pixel_dit_arch_args(parser)
    args = parser.parse_args([])
    args.hf_path = str(hf_model)
    args.ckpt = None if ckpt is None else str(ckpt)
    args.iteration = iteration
    for key, value in (arch_overrides or {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    model, _ = build_pixel_dit_from_args(args, torch.device("cuda"), dtype=getattr(torch, dtype))
    return model, args


def save_weights(bridge, model, out: Path, show_progress=True, strict=True):
    """Write Qwen3-VL keys in the base model's shard layout, ``transfusion.*`` to a sidecar.

    The base layout only knows Qwen3-VL keys, so Pixel-DiT tensors are diverted
    into ``transfusion_weights.safetensors`` (read by ``hf_inference/export.py``).
    """
    from safetensors.torch import save_file

    sidecar = {}

    def base_weights():
        for name, tensor in bridge.export_hf_weights([model], cpu=True, show_progress=show_progress):
            if name.startswith("transfusion."):
                sidecar[name] = tensor.contiguous()
            else:
                yield name, tensor

    bridge.hf_pretrained.state.source.save_generator(base_weights(), str(out), strict=strict)
    if not sidecar:
        raise RuntimeError("Bridge exported no transfusion.* tensors; is the Transfusion bridge registered?")
    save_file(sidecar, str(out / "transfusion_weights.safetensors"))
    print(f"Saved {len(sidecar)} transfusion tensors to transfusion_weights.safetensors", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--megatron-path", required=True, type=Path,
                        help="checkpoint root or iter_XXXXXXX directory")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--hf-model", type=Path, default=None,
                        help="base Qwen3-VL HF directory (config + tokenizer); "
                             "defaults to the path recorded in run_config.yaml")
    parser.add_argument("--hf-path", required=True, type=Path, help="output directory")
    parser.add_argument("--export-dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--no-strict", action="store_true",
                        help="do not fail on Bridge key-coverage validation")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from hf_inference.configuration_pixel_dit import validate_architecture

    iter_dir = resolve_iteration_dir(args.megatron_path, args.iteration)
    run_config = read_run_config(iter_dir)
    validate_architecture({**(run_config.get("model") or {}), **(run_config.get("transfusion") or {})})
    hf_model = args.hf_model or hf_model_from_run_config(run_config)
    if hf_model is None or not (Path(hf_model) / "config.json").is_file():
        raise SystemExit("--hf-model is required: run_config.yaml does not point to a local Qwen3-VL directory")
    out = args.hf_path.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Register the Transfusion bridge before AutoBridge dispatch.
    import model.transfusion.transfusion_qwen3vl_bridge  # noqa: F401
    from megatron.bridge import AutoBridge

    from recipe.transfusion.eval.pixel_dit_runtime import cleanup_dist

    view = prepare_bridge_hf_dir(Path(hf_model), out)
    try:
        model, _ = build_megatron_model(view, iter_dir, None, DTYPES[args.export_dtype])
        print(f"Loaded {iter_dir}", flush=True)
        bridge = AutoBridge.from_hf_pretrained(str(view), trust_remote_code=True)
        bridge.hf_pretrained.save_artifacts(str(out))
        save_weights(bridge, model, out, show_progress=not args.no_progress, strict=not args.no_strict)
    finally:
        shutil.rmtree(view, ignore_errors=True)
    for stale in ("modeling_transfusion.py",):
        (out / stale).unlink(missing_ok=True)
    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = ["Qwen3VLForConditionalGeneration"]
    config.pop("auto_map", None)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported Bridge-layout weights to {out}", flush=True)
    cleanup_dist(True)


if __name__ == "__main__":
    main()
