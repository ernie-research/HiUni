#!/usr/bin/env python3
"""Import an HF Pixel-DiT export into a Megatron torch_dist checkpoint.

This is the inverse of ``hf_inference/export.py``:

    python recipe/transfusion/import_hf_to_megatron.py \\
        --hf-path checkpoints/pixel_dit_hf \\
        --megatron-path checkpoints/pixel_dit_megatron

The result is a parameter-only checkpoint (no optimizer / RNG state) with a
``run_config.yaml`` carrying the Pixel-DiT architecture and sampling flags.
Use it as ``checkpoint.pretrained_checkpoint`` for hot-start training, or pass
it to the Megatron inference entrypoints and ``export_transfusion_ckpt.py``.

Megatron parameters the HF model does not carry (the unused image input
projection, ViT LayerNorms and ViT statistics buffers) are zero-filled; every
other parameter must be covered and every HF tensor must be consumed.
"""

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE = _PROJECT_ROOT / "3rdparty" / "Megatron-Bridge"
for _p in (_PROJECT_ROOT, _BRIDGE / "src", _BRIDGE / "3rdparty" / "Megatron-LM"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

# HF config fields that select the Megatron Pixel-DiT architecture.
ARCH_KEYS = (
    "num_diffusion_layers", "pixel_dit_hidden_size", "num_diffusion_heads", "num_diffusion_kv_heads",
    "diffusion_intermediate_size", "pixel_dit_gate_inject_layer", "pixel_dit_gate_input",
    "use_pixel_dit_pixel_residual", "pixel_dit_ref_gate_mode", "dit_head_impl", "dit_block_adaln",
    "adaln_post_modulation",
)
# Flags the Megatron model does not hold but inference and re-export read from run_config.yaml.
MODEL_FLAGS = ("use_pixel_dit", "pixel_dit_use_text_tokens", "predict_x0", "predict_gate_mode")
SCHEDULER_FLAGS = ("use_dynamic_noise_scale", "dynamic_noise_ref_tokens", "dynamic_noise_base_scale")
# Megatron-only tensors the HF model drops because Pixel-DiT never reads them.
UNUSED_PREFIXES = ("image_input_projection.", "vit_channel_", "vit_stats_count", "vision_input_layernorm.",
                   "vision_output_layernorm.")
TIED_HF_KEYS = ("lm_head.weight",)


def read_hf_config(hf_path: Path) -> dict:
    """Load and validate the exported PixelDiTConfig."""
    from hf_inference.configuration_pixel_dit import validate_architecture

    config = json.loads((hf_path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "pixel_dit_qwen3_vl":
        raise ValueError(f"{hf_path} is not an hf_inference/export.py output (model_type={config.get('model_type')!r})")
    validate_architecture(config)
    return config


class _BridgeNames(Mapping):
    """Expose the export's bare Pixel-DiT keys under the Bridge's ``transfusion.*`` names."""

    def __init__(self, state, keys):
        self.state, self.keys_ = state, keys

    @staticmethod
    def canonical(name):
        return name.removeprefix("transfusion.")

    def __getitem__(self, name):
        return self.state[self.canonical(name)]

    def __contains__(self, name):
        return self.canonical(name) in self.keys_

    def __iter__(self):
        return iter(self.keys_)

    def __len__(self):
        return len(self.keys_)


def load_hf_into_megatron(bridge, pretrained, model):
    """Run the Bridge HF->Megatron tasks, skipping only mappings for tensors HF does not carry."""
    import torch

    model_bridge = bridge._model_bridge
    keys = set(pretrained.state.source.get_all_keys())
    state = _BridgeNames(pretrained.state, keys)
    loaded, consumed = set(), set()
    for task in model_bridge.build_conversion_tasks(pretrained, [model]):
        if task is None or task.megatron_module is None:
            continue
        hf_param = task.mapping.hf_param
        names = [hf_param] if isinstance(hf_param, str) else list(hf_param.values())
        if not all(name in state for name in names):
            continue
        weights = model_bridge.maybe_modify_loaded_hf_weight(hf_param, state)
        converted = task.mapping.hf_to_megatron(weights, task.megatron_module)
        if converted is None:
            continue
        if converted.shape != task.param_weight.shape:
            raise ValueError(f"Shape mismatch for {task.param_name}: HF {tuple(converted.shape)} "
                             f"vs Megatron {tuple(task.param_weight.shape)} ({hf_param})")
        task.param_weight.data.copy_(converted)
        loaded.add(id(task.param_weight))
        consumed.update(_BridgeNames.canonical(name) for name in names)
    model_bridge._broadcast_shared_embeddings([model])

    leftover = sorted(keys - consumed - set(TIED_HF_KEYS))
    if leftover:
        raise ValueError(f"{len(leftover)} HF tensors have no Megatron target: {leftover[:10]}")
    uncovered = [name for name, param in model.named_parameters() if id(param) not in loaded]
    unexpected = [name for name in uncovered if not name.startswith(UNUSED_PREFIXES)]
    if unexpected:
        raise ValueError(f"{len(unexpected)} Megatron parameters are missing from the HF export: {unexpected[:10]}")
    # perform_initialization=False leaves these uninitialised; make the checkpoint deterministic.
    with torch.no_grad():
        for param in model.parameters():
            if id(param) not in loaded:
                param.zero_()
    print(f"Loaded {len(loaded)} Megatron parameters from {len(consumed)} HF tensors; "
          f"zero-filled unused: {uncovered}", flush=True)


def annotate_run_config(checkpoint: Path, hf_config: dict) -> Path:
    """Record the HF flags Megatron does not store so the checkpoint is self-describing."""
    import yaml

    from recipe.transfusion.export_transfusion_ckpt import resolve_iteration_dir

    iter_dir = resolve_iteration_dir(checkpoint)
    path = iter_dir / "run_config.yaml"
    with path.open(encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle) or {}
    model_cfg = run_config.setdefault("model", {})
    for key in ARCH_KEYS + MODEL_FLAGS:
        if key in hf_config:
            model_cfg[key] = hf_config[key]
    run_config["scheduler_config"] = {key: hf_config[key] for key in SCHEDULER_FLAGS if key in hf_config}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False, allow_unicode=True)
    return iter_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-path", required=True, type=Path, help="hf_inference/export.py output directory")
    parser.add_argument("--megatron-path", required=True, type=Path, help="output checkpoint root (must be empty)")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    hf_path, out = args.hf_path.resolve(), args.megatron_path.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output must be empty: {out}")
    hf_config = read_hf_config(hf_path)
    # Single-process conversion: let TCPStore pick a free port instead of racing a probed one.
    for name, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                        ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "0")):
        os.environ.setdefault(name, value)

    # Register the Transfusion bridge before AutoBridge dispatch.
    import model.transfusion.transfusion_qwen3vl_bridge  # noqa: F401
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.model_load_save import save_megatron_model

    from recipe.transfusion.eval.pixel_dit_runtime import cleanup_dist
    from recipe.transfusion.export_transfusion_ckpt import build_megatron_model, prepare_bridge_hf_dir

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-import-", dir=out.parent) as workdir:
        # The view links the exported shards and dispatches config.json to the Transfusion bridge.
        view = prepare_bridge_hf_dir(hf_path, Path(workdir))
        overrides = {key: hf_config[key] for key in ARCH_KEYS if key in hf_config}
        megatron_model, _ = build_megatron_model(view, None, None, args.dtype, overrides)
        bridge = AutoBridge.from_hf_pretrained(str(view), trust_remote_code=True)
        load_hf_into_megatron(bridge, bridge.hf_pretrained, megatron_model)
        megatron_model.config.predict_x0 = bool(hf_config.get("predict_x0", False))
        save_megatron_model([megatron_model], out)
    iter_dir = annotate_run_config(out, hf_config)
    print(f"Saved Megatron checkpoint to {iter_dir}", flush=True)
    cleanup_dist(True)


if __name__ == "__main__":
    main()
