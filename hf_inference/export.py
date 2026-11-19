#!/usr/bin/env python3
"""Reuse the Megatron exporter and package a standalone, validated HF Pixel-DiT model."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_FILES = ("configuration_pixel_dit.py", "modeling_pixel_dit.py", "diffusion_head.py", "sampling.py")
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json",
    "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
)
UNUSED_PREFIXES = ("image_input_projection.", "vit_channel_", "vit_stats_count", "vision_input_layernorm.",
                   "vision_output_layernorm.")


def resolve_checkpoint(path):
    """Resolve an explicit iteration or the iteration selected by Megatron's tracker."""
    path = Path(path).resolve()
    if (path / ".metadata").is_file():
        return path
    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        value = tracker.read_text().strip()
        name = "release" if value == "release" else f"iter_{int(value):07d}"
        target = path / name
        if (target / ".metadata").is_file():
            return target
        raise FileNotFoundError(f"Tracker points to an incomplete checkpoint: {target}")
    raise FileNotFoundError(f"Pass an iteration directory containing .metadata: {path}")


def load_training_settings(path):
    """Read architecture flags from the authoritative checkpoint run_config.yaml."""
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle)
    if not isinstance(run_config, dict):
        raise ValueError("run_config.yaml must contain a mapping")
    settings = dict(run_config.get("model") or {})
    settings.update(run_config.get("transfusion") or {})
    scheduler = run_config.get("scheduler_config") or {}
    for name in ("use_dynamic_noise_scale", "dynamic_noise_ref_tokens", "dynamic_noise_base_scale"):
        if name in scheduler:
            settings[name] = scheduler[name]
    return settings


def build_config(source, settings):
    """Retain HF backbone settings and explicitly preserve all supported generation flags."""
    if __package__:
        from .configuration_pixel_dit import PIXEL_DIT_DEFAULTS, PixelDiTConfig, validate_architecture
    else:
        from configuration_pixel_dit import PIXEL_DIT_DEFAULTS, PixelDiTConfig, validate_architecture
    validate_architecture(settings)
    if "use_dynamic_noise_scale" not in settings:
        print("scheduler_config is absent from this checkpoint; using the Pixel-DiT "
              "benchmark default dynamic_noise_scale=true (reference tokens=256, base scale=1). "
              "Override with --no-dynamic-noise-scale if training disabled it.", flush=True)
    base = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    if base.get("model_type") not in ("qwen3_vl", "pixel_dit_qwen3_vl"):
        raise ValueError(f"Expected a dense Qwen3-VL export, got {base.get('model_type')!r}")
    for name in ("model_type", "auto_map", "architectures", "_name_or_path", "_original_architectures"):
        base.pop(name, None)
    # Defaults describe historical training, not the public inference preset.
    defaults = {**PIXEL_DIT_DEFAULTS, "pixel_dit_gate_inject_layer": 0, "predict_x0": False}
    base.update({key: settings.get(key, default) for key, default in defaults.items()})
    if base.get("text_config", {}).get("num_experts"):
        raise ValueError("MoE checkpoints are not supported by this release")
    config = PixelDiTConfig(**base)
    config.architectures = ["PixelDiTForConditionalGeneration"]
    config.auto_map = {
        "AutoConfig": "configuration_pixel_dit.PixelDiTConfig",
        "AutoModelForCausalLM": "modeling_pixel_dit.PixelDiTForConditionalGeneration",
    }
    return config


def source_weights(source):
    """Find indexed and sidecar safetensors without loading tensor contents."""
    from safetensors import safe_open

    source = Path(source)
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = sorted(set(mapping.values()))
        if (source / "transfusion_weights.safetensors").exists():
            files = sorted(set(files) | {"transfusion_weights.safetensors"})
    else:
        files = [path.name for path in sorted(source.glob("*.safetensors"))]
        mapping = None
    candidates = {}
    for filename in files:
        path = (source / filename).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError(f"Weight shard escapes model directory: {filename}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if mapping is not None and mapping.get(key, filename) != filename:
                    continue
                canonical = key.removeprefix("transfusion.")
                # The existing exporter writes both Bridge-prefixed keys and a bare sidecar.
                priority = (int(key.startswith("transfusion.")), int(filename != "transfusion_weights.safetensors"))
                candidate = (priority, path, key, tuple(handle.get_slice(key).get_shape()))
                if canonical not in candidates or priority < candidates[canonical][0]:
                    candidates[canonical] = candidate
    if not candidates:
        raise ValueError(f"No safetensors weights found in {source}")
    return candidates


def package_export(source, output, settings, tokenizer_source=None, max_shard_bytes=2 * 1024**3, dtype="bfloat16"):
    """Validate every model tensor, deduplicate exported heads, and write portable HF shards."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if __package__:
        from .modeling_pixel_dit import PixelDiTForConditionalGeneration
    else:
        # Keep model relative imports intact when this file is executed directly.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from hf_inference.modeling_pixel_dit import PixelDiTForConditionalGeneration

    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("Output must be separate from the source model directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty: {output}")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    config = build_config(source, settings)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = PixelDiTForConditionalGeneration(config)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    del model
    weights = source_weights(source)
    # Tied lm_head may be omitted from a standard HF export.
    if config.tie_word_embeddings and "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights.get("model.language_model.embed_tokens.weight")
    missing = [name for name in expected if not weights.get(name)]
    mismatched = [(name, weights[name][3], shape) for name, shape in expected.items()
                  if weights.get(name) and weights[name][3] != shape]
    unexpected = [name for name in weights if name not in expected and not name.startswith(UNUSED_PREFIXES)]
    if missing or mismatched or unexpected:
        raise ValueError(f"Checkpoint validation failed: missing={missing}, shapes={mismatched}, unexpected={unexpected}")

    target_dtype = getattr(torch, dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-package-", dir=output.parent) as temporary:
        destination = Path(temporary)
        shard, shard_bytes, index, total_size = {}, 0, {}, 0
        shard_number = 0

        def flush():
            """Flush the current shard and record its canonical keys."""
            nonlocal shard, shard_bytes, shard_number
            if not shard:
                return
            shard_number += 1
            name = f"model-{shard_number:05d}.safetensors"
            save_file(shard, str(destination / name), metadata={"format": "pt"})
            index.update({key: name for key in shard})
            shard, shard_bytes = {}, 0

        for name in expected:
            if name == "lm_head.weight" and config.tie_word_embeddings:
                continue
            _, filename, original, _ = weights[name]
            with safe_open(filename, framework="pt", device="cpu") as handle:
                value = handle.get_tensor(original).to(target_dtype).contiguous()
            size = value.numel() * value.element_size()
            if shard_bytes + size > max_shard_bytes:
                flush()
            shard[name] = value
            shard_bytes += size
            total_size += size
        flush()
        (destination / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_size}, "weight_map": index}, indent=2) + "\n",
            encoding="utf-8",
        )
        config.dtype = target_dtype
        config.save_pretrained(destination)
        assets = Path(tokenizer_source) if tokenizer_source else source
        for name in TOKENIZER_FILES:
            asset = assets / name
            if asset.is_file():
                shutil.copy2(asset, destination / name)
        if not (destination / "tokenizer_config.json").is_file():
            raise ValueError("Missing tokenizer_config.json; provide --tokenizer-source")
        if not any((destination / name).is_file() for name in ("tokenizer.json", "vocab.json")):
            raise ValueError("Missing tokenizer vocabulary; provide --tokenizer-source")
        for name in MODEL_FILES:
            shutil.copy2(Path(__file__).parent / name, destination / name)
        output.mkdir(parents=True, exist_ok=True)
        for artifact in destination.iterdir():
            shutil.move(str(artifact), output / artifact.name)
    print(f"Packaged {len(index)} tensors in {shard_number} shards: {output}")
    return config


def main(argv=None):
    """Convert a Megatron checkpoint or package a previously converted HF directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--megatron-path", type=Path)
    source.add_argument("--from-export", type=Path, help="Existing export_transfusion_ckpt.py output")
    parser.add_argument("--hf-model", type=Path, help="Original Qwen3-VL config/tokenizer for Megatron conversion")
    parser.add_argument("--run-config", type=Path, help="Required for --from-export; otherwise read from checkpoint")
    parser.add_argument("--hf-path", type=Path, required=True)
    parser.add_argument("--tokenizer-source", type=Path)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-shard-gb", type=float, default=2.0)
    parser.add_argument("--dynamic-noise-scale", action=argparse.BooleanOptionalAction, default=None,
                        help="Override noise scaling when scheduler_config was not saved in run_config.yaml")
    parser.add_argument("--dynamic-noise-ref-tokens", type=int)
    parser.add_argument("--dynamic-noise-base-scale", type=float)
    args = parser.parse_args(argv)
    overrides = {key: value for key, value in {
        "use_dynamic_noise_scale": args.dynamic_noise_scale,
        "dynamic_noise_ref_tokens": args.dynamic_noise_ref_tokens,
        "dynamic_noise_base_scale": args.dynamic_noise_base_scale,
    }.items() if value is not None}
    if args.from_export:
        if args.run_config is None:
            parser.error("--from-export requires --run-config from the original training checkpoint")
        package_export(args.from_export, args.hf_path, {**load_training_settings(args.run_config), **overrides},
                       args.tokenizer_source, int(args.max_shard_gb * 1024**3), args.dtype)
        return
    checkpoint = resolve_checkpoint(args.megatron_path)
    settings = {**load_training_settings(args.run_config or checkpoint / "run_config.yaml"), **overrides}
    if __package__:
        from .configuration_pixel_dit import validate_architecture
    else:
        from configuration_pixel_dit import validate_architecture
    validate_architecture(settings)
    repo = Path(__file__).resolve().parent.parent
    exporter = repo / "recipe/transfusion/export_transfusion_ckpt.py"
    if not exporter.is_file():
        raise FileNotFoundError("Megatron conversion runs inside the training repository; use --from-export here")
    if args.hf_path.exists() and any(args.hf_path.iterdir()):
        raise ValueError(f"Output must be empty: {args.hf_path}")
    args.hf_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".megatron-hf-", dir=args.hf_path.parent) as temporary:
        command = [sys.executable, str(exporter), "--megatron-path", str(checkpoint),
                   "--hf-path", str(Path(temporary).resolve()), "--export-dtype",
                   {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}[args.dtype]]
        if args.hf_model:
            command += ["--hf-model", str(args.hf_model.resolve())]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(repo), str(repo / "3rdparty/Megatron-Bridge/src"),
            str(repo / "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM"), env.get("PYTHONPATH", ""),
        ])
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)
        # This conversion uses world_size=1: let TCPStore bind its own free port.
        # Probing a socket and closing it first races with other local workers.
        env["MASTER_PORT"] = "0"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(command, cwd=repo, env=env, check=True)
        package_export(temporary, args.hf_path, settings, args.tokenizer_source,
                       int(args.max_shard_gb * 1024**3), args.dtype)


if __name__ == "__main__":
    main()
