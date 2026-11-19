#!/usr/bin/env python3
"""Generate OneIG, LongText, or QwenImageBench images from an exported HF model."""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path

if __package__:
    from .benchmarks import (finalize_outputs, load_prompts, prompt_digest, sample_path,
                             save_image_atomic, valid_image, validate_model_name, write_json_atomic)
else:
    from benchmarks import (finalize_outputs, load_prompts, prompt_digest, sample_path,
                            save_image_atomic, valid_image, validate_model_name, write_json_atomic)


def parse_args(argv=None):
    """Parse portable paths and benchmark-specific generation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local exported HF model directory")
    parser.add_argument("--tokenizer", help="Optional tokenizer override")
    parser.add_argument("--benchmark", required=True, choices=["OneIG", "LongText", "QwenImageBench"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--language", choices=["EN", "ZH"], default="EN")
    parser.add_argument("--prompt-column")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for one benchmark/language")
    parser.add_argument("--model-name", default="pixel-dit")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--shift", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, help="Default: OneIG/LongText=4, QwenImageBench=1")
    parser.add_argument("--limit", type=int, help="Use the first N prompts for a smoke test")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--instruction", help="Fixed instruction prefix; default samples the training prefix list")
    parser.add_argument("--cfg-keep-chat-template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-noise-scale", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dynamic-noise-ref-tokens", type=int)
    parser.add_argument("--dynamic-noise-base-scale", type=float)
    parser.add_argument("--predict-x0", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--drop-dit-text", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and show paths without loading torch or weights")
    parser.add_argument("--finalize-only", action="store_true", help="Validate images and rebuild evaluator artifacts")
    args = parser.parse_args(argv)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.benchmark == "QwenImageBench" else 4)
    if args.repeats <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("repeats and limit must be positive")
    if args.benchmark == "OneIG" and args.repeats not in (1, 4):
        parser.error("OneIG supports one image for smoke tests or four images for a 2x2 gallery")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        parser.error("height and width must be positive multiples of 32 pixels")
    if args.steps <= 0 or not math.isfinite(args.shift) or args.shift <= 0:
        parser.error("steps and shift must be positive")
    if not math.isfinite(args.cfg) or args.cfg < 1:
        parser.error("cfg must be finite and at least 1")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.dynamic_noise_ref_tokens is not None and args.dynamic_noise_ref_tokens <= 0:
        parser.error("dynamic-noise-ref-tokens must be positive")
    if args.dynamic_noise_base_scale is not None and (
        not math.isfinite(args.dynamic_noise_base_scale) or args.dynamic_noise_base_scale <= 0
    ):
        parser.error("dynamic-noise-base-scale must be finite and positive")
    validate_model_name(args.model_name)
    return args


def run_signature(args, prompts):
    """Record all image-affecting arguments while allowing a different worker count."""
    ignored = {"resume", "dry_run", "finalize_only", "output", "device", "limit", "data", "prompt_column"}
    signature = {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items() if key not in ignored}
    signature["model"] = str(Path(args.model).resolve())
    for key, directory in (("model_files", args.model), ("tokenizer_files", args.tokenizer)):
        if directory and Path(directory).is_dir():
            signature[key] = [[path.name, path.stat().st_size, path.stat().st_mtime_ns]
                              for path in sorted(Path(directory).iterdir()) if path.is_file()]
    signature["prompts_sha256"] = prompt_digest(prompts)
    return signature


def main(argv=None):
    """Shard complete prompts across torchrun workers, then publish evaluator input."""
    args = parse_args(argv)
    prompts = load_prompts(args.data, args.benchmark, args.language, args.prompt_column)
    if args.limit is not None:
        prompts = prompts[:args.limit]
    if not prompts:
        raise ValueError("No prompts selected")
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if not 0 <= rank < world_size:
        raise ValueError("Invalid RANK/WORLD_SIZE")
    if args.dry_run:
        if rank == 0:
            print(json.dumps({"prompts": len(prompts), "images": len(prompts) * args.repeats,
                              "output": str(args.output), "first_prompt": vars(prompts[0]),
                              "first_image": sample_path(prompts[0], args.benchmark, args.model_name, 0, args.repeats)},
                             ensure_ascii=False, indent=2))
        return
    if args.finalize_only:
        if rank == 0:
            finalize_outputs(prompts, args.output, args.benchmark, args.model_name,
                             args.repeats, (args.width, args.height))
        return

    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    distributed = world_size > 1
    if distributed:
        dist.init_process_group("gloo", timeout=timedelta(hours=6))
    try:
        manifest = args.output / "run_config.json"
        signature = run_signature(args, prompts)
        if rank == 0:
            if manifest.exists():
                if not args.resume:
                    raise ValueError("Output already has a run; use --resume or a new output directory")
                if json.loads(manifest.read_text(encoding="utf-8")) != signature:
                    raise ValueError("Resume arguments or prompts differ from run_config.json")
            elif args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Nonempty output directory has no run_config.json; choose a new directory")
            write_json_atomic(manifest, signature)
        if distributed:
            dist.barrier()
        model = tokenizer = None
        for ordinal in range(rank, len(prompts), world_size):
            prompt = prompts[ordinal]
            for repeat in range(args.repeats):
                output = args.output / sample_path(prompt, args.benchmark, args.model_name, repeat, args.repeats)
                if args.resume and valid_image(output, (args.width, args.height)):
                    continue
                if model is None:
                    device = args.device
                    if device == "cuda":
                        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
                    if device.startswith("cuda"):
                        torch.cuda.set_device(torch.device(device))
                    model, info = AutoModelForCausalLM.from_pretrained(
                        args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype),
                        attn_implementation="sdpa", output_loading_info=True,
                    )
                    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
                        raise ValueError(f"Incomplete/incompatible HF checkpoint: {info}")
                    if not hasattr(model, "generate_image"):
                        raise ValueError("Model lacks generate_image; run hf_inference/export.py first")
                    model = model.to(device).eval()
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
                seed = args.seed + ordinal * args.repeats + repeat
                image = model.generate_image(
                    prompt.text, tokenizer=tokenizer, height=args.height, width=args.width,
                    num_steps=args.steps, guidance_scale=args.cfg, shift=args.shift, seed=seed,
                    negative_prompt=args.negative_prompt, instruction=args.instruction,
                    cfg_keep_chat_template=args.cfg_keep_chat_template, predict_x0=args.predict_x0,
                    drop_dit_text=args.drop_dit_text,
                    dynamic_noise_scale=args.dynamic_noise_scale,
                    dynamic_noise_ref_tokens=args.dynamic_noise_ref_tokens,
                    dynamic_noise_base_scale=args.dynamic_noise_base_scale,
                )
                save_image_atomic(image, output)
                print(f"rank={rank} id={prompt.item_id} repeat={repeat} seed={seed} -> {output}", flush=True)
        if distributed:
            dist.barrier()
        if rank == 0:
            finalize_outputs(prompts, args.output, args.benchmark, args.model_name,
                             args.repeats, (args.width, args.height))
            print(f"Completed {len(prompts) * args.repeats} images: {args.output}")
        if distributed:
            # Keep every worker alive until rank 0 finishes evaluator artifacts.
            dist.barrier()
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
