#!/usr/bin/env python3
"""Text-to-image generation (prompt -> image) with an HF Pixel-DiT export.

    python hf_inference/t2i.py --model exports/pixel_dit_hf \\
        --prompt "A red fox in the snow." --output t2i/
    torchrun --nproc_per_node 8 hf_inference/t2i.py --model exports/pixel_dit_hf \\
        --data prompts.jsonl --output t2i/ --resume

jsonl records: {"id", "prompt" | "caption", "height"?, "width"?}. Images are
written to ``<output>/<id>.png`` (``<id>_<repeat>.png`` with ``--repeats > 1``)
and every finished sample is appended to ``<output>/results.jsonl``. For
benchmark layouts (OneIG / LongText / QwenImageBench) use ``generate.py``.
"""

import argparse
import json
import math
import os
import re

if __package__:
    from . import runtime
else:
    import runtime


def parse_args(argv=None):
    """Parse model, data and sampling arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Exported HF Pixel-DiT directory")
    parser.add_argument("--tokenizer", help="Optional tokenizer override")
    parser.add_argument("--prompt", default="", help="Single prompt; otherwise use --data")
    parser.add_argument("--data", default="", help="jsonl: {id, prompt|caption, height?, width?}")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Skip samples whose image already exists")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Use the first N records (0 = all)")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels (multiple of 32)")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels (multiple of 32)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--shift", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--instruction", help="Fixed instruction prefix; default samples the training prefix list")
    parser.add_argument("--cfg-keep-chat-template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--predict-x0", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dynamic-noise-scale", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dynamic-noise-ref-tokens", type=int)
    parser.add_argument("--dynamic-noise-base-scale", type=float)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.limit < 0 or args.steps <= 0 or args.seed < 0:
        parser.error("repeats and steps must be positive; limit and seed nonnegative")
    if not math.isfinite(args.shift) or args.shift <= 0:
        parser.error("shift must be finite and positive")
    if not math.isfinite(args.cfg) or args.cfg < 1:
        parser.error("cfg must be finite and at least 1")
    if bool(args.prompt) == bool(args.data):
        parser.error("pass either --prompt or --data")
    return args


def check_size(height, width, where):
    """Reject sizes the 32-pixel patch grid cannot represent."""
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError(f"{where}: height and width must be positive multiples of 32 pixels")
    return int(height), int(width)


def load_items(args):
    """Return [{id, prompt, height, width}] from --prompt or --data."""
    size = check_size(args.height, args.width, "--height/--width")
    if args.prompt:
        return [{"id": "0", "prompt": args.prompt, "height": size[0], "width": size[1]}]
    items = []
    for index, record in enumerate(runtime.read_jsonl(args.data)):
        prompt = record.get("prompt", record.get("caption"))
        if not prompt:
            raise ValueError(f"{args.data} record {index} needs prompt/caption")
        item_id = str(record.get("id", index))
        if not re.fullmatch(r"[\w.\-]+", item_id):
            raise ValueError(f"{args.data} record {index}: id {item_id!r} is not a safe file name")
        height, width = check_size(record.get("height", size[0]), record.get("width", size[1]),
                                   f"{args.data} record {index}")
        items.append({"id": item_id, "prompt": prompt, "height": height, "width": width})
    return items[:args.limit] if args.limit else items


def output_name(item_id, repeat, repeats):
    """File name of one sample."""
    return f"{item_id}.png" if repeats == 1 else f"{item_id}_{repeat}.png"


def main(argv=None):
    """Generate every prompt, sharded over torchrun workers."""
    args = parse_args(argv)
    rank, world, local_rank = runtime.dist_info()
    items = load_items(args)
    os.makedirs(args.output, exist_ok=True)
    runtime.init_dist(world)
    try:
        device = runtime.resolve_device(args.device, local_rank)
        model = runtime.load_model(args.model, args.dtype, device)
        tokenizer = runtime.load_tokenizer(args.model, args.tokenizer)
        if rank == 0:
            print(f"[t2i] model={args.model} items={len(items)} repeats={args.repeats} steps={args.steps} "
                  f"cfg={args.cfg} shift={args.shift}", flush=True)
        jobs = [(index, item, repeat) for index, item in enumerate(items) for repeat in range(args.repeats)]
        results = os.path.join(args.output, "results.jsonl")
        with open(runtime.shard_path(results, rank, world), "a" if args.resume else "w", encoding="utf-8") as handle:
            for index, item, repeat in jobs[rank::world]:
                name = output_name(item["id"], repeat, args.repeats)
                path = os.path.join(args.output, name)
                if args.resume and os.path.isfile(path):
                    continue
                # Same per-sample seed layout as generate.py.
                seed = args.seed + index * args.repeats + repeat
                image = model.generate_image(
                    item["prompt"], tokenizer=tokenizer, height=item["height"], width=item["width"],
                    num_steps=args.steps, guidance_scale=args.cfg, shift=args.shift, seed=seed,
                    negative_prompt=args.negative_prompt, instruction=args.instruction,
                    cfg_keep_chat_template=args.cfg_keep_chat_template, predict_x0=args.predict_x0,
                    dynamic_noise_scale=args.dynamic_noise_scale,
                    dynamic_noise_ref_tokens=args.dynamic_noise_ref_tokens,
                    dynamic_noise_base_scale=args.dynamic_noise_base_scale,
                )
                temporary = f"{path}.tmp.png"
                image.save(temporary)
                os.replace(temporary, path)
                record = {"id": item["id"] if args.repeats == 1 else f"{item['id']}_{repeat}",
                          "source_id": item["id"], "repeat": repeat, "seed": seed, "prompt": item["prompt"],
                          "output": name, "size": list(image.size)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"rank={rank} id={record['id']} -> {path}", flush=True)
        runtime.barrier()
        if world > 1 and rank == 0:
            order = [item["id"] if args.repeats == 1 else f"{item['id']}_{repeat}" for _, item, repeat in jobs]
            runtime.merge_shards(results, world, order)
    finally:
        runtime.close_dist()


if __name__ == "__main__":
    main()
