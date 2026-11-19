#!/usr/bin/env python3
"""Instruction-based image editing (reference image + instruction -> image) with an HF Pixel-DiT export.

    python hf_inference/edit.py --model exports/pixel_dit_hf \\
        --image input.jpg --prompt "Make the sky purple." --output edits/
    torchrun --nproc_per_node 8 hf_inference/edit.py --model exports/pixel_dit_hf \\
        --data edits.jsonl --image-root images/ --output edits/ --resume

jsonl records: {"id", "image", "prompt" | "instruction"}. Images are written to
``<output>/<id>.png`` (``<id>_<repeat>.png`` with ``--repeats > 1``) and every
finished sample is appended to ``<output>/results.jsonl``.
"""

import argparse
import json
import math
import os
import re

if __package__:
    from . import runtime
    from .editing import CFG_MODES, SCHEDULERS, edit_image
else:
    import runtime
    from editing import CFG_MODES, SCHEDULERS, edit_image

REPEAT_SEED_STRIDE = 100003  # Same per-repeat seed offset as the Megatron edit entrypoint.


def parse_args(argv=None):
    """Parse model, data and sampling arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Exported HF Pixel-DiT directory")
    parser.add_argument("--tokenizer", help="Optional tokenizer override")
    parser.add_argument("--image", default="", help="Reference image for --prompt")
    parser.add_argument("--prompt", default="", help="Edit instruction; otherwise use --data")
    parser.add_argument("--data", default="", help="jsonl: {id, image, prompt|instruction}")
    parser.add_argument("--image-root", default="", help="Prefix for relative jsonl image paths")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Skip samples whose image already exists")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Use the first N records (0 = all)")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--shift", type=float, default=4.0)
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="euler")
    parser.add_argument("--cfg-mode", choices=CFG_MODES, default="edit3")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Guidance for --cfg-mode text")
    parser.add_argument("--w-text", type=float, default=4.0, help="edit3 text guidance")
    parser.add_argument("--w-img", type=float, default=1.0, help="edit3 image guidance (1 = two branches)")
    parser.add_argument("--cfg-keep-chat-template", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep the empty chat template in the edit3 unconditional branch")
    parser.add_argument("--route-token", action=argparse.BooleanOptionalAction, default=None,
                        help="<|image_gate|> before the output image: auto (default), force or disable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match-ref-resolution", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate at the processed reference resolution")
    parser.add_argument("--resolution", type=int, default=512, help="Square output pixels without matching")
    parser.add_argument("--height", type=int, help="Output height in pixels (with --width)")
    parser.add_argument("--width", type=int, help="Output width in pixels (with --height)")
    parser.add_argument("--min-pixels", type=int, help="Reference processor min_pixels")
    parser.add_argument("--max-pixels", type=int, help="Reference processor max_pixels")
    parser.add_argument("--resize-to-ref", action=argparse.BooleanOptionalAction, default=True,
                        help="Resize outputs to the original reference size")
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
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be set together")
    if bool(args.prompt) == bool(args.data):
        parser.error("pass either --prompt (with --image) or --data")
    if args.prompt and not args.image:
        parser.error("--prompt requires --image")
    return args


def load_items(args):
    """Return [{id, image, prompt}] from --prompt or --data."""
    if args.prompt:
        return [{"id": "0", "image": args.image, "prompt": args.prompt}]
    items = []
    for index, record in enumerate(runtime.read_jsonl(args.data)):
        prompt = record.get("prompt", record.get("instruction"))
        image = record.get("image") or (record.get("images") or [None])[0]
        if prompt is None or not image:
            raise ValueError(f"{args.data} record {index} needs image and prompt/instruction")
        item_id = str(record.get("id", index))
        if not re.fullmatch(r"[\w.\-]+", item_id):
            raise ValueError(f"{args.data} record {index}: id {item_id!r} is not a safe file name")
        items.append({"id": item_id, "image": runtime.resolve_image_paths([image], args.image_root)[0],
                      "prompt": prompt})
    return items[:args.limit] if args.limit else items


def output_name(item_id, repeat, repeats):
    """File name of one sample."""
    return f"{item_id}.png" if repeats == 1 else f"{item_id}_{repeat}.png"


def main(argv=None):
    """Edit every (image, instruction) pair, sharded over torchrun workers."""
    args = parse_args(argv)
    from PIL import Image

    rank, world, local_rank = runtime.dist_info()
    items = load_items(args)
    os.makedirs(args.output, exist_ok=True)
    runtime.init_dist(world)
    try:
        device = runtime.resolve_device(args.device, local_rank)
        model = runtime.load_model(args.model, args.dtype, device)
        tokenizer = runtime.load_tokenizer(args.model, args.tokenizer)
        processor = runtime.load_processor(args.model)
        if rank == 0:
            print(f"[edit] model={args.model} items={len(items)} repeats={args.repeats} steps={args.steps} "
                  f"shift={args.shift} scheduler={args.scheduler} cfg_mode={args.cfg_mode} "
                  f"w_text={args.w_text} w_img={args.w_img} cfg_scale={args.cfg_scale}", flush=True)
        jobs = [(item, repeat) for item in items for repeat in range(args.repeats)]
        results = os.path.join(args.output, "results.jsonl")
        shard = runtime.shard_path(results, rank, world)
        with open(shard, "a" if args.resume else "w", encoding="utf-8") as handle:
            for item, repeat in jobs[rank::world]:
                name = output_name(item["id"], repeat, args.repeats)
                path = os.path.join(args.output, name)
                if args.resume and os.path.isfile(path):
                    continue
                seed = args.seed + repeat * REPEAT_SEED_STRIDE
                image = edit_image(
                    model, tokenizer, processor, item["prompt"], Image.open(item["image"]).convert("RGB"),
                    steps=args.steps, shift=args.shift, scheduler=args.scheduler, cfg_mode=args.cfg_mode,
                    cfg_scale=args.cfg_scale, w_text=args.w_text, w_img=args.w_img, seed=seed,
                    match_ref_resolution=args.match_ref_resolution, resolution=args.resolution,
                    height=args.height, width=args.width, min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels, keep_chat_template=args.cfg_keep_chat_template,
                    route=args.route_token, dynamic_noise_scale=args.dynamic_noise_scale,
                    dynamic_noise_ref_tokens=args.dynamic_noise_ref_tokens,
                    dynamic_noise_base_scale=args.dynamic_noise_base_scale, resize_to_ref=args.resize_to_ref,
                )
                temporary = f"{path}.tmp.png"
                image.save(temporary)
                os.replace(temporary, path)
                record = {"id": f"{item['id']}" if args.repeats == 1 else f"{item['id']}_{repeat}",
                          "source_id": item["id"], "repeat": repeat, "seed": seed, "image": item["image"],
                          "prompt": item["prompt"], "output": name, "size": list(image.size)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"rank={rank} id={record['id']} -> {path}", flush=True)
        runtime.barrier()
        if world > 1 and rank == 0:
            order = [item["id"] if args.repeats == 1 else f"{item['id']}_{repeat}" for item, repeat in jobs]
            runtime.merge_shards(results, world, order)
    finally:
        runtime.close_dist()


if __name__ == "__main__":
    main()
