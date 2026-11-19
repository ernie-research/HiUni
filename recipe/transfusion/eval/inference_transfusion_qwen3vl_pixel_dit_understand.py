#!/usr/bin/env python3
"""Image understanding (images + text -> text) with the dense Pixel-DiT Megatron model.

Single sample::

    python inference_transfusion_qwen3vl_pixel_dit_understand.py \
        --ckpt <megatron_ckpt> --hf_path <Qwen3-VL-4B-Instruct> \
        --image cat.jpg --prompt "Describe the image."

Batch (jsonl records ``{"id", "prompt", "image" | "images", "system"?}``)::

    torchrun --nproc_per_node 8 inference_transfusion_qwen3vl_pixel_dit_understand.py \
        --ckpt <ckpt> --hf_path <hf> --data_path vqa.jsonl --output answers.jsonl

Each output record holds ``raw`` (with the think block), ``answer`` (think
block removed), token counts and ``finish_reason``.
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
_BRIDGE = os.path.join(_PROJECT_ROOT, "3rdparty", "Megatron-Bridge")
for _p in (_PROJECT_ROOT, os.path.join(_BRIDGE, "src"), os.path.join(_BRIDGE, "3rdparty", "Megatron-LM")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

from recipe.transfusion.eval.inference_transfusion_qwen3vl_pixel_dit_i2i import (  # noqa: E402
    add_pixel_dit_arch_args,
)
from recipe.transfusion.eval.pixel_dit_text_decode import add_decode_args, run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Qwen3-VL dense Pixel-DiT image understanding")
    add_decode_args(parser, with_images=True)
    add_pixel_dit_arch_args(parser)
    args = parser.parse_args()
    if args.prompt and not args.image:
        parser.error("--prompt needs at least one --image; use the text entrypoint for text-only prompts")
    return args


if __name__ == "__main__":
    run(parse_args(), with_images=True)
