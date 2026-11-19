#!/usr/bin/env python3
"""Pure-text chat (text -> text) with the dense Pixel-DiT Megatron model.

Single prompt::

    python inference_transfusion_qwen3vl_pixel_dit_text.py \
        --ckpt <megatron_ckpt> --hf_path <Qwen3-VL-4B-Instruct> \
        --prompt "Explain flow matching in two sentences."

Batch (jsonl records ``{"id", "prompt", "system"?}``)::

    torchrun --nproc_per_node 8 inference_transfusion_qwen3vl_pixel_dit_text.py \
        --ckpt <ckpt> --hf_path <hf> --data_path prompts.jsonl --output answers.jsonl

The input layout matches ``sft_text`` training samples: chat template, linear
positions and a causal mask.
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
    parser = argparse.ArgumentParser("Qwen3-VL dense Pixel-DiT text chat")
    add_decode_args(parser, with_images=False)
    add_pixel_dit_arch_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args(), with_images=False)
