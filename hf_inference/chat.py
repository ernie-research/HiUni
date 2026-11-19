#!/usr/bin/env python3
"""Pure-text chat (prompt -> text) with an exported HF Pixel-DiT model.

    python hf_inference/chat.py --model exports/pixel_dit_hf --prompt "What is 17 * 23?"
    torchrun --nproc_per_node 8 hf_inference/chat.py --model exports/pixel_dit_hf \\
        --data prompts.jsonl --output answers.jsonl --max-think-tokens 2048

jsonl records: {"id", "prompt" | "question", "system"?}.
"""

import argparse

if __package__:
    from .text_decode import add_decode_args, run
else:
    from text_decode import add_decode_args, run


def main(argv=None):
    """Parse arguments and decode every request."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_decode_args(parser, with_images=False)
    run(parser.parse_args(argv), with_images=False)


if __name__ == "__main__":
    main()
