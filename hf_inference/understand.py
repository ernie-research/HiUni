#!/usr/bin/env python3
"""Image understanding (images + prompt -> text) with an exported HF Pixel-DiT model.

    python hf_inference/understand.py --model exports/pixel_dit_hf \\
        --image demo.jpg --prompt "Describe this image."
    torchrun --nproc_per_node 8 hf_inference/understand.py --model exports/pixel_dit_hf \\
        --data questions.jsonl --image-root images/ --output answers.jsonl --max-think-tokens 2048

jsonl records: {"id", "prompt" | "question", "image" | "images", "system"?}.
"""

import argparse

if __package__:
    from .text_decode import add_decode_args, run
else:
    from text_decode import add_decode_args, run


def main(argv=None):
    """Parse arguments and decode every request."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_decode_args(parser, with_images=True)
    run(parser.parse_args(argv), with_images=True)


if __name__ == "__main__":
    main()
