#!/usr/bin/env python3
"""Turn an averaged (EMA) checkpoint into a loadable Megatron release checkpoint.

``average_model_checkpoints.py`` writes parameter-only torch_dist shards plus
``metadata.json``.  Megatron's ``dist_checkpointing.load`` additionally needs
``common.pt``; the inference, export and hot-start code read structural
hyperparameters from ``run_config.yaml``; and Megatron-Bridge resolves the
iteration directory through ``latest_checkpointed_iteration.txt``.  This script
adds all three and, by default, strips machine-specific absolute paths from the
copied ``run_config.yaml`` and ``provenance.json``.

Example::

    python recipe/transfusion/scripts/prepare_release_checkpoint.py \\
        --averaged-dir output/run-ema/ema_0.95/iter_0008000 \\
        --run-config output/run/iter_0008000/run_config.yaml \\
        --output checkpoints-megatron/pixel-dit-ema0.95-4k-8k

The output is a regular Megatron checkpoint root.  It contains model weights
only (no optimizer, RNG or dataloader state), so resume it with
``checkpoint.finetune: true`` and ``checkpoint.load_optim: false``.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch

ITER_RE = re.compile(r"^iter_(\d{7})$")
PATH_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+): (?P<value>/\S+)\s*$")
NULL_KEYS = {
    "load", "save", "pretrained_checkpoint", "dataloader_load", "dataloader_save",
    "tensorboard_dir", "wandb_save_dir", "save_dir",
}
HF_KEYS = {"hf_model_id", "hf_path", "hf_model_path", "tokenizer_model"}
# Mirrors megatron.core.dist_checkpointing for checkpoints saved by Megatron-Bridge.
COMMON_STATE = {
    "checkpoint_version": 3.0,
    "content_metadata": {
        "distrib_optim_sharding_type": "dp_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    },
}


def parse_args(argv=None):
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--averaged-dir", required=True, type=Path,
                        help="iter_XXXXXXX directory produced by average_model_checkpoints.py")
    parser.add_argument("--run-config", required=True, type=Path,
                        help="run_config.yaml of the training run that produced the averaged steps")
    parser.add_argument("--output", required=True, type=Path, help="release checkpoint root to create")
    parser.add_argument("--iteration", type=int, default=None,
                        help="iteration recorded in the release (default: parsed from --averaged-dir)")
    parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy",
                        help="how shard files are placed in the output (default: copy)")
    parser.add_argument("--hf-model-id", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="replacement for absolute HF base model paths in run_config.yaml")
    parser.add_argument("--keep-paths", action="store_true",
                        help="copy run_config.yaml and provenance.json verbatim instead of removing absolute paths")
    return parser.parse_args(argv)


def sanitize_run_config(text, hf_model_id):
    """Replace absolute paths with portable values and report every change."""
    changes = []
    lines = []
    for line in text.splitlines(keepends=True):
        match = PATH_LINE_RE.match(line.rstrip("\n"))
        if match is None:
            lines.append(line)
            continue
        key, value = match["key"], match["value"]
        if key in HF_KEYS:
            new = hf_model_id
        elif key in NULL_KEYS:
            new = "null"
        elif value.endswith((".yaml", ".yml")):
            new = f"recipe/transfusion/probs/{Path(value).name}"
        else:
            new = "null"
        changes.append((key, value, new))
        lines.append(f"{match['indent']}{key}: {new}\n")
    return "".join(lines), changes


def sanitize_provenance(data):
    """Make the averaging provenance portable: paths become relative to its output_root."""
    root = Path(data.get("output_root", "/"))

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and value.startswith("/"):
            path = Path(value)
            return str(path.relative_to(root)) if path.is_relative_to(root) and path != root else path.name
        return value

    return portable(data)


def place(src, dst, mode):
    """Copy, hardlink or symlink one file."""
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def prepare(args):
    """Build the release checkpoint in a temporary directory, then publish it."""
    src = args.averaged_dir
    if not (src / ".metadata").is_file() or not (src / "metadata.json").is_file():
        raise SystemExit(f"{src} is not a torch_dist checkpoint (missing .metadata or metadata.json)")
    if not args.run_config.is_file():
        raise SystemExit(f"run_config.yaml not found: {args.run_config}")
    iteration = args.iteration
    if iteration is None:
        match = ITER_RE.match(src.name)
        if match is None:
            raise SystemExit("cannot parse iteration from --averaged-dir; pass --iteration")
        iteration = int(match.group(1))
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    staging = args.output.with_name(f".{args.output.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    iter_dir = staging / f"iter_{iteration:07d}"
    iter_dir.mkdir(parents=True)

    for item in sorted(src.iterdir()):
        if not item.is_file() or item.name in ("common.pt", "run_config.yaml"):
            continue
        if item.name == "provenance.json" and not args.keep_paths:
            data = sanitize_provenance(json.loads(item.read_text()))
            (iter_dir / item.name).write_text(json.dumps(data, indent=2) + "\n")
            continue
        place(item, iter_dir / item.name, args.mode)

    torch.save({**COMMON_STATE, "iteration": iteration}, iter_dir / "common.pt")

    text = args.run_config.read_text()
    if not args.keep_paths:
        text, changes = sanitize_run_config(text, args.hf_model_id)
        for key, old, new in changes:
            print(f"run_config: {key}: {old} -> {new}")
    (iter_dir / "run_config.yaml").write_text(text)
    (staging / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    staging.rename(args.output)
    print(f"release checkpoint ready: {args.output} (iteration {iteration}, mode {args.mode})")


def main(argv=None):
    """Script entry point."""
    prepare(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
