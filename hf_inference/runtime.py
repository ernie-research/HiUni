"""Model loading, torchrun sharding and jsonl helpers shared by the per-task HF CLIs."""

import json
import os
from datetime import timedelta
from pathlib import Path


def dist_info():
    """Return (rank, world_size, local_rank) from torchrun's environment."""
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if not 0 <= rank < world:
        raise ValueError("Invalid RANK/WORLD_SIZE")
    return rank, world, int(os.environ.get("LOCAL_RANK", 0))


def init_dist(world):
    """Start a CPU process group used only for barriers between workers."""
    import torch.distributed as dist

    if world > 1 and not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(hours=6))


def barrier():
    """Synchronize workers when torchrun started more than one."""
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def close_dist():
    """Tear down the process group created by init_dist."""
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(device, local_rank):
    """Map a bare ``cuda`` to this worker's GPU and make it current."""
    import torch

    if device == "cuda":
        device = f"cuda:{local_rank}"
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    return device


def load_model(path, dtype="bfloat16", device="cuda"):
    """Load an exported Pixel-DiT directory and reject partial checkpoints."""
    import torch
    from transformers import AutoModelForCausalLM

    model, info = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=getattr(torch, dtype),
        attn_implementation="sdpa", output_loading_info=True,
    )
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete/incompatible HF checkpoint: {info}")
    if not hasattr(model, "diffusion_head_a"):
        raise ValueError("Not a Pixel-DiT export; run hf_inference/export.py first")
    return model.to(device).eval()


def load_tokenizer(model_dir, tokenizer_dir=None):
    """Load the tokenizer shipped with the export unless overridden."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir or model_dir)


def load_processor(model_dir):
    """Use the slow Qwen2-VL processor, matching the Megatron entrypoints numerically."""
    from transformers import Qwen2VLImageProcessor

    return Qwen2VLImageProcessor.from_pretrained(model_dir)


def read_jsonl(path):
    """Read non-empty jsonl lines as dictionaries."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def done_ids(path):
    """Return the ids already written to a jsonl output."""
    return {record["id"] for record in read_jsonl(path)} if os.path.isfile(path) else set()


def shard_path(output, rank, world):
    """Per-rank output file; a single worker writes the final file directly."""
    return output if world == 1 else f"{output}.rank{rank}-of-{world}"


def merge_shards(output, world, order):
    """Merge per-rank jsonl shards into ``output`` in input order, keeping prior records."""
    records = {record["id"]: record for record in read_jsonl(output)} if os.path.isfile(output) else {}
    shards = [shard_path(output, rank, world) for rank in range(world)]
    for shard in shards:
        if os.path.isfile(shard):
            records.update({record["id"]: record for record in read_jsonl(shard)})
    temporary = f"{output}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        for key in order:
            if key in records:
                handle.write(json.dumps(records[key], ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    for shard in shards:
        if os.path.isfile(shard):
            os.remove(shard)


def resolve_image_paths(paths, root):
    """Prefix relative paths with ``root``; absolute paths are kept."""
    return [path if not root or os.path.isabs(path) else str(Path(root) / path) for path in paths]
