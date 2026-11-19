"""Small runtime helpers shared by dense Pixel-DiT inference entrypoints."""

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank
    return False, 0, 1, 0


def cleanup_dist(is_ddp: bool):
    if is_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def _to01(x):
    return torch.clamp((x.float() + 1.0) / 2.0, 0.0, 1.0)


ONEIG_CLASS_ITEM = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
    "Multilingualism": "multilingualism",
}


def _init_megatron_parallel_state(
    seed: int = 42,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    from megatron.core import parallel_state

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
        )
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(seed)


def _resolve_megatron_ckpt_dir(base_path: str, iteration: int = None) -> str:
    base = Path(base_path)
    if iteration is None:
        latest = base / "latest_checkpointed_iteration.txt"
        if latest.exists():
            iteration = int(latest.read_text().strip())
        elif (base / "metadata.json").exists():
            return str(base)
        else:
            raise FileNotFoundError(f"No checkpoint iteration found in {base}")
    path = base / f"iter_{int(iteration):07d}"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")
    return str(path)


# Statistics buffers that Pixel-DiT never reads. Parameter-only checkpoints,
# such as averaged EMA releases, do not store them.
OPTIONAL_CHECKPOINT_KEYS = ("vit_channel_mean", "vit_channel_std", "vit_stats_count")


def load_megatron_checkpoint(model, ckpt_dir: str):
    """Load a torch_dist checkpoint, tolerating only absent unused buffers."""
    from megatron.core.dist_checkpointing import load as dist_ckpt_load
    from megatron.core.dist_checkpointing.validation import StrictHandling

    # Megatron calls keys requested by the model but absent from the
    # checkpoint "unexpected"; RETURN_ALL drops them from the load request.
    loaded, _, absent = dist_ckpt_load(
        sharded_state_dict=model.sharded_state_dict(prefix=""),
        checkpoint_dir=ckpt_dir,
        strict=StrictHandling.RETURN_ALL,
    )
    absent = sorted(absent)
    # TransformerEngine `_extra_state` holds FP8 scaling metadata that is
    # re-initialised on load; parameter-only checkpoints omit it as well.
    required = [key for key in absent
                if not key.startswith(OPTIONAL_CHECKPOINT_KEYS) and not key.endswith("._extra_state")]
    if required:
        raise KeyError(f"Checkpoint {ckpt_dir} lacks {len(required)} model tensors: {required[:10]}")
    if absent:
        logger.info("Checkpoint omits %d optional entries (buffers / TE extra state)", len(absent))
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    missing = [key for key in missing if key not in absent and not key.endswith("._extra_state")]
    if missing:
        logger.warning("Model keys not restored from checkpoint: %d", len(missing))
        for key in missing[:10]:
            logger.warning("  [missing] %s", key)
    if unexpected:
        logger.warning("Unexpected checkpoint keys: %d", len(unexpected))
        for key in unexpected[:10]:
            logger.warning("  [unexpected] %s", key)


def save_image_tensor(tensor: torch.Tensor, path: str):
    from torchvision.utils import save_image

    if tensor.dim() == 4:
        tensor = tensor[0]
    save_image(tensor.clamp(0, 1), path)


class CLIPScoreEvaluator:
    def __init__(self, device="cuda"):
        try:
            import clip

            self.model, self.preprocess = clip.load("ViT-B/32", device=device)
            self.device = device
        except ImportError:
            logger.warning("CLIP is not installed; CLIP evaluation is disabled")
            self.model = None

    def compute_score(self, images: torch.Tensor, texts: List[str]):
        if self.model is None:
            return 0.0, 0
        import clip

        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features * text_features).sum(dim=-1).sum().item(), len(texts)


class FIDFeatureExtractor:
    def __init__(self, device="cuda"):
        try:
            from torchvision.models import inception_v3

            self.model = inception_v3(pretrained=True, transform_input=False)
            self.model.fc = torch.nn.Identity()
            self.model = self.model.to(device).eval()
        except Exception as exc:
            logger.warning("Failed to load Inception model: %s", exc)
            self.model = None

    def __call__(self, images: torch.Tensor):
        if self.model is None:
            return torch.empty(0, 2048)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.model(images)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
