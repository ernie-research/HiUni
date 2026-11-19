#!/usr/bin/env python3
"""Dense Pixel-DiT image editing inference for Qwen3-VL Transfusion.

The edit layout is aligned with TransfusionTaskEncoder._encode_editing_vit:
reference images are ViT merged tokens, the generated image is a pixel-space
diffusion target, and image_gen_grid_thw keeps the raw pixel patch grid.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
_MEGATRON_BRIDGE_SRC = os.path.join(_PROJECT_ROOT, "3rdparty", "Megatron-Bridge", "src")
_MEGATRON_LM = os.path.join(
    _PROJECT_ROOT, "3rdparty", "Megatron-Bridge", "3rdparty", "Megatron-LM"
)
for _p in (_PROJECT_ROOT, _MEGATRON_BRIDGE_SRC, _MEGATRON_LM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

from transformers import AutoTokenizer, Qwen2VLImageProcessor

from data.transfusion.transfusion_task_encoder import V5_IMAGE_GATE_TOKEN_ID
from data.transfusion.transfusion_data_cooker import open_image_checked
from model.transfusion.transfusion_sampler import (
    ASSISTANT_ID,
    BOI_ID,
    EOI_ID,
    EOS_ID,
    FlowMatchScheduler,
    IMG_PAD_ID,
    IM_END_ID,
    IM_START_ID,
    NL_ID,
    TIME_TOKEN_ID,
    USER_ID,
)
from recipe.transfusion.eval.pixel_dit_runtime import (
    cleanup_dist,
    is_main,
    setup_dist,
)
from recipe.transfusion.eval.pixel_dit_runtime import _resolve_megatron_ckpt_dir
from recipe.transfusion.eval.inference_transfusion_qwen3vl_pixel_dit import (
    _read_pixel_dit_config_from_ckpt,
    build_model_pixel_dit,
    patch_qwen3vl_rotary_packed_seq_compat,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TASK_EDITING = 2
PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
CHANNELS = 3
PIXEL_DIM = CHANNELS * PATCH_SIZE * PATCH_SIZE
VISION_PATCH_DIM = CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE


def _encode_text(tokenizer, text: str) -> List[int]:
    if not text:
        return []
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.tokenize(text)


def _supports_image_route_token(tokenizer) -> bool:
    if not hasattr(tokenizer, "encode"):
        return False
    try:
        encoded = tokenizer.encode("<|image_gate|>", add_special_tokens=False)
        converted = (
            tokenizer.convert_tokens_to_ids("<|image_gate|>")
            if hasattr(tokenizer, "convert_tokens_to_ids")
            else None
        )
    except Exception:
        return False
    return encoded == [V5_IMAGE_GATE_TOKEN_ID] and converted == V5_IMAGE_GATE_TOKEN_ID


def _resolve_predict_gate_mode(tokenizer, predict_gate_mode: Optional[bool]) -> bool:
    supported = _supports_image_route_token(tokenizer)
    if predict_gate_mode is True and not supported:
        raise ValueError(
            "predict_gate_mode=True requires <|image_gate|> to map to "
            f"the single V5 ID {V5_IMAGE_GATE_TOKEN_ID}"
        )
    return supported if predict_gate_mode is None else bool(predict_gate_mode)


def _build_edit_sequence(
    tokenizer,
    prompt: str,
    ref_tokens: int,
    gen_h: int,
    gen_w: int,
    cfg_mode: str,
    cfg_keep_chat_template: bool = False,
    predict_gate_mode: Optional[bool] = None,
) -> torch.Tensor:
    """Build one edit sequence for cfg_mode in none/text/full.

    By default the full-uncond branch is the same image-only layout used by
    T2I. Set cfg_keep_chat_template=True to retain the legacy empty chat
    wrapper. The text-uncond branch always keeps its reference-image wrapper.
    """
    if cfg_mode not in ("none", "text", "full"):
        raise ValueError(f"invalid cfg_mode: {cfg_mode}")

    num_gen_tokens = gen_h * gen_w
    user_content: List[int] = []
    if cfg_mode != "full" and ref_tokens > 0:
        user_content += [BOI_ID] + [IMG_PAD_ID] * ref_tokens + [EOI_ID]
    if cfg_mode == "none":
        user_content += _encode_text(tokenizer, prompt)

    route_tokens = (
        [V5_IMAGE_GATE_TOKEN_ID]
        if cfg_mode == "none" and _resolve_predict_gate_mode(tokenizer, predict_gate_mode)
        else []
    )
    image_tokens = (
        route_tokens
        + [BOI_ID, TIME_TOKEN_ID]
        + [IMG_PAD_ID] * num_gen_tokens
        + [EOI_ID, EOS_ID]
    )
    if cfg_mode == "full" and not cfg_keep_chat_template:
        seq = image_tokens
    else:
        seq = (
            [IM_START_ID, USER_ID, NL_ID]
            + user_content
            + [IM_END_ID, NL_ID]
            + [IM_START_ID, ASSISTANT_ID, NL_ID]
            + image_tokens
        )
    return torch.tensor([seq], dtype=torch.long)


def _find_img_blocks(input_ids: torch.Tensor) -> List[Tuple[int, int]]:
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    starts = (input_ids == BOI_ID).nonzero(as_tuple=True)[0].tolist()
    ends = (input_ids == EOI_ID).nonzero(as_tuple=True)[0].tolist()
    return [(int(s), int(e)) for s, e in zip(starts, ends) if int(s) < int(e)]


def _build_edit_startend(
    input_ids: torch.Tensor,
    block_bidirectional_flags: List[bool],
    device=None,
) -> torch.Tensor:
    """FlashMask start/end rows aligned with editing training.

    Dense Pixel-DiT editing uses bidirectional attention inside ViT image blocks.
    The generated noisy block is also self-closed so later text cannot see its
    noisy content.
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    input_ids = input_ids.to(device)
    L = input_ids.shape[0]
    col_idx = torch.arange(L, device=input_ids.device, dtype=torch.int32)
    lts = torch.full((L,), max(L - 1, 0), device=input_ids.device, dtype=torch.int32)
    ute = col_idx.clone()

    # 按图像块出现顺序记录块内 attention 方向; True 表示块内双向,
    # 同时保持块外 causal 隔离。
    blocks = _find_img_blocks(input_ids)
    for block_idx, (s, e) in enumerate(blocks):
        block_ids = input_ids[s + 1: e]
        has_time = bool((block_ids == TIME_TOKEN_ID).any())
        bidirectional = (
            bool(block_bidirectional_flags[block_idx])
            if block_idx < len(block_bidirectional_flags)
            else has_time
        )
        if has_time:
            time_rel = (block_ids == TIME_TOKEN_ID).nonzero(as_tuple=True)[0][0].item()
            patch_start = s + 1 + time_rel + 1
        else:
            patch_start = s + 1
        patch_mask = (col_idx >= patch_start) & (col_idx < e)
        if bidirectional and patch_mask.any():
            ute[patch_mask] = int(patch_start)
        if has_time:
            cutoff = int(e + 1)
            for pos in range(s, e + 1):
                lts[pos] = min(int(lts[pos].item()), cutoff)

    return torch.stack([lts, ute], dim=-1).unsqueeze(0).unsqueeze(0)


def _build_edit_position_ids(
    input_ids: torch.Tensor,
    block_grids: List[List[List[int]]],
    device=None,
) -> torch.Tensor:
    """Build M-RoPE position ids for all IMG_PAD blocks."""
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    input_ids = input_ids.to(device)
    L = input_ids.shape[0]
    position_ids = torch.zeros(3, 1, L, dtype=torch.long, device=input_ids.device)

    img_blocks: List[Tuple[int, int]] = []
    in_block = False
    block_start = 0
    for i, tok in enumerate(input_ids.tolist()):
        if tok == IMG_PAD_ID:
            if not in_block:
                block_start = i
                in_block = True
        else:
            if in_block:
                img_blocks.append((block_start, i))
                in_block = False
    if in_block:
        img_blocks.append((block_start, L))

    pos = 0
    prev_end = 0
    for block_idx, (start, end) in enumerate(img_blocks):
        for i in range(prev_end, start):
            position_ids[:, 0, i] = pos
            pos += 1

        sub_grids = block_grids[block_idx] if block_idx < len(block_grids) else [[1, 1, end - start]]
        sub_start = start
        total_k = 0
        for grid in sub_grids:
            t, h, w = (int(grid[0]), int(grid[1]), int(grid[2]))
            k = t * h * w
            sub_end = sub_start + k
            offset = pos
            t_idx = torch.arange(t, device=input_ids.device).view(-1, 1).expand(-1, h * w).flatten()
            h_idx = torch.arange(h, device=input_ids.device).view(1, -1, 1).expand(t, -1, w).flatten()
            w_idx = torch.arange(w, device=input_ids.device).view(1, 1, -1).expand(t, h, -1).flatten()
            position_ids[0, 0, sub_start:sub_end] = offset + t_idx
            position_ids[1, 0, sub_start:sub_end] = offset + h_idx
            position_ids[2, 0, sub_start:sub_end] = offset + w_idx
            sub_start = sub_end
            pos = offset + max(t, h, w)
            total_k += k
        if total_k != end - start:
            raise ValueError(f"image block {block_idx} length {end - start} != grid tokens {total_k}")
        prev_end = end

    for i in range(prev_end, L):
        position_ids[:, 0, i] = pos
        pos += 1
    return position_ids


def _pad_ids(ids: torch.Tensor, max_len: int, device) -> torch.Tensor:
    if ids.dim() == 2:
        ids = ids[0]
    ids = ids.to(device)
    if ids.shape[0] >= max_len:
        return ids.unsqueeze(0)
    pad = torch.full((max_len - ids.shape[0],), EOS_ID, dtype=ids.dtype, device=device)
    return torch.cat([ids, pad], dim=0).unsqueeze(0)


def _extend_position_ids(position_ids: torch.Tensor, max_len: int, device) -> torch.Tensor:
    position_ids = position_ids.to(device)
    cur_len = position_ids.shape[-1]
    if cur_len >= max_len:
        return position_ids
    last = position_ids[:, :, -1:]
    extra = torch.arange(1, max_len - cur_len + 1, dtype=position_ids.dtype, device=device)
    extra = last + extra.view(1, 1, -1)
    return torch.cat([position_ids, extra], dim=-1)


def _pad_startend(startend: torch.Tensor, cur_len: int, max_len: int, device) -> torch.Tensor:
    startend = startend.to(device)
    if cur_len >= max_len:
        return startend
    pad_len = max_len - cur_len
    pad = torch.zeros((1, 1, pad_len, 2), dtype=startend.dtype, device=device)
    pad[0, 0, :, 0] = max_len
    pad[0, 0, :, 1] = torch.arange(cur_len, max_len, dtype=startend.dtype, device=device)
    return torch.cat([startend, pad], dim=2)


def _latent_to_vision_data(latents: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """Convert raw pixel patches [N, 768] to Qwen-VL ViT pixel_values [N, 1536]."""
    latents = latents.float()
    grid_thw = grid_thw.to(device=latents.device, dtype=torch.long)
    raw_sizes = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    chunks = []
    offset = 0
    for grid, raw_size in zip(grid_thw, raw_sizes):
        h = int(grid[1].item())
        w = int(grid[2].item())
        raw_size = int(raw_size)
        z = latents[offset: offset + raw_size].reshape(h, w, CHANNELS, PATCH_SIZE, PATCH_SIZE)
        offset += raw_size
        image = z.permute(2, 0, 3, 1, 4).reshape(CHANNELS, h * PATCH_SIZE, w * PATCH_SIZE)
        patches = image.reshape(
            1,
            CHANNELS,
            h // MERGE_SIZE,
            MERGE_SIZE,
            PATCH_SIZE,
            w // MERGE_SIZE,
            MERGE_SIZE,
            PATCH_SIZE,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        patches = patches.unsqueeze(6).expand(
            -1, -1, -1, -1, -1, -1, TEMPORAL_PATCH_SIZE, -1, -1
        )
        chunks.append(patches.reshape(raw_size, VISION_PATCH_DIM))
    return torch.cat(chunks, dim=0)


def _decode_pixel_latents(latents: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """Decode raw pixel patches [N, 768] to images [B, 3, H, W] in [0, 1]."""
    latents = latents.float()
    grid_thw = grid_thw.to(device=latents.device, dtype=torch.long)
    raw_sizes = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    images = []
    offset = 0
    for grid, raw_size in zip(grid_thw, raw_sizes):
        h = int(grid[1].item())
        w = int(grid[2].item())
        raw_size = int(raw_size)
        z = latents[offset: offset + raw_size]
        offset += raw_size
        z = z.reshape(h, w, CHANNELS, PATCH_SIZE, PATCH_SIZE)
        z = z.permute(2, 0, 3, 1, 4).reshape(CHANNELS, h * PATCH_SIZE, w * PATCH_SIZE)
        z = z * 0.5 + 0.5
        images.append(z.unsqueeze(0).clamp(0, 1))
    return torch.cat(images, dim=0)


def _resize_tensor_image_to_pil_size(image: torch.Tensor, pil_size: Tuple[int, int]) -> torch.Tensor:
    """将生成图 tensor resize 回 PIL 原图尺寸。

    Args:
        image: [3, H, W] 或 [1, 3, H, W], 值域 [0, 1]。
        pil_size: PIL 的 (width, height)。
    """
    width, height = int(pil_size[0]), int(pil_size[1])
    if image.dim() == 3:
        image_b = image.unsqueeze(0)
        squeeze = True
    elif image.dim() == 4 and image.shape[0] == 1:
        image_b = image
        squeeze = False
    else:
        raise ValueError(f"expected image shape [3,H,W] or [1,3,H,W], got {tuple(image.shape)}")
    if image_b.shape[-2:] != (height, width):
        image_b = F.interpolate(
            image_b.float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    image_b = image_b.clamp(0, 1)
    return image_b[0] if squeeze else image_b


def _load_image(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    image_path = str(image_obj)
    if image_path.startswith("http"):
        from megatron.energon.wrappers import SkipSample

        from data.transfusion.transfusion_data_cooker import load_images

        imgs = []
        for _ in range(5):
            try:
                imgs = load_images([image_path])
            except SkipSample:
                imgs = []
            if imgs:
                break
        if not imgs:
            raise RuntimeError(f"failed to load image: {image_path}")
        return imgs[0].convert("RGB")
    return open_image_checked(image_path).convert("RGB")


def _processor_image_to_pixel_values(
    processor,
    image: Image.Image,
    device,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kwargs = {}
    if min_pixels is not None and min_pixels > 0:
        kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None and max_pixels > 0:
        kwargs["max_pixels"] = int(max_pixels)
    inputs = processor(images=[image], videos=None, return_tensors="pt", **kwargs)
    return inputs["pixel_values"].to(device), inputs["image_grid_thw"].to(device)


def _resolve_gen_grid(ref_grid: torch.Tensor, args) -> torch.Tensor:
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("--height and --width must be set together")
        return torch.tensor([[1, int(args.height) * 2, int(args.width) * 2]], dtype=torch.long)
    if args.match_ref_resolution:
        return ref_grid.detach().cpu().long().clone()
    if args.resolution % 32 != 0:
        raise ValueError("--resolution must be divisible by 32")
    merged = args.resolution // 32
    return torch.tensor([[1, merged * 2, merged * 2]], dtype=torch.long)


def _select_cfg_modes(
    cfg_mode: str,
    cfg_scale: float,
    cfg_w_img: float,
) -> Tuple[List[str], bool]:
    """返回实际需要 forward 的 CFG 分支，以及 edit3 是否可退化为两分支。"""
    edit3_two_branch = False
    if cfg_mode == "none" or (cfg_scale <= 1.0 and cfg_mode == "text"):
        modes = ["none"]
    elif cfg_mode == "text":
        modes = ["none", "text"]
    elif cfg_mode == "edit3":
        # edit3 公式: full + w_text * (none - text) + w_img * (text - full)。
        # 当 w_img == 1 时 full 项完全抵消, 只需 none/text 两个分支。
        edit3_two_branch = abs(float(cfg_w_img) - 1.0) < 1e-6
        modes = ["none", "text"] if edit3_two_branch else ["none", "text", "full"]
    else:
        raise ValueError(f"unsupported cfg_mode: {cfg_mode}")
    return modes, edit3_two_branch


def _combine_cfg_predictions(
    mode_preds: Dict[str, torch.Tensor],
    cfg_mode: str,
    cfg_scale: float,
    cfg_w_text: float,
    cfg_w_img: float,
    edit3_two_branch: bool,
) -> torch.Tensor:
    """按 CFG 公式把各分支预测合成一个 x0/eps 预测。"""
    if "none" not in mode_preds:
        raise ValueError("missing required CFG branch: none")
    if cfg_mode == "none" or (cfg_scale <= 1.0 and cfg_mode == "text"):
        return mode_preds["none"]
    if cfg_mode == "text":
        if "text" not in mode_preds:
            raise ValueError("missing required CFG branch: text")
        return mode_preds["text"] + cfg_scale * (mode_preds["none"] - mode_preds["text"])
    if cfg_mode == "edit3" and edit3_two_branch:
        if "text" not in mode_preds:
            raise ValueError("missing required CFG branch: text")
        return mode_preds["text"] + cfg_w_text * (mode_preds["none"] - mode_preds["text"])
    if cfg_mode == "edit3":
        for mode in ("text", "full"):
            if mode not in mode_preds:
                raise ValueError(f"missing required CFG branch: {mode}")
        return (
            mode_preds["full"]
            + cfg_w_text * (mode_preds["none"] - mode_preds["text"])
            + cfg_w_img * (mode_preds["text"] - mode_preds["full"])
        )
    raise ValueError(f"unsupported cfg_mode: {cfg_mode}")


def _build_forward_batch(
    tokenizer,
    prompt: str,
    ref_pixel_values: torch.Tensor,
    ref_grid_thw: torch.Tensor,
    gen_latents: torch.Tensor,
    gen_grid_thw: torch.Tensor,
    t: torch.Tensor,
    cfg_modes: List[str],
    device,
    cfg_keep_chat_template: bool = False,
    predict_gate_mode: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    raw_gen_tokens = int((gen_grid_thw[:, 0] * gen_grid_thw[:, 1] * gen_grid_thw[:, 2]).sum().item())
    gen_h = int(gen_grid_thw[0, 1].item()) // MERGE_SIZE
    gen_w = int(gen_grid_thw[0, 2].item()) // MERGE_SIZE
    ref_h = int(ref_grid_thw[0, 1].item()) // MERGE_SIZE
    ref_w = int(ref_grid_thw[0, 2].item()) // MERGE_SIZE
    ref_tokens = ref_h * ref_w
    gen_tokens = gen_h * gen_w

    gen_noise_vision = _latent_to_vision_data(gen_latents, gen_grid_thw.to(device))
    gen_zero_vision = torch.zeros_like(gen_noise_vision)

    ids_list = []
    pos_list = []
    se_list = []
    pixel_values_list = []
    image_grid_list = []
    total_img_und_mask_list = []  # input-token 空间; True = clean ref token, False = noisy gen token
    image_generate_mask_list = []  # 此处为 input-token 空间; True = 生成图 token
    image_understand_mask_list = []  # 此处为 input-token 空间; True = ref/understanding 图像 token
    image_gen_grid_list = []  # 每张生成图的 raw grid, 与 noise_list 对齐
    noise_list = []  # 仅生成图使用的 raw pixel-patch noise/latent token
    timestep_token_values = []  # 每个 CFG branch/sample 对应一个 timestep 值

    for mode in cfg_modes:
        include_ref = mode != "full"
        ids = _build_edit_sequence(
            tokenizer=tokenizer,
            prompt=prompt,
            ref_tokens=ref_tokens if include_ref else 0,
            gen_h=gen_h,
            gen_w=gen_w,
            cfg_mode=mode,
            cfg_keep_chat_template=cfg_keep_chat_template,
            predict_gate_mode=predict_gate_mode,
        )
        block_grids = []
        block_bidir = []
        if include_ref:
            block_grids.append([[1, ref_h, ref_w]])
            block_bidir.append(True)
        block_grids.append([[1, gen_h, gen_w]])
        block_bidir.append(True)

        ids = ids.to(device)
        ids_list.append(ids)
        pos_list.append(_build_edit_position_ids(ids, block_grids, device=device))
        se_list.append(_build_edit_startend(ids, block_bidir, device=device))

        if include_ref:
            pixel_values_list.append(ref_pixel_values.to(device))
            pixel_values_list.append(gen_zero_vision)
            image_grid_list.append(ref_grid_thw.to(device))
            image_grid_list.append(gen_grid_thw.to(device))
            total_img_und_mask_list.append(torch.ones(ref_tokens, dtype=torch.bool, device=device))
            total_img_und_mask_list.append(torch.zeros(gen_tokens, dtype=torch.bool, device=device))
            image_generate_mask_list.append(torch.zeros(ref_tokens, dtype=torch.bool, device=device))
            image_generate_mask_list.append(torch.ones(gen_tokens, dtype=torch.bool, device=device))
            image_understand_mask_list.append(torch.ones(ref_tokens, dtype=torch.bool, device=device))
            image_understand_mask_list.append(torch.zeros(gen_tokens, dtype=torch.bool, device=device))
        else:
            pixel_values_list.append(gen_zero_vision)
            image_grid_list.append(gen_grid_thw.to(device))
            total_img_und_mask_list.append(torch.zeros(gen_tokens, dtype=torch.bool, device=device))
            image_generate_mask_list.append(torch.ones(gen_tokens, dtype=torch.bool, device=device))
            image_understand_mask_list.append(torch.zeros(gen_tokens, dtype=torch.bool, device=device))

        image_gen_grid_list.append(gen_grid_thw.to(device))
        noise_list.append(gen_noise_vision)
        timestep_token_values.append(t)

    max_len = max(ids.shape[1] for ids in ids_list)
    padded_ids = []
    padded_pos = []
    padded_se = []
    for ids, pos, se in zip(ids_list, pos_list, se_list):
        cur_len = ids.shape[1]
        padded_ids.append(_pad_ids(ids, max_len, device))
        padded_pos.append(_extend_position_ids(pos, max_len, device))
        padded_se.append(_pad_startend(se, cur_len, max_len, device))

    noise = torch.cat(noise_list, dim=0).to(device)
    return {
        "input_ids": torch.cat(padded_ids, dim=0),
        "position_ids": torch.cat(padded_pos, dim=1),
        "startend_row_indices": torch.cat(padded_se, dim=0),
        "pixel_values": torch.cat(pixel_values_list, dim=0).to(device),
        "image_grid_thw": torch.cat(image_grid_list, dim=0).to(device),
        "image_input_mask": torch.cat(padded_ids, dim=0).eq(IMG_PAD_ID),
        "image_generate_mask": torch.cat(image_generate_mask_list, dim=0),
        "image_understand_mask": torch.cat(image_understand_mask_list, dim=0),
        "total_img_und_mask": torch.cat(total_img_und_mask_list, dim=0),
        "image_gen_grid_thw": torch.cat(image_gen_grid_list, dim=0).to(device),
        "timesteps": t.to(device).expand(noise.shape[0]),
        "timesteps_samples": torch.stack(timestep_token_values).to(device).view(-1),
        "sigmas": torch.ones(noise.shape[0], device=device, dtype=torch.float32),
        "noise": noise,
        "task_type": torch.full((len(cfg_modes),), TASK_EDITING, dtype=torch.long, device=device),
        "raw_gen_tokens": raw_gen_tokens,
    }


def _build_packed_forward_batch(
    tokenizer,
    states: List[Dict],
    t: torch.Tensor,
    cfg_modes: List[str],
    device,
    cfg_keep_chat_template: bool = False,
    predict_gate_mode: Optional[bool] = None,
) -> Dict:
    """把多个 edit 样本及其 CFG 分支 pack 成一次 model forward。

    每个 state 表示一个待编辑样本; 同一个 denoise step 下, 所有样本共享 t,
    但各自可以有不同的 ref/gen 分辨率, 因此只 pad 文本序列, 图像 token 仍按
    packing 方式连续拼接。
    """
    if not states:
        raise ValueError("states must not be empty")

    t_scalar = t.to(device).view(-1)[0]
    ids_list = []
    pos_list = []
    se_list = []
    pixel_values_list = []
    image_grid_list = []
    total_img_und_mask_list = []  # input-token 空间; True = clean ref token, False = noisy gen token
    image_generate_mask_list = []  # input-token 空间; True = 生成图 token
    image_understand_mask_list = []  # input-token 空间; True = ref/understanding 图像 token
    image_gen_grid_list = []  # 每个 CFG branch 对应一张生成图 raw grid, 与 noise_list 对齐
    noise_list = []  # 每个 CFG branch 的 noisy gen raw pixel-patch token
    timestep_token_values = []  # 每个 CFG branch/sample 对应一个 timestep token 值
    branch_records = []  # 记录 image_pred 中每段 raw token 属于哪个样本和 CFG 分支
    branch_raw_gen_tokens: List[int] = []

    for sample_idx, state in enumerate(states):
        prompt = state["prompt"]
        ref_pixel_values = state["ref_pixel_values"].to(device)
        ref_grid_thw = state["ref_grid_thw"].to(device)
        gen_latents = state["latents"].to(device)
        gen_grid_thw = state["gen_grid_thw"].to(device)

        raw_gen_tokens = int((gen_grid_thw[:, 0] * gen_grid_thw[:, 1] * gen_grid_thw[:, 2]).sum().item())
        gen_h = int(gen_grid_thw[0, 1].item()) // MERGE_SIZE
        gen_w = int(gen_grid_thw[0, 2].item()) // MERGE_SIZE
        ref_h = int(ref_grid_thw[0, 1].item()) // MERGE_SIZE
        ref_w = int(ref_grid_thw[0, 2].item()) // MERGE_SIZE
        ref_tokens = ref_h * ref_w
        gen_tokens = gen_h * gen_w

        gen_noise_vision = _latent_to_vision_data(gen_latents, gen_grid_thw)
        gen_zero_vision = torch.zeros_like(gen_noise_vision)

        for mode in cfg_modes:
            include_ref = mode != "full"
            ids = _build_edit_sequence(
                tokenizer=tokenizer,
                prompt=prompt,
                ref_tokens=ref_tokens if include_ref else 0,
                gen_h=gen_h,
                gen_w=gen_w,
                cfg_mode=mode,
                cfg_keep_chat_template=cfg_keep_chat_template,
                predict_gate_mode=predict_gate_mode,
            )
            block_grids = []
            block_bidir = []
            if include_ref:
                block_grids.append([[1, ref_h, ref_w]])
                block_bidir.append(True)
            block_grids.append([[1, gen_h, gen_w]])
            block_bidir.append(True)

            ids = ids.to(device)
            ids_list.append(ids)
            pos_list.append(_build_edit_position_ids(ids, block_grids, device=device))
            se_list.append(_build_edit_startend(ids, block_bidir, device=device))

            if include_ref:
                pixel_values_list.append(ref_pixel_values)
                pixel_values_list.append(gen_zero_vision)
                image_grid_list.append(ref_grid_thw)
                image_grid_list.append(gen_grid_thw)
                total_img_und_mask_list.append(torch.ones(ref_tokens, dtype=torch.bool, device=device))
                total_img_und_mask_list.append(torch.zeros(gen_tokens, dtype=torch.bool, device=device))
                image_generate_mask_list.append(torch.zeros(ref_tokens, dtype=torch.bool, device=device))
                image_generate_mask_list.append(torch.ones(gen_tokens, dtype=torch.bool, device=device))
                image_understand_mask_list.append(torch.ones(ref_tokens, dtype=torch.bool, device=device))
                image_understand_mask_list.append(torch.zeros(gen_tokens, dtype=torch.bool, device=device))
            else:
                pixel_values_list.append(gen_zero_vision)
                image_grid_list.append(gen_grid_thw)
                total_img_und_mask_list.append(torch.zeros(gen_tokens, dtype=torch.bool, device=device))
                image_generate_mask_list.append(torch.ones(gen_tokens, dtype=torch.bool, device=device))
                image_understand_mask_list.append(torch.zeros(gen_tokens, dtype=torch.bool, device=device))

            image_gen_grid_list.append(gen_grid_thw)
            noise_list.append(gen_noise_vision)
            timestep_token_values.append(t_scalar)
            branch_records.append({"sample_idx": sample_idx, "mode": mode})
            branch_raw_gen_tokens.append(raw_gen_tokens)

    max_len = max(ids.shape[1] for ids in ids_list)
    padded_ids = []
    padded_pos = []
    padded_se = []
    for ids, pos, se in zip(ids_list, pos_list, se_list):
        cur_len = ids.shape[1]
        padded_ids.append(_pad_ids(ids, max_len, device))
        padded_pos.append(_extend_position_ids(pos, max_len, device))
        padded_se.append(_pad_startend(se, cur_len, max_len, device))

    noise = torch.cat(noise_list, dim=0).to(device)
    return {
        "input_ids": torch.cat(padded_ids, dim=0),
        "position_ids": torch.cat(padded_pos, dim=1),
        "startend_row_indices": torch.cat(padded_se, dim=0),
        "pixel_values": torch.cat(pixel_values_list, dim=0).to(device),
        "image_grid_thw": torch.cat(image_grid_list, dim=0).to(device),
        "image_input_mask": torch.cat(padded_ids, dim=0).eq(IMG_PAD_ID),
        "image_generate_mask": torch.cat(image_generate_mask_list, dim=0),
        "image_understand_mask": torch.cat(image_understand_mask_list, dim=0),
        "total_img_und_mask": torch.cat(total_img_und_mask_list, dim=0),
        "image_gen_grid_thw": torch.cat(image_gen_grid_list, dim=0).to(device),
        "timesteps": t_scalar.view(1).expand(noise.shape[0]),
        "timesteps_samples": torch.stack(timestep_token_values).to(device).view(-1),
        "sigmas": torch.ones(noise.shape[0], device=device, dtype=torch.float32),
        "noise": noise,
        "task_type": torch.full((len(branch_records),), TASK_EDITING, dtype=torch.long, device=device),
        "branch_records": branch_records,
        "branch_raw_gen_tokens": branch_raw_gen_tokens,
    }


def _forward_edit_model(
    model,
    scheduler,
    batch: Dict,
) -> torch.Tensor:
    output = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        labels=None,
        attention_mask=None,
        extra_block_kwargs={
            "startend_row_indices": batch["startend_row_indices"],
            "causal": False,
        },
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
        image_input_mask=batch["image_input_mask"],
        image_generate_mask=batch["image_generate_mask"],
        image_understand_mask=batch["image_understand_mask"],
        image_gen_grid_thw=batch["image_gen_grid_thw"],
        total_img_und_mask=batch["total_img_und_mask"],
        timesteps_samples=batch["timesteps_samples"],
        timesteps=batch["timesteps"],
        sigmas=batch["sigmas"],
        noise=batch["noise"],
        diff_scheduler=scheduler,
        task_type=batch["task_type"],
        edit_clean_img_mask=None,
        predict_x0=True,
    )
    return output["image_pred"].float()


def _predict_edit(
    model,
    tokenizer,
    scheduler,
    prompt: str,
    ref_pixel_values: torch.Tensor,
    ref_grid_thw: torch.Tensor,
    gen_latents: torch.Tensor,
    gen_grid_thw: torch.Tensor,
    t: torch.Tensor,
    cfg_mode: str,
    cfg_scale: float,
    cfg_w_text: float,
    cfg_w_img: float,
    device,
    cfg_keep_chat_template: bool = False,
    predict_gate_mode: Optional[bool] = None,
) -> torch.Tensor:
    modes, edit3_two_branch = _select_cfg_modes(
        cfg_mode=cfg_mode,
        cfg_scale=cfg_scale,
        cfg_w_img=cfg_w_img,
    )

    batch = _build_forward_batch(
        tokenizer=tokenizer,
        prompt=prompt,
        ref_pixel_values=ref_pixel_values,
        ref_grid_thw=ref_grid_thw,
        gen_latents=gen_latents,
        gen_grid_thw=gen_grid_thw,
        t=t,
        cfg_modes=modes,
        device=device,
        cfg_keep_chat_template=cfg_keep_chat_template,
        predict_gate_mode=predict_gate_mode,
    )

    pred = _forward_edit_model(model=model, scheduler=scheduler, batch=batch)
    raw_gen_tokens = int(batch["raw_gen_tokens"])
    pred = pred.view(len(modes), raw_gen_tokens, -1)

    mode_preds = {mode: pred[idx] for idx, mode in enumerate(modes)}
    return _combine_cfg_predictions(
        mode_preds=mode_preds,
        cfg_mode=cfg_mode,
        cfg_scale=cfg_scale,
        cfg_w_text=cfg_w_text,
        cfg_w_img=cfg_w_img,
        edit3_two_branch=edit3_two_branch,
    )


def _predict_edit_packed(
    model,
    tokenizer,
    scheduler,
    states: List[Dict],
    t: torch.Tensor,
    cfg_mode: str,
    cfg_scale: float,
    cfg_w_text: float,
    cfg_w_img: float,
    device,
    cfg_keep_chat_template: bool = False,
    predict_gate_mode: Optional[bool] = None,
) -> List[torch.Tensor]:
    modes, edit3_two_branch = _select_cfg_modes(
        cfg_mode=cfg_mode,
        cfg_scale=cfg_scale,
        cfg_w_img=cfg_w_img,
    )
    batch = _build_packed_forward_batch(
        tokenizer=tokenizer,
        states=states,
        t=t,
        cfg_modes=modes,
        device=device,
        cfg_keep_chat_template=cfg_keep_chat_template,
        predict_gate_mode=predict_gate_mode,
    )
    pred = _forward_edit_model(model=model, scheduler=scheduler, batch=batch)
    pred_chunks = torch.split(pred, batch["branch_raw_gen_tokens"], dim=0)

    mode_preds_by_sample: List[Dict[str, torch.Tensor]] = [dict() for _ in states]
    for chunk, record in zip(pred_chunks, batch["branch_records"]):
        mode_preds_by_sample[int(record["sample_idx"])][str(record["mode"])] = chunk

    return [
        _combine_cfg_predictions(
            mode_preds=mode_preds,
            cfg_mode=cfg_mode,
            cfg_scale=cfg_scale,
            cfg_w_text=cfg_w_text,
            cfg_w_img=cfg_w_img,
            edit3_two_branch=edit3_two_branch,
        )
        for mode_preds in mode_preds_by_sample
    ]


def _scheduler_step(
    scheduler,
    scheduler_type: str,
    pred: torch.Tensor,
    t: torch.Tensor,
    latents: torch.Tensor,
    x0_history: List[Tuple[float, torch.Tensor]],
    predict_x0: bool = True,
) -> torch.Tensor:
    if scheduler_type == "euler":
        return scheduler.step(pred, t, latents, predict_x0=predict_x0, predict_x0_sigma_min=None)

    timestep_id = torch.argmin((scheduler.timesteps - t.to(scheduler.timesteps.device)).abs())
    sigma = scheduler.sigmas[timestep_id]
    if predict_x0:
        x0_pred = pred
    else:
        x0_pred = latents - sigma * pred
    sigma_next = (
        torch.zeros_like(sigma)
        if timestep_id + 1 >= len(scheduler.timesteps)
        else scheduler.sigmas[timestep_id + 1]
    )
    eps_safe = 1e-8
    lam = torch.log((1 - sigma).clamp(min=eps_safe) / sigma.clamp(min=eps_safe))
    lam_next = torch.log((1 - sigma_next).clamp(min=eps_safe) / sigma_next.clamp(min=eps_safe))
    h = lam_next - lam
    x0_history.append((lam.item(), x0_pred))

    order = 2 if scheduler_type == "dpm2" else 3
    eff_order = min(order, len(x0_history))
    if eff_order == 1:
        latents = (1 - sigma_next) * x0_pred + sigma_next * (
            latents - (1 - sigma) * x0_pred
        ) / sigma.clamp(min=eps_safe)
    elif eff_order == 2:
        lam_prev, x0_prev = x0_history[-2]
        h_prev = lam.item() - lam_prev
        r = h.item() / h_prev if h_prev != 0 else 0.0
        x0_corrected = x0_pred + (1.0 / (2.0 * r)) * (x0_pred - x0_prev)
        latents = (1 - sigma_next) * x0_corrected + sigma_next * (
            latents - (1 - sigma) * x0_pred
        ) / sigma.clamp(min=eps_safe)
    else:
        lam_2, x0_2 = x0_history[-2]
        lam_3, x0_3 = x0_history[-3]
        h_1 = lam.item() - lam_2
        h_2 = lam_2 - lam_3
        r1 = h.item() / h_1 if h_1 != 0 else 0.0
        r2 = h_1 / h_2 if h_2 != 0 else 0.0
        d1_0 = (1.0 / r1) * (x0_pred - x0_2)
        d1_1 = (1.0 / r2) * (x0_2 - x0_3)
        d1 = d1_0 + (r1 / (r1 + r2)) * (d1_0 - d1_1)
        d2 = (1.0 / (r1 + r2)) * (d1_0 - d1_1)
        x0_corrected = x0_pred + d1 + d2
        latents = (1 - sigma_next) * x0_corrected + sigma_next * (
            latents - (1 - sigma) * x0_pred
        ) / sigma.clamp(min=eps_safe)

    if len(x0_history) > 3:
        x0_history.pop(0)
    return latents


def _make_latent_generator(device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator_device = "cuda" if torch.device(device).type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _prepare_edit_state(
    processor,
    prompt: str,
    ref_image: Image.Image,
    args,
    device,
    seed: Optional[int] = None,
) -> Dict:
    ref_pixel_values, ref_grid_thw = _processor_image_to_pixel_values(
        processor,
        ref_image,
        device,
        min_pixels=args.ref_min_pixels,
        max_pixels=args.ref_max_pixels,
    )
    gen_grid_thw = _resolve_gen_grid(ref_grid_thw, args).to(device)

    raw_gen_tokens = int((gen_grid_thw[:, 0] * gen_grid_thw[:, 1] * gen_grid_thw[:, 2]).sum().item())
    generator = _make_latent_generator(device, seed)
    latents = torch.randn(
        raw_gen_tokens,
        PIXEL_DIM,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    if args.dynamic_noise_scale:
        scale = math.sqrt(raw_gen_tokens / args.dynamic_noise_ref_tokens) * args.dynamic_noise_base_scale
        latents = latents * scale

    return {
        "prompt": prompt,
        "ref_pixel_values": ref_pixel_values,
        "ref_grid_thw": ref_grid_thw,
        "gen_grid_thw": gen_grid_thw,
        "latents": latents,
        "x0_history": [],
    }


@torch.no_grad()
def sample_edit_images_packed(
    model,
    tokenizer,
    processor,
    jobs: List[Dict],
    args,
    device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """多样本 packing 推理; jobs 中每个元素需要包含 prompt/ref_image。"""
    states = [
        _prepare_edit_state(
            processor=processor,
            prompt=job["prompt"],
            ref_image=job["ref_image"],
            args=args,
            device=device,
            seed=job.get("seed"),
        )
        for job in jobs
    ]

    scheduler = FlowMatchScheduler(
        num_train_timesteps=1000,
        num_inference_steps=args.steps,
        shift=args.shift,
        extra_one_step=True,
    )
    scheduler.set_timesteps(args.steps)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)

    for step_idx, t in enumerate(scheduler.timesteps):
        t_tensor = t.to(device).unsqueeze(0)
        preds = _predict_edit_packed(
            model=model,
            tokenizer=tokenizer,
            scheduler=scheduler,
            states=states,
            t=t_tensor,
            cfg_mode=args.cfg_mode,
            cfg_scale=args.cfg_scale,
            cfg_w_text=args.cfg_w_text,
            cfg_w_img=args.cfg_w_img,
            device=device,
            cfg_keep_chat_template=args.cfg_keep_chat_template,
            predict_gate_mode=args.predict_gate_mode,
        )
        for state, pred in zip(states, preds):
            state["latents"] = _scheduler_step(
                scheduler=scheduler,
                scheduler_type=args.scheduler,
                pred=pred,
                t=t,
                latents=state["latents"],
                x0_history=state["x0_history"],
                predict_x0=True,
            )
        if args.log_interval > 0 and (
            step_idx % args.log_interval == 0 or step_idx == len(scheduler.timesteps) - 1
        ):
            lat_std = torch.stack([state["latents"].std() for state in states]).mean()
            pred_std = torch.stack([pred.std() for pred in preds]).mean()
            logger.info(
                "packed denoise step=%d/%d t=%.2f batch=%d latents_std=%.4f pred_std=%.4f",
                step_idx,
                len(scheduler.timesteps),
                float(t.item()),
                len(states),
                float(lat_std.item()),
                float(pred_std.item()),
            )

    return [
        (_decode_pixel_latents(state["latents"], state["gen_grid_thw"])[0], state["gen_grid_thw"])
        for state in states
    ]


@torch.no_grad()
def sample_edit_image(
    model,
    tokenizer,
    processor,
    prompt: str,
    ref_image: Image.Image,
    args,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """sample_edit_image"""
    ref_pixel_values, ref_grid_thw = _processor_image_to_pixel_values(
        processor,
        ref_image,
        device,
        min_pixels=args.ref_min_pixels,
        max_pixels=args.ref_max_pixels,
    )
    gen_grid_thw = _resolve_gen_grid(ref_grid_thw, args).to(device)

    raw_gen_tokens = int((gen_grid_thw[:, 0] * gen_grid_thw[:, 1] * gen_grid_thw[:, 2]).sum().item())
    latents = torch.randn(raw_gen_tokens, PIXEL_DIM, device=device, dtype=torch.float32)
    if args.dynamic_noise_scale:
        scale = math.sqrt(raw_gen_tokens / args.dynamic_noise_ref_tokens) * args.dynamic_noise_base_scale
        latents = latents * scale

    scheduler = FlowMatchScheduler(
        num_train_timesteps=1000,
        num_inference_steps=args.steps,
        shift=args.shift,
        extra_one_step=True,
    )
    scheduler.set_timesteps(args.steps)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    x0_history: List[Tuple[float, torch.Tensor]] = []

    for step_idx, t in enumerate(scheduler.timesteps):
        t_tensor = t.to(device).unsqueeze(0)
        pred = _predict_edit(
            model=model,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=prompt,
            ref_pixel_values=ref_pixel_values,
            ref_grid_thw=ref_grid_thw,
            gen_latents=latents,
            gen_grid_thw=gen_grid_thw,
            t=t_tensor,
            cfg_mode=args.cfg_mode,
            cfg_scale=args.cfg_scale,
            cfg_w_text=args.cfg_w_text,
            cfg_w_img=args.cfg_w_img,
            device=device,
            cfg_keep_chat_template=args.cfg_keep_chat_template,
            predict_gate_mode=args.predict_gate_mode,
        )
        latents = _scheduler_step(
            scheduler=scheduler,
            scheduler_type=args.scheduler,
            pred=pred,
            t=t,
            latents=latents,
            x0_history=x0_history,
            predict_x0=True,
        )
        if args.log_interval > 0 and (
            step_idx % args.log_interval == 0 or step_idx == len(scheduler.timesteps) - 1
        ):
            logger.info(
                "denoise step=%d/%d t=%.2f latents_mean=%.4f latents_std=%.4f pred_std=%.4f",
                step_idx,
                len(scheduler.timesteps),
                float(t.item()),
                float(latents.mean().item()),
                float(latents.std().item()),
                float(pred.std().item()),
            )

    return _decode_pixel_latents(latents, gen_grid_thw)[0], gen_grid_thw


def _apply_pixel_dit_ckpt_config(args, ckpt_dir: str) -> None:
    cfg = _read_pixel_dit_config_from_ckpt(ckpt_dir)
    if cfg is None:
        return
    if cfg.get("num_diffusion_layers") is not None:
        args.num_diffusion_layers = int(cfg["num_diffusion_layers"])
    if cfg.get("pixel_dit_hidden_size") is not None:
        args.pixel_dit_hidden_size = int(cfg["pixel_dit_hidden_size"])
    if cfg.get("num_diffusion_heads") is not None:
        args.num_diffusion_heads = int(cfg["num_diffusion_heads"])
    if cfg.get("num_diffusion_kv_heads") is not None:
        args.num_diffusion_kv_heads = int(cfg["num_diffusion_kv_heads"])
    if cfg.get("diffusion_intermediate_size") is not None:
        args.diffusion_intermediate_size = int(cfg["diffusion_intermediate_size"])
    if cfg.get("pixel_dit_gate_inject_layer") is not None:
        args.pixel_dit_gate_inject_layer = int(cfg["pixel_dit_gate_inject_layer"])
    if cfg.get("pixel_dit_gate_input") is not None:
        args.pixel_dit_gate_input = str(cfg["pixel_dit_gate_input"])
    if cfg.get("use_pixel_dit_pixel_residual") is not None:
        args.use_pixel_dit_pixel_residual = bool(
            cfg["use_pixel_dit_pixel_residual"]
        )
    if cfg.get("pixel_dit_ref_gate_mode") is not None:
        args.pixel_dit_ref_gate_mode = str(cfg["pixel_dit_ref_gate_mode"])
    if cfg.get("dit_head_impl") is not None:
        args.dit_head_impl = str(cfg["dit_head_impl"])
    if cfg.get("dit_block_adaln") is not None:
        args.dit_block_adaln = bool(cfg["dit_block_adaln"])
    if cfg.get("adaln_post_modulation") is not None:
        args.adaln_post_modulation = bool(cfg["adaln_post_modulation"])
    if cfg.get("predict_gate_mode") is not None and not getattr(
        args, "_predict_gate_cli_override", False
    ):
        args.predict_gate_mode = bool(cfg["predict_gate_mode"])
    if cfg.get("use_dynamic_noise_scale") is not None:
        args.dynamic_noise_scale = bool(cfg["use_dynamic_noise_scale"])
    if cfg.get("dynamic_noise_ref_tokens") is not None:
        args.dynamic_noise_ref_tokens = int(cfg["dynamic_noise_ref_tokens"])
    if cfg.get("dynamic_noise_base_scale") is not None:
        args.dynamic_noise_base_scale = float(cfg["dynamic_noise_base_scale"])


def build_pixel_dit_from_args(args, device, dtype=torch.bfloat16):
    """Build the model; ``args.ckpt=None`` keeps CLI architecture flags and skips loading."""
    patch_qwen3vl_rotary_packed_seq_compat()
    ckpt_dir = None
    if args.ckpt is not None:
        ckpt_dir = _resolve_megatron_ckpt_dir(args.ckpt, args.iteration)
        _apply_pixel_dit_ckpt_config(args, ckpt_dir)
    model, _ = build_model_pixel_dit(
            hf_path=args.hf_path,
            ckpt_path=args.ckpt,
            num_diffusion_layers=args.num_diffusion_layers,
            pixel_dit_hidden_size=args.pixel_dit_hidden_size,
            num_diffusion_heads=args.num_diffusion_heads,
            num_diffusion_kv_heads=args.num_diffusion_kv_heads,
            diffusion_intermediate_size=args.diffusion_intermediate_size,
            pixel_dit_gate_inject_layer=args.pixel_dit_gate_inject_layer,
            pixel_dit_gate_input=args.pixel_dit_gate_input,
            use_pixel_dit_pixel_residual=args.use_pixel_dit_pixel_residual,
            pixel_dit_ref_gate_mode=args.pixel_dit_ref_gate_mode,
            dit_head_impl=args.dit_head_impl,
            dit_block_adaln=args.dit_block_adaln,
            adaln_post_modulation=args.adaln_post_modulation,
            device=device,
            dtype=dtype,
            iteration=args.iteration,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        )

    model.eval()
    return model, ckpt_dir


def _load_items(args) -> List[Dict]:
    if args.ref_image and args.prompt:
        return [{
            "key": "quick_0000",
            "instruction": args.prompt,
            "image": _load_image(args.ref_image),
            "task_type": "quick",
            "instruction_language": "unknown",
        }]

    from datasets import load_from_disk

    ds = load_from_disk(args.data_path)
    total = len(ds) if args.num_samples < 0 else min(args.num_samples, len(ds))
    items = []
    for idx in range(total):
        item = ds[idx]
        image_obj = item["input_image_raw"] if args.use_raw_image else item["input_image"]
        items.append({
            "key": str(item["key"]),
            "instruction": item["instruction"],
            "image": _load_image(image_obj),
            "task_type": item.get("task_type", "unknown"),
            "instruction_language": item.get("instruction_language", "unknown"),
        })
    return items


def _output_path(args, item: Dict, repeat_id: int) -> str:
    if args.ref_image and args.prompt:
        name = f"{item['key']}_{repeat_id}.png" if args.repeat_times > 1 else f"{item['key']}.png"
        return os.path.join(args.out_dir, "images", name)
    name = f"{item['key']}_{repeat_id}.png" if args.repeat_times > 1 else f"{item['key']}.png"
    return os.path.join(
        args.out_dir,
        "images",
        str(item.get("task_type", "unknown")),
        str(item.get("instruction_language", "unknown")),
        name,
    )


def _write_metadata(args, items: List[Dict], rank: int) -> None:
    if not is_main(rank) or not args.save_images:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for item in items:
            entry = {
                "key": item["key"],
                "instruction": item["instruction"],
                "images": [
                    os.path.relpath(_output_path(args, item, r), args.out_dir)
                    for r in range(args.repeat_times)
                ],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser("Qwen3-VL dense Pixel-DiT edit inference")
    parser.add_argument("--model_mode", choices=["pixel_dit"], default="pixel_dit")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hf_path", required=True)
    parser.add_argument("--out_dir", default="eval_results_i2i_pixel_dit")
    parser.add_argument("--iteration", type=int, default=None)

    parser.add_argument("--data_path", default="")
    parser.add_argument("--ref_image", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--use_raw_image", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="number of edit samples packed per denoise step")
    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--save_ref", action="store_true", default=False)

    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["euler", "dpm2", "dpm3"], default="euler")
    parser.add_argument("--cfg_mode", choices=["none", "text", "edit3"], default="edit3")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--cfg_w_text", type=float, default=4.0)
    parser.add_argument("--cfg_w_img", type=float, default=1.0)
    parser.add_argument(
        "--cfg_keep_chat_template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the legacy empty user/assistant template in edit full-uncond; "
             "otherwise use the T2I-aligned image-only layout.",
    )
    parser.add_argument(
        "--predict_gate_mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="V5 image gate: auto-detect (default), force enable, or disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=0)

    parser.add_argument("--match_ref_resolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--height", type=int, default=None, help="merged grid H; overrides ref/resolution")
    parser.add_argument("--width", type=int, default=None, help="merged grid W; overrides ref/resolution")
    parser.add_argument("--ref_min_pixels", type=int, default=None)
    parser.add_argument("--ref_max_pixels", type=int, default=None)

    parser.add_argument("--dynamic_noise_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic_noise_ref_tokens", type=int, default=256)
    parser.add_argument("--dynamic_noise_base_scale", type=float, default=1.0)

    add_pixel_dit_arch_args(parser)
    parser.add_argument("--diffusion_head_recompute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model_name", default="")
    return parser.parse_args()


def add_pixel_dit_arch_args(parser):
    """Architecture and parallelism flags; run_config.yaml values override them."""
    parser.add_argument("--num_diffusion_layers", type=int, default=10)
    parser.add_argument("--pixel_dit_hidden_size", type=int, default=0)
    parser.add_argument("--num_diffusion_heads", type=int, default=0)
    parser.add_argument("--num_diffusion_kv_heads", type=int, default=0)
    parser.add_argument("--diffusion_intermediate_size", type=int, default=0)
    parser.add_argument("--pixel_dit_gate_inject_layer", type=int, default=3)
    parser.add_argument(
        "--pixel_dit_gate_input", choices=["pixel", "vit"], default="pixel"
    )
    parser.add_argument(
        "--use_pixel_dit_pixel_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pixel_dit_ref_gate_mode", choices=["none", "shared", "separate"], default="none")
    parser.add_argument("--dit_head_impl", choices=["torch", "megatron"], default="torch")
    parser.add_argument("--dit_block_adaln", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaln_post_modulation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_model_parallel_size", type=int, default=1)
    parser.add_argument("--expert_model_parallel_size", type=int, default=1)


@torch.no_grad()
def main():
    args = parse_args()
    args._predict_gate_cli_override = any(
        arg == "--predict_gate_mode" or arg.startswith("--predict_gate_mode=")
        or arg == "--no-predict_gate_mode" or arg.startswith("--no-predict_gate_mode=")
        for arg in sys.argv[1:]
    )
    is_ddp, rank, _world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model, ckpt_dir = build_pixel_dit_from_args(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
    processor = Qwen2VLImageProcessor.from_pretrained(args.hf_path)

    from megatron.core import parallel_state
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world = parallel_state.get_data_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    torch.manual_seed(args.seed + dp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + dp_rank)

    if is_main(rank):
        logger.info(
            "Edit inference mode=%s ckpt=%s steps=%d shift=%.3f cfg_mode=%s batch_size=%d",
            args.model_mode,
            ckpt_dir,
            args.steps,
            args.shift,
            args.cfg_mode,
            args.batch_size,
        )
        logger.info("predict_gate_mode=%s (auto/forced/disabled)", args.predict_gate_mode)

    items = _load_items(args)
    _write_metadata(args, items, rank)
    local_items = items[dp_rank::dp_world] if dp_world > 1 else items

    generated = 0
    skipped = 0
    batch_size = max(1, int(args.batch_size))
    if batch_size == 1:
        for item in tqdm(local_items, disable=not is_main(rank)):
            for repeat_id in range(args.repeat_times):
                out_path = _output_path(args, item, repeat_id)
                if args.resume and os.path.exists(out_path):
                    skipped += 1
                    continue
                if repeat_id > 0:
                    torch.manual_seed(args.seed + dp_rank + repeat_id * 100003)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(args.seed + dp_rank + repeat_id * 100003)

                image, _gen_grid = sample_edit_image(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    prompt=item["instruction"],
                    ref_image=item["image"],
                    args=args,
                    device=device,
                )
                generated += 1
                if args.save_images and tp_rank == 0:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    image_to_save = _resize_tensor_image_to_pil_size(
                        image.clamp(0, 1),
                        item["image"].size,
                    )
                    save_image(image_to_save, out_path)
                    if args.save_ref:
                        ref_path = os.path.splitext(out_path)[0] + "_ref.png"
                        item["image"].save(ref_path)
    else:
        pending_jobs = []
        for item in local_items:
            for repeat_id in range(args.repeat_times):
                out_path = _output_path(args, item, repeat_id)
                if args.resume and os.path.exists(out_path):
                    skipped += 1
                    continue
                repeat_seed = None
                if repeat_id > 0:
                    repeat_seed = args.seed + dp_rank + repeat_id * 100003
                pending_jobs.append({
                    "item": item,
                    "repeat_id": repeat_id,
                    "out_path": out_path,
                    "prompt": item["instruction"],
                    "ref_image": item["image"],
                    "seed": repeat_seed,
                })

        for start in tqdm(range(0, len(pending_jobs), batch_size), disable=not is_main(rank)):
            jobs = pending_jobs[start:start + batch_size]
            results = sample_edit_images_packed(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                jobs=jobs,
                args=args,
                device=device,
            )
            for job, (image, _gen_grid) in zip(jobs, results):
                generated += 1
                if args.save_images and tp_rank == 0:
                    item = job["item"]
                    out_path = job["out_path"]
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    image_to_save = _resize_tensor_image_to_pil_size(
                        image.clamp(0, 1),
                        item["image"].size,
                    )
                    save_image(image_to_save, out_path)
                    if args.save_ref:
                        ref_path = os.path.splitext(out_path)[0] + "_ref.png"
                        item["image"].save(ref_path)

    if is_ddp:
        dist.barrier()

    if is_main(rank):
        os.makedirs(args.out_dir, exist_ok=True)
        summary = {
            "model_mode": args.model_mode,
            "ckpt": ckpt_dir,
            "out_dir": args.out_dir,
            "num_items": len(items),
            "generated_local": generated,
            "skipped_local": skipped,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "shift": args.shift,
            "cfg_mode": args.cfg_mode,
            "cfg_scale": args.cfg_scale,
            "cfg_w_text": args.cfg_w_text,
            "cfg_w_img": args.cfg_w_img,
            "cfg_keep_chat_template": args.cfg_keep_chat_template,
            "predict_gate_mode": args.predict_gate_mode,
        }
        with open(os.path.join(args.out_dir, "inference_results.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Done. generated=%d skipped=%d out_dir=%s", generated, skipped, args.out_dir)

    cleanup_dist(is_ddp)


if __name__ == "__main__":
    main()
