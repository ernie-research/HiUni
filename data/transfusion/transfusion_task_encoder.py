"""
多模态任务编码模块 (Task Encoder)

该模块负责将清洗后的 RawMultimodalSample 编码为 Transfusion 训练所需的
Tensor 格式，包括图像/视频特征处理、Token 序列构建、Attention Mask 生成等。

主要组件:
    - TransfusionTaskSample: 编码后的样本数据结构
    - TransfusionTaskBatch: 编码后的批次数据结构
    - TransfusionTaskEncoder: 主任务编码器，实现从样本到批次的转换流程
    - 辅助函数: 序列掩码构建、视觉特征处理、样本打包 (Packing) 等

注意:
    同一 micro batch 中只允许单一模态的数据 (纯文本 / 理解 / 生成)。
    batch() 函数会验证这一约束，如果发现混合模态会抛出 ValueError。
    确保数据采样配置正确，每个 batch 只包含同一种 task_type 的样本。
"""

import dataclasses
import bisect
import functools
import json
import logging
import os
import random
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional
import numpy as np
import torch
from PIL import Image
from megatron.energon import Batch, DefaultTaskEncoder
from megatron.energon.task_encoder.cooking import Cooker
from megatron.energon.task_encoder.base import stateless
from megatron.energon.wrappers import SkipSample
from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs

from data.transfusion.transfusion_data_cooker import (
    GenerationCotCooker,
    RawMultimodalSample,
    cook_multimodal_sample,
)

logger = logging.getLogger(__name__)

# 匹配 refbox 格式: <|object_ref_start|>文本<|object_ref_end|><|box_start|>...<|box_end|>
_REFBOX_PATTERN = re.compile(
    r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>'
    r'(?:<\|box_start\|>.*?<\|box_end\|>)+'
)


def strip_refbox(text: str) -> str:
    """去除 refbox 标记，只保留文本内容（用于 wobox 源）"""
    return _REFBOX_PATTERN.sub(r'\1', text)


# 检测文本是否已包含对话模板
_HAS_CHAT_TEMPLATE = re.compile(r'<\|im_start\|>')
_IMAGE_PLACEHOLDER = "<image>"
_TEXTUAL_DATA_MARKERS = (
    _IMAGE_PLACEHOLDER,
    "<|image_pad|>",
    "<image_pad>",
    "<|endoftext|>",
)
_THINK_TAG_PATTERN = re.compile(r"</?think>")


def _has_balanced_think_tags(text: str) -> bool:
    """Return whether a rendered assistant text has valid think blocks.

    SFT data may contain literal think tags in ``content`` in addition to the
    separate ``reasoning_content`` field.  A simple count is insufficient: a
    closing tag before an opening tag and nested blocks are malformed too.
    Multiple sequential, non-nested blocks are allowed.
    """
    depth = 0
    for match in _THINK_TAG_PATTERN.finditer(text):
        if match.group() == "<think>":
            if depth:
                return False
            depth = 1
        else:
            if not depth:
                return False
            depth = 0
    return depth == 0


def _count_token_subsequence(tokens: list, marker_tokens: list) -> int:
    """Count exact, non-overlapping marker-token occurrences in a sequence."""
    marker_len = len(marker_tokens)
    if marker_len == 0 or len(tokens) < marker_len:
        return 0

    count = 0
    start = 0
    last_start = len(tokens) - marker_len
    while start <= last_start:
        if tokens[start : start + marker_len] == marker_tokens:
            count += 1
            start += marker_len
        else:
            start += 1
    return count


def _has_chat_template(text_fragments):
    """检测 text_fragments 中是否已包含对话模板标记"""
    return any(
        isinstance(frag, str) and _HAS_CHAT_TEMPLATE.search(frag)
        for frag in text_fragments
    )


def strip_textual_image_placeholders(text: str) -> str:
    """Remove data-format markers that must not become natural-language labels."""
    for marker in _TEXTUAL_DATA_MARKERS:
        text = text.replace(marker, "")
    return text


def _move_template_images_after_user_header(text_fragments, mask):
    """Keep rendered ChatML valid while enforcing image(s) -> question order.

    Some understanding sources render the user header and question in one
    fragment, while storing the image as a separate fragment after it. Moving
    the image fragment after the header is enough; the remaining ChatML text
    and its ownership mask must retain their original order.
    """
    image_pairs = [
        (fragment, mask_value)
        for fragment, mask_value in zip(text_fragments, mask)
        if fragment == _IMAGE_PLACEHOLDER
    ]
    if not image_pairs:
        return text_fragments, mask

    remaining = [
        (fragment, mask_value)
        for fragment, mask_value in zip(text_fragments, mask)
        if fragment != _IMAGE_PLACEHOLDER
    ]
    user_header_pattern = re.compile(r"<\|im_start\|>user[ \t]*\n")

    for index, (fragment, mask_value) in enumerate(remaining):
        if not isinstance(fragment, str):
            continue
        match = user_header_pattern.search(fragment)
        if match is None:
            continue

        prefix = fragment[: match.end()]
        suffix = fragment[match.end() :]
        reordered = remaining[:index]
        reordered.append((prefix, mask_value))
        reordered.extend(image_pairs)
        if suffix:
            reordered.append((suffix, mask_value))
        reordered.extend(remaining[index + 1 :])
        return (
            [fragment for fragment, _ in reordered],
            [mask_value for _, mask_value in reordered],
        )

    return text_fragments, mask


def get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    eod_mask_loss: bool,
    reset_attention_mask: bool,
    reset_position_ids: bool,
    compute_attention_mask: bool = True,
):
    """Build masks and position ids for a left-to-right model.

    Returns:
        attention_mask: [att_mask_batch, 1, s, s] boolean mask (True means masked)
        loss_mask: [b, s] float mask (1.0 to keep loss, 0.0 to drop)
        position_ids: [b, s] positions
    """
    micro_batch_size, seq_length = data.size()

    att_mask_batch = micro_batch_size if reset_attention_mask else 1
    attention_mask = None
    if compute_attention_mask:
        attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length
        )

    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(micro_batch_size):
            eod_index = position_ids[b, data[b] == eod_token]
            if reset_position_ids:
                eod_index = eod_index.clone()
            prev_index = 0
            for j in range(eod_index.size(0)):
                i = eod_index[j]
                if reset_attention_mask and attention_mask is not None:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    if compute_attention_mask and attention_mask is not None:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def create_sample_startend_row_indices(
    input_ids, img_start_id, img_end_id, bidirectional=True
):
    """
    为单个样本构造 FlashMask 的 startend_row_indices。

    Args:
        input_ids: [L] 的 token ids
        img_start_id: <image_start> token id
        img_end_id: <image_end> token id
        bidirectional (bool):
            True  -> 修改 UTE，使图像列对图像行可见 (双向)。用于生成任务。
            False -> 不修改 UTE，保持初始化的 Causal 状态 (单向)。用于理解任务。

    Returns:
        cond_startend_row_indices: [1, L, 2] 条件 attention 的 (LTS, UTE)
        nocond_startend_row_indices: [1, L, 2] 无条件 attention 的 (LTS, UTE) (CFG用)
    """
    L = len(input_ids)
    device = input_ids.device

    # === 1. 初始化: 标准 Causal Attention ===
    # LTS (Lower Triangle Start) = L: 下三角全可见 (不Mask)
    # UTE (Upper Triangle End) = col_idx: 上三角全Mask (只保留对角线及以下)
    col_idx = torch.arange(L, device=device, dtype=torch.int32)  # [L]

    # NOTE: 仅倒数第 2 token需要看见前面所有 token，预测最后一个 token
    # cond_LTS = torch.full((L,), L, device=device, dtype=torch.int32)
    cond_LTS = torch.full((L,), max(L - 1, 0), device=device, dtype=torch.int32)
    cond_UTE = col_idx.clone()

    # nocond_LTS = torch.full((L,), L, device=device, dtype=torch.int32)
    nocond_LTS = torch.full((L,), max(L - 1, 0), device=device, dtype=torch.int32)
    nocond_UTE = col_idx.clone()

    # === 2. 找到图像区域并修改 ===
    starts = (input_ids == img_start_id).nonzero(as_tuple=True)[0]
    ends = (input_ids == img_end_id).nonzero(as_tuple=True)[0]

    min_len = min(len(starts), len(ends))

    for k in range(min_len):
        s, e = starts[k], ends[k] # 由于 timestep 存在
        # s: <boi> index, e: <eoi> index
        # Image Patch range: s+1 到 e-1 (包含)

        # --- (A) 处理图像内部双向注意力 (Cond & NoCond) ---
        # 对应 Dense: mask[s+1:e, s+1:e] = 1
        # 逻辑: 图像 Patch 列 (s+1 .. e-1) 可以被 图像 Patch 行 (s+1 .. e-1) 看到
        # FlashMask: 修改 UTE。对于 Patch 列 j，设置 UTE = s+1。
        # 效果: 行 i < s+1 (文本) 不可见; 行 i >= s+1 (图像内部+未来) 可见。

        # NOTE: 多了 timestep
        if bidirectional:
            patch_cols_mask = (col_idx > s + 1) & (col_idx < e)
            if patch_cols_mask.any():
                # 修改 UTE：让这些列在 s+1 行就能被看到 -> 形成双向块
                patch_start_idx = (s + 2).int()
                cond_UTE[patch_cols_mask] = patch_start_idx
                nocond_UTE[patch_cols_mask] = patch_start_idx

        # --- (B) 处理 NoCond 屏蔽前文 (NoCond Only) ---
        # 此逻辑与图像内部是否双向无关，只要是 NoCond 就需要屏蔽文本
        cutoff = s

        text_cols_mask = col_idx < cutoff
        if text_cols_mask.any():
            current_lts = nocond_LTS[text_cols_mask]
            new_lts_val = s.int()
            nocond_LTS[text_cols_mask] = torch.minimum(current_lts, new_lts_val)

    # === 3. 格式封装 ===
    cond_startend_row_indices = torch.stack([cond_LTS, cond_UTE], dim=-1).unsqueeze(0)
    nocond_startend_row_indices = torch.stack(
        [nocond_LTS, nocond_UTE], dim=-1
    ).unsqueeze(0)

    return cond_startend_row_indices, nocond_startend_row_indices


def _center_crop_square(img):
    """将 PIL Image center crop 成正方形。"""
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def align_single_edit_reference(images, text, mask):
    """Match single-edit images at the smaller of their native resolutions.

    Editing samples with multiple references/targets can intentionally use
    different geometries, so this normalization is limited to the unambiguous
    one-reference/one-target case. The larger image is resized to the smaller
    image's exact dimensions; the returned list is copy-on-write and the input
    list is never mutated.
    """
    if not isinstance(images, list) or not images:
        return images
    if not isinstance(text, list) or not isinstance(mask, list) or len(text) != len(mask):
        return images

    image_positions = [idx for idx, value in enumerate(text) if value == "<image>"]
    ref_positions = [idx for idx in image_positions if int(mask[idx]) == 0]
    target_positions = [idx for idx in image_positions if int(mask[idx]) == 1]
    if len(ref_positions) != 1 or len(target_positions) != 1:
        return images

    ref_image_idx = image_positions.index(ref_positions[0])
    target_image_idx = image_positions.index(target_positions[0])
    if ref_image_idx >= len(images) or target_image_idx >= len(images):
        return images

    ref_image = images[ref_image_idx]
    target_image = images[target_image_idx]
    ref_pixels = ref_image.size[0] * ref_image.size[1]
    target_pixels = target_image.size[0] * target_image.size[1]
    if ref_pixels == target_pixels and ref_image.size == target_image.size:
        return images

    aligned = list(images)
    if ref_pixels >= target_pixels:
        aligned[ref_image_idx] = ref_image.resize(
            target_image.size, Image.Resampling.LANCZOS
        )
    else:
        aligned[target_image_idx] = target_image.resize(
            ref_image.size, Image.Resampling.LANCZOS
        )
    return aligned


def _validate_max_input_pixels(images, max_input_pixels):
    """Reject native images over the configured source-pixel limit."""
    if images is None or max_input_pixels is None:
        return
    if max_input_pixels <= 0:
        raise ValueError(
            "max_input_pixels must be positive or None, "
            f"got {max_input_pixels!r}"
        )
    for index, image in enumerate(images):
        width, height = image.size
        pixels = width * height
        if pixels > max_input_pixels:
            logger.debug(
                "Skip image over max_input_pixels: index=%s size=%sx%s "
                "pixels=%s limit=%s",
                index,
                width,
                height,
                pixels,
                max_input_pixels,
            )
            raise SkipSample()


def process_vision(
    processor,
    images,
    videos,
    fps=None,
    model_version: str = "qwen-vl",
    min_pixels=None,
    max_pixels=None,
    max_input_pixels=None,
    square_crop: bool = False,
    resolution_buckets=None,
    resolution_bucket_threshold: float = 0.95,
):
    """
    Minimal vision preprocessing wrapper using the provided processor (e.g., HF AutoProcessor).
    images: [image_source, image_source, ...]
    max_input_pixels: optional native (pre-resize) image area limit.
    """
    if not 0.0 < float(resolution_bucket_threshold) <= 1.0:
        raise ValueError(
            "resolution_bucket_threshold must be in (0, 1], got "
            f"{resolution_bucket_threshold!r}"
        )

    if resolution_buckets is not None:
        if isinstance(resolution_buckets, (int, float)):
            resolution_buckets = [resolution_buckets]
        try:
            resolution_buckets = sorted({int(size) for size in resolution_buckets})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "resolution_buckets must be a sequence of positive side lengths"
            ) from exc
        if not resolution_buckets or any(size <= 0 for size in resolution_buckets):
            raise ValueError(
                "resolution_buckets must contain at least one positive side length"
            )

    def _select_resolution_bucket(image):
        """Map native image area to the first larger resolution tier.

        For buckets ``[1024, 2048]`` and threshold ``0.95``, areas up to
        ``2048**2 * 0.95`` use the 1024 tier and larger areas use 2048.
        Images below the first tier also use the first tier.
        """
        area = image.size[0] * image.size[1]
        for current, next_size in zip(resolution_buckets, resolution_buckets[1:]):
            if area <= next_size * next_size * resolution_bucket_threshold:
                return current
        return resolution_buckets[-1]

    if images is not None:
        _validate_max_input_pixels(images, max_input_pixels)
        if square_crop:
            images = [_center_crop_square(img) for img in images]
        for img in images:
            width, height = img.size
            short_side = min(width, height)
            # 与 Qwen2VLImageProcessor.smart_resize 的约束对齐。提前转成
            # SkipSample，避免单个异常图片终止整个 dataloader worker。
            if short_side <= 4:
                raise SkipSample()
            aspect_ratio = max(width, height) / short_side
            if aspect_ratio > 200:
                logger.debug(
                    "Skip image with unsupported aspect ratio: size=%sx%s ratio=%.2f",
                    width,
                    height,
                    aspect_ratio,
                )
                raise SkipSample()
        if resolution_buckets is None:
            kwargs = {}
            if min_pixels is not None:
                kwargs["min_pixels"] = min_pixels
            if max_pixels is not None:
                kwargs["max_pixels"] = max_pixels
            image_inputs = processor(images=images, videos=None, return_tensors="pt", **kwargs)
        else:
            # A processor call has one shared min/max-pixel range. Process each
            # image separately so samples in one packed sequence can use
            # different resolution tiers without changing the token ordering.
            per_image_inputs = []
            for image in images:
                target_side = _select_resolution_bucket(image)
                target_pixels = target_side * target_side
                per_image_inputs.append(
                    processor(
                        images=[image],
                        videos=None,
                        return_tensors="pt",
                        min_pixels=target_pixels,
                        max_pixels=target_pixels,
                    )
                )

            image_inputs = {}
            for key in ("pixel_values", "image_grid_thw"):
                values = [item.get(key) for item in per_image_inputs]
                values = [value for value in values if value is not None]
                if values:
                    image_inputs[key] = torch.cat(values, dim=0)

        image_grid_thw = image_inputs.get("image_grid_thw", None)
    else:
        image_inputs = {}
        image_grid_thw = None

    if videos is not None:
        videos_inputs = processor(images=None, videos=videos, return_tensors="pt")
        video_grid_thw = videos_inputs.get("video_grid_thw", None)
    else:
        videos_inputs = {}
        video_grid_thw = None
    return {
        "image_inputs": image_inputs,
        "image_grid_thw": image_grid_thw,
        "video_inputs": videos_inputs,
        "video_grid_thw": video_grid_thw,
    }


def _resolve_hf_mm_token_ids(hf_tokenizer):
    """Resolve HF tokenizer ids for Qwen3-VL special tokens."""

    def _get(token_str: str, default_id: int) -> int:
        """Resolve one tokenizer id, falling back when the token is unavailable."""
        token_attr = getattr(hf_tokenizer, f"{token_str.strip('<>|')}_token_id", None)
        if token_attr is not None:
            return int(token_attr)
        try:
            return int(hf_tokenizer.convert_tokens_to_ids(token_str))
        except Exception:
            return default_id

    # Qwen3-VL vision tokens
    image_id = _get("<|image_pad|>", 151655)
    video_id = _get("<|video_pad|>", 151656)
    image_start_id = _get("<|vision_start|>", 151652)
    image_end_id = _get("<|vision_end|>", 151653)
    video_start_id = _get("<|vision_start|>", 151652)
    video_end_id = _get("<|vision_end|>", 151653)
    time_id = _get("<time_token>", 151661)

    # Qwen3-VL chat template tokens: 
    # <|im_start|>user\n{text}<|im_end|>\n
    # <|im_start|>assistant\n
    im_start_id = _get("<|im_start|>", 151644)
    im_end_id = _get("<|im_end|>", 151645)
    nl_ids = hf_tokenizer.encode("\n", add_special_tokens=False)
    user_ids = hf_tokenizer.encode("user", add_special_tokens=False)
    assistant_ids = hf_tokenizer.encode("assistant", add_special_tokens=False)

    # pad token id
    pad_id = hf_tokenizer.pad_token_id

    return (image_id, video_id, image_start_id, image_end_id, video_start_id, video_end_id, time_id,
            im_start_id, im_end_id, nl_ids, user_ids, assistant_ids, pad_id)


# V5 route tokens are part of the checkpoint/tokenizer contract.
V5_TEXT_GATE_TOKEN_ID = 151669
V5_IMAGE_GATE_TOKEN_ID = 151670
V5_RECAPTION_END_TOKEN_ID = 151671


def search_for_fit(numbers: List[int], capacity: int) -> int:
    """Finds index of largest number that fits into knapsack with given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(item_sizes: List[int], samples: List, max_capacity: int) -> List:
    """Greedy algorithm with binary search for the knapsack problem.

    Pack as many samples as possible given a maximum capacity and capacities of individual samples.
    Used if sequence packing is enabled.

    Args:
        item_sizes: List of sample sizes
        samples: List of samples
        max_capacity: Maximum sequence length for packing

    Returns:
        List of packed sample groups
    """
    assert len(item_sizes) == len(samples), "sample lengths and samples must have same length."

    knapsacks = []

    if len(item_sizes) == 0:
        return knapsacks

    # Sort sample lengths and samples together.
    sorted_item_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples), key=lambda x: x[0]))
    sorted_item_sizes = list(sorted_item_sizes)
    sorted_samples = list(sorted_samples)

    # Check if all samples fit in knapsack capacity.
    if sorted_item_sizes[-1] > max_capacity:
        raise ValueError(
            f"knapsack: A sample is larger {sorted_item_sizes[-1]} than max_sequence_length {max_capacity}."
        )

    while sorted_item_sizes:
        current_knapsack = []
        remaining_capacity = max_capacity

        while True:
            idx = search_for_fit(sorted_item_sizes, remaining_capacity)
            if idx == -1:
                break  # Can't fit more samples.

            remaining_capacity -= sorted_item_sizes[idx]

            sorted_item_sizes.pop(idx)
            sample = sorted_samples.pop(idx)
            current_knapsack.append(sample)

        knapsacks.append(current_knapsack)

    return knapsacks


class TaskType(IntEnum):
    """Task Type"""

    UNDERSTANDING = 0
    GENERATION = 1
    EDITING = 2
    TEXT = 3
    MESSAGE = 4
    SFT_TEXT = 5
    SFT_IMAGE_UND = 6
    GENERATION_COT = 7
    GENERATION_COT_TEXT = 8

    @classmethod
    def from_str(cls, s: str) -> "TaskType":
        """map str to enum"""
        mapping = {
            "understanding": cls.UNDERSTANDING,
            "generation": cls.GENERATION,
            "editing": cls.EDITING,
            "text": cls.TEXT,
            "message": cls.MESSAGE,
            "sft_text": cls.SFT_TEXT,
            "sft_image_und": cls.SFT_IMAGE_UND,
            "generation_cot": cls.GENERATION_COT,
            "generation_cot_text": cls.GENERATION_COT_TEXT,
        }
        if s not in mapping:
            raise ValueError(f"Unknown task type: {s}")
        return mapping[s]


@dataclass
class TransfusionTaskSample:
    """Encoded Sample Format For Transfusion"""

    __key__: str
    __subflavors__: Dict

    # 用于 packing 时分类隔离
    task_type: str

    imgs: List[torch.Tensor]
    videos: List[torch.Tensor]

    image_thw_grids: List[torch.Tensor]
    video_thw_grids: List[torch.Tensor]
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    text: torch.Tensor
    target: torch.Tensor
    total_len: int

    # Transfusion 图像掩码涉及三种坐标空间:
    #   input-token: spatial merge 后的 image token, 与 input_ids/image_input_mask 对齐
    #   raw ViT:     merge 前的像素 patch, 与 pixel_values/image_grid_thw 对齐
    #   ViT output:  vision_model 输出 token, ref/understanding 路径已 spatial merge
    total_img_und_mask: torch.Tensor  # input-token 空间; True = ref/understanding/feedback 图像 token
    image_generate_mask: torch.Tensor  # ViT/raw 特征空间; True = 生成目标 token
    image_understand_mask: torch.Tensor  # ViT output 空间; True = 作为 clean 条件的 ref/understanding token
    image_gen_seq_lens: List[int]  # 每张生成图的 raw token 数, 用于 diffusion scheduler/noise
    image_gen_grid_thw: List[torch.Tensor] = None  # 每张生成图的 raw ViT grid, 用于 pixel-space noise

    # 3-channel M-RoPE position_ids: (3, L), 图像任务由 task encoder 构造
    position_ids: torch.Tensor = None

    # FlashMask startend_row_indices: (1, L, 2), 单个样本的 attention mask
    # [0, :, 0] = LTS (Lower Triangle Start), [0, :, 1] = UTE (Upper Triangle End)
    cond_startend_row_indices: torch.Tensor = None  # 条件 attention
    nocond_startend_row_indices: torch.Tensor = None  # 无条件 attention (CFG用)

    # Packing 时各子样本的 token 长度 (用于 sample-level loss averaging)
    packed_sample_lens: List[int] = None  # [num_packed_samples], e.g. [1200, 800]


@dataclass
class TransfusionTaskBatch(Batch):  # 👈 改名了
    """Encoded Batch Format For Transfusion"""

    __keys__: List[str]
    __subflavors__: List[Dict]
    pixel_values: torch.Tensor
    pixel_values_videos: torch.Tensor
    image_grid_thw: torch.Tensor
    video_grid_thw: torch.Tensor
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor

    # batch 后仍保持与 TransfusionTaskSample 相同的坐标空间定义。
    total_img_und_mask: torch.Tensor = None  # input-token 空间; True = ref/understanding/feedback 图像 token
    image_generate_mask: torch.Tensor = None  # ViT/raw 特征空间; True = 生成目标 token
    image_understand_mask: torch.Tensor = None  # ViT output 空间; True = ref/understanding token
    image_gen_seq_lens: torch.Tensor = None  # [num_gen_images], 每张生成图的 raw token 数
    image_gen_grid_thw: torch.Tensor = None  # [num_gen_images, 3], pixel-space noise 使用的 raw grid
    num_gen_images_per_sample: torch.Tensor = None  # [B] 每个 sample 贡献的生成图数量
    startend_row_indices: torch.Tensor = None
    task_type: torch.Tensor = None
    # Packing 时各子样本的 token 长度 (用于 sample-level loss averaging)
    # [B, max_num_samples], padding 用 0 填充
    packed_sample_lens: torch.Tensor = None

    max_lengths: List[int] = None

    # 统计信息 (用于 TensorBoard 记录)
    num_images: int = 0  # batch 中的图片总数
    num_gen_images: int = 0  # 普通生成任务的图片数量
    num_gen_cot_images: int = 0  # 带 COT 生成任务的图片数量
    num_und_images: int = 0  # 理解任务的图片数量
    num_edit_images: int = 0  # 编辑任务的图片数量

    # 不同模态的训练行数统计 (用于 TensorBoard 记录)
    # 注意: 有 packing 时一行可能包含多个样本，此处统计的是 batch 行数
    num_text_rows: int = 0  # 纯文本行数
    num_und_rows: int = 0  # 理解任务行数
    num_gen_rows: int = 0  # 普通生成任务行数
    num_gen_cot_rows: int = 0  # 带 COT 生成任务行数
    num_gen_cot_text_rows: int = 0  # 纯文本 generation COT 行数
    num_edit_rows: int = 0  # 编辑任务行数


class TransfusionTaskEncoder(
    DefaultTaskEncoder[RawMultimodalSample, TransfusionTaskSample, TransfusionTaskBatch, dict]
):
    """A simple task encoder for Transfusion."""

    cookers = [Cooker(cook_multimodal_sample)]

    def __init__(
        self,
        tokenizer,
        image_processor,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        patch_size: int = 14,
        max_padding_length: int = 8192,
        min_pixels: int = 200704,
        max_pixels: int = 1003520,
        gen_min_pixels: int = None,
        gen_max_pixels: int = None,
        gen_max_input_pixels: int = None,
        edit_min_pixels: int = None,
        edit_max_pixels: int = None,
        edit_max_input_pixels: int = None,
        und_min_pixels: int = None,
        und_max_pixels: int = None,
        gen_resolution_buckets=None,
        edit_resolution_buckets=None,
        resolution_bucket_threshold: float = 0.95,
        packing_seq_length: int = 8192,
        cfg_drop_rate: float = 0.1,
        cfg_keep_chat_template: bool = False,
        square_crop: bool = False,
        enable_und_bidirection: bool = False,
        prompt_mapping_file: str = "data/image_und_midtrain_v0214_prompt_v4.list",
        editing_encode_type: str = "vit",
        sft_chat_template_path: str = None,  # SFT 纯文本 (sft_text) jinja 模板路径，None=用默认 qwen3 模板
        vl_chat_template_path: str = None,  # SFT 多模态理解 (sft_image_und) jinja 模板路径，None=用默认 Qwen3-VL 模板
        enable_think: bool = False,  # 是否将 reasoning_content 注入 <think> 块；False 时丢弃思考链
        generation_empty_think: bool = False,
        generation_cot_prompt_sampling_weights: dict = None,
        generation_cot_long_template_prob: float = 0.0,
        generation_cot_long_template_version: str = "cpt_v5",
        batch_schedule: dict = None,
        batch_schedule_strict: bool = True,
        predict_gate_mode: bool = None,  # None=自动探测, True=强制启用, False=关闭
    ):
        """Initialize tokenizer, visual processing, routing, and packing settings."""
        super().__init__()
        self.hf_tokenizer = tokenizer

        self.generation_cot_prompt_sampling_weights = (
            GenerationCotCooker.normalize_prompt_sampling_weights(
                generation_cot_prompt_sampling_weights
            )
        )
        if not 0.0 <= float(generation_cot_long_template_prob) <= 1.0:
            raise ValueError("generation_cot_long_template_prob must be in [0, 1]")
        if generation_cot_long_template_version != "cpt_v5":
            raise ValueError(
                "unsupported generation_cot_long_template_version: "
                f"{generation_cot_long_template_version!r}"
            )
        self.generation_cot_long_template_prob = float(generation_cot_long_template_prob)
        self.generation_cot_long_template_version = generation_cot_long_template_version
        self.generation_empty_think = bool(generation_empty_think)
        # The V5 route is deliberately fixed; there is no protocol selector.
        self.image_gate_token_id = V5_IMAGE_GATE_TOKEN_ID
        self.recaption_end_token_id = V5_RECAPTION_END_TOKEN_ID
        self.text_gate_token_id = V5_TEXT_GATE_TOKEN_ID
        self.think_start_ids = self.hf_tokenizer.encode("<think>", add_special_tokens=False)
        self.think_end_ids = self.hf_tokenizer.encode("</think>", add_special_tokens=False)
        self.cookers = [
            Cooker(
                functools.partial(
                    cook_multimodal_sample,
                    generation_cot_prompt_sampling_weights=self.generation_cot_prompt_sampling_weights,
                    generation_cot_long_template_prob=self.generation_cot_long_template_prob,
                    generation_cot_long_template_version=self.generation_cot_long_template_version,
                )
            )
        ]

        self.image_processor = image_processor
        self.seq_length = max_padding_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.gen_max_input_pixels = gen_max_input_pixels
        self.edit_max_input_pixels = (
            edit_max_input_pixels
            if edit_max_input_pixels is not None
            else gen_max_input_pixels
        )
        for name, value in (
            ("gen_max_input_pixels", self.gen_max_input_pixels),
            ("edit_max_input_pixels", self.edit_max_input_pixels),
        ):
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} must be positive or None, got {value!r}"
                )
        self.square_crop = square_crop
        # 按任务类型区分的像素范围，未指定时回退到通用值
        self.pixel_config = {
            "generation": {
                "min_pixels": gen_min_pixels if gen_min_pixels is not None else min_pixels,
                "max_pixels": gen_max_pixels if gen_max_pixels is not None else max_pixels,
                "max_input_pixels": gen_max_input_pixels,
            },
            "generation_cot": {
                "min_pixels": gen_min_pixels if gen_min_pixels is not None else min_pixels,
                "max_pixels": gen_max_pixels if gen_max_pixels is not None else max_pixels,
                "max_input_pixels": gen_max_input_pixels,
            },
            "generation_cot_text": {
                "min_pixels": 0,
                "max_pixels": 0,
                "max_input_pixels": None,
            },
            "understanding": {
                "min_pixels": und_min_pixels if und_min_pixels is not None else min_pixels,
                "max_pixels": und_max_pixels if und_max_pixels is not None else max_pixels,
                "max_input_pixels": None,
            },
            "sft_image_und": {
                "min_pixels": und_min_pixels if und_min_pixels is not None else min_pixels,
                "max_pixels": und_max_pixels if und_max_pixels is not None else max_pixels,
                "max_input_pixels": None,
            },
            "editing": {
                "min_pixels": (
                    edit_min_pixels
                    if edit_min_pixels is not None
                    else (gen_min_pixels if gen_min_pixels is not None else min_pixels)
                ),
                "max_pixels": (
                    edit_max_pixels
                    if edit_max_pixels is not None
                    else (gen_max_pixels if gen_max_pixels is not None else max_pixels)
                ),
                "max_input_pixels": self.edit_max_input_pixels,
            },
            "text": {
                "min_pixels": 0,
                "max_pixels": 0,
                "max_input_pixels": None,
            },
        }
        # These buckets apply to processor-backed generation/editing images.
        self.resolution_buckets = {
            "generation": gen_resolution_buckets,
            "generation_cot": gen_resolution_buckets,
            "editing": edit_resolution_buckets,
        }
        self.resolution_bucket_threshold = float(resolution_bucket_threshold)
        if not 0.0 < self.resolution_bucket_threshold <= 1.0:
            raise ValueError(
                "resolution_bucket_threshold must be in (0, 1], got "
                f"{resolution_bucket_threshold!r}"
            )

        self.temporal_patch_size = temporal_patch_size
        self.merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.enable_und_bidirection = enable_und_bidirection
        self.seq_len = max_padding_length

        # Dense Pixel-DiT editing always uses the ViT/raw-pixel route.
        _valid_editing_types = {"vit"}
        if editing_encode_type not in _valid_editing_types:
            raise ValueError(
                f"editing_encode_type={editing_encode_type!r} 不合法，"
                f"可选值: {_valid_editing_types}"
            )
        self.editing_encode_type = editing_encode_type
        (
            self.image_token_id,
            self.video_token_id,
            self.image_start_id,
            self.image_end_id,
            self.video_start_id,
            self.video_end_id,
            self.time_token_id,
            self.im_start_id,
            self.im_end_id,
            self.nl_ids,
            self.user_ids,
            self.assistant_ids,
            self.pad_id,
        ) = _resolve_hf_mm_token_ids(self.hf_tokenizer)
        route_token_support = True
        for token, token_id in (
            ("<|text_gate|>", V5_TEXT_GATE_TOKEN_ID),
            ("<|image_gate|>", V5_IMAGE_GATE_TOKEN_ID),
            ("<|recaption_end|>", V5_RECAPTION_END_TOKEN_ID),
        ):
            try:
                encoded = self.hf_tokenizer.encode(token, add_special_tokens=False)
                converted = self.hf_tokenizer.convert_tokens_to_ids(token)
            except (AttributeError, KeyError, TypeError, ValueError):
                route_token_support = False
                break
            if encoded != [token_id] or converted != token_id:
                route_token_support = False
                break
        if predict_gate_mode is True and not route_token_support:
            raise ValueError(
                "predict_gate_mode=True requires V5 route tokens to each encode "
                "as one fixed ID: <|text_gate|>=151669, "
                "<|image_gate|>=151670, <|recaption_end|>=151671"
            )
        self.predict_gate_mode = predict_gate_mode
        # None preserves legacy auto-detection; an explicit bool is authoritative.
        self.use_v5_route_tokens = (
            route_token_support if predict_gate_mode is None else bool(predict_gate_mode)
        )

        # Packing settings
        self.packing_seq_length = packing_seq_length
        self.is_packing_enabled = packing_seq_length is not None and packing_seq_length > 0

        # CFG (Classifier-Free Guidance) 训练: 以一定概率让生成图片看不到前面的文本
        self.cfg_drop_rate = cfg_drop_rate
        # False: T2I/full-edit uncond 仅保留生成图块；True: 保留 chat template。
        # T2I 的兼容格式还会保留通用生成指令；edit full 的 user content 为空。
        self.cfg_keep_chat_template = cfg_keep_chat_template

        # CFG mode 采样统计 (editing 任务): 用于验证三态采样比例约为 1:1:(1-2*p)
        # 通过环境变量 CFG_MODE_LOG_INTERVAL 控制打印频率, 0 表示关闭
        self._cfg_mode_counts = {"none": 0, "text": 0, "full": 0}
        self._cfg_mode_log_interval = int(os.environ.get("CFG_MODE_LOG_INTERVAL", "0"))

        # 是否开启 think 模式：True 时 reasoning_content → <think>...</think>，False 时丢弃
        self.enable_think = enable_think

        # Debug: 每种 task_type 打印前 N 条样本的 text / label 对齐情况
        self._debug_print_limit = int(os.environ.get("SFT_DEBUG_PRINT", "2"))
        self._debug_print_count: Dict[str, int] = defaultdict(int)

        # 模态缓冲区: 按 task_type 分组存储样本，用于保证同一 batch 只有单一模态
        self._modal_buffers: Dict[TaskType, List[TransfusionTaskSample]] = defaultdict(list)

        # Round-robin batch 调度器
        self._batch_schedule = batch_schedule
        self._batch_schedule_strict = batch_schedule_strict
        self._schedule_weights: Dict[TaskType, float] = {}
        self._schedule_deficit: Dict[TaskType, float] = {}
        self._batch_step_counter = 0
        self._batch_modality_counts: Dict[TaskType, int] = defaultdict(int)
        _MAX_DEFICIT = 10.0
        self._max_deficit = _MAX_DEFICIT

        if batch_schedule:
            _name_to_type = {
                "text": TaskType.TEXT,
                "understanding": TaskType.UNDERSTANDING,
                "generation": TaskType.GENERATION,
                "generation_cot": TaskType.GENERATION_COT,
                "generation_cot_text": TaskType.GENERATION_COT_TEXT,
                "editing": TaskType.EDITING,
            }
            total_weight = sum(batch_schedule.values())
            for name, weight in batch_schedule.items():
                if name not in _name_to_type:
                    logger.warning(
                        f"batch_schedule 中未知模态: {name}, 跳过"
                    )
                    continue
                task_type = _name_to_type[name]
                self._schedule_weights[task_type] = weight / total_weight
                self._schedule_deficit[task_type] = 0.0

        # Prompt 映射: source_id -> [prompt_file_path, ...]
        self.prompt_mapping = {}
        self.prompt_cache = {}
        if prompt_mapping_file:
            self._load_prompt_mapping(prompt_mapping_file)

        # ==========================================
        # SFT 纯文本 jinja 模板（用于 task_type="sft_text"）
        # ==========================================
        # 默认复用 qwen_text recipe 提供的 Qwen3 官方 chat template
        if sft_chat_template_path is None:
            sft_chat_template_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "recipe", "qwen_text", "tokenizer", "chat_template.jinja",
            )
        self.sft_chat_template = None
        try:
            from jinja2 import Environment

            def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
                """Serialize a Jinja value using the same options as json.dumps."""
                return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                                  separators=separators, sort_keys=sort_keys)

            with open(sft_chat_template_path, "r", encoding="utf-8") as f:
                template_str = f.read()
            env = Environment()
            env.filters["tojson"] = _tojson
            self.sft_chat_template = env.from_string(template_str)
            logger.info(f"[SFT] chat template loaded from {sft_chat_template_path}")
        except FileNotFoundError:
            logger.warning(
                f"[SFT] chat template not found at {sft_chat_template_path}; "
                f"task_type=sft_text will be unavailable."
            )
        except Exception as e:
            logger.warning(f"[SFT] failed to load chat template: {e}")

        # \n token id (单 token)，用于 SFT loss mask 中跳过 role header
        nl_encoded = self.hf_tokenizer.encode("\n", add_special_tokens=False)
        self.nl_single_id = nl_encoded[0] if nl_encoded else None

        # ==========================================
        # Qwen3-VL chat 模板（用于 task_type="sft_image_und"）
        # ==========================================
        # 默认从 hf_tokenizer 自带的 chat_template 加载（HF Qwen3-VL Processor
        # 会把 chat_template.json 写入 tokenizer），失败回退到查找本地 hf_path/chat_template.json。
        self.vl_chat_template = None
        try:
            from jinja2 import Environment

            def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
                """Serialize a Jinja value using the same options as json.dumps."""
                return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                                  separators=separators, sort_keys=sort_keys)

            template_str = None
            if vl_chat_template_path is not None:
                with open(vl_chat_template_path, "r", encoding="utf-8") as f:
                    template_str = f.read()
            else:
                # 1) 优先使用 tokenizer 自带的 chat_template
                tk_tpl = getattr(self.hf_tokenizer, "chat_template", None)
                if isinstance(tk_tpl, str) and "<|vision_start|>" in tk_tpl:
                    template_str = tk_tpl
                else:
                    # 2) 回退：从 tokenizer name_or_path 同目录找 chat_template.json
                    name_or_path = getattr(self.hf_tokenizer, "name_or_path", None)
                    if name_or_path:
                        candidate = os.path.join(name_or_path, "chat_template.json")
                        if os.path.exists(candidate):
                            with open(candidate, "r", encoding="utf-8") as f:
                                template_str = json.load(f).get("chat_template")

            if template_str:
                env = Environment()
                env.filters["tojson"] = _tojson
                self.vl_chat_template = env.from_string(template_str)
                logger.info("[VL-SFT] Qwen3-VL chat template loaded")
            else:
                logger.warning(
                    "[VL-SFT] Qwen3-VL chat template not found; "
                    "task_type=sft_image_und will be unavailable."
                )
        except Exception as e:
            logger.warning(f"[VL-SFT] failed to load Qwen3-VL chat template: {e}")

    def _load_prompt_mapping(self, mapping_file: str):
        """
        加载 prompt 映射文件并预加载所有 prompt 文件内容
        格式: source_id\tprompt_file_path (每行一条，同一 source_id 可对应多个 prompt 文件)
        """
        import os
        if not os.path.exists(mapping_file):
            logger.warning(f"Prompt mapping file not found: {mapping_file}")
            return

        # 收集所有唯一的 prompt 文件路径
        all_prompt_files = set()

        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                source_id, prompt_path = parts
                if source_id not in self.prompt_mapping:
                    self.prompt_mapping[source_id] = []
                self.prompt_mapping[source_id].append(prompt_path)
                all_prompt_files.add(prompt_path)

        # 预加载所有 prompt 文件
        total_prompts = 0
        for prompt_file in all_prompt_files:
            if os.path.exists(prompt_file):
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompts = [line.strip() for line in f if line.strip()]
                self.prompt_cache[prompt_file] = prompts
                total_prompts += len(prompts)
            else:
                logger.warning(f"Prompt file not found: {prompt_file}")
                self.prompt_cache[prompt_file] = []

        logger.warning(f"Loaded prompt mapping: {len(self.prompt_mapping)} source_ids, "
                    f"{len(all_prompt_files)} prompt files, {total_prompts} total prompts")

    def _get_random_prompt(self, source_id: str) -> str:
        """根据 source_id 随机获取一个 prompt"""
        if not self.prompt_mapping or not source_id:
            return None

        prompt_key = source_id
        if prompt_key not in self.prompt_mapping:
            # 兼容旧数据只保存 release_no（例如 8010010326），而映射表使用
            # release_no_版本（例如 8010010326_88）的格式。
            is_wobox = source_id.endswith("_wobox")
            source_base = source_id.removesuffix("_wobox")
            prompt_key = next(
                (
                    key
                    for key in self.prompt_mapping
                    if key.endswith("_wobox") == is_wobox and source_base in key
                ),
                None,
            )
            if prompt_key is None:
                return None

        # 随机选择一个 prompt 文件
        prompt_files = [
            path
            for path in self.prompt_mapping[prompt_key]
            if self.prompt_cache.get(path)
        ]
        if not prompt_files:
            return None
        prompt_file = random.choice(prompt_files)

        # 从缓存中获取 prompts
        prompts = self.prompt_cache.get(prompt_file, [])
        if not prompts:
            return None

        return random.choice(prompts)

    def _build_position_ids(self, input_ids_list, image_blocks):
        """
        为任意任务类型构造 3-channel M-RoPE position_ids (3, L).
        对齐 get_rope_index 的语义，但不依赖 vision_start_token_id，纯粹通过
        检测 IMG_PAD (image_token_id) 连续块来定位图像区域。

        基本规则:
          - 非图像 token (text, BOI, TIME, EOI, EOS 等): 3 通道 = 相同的递增位置
          - 图像 token (IMG_PAD 连续块): 每个子网格用 3D ramp
              通道 0 (temporal): offset + t_index
              通道 1 (height):   offset + h_index
              通道 2 (width):    offset + w_index
            offset = 该子网格前的 text position + 1; 子网格在块内顺序排布,
            下一段 offset = 上一段 offset + max(T,H,W) (与 get_rope_index 一致)。
          - 图像块之后的 text 从最后一个 offset + max(T,H,W) 继续递增。

        image_blocks 支持两种形式 (会被归一化为统一的"块描述符"):
          1) 兼容旧版: tensor[N,3] / list[[t,h,w], ...], 每个 IMG_PAD 块一个普通 2D 网格;
             某元素也可是子网格列表 [[t,h,w],[t,h,w]] 表示复合视觉块。
          2) 新版(编辑): list[dict], 每个 IMG_PAD 块一个 dict:
               {
                 "sub_grids": [[t,h,w], ...],  # 一个或多个子网格, 块内顺序排布
                 "role":      "normal"|"gen"|"feedback",
                 "gen_id":    int,             # gen: 自身 id; feedback: 镜像的 gen id
               }
        feedback 块语义 (透明镜像): 各子网格 offset = 对应 gen 块的起始 offset
        (参考子网格必须与 gen 完全一致; ViT 子网格用同一 offset、独立 ramp、允许重叠)。
        处理完后【全局 pos 不前进】, 使 feedback 在位置空间里"零占用",
        下游 text 的位置与"没有 feedback"时一致。

        Args:
            input_ids_list: token id 列表 (python list of int)
            image_blocks:   见上 (tensor / list[list] / list[dict]) 或 None
        Returns:
            position_ids: (3, L) long tensor
        """
        L = len(input_ids_list)
        position_ids = torch.zeros(3, L, dtype=torch.long)

        # 找出所有 IMG_PAD 连续块的 [start, end) 区间
        img_token_id = self.image_token_id
        img_blocks = []
        in_block = False
        for i, tok in enumerate(input_ids_list):
            if tok == img_token_id:
                if not in_block:
                    block_start = i
                    in_block = True
            else:
                if in_block:
                    img_blocks.append((block_start, i))
                    in_block = False
        if in_block:
            img_blocks.append((block_start, L))

        # 无图像: 全部线性位置
        if not img_blocks:
            pos = torch.arange(L)
            position_ids[0] = pos
            position_ids[1] = pos
            position_ids[2] = pos
            return position_ids

        def _to_int(x):
            """Convert a scalar tensor or numeric value to a Python integer."""
            return x.item() if isinstance(x, torch.Tensor) else int(x)

        # ---- 归一化 image_blocks 为统一的块描述符列表 ----
        # 每项: {"sub_grids": [[t,h,w],...], "role": str, "gen_id": int|None}
        num_images = len(img_blocks)

        def _normalize_entry(entry):
            """Normalize one image block descriptor to the internal dictionary form."""
            # dict: 新版描述符, 直接用
            if isinstance(entry, dict):
                return {
                    "sub_grids": [list(sg) for sg in entry["sub_grids"]],
                    "role": entry.get("role", "normal"),
                    "gen_id": entry.get("gen_id", None),
                }
            # 复合块: entry 的首元素本身是 [t,h,w] (list/tuple 或 ndim>=1 的 tensor)
            _first = entry[0]
            _is_composite = isinstance(_first, (list, tuple)) or (
                isinstance(_first, torch.Tensor) and _first.ndim >= 1
            )
            sub_grids = (
                [[_to_int(c) for c in sg] for sg in entry]
                if _is_composite
                else [[_to_int(entry[0]), _to_int(entry[1]), _to_int(entry[2])]]
            )
            return {"sub_grids": sub_grids, "role": "normal", "gen_id": None}

        assert image_blocks is not None and len(image_blocks) == num_images, (
            f"IMG_PAD 连续块数 {num_images} != image_blocks 数 "
            f"{len(image_blocks) if image_blocks is not None else 0}"
        )
        norm_blocks = [_normalize_entry(image_blocks[i]) for i in range(num_images)]

        gen_offset_map = {}  # gen_id -> 该 gen 子网格的起始 offset
        pos = 0              # 当前 text position counter
        prev_end = 0

        def _lay_subgrid(sub_start, sg, offset):
            """Write temporal and spatial position ids for one image sub-grid."""
            T, H, W = int(sg[0]), int(sg[1]), int(sg[2])
            K = T * H * W
            sub_end = sub_start + K
            t_index = torch.arange(T).view(-1, 1).expand(-1, H * W).flatten()
            h_index = torch.arange(H).view(1, -1, 1).expand(T, -1, W).flatten()
            w_index = torch.arange(W).view(1, 1, -1).expand(T, H, -1).flatten()
            position_ids[0, sub_start:sub_end] = offset + t_index
            position_ids[1, sub_start:sub_end] = offset + h_index
            position_ids[2, sub_start:sub_end] = offset + w_index
            return sub_end, max(T, H, W), K

        for img_idx, (blk_start, blk_end) in enumerate(img_blocks):
            # 1. 图像块之前的 text tokens (包括 BOI, TIME 等)
            for i in range(prev_end, blk_start):
                position_ids[:, i] = pos
                pos += 1

            spec = norm_blocks[img_idx]
            sub_grids = spec["sub_grids"]
            role = spec["role"]
            gen_id = spec["gen_id"]

            # 2. 图像 tokens
            is_feedback = role == "feedback" and gen_id in gen_offset_map
            sub_start = blk_start
            total_k = 0

            if is_feedback:
                # 透明镜像: 全部子网格 offset = 对应 gen 的起始 offset; pos 不前进
                base = gen_offset_map[gen_id]
                for sg in sub_grids:
                    sub_start, _span, K = _lay_subgrid(sub_start, sg, base)
                    total_k += K
                # 结束后 pos 保持不变 (零占用)
            else:
                first_offset = pos
                for sg in sub_grids:
                    offset = pos
                    sub_start, span, K = _lay_subgrid(sub_start, sg, offset)
                    pos = offset + span
                    total_k += K
                if role == "gen" and gen_id is not None:
                    gen_offset_map[gen_id] = first_offset

            assert blk_end - blk_start == total_k, (
                f"Image block size {blk_end - blk_start} != sum(T*H*W) = {total_k}"
            )
            prev_end = blk_end

        # 3. 最后一个图像块之后的 text tokens (EOI, EOS 等)
        for i in range(prev_end, L):
            position_ids[:, i] = pos
            pos += 1

        return position_ids

    def _normalize_task_type(self, task_type: TaskType) -> TaskType:
        """将 MESSAGE / SFT_TEXT 映射到 TEXT，将 SFT_IMAGE_UND 映射到 UNDERSTANDING，
        使它们各自共用一个 buffer"""
        if task_type == TaskType.MESSAGE or task_type == TaskType.SFT_TEXT:
            return TaskType.TEXT
        if task_type == TaskType.SFT_IMAGE_UND:
            return TaskType.UNDERSTANDING
        return task_type

    def _select_round_robin(self, batch_size: int) -> "Optional[TaskType]":
        """Deficit-based 确定性调度。

        strict 模式：目标模态 buffer 空时 SkipSample 等待，严格保比例。
        relaxed 模式：目标模态 buffer 空时退而选下一个有数据的模态，
                     饥饿模态 deficit 继续累积，来数据后优先补偿。
        """
        for task_type, weight in self._schedule_weights.items():
            self._schedule_deficit[task_type] += weight

        candidates = sorted(
            self._schedule_deficit.keys(),
            key=lambda t: (-self._schedule_deficit[t], t.value),
        )

        for task_type in candidates:
            if self._schedule_deficit[task_type] > self._max_deficit:
                logger.warning(
                    f"[BatchScheduler] 模态 {task_type.name} deficit "
                    f"超过阈值 {self._max_deficit}，数据供给不足"
                )
                if self._batch_schedule_strict:
                    continue
                else:
                    self._schedule_deficit[task_type] = self._max_deficit
                    continue
            if len(self._modal_buffers[task_type]) >= batch_size:
                self._schedule_deficit[task_type] -= 1.0
                return task_type
            if self._batch_schedule_strict:
                # strict: 目标模态没数据，直接等待
                for tt, w in self._schedule_weights.items():
                    self._schedule_deficit[tt] -= w
                return None

        # 所有模态都没数据
        for task_type, weight in self._schedule_weights.items():
            self._schedule_deficit[task_type] -= weight
        return None

    def _select_max_buffer(self, batch_size: int) -> "Optional[TaskType]":
        """原有逻辑：选择 buffer 中样本最多且 >= batch_size 的模态。"""
        selected_task_type = None
        max_count = 0
        for task_type, buffer in self._modal_buffers.items():
            if len(buffer) >= batch_size and len(buffer) > max_count:
                max_count = len(buffer)
                selected_task_type = task_type
        return selected_task_type

    def batch(self, samples: List[TransfusionTaskSample]) -> TransfusionTaskBatch:
        """
        组装最终的 Tensor Batch。
        负责动态 Padding、-100 转换，以及触发 Megatron 原生的 Attention 隔离。

        注意: 同一 batch 中只允许单一模态的数据。
        使用 buffer 机制按模态分组，MESSAGE 和 TEXT 共用一个 buffer。
        """
        batch_size = len(samples)

        # 将新样本按模态分组放入 buffer (MESSAGE 和 TEXT 合并)
        for s in samples:
            normalized_type = self._normalize_task_type(s.task_type)
            self._modal_buffers[normalized_type].append(s)

        # 模态选择
        if self._batch_schedule:
            selected_task_type = self._select_round_robin(batch_size)
        else:
            selected_task_type = self._select_max_buffer(batch_size)

        if selected_task_type is None:
            raise SkipSample()

        # 从选中的 buffer 取出 batch_size 个样本
        buffer = self._modal_buffers[selected_task_type]
        samples = buffer[:batch_size]
        self._modal_buffers[selected_task_type] = buffer[batch_size:]

        # 调度统计
        self._batch_step_counter += 1
        self._batch_modality_counts[selected_task_type] += 1
        if self._batch_schedule and self._batch_step_counter % 100 == 0:
            total = self._batch_step_counter
            ratios = {
                k.name: f"{v}/{total}({v / total * 100:.1f}%)"
                for k, v in self._batch_modality_counts.items()
            }
            logger.info(f"[BatchScheduler] step {total}, 模态比例: {ratios}")

        imgs, image_thw_grids = [], []
        videos, video_thw_grids = [], []

        # batch 拼接时保持每类 mask 原本的坐标空间。
        batch_total_img_und_masks = []  # input-token 掩码; True = clean ref/understanding/feedback
        batch_image_gen_masks = []  # ViT/raw 空间掩码; True = 生成目标 token
        batch_gen_seq_lens = []  # 每张生成图的 raw token 数
        batch_image_gen_grid_thw = []  # 每张生成图的 raw grid
        batch_image_und_masks = []  # ViT-output 掩码; True = clean 条件 token
        # 1. 收集视觉特征和生成掩码
        for s in samples:
            if s.imgs is not None and len(s.imgs) > 0:
                # 兼容 imgs 可能是 list of tensors 或者是被 packing 好的单个大 tensor
                if isinstance(s.imgs, list):
                    imgs.extend(s.imgs)
                else:
                    imgs.append(s.imgs)

            if s.image_thw_grids is not None and len(s.image_thw_grids) > 0:
                image_thw_grids.extend([thw for thw in s.image_thw_grids])

            # 视频逻辑暂时为空
            if s.videos is not None and len(s.videos) > 0:
                raise NotImplementedError

            # 过滤收集一维的生成掩码 (剔除空张量，防止纯理解任务报错)
            if s.total_img_und_mask is not None and len(s.total_img_und_mask) > 0:
                batch_total_img_und_masks.append(s.total_img_und_mask)
            if s.image_generate_mask is not None and len(s.image_generate_mask) > 0:
                batch_image_gen_masks.append(s.image_generate_mask)
            if s.image_gen_seq_lens is not None and len(s.image_gen_seq_lens) > 0:
                batch_gen_seq_lens.extend(s.image_gen_seq_lens)
            if s.image_gen_grid_thw is not None and len(s.image_gen_grid_thw) > 0:
                if isinstance(s.image_gen_grid_thw, torch.Tensor):
                    if s.image_gen_grid_thw.ndim == 1:
                        batch_image_gen_grid_thw.append(s.image_gen_grid_thw.unsqueeze(0))
                    else:
                        batch_image_gen_grid_thw.append(s.image_gen_grid_thw)
                else:
                    batch_image_gen_grid_thw.extend(s.image_gen_grid_thw)
            if s.image_understand_mask is not None and len(s.image_understand_mask) > 0:
                batch_image_und_masks.append(s.image_understand_mask)

        # 固定 pad 到 seq_length，确保 SP scatter 时 seq_len 能被 TP 整除
        max_seq_len = self.packing_seq_length
        pad_id = self.pad_id
        batch_size = len(samples)

        text_mat = np.full((batch_size, max_seq_len), pad_id, dtype=np.int64)
        target_mat = np.full((batch_size, max_seq_len), pad_id, dtype=np.int64)
        image_input_masks = np.zeros_like(text_mat, dtype=bool)

        for i, s in enumerate(samples):
            seq_len = min(max_seq_len, len(s.text))
            text_mat[i, :seq_len] = s.text.numpy()[:seq_len]
            target_mat[i, :seq_len] = s.target.numpy()[:seq_len]
            image_input_masks[i, :seq_len] = s.image_input_mask.numpy()[:seq_len]

        tokens = torch.from_numpy(text_mat)

        labels = torch.from_numpy(target_mat)
        labels[labels == pad_id] = -100

        # 构造 loss_mask: 对 labels < 0 的位置不计算 loss
        loss_mask = torch.ones(tokens.size(), dtype=torch.float)

        # 构造 FlashMask startend_row_indices [bs, 1, seq, 2] (int32)
        # 从各样本的 cond_startend_row_indices 中取出（已在 encode 阶段构建）
        # 注: CFG drop 已在 encode 阶段处理，此处直接使用 cond_indices。
        startend_row_indices = torch.zeros((batch_size, 1, max_seq_len, 2), dtype=torch.int32)

        for b in range(batch_size):
            s = samples[b]
            cond_indices = s.cond_startend_row_indices

            if cond_indices is not None:
                seq_len = min(cond_indices.shape[1], max_seq_len)
                startend_row_indices[b, 0, :seq_len, :] = cond_indices[0, :seq_len, :]

        # attention_mask 由 startend_row_indices 接管
        attention_mask = None

        # position_ids: 从 per-sample 构造的 (3, L) pad 到 (3, B, max_seq_len)
        has_position_ids = any(s.position_ids is not None for s in samples)
        if has_position_ids:
            position_ids = torch.zeros(3, batch_size, max_seq_len, dtype=torch.long)
            for i, s in enumerate(samples):
                if s.position_ids is not None:
                    seq_len = min(max_seq_len, s.position_ids.shape[1])
                    position_ids[:, i, :seq_len] = s.position_ids[:, :seq_len]
        else:
            position_ids = None

        loss_mask[labels < 0] = 0.0

        # 组装 packed_sample_lens: [B, max_num_samples], 用于 sample-level loss averaging
        all_sample_lens = []
        for s in samples:
            if s.packed_sample_lens is not None:
                all_sample_lens.append(s.packed_sample_lens)
            else:
                # 非 packing 场景: 一行就是一个样本
                all_sample_lens.append([len(s.text)])
        max_num_samples = max(len(sl) for sl in all_sample_lens)
        packed_sample_lens_mat = torch.zeros(batch_size, max_num_samples, dtype=torch.long)
        for i, sl in enumerate(all_sample_lens):
            for j, l in enumerate(sl):
                packed_sample_lens_mat[i, j] = l

        # 统计各任务类型的图片数量 (通过 image_start token 计数)
        num_images = 0
        num_gen_images = 0
        num_gen_cot_images = 0
        num_und_images = 0
        num_edit_images = 0
        # 统计各模态的训练行数
        num_text_rows = 0
        num_und_rows = 0
        num_gen_rows = 0
        num_gen_cot_rows = 0
        num_gen_cot_text_rows = 0
        num_edit_rows = 0
        for i, s in enumerate(samples):
            # 统计该样本中 image_start token 的数量
            sample_num_images = (s.text == self.image_start_id).sum().item()
            num_images += sample_num_images
            task_type = TaskType.from_str(s.task_type)
            if task_type == TaskType.GENERATION:
                num_gen_images += sample_num_images
                num_gen_rows += 1
            elif task_type == TaskType.GENERATION_COT:
                num_gen_cot_images += sample_num_images
                num_gen_cot_rows += 1
            elif task_type == TaskType.GENERATION_COT_TEXT:
                num_gen_cot_text_rows += 1
            elif task_type == TaskType.UNDERSTANDING or task_type == TaskType.SFT_IMAGE_UND:
                num_und_images += sample_num_images
                num_und_rows += 1
            elif task_type == TaskType.EDITING:
                num_edit_images += len(s.image_gen_seq_lens) if s.image_gen_seq_lens else 0
                num_edit_rows += 1
            elif task_type == TaskType.TEXT or task_type == TaskType.MESSAGE or task_type == TaskType.SFT_TEXT:
                num_text_rows += 1

        return TransfusionTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            pixel_values=torch.vstack(imgs) if len(imgs) > 0 else None,
            pixel_values_videos=None,
            image_grid_thw=torch.from_numpy(np.array(image_thw_grids)) if len(image_thw_grids) > 0 else None,
            video_grid_thw=None,
            image_input_mask=torch.from_numpy(image_input_masks),
            video_input_mask=torch.zeros(image_input_masks.shape, dtype=torch.bool),
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
            total_img_und_mask=torch.cat(batch_total_img_und_masks, dim=0) if batch_total_img_und_masks else None,
            image_generate_mask=torch.cat(batch_image_gen_masks, dim=0) if batch_image_gen_masks else None,
            image_understand_mask=torch.cat(batch_image_und_masks, dim=0) if batch_image_und_masks else None,
            image_gen_seq_lens=torch.tensor(batch_gen_seq_lens, dtype=torch.long) if batch_gen_seq_lens else None,
            image_gen_grid_thw=(
                torch.cat([
                    thw if isinstance(thw, torch.Tensor) and thw.ndim == 2
                    else torch.as_tensor(thw, dtype=torch.long).view(1, 3)
                    for thw in batch_image_gen_grid_thw
                ], dim=0)
                if batch_image_gen_grid_thw else None
            ),
            num_gen_images_per_sample=torch.tensor(
                [len(s.image_gen_seq_lens) if s.image_gen_seq_lens else 0 for s in samples],
                dtype=torch.long,
            ),
            startend_row_indices=startend_row_indices,
            task_type=torch.tensor(
                [self._normalize_task_type(TaskType.from_str(s.task_type)) for s in samples],
                dtype=torch.long,
            ),
            packed_sample_lens=packed_sample_lens_mat,
            num_images=num_images,
            num_gen_images=num_gen_images,
            num_gen_cot_images=num_gen_cot_images,
            num_und_images=num_und_images,
            num_edit_images=num_edit_images,
            num_text_rows=num_text_rows,
            num_und_rows=num_und_rows,
            num_gen_rows=num_gen_rows,
            num_gen_cot_rows=num_gen_cot_rows,
            num_gen_cot_text_rows=num_gen_cot_text_rows,
            num_edit_rows=num_edit_rows,
        )

    def encode_batch(self, batch: TransfusionTaskBatch) -> dict:
        """Convert a collated task batch into the model-facing input dictionary."""
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]

        raw["visual_inputs"] = Qwen2_5_VLVisualInputs(
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
        )

        raw["tokens"] = batch.input_ids
        raw["labels"] = batch.labels
        raw["attention_mask"] = batch.attention_mask
        raw["pixel_values"] = batch.pixel_values
        raw["image_grid_thw"] = batch.image_grid_thw
        raw["image_input_mask"] = batch.image_input_mask
        raw["image_generate_mask"] = batch.image_generate_mask
        raw["image_understand_mask"] = batch.image_understand_mask
        raw["image_gen_seq_lens"] = batch.image_gen_seq_lens
        raw["image_gen_grid_thw"] = batch.image_gen_grid_thw
        raw["num_gen_images_per_sample"] = batch.num_gen_images_per_sample
        raw["total_img_und_mask"] = batch.total_img_und_mask
        raw["startend_row_indices"] = batch.startend_row_indices
        raw["position_ids"] = batch.position_ids
        raw["packed_sample_lens"] = batch.packed_sample_lens

        return raw

    @stateless
    def encode_sample(self, sample: RawMultimodalSample) -> TransfusionTaskSample:
        """Encode one source sample, dropping intentionally rejected samples."""
        try:
            yield from self._encode_sample_items(sample)
        except SkipSample:
            logger.debug(
                "[SkipSample] Drop sample during encoding, key=%s, task_type=%s",
                getattr(sample, "__key__", None),
                getattr(sample, "task_type", None),
            )

    def _encode_sample_items(self, sample: RawMultimodalSample) -> TransfusionTaskSample:
        """
        这是主入口，充当路由器 (Dispatcher)
        """

        task_type = TaskType.from_str(sample.task_type)

        # Text-only routes do not need image preprocessing.
        if task_type in (
            TaskType.TEXT,
            TaskType.MESSAGE,
            TaskType.SFT_TEXT,
            TaskType.GENERATION_COT_TEXT,
        ):
            processed_vision = None
        else:
            # 根据任务类型选择对应的像素范围
            px_cfg = self.pixel_config.get(
                sample.task_type,
                {
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    "max_input_pixels": None,
                },
            )
            vision_images = sample.imgs
            _validate_max_input_pixels(vision_images, px_cfg.get("max_input_pixels"))
            if task_type == TaskType.EDITING:
                vision_images = align_single_edit_reference(
                    sample.imgs, sample.text, sample.mask
                )
            processed_vision = process_vision(
                self.image_processor,
                vision_images,
                sample.videos,
                min_pixels=px_cfg["min_pixels"],
                max_pixels=px_cfg["max_pixels"],
                max_input_pixels=px_cfg.get("max_input_pixels"),
                square_crop=self.square_crop,
                resolution_buckets=self.resolution_buckets.get(sample.task_type),
                resolution_bucket_threshold=self.resolution_bucket_threshold,
            )

        # 核心分发：根据任务类型，调用专门的私有方法去排布 Token 和 Mask
        if task_type == TaskType.UNDERSTANDING:
            result = self._encode_understanding(sample, processed_vision)
        elif task_type == TaskType.GENERATION:
            # CFG: 在 encode 阶段根据 cfg_drop_rate 决定是否丢弃真实 prompt 文本。
            cfg_dropped = self.cfg_drop_rate > 0 and random.random() < self.cfg_drop_rate
            result = self._encode_generation(sample, processed_vision, cfg_dropped=cfg_dropped)
        elif task_type == TaskType.GENERATION_COT:
            # COT 样本需要监督 <think> 内 refined prompt，不做 CFG 文本丢弃。
            result = self._encode_generation_cot(sample, processed_vision)
        elif task_type == TaskType.GENERATION_COT_TEXT:
            result = self._encode_generation_cot_text(sample)
        elif task_type == TaskType.EDITING:
            # CFG 三态采样:
            #   - "full": 概率 cfg_drop_rate, 仅最后一轮丢文本 + 条件图像
            #   - "text": 概率 cfg_drop_rate, 仅最后一轮丢文本, 保留图像条件
            #   - "none": 其余情况, 不丢
            # 单次均匀采样保证两区间互斥, 总 drop 概率 = 2 * cfg_drop_rate
            if self.cfg_drop_rate > 0:
                r = random.random()
                if r < self.cfg_drop_rate:
                    cfg_mode = "full"
                elif r < 2 * self.cfg_drop_rate:
                    cfg_mode = "text"
                else:
                    cfg_mode = "none"
            else:
                cfg_mode = "none"

            # 统计三态采样占比, 期望比例 full : text : none ≈ p : p : (1-2p)
            self._cfg_mode_counts[cfg_mode] += 1
            if self._cfg_mode_log_interval > 0:
                total = sum(self._cfg_mode_counts.values())
                if total % self._cfg_mode_log_interval == 0:
                    n_full = self._cfg_mode_counts["full"]
                    n_text = self._cfg_mode_counts["text"]
                    n_none = self._cfg_mode_counts["none"]
                    logger.info(
                        f"[CFG-mode-stats] total={total} "
                        f"full={n_full} ({n_full/total:.3f}) "
                        f"text={n_text} ({n_text/total:.3f}) "
                        f"none={n_none} ({n_none/total:.3f}) "
                        f"(expected ~ {self.cfg_drop_rate:.2f}:{self.cfg_drop_rate:.2f}:{1-2*self.cfg_drop_rate:.2f})"
                    )
            result = self._encode_editing_vit(sample, processed_vision, cfg_mode=cfg_mode)
        elif task_type == TaskType.TEXT:
            result = self._encode_text(sample)
        elif task_type == TaskType.MESSAGE:
            result = self._encode_message(sample)
        elif task_type == TaskType.SFT_TEXT:
            result = self._encode_sft_text(sample)
        elif task_type == TaskType.SFT_IMAGE_UND:
            result = self._encode_sft_image_und(sample, processed_vision)
        else:
            raise ValueError(f"未知的任务类型: {task_type}")

        # 统一处理：将生成器和普通返回值都转为迭代器
        if isinstance(result, types.GeneratorType):
            results = result  # 已经是生成器，直接使用
        else:
            results = [result]  # 包装为列表

        for item in results:
            # 检查单条样本是否超过 packing 长度限制，超长则跳过
            if self.is_packing_enabled and item.total_len > self.packing_seq_length:
                logger.warning(
                    f"[SkipSample] Sample exceeds packing_seq_length: "
                    f"total_len={item.total_len}, packing_seq={self.packing_seq_length}, "
                    f"__key__={sample.__key__}, task_type={sample.task_type}"
                )
                raise SkipSample()

            yield item

    def _encode_understanding(self, sample, processed_vision) -> TransfusionTaskSample:
        """
        专门处理【多模态理解/预训练】任务的策略函数。

        数据格式:
        - sample.text: List[str], e.g. ["Hello", "<image>", "world", "<image>", "friend"]
        - sample.mask: List[int], e.g. [1, 0, 1, 0, 1]
        - sample.imgs: List[Image.Image], 按顺序对应 <image> 占位符
        - processed_vision: Qwen2VLImageProcessor 处理后的结果
          - image_inputs: {"pixel_values": tensor, "image_grid_thw": tensor}
          - image_grid_thw: 每张图的 [t, h, w] 网格维度

        对齐 Qwen3-VL chat template:
        <|im_start|>user\n{input with <|vision_start|><|image_pad|>...<|vision_end|>}<|im_end|>\n
        <|im_start|>assistant\n{output}<|im_end|>
        """
        
        if not isinstance(sample.text, list) or not isinstance(sample.mask, list):
            logger.warning(
                "[Understanding] Skip malformed sample without list text/mask, key=%s",
                sample.__key__,
            )
            raise SkipSample()
        if len(sample.text) != len(sample.mask) or not sample.text:
            logger.warning(
                "[Understanding] Skip malformed text/mask lengths: text=%d mask=%d key=%s",
                len(sample.text),
                len(sample.mask),
                sample.__key__,
            )
            raise SkipSample()
        if any(value not in (0, 1) for value in sample.mask):
            logger.warning(
                "[Understanding] Skip sample with non-binary mask, key=%s", sample.__key__
            )
            raise SkipSample()
        if processed_vision is None:
            raise SkipSample()

        # 1. 从 processed_vision 提取视觉特征
        image_inputs = processed_vision.get("image_inputs")
        image_grid_thw = processed_vision.get("image_grid_thw")
        image_grid_thw_ori = image_grid_thw.clone() if image_grid_thw is not None else None
        if image_grid_thw is not None:
            # 暂时写死merge size = 2
            image_grid_thw[..., 1] = image_grid_thw[..., 1] // 2
            image_grid_thw[..., 2] = image_grid_thw[..., 2] // 2

        # 提取 pixel_values (Qwen2VLImageProcessor 返回格式)
        if image_inputs:
            pixel_values = image_inputs.get("pixel_values")
        else:
            pixel_values = None

        # 2. 处理文本特征
        text_fragments = list(sample.text)  # 复制一份避免修改原始数据
        mask = list(sample.mask)

        # 2.0 检测文本是否已包含对话模板（如 <|im_start|>system/user/assistant）
        has_template = _has_chat_template(text_fragments)

        # 2.1 根据 source_id 查找并插入 question prompt。理解任务的输入协议固定为
        # image(s) -> question -> answer；因此 question 必须放在最后一个真实图片块后。
        # 已经渲染好的 ChatML 样本包含自己的 question，不能再次注入映射 prompt。
        # MidTrain 的部分答案已经包含 <think>...</think>，其推理内容仍由原始 mask
        # 负责监督（opening <think> 本身除外），不受 enable_think 控制。
        source_id = getattr(sample, "source_id", None)
        has_explicit_user_text = any(
            mask_value == 0
            and fragment != _IMAGE_PLACEHOLDER
            and isinstance(fragment, str)
            and fragment.strip()
            for fragment, mask_value in zip(text_fragments, mask)
        )
        if (
            not has_template
            and not has_explicit_user_text
            and source_id
            and self.prompt_mapping
        ):
            prompt = self._get_random_prompt(source_id)
            if prompt:
                prompt = strip_textual_image_placeholders(prompt)
                last_image_pos = max(
                    (
                        idx
                        for idx, fragment in enumerate(text_fragments)
                        if fragment == _IMAGE_PLACEHOLDER
                    ),
                    default=-1,
                )
                insert_pos = last_image_pos + 1
                text_fragments.insert(insert_pos, prompt)
                mask.insert(insert_pos, 0)  # prompt 不参与训练

        # 2.2 wobox 源：去除 refbox 标记，只保留纯文本
        if source_id and source_id.endswith("_wobox"):
            text_fragments = [
                strip_refbox(frag) if isinstance(frag, str) and "<|object_ref_start|>" in frag else frag
                for frag in text_fragments
            ]

        # 独立的 <image> fragment 是真实图片的结构位置，必须保留；普通文本中
        # 残留的字面 <image> 不应进入 CE 标签，否则模型会学会在理解回答中输出它。
        cleaned_fragments = []
        cleaned_mask = []
        for frag, mask_value in zip(text_fragments, mask):
            if frag == _IMAGE_PLACEHOLDER:
                if mask_value != 0:
                    logger.warning(
                        "[Understanding] Skip image placeholder with mask=%s, key=%s",
                        mask_value,
                        sample.__key__,
                    )
                    raise SkipSample()
                cleaned_fragments.append(frag)
                cleaned_mask.append(mask_value)
                continue
            if not isinstance(frag, str):
                logger.warning(
                    "[Understanding] Skip non-string text fragment, key=%s", sample.__key__
                )
                raise SkipSample()
            frag = strip_textual_image_placeholders(frag)
            if frag.strip():
                cleaned_fragments.append(frag)
                cleaned_mask.append(mask_value)
        text_fragments = cleaned_fragments
        mask = cleaned_mask

        if has_template:
            text_fragments, mask = _move_template_images_after_user_header(
                text_fragments, mask
            )
        else:
            # converter 按 matched_text_index 保留了原始文本顺序。理解协议要求
            # 图片先于文本，但非图片片段可能是交错的多轮问答或结构化字段，不能
            # 再按 mask 全局重排，否则会把 ``key -> value -> key -> value`` 改乱。
            ordered_fragments = [
                (fragment, mask_value)
                for fragment, mask_value in zip(text_fragments, mask)
                if fragment == _IMAGE_PLACEHOLDER
            ]
            ordered_fragments.extend(
                (fragment, mask_value)
                for fragment, mask_value in zip(text_fragments, mask)
                if fragment != _IMAGE_PLACEHOLDER
            )
            text_fragments = [fragment for fragment, _ in ordered_fragments]
            mask = [mask_value for _, mask_value in ordered_fragments]

        # Think blocks in ordinary understanding data are already part of the
        # supervised text. Preserve every marker through length truncation;
        # otherwise a clipped reasoning block would train an invalid protocol.
        expected_think_start_count = sum(
            fragment.count("<think>") for fragment in text_fragments
        )
        expected_think_end_count = sum(
            fragment.count("</think>") for fragment in text_fragments
        )
        if expected_think_start_count != expected_think_end_count:
            logger.debug(
                "[Understanding] Skip sample with unbalanced think tags: "
                "start=%d end=%d key=%s",
                expected_think_start_count,
                expected_think_end_count,
                sample.__key__,
            )
            raise SkipSample()

        has_answer = any(
            mask_value == 1 and frag != _IMAGE_PLACEHOLDER
            for frag, mask_value in zip(text_fragments, mask)
        )
        if not has_answer:
            logger.warning(
                "[Understanding] Skip sample without supervised answer, source_id=%s key=%s",
                source_id,
                sample.__key__,
            )
            raise SkipSample()

        # 3. 预处理图片信息：计算每张图的 token 数量
        imgs_count = len(sample.imgs) if sample.imgs else 0

        # image_grid_thw 格式: [num_images, 3] -> [t, h, w] for each image
        # 如果是单张图，可能是 [3] 一维数组
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, np.ndarray):
                if image_grid_thw.ndim == 1:
                    # 单张图: [t, h, w]
                    image_thw_list = [image_grid_thw]
                else:
                    # 多张图: [num_images, 3]
                    image_thw_list = [image_grid_thw[i] for i in range(image_grid_thw.shape[0])]
            elif isinstance(image_grid_thw, torch.Tensor):
                if image_grid_thw.ndim == 1:
                    image_thw_list = [image_grid_thw]
                else:
                    image_thw_list = [image_grid_thw[i] for i in range(image_grid_thw.shape[0])]
            else:
                image_thw_list = [image_grid_thw]
        else:
            image_thw_list = []

        # 计算每张图的 token 数量
        image_patch_counts = []
        for thw in image_thw_list:
            if thw is not None:
                num_patches = int(np.prod(thw)) if isinstance(thw, np.ndarray) else int(thw.prod())
            else:
                num_patches = 256  # 默认值
            image_patch_counts.append(num_patches)

        # 4. 构建 token 序列 - 对齐 Qwen3-VL chat template
        input_ids_list = []
        loss_mask_list = []

        # 基于 mask 判断是否需要训练：mask=0 表示 user（不训练），mask=1 表示 assistant（训练）
        # 当 mask 变化时，需要插入相应的 chat template token
        prev_mask = None
        img_idx = 0

        for i, fragment in enumerate(text_fragments):
            current_mask = mask[i] if i < len(mask) else 1

            # 检测 mask 变化，插入角色切换 token（仅在数据没有自带模板时）
            if not has_template:
                if prev_mask is None:
                    # 第一个 fragment，根据 mask 决定开头
                    if current_mask == 0:
                        # user 开头: <|im_start|>user\n
                        role_prefix = [self.im_start_id] + self.user_ids + self.nl_ids
                    else:
                        # assistant 开头: <|im_start|>assistant\n
                        role_prefix = [self.im_start_id] + self.assistant_ids + self.nl_ids
                    input_ids_list.extend(role_prefix)
                    loss_mask_list.extend([0] * len(role_prefix))
                elif prev_mask != current_mask:
                    # mask 发生变化，需要切换角色
                    if current_mask == 1:
                        # user(0) → assistant(1): <|im_end|>\n<|im_start|>assistant\n
                        role_switch = ([self.im_end_id] + self.nl_ids
                                       + [self.im_start_id] + self.assistant_ids + self.nl_ids)
                    else:
                        # assistant(1) → user(0): <|im_end|>\n<|im_start|>user\n
                        role_switch = ([self.im_end_id] + self.nl_ids
                                       + [self.im_start_id] + self.user_ids + self.nl_ids)
                    input_ids_list.extend(role_switch)
                    role_switch_mask = [1 if prev_mask == 1 else 0] + [0] * (len(role_switch) - 1)
                    loss_mask_list.extend(role_switch_mask)

            if fragment == "<image>":
                # 插入图片 token 序列
                # Dense Pixel-DiT inserts one timestep token before the image patches.
                if img_idx < imgs_count:
                    num_patches = image_patch_counts[img_idx] if img_idx < len(image_patch_counts) else 256

                    # 超长跳过机制：检查加入该 image token 后是否会在截断位置
                    # 如果 input_ids[packing_seq_length - 1] 会落在 image_token 上，则不加入该 image
                    current_len = len(input_ids_list)
                    extra_tokens = 3  # start + timestep + patches + end
                    image_token_len = num_patches + extra_tokens
                    new_len = current_len + image_token_len
                    if new_len > self.packing_seq_length:
                        pad_offset = 2  # start + timestep precede image patches
                        image_token_start = current_len + pad_offset
                        image_token_end = image_token_start + num_patches
                        truncate_idx = self.packing_seq_length - 1
                        if image_token_start <= truncate_idx < image_token_end:
                            logger.debug(
                                f"[Understanding] Skip image token due to truncation at image_token, "
                                f"current_len={current_len}, image_token_len={image_token_len}, "
                                f"packing_seq={self.packing_seq_length}, __key__={sample.__key__}"
                            )
                            break

                    input_ids_list.extend(
                        [self.image_start_id]
                        + [self.time_token_id]
                        + [self.image_token_id] * num_patches
                        + [self.image_end_id]
                    )

                    # 图片不参与 loss 计算
                    loss_mask_list.extend([0] * (num_patches + extra_tokens))

                    img_idx += 1
                else:
                    # 图片数量不足，跳过
                    input_ids_list.append(self.image_token_id)
                    loss_mask_list.append(0)
            elif fragment:
                # 处理普通文本，根据 mask 决定是否参与训练
                text_tokens = self.hf_tokenizer.encode(fragment, add_special_tokens=False)

                # 检查是否超长，如果超长则截断
                current_len = len(input_ids_list)
                remaining_len = self.packing_seq_length - current_len
                if remaining_len <= 0:
                    break
                if len(text_tokens) > remaining_len:
                    text_tokens = text_tokens[:remaining_len]

                input_ids_list.extend(text_tokens)
                # mask=0 表示 user（不训练），mask=1 表示 assistant（训练）
                fragment_loss_mask = [current_mask] * len(text_tokens)
                if current_mask == 1:
                    self._mask_opening_think_in_fragment(
                        fragment, text_tokens, fragment_loss_mask
                    )
                loss_mask_list.extend(fragment_loss_mask)
            else:
                # 空 fragment 不更新 prev_mask，避免触发无意义的角色切换
                continue

            prev_mask = current_mask

        # 4.2 添加结尾: <|im_end|>（仅在数据没有自带模板时，且还有空间）
        if not has_template and len(input_ids_list) < self.packing_seq_length:
            input_ids_list.append(self.im_end_id)
            # 如果最后是 assistant，结束标记参与训练；如果是 user，不参与
            loss_mask_list.append(1 if prev_mask == 1 else 0)

        # Text fragments may have been truncated to fit packing_seq_length.
        # Reject the sample if either side of an original think block was clipped.
        think_start_ids = getattr(self, "think_start_ids", None)
        if not think_start_ids:
            think_start_ids = self.hf_tokenizer.encode(
                "<think>", add_special_tokens=False
            )
        think_end_ids = getattr(self, "think_end_ids", None)
        if not think_end_ids:
            think_end_ids = self.hf_tokenizer.encode(
                "</think>", add_special_tokens=False
            )
        retained_think_start_count = _count_token_subsequence(
            input_ids_list, think_start_ids
        )
        retained_think_end_count = _count_token_subsequence(
            input_ids_list, think_end_ids
        )
        if (
            retained_think_start_count != expected_think_start_count
            or retained_think_end_count != expected_think_end_count
        ):
            logger.debug(
                "[Understanding] Skip sample with truncated think block: "
                "expected=(%d,%d) retained=(%d,%d) packing_seq=%d key=%s",
                expected_think_start_count,
                expected_think_end_count,
                retained_think_start_count,
                retained_think_end_count,
                self.packing_seq_length,
                sample.__key__,
            )
            raise SkipSample()

        if getattr(self, "use_v5_route_tokens", False):
            self._insert_route_tokens_after_assistant_headers(
                input_ids_list,
                self.text_gate_token_id,
                token_ownership=loss_mask_list,
            )
            if len(input_ids_list) > self.packing_seq_length:
                raise SkipSample()

        # 理解样本必须至少保留一张完整图。若长文本导致所有图像块都无法放入
        # packing_seq_length，直接跳过，避免无图理解样本进入训练。
        if img_idx == 0:
            logger.debug(
                f"[Understanding] Skip sample without usable image after truncation, "
                f"packing_seq={self.packing_seq_length}, __key__={sample.__key__}"
            )
            raise SkipSample()

        # 5. 转换为 numpy/torch
        input_ids = np.array(input_ids_list, dtype=np.int64)
        # image_input_mask 直接由 input_ids 计算得到
        image_input_mask = torch.from_numpy(input_ids == self.image_token_id)
        loss_mask = torch.from_numpy(
            self._align_loss_mask_to_next_token(loss_mask_list)
        )

        # 6. 构建 Labels (用于训练): 下一个 token 预测
        pad_id = self.pad_id
        target = input_ids.copy()
        target = np.roll(target, shift=-1)
        target[-1] = pad_id

        # 应用 loss_mask 到 labels
        loss_mask_np = loss_mask.numpy()
        target[loss_mask_np == 0] = pad_id
        if not np.any(target != pad_id):
            logger.warning(
                "[Understanding] Skip sample without CE labels after truncation, key=%s",
                sample.__key__,
            )
            raise SkipSample()

        # 7. 理解任务 mask 构建
        num_image_tokens = np.sum(input_ids == self.image_token_id)
        # Understanding images are clean ViT inputs, never diffusion targets.
        image_generate_mask = np.zeros(0, dtype=bool)
        image_understand_mask = np.ones(num_image_tokens, dtype=bool)
        total_img_und_mask = np.ones(num_image_tokens, dtype=bool)
        image_gen_seq_lens = []

        # 7.1 构建 3D M-RoPE position_ids（理解任务）。
        # 直接按连续 image_pad 块定位图像，因此同时支持：
        # base:   vision_start + image_pad...
        # Dense Pixel-DiT: vision_start + time_token + image_pad...
        position_ids = self._build_position_ids(
            input_ids.tolist(), image_thw_list[:img_idx]
        )

        # 8. 转换 image_thw_grids 为 List[torch.Tensor]
        # 只使用实际添加的图片 (img_idx 个)，跳过因超长未添加的图片
        image_thw_final = []
        if image_grid_thw_ori is not None:
            if isinstance(image_grid_thw_ori, np.ndarray):
                if image_grid_thw_ori.ndim == 1:
                    image_thw_final = [torch.from_numpy(image_grid_thw_ori)] if img_idx > 0 else []
                else:
                    image_thw_final = [
                        torch.from_numpy(image_grid_thw_ori[i]) 
                        for i in range(min(img_idx, image_grid_thw_ori.shape[0]))
                    ]
            elif isinstance(image_grid_thw_ori, torch.Tensor):
                if image_grid_thw_ori.ndim == 1:
                    image_thw_final = [image_grid_thw_ori] if img_idx > 0 else []
                else:
                    image_thw_final = [
                        image_grid_thw_ori[i] for i in range(min(img_idx, image_grid_thw_ori.shape[0]))
                        ]

        # 8.1 截取 pixel_values，只保留实际使用的图片 patches
        # pixel_values 格式: [total_patches, channels] 或 [total_patches, ...]
        # 注意: pixel_values 行数对应 merge 前的 patch 数 (t*h*w)，
        #       必须使用 image_grid_thw_ori (未除以 merge_size) 来计算截断位置，
        #       而不是 image_patch_counts（基于除以 merge_size 后的 grid 计算的）。
        if pixel_values is not None and img_idx < imgs_count:
            # 基于原始 grid (merge 前) 计算实际使用的 patches 总数
            if image_grid_thw_ori is not None:
                if isinstance(image_grid_thw_ori, np.ndarray):
                    if image_grid_thw_ori.ndim == 1:
                        used_patches = int(np.prod(image_grid_thw_ori)) if img_idx > 0 else 0
                    else:
                        used_patches = sum(
                            int(np.prod(image_grid_thw_ori[i]))
                            for i in range(min(img_idx, image_grid_thw_ori.shape[0]))
                        )
                elif isinstance(image_grid_thw_ori, torch.Tensor):
                    if image_grid_thw_ori.ndim == 1:
                        used_patches = int(image_grid_thw_ori.prod().item()) if img_idx > 0 else 0
                    else:
                        used_patches = sum(
                            int(image_grid_thw_ori[i].prod().item())
                            for i in range(min(img_idx, image_grid_thw_ori.shape[0]))
                        )
                else:
                    used_patches = sum(image_patch_counts[:img_idx])
            else:
                used_patches = 0
            if hasattr(pixel_values, 'shape') and len(pixel_values.shape) > 0:
                pixel_values = pixel_values[:used_patches]

        # 9. 构建 FlashMask startend_row_indices
        cond_indices, nocond_indices = create_sample_startend_row_indices(
            input_ids=torch.from_numpy(input_ids),
            img_start_id=self.image_start_id,
            img_end_id=self.image_end_id,
            bidirectional=True,
        )

        # ===== DEBUG: 每个 source_id 采样一条，前 N 条样本内采集后关闭 =====
        import os
        debug_dir = os.environ.get("TRANSFUSION_DEBUG_DIR", "")
        _debug_limit = int(os.environ.get("TRANSFUSION_DEBUG_NUM", "1024"))
        if debug_dir:
            if not hasattr(self, "_debug_und_seen"):
                self._debug_und_seen = set()
                self._debug_und_total = 0
            if self._debug_und_total < _debug_limit:
                self._debug_und_total += 1
                _source = getattr(sample, "source_id", None) or "unknown"
                if _source not in self._debug_und_seen:
                    self._debug_und_seen.add(_source)
                    # 跨进程去重：原子创建标记文件，成功则写，失败则跳过
                    os.makedirs(debug_dir, exist_ok=True)
                    safe_source = str(_source).replace("/", "_")[:60]
                    marker = os.path.join(debug_dir, f".seen_{safe_source}")
                    try:
                        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                    except FileExistsError:
                        pass
                    else:
                        sample_key = getattr(sample, "__key__", "unknown")

                        decoded_text = self.hf_tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
                        image_pad_token = self.hf_tokenizer.decode([self.image_token_id])
                        for cnt in image_patch_counts[:img_idx]:
                            decoded_text = decoded_text.replace(image_pad_token * cnt, f"{image_pad_token}*{cnt}", 1)

                        train_tokens = self.hf_tokenizer.decode(
                            [tid for tid, m in zip(input_ids.tolist(), loss_mask_list) if m == 1],
                            skip_special_tokens=False
                        )

                        debug_info = (
                            f"{'='*80}\n"
                            f"[source={_source}] key={sample_key}\n"
                            f"[完整输入序列] (len={len(input_ids)}):\n{decoded_text}\n"
                            f"{'-'*40}\n"
                            f"[训练部分]:\n{train_tokens}\n"
                            f"{'-'*40}\n"
                            f"[图片] 原始={imgs_count}, 实际使用={img_idx}, grid_thw={image_thw_list[:img_idx]}\n"
                            f"{'='*80}\n\n"
                        )

                        debug_file = os.path.join(debug_dir, "debug_understanding.txt")
                        with open(debug_file, "a", encoding="utf-8") as f:
                            f.write(debug_info)

                        if sample.imgs:
                            for i, img in enumerate(sample.imgs[:img_idx]):
                                if img is None:
                                    continue
                                if isinstance(img, str):
                                    with open(debug_file, "a", encoding="utf-8") as f:
                                        f.write(f"  img[{i}] source: {img}\n")
                                else:
                                    img_path = os.path.join(debug_dir, f"{safe_source}_img{i}.png")
                                    try:
                                        img.save(img_path)
                                    except Exception:
                                        pass
        # ===== END DEBUG =====

        return TransfusionTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=pixel_values if pixel_values is not None else [],
            image_thw_grids=image_thw_final,
            image_input_mask=image_input_mask,
            total_img_und_mask=torch.from_numpy(total_img_und_mask),
            image_generate_mask=torch.from_numpy(image_generate_mask),
            image_understand_mask=torch.from_numpy(image_understand_mask),
            image_gen_seq_lens=image_gen_seq_lens,
            videos=[],
            video_thw_grids=[],
            video_input_mask=torch.zeros_like(image_input_mask)
            if len(image_input_mask) > 0
            else torch.empty(0, dtype=torch.bool),
            task_type=sample.task_type,
            text=torch.from_numpy(input_ids),
            target=torch.from_numpy(target),
            total_len=len(input_ids),
            position_ids=position_ids,
            cond_startend_row_indices=cond_indices,
            nocond_startend_row_indices=nocond_indices,
        )

    # 生成任务的指令性 prompt 模板
    GEN_INSTRUCTION_PROMPTS = [
        # 中文指令
        "请根据以下描述生成一张图片：",
        "根据下面的描述生成图片：",
        "生成图片：",
        "请画一张图：",
        "帮我生成一张图片：",
        "根据描述创建图像：",
        "请根据文字描述绘制图片：",
        "按照以下内容生成图片：",
        "生成一张符合描述的图片：",
        "请为以下内容创作一张图片：",
        "根据提示生成图像：",
        "图片生成：",
        # 英文指令
        "Generate an image based on the following description: ",
        "Create an image: ",
        "Please generate an image: ",
        "Draw an image based on: ",
        "Create a picture: ",
        "Generate: ",
        "Please draw: ",
        "Create an image from the description: ",
        "Generate a picture: ",
        "Please create an image based on: ",
    ]

    # 带 prompt refine 思考链的生成任务指令。按样本语言抽取，避免中英混写。
    GEN_COT_INSTRUCTION_PROMPTS_ZH = [
        "请先思考并细化下面的图像需求，再生成图片：",
        "请先将下面的描述改写成更完整的图像生成提示，再生成图片：",
        "请先理解并补充画面细节，然后根据结果生成图片：",
        "请先在心里完善这段图像描述，再生成对应图片：",
        "请先梳理主体、场景、风格和细节，再创作图片：",
    ]

    GEN_COT_INSTRUCTION_PROMPTS_EN = [
        "First refine the following image request, then generate the image: ",
        "Think through and enrich the following description before generating the image: ",
        "First turn the request into a more complete image prompt, then create the image: ",
        "Refine the subject, scene, style, and details before generating the image: ",
        "First reason about the visual details, then generate an image from the request: ",
    ]

    def _choose_generation_cot_instruction(self, language: str) -> str:
        """Choose a language-appropriate prompt-refinement instruction."""
        lang = (language or "").lower()
        if lang.startswith("zh") or lang in ("cn", "chinese"):
            return random.choice(self.GEN_COT_INSTRUCTION_PROMPTS_ZH)
        if lang.startswith("en") or lang == "english":
            return random.choice(self.GEN_COT_INSTRUCTION_PROMPTS_EN)
        return random.choice(
            self.GEN_COT_INSTRUCTION_PROMPTS_ZH + self.GEN_COT_INSTRUCTION_PROMPTS_EN
        )

    def _encode_generation(self, sample, processed_vision, cfg_dropped: bool = False) -> TransfusionTaskSample:
        """Encode a text-to-image sample for Dense Pixel-DiT."""
        image_thw_grids_ori = processed_vision["image_grid_thw"].clone()
        if image_thw_grids_ori.numel() == 0:
            raise SkipSample()
        image_thw_grids = image_thw_grids_ori.clone()
        image_thw_grids[..., 1] //= 2
        image_thw_grids[..., 2] //= 2
        flattened_imgs = processed_vision.get("image_inputs")
        num_image_patches = int(image_thw_grids.prod().item())

        instruction = random.choice(self.GEN_INSTRUCTION_PROMPTS)
        instruction_tokens = self.hf_tokenizer.encode(instruction, add_special_tokens=False)
        text_tokens = self.hf_tokenizer.encode(sample.text[0], add_special_tokens=False)
        if cfg_dropped and not self.cfg_keep_chat_template:
            input_ids_list = (
                [self.image_start_id, self.time_token_id]
                + [self.image_token_id] * num_image_patches
                + [self.image_end_id, self.im_end_id]
            )
        else:
            user_content_tokens = instruction_tokens if cfg_dropped else instruction_tokens + text_tokens
            route_tokens = []
            if self.use_v5_route_tokens and not cfg_dropped:
                route_tokens.append(self.image_gate_token_id)
                if self.generation_empty_think:
                    route_tokens.extend(self.hf_tokenizer.encode("<think>\n\n</think>", add_special_tokens=False))
                    route_tokens.append(self.recaption_end_token_id)
            input_ids_list = (
                [self.im_start_id] + self.user_ids + self.nl_ids + user_content_tokens
                + [self.im_end_id] + self.nl_ids
                + [self.im_start_id] + self.assistant_ids + self.nl_ids
                + route_tokens + [self.image_start_id, self.time_token_id]
                + [self.image_token_id] * num_image_patches
                + [self.image_end_id, self.im_end_id]
            )
        input_ids = np.array(input_ids_list, dtype=np.int64)

        target = np.roll(np.full_like(input_ids, self.pad_id), shift=-1)
        target[-1] = self.pad_id
        if self.use_v5_route_tokens and not cfg_dropped:
            gate_pos = input_ids_list.index(self.image_gate_token_id)
            target[gate_pos - 1] = self.image_gate_token_id

        image_input_mask = torch.from_numpy(input_ids == self.image_token_id)
        total_img_und_mask = torch.zeros(num_image_patches, dtype=torch.bool)
        image_generate_mask = torch.ones(num_image_patches, dtype=torch.bool)
        image_understand_mask = torch.zeros(num_image_patches, dtype=torch.bool)
        image_gen_seq_lens = [int(image_thw_grids_ori.prod().item())]
        position_ids = self._build_position_ids(input_ids_list, image_thw_grids)
        cond_indices, nocond_indices = create_sample_startend_row_indices(
            input_ids=torch.from_numpy(input_ids),
            img_start_id=self.image_start_id,
            img_end_id=self.image_end_id,
            bidirectional=True,
        )
        return TransfusionTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs["pixel_values"] if flattened_imgs else [],
            image_thw_grids=image_thw_grids_ori,
            image_input_mask=image_input_mask,
            total_img_und_mask=total_img_und_mask,
            image_generate_mask=image_generate_mask,
            image_understand_mask=image_understand_mask,
            image_gen_seq_lens=image_gen_seq_lens,
            image_gen_grid_thw=[image_thw_grids_ori.squeeze(0).clone()],
            videos=[],
            video_thw_grids=[],
            video_input_mask=torch.zeros_like(image_input_mask),
            task_type=sample.task_type,
            text=torch.from_numpy(input_ids),
            target=torch.from_numpy(target),
            total_len=len(input_ids),
            position_ids=position_ids,
            cond_startend_row_indices=cond_indices,
            nocond_startend_row_indices=None if cfg_dropped else nocond_indices,
        )
    def _encode_generation_cot(self, sample, processed_vision) -> TransfusionTaskSample:
        """Encode prompt-refinement generation for Dense Pixel-DiT."""
        data = getattr(sample, "raw_message", None) or {}
        original_prompt = str(sample.text[0] if isinstance(sample.text, list) and sample.text else data.get("original_prompt") or "").strip()
        refined_prompt = str(sample.text[1] if isinstance(sample.text, list) and len(sample.text) > 1 else data.get("refined_prompt") or "").strip()
        language = str(sample.text[2] if isinstance(sample.text, list) and len(sample.text) > 2 else data.get("language") or "").strip().lower()
        thinking = str(sample.text[3] if isinstance(sample.text, list) and len(sample.text) > 3 else data.get("thinking") or "").strip()
        if not original_prompt or not refined_prompt:
            raise SkipSample()
        raw_grid = processed_vision["image_grid_thw"].clone()
        if raw_grid.numel() == 0:
            raise SkipSample()
        grid = raw_grid.clone()
        grid[..., 1] //= 2
        grid[..., 2] //= 2
        num_image_patches = int(grid.prod().item())
        flattened_imgs = processed_vision.get("image_inputs")
        instruction_tokens = self.hf_tokenizer.encode(self._choose_generation_cot_instruction(language), add_special_tokens=False)
        original_tokens = self.hf_tokenizer.encode(original_prompt, add_special_tokens=False)
        think_body = thinking or refined_prompt
        think_tokens = self.hf_tokenizer.encode(f"<think>\n{think_body}\n</think>", add_special_tokens=False)
        if not think_tokens:
            raise SkipSample()
        prefix = (
            [self.im_start_id] + self.user_ids + self.nl_ids + instruction_tokens + original_tokens
            + [self.im_end_id] + self.nl_ids + [self.im_start_id] + self.assistant_ids + self.nl_ids
        )
        input_ids_list = (
            prefix + [self.image_gate_token_id] + think_tokens + [self.recaption_end_token_id]
            + [self.image_start_id, self.time_token_id]
            + [self.image_token_id] * num_image_patches + [self.image_end_id, self.im_end_id]
        )
        input_ids = np.asarray(input_ids_list, dtype=np.int64)
        loss_mask = np.zeros_like(input_ids, dtype=np.float32)
        loss_mask[len(prefix) - 1:len(prefix) + 1 + len(think_tokens) + 1] = 1.0
        target = np.roll(input_ids, -1)
        target[-1] = self.pad_id
        target[loss_mask == 0] = self.pad_id
        image_input_mask = torch.from_numpy(input_ids == self.image_token_id)
        total_img_und_mask = torch.zeros(num_image_patches, dtype=torch.bool)
        image_generate_mask = torch.ones(num_image_patches, dtype=torch.bool)
        image_understand_mask = torch.zeros(num_image_patches, dtype=torch.bool)
        cond_indices, nocond_indices = create_sample_startend_row_indices(
            input_ids=torch.from_numpy(input_ids), img_start_id=self.image_start_id,
            img_end_id=self.image_end_id, bidirectional=True,
        )
        return TransfusionTaskSample(
            __key__=sample.__key__, __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs["pixel_values"] if flattened_imgs else [],
            image_thw_grids=raw_grid, image_input_mask=image_input_mask,
            total_img_und_mask=total_img_und_mask, image_generate_mask=image_generate_mask,
            image_understand_mask=image_understand_mask,
            image_gen_seq_lens=[int(raw_grid.prod().item())],
            image_gen_grid_thw=[raw_grid.squeeze(0).clone()], videos=[], video_thw_grids=[],
            video_input_mask=torch.zeros_like(image_input_mask), task_type=sample.task_type,
            text=torch.from_numpy(input_ids), target=torch.from_numpy(target),
            total_len=len(input_ids), position_ids=self._build_position_ids(input_ids_list, grid),
            cond_startend_row_indices=cond_indices, nocond_startend_row_indices=nocond_indices,
        )
    def _encode_editing_vit(self, sample, processed_vision, cfg_mode: str = "none") -> TransfusionTaskSample:
        """Encode ViT-conditioned image editing for Dense Pixel-DiT."""
        if cfg_mode not in ("none", "text", "full"):
            raise ValueError(f"invalid cfg_mode={cfg_mode!r}")
        grids_raw = processed_vision["image_grid_thw"].clone()
        if grids_raw.ndim == 1:
            grids_raw = grids_raw.unsqueeze(0)
        grids = grids_raw.clone()
        grids[..., 1] //= 2
        grids[..., 2] //= 2
        flattened_imgs = processed_vision.get("image_inputs")
        input_ids_list = [self.im_start_id] + self.user_ids + self.nl_ids
        total_img_und = []
        image_generate = []
        image_understand = []
        image_gen_seq_lens = []
        image_gen_grid_thw = []
        image_block_grids = []
        gen_ranges = []
        image_index = 0
        last_text = None
        for text, mask_value in zip(sample.text, sample.mask):
            if text != "<image>":
                if text:
                    input_ids_list.extend(self.hf_tokenizer.encode(text, add_special_tokens=False))
                last_text = text
                continue
            if image_index >= grids.shape[0]:
                raise SkipSample()
            grid = grids[image_index]
            raw_grid = grids_raw[image_index]
            tokens = int((grid[0] * grid[1] * grid[2]).item())
            is_gen = int(mask_value) == 1
            if is_gen:
                input_ids_list.extend([self.image_start_id, self.time_token_id])
                first_patch = len(input_ids_list)
                input_ids_list.extend([self.image_token_id] * tokens)
                input_ids_list.append(self.image_end_id)
                gen_ranges.append((first_patch, first_patch + tokens))
                total_img_und.extend([False] * tokens)
                image_generate.extend([True] * tokens)
                image_understand.extend([False] * tokens)
                image_gen_seq_lens.append(int((raw_grid[0] * raw_grid[1] * raw_grid[2]).item()))
                image_gen_grid_thw.append(raw_grid.clone())
            else:
                input_ids_list.extend([self.image_start_id])
                input_ids_list.extend([self.image_token_id] * tokens)
                input_ids_list.append(self.image_end_id)
                total_img_und.extend([True] * tokens)
                image_generate.extend([False] * tokens)
                image_understand.extend([True] * tokens)
            image_block_grids.append(grid.clone())
            image_index += 1
        input_ids_list.extend([self.im_end_id])
        input_ids = np.asarray(input_ids_list, dtype=np.int64)
        target = np.roll(input_ids, -1)
        target[-1] = self.pad_id
        image_input_mask = torch.from_numpy(input_ids == self.image_token_id)
        total_img_und_mask = torch.tensor(total_img_und, dtype=torch.bool)
        image_generate_mask = torch.tensor(image_generate, dtype=torch.bool)
        image_understand_mask = torch.tensor(image_understand, dtype=torch.bool)
        cond_indices = self._build_editing_startend_row_indices(
            torch.from_numpy(input_ids), gen_ranges
        )
        position_ids = self._build_position_ids(input_ids_list, image_block_grids)
        return TransfusionTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs["pixel_values"] if flattened_imgs else [],
            image_thw_grids=grids_raw,
            image_input_mask=image_input_mask,
            total_img_und_mask=total_img_und_mask,
            image_generate_mask=image_generate_mask,
            image_understand_mask=image_understand_mask,
            image_gen_seq_lens=image_gen_seq_lens,
            image_gen_grid_thw=image_gen_grid_thw or None,
            videos=[], video_thw_grids=[], video_input_mask=torch.zeros_like(image_input_mask),
            task_type=sample.task_type,
            text=torch.from_numpy(input_ids), target=torch.from_numpy(target),
            total_len=len(input_ids), position_ids=position_ids,
            cond_startend_row_indices=cond_indices,
            nocond_startend_row_indices=None,
        )

    def _build_editing_startend_row_indices(
        self,
        input_ids,
        gen_img_ranges,
        block_bidirectional_flags=None,
    ):
        """
        为 editing 任务构建 FlashMask startend_row_indices。
        对齐 create_sample_startend_row_indices 的初始化方式:
          LTS = max(L-1, 0)  (下三角全可见)
          UTE = col_idx       (标准 causal)

        规则:
        - <vis_start>...<vis_end> 块内部默认按 block_bidirectional_flags 逐块决定:
            * True  → bidirectional (reference / feedback / noisy generation)
            * False → causal        (ViT ref / ViT feedback)
          block_bidirectional_flags 按 input_ids 中 <vis_start> 出现序对齐;
          若为 None, 兼容旧行为: 所有块 bidirectional。
        - CFG 通过丢弃 text token 实现，不需要 nocond

        Args:
            input_ids: [L] token ids
            gen_img_ranges: List[(start_pos, end_pos)] 每张生成图的 noise token 范围
            block_bidirectional_flags: Optional[List[bool]] 按块出现序的块内方向标志

        Returns:
            cond_startend_row_indices: [1, L, 2]  (LTS, UTE)
        """
        L = len(input_ids)
        device = input_ids.device
        col_idx = torch.arange(L, device=device, dtype=torch.int32)

        # === 1. 初始化: 对齐 create_sample_startend_row_indices ===
        cond_LTS = torch.full((L,), max(L - 1, 0), device=device, dtype=torch.int32)
        cond_UTE = col_idx.clone()

        # === 2. 按 flags 逐块决定块内方向 ===
        # flags[k] == True  → 覆盖 UTE = first_patch, 块内 bidirectional
        # flags[k] == False → 保留 UTE = col_idx,    块内 causal
        starts = (input_ids == self.image_start_id).nonzero(as_tuple=True)[0]
        ends = (input_ids == self.image_end_id).nonzero(as_tuple=True)[0]
        min_len = min(len(starts), len(ends))

        if block_bidirectional_flags is not None:
            assert len(block_bidirectional_flags) == min_len, (
                f"block_bidirectional_flags 长度 {len(block_bidirectional_flags)} 与"
                f" <vis_start>/<vis_end> 匹配块数 {min_len} 不一致"
            )

        for k in range(min_len):
            s, e = starts[k].item(), ends[k].item()
            if s + 1 < L and input_ids[s + 1].item() == self.time_token_id:
                first_patch = s + 2  # <vis_start> <time> [patches...]
            else:
                first_patch = s + 1  # <vis_start> [patches...]
            bidir = True if block_bidirectional_flags is None else bool(block_bidirectional_flags[k])
            if not bidir:
                continue  # 保留默认 UTE=col_idx: 块内 causal
            patch_cols_mask = (col_idx >= first_patch) & (col_idx < e)
            if patch_cols_mask.any():
                cond_UTE[patch_cols_mask] = first_patch

        # === 3. noise gen block 自闭合: 块外任何 query 都看不到 noise 块 ===
        # FlashMask 语义 (causal=False, dim=2):
        #   LTS[k]: key k 的下三角从 LTS[k] 行起被 mask → query q >= LTS[k] 且 q > k 看不到 key k
        #   UTE[k]: key k 的上三角在 UTE[k] 行以上被 mask → query q < UTE[k] 且 q < k 看不到 key k
        #
        # noise 是"正在去噪的隐变量", 后文应当看它的 clean feedback、而非 noise 本身。
        # 因此每个 noise gen block 在自己的 <vis_end> 处就【截止】: 只有块内 token 能看到它
        # (块内 bidirectional 由第 2 步保证), 块的 <vis_end> 之后的任何 query 都不可见。
        # 实现: 对该 noise block 的【全部 key 列】(vis_start, time, patches, vis_end)
        #       把 LTS 收紧到 vis_end + 1, 使行 >= vis_end+1 的 query 都看不到本块。
        # 这样:
        #   - noise patch 之间完全互不可见 (后续 gen block 在更后面, 必然 >= vis_end+1);
        #   - 该图自己的 clean feedback (紧跟其后) 也看不到 noise, 只提供 clean 条件;
        #   - 仅最后一张 gen 无需此处理 (其后无 feedback / 其它 gen)，但统一处理也安全。
        for gen_start, gen_end in gen_img_ranges:
            vis_start_pos = gen_start - 2   # <vis_start> 位置 (gen_start = first_patch)
            vis_end_pos = gen_end            # <vis_end> 位置
            cutoff = vis_end_pos + 1         # 块外第一行起即不可见
            for pos in range(vis_start_pos, vis_end_pos + 1):
                if 0 <= pos < L:
                    cond_LTS[pos] = min(cond_LTS[pos].item(), cutoff)

        # === 4. 封装 ===
        cond_startend_row_indices = torch.stack([cond_LTS, cond_UTE], dim=-1).unsqueeze(0)

        return cond_startend_row_indices

    def _encode_text(self, sample):
        """
        专门处理【纯文本预训练】任务的策略函数 (生成器版本)。
        纯文本预训练数据不需要加 template，直接编码原始文本。

        数据格式 (来自 TextCooker 流式读取):
        - sample.text: List[str], 文本片段列表 (通常是单个文本)
        - sample.mask: List[int], 与 text 对应的训练掩码 (1=训练, 0=跳过)
        - sample.imgs: None
        - sample.videos: None

        按 packing_seq_length 切分长文本，每次 yield 一个 TransfusionTaskSample。
        """
        # 1. 处理文本片段 - 流式处理，逐片段编码
        text_fragments = sample.text
        mask = sample.mask if sample.mask else [1] * len(text_fragments)

        # 2. 构建 token 序列 - 纯文本预训练不加 template，直接编码
        input_ids_list = []
        loss_mask_list = []

        for i, fragment in enumerate(text_fragments):
            fragment_mask = mask[i] if i < len(mask) else 1

            if fragment and isinstance(fragment, str) and fragment.strip():
                # 编码文本片段
                for j in range(0, len(fragment), self.hf_tokenizer.max_len_single_sentence):
                    chunk = fragment[j : min(j + self.hf_tokenizer.max_len_single_sentence, len(fragment))]
                    text_tokens = self.hf_tokenizer.encode(chunk, add_special_tokens=False)

                    # 添加文本内容，根据 mask 决定是否参与训练
                    input_ids_list.extend(text_tokens)
                    loss_mask_list.extend([fragment_mask] * len(text_tokens))

        # 3. 为完整 sample 构造一次纯文自回归序列：
        #   input  = [EOS, body...]
        #   target = [body..., EOS]
        # EOS 表示 sample 边界，不是后续切分 chunk 的边界。
        body_ids = np.array(input_ids_list, dtype=np.int64)
        body_loss_mask = np.array(loss_mask_list, dtype=np.float32)

        if len(body_ids) == 0:
            return  # 空文本，不产出任何样本

        input_ids = np.concatenate((
            np.array([self.im_end_id], dtype=np.int64),
            body_ids,
        ))
        target = np.concatenate((
            body_ids,
            np.array([self.im_end_id], dtype=np.int64),
        ))
        prediction_mask = np.concatenate((
            body_loss_mask,
            np.array([1.0], dtype=np.float32),
        ))
        target[prediction_mask == 0] = self.pad_id

        # 4. 同步切分完整 sample 的 input/target。只有 chunk0 以 EOS 开头；
        # 后续 chunk 是同一 sample 的正文延续，不额外插入 EOS 或结束目标。
        max_seq_len = self.packing_seq_length
        if max_seq_len < 1:
            raise ValueError("TEXT packing_seq_length must be positive")
        total_len = len(input_ids)
        chunk_idx = 0

        for start in range(0, total_len, max_seq_len):
            end = min(start + max_seq_len, total_len)
            chunk_input_ids = input_ids[start:end]
            chunk_target = target[start:end]

            chunk_len = len(chunk_input_ids)
            # 保留真实长度供 packing 使用；固定长度 padding 由 batch() 统一完成。
            seq_len = chunk_len

            # 6. 纯文本任务：没有图片和视频
            image_input_mask = torch.zeros(seq_len, dtype=torch.bool)

            # 7. 构建 3D M-RoPE position_ids (纯文本：3个通道值相同)
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, -1).clone()

            # 8. 构建 FlashMask startend_row_indices (纯文本: 标准 causal attention)
            cond_indices, nocond_indices = create_sample_startend_row_indices(
                input_ids=torch.from_numpy(chunk_input_ids),
                img_start_id=self.image_start_id,
                img_end_id=self.image_end_id,
                bidirectional=False,  # 纯文本使用标准 causal attention
            )

            yield TransfusionTaskSample(
                __key__=f"{sample.__key__}_chunk{chunk_idx}",
                __subflavors__=sample.__subflavors__,
                imgs=[],
                image_thw_grids=[],
                image_input_mask=image_input_mask,
                total_img_und_mask=torch.empty(0, dtype=torch.bool),
                image_generate_mask=torch.empty(0, dtype=torch.bool),
                image_understand_mask=torch.empty(0, dtype=torch.bool),
                image_gen_seq_lens=[],
                videos=[],
                video_thw_grids=[],
                video_input_mask=torch.zeros(seq_len, dtype=torch.bool),
                task_type=sample.task_type,
                text=torch.from_numpy(chunk_input_ids),
                target=torch.from_numpy(chunk_target),
                total_len=chunk_len,  # 实际有效长度（不含 padding）
                position_ids=position_ids,
                cond_startend_row_indices=cond_indices,
                nocond_startend_row_indices=nocond_indices,
            )
            chunk_idx += 1

    def _encode_message(self, sample):
        """
        专门处理【message 文本预训练】任务的策略函数。

        数据格式 (来自 MessageCooker):
        - sample.text: List[str], 文本片段列表
        - sample.mask: List[int], 对应每个片段的 mask，0 表示 user（不训练），1 表示 assistant（训练）
        - sample.imgs: None
        - sample.videos: None

        与 _encode_text 类似，按 packing_seq_length 切分长文本，每次 yield 一个 TransfusionTaskSample。

        对齐 Qwen3-VL chat template:
        <|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>
        """
        # 1. 处理文本片段
        text_fragments = list(sample.text)
        mask = list(sample.mask) if sample.mask else [0] * len(text_fragments)

        # 2. 构建 token 序列 - 对齐 Qwen3-VL chat template
        input_ids_list = []
        loss_mask_list = []

        # 基于 mask 判断是否需要训练：mask=0 表示 user（不训练），mask=1 表示 assistant（训练）
        # 当 mask 变化时，需要插入相应的 chat template token
        prev_mask = None

        for i, fragment in enumerate(text_fragments):
            current_mask = mask[i] if i < len(mask) else 1

            # 检测 mask 变化，插入角色切换 token
            if prev_mask is None:
                # 第一个 fragment，根据 mask 决定开头
                if current_mask == 0:
                    # user 开头: <|im_start|>user\n
                    role_prefix = [self.im_start_id] + self.user_ids + self.nl_ids
                else:
                    # assistant 开头: <|im_start|>assistant\n
                    role_prefix = [self.im_start_id] + self.assistant_ids + self.nl_ids
                input_ids_list.extend(role_prefix)
                loss_mask_list.extend([0] * len(role_prefix))
            elif prev_mask != current_mask:
                # mask 发生变化，需要切换角色
                if current_mask == 1:
                    # user(0) → assistant(1): <|im_end|>\n<|im_start|>assistant\n
                    role_switch = [self.im_end_id] + self.nl_ids +\
                          [self.im_start_id] + self.assistant_ids + self.nl_ids
                else:
                    # assistant(1) → user(0): <|im_end|>\n<|im_start|>user\n
                    role_switch = [self.im_end_id] + self.nl_ids +\
                          [self.im_start_id] + self.user_ids + self.nl_ids
                input_ids_list.extend(role_switch)
                role_switch_mask = [1 if prev_mask == 1 else 0] + [0] * (len(role_switch) - 1)
                loss_mask_list.extend(role_switch_mask)

            if fragment and isinstance(fragment, str):
                # 处理普通文本，根据 mask 决定是否参与训练
                text_tokens = self.hf_tokenizer.encode(fragment, add_special_tokens=False)
                input_ids_list.extend(text_tokens)
                # mask=0 表示 user（不训练），mask=1 表示 assistant（训练）
                loss_mask_list.extend([current_mask] * len(text_tokens))

            prev_mask = current_mask

        # 添加结尾: <|im_end|>
        input_ids_list.append(self.im_end_id)
        # 如果最后是 assistant，结束标记参与训练；如果是 user，不参与
        loss_mask_list.append(1 if prev_mask == 1 else 0)

        if getattr(self, "use_v5_route_tokens", False):
            self._insert_route_tokens_after_assistant_headers(
                input_ids_list,
                self.text_gate_token_id,
                token_ownership=loss_mask_list,
            )

        # 4. 转换为 numpy/torch
        input_ids = np.array(input_ids_list, dtype=np.int64)
        loss_mask = np.array(loss_mask_list, dtype=np.float32)

        total_len = len(input_ids)
        if total_len == 0:
            return  # 空文本，不产出任何样本

        # 5. 按 packing_seq_length 切分，流式 yield
        max_seq_len = self.packing_seq_length
        pad_id = self.pad_id
        chunk_idx = 0

        for start in range(0, total_len, max_seq_len):
            end = min(start + max_seq_len, total_len)
            chunk_input_ids = input_ids[start:end]
            chunk_loss_mask = loss_mask[start:end]

            # 构建 Labels (用于训练): 下一个 token 预测
            chunk_prediction_mask = self._align_loss_mask_to_next_token(
                chunk_loss_mask
            )
            chunk_target = chunk_input_ids.copy()
            chunk_target = np.roll(chunk_target, shift=-1)
            chunk_target[-1] = pad_id

            # token ownership mask 需要左移到预测该 token 的位置。
            chunk_target[chunk_prediction_mask == 0] = pad_id

            chunk_len = len(chunk_input_ids)
            # 保留真实长度供 packing 使用；固定长度 padding 由 batch() 统一完成。
            seq_len = chunk_len

            # 6. 纯文本任务：没有图片和视频
            image_input_mask = torch.zeros(seq_len, dtype=torch.bool)

            # 7. 构建 3D M-RoPE position_ids (纯文本：3个通道值相同)
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, -1).clone()

            # 8. 构建 FlashMask startend_row_indices (message: 标准 causal attention)
            cond_indices, nocond_indices = create_sample_startend_row_indices(
                input_ids=torch.from_numpy(chunk_input_ids),
                img_start_id=self.image_start_id,
                img_end_id=self.image_end_id,
                bidirectional=False,  # message 使用标准 causal attention
            )

            yield TransfusionTaskSample(
                __key__=f"{sample.__key__}_chunk{chunk_idx}",
                __subflavors__=sample.__subflavors__,
                imgs=[],
                image_thw_grids=[],
                image_input_mask=image_input_mask,
                total_img_und_mask=torch.empty(0, dtype=torch.bool),
                image_generate_mask=torch.empty(0, dtype=torch.bool),
                image_understand_mask=torch.empty(0, dtype=torch.bool),
                image_gen_seq_lens=[],
                videos=[],
                video_thw_grids=[],
                video_input_mask=torch.zeros(seq_len, dtype=torch.bool),
                task_type=sample.task_type,
                text=torch.from_numpy(chunk_input_ids),
                target=torch.from_numpy(chunk_target),
                total_len=chunk_len,  # 实际有效长度（不含 padding）
                position_ids=position_ids,
                cond_startend_row_indices=cond_indices,
                nocond_startend_row_indices=nocond_indices,
            )
            chunk_idx += 1

    def _compute_sft_loss_mask(
        self,
        input_ids: list,
        all_messages: list,
        prompt_messages: list,
        labels: list,
        *,
        has_tools: bool = False,
        render_nonfirst_system: bool = True,
    ) -> list:
        """基于 <|im_end|> token 切分消息并计算 SFT loss mask。

        策略（移植自 qwen_text 数据编码逻辑）
          - candidates 中 assistant 强制学习
          - prompt 中 assistant 默认学习；labels[i]=0 关闭
          - prompt 中 user/system 默认不学习；labels[i]=1 打开
          - assistant 段：跳过 `<|im_start|>{role}\n` header（找首个 `<|im_start|>`
            之后的首个 `\n` token，content_start = 该 `\n` + 1），从 content_start
            到含 `<|im_end|>` 都学习
          - user/system 段（学习时）：全部学，但不含 `<|im_end|>`

        Qwen3 模板在 tools 非空时会额外渲染 system 工具说明，并把连续 tool
        message 合并成一个 ChatML 段。loss segment 必须复现该结构，不能直接与
        all_messages 按下标对齐。
        """
        loss_mask = [0.0] * len(input_ids)
        eos_id = self.im_end_id

        im_end_positions = [i for i, t in enumerate(input_ids) if t == eos_id]
        if not im_end_positions:
            return loss_mask

        prompt_count = len(prompt_messages)

        def should_learn(message_indices, role):
            """Determine whether a rendered message segment contributes to loss."""
            if not message_indices:
                return False

            if role == "assistant":
                message_idx = message_indices[0]
                if message_idx >= prompt_count:
                    return True
                return not (
                    labels
                    and message_idx < len(labels)
                    and labels[message_idx] == 0
                )

            return any(
                message_idx < prompt_count
                and labels
                and message_idx < len(labels)
                and labels[message_idx] == 1
                for message_idx in message_indices
            )

        # Each item is (rendered role, should_learn, skip_opening_think). The role
        # for a grouped tool response remains "tool" even though the template uses
        # a user ChatML header.
        rendered_segments = []
        message_idx = 0
        first_role = all_messages[0].get("role") if all_messages else None
        if first_role == "system":
            rendered_segments.append(("system", should_learn([0], "system"), False))
            message_idx = 1
        elif has_tools:
            # The Qwen3 templates synthesize a system message containing tool schemas.
            rendered_segments.append(("system", False, False))

        while message_idx < len(all_messages):
            message = all_messages[message_idx]
            role = message.get("role")

            if role == "tool":
                tool_indices = []
                while (
                    message_idx < len(all_messages)
                    and all_messages[message_idx].get("role") == "tool"
                ):
                    tool_indices.append(message_idx)
                    message_idx += 1
                rendered_segments.append(
                    ("tool", should_learn(tool_indices, "tool"), False)
                )
                continue

            if role in ("user", "assistant") or (
                role == "system" and render_nonfirst_system
            ):
                rendered_segments.append(
                    (
                        role,
                        should_learn([message_idx], role),
                        role == "assistant"
                        and isinstance(message.get("content"), str)
                        and message["content"].lstrip().startswith("<think>"),
                    )
                )
            message_idx += 1

        if len(im_end_positions) != len(rendered_segments):
            logger.warning(
                f"[SFT] <|im_end|> count {len(im_end_positions)} != rendered segment "
                f"count {len(rendered_segments)}; skip malformed template alignment"
            )
            return loss_mask

        for seg_idx, im_end_pos in enumerate(im_end_positions):
            start_pos = 0 if seg_idx == 0 else im_end_positions[seg_idx - 1] + 1
            role, learn_segment, skip_opening_think = rendered_segments[seg_idx]

            if learn_segment and role == "assistant":
                # 在段内先定位 <|im_start|>，再在其后找首个 \n —— 这样能精确跳过
                # `<|im_start|>{role}\n` header（包括 role 本身被分成多 token 的情况），
                # 而不会被消息间分隔的换行误导
                content_start = start_pos
                if self.nl_single_id is not None:
                    ims_pos = -1
                    for i in range(start_pos, im_end_pos):
                        if input_ids[i] == self.im_start_id:
                            ims_pos = i
                            break
                    if ims_pos >= 0:
                        for i in range(ims_pos + 1, im_end_pos):
                            if input_ids[i] == self.nl_single_id:
                                content_start = i + 1
                                break
                for i in range(content_start, im_end_pos + 1):
                    if i < len(loss_mask):
                        loss_mask[i] = 1.0
                if skip_opening_think:
                    think_tokens = self.hf_tokenizer.encode(
                        "<think>", add_special_tokens=False
                    )
                    think_start = content_start
                    if (
                        think_start < im_end_pos
                        and input_ids[think_start]
                        in (
                            getattr(self, "text_gate_token_id", -1),
                            getattr(self, "image_gate_token_id", -1),
                        )
                    ):
                        think_start += 1
                    think_end = think_start + len(think_tokens)
                    if (
                        think_tokens
                        and think_end <= im_end_pos
                        and input_ids[think_start:think_end] == think_tokens
                    ):
                        loss_mask[think_start:think_end] = [0.0] * len(think_tokens)
            elif learn_segment:
                for i in range(start_pos, im_end_pos):
                    loss_mask[i] = 1.0

        return loss_mask

    def _mask_opening_think_in_fragment(
        self, fragment: str, text_tokens: list, token_loss_mask: list
    ) -> None:
        """Exclude an answer's opening ``<think>`` marker from CE targets."""
        marker_pos = fragment.find("<think>")
        if marker_pos < 0 or fragment[:marker_pos].strip():
            return

        prefix_tokens = self.hf_tokenizer.encode(
            fragment[:marker_pos], add_special_tokens=False
        )
        think_tokens = self.hf_tokenizer.encode(
            "<think>", add_special_tokens=False
        )
        marker_start = len(prefix_tokens)
        marker_end = marker_start + len(think_tokens)
        if (
            think_tokens
            and marker_end <= len(text_tokens)
            and text_tokens[marker_start:marker_end] == think_tokens
        ):
            token_loss_mask[marker_start:marker_end] = [0] * len(think_tokens)

    @staticmethod
    def _align_loss_mask_to_next_token(token_loss_mask: list) -> np.ndarray:
        """Map token ownership to positions predicting those tokens."""
        token_loss_mask = np.asarray(token_loss_mask, dtype=np.float32)
        prediction_loss_mask = np.zeros_like(token_loss_mask)
        if token_loss_mask.size > 1:
            prediction_loss_mask[:-1] = token_loss_mask[1:]
        return prediction_loss_mask

    def _insert_route_tokens_after_assistant_headers(
        self,
        input_ids: list,
        route_token_id: int,
        *,
        token_ownership: list = None,
    ) -> list:
        """Insert a route token at the start of each assistant response.

        When ``token_ownership`` is provided, the route token inherits whether
        that assistant segment is supervised. The ownership mask is shifted
        later, so the token immediately before the route token predicts it.
        """
        if token_ownership is not None and len(token_ownership) != len(input_ids):
            raise ValueError("route-token ownership must align with input_ids")

        header = [self.im_start_id] + self.assistant_ids + self.nl_ids
        if not header:
            return []

        route_positions = []
        index = 0
        while index <= len(input_ids) - len(header):
            if input_ids[index:index + len(header)] != header:
                index += 1
                continue

            content_start = index + len(header)
            segment_end = content_start
            while segment_end < len(input_ids) and input_ids[segment_end] != self.im_end_id:
                segment_end += 1
            learn_route = True
            if token_ownership is not None:
                learn_route = any(
                    float(value) > 0
                    for value in token_ownership[content_start:segment_end]
                )

            if content_start < len(input_ids) and input_ids[content_start] == route_token_id:
                if token_ownership is not None and learn_route:
                    token_ownership[content_start] = 1
                route_positions.append(content_start)
                index = content_start + 1
                continue

            input_ids.insert(content_start, route_token_id)
            if token_ownership is not None:
                token_ownership.insert(content_start, 1 if learn_route else 0)
            route_positions.append(content_start)
            index = content_start + 1

        return route_positions

    @staticmethod
    def _normalize_sft_messages(
        prompt_messages,
        candidate_messages,
        *,
        keep_image_url: bool,
        enable_think: bool = True,
        strip_image_placeholders: bool = False,
    ):
        """SFT 消息预处理（理解与纯文共用，唯一差别在 user/system 是否保留 image_url）。

        - assistant: 从 content（list 或 str）抽取所有 text，若 enable_think=True
          且有 reasoning_content 则拼为 `<think>\n{rc}\n</think>\n{text}`；
          enable_think=False 时丢弃 reasoning_content
        - tool:   content list → "\n"-join 的纯文本 string
        - user/system:
            keep_image_url=True  → 保留 content list（VL 模板自行处理 image_url）
            keep_image_url=False → 拍扁为 "\n"-join 的纯文本 string
        """
        all_messages = []
        for msg in list(prompt_messages) + list(candidate_messages):
            new_msg = dict(msg)
            role = new_msg.get("role")
            content = new_msg.get("content")
            if strip_image_placeholders:
                if isinstance(content, str):
                    content = strip_textual_image_placeholders(content)
                elif isinstance(content, list):
                    cleaned_content = []
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            part = dict(part)
                            part["text"] = strip_textual_image_placeholders(part["text"])
                        cleaned_content.append(part)
                    content = cleaned_content
                new_msg["content"] = content
            if role == "assistant":
                if isinstance(content, list):
                    text_str = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and "text" in part
                    )
                else:
                    text_str = content if isinstance(content, str) else ""
                rc = new_msg.get("reasoning_content") or ""
                if strip_image_placeholders and isinstance(rc, str):
                    rc = strip_textual_image_placeholders(rc)
                if rc and enable_think:
                    assistant_text = f"<think>\n{rc}\n</think>\n{text_str}"
                else:
                    assistant_text = text_str
                if not _has_balanced_think_tags(assistant_text):
                    raise SkipSample()
                new_msg["content"] = assistant_text
                new_msg.pop("reasoning_content", None)
            elif role == "tool":
                if isinstance(content, list):
                    new_msg["content"] = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and "text" in part
                    )
            else:
                # user / system
                if isinstance(content, list) and not keep_image_url:
                    new_msg["content"] = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                # keep_image_url=True 时保持 list，让 VL 模板处理 image_url
            all_messages.append(new_msg)
        return all_messages

    def _debug_print_sft_sample(self, task_type, key, full_text, input_ids, target, loss_mask):
        """以单块文本输出 SFT 样本的渲染、分段 mask、token→target 对齐表。

        - 通过 SFT_DEBUG_PRINT 控制每种 task_type 的打印份数（默认 2）。
        - 通过 SFT_DEBUG_FILE 指定输出文件；为空则写到 stderr，避免被 logger 前缀污染。
        - 仅在 rank 0 打印，避免多卡交错。
        """
        rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        try:
            if int(rank_env) != 0:
                return
        except ValueError:
            return

        try:
            import torch.utils.data as _tud
            wi = _tud.get_worker_info()
            if wi is not None and wi.id != 0:
                return
        except Exception:
            pass

        if self._debug_print_count[task_type] >= self._debug_print_limit:
            return
        self._debug_print_count[task_type] += 1

        n = len(input_ids)
        n_loss = int(loss_mask.sum())
        ratio = n_loss / max(n, 1) * 100

        # 1) 切段
        segments = []
        cur_mask = int(loss_mask[0])
        start = 0
        for i in range(1, n):
            m = int(loss_mask[i])
            if m != cur_mask:
                segments.append((start, i, cur_mask))
                cur_mask = m
                start = i
        segments.append((start, n, cur_mask))

        # 2) token-level 对齐表（前 30 个 predict 位置）
        predict_pos = (loss_mask > 0).nonzero()[0][:30]

        def _esc(s, w):
            """转义换行/制表，截断到宽度 w，便于单行展示。"""
            s = s.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            return s if len(s) <= w else s[: w - 1] + "…"

        align_rows = []
        if len(predict_pos) > 0:
            for p in predict_pos:
                p = int(p)
                in_id = int(input_ids[p])
                tg_id = int(target[p])
                in_tok = self.hf_tokenizer.decode([in_id], skip_special_tokens=False)
                tg_tok = self.hf_tokenizer.decode([tg_id], skip_special_tokens=False)
                align_rows.append(
                    f"  {p:>6d}  {in_id:>7d} {_esc(in_tok, 18):<20} →  "
                    f"{tg_id:>7d} {_esc(tg_tok, 18):<20}"
                )

        # 3) 段表
        seg_lines = []
        for s, e, m in segments:
            tag = "PRED" if m else "CTX "
            star = "*" if m else " "
            seg_text = self.hf_tokenizer.decode(
                input_ids[s:e].tolist(), skip_special_tokens=False
            )
            seg_lines.append(
                f"  [{tag}{star}] [{s:>6d}:{e:>6d}] ({e - s:>5d}t) {_esc(seg_text, 140)}"
            )

        # 4) 拼成单块
        bar = "═" * 88
        sub = "─" * 88
        rendered = full_text[:1500]
        if len(full_text) > 1500:
            rendered += "\n…[truncated]"

        block = [
            "",
            bar,
            f" SFT  task={task_type}  len={n}  loss={n_loss}/{n} ({ratio:.1f}%)",
            f" key={key}",
            bar,
            f"{sub} rendered_text (chars 0:{min(len(full_text), 1500)}) {sub[:0]}",
            rendered,
            f"{sub} segments  (CTX=loss_mask 0, PRED=loss_mask 1) {sub[:0]}",
            *seg_lines,
            f"{sub} align  pos | input_id input_tok → target_id target_tok {sub[:0]}",
            *align_rows,
            bar,
            "",
        ]
        out = "\n".join(block)

        path = os.environ.get("SFT_DEBUG_FILE", "").strip()
        if path:
            try:
                with open(path, "a", encoding="utf-8") as fp:
                    fp.write(out)
                return
            except Exception as ex:
                sys.stderr.write(f"[SFT-DEBUG] write {path} failed: {ex}\n")
        sys.stderr.write(out)
        sys.stderr.flush()

    def _encode_sft_text(self, sample):
        """处理 task_type="sft_text" 样本：
        从 sample.raw_message 渲染 jinja chat template，tokenize，
        按 <|im_end|> 切段算 loss mask；超长直接丢弃（不切 chunk）。
        """
        # 优先用 VL 模板（与 Qwen3-VL backbone 预训练格式一致）；纯文本 backbone
        # （vl_chat_template 为 None）回退到 sft_chat_template。
        chat_template = self.vl_chat_template or self.sft_chat_template
        if chat_template is None:
            logger.warning("[SFT] chat template 未加载，跳过 sft_text 样本")
            raise SkipSample()

        data = getattr(sample, "raw_message", None)
        if not isinstance(data, dict):
            logger.warning(f"[SFT] raw_message 非 dict, key={sample.__key__}")
            raise SkipSample()

        prompt_messages = list(data.get("prompt", []))
        candidates = data.get("candidates", [])
        if not prompt_messages or not candidates or not candidates[0]:
            raise SkipSample()

        labels = list(data.get("labels", []))
        if labels and len(labels) != len(prompt_messages):
            logger.warning(
                f"[SFT] labels 长度({len(labels)}) != prompt 长度({len(prompt_messages)}), 忽略 labels, key={sample.__key__}"
            )
            labels = []

        all_messages = self._normalize_sft_messages(
            prompt_messages, candidates[0],
            keep_image_url=False, enable_think=self.enable_think
        )

        temp_data = dict(data)
        temp_data["messages"] = all_messages
        temp_data["add_generation_prompt"] = False

        try:
            full_text = chat_template.render(temp_data)
        except Exception as e:
            logger.warning(f"[SFT] jinja render 失败: {e}, key={sample.__key__}")
            raise SkipSample()

        full_input_ids = self.hf_tokenizer.encode(full_text, add_special_tokens=False)

        if self.use_v5_route_tokens:
            self._insert_route_tokens_after_assistant_headers(
                full_input_ids,
                self.text_gate_token_id,
            )

        # 超长直接丢弃
        if len(full_input_ids) > self.packing_seq_length:
            raise SkipSample()

        loss_mask_list = self._compute_sft_loss_mask(
            full_input_ids,
            all_messages,
            prompt_messages,
            labels,
            has_tools=bool(data.get("tools")),
            render_nonfirst_system=True,
        )

        if not any(m > 0 for m in loss_mask_list):
            raise SkipSample()

        input_ids = np.array(full_input_ids, dtype=np.int64)
        loss_mask = self._align_loss_mask_to_next_token(loss_mask_list)
        total_len = len(input_ids)
        if total_len == 0:
            return

        pad_id = self.pad_id

        # Labels: shifted next-token; loss_mask=0 → pad_id（batch 阶段会转为 -100）
        target = input_ids.copy()
        target = np.roll(target, shift=-1)
        target[-1] = pad_id
        target[loss_mask == 0] = pad_id

        # self._debug_print_sft_sample(
        #     "sft_text", sample.__key__, full_text,
        #     input_ids, target, loss_mask,
        # )

        # 不做 chunk padding；batch 阶段会 pad 到 packing_seq_length
        seq_len = total_len
        image_input_mask = torch.zeros(seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, -1).clone()
        cond_indices, nocond_indices = create_sample_startend_row_indices(
            input_ids=torch.from_numpy(input_ids),
            img_start_id=self.image_start_id,
            img_end_id=self.image_end_id,
            bidirectional=False,
        )

        yield TransfusionTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=[],
            image_thw_grids=[],
            image_input_mask=image_input_mask,
            total_img_und_mask=torch.empty(0, dtype=torch.bool),
            image_generate_mask=torch.empty(0, dtype=torch.bool),
            image_understand_mask=torch.empty(0, dtype=torch.bool),
            image_gen_seq_lens=[],
            videos=[],
            video_thw_grids=[],
            video_input_mask=torch.zeros(seq_len, dtype=torch.bool),
            task_type=sample.task_type,
            text=torch.from_numpy(input_ids),
            target=torch.from_numpy(target),
            total_len=total_len,
            position_ids=position_ids,
            cond_startend_row_indices=cond_indices,
            nocond_startend_row_indices=nocond_indices,
        )

    def _encode_sft_image_und(self, sample, processed_vision):
        """处理 task_type="sft_image_und" 样本：
        从 sample.raw_message 渲染 Qwen3-VL chat template，tokenize，
        把每个 <|image_pad|> 扩展为 N 个 (N = 该图 merged token 数)，
        按 <|im_end|> 切段计算 loss mask；超长直接丢弃。
        """
        if self.vl_chat_template is None:
            logger.warning("[VL-SFT] Qwen3-VL chat template 未加载，跳过 sft_image_und 样本")
            raise SkipSample()

        data = getattr(sample, "raw_message", None)
        if not isinstance(data, dict):
            logger.warning(f"[VL-SFT] raw_message 非 dict, key={sample.__key__}")
            raise SkipSample()

        prompt_messages = list(data.get("prompt", []))
        candidates = data.get("candidates", [])
        if not prompt_messages or not candidates or not candidates[0]:
            raise SkipSample()

        labels = list(data.get("labels", []))
        if labels and len(labels) != len(prompt_messages):
            logger.warning(
                f"[VL-SFT] labels 长度({len(labels)}) != prompt 长度({len(prompt_messages)}), "
                f"忽略 labels, key={sample.__key__}"
            )
            labels = []

        # ---- 视觉特征 ----
        image_inputs = (processed_vision or {}).get("image_inputs") or {}
        image_grid_thw = (processed_vision or {}).get("image_grid_thw")
        image_grid_thw_ori = image_grid_thw.clone() if image_grid_thw is not None else None
        if image_grid_thw is not None:
            image_grid_thw[..., 1] = image_grid_thw[..., 1] // self.merge_size
            image_grid_thw[..., 2] = image_grid_thw[..., 2] // self.merge_size
        pixel_values = image_inputs.get("pixel_values") if image_inputs else None

        # 每张图 merge 后 token 数（与 _encode_understanding 中 image_patch_counts 一致）
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, np.ndarray):
                thw_arr = image_grid_thw if image_grid_thw.ndim == 2 else image_grid_thw[None, :]
                image_patch_counts = [int(np.prod(thw_arr[i])) for i in range(thw_arr.shape[0])]
            elif isinstance(image_grid_thw, torch.Tensor):
                thw_arr = image_grid_thw if image_grid_thw.ndim == 2 else image_grid_thw.unsqueeze(0)
                image_patch_counts = [int(thw_arr[i].prod().item()) for i in range(thw_arr.shape[0])]
            else:
                image_patch_counts = []
        else:
            image_patch_counts = []

        # ---- 预处理 messages（与 sft_text 共用 helper，唯一差别：保留 user/system 中
        # 的 image_url，让 Qwen3-VL 模板自行展开为 <|vision_start|><|image_pad|><|vision_end|>）----
        all_messages = self._normalize_sft_messages(
            prompt_messages, candidates[0],
            keep_image_url=True,
            enable_think=self.enable_think,
            strip_image_placeholders=True,
        )

        temp_data = dict(data)
        temp_data["messages"] = all_messages
        temp_data["add_generation_prompt"] = False
        temp_data.setdefault("add_vision_id", False)

        try:
            full_text = self.vl_chat_template.render(temp_data)
        except Exception as e:
            logger.warning(f"[VL-SFT] jinja render 失败: {e}, key={sample.__key__}")
            raise SkipSample()

        full_input_ids = self.hf_tokenizer.encode(full_text, add_special_tokens=False)

        # ---- 扩展视觉块 ----
        # 模板每张图只产出一个 <|image_pad|>，需要按该图 merge 后的 token 数复制。
        # Dense Pixel-DiT places one timestep token before each image patch block.
        image_token_id = self.image_token_id
        num_pad_in_text = sum(1 for t in full_input_ids if t == image_token_id)
        if num_pad_in_text != len(image_patch_counts):
            n_img = len(image_patch_counts)
            logger.warning(
                f"[VL-SFT] 渲染后 image_pad 数({num_pad_in_text})"
                f" != 图片数({n_img}), key={sample.__key__}"
            )
            raise SkipSample()

        if num_pad_in_text > 0:
            expanded = []
            img_iter = iter(image_patch_counts)
            for tok in full_input_ids:
                if tok == image_token_id:
                    expanded.append(self.time_token_id)
                    expanded.extend([image_token_id] * next(img_iter))
                else:
                    expanded.append(tok)
            full_input_ids = expanded

        if self.use_v5_route_tokens:
            self._insert_route_tokens_after_assistant_headers(
                full_input_ids,
                self.text_gate_token_id,
            )

        # 超长直接丢弃
        if len(full_input_ids) > self.packing_seq_length:
            raise SkipSample()

        # ---- 计算 loss mask（沿用文本 SFT 逻辑） ----
        loss_mask_list = self._compute_sft_loss_mask(
            full_input_ids,
            all_messages,
            prompt_messages,
            labels,
            has_tools=bool(data.get("tools")),
            render_nonfirst_system=False,
        )
        if not any(m > 0 for m in loss_mask_list):
            raise SkipSample()

        input_ids = np.array(full_input_ids, dtype=np.int64)
        loss_mask = self._align_loss_mask_to_next_token(loss_mask_list)
        total_len = len(input_ids)
        if total_len == 0:
            return

        pad_id = self.pad_id

        # Labels: shifted next-token; loss_mask=0 → pad_id（batch 阶段会转为 -100）
        target = input_ids.copy()
        target = np.roll(target, shift=-1)
        target[-1] = pad_id
        target[loss_mask == 0] = pad_id

        # self._debug_print_sft_sample(
        #     "sft_image_und", sample.__key__, full_text,
        #     input_ids, target, loss_mask,
        # )

        # ---- 构建图像/视觉相关掩码 ----
        seq_len = total_len
        image_input_mask = torch.from_numpy(input_ids == image_token_id)
        num_image_tokens = int(image_input_mask.sum().item())
        image_generate_mask = torch.zeros(num_image_tokens * 4, dtype=torch.bool)
        image_understand_mask = torch.ones(num_image_tokens, dtype=torch.bool)
        total_img_und_mask = torch.ones(num_image_tokens, dtype=torch.bool)

        # ---- 3D M-RoPE position_ids ----
        # image_grid_thw 已在上面转换为 merged grid；按 image_pad 连续块构造，
        # 避免原始 get_rope_index 无法跨过 time_token 的问题。
        if image_grid_thw is None:
            rope_image_grids = []
        elif isinstance(image_grid_thw, np.ndarray):
            rope_grid_array = (
                image_grid_thw[None, :] if image_grid_thw.ndim == 1 else image_grid_thw
            )
            rope_image_grids = [
                rope_grid_array[i] for i in range(rope_grid_array.shape[0])
            ]
        elif isinstance(image_grid_thw, torch.Tensor):
            rope_grid_tensor = (
                image_grid_thw.unsqueeze(0) if image_grid_thw.ndim == 1 else image_grid_thw
            )
            rope_image_grids = [
                rope_grid_tensor[i] for i in range(rope_grid_tensor.shape[0])
            ]
        else:
            rope_image_grids = [image_grid_thw]
        position_ids = self._build_position_ids(
            input_ids.tolist(), rope_image_grids
        )

        # ---- image_thw_grids（merge 后） ----
        image_thw_final = []
        if image_grid_thw_ori is not None:
            if isinstance(image_grid_thw_ori, np.ndarray):
                if image_grid_thw_ori.ndim == 1:
                    image_thw_final = [torch.from_numpy(image_grid_thw_ori)]
                else:
                    image_thw_final = [
                        torch.from_numpy(image_grid_thw_ori[i])
                        for i in range(image_grid_thw_ori.shape[0])
                    ]
            elif isinstance(image_grid_thw_ori, torch.Tensor):
                if image_grid_thw_ori.ndim == 1:
                    image_thw_final = [image_grid_thw_ori]
                else:
                    image_thw_final = [
                        image_grid_thw_ori[i] for i in range(image_grid_thw_ori.shape[0])
                    ]

        cond_indices, nocond_indices = create_sample_startend_row_indices(
            input_ids=torch.from_numpy(input_ids),
            img_start_id=self.image_start_id,
            img_end_id=self.image_end_id,
            bidirectional=True,
        )

        yield TransfusionTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=pixel_values if pixel_values is not None else [],
            image_thw_grids=image_thw_final,
            image_input_mask=image_input_mask,
            total_img_und_mask=total_img_und_mask,
            image_generate_mask=image_generate_mask,
            image_understand_mask=image_understand_mask,
            image_gen_seq_lens=[],
            videos=[],
            video_thw_grids=[],
            video_input_mask=torch.zeros(seq_len, dtype=torch.bool),
            task_type=sample.task_type,
            text=torch.from_numpy(input_ids),
            target=torch.from_numpy(target),
            total_len=total_len,
            position_ids=position_ids,
            cond_startend_row_indices=cond_indices,
            nocond_startend_row_indices=nocond_indices,
        )

    def select_samples_to_pack(self, samples: List[TransfusionTaskSample]) -> List[List[TransfusionTaskSample]]:
        """
        按任务类型分组并打包样本。

        确保返回的每组样本都是同一模态，这样后续 batch 时不会混合不同模态。
        """
        # 1. 按任务类型把样本分流到不同的桶里，避免交叉污染
        task_groups = defaultdict(list)
        for s in samples:
            t_type = getattr(s, "task_type", "unknown")
            task_groups[t_type].append(s)

        packed_groups = []

        if not self.is_packing_enabled:
            # 非 packing 模式: 每个样本单独成组，但仍按任务类型排序
            # 这样相邻的样本会是同一模态，有助于后续 batch 时按模态分组
            for task_type, group_samples in task_groups.items():
                for s in group_samples:
                    packed_groups.append([s])
            return packed_groups

        # 2. 在各自的桶内独立进行背包打包
        for task_type, group_samples in task_groups.items():
            lengths = [s.total_len for s in group_samples]
            groups = greedy_knapsack(lengths, group_samples, self.packing_seq_length)

            # ── shuffle 打包结果, 打散 knapsack 产生的 "[大,大,大,残]" 确定性 FIFO 顺序,
            # 消除 num_edit_images 的阶梯状周期震荡
            random.shuffle(groups)

            packed_groups.extend(groups)
        return packed_groups

    def pack_selected_samples(self, samples: List[TransfusionTaskSample]) -> TransfusionTaskSample:
        """Concatenate selected samples into one packed training sample."""
        if not self.is_packing_enabled:
            raise RuntimeError("pack_selected_samples 被调用，但 Packing 未开启")

        # 提取共同的 task_type
        task_type = getattr(samples[0], "task_type", "unknown")
        # 准备拼接
        packed_texts = [s.text for s in samples]
        packed_targets = [s.target for s in samples]

        packed_imgs = [s.imgs for s in samples if s.imgs is not None and len(s.imgs) > 0]
        packed_image_thw_grids = [
            s.image_thw_grids for s in samples if s.image_thw_grids is not None and len(s.image_thw_grids) > 0
        ]
        packed_image_input_masks = [s.image_input_mask for s in samples]

        packed_image_generate_masks = [
            s.image_generate_mask
            for s in samples
            if s.image_generate_mask is not None and len(s.image_generate_mask) > 0
        ]
        packed_image_understand_masks = [
            s.image_understand_mask
            for s in samples
            if s.image_understand_mask is not None and len(s.image_understand_mask) > 0
        ]
        packed_total_img_und_mask = [
            s.total_img_und_mask for s in samples if s.total_img_und_mask is not None and len(s.total_img_und_mask) > 0
        ]
        packed_image_gen_seq_lens = []
        packed_image_gen_grid_thw = []
        for s in samples:
            if s.image_gen_seq_lens is not None and len(s.image_gen_seq_lens) > 0:
                packed_image_gen_seq_lens.extend(s.image_gen_seq_lens)
            if s.image_gen_grid_thw is not None and len(s.image_gen_grid_thw) > 0:
                if isinstance(s.image_gen_grid_thw, torch.Tensor):
                    if s.image_gen_grid_thw.ndim == 1:
                        packed_image_gen_grid_thw.append(s.image_gen_grid_thw.unsqueeze(0))
                    else:
                        packed_image_gen_grid_thw.append(s.image_gen_grid_thw)
                else:
                    packed_image_gen_grid_thw.extend(s.image_gen_grid_thw)

        # 物理张量拼接 (Torch)
        packed_text = torch.cat(packed_texts, dim=0)
        packed_target = torch.cat(packed_targets, dim=0)
        packed_image_input_mask = torch.cat(packed_image_input_masks, dim=0)
        # 物理张量拼接 (Numpy)
        final_image_thw_grids = (
            np.concatenate(packed_image_thw_grids, axis=0) if packed_image_thw_grids else np.array([])
        )
        # 拼接生成专属掩码
        final_image_generate_mask = (
            torch.cat(packed_image_generate_masks, dim=0)
            if packed_image_generate_masks
            else torch.empty(0, dtype=torch.bool)
        )
        # 拼接理解专属掩码
        final_image_understand_mask = (
            torch.cat(packed_image_understand_masks, dim=0)
            if packed_image_understand_masks
            else torch.empty(0, dtype=torch.bool)
        )
        final_total_img_und_mask = (
            torch.cat(packed_total_img_und_mask, dim=0)
            if packed_total_img_und_mask
            else torch.empty(0, dtype=torch.bool)
        )

        # 拼接 position_ids: 直接 concat，每个 sample 的 position_ids 保持独立
        # (cu_seqlens 已标记 sample 边界，不需要 offset)
        packed_position_ids_list = [s.position_ids for s in samples if s.position_ids is not None]
        if packed_position_ids_list:
            packed_position_ids = torch.cat(packed_position_ids_list, dim=1)
        else:
            packed_position_ids = None

        final_image_gen_grid_thw = (
            torch.cat([
                thw if isinstance(thw, torch.Tensor) and thw.ndim == 2
                else torch.as_tensor(thw, dtype=torch.long).view(1, 3)
                for thw in packed_image_gen_grid_thw
            ], dim=0)
            if packed_image_gen_grid_thw else None
        )

        # 拼接 startend_row_indices: 每个样本的 indices 需要偏移
        # 由于 packing 会把多个样本拼接成一个序列，需要调整每个样本内的索引
        packed_cond_indices_list = []
        packed_nocond_indices_list = []
        offset = 0
        for s in samples:
            if s.cond_startend_row_indices is not None:
                # cond_startend_row_indices: [1, L, 2], 需要偏移 offset
                cond_idx = s.cond_startend_row_indices.clone()
                # LTS (第0列) 和 UTE (第1列) 都需要加上 offset
                # 但是 LTS = L 表示"全可见"，不需要偏移；只偏移实际的行索引
                # 实际上 LTS 和 UTE 都是行索引，拼接时需要整体偏移
                cond_idx = cond_idx + offset
                packed_cond_indices_list.append(cond_idx[0])  # [L, 2]
            if s.nocond_startend_row_indices is not None:
                nocond_idx = s.nocond_startend_row_indices.clone()
                nocond_idx = nocond_idx + offset
                packed_nocond_indices_list.append(nocond_idx[0])  # [L, 2]
            offset += s.total_len

        final_cond_indices = None
        final_nocond_indices = None
        if packed_cond_indices_list:
            final_cond_indices = torch.cat(packed_cond_indices_list, dim=0).unsqueeze(0)  # [1, total_L, 2]
        if packed_nocond_indices_list:
            final_nocond_indices = torch.cat(packed_nocond_indices_list, dim=0).unsqueeze(0)

        return TransfusionTaskSample(
            __key__="|".join([s.__key__ for s in samples]),
            __subflavors__=samples[0].__subflavors__,
            imgs=torch.cat(packed_imgs, dim=0) if packed_imgs else None,
            videos=None,
            image_thw_grids=final_image_thw_grids,
            video_thw_grids=np.array([]),
            image_input_mask=packed_image_input_mask,
            video_input_mask=torch.zeros_like(packed_image_input_mask),
            text=packed_text,
            target=packed_target,
            total_len=len(packed_text),
            total_img_und_mask=final_total_img_und_mask,
            image_generate_mask=final_image_generate_mask,
            image_understand_mask=final_image_understand_mask,
            image_gen_seq_lens=packed_image_gen_seq_lens,
            image_gen_grid_thw=final_image_gen_grid_thw,
            task_type=task_type,
            position_ids=packed_position_ids,
            cond_startend_row_indices=final_cond_indices,
            nocond_startend_row_indices=final_nocond_indices,
            packed_sample_lens=[s.total_len for s in samples],
        )
