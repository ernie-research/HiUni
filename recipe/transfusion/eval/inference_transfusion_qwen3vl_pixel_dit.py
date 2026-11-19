#!/usr/bin/env python3
"""
inference_transfusion_qwen3vl_pixel_dit.py
Pixel-DiT 模式 TransfusionQwen3VL 批量推理评测脚本

支持:
  - LongText / OneIG / DPG benchmark (txt/csv/jsonl prompt 列表)
  - 自定义 prompt 文件
  - FID / CLIP Score 指标
  - resume (跳过已生成图片)

与 inference_transfusion_qwen3vl.py 的区别:
  - dense Pixel-DiT: ViT conditioning + pixel-space diffusion head,
    latent 直接是 raw pixel patch [H*W, 768]。
  - model/transfusion/transfusion_sampler.py 的 predict_eps/sample_images 支持同一
    H/W 下多 prompt batch 推理；本脚本按 --batch_size 分 chunk 生成和保存。
  - --height/--width 直接对应 sample_images 的 merged grid 参数 (raw 分辨率 = 2H x 2W patch,
    16px/patch), 替代 base 模式的 --resolution。
  - 只支持 dense Pixel-DiT。
    --drop_dit_text 可屏蔽 Pixel-DiT head 的 text→image 注意力。

用法:
    # 单卡
    python recipe/transfusion/eval/inference_transfusion_qwen3vl_pixel_dit.py \
        --ckpt output/pixel_dit_overfit --hf_path /path/to/Qwen3-VL-4B-Instruct \
        --out_dir ./LongText_results/pixel_dit/ZH \
        --data_format txt --data_file data/benchmark/LongText_zh.txt \
        --height 9 --width 9 --steps 50 --shift 1.0 --save_images
"""
import os
import sys
import math
import json
import logging
import argparse
import copy
import inspect
import random
import re
import subprocess
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import dataclasses
import model.transfusion.transfusion_qwen3vl_bridge  # noqa: F401

from megatron.bridge import AutoBridge
from model.transfusion.transfusion_qwen3vl_provider import TransfusionQwen3VLProvider

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformers import AutoTokenizer

from data.transfusion.transfusion_data_cooker import open_image_checked
from hf_inference.configuration_pixel_dit import reject_enabled_refiner
from model.transfusion.transfusion_sampler import sample_images

# ── 从 inference_transfusion_qwen3 复用共享组件 ──
from recipe.transfusion.eval.pixel_dit_runtime import (
    setup_dist,
    cleanup_dist,
    is_main,
    _to01,
    ONEIG_CLASS_ITEM,
    CLIPScoreEvaluator,
    FIDFeatureExtractor,
    calculate_frechet_distance,
)
from recipe.transfusion.eval.pixel_dit_runtime import (
    _init_megatron_parallel_state,
    _resolve_megatron_ckpt_dir,
    load_megatron_checkpoint,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def patch_qwen3vl_rotary_packed_seq_compat() -> bool:
    """Adapt Megatron Core's mRoPE call to the Bridge rotary API."""
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import (
        Qwen3VLTextRotaryEmbedding,
    )

    original_forward = Qwen3VLTextRotaryEmbedding.forward
    if "packed_seq_params" in inspect.signature(original_forward).parameters:
        return False

    def forward(self, position_ids, mrope_section=None, packed_seq_params=None):
        if packed_seq_params is not None:
            raise ValueError(
                "Qwen3VLTextRotaryEmbedding compatibility path requires "
                "packed_seq_params=None"
            )
        return original_forward(self, position_ids, mrope_section=mrope_section)

    Qwen3VLTextRotaryEmbedding.forward = forward
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 模型构建 (dense Pixel-DiT)
# ═══════════════════════════════════════════════════════════════════════════════
def _read_pixel_dit_config_from_ckpt(ckpt_dir: str):
    """从 checkpoint 的 run_config.yaml 读取 pixel-dit 结构超参。

    num_diffusion_layers / pixel_dit_gate_inject_layer / dit_head_impl 必须训推一致,
    硬编码易错位, 优先从训练时落盘的 run_config.yaml 读取。读取失败则返回 None
    让调用方用 CLI 默认值。
    """
    import yaml

    cfg_path = os.path.join(ckpt_dir, "run_config.yaml")
    if not os.path.isfile(cfg_path):
        logger.warning(f"run_config.yaml 不存在于 {ckpt_dir}, 使用 CLI 默认结构超参")
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict):
        reject_enabled_refiner(
            {**(cfg.get("model") or {}), **(cfg.get("transfusion") or {})}
        )

    def _search(d, key):
        """递归查找首个匹配 key (run_config 结构层级不固定)。"""
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = _search(v, key)
                if r is not None:
                    return r
        return None

    return {
        "num_diffusion_layers": _search(cfg, "num_diffusion_layers"),
        "pixel_dit_hidden_size": _search(cfg, "pixel_dit_hidden_size"),
        "num_diffusion_heads": _search(cfg, "num_diffusion_heads"),
        "num_diffusion_kv_heads": _search(cfg, "num_diffusion_kv_heads"),
        "diffusion_intermediate_size": _search(cfg, "diffusion_intermediate_size"),
        "pixel_dit_gate_inject_layer": _search(cfg, "pixel_dit_gate_inject_layer"),
        "pixel_dit_gate_input": _search(cfg, "pixel_dit_gate_input"),
        "use_pixel_dit_pixel_residual": _search(
            cfg, "use_pixel_dit_pixel_residual"
        ),
        "pixel_dit_ref_gate_mode": _search(cfg, "pixel_dit_ref_gate_mode"),
        "dit_head_impl": _search(cfg, "dit_head_impl"),
        "dit_block_adaln": _search(cfg, "dit_block_adaln"),
        "adaln_post_modulation": _search(cfg, "adaln_post_modulation"),
        "predict_x0": _search(cfg, "predict_x0"),
        "diffusion_loss_type": _search(cfg, "diffusion_loss_type"),
        "predict_gate_mode": _search(cfg, "predict_gate_mode"),
        "use_dynamic_noise_scale": _search(cfg, "use_dynamic_noise_scale"),
        "dynamic_noise_ref_tokens": _search(cfg, "dynamic_noise_ref_tokens"),
        "dynamic_noise_base_scale": _search(cfg, "dynamic_noise_base_scale"),
    }


def build_model_pixel_dit(
    hf_path: str,
    ckpt_path: str,
    num_diffusion_layers: int,
    pixel_dit_gate_inject_layer: int,
    pixel_dit_hidden_size: int = 0,
    num_diffusion_heads: int = 0,
    num_diffusion_kv_heads: int = 0,
    diffusion_intermediate_size: int = 0,
    pixel_dit_gate_input: str = "pixel",
    use_pixel_dit_pixel_residual: bool = True,
    pixel_dit_ref_gate_mode: str = "none",
    dit_head_impl: str = "torch",
    dit_block_adaln: bool = True,
    adaln_post_modulation: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    iteration: int = None,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    """Build a dense Pixel-DiT model from a Megatron checkpoint."""
    _init_megatron_parallel_state(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    logger.info(f"Building Pixel-DiT TransfusionQwen3VL Model from {hf_path}")

    bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True)
    base_model_cfg = bridge.to_megatron_provider(load_weights=False)

    kwargs = {f.name: getattr(base_model_cfg, f.name) for f in dataclasses.fields(base_model_cfg)}
    kwargs.update({
        "tensor_model_parallel_size": tensor_model_parallel_size,
        "pipeline_model_parallel_size": pipeline_model_parallel_size,
        "sequence_parallel": False,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "num_diffusion_layers": num_diffusion_layers,
        "pixel_dit_hidden_size": pixel_dit_hidden_size,
        "num_diffusion_heads": num_diffusion_heads,
        "num_diffusion_kv_heads": num_diffusion_kv_heads,
        "diffusion_intermediate_size": diffusion_intermediate_size,
        "pixel_dit_gate_inject_layer": pixel_dit_gate_inject_layer,
        "pixel_dit_gate_input": pixel_dit_gate_input,
        "use_pixel_dit_pixel_residual": use_pixel_dit_pixel_residual,
        "pixel_dit_ref_gate_mode": pixel_dit_ref_gate_mode,
        "dit_head_impl": dit_head_impl,
        "dit_block_adaln": dit_block_adaln,
        "adaln_post_modulation": adaln_post_modulation,
        "perform_initialization": False,
        "use_flashmask": True,
    })

    kwargs.pop("init_method", None)
    kwargs.pop("output_layer_init_method", None)

    valid_fields = {f.name for f in dataclasses.fields(TransfusionQwen3VLProvider)}
    kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    provider = TransfusionQwen3VLProvider(**kwargs)
    model = provider.provide(pre_process=True, post_process=True)
    model = model.to(dtype=dtype, device=device)

    # ckpt_path=None builds an empty model, e.g. as the target of an HF import.
    ckpt_dir = None
    if ckpt_path is not None:
        ckpt_dir = _resolve_megatron_ckpt_dir(ckpt_path, iteration)
        logger.info(f"Loading Megatron checkpoint from {ckpt_dir}")
        load_megatron_checkpoint(model, ckpt_dir)

    # A checkpoint can restore the base Qwen config with FlashMask disabled.
    # Pixel-DiT sampling requires the bidirectional image block used in training.
    for cfg_owner in (model, getattr(model, "language_model", None)):
        cfg = getattr(cfg_owner, "config", None)
        if cfg is not None and hasattr(cfg, "use_flashmask"):
            cfg.use_flashmask = True

    model.eval()
    logger.info("Megatron pixel-dit model loaded successfully")

    return model, ckpt_dir


def seed_inference_rank(seed: int, dp_rank: int) -> int:
    """Synchronize RNGs inside a TP group while varying them across DP ranks."""
    effective_seed = seed + dp_rank
    random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 加载 (轻量版: 只读文本, 不构造序列 — 序列构造由 sample_images 内部完成)
# ═══════════════════════════════════════════════════════════════════════════════

# A ratio in a prompt is not necessarily an image ratio: timestamps such as
# ``3:07`` and cinematic crop notes such as ``2.39:1`` are common in this
# benchmark.  Keep the parser deliberately conservative and require either a
# common ratio or nearby aspect/format wording before accepting an arbitrary
# numeric pair.
_PROMPT_RATIO_RE = re.compile(
    r"(?<![\d.:])"
    r"(?P<w>\d{1,3}(?:\.\d+)?)\s*[:：]\s*"
    r"(?P<h>\d{1,3}(?:\.\d+)?)"
    r"(?![\d.:])"
)
_COMMON_PROMPT_RATIOS = {
    (1.0, 1.0),
    (2.0, 3.0),
    (3.0, 2.0),
    (3.0, 4.0),
    (4.0, 3.0),
    (9.0, 16.0),
    (16.0, 9.0),
    (21.0, 9.0),
}
_RATIO_CONTEXT_RE = re.compile(
    r"aspect\s*[-_ ]?ratio|ratio|宽高比|宽高|画面比例|图片比例|图像比例|"
    r"照片比例|生成比例|画幅|横纵比|比例|format|orientation|"
    r"portrait|landscape|frame|canvas|poster|海报|画面|照片|图片|图像",
    re.IGNORECASE,
)
_TIME_CONTEXT_RE = re.compile(
    r"\b(?:rec|time|timestamp|clock|timer|date)\b|时间|日期|计时器|"
    r"挂钟|时钟|指针|停在|遮幅|影院|电影宽银幕",
    re.IGNORECASE,
)


def parse_prompt_aspect_ratio(prompt: str):
    """Return ``(width_ratio, height_ratio, label)`` when *prompt* asks for it.

    The benchmark embeds ratios in free-form Chinese/English text instead of
    providing a separate field.  Invalid pairs and likely timestamps are
    ignored.  Natural-language orientation words are accepted only when no
    numeric ratio was found.
    """
    if not isinstance(prompt, str):
        return None

    for match in _PROMPT_RATIO_RE.finditer(prompt):
        width_ratio = float(match.group("w"))
        height_ratio = float(match.group("h"))
        if width_ratio <= 0 or height_ratio <= 0:
            continue

        before = prompt[max(0, match.start() - 48):match.start()]
        after = prompt[match.end():match.end() + 48]
        context = before + after

        # ``00:45``/``20:00`` and similar clock values are not dimensions.
        if width_ratio == 0 or height_ratio == 0:
            continue
        if width_ratio < 10 and height_ratio < 60 and _TIME_CONTEXT_RE.search(context):
            continue

        ratio_pair = (width_ratio, height_ratio)
        has_context = bool(_RATIO_CONTEXT_RE.search(context))
        if ratio_pair not in _COMMON_PROMPT_RATIOS and not has_context:
            continue

        # Do not turn a cinematic crop/letterbox mention (for example
        # ``2.39:1``) into the requested canvas unless the prompt explicitly
        # says it is the image/画面比例.
        if (width_ratio != int(width_ratio) or height_ratio != int(height_ratio)) and not has_context:
            continue
        label = f"{width_ratio:g}:{height_ratio:g}"
        return width_ratio, height_ratio, label

    # A few benchmark prompts use orientation words instead of a numeric pair.
    orientation_patterns = (
        # Keep ``portrait``/``landscape`` broad because they are commonly
        # used as the image subject/orientation ("realistic portrait",
        # "landscape image").  Directional adjectives need a layout noun;
        # otherwise phrases such as "vertical path" are object geometry.
        (r"\bportrait\b", 9.0, 16.0),
        (
            r"\bvertical\s+(?:orientation|format|image|photo(?:graph)?|"
            r"poster|storyboard|layout|composition|frame|canvas|panel|shot|view)\b|"
            r"竖版|竖幅|竖屏|竖向(?:构图|画面|海报|分镜|排列)|"
            r"纵版|纵向(?:构图|画面|海报|分镜|排列)",
            9.0,
            16.0,
        ),
        (r"\blandscape\s+(?:orientation|format|image|photo(?:graph)?|painting|view|scene|composition)\b", 16.0, 9.0),
        (
            r"\bhorizontal\s+(?:orientation|format|image|photo(?:graph)?|"
            r"poster|storyboard|layout|composition|frame|canvas|panel|shot|view)\b|"
            r"横版|横幅|横屏|横向(?:构图|画面|海报|分镜|排列)",
            16.0,
            9.0,
        ),
        (r"\bpanoramic\b|\bpanorama\b|全景", 2.0, 1.0),
        (
            r"\bsquare\s+(?:image|photo|format|canvas|composition|layout|frame|poster|picture)\b|"
            r"方形(?:构图|画面|海报|照片|图片)|正方形(?:构图|画面|海报|照片|图片)",
            1.0,
            1.0,
        ),
    )
    for pattern, width_ratio, height_ratio in orientation_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return width_ratio, height_ratio, f"{width_ratio:g}:{height_ratio:g}"
    return None


def resolve_prompt_grid(args, prompt: str):
    """Resolve the merged Pixel-DiT grid for one prompt.

    ``args.height``/``args.width`` are the baseline grid.  When a prompt has
    an aspect ratio, preserve the baseline area and derive both edges from the
    ratio.  This keeps a 2048 baseline close to 2048x2048 pixels in area even
    for non-square prompts.  Returned sizes are merged-grid units and therefore
    decode to dimensions in 32-pixel increments.
    """
    base_height = int(args.height)
    base_width = int(args.width)
    if not getattr(args, "respect_prompt_aspect_ratio", True):
        return base_height, base_width, None

    parsed = parse_prompt_aspect_ratio(prompt)
    if parsed is None:
        return base_height, base_width, None
    width_ratio, height_ratio, label = parsed

    target_area = float(base_height * base_width)
    ratio = width_ratio / height_ratio
    grid_width = max(1, int(round(math.sqrt(target_area * ratio))))
    grid_height = max(1, int(round(math.sqrt(target_area / ratio))))
    return grid_height, grid_width, label


def load_prompts(
    data_format: str,
    data_file: str,
    reasoning_category: str = None,
    prompt_texts=None,
):
    """读取 prompt 列表, 返回 [(item_id, prompt_text, category_or_None), ...]。

    prompt_texts: CLI 直接传入的 prompt，优先于 data_file，供 quick generation 使用。
    txt:   每行一条, 跳过空行和 # 开头 (与 T2iEvalDataset._load_data 的 txt 分支一致)
    jsonl: 每行 json, 优先取 "prompt"/"text"，也兼容原始 benchmark 的
           "prompt_cn"/"prompt_en" 字段
    csv:   OneIG (reasoning_category="run_OneIG") 取 category/id/prompt_cn 列;
           其余取 item_id/text 列 (drop_duplicates)
    """
    if prompt_texts:
        return [(idx, prompt, None) for idx, prompt in enumerate(prompt_texts)]

    if not data_file:
        raise ValueError("必须提供 --data_file 或至少一个 --prompt")

    items = []
    if data_format == "txt":
        with open(data_file, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    items.append((idx, line, None))
                    idx += 1
    elif data_format == "jsonl":
        with open(data_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Qwen-Image-Bench uses ``ID``/``prompt`` while the older
                # benchmark files used ``item_id``/``text``.  Keep both
                # conventions accepted so the inference output can be fed
                # directly to the Qwen-Image-Bench judge.
                item_id = data.get("ID", data.get("id", data.get("item_id", idx)))
                prompt = data.get("prompt", data.get("text"))
                if prompt is None:
                    # The raw Qwen-Image-Bench export keeps both language
                    # columns; use Chinese as the deterministic default when
                    # a language-specific prompt file was not prepared.
                    prompt = data.get("prompt_cn", data.get("prompt_en"))
                if prompt is None:
                    raise KeyError(
                        "JSONL prompt record must contain one of: prompt, text, prompt_cn, prompt_en"
                    )
                items.append((item_id, prompt, None))
    elif data_format == "csv":
        import pandas as pd
        if reasoning_category == "run_OneIG":
            df = pd.read_csv(data_file)
            prompt_col = "prompt_cn" if "prompt_cn" in df.columns else "prompt_en"
            render_tasks = df[["category", "id", prompt_col]]
            for row in render_tasks.itertuples(index=False):
                items.append((row.id, getattr(row, prompt_col), row.category))
        else:
            df = pd.read_csv(data_file)
            render_tasks = df[["item_id", "text"]].drop_duplicates()
            for row in render_tasks.itertuples(index=False):
                items.append((row.item_id, row.text, None))
    else:
        raise NotImplementedError(f"Unsupported data_format: {data_format}")
    return items


def format_item_id(item_id):
    """Match pandas-inferred OneIG ids and image filenames such as 0.webp."""
    if isinstance(item_id, (int, np.integer)):
        return str(int(item_id))
    item_id_str = str(item_id).strip()
    return str(int(item_id_str)) if item_id_str.isdigit() else item_id_str


def get_output_relpath(args, item_id, category, repeat_id):
    """生成图片的相对路径。OneIG 需按 category 存储在不同目录."""
    item_id_str = format_item_id(item_id)
    if args.run_OneIG:
        mapped_cat = ONEIG_CLASS_ITEM.get(category, category)
        filename = f"{item_id_str}.png" if args.repeat_times == 1 else f"{item_id_str}_{repeat_id}.png"
        return os.path.join("images", mapped_cat, args.model_name, filename)
    if args.run_DPG:
        return os.path.join("images", f"{item_id_str}_{repeat_id}.png")
    if isinstance(item_id, (int, np.integer)):
        filename = f"{int(item_id):05d}_{repeat_id}.png"
    else:
        filename = f"{item_id_str}_{repeat_id}.png"
    return os.path.join("images", filename)


def get_oneig_stitched_relpath(args, item_id, category):
    """Return the final OneIG gallery path produced after repeat images are stitched."""
    item_id_str = format_item_id(item_id)
    mapped_cat = ONEIG_CLASS_ITEM.get(category, category)
    return os.path.join("images", mapped_cat, args.model_name, f"{item_id_str}.png")


def _image_matches_grid(path, grid_height: int, grid_width: int, stitched: bool = False):
    """Check an existing output header without decoding the image pixels."""
    expected_height = int(grid_height) * 32
    expected_width = int(grid_width) * 32
    if stitched:
        expected_height *= 2
        expected_width *= 2
    try:
        with open_image_checked(path) as image:
            return image.size == (expected_width, expected_height)
    except (
        FileNotFoundError,
        OSError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ):
        return False


def find_existing_output_for_resume(
    args,
    item_id,
    category,
    repeat_id,
    out_path,
    expected_grid=None,
):
    """Find an output for resume, rejecting images from a different aspect ratio."""
    if os.path.exists(out_path):
        if expected_grid is None or _image_matches_grid(out_path, *expected_grid):
            return out_path
        logger.warning("Ignoring stale output with wrong size: %s", out_path)
    if args.run_OneIG and args.repeat_times > 1:
        stitched_path = os.path.join(
            args.out_dir,
            get_oneig_stitched_relpath(args, item_id, category),
        )
        if os.path.exists(stitched_path) and (
            expected_grid is None
            or _image_matches_grid(stitched_path, *expected_grid, stitched=True)
        ):
            return stitched_path
        if os.path.exists(stitched_path) and expected_grid is not None:
            logger.warning("Ignoring stale stitched output with wrong size: %s", stitched_path)
    return None


def oneig_stitched_output_needs_rerun(args, item_id, category):
    """Detect a stale OneIG gallery with a black/missing quadrant.

    Older runs could stitch before all repeat images were written and pad the
    missing slots with black pixels. Such a gallery must not make resume skip
    the prompt permanently.
    """
    if not (args.run_OneIG and args.repeat_times > 1):
        return False
    stitched_path = os.path.join(args.out_dir, get_oneig_stitched_relpath(args, item_id, category))
    if not os.path.isfile(stitched_path):
        repeat_paths = [
            os.path.join(args.out_dir, get_output_relpath(args, item_id, category, repeat_id))
            for repeat_id in range(args.repeat_times)
        ]
        existing_repeats = sum(os.path.isfile(path) for path in repeat_paths)
        return 0 < existing_repeats < args.repeat_times
    try:
        with open_image_checked(stitched_path) as image:
            arr = np.asarray(image.convert("RGB"))
        if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return True
        half_h, half_w = arr.shape[0] // 2, arr.shape[1] // 2
        tiles = (
            arr[:half_h, :half_w],
            arr[:half_h, half_w:],
            arr[half_h:, :half_w],
            arr[half_h:, half_w:],
        )
        return any(np.max(tile) <= 1 for tile in tiles)
    except Exception as exc:
        logger.warning("Unable to validate OneIG gallery %s: %s", stitched_path, exc)
        return True


def remove_oneig_stitched_output(args, item_id, category):
    if not (args.run_OneIG and args.repeat_times > 1):
        return
    stitched_path = os.path.join(args.out_dir, get_oneig_stitched_relpath(args, item_id, category))
    try:
        os.remove(stitched_path)
        logger.warning("Removed stale OneIG gallery; rerunning prompt %s", format_item_id(item_id))
    except FileNotFoundError:
        pass


def write_metadata(args, prompts_all, rank):
    """保存 prompt 列表到 metadata.jsonl"""
    if not (is_main(rank) and args.save_images and prompts_all):
        return

    meta_path = os.path.join(args.out_dir, "metadata.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"Generating metadata at {meta_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for item_id, prompt, category in prompts_all:
            images = [get_output_relpath(args, item_id, category, r) for r in range(args.repeat_times)]
            grid_height, grid_width, aspect_ratio = resolve_prompt_grid(args, prompt)
            entry = {
                "prompt_id": format_item_id(item_id),
                "prompt": prompt,
                "images": images,
                "height": grid_height * 32,
                "width": grid_width * 32,
            }
            if aspect_ratio is not None:
                entry["aspect_ratio"] = aspect_ratio
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_benchmark_spec(args, benchmark_name: str):
    model_name = args.model_name
    if benchmark_name == "LongText_ZH":
        return {
            "out_dir": f"./LongText_results/{model_name}/ZH",
            "data_format": "txt",
            "data_file": "data/benchmark/LongText_zh.txt",
            "run_OneIG": False,
            "run_DPG": False,
        }
    if benchmark_name == "LongText_EN":
        return {
            "out_dir": f"./LongText_results/{model_name}/EN",
            "data_format": "txt",
            "data_file": "data/benchmark/LongText_en.txt",
            "run_OneIG": False,
            "run_DPG": False,
        }
    if benchmark_name == "OneIG_ZH":
        return {
            "out_dir": f"./OneIG_results/{model_name}/ZH",
            "data_format": "csv",
            "data_file": "OneIG-Benchmark/OneIG-Bench-ZH.csv",
            "run_OneIG": True,
            "run_DPG": False,
        }
    if benchmark_name == "OneIG_EN":
        # 别被 _old 的名字误导：OneIG-Bench_old.csv 才是官方 OneIG-Bench.csv 的原版
        # （1120 条逐条一致，只是把 prompt_en 列改名成 loader 需要的 prompt_cn）。
        # 同目录下的 OneIG-Bench.csv 是本地把 tag 串扩写成长描述的改写版，1120 条
        # 全部与官方不同，换过去分数将无法与历史结果和官方 leaderboard 对比。
        return {
            "out_dir": f"./OneIG_results/{model_name}/EN",
            "data_format": "csv",
            "data_file": "OneIG-Benchmark/OneIG-Bench_old.csv",
            "run_OneIG": True,
            "run_DPG": False,
        }
    if benchmark_name == "DPG":
        return {
            "out_dir": f"./DPG_results/{model_name}",
            "data_format": "csv",
            "data_file": "data/benchmark/dpg_bench.csv",
            "run_OneIG": False,
            "run_DPG": True,
        }
    raise ValueError(f"Unknown benchmark: {benchmark_name}")


def parse_benchmark_list(args):
    if not args.benchmark_list:
        return []
    return [name.strip() for name in args.benchmark_list.split(",") if name.strip()]


def build_benchmark_args(args, benchmark_name: str):
    benchmark_args = copy.copy(args)
    spec = get_benchmark_spec(args, benchmark_name)
    for key, value in spec.items():
        setattr(benchmark_args, key, value)
    benchmark_args.benchmark_name = benchmark_name
    return benchmark_args


def launch_oneig_postprocess(args, rank: int):
    if not (is_main(rank) and args.save_images and args.oneig_postprocess and args.run_OneIG):
        return None

    image_dir = os.path.join(args.out_dir, "images")
    if not os.path.isdir(image_dir):
        logger.warning("OneIG postprocess skipped: missing %s", image_dir)
        return None

    status_path = os.path.join(args.out_dir, ".oneig_postprocess.status.json")
    cmd = [
        sys.executable,
        args.oneig_postprocess_script,
        "--input_root", image_dir,
        "--output_root", image_dir,
        "--metadata_path", os.path.join(args.out_dir, "metadata.jsonl"),
        "--status_path", status_path,
        "--workers", str(args.oneig_postprocess_workers),
    ]
    proc = subprocess.Popen(cmd, start_new_session=args.oneig_postprocess_detach)

    if args.oneig_postprocess_queue_file:
        queue_dir = os.path.dirname(args.oneig_postprocess_queue_file)
        if queue_dir:
            os.makedirs(queue_dir, exist_ok=True)
        with open(args.oneig_postprocess_queue_file, "a", encoding="utf-8") as f:
            f.write(f"{proc.pid}\t{status_path}\t{args.model_name}\t{args.benchmark_name}\n")

    logger.info(
        "OneIG postprocess queued: pid=%s benchmark=%s status=%s",
        proc.pid,
        getattr(args, "benchmark_name", args.out_dir),
        status_path,
    )
    return proc


def run_pixel_dit_benchmark(
    benchmark_args,
    model,
    tokenizer,
    device,
    rank: int,
    dp_rank: int,
    dp_world_size: int,
    tp_local_rank: int,
    predict_x0: bool,
    predict_gate_mode: Optional[bool],
    dynamic_noise_scale: bool,
    dynamic_noise_ref_tokens: int,
    dynamic_noise_base_scale: float,
    inception_model=None,
    clip_evaluator=None,
):
    benchmark_name = getattr(benchmark_args, "benchmark_name", "")
    save_img_dir = os.path.join(benchmark_args.out_dir, "images")
    if benchmark_args.save_images and is_main(rank):
        os.makedirs(save_img_dir, exist_ok=True)
        logger.info(f"Output dir: {benchmark_args.out_dir}")

    if is_main(rank):
        if benchmark_name:
            logger.info(f"[benchmark] {benchmark_name}")
        prompt_source = "CLI --prompt" if benchmark_args.prompt else benchmark_args.data_file
        logger.info(f"[3/4] Loading prompts from {prompt_source}")

    reasoning_category = "run_OneIG" if benchmark_args.run_OneIG else None
    prompts_all = load_prompts(
        benchmark_args.data_format,
        benchmark_args.data_file,
        reasoning_category,
        prompt_texts=benchmark_args.prompt,
    )
    if benchmark_args.num_samples > 0:
        prompts_all = prompts_all[:benchmark_args.num_samples]

    write_metadata(benchmark_args, prompts_all, rank)

    # Keep completed-sample accounting on the original DP shard, but scan the
    # complete prompt list for pending work.  A rank-local resume scan can
    # leave all missing jobs on one rank (typically rank 0), causing the other
    # DP workers to sit idle at the next collective.
    prompts_local = prompts_all[dp_rank::dp_world_size] if dp_world_size > 1 else prompts_all
    assigned_item_ids = {
        item_id for item_id, _prompt, _category in prompts_local
    }

    if is_main(rank):
        logger.info(f"  Total prompts: {len(prompts_all)}, this DP rank handles: {len(prompts_local)}")

    generated_count = 0
    skipped_count = 0
    real_features_list = []
    fake_features_list = []
    local_clip_score_sum = 0.0
    local_clip_count = 0

    # Only rank 0 inspects existing OneIG galleries. Decoding every 4K grid on
    # every DP rank is needlessly expensive, and concurrent deletion can make
    # the resume decision differ between ranks.
    rerun_oneig_prompts = set()
    if benchmark_args.resume and benchmark_args.run_OneIG and benchmark_args.repeat_times > 1:
        if is_main(rank):
            for item_id, _prompt, category in prompts_all:
                if oneig_stitched_output_needs_rerun(benchmark_args, item_id, category):
                    rerun_oneig_prompts.add((format_item_id(item_id), category))
                    remove_oneig_stitched_output(benchmark_args, item_id, category)
            logger.warning(
                "OneIG resume scan found %d prompt(s) requiring full repeat rerun",
                len(rerun_oneig_prompts),
            )
        if dp_world_size > 1:
            rerun_payload = [list(rerun_oneig_prompts) if is_main(rank) else None]
            dist.broadcast_object_list(rerun_payload, src=0)
            rerun_oneig_prompts = set(tuple(value) for value in rerun_payload[0])

    all_pending_jobs = []
    for item_id, prompt, category in prompts_all:
        rerun_oneig_prompt = (format_item_id(item_id), category) in rerun_oneig_prompts
        prompt_height, prompt_width, prompt_aspect_ratio = resolve_prompt_grid(
            benchmark_args, prompt
        )
        for r in range(benchmark_args.repeat_times):
            out_path = os.path.join(benchmark_args.out_dir, get_output_relpath(benchmark_args, item_id, category, r))

            existing_path = None
            if benchmark_args.resume and not rerun_oneig_prompt:
                existing_path = find_existing_output_for_resume(
                    benchmark_args,
                    item_id,
                    category,
                    r,
                    out_path,
                    expected_grid=(prompt_height, prompt_width),
                )

            if existing_path is not None:
                # Resume-time metric accounting remains shard-local so every
                # existing output is counted exactly once across DP ranks.
                if item_id not in assigned_item_ids:
                    continue
                skipped_count += 1
                if benchmark_args.calc_fid or benchmark_args.calc_clip:
                    # A stitched OneIG gallery represents all repeats; evaluate it only once.
                    if existing_path != out_path and r > 0:
                        continue
                    img = TF.to_tensor(open_image_checked(existing_path).convert("RGB")).unsqueeze(0).to(device)
                else:
                    continue

                generated_count += 1
                if benchmark_args.calc_clip and clip_evaluator is not None and tp_local_rank == 0:
                    batch_score, batch_n = clip_evaluator.compute_score(img, [prompt])
                    local_clip_score_sum += batch_score
                    local_clip_count += batch_n

                if benchmark_args.calc_fid and inception_model is not None and tp_local_rank == 0:
                    fake_features_list.append(inception_model(img.to(device)).cpu().numpy())
                    if benchmark_args.image_path:
                        ref_path = os.path.join(benchmark_args.image_path, f"{item_id}.jpg")
                        if os.path.exists(ref_path):
                            try:
                                raw_img = TF.to_tensor(open_image_checked(ref_path).convert("RGB")).unsqueeze(0).to(device)
                            except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
                                logger.warning("Skipping oversized FID reference image %s: %s", ref_path, exc)
                            else:
                                raw_img = F.interpolate(raw_img.float(), size=(299, 299),
                                                         mode="bilinear", align_corners=False)
                                real_features_list.append(inception_model(raw_img).cpu().numpy())
            else:
                all_pending_jobs.append(
                    (
                        item_id,
                        prompt,
                        category,
                        r,
                        out_path,
                        prompt_height,
                        prompt_width,
                        prompt_aspect_ratio,
                    )
                )

    pending_jobs = (
        all_pending_jobs[dp_rank::dp_world_size]
        if dp_world_size > 1
        else all_pending_jobs
    )

    if is_main(rank):
        logger.info(
            "  Resume scan: global_pending=%d, this DP rank pending=%d, skipped=%d",
            len(all_pending_jobs),
            len(pending_jobs),
            skipped_count,
        )

    batch_size = max(1, int(benchmark_args.batch_size))
    # sample_images constructs one packed sequence and one latent shape for a
    # whole batch.  Group jobs by their resolved grid so a 16:9 prompt is never
    # accidentally sampled with a neighboring 1:1 prompt's dimensions.
    jobs_by_grid = {}
    for job in pending_jobs:
        jobs_by_grid.setdefault((job[5], job[6]), []).append(job)
    total_batches = sum(
        (len(grid_jobs) + batch_size - 1) // batch_size
        for grid_jobs in jobs_by_grid.values()
    )
    progress = tqdm(
        range(total_batches),
        disable=not is_main(rank),
    )
    max_batch_retries = 3
    batched_jobs = []
    for grid_height, grid_width in sorted(jobs_by_grid):
        grid_jobs = jobs_by_grid[(grid_height, grid_width)]
        for start in range(0, len(grid_jobs), batch_size):
            batched_jobs.append(
                (grid_height, grid_width, grid_jobs[start:start + batch_size])
            )

    for _batch_index, (grid_height, grid_width, jobs) in zip(progress, batched_jobs):
        progress.set_description(f"grid {grid_height}x{grid_width}")
        batch_prompts = [job[1] for job in jobs]
        images = None
        for attempt in range(1, max_batch_retries + 1):
            images = sample_images(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                H=grid_height,
                W=grid_width,
                num_steps=benchmark_args.steps,
                shift=benchmark_args.shift,
                device=device,
                guidance_scale=benchmark_args.cfg_scale,
                negative_prompt=benchmark_args.negative_prompt,
                cfg_keep_chat_template=benchmark_args.cfg_keep_chat_template,
                drop_dit_text=benchmark_args.drop_dit_text,
                predict_gate_mode=predict_gate_mode,
                predict_x0=predict_x0,
                scheduler_type=benchmark_args.scheduler,
                timestep_sampling=benchmark_args.timestep_sampling,
                logit_norm_mu=benchmark_args.logit_norm_mu,
                logit_norm_sigma=benchmark_args.logit_norm_sigma,
                dynamic_noise_scale=dynamic_noise_scale,
                dynamic_noise_ref_tokens=dynamic_noise_ref_tokens,
                dynamic_noise_base_scale=dynamic_noise_base_scale,
            )
            valid_shape = (
                images is not None
                and images.dim() == 4
                and images.shape[1] == 3
                and images.shape[0] == len(jobs)
            )
            if valid_shape:
                break
            returned = None if images is None else tuple(images.shape)
            logger.warning(
                "Generation returned %s for %d jobs (attempt %d/%d); retrying batch",
                returned,
                len(jobs),
                attempt,
                max_batch_retries,
            )
        if images is None or images.dim() != 4 or images.shape[1] != 3 or images.shape[0] != len(jobs):
            raise RuntimeError(
                f"Generation failed after {max_batch_retries} attempts for batch starting at local job {start}"
            )

        valid_count = min(images.shape[0], len(jobs))
        images = images[:valid_count].float()

        if benchmark_args.save_images and tp_local_rank == 0:
            for img_i, job in enumerate(jobs[:valid_count]):
                out_path = job[4]
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                save_image(images[img_i].clamp(0, 1), out_path)

        generated_count += valid_count

        if benchmark_args.calc_clip and clip_evaluator is not None and tp_local_rank == 0:
            batch_score, batch_n = clip_evaluator.compute_score(
                images,
                batch_prompts[:valid_count],
            )
            local_clip_score_sum += batch_score
            local_clip_count += batch_n

        if benchmark_args.calc_fid and inception_model is not None and tp_local_rank == 0:
            fake_features_list.append(inception_model(images.to(device)).cpu().numpy())
            if benchmark_args.image_path:
                for img_i, job in enumerate(jobs[:valid_count]):
                    item_id = job[0]
                    ref_path = os.path.join(benchmark_args.image_path, f"{item_id}.jpg")
                    if os.path.exists(ref_path):
                        try:
                            raw_img = TF.to_tensor(open_image_checked(ref_path).convert("RGB")).unsqueeze(0).to(device)
                        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
                            logger.warning("Skipping oversized FID reference image %s: %s", ref_path, exc)
                        else:
                            raw_img = F.interpolate(raw_img.float(), size=(299, 299),
                                                     mode="bilinear", align_corners=False)
                            real_features_list.append(inception_model(raw_img).cpu().numpy())

        if is_main(rank) and generated_count % 10 == 0:
            logger.info(f"Generated: {generated_count}, Skipped: {skipped_count}")

    metrics_str = []
    if benchmark_args.calc_clip:
        if dp_world_size > 1:
            stat_tensor = torch.tensor([local_clip_score_sum, local_clip_count], device=device, dtype=torch.float64)
            dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
            total_score = stat_tensor[0].item()
            total_count = stat_tensor[1].item()
        else:
            total_score = local_clip_score_sum
            total_count = local_clip_count
        if total_count > 0:
            avg_clip_score = total_score / total_count
            metrics_str.append(f"CLIP Score: {avg_clip_score:.4f}")

    if benchmark_args.calc_fid and is_main(rank):
        local_fake_feats = np.concatenate(fake_features_list, axis=0) if fake_features_list else np.empty((0, 2048))
        local_real_feats = np.concatenate(real_features_list, axis=0) if real_features_list else np.empty((0, 2048))
        logger.info(f"FID: Real={local_real_feats.shape[0]}, Fake={local_fake_feats.shape[0]}")
        if local_real_feats.shape[0] >= 2 and local_fake_feats.shape[0] >= 2:
            mu_real = np.mean(local_real_feats, axis=0)
            sigma_real = np.cov(local_real_feats, rowvar=False)
            mu_fake = np.mean(local_fake_feats, axis=0)
            sigma_fake = np.cov(local_fake_feats, rowvar=False)
            fid_score = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
            metrics_str.append(f"FID Score: {fid_score:.4f}")
            os.makedirs(benchmark_args.out_dir, exist_ok=True)
            with open(os.path.join(benchmark_args.out_dir, "fid_score.txt"), "w", encoding="utf-8") as f:
                f.write(f"{fid_score}\n")

    # Stitching is launched by the outer loop after all DP ranks reach a
    # barrier. Starting it here races with late ranks still writing repeats.
    postprocess_proc = None

    if is_main(rank):
        logger.info("=" * 50)
        logger.info("Pixel-DiT Inference Completed")
        logger.info("Benchmark: %s", benchmark_name or benchmark_args.out_dir)
        logger.info("Generated: %d images (Skipped: %d)", generated_count, skipped_count)
        logger.info("Output: %s", benchmark_args.out_dir)
        if metrics_str:
            logger.info("Evaluation Results:")
            for m in metrics_str:
                logger.info("  - %s", m)
        logger.info("=" * 50)

    return {
        "generated_count": generated_count,
        "skipped_count": skipped_count,
        "postprocess_proc": postprocess_proc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="TransfusionQwen3VL Pixel-DiT Inference & Evaluation")

    # 模型参数
    parser.add_argument("--ckpt", type=str, required=True, help="Megatron checkpoint 路径")
    parser.add_argument("--hf_path", type=str, default=None,
                         help="HuggingFace Qwen3-VL 模型路径 (tokenizer + vision config)")
    parser.add_argument("--out_dir", type=str, default="inference_pixel_dit_results", help="输出目录")
    parser.add_argument("--iteration", type=int, default=None, help="迭代步数")

    # 推理参数
    parser.add_argument("--steps", type=int, default=50, help="采样步数")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG guidance scale")
    parser.add_argument("--shift", type=float, default=1.0, help="Flow matching shift (pixel-dit 默认 1.0)")
    parser.add_argument("--scheduler", type=str, default="euler", choices=["euler", "dpm2", "dpm3"],
                         help="Scheduler: euler (1st-order), dpm2 (DPM-Solver++ 2nd), dpm3 (3rd)")
    parser.add_argument("--timestep_sampling", type=str, default="uniform",
                        choices=["uniform", "shift_logit_norm"],
                        help="Inference timestep grid; shift_logit_norm aligns training sampling")
    parser.add_argument("--logit_norm_mu", type=float, default=0.8)
    parser.add_argument("--logit_norm_sigma", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=8, help="merged grid H (raw 分辨率 = 2H*16 px)")
    parser.add_argument("--width", type=int, default=8, help="merged grid W (raw 分辨率 = 2W*16 px)")
    parser.add_argument(
        "--respect_prompt_aspect_ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="从 prompt 中解析宽高比并按样本调整输出尺寸",
    )
    parser.add_argument("--batch_size", type=int, default=1,
                         help="每次 sample_images 处理的 prompt 数量; 同一 batch 使用相同 height/width")
    parser.add_argument("--repeat_times", type=int, default=1, help="每个 Prompt 生成几张图片")

    # Pixel-DiT 结构超参 (优先从 checkpoint 的 run_config.yaml 自动读取)
    parser.add_argument("--num_diffusion_layers", type=int, default=6, help="多层 DiT head 层数")
    parser.add_argument(
        "--pixel_dit_hidden_size", type=int, default=0,
        help="Pixel-DiT hidden size; 0 follows the language backbone",
    )
    parser.add_argument("--num_diffusion_heads", type=int, default=0)
    parser.add_argument("--num_diffusion_kv_heads", type=int, default=0)
    parser.add_argument("--diffusion_intermediate_size", type=int, default=0)
    parser.add_argument("--pixel_dit_gate_inject_layer", type=int, default=3, help="Gate 注入层 (0-based)")
    parser.add_argument(
        "--pixel_dit_gate_input",
        choices=["pixel", "vit"],
        default="pixel",
        help="Pixel-DiT HiGate high-info 输入: merged pixel 或 merged ViT feat",
    )
    parser.add_argument(
        "--use_pixel_dit_pixel_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject pixel_proj(x_t) into the Pixel-DiT hidden state through HiGate.",
    )
    parser.add_argument("--pixel_dit_ref_gate_mode", type=str, default="none",
                        choices=["none", "shared", "separate"],
                        help="Pixel-DiT edit ref pixel gate mode")
    parser.add_argument("--dit_head_impl", type=str, default="torch", choices=["torch", "megatron"],
                         help="DiT head 实现: torch=原生 (默认) / megatron=TE 并行层")
    parser.add_argument("--dit_block_adaln", action=argparse.BooleanOptionalAction, default=True,
                        help="DiT block 内是否启用 AdaLN 调制; 优先从 checkpoint run_config.yaml 读取")
    parser.add_argument("--adaln_post_modulation", action=argparse.BooleanOptionalAction, default=False,
                        help="是否使用 post-modulation AdaLN; 优先从 checkpoint run_config.yaml 读取")
    parser.add_argument("--predict_x0", action=argparse.BooleanOptionalAction, default=True,
                        help="x0 prediction; 显式 CLI 参数优先于 checkpoint run_config.yaml")
    parser.add_argument("--diffusion_loss_type", type=str, default=None,
                        choices=["v-loss", "x-loss"],
                        help="扩散 loss 类型；独立于模型输出目标 predict_x0")
    parser.add_argument("--dynamic_noise_scale", action=argparse.BooleanOptionalAction, default=True,
                        help="采样初始噪声是否按 token 数动态缩放; 优先从 checkpoint run_config.yaml 读取")
    parser.add_argument("--dynamic_noise_ref_tokens", type=int, default=256,
                        help="dynamic noise scale 参考 token 数")
    parser.add_argument("--dynamic_noise_base_scale", type=float, default=1.0,
                        help="dynamic noise scale 基础倍率")

    # 分布式并行配置
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1, help="张量模型并行度 (TP)")
    parser.add_argument("--pipeline_model_parallel_size", type=int, default=1, help="流水线模型并行度 (PP)")

    # Benchmark 选项
    parser.add_argument("--run_OneIG", action="store_true", help="OneIG benchmark (csv 输出目录布局)")
    parser.add_argument("--run_DPG", action="store_true", help="DPG benchmark (csv 输出目录布局)")
    parser.add_argument("--model_name", type=str, default="transfusion_qwen3vl_pixel_dit", help="模型名称 (用于保存)")

    # 数据参数
    parser.add_argument("--data_format", type=str, default="txt", help="数据格式 (jsonl/txt/csv)")
    parser.add_argument("--data_file", type=str, default="", help="数据文件路径")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="直接生成一条 prompt；可重复传入。设置后优先于 --data_file",
    )
    parser.add_argument("--image_path", type=str, default="", help="真实图像路径 (FID 用)")

    # 其他选项
    parser.add_argument("--num_samples", type=int, default=-1, help="推理样本数, -1 表示全部")
    parser.add_argument("--save_images", action="store_true", help="保存生成的图像")
    parser.add_argument("--calc_fid", action="store_true", help="计算 FID")
    parser.add_argument("--calc_clip", action="store_true", help="计算 CLIP Score")
    parser.add_argument("--resume", action="store_true", help="跳过已生成的图片")
    parser.add_argument("--benchmark_list", type=str, default="", help="逗号分隔的 benchmark 列表, 单次加载模型后依次执行")
    parser.add_argument("--oneig_postprocess", action=argparse.BooleanOptionalAction, default=False,
                        help="对 OneIG 结果启动后台拼接任务")
    parser.add_argument("--oneig_postprocess_detach", action=argparse.BooleanOptionalAction, default=True,
                        help="OneIG 拼接任务是否从当前进程组脱离")
    parser.add_argument("--oneig_postprocess_queue_file", type=str, default="",
                        help="记录后台拼接任务 pid/status 的队列文件")
    parser.add_argument("--oneig_postprocess_script", type=str,
                        default="recipe/transfusion/scripts/oneig_postprocess_parallel.py")
    parser.add_argument("--oneig_postprocess_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--negative_prompt", type=str, default="", help="CFG 负向 prompt")
    parser.add_argument(
        "--drop_dit_text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="屏蔽 Pixel-DiT head 中 image token 对 text token 的注意力。",
    )
    parser.add_argument(
        "--cfg_keep_chat_template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CFG uncond 保留 Qwen chat template, 仅清空/替换 user content; "
             "同时复用 cond 分支的 generation instruction prefix。"
             "默认关闭，即使用纯 image-only uncond。",
    )
    parser.add_argument(
        "--predict_gate_mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="V5 image gate: auto-detect (default), force enable, or disable.",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    if patch_qwen3vl_rotary_packed_seq_compat():
        logger.info("Enabled Qwen3VL mRoPE packed_seq_params compatibility")

    # An explicit CLI override must win over metadata saved in the checkpoint.
    # This is needed when validating older Pixel-DiT checkpoints whose config
    # predates the x0-target semantics used by the model branch.
    predict_x0_cli = getattr(args, "_predict_x0_cli_override", False) or any(
        arg == "--predict_x0" or arg.startswith("--predict_x0=")
        or arg == "--no-predict_x0" or arg.startswith("--no-predict_x0=")
        for arg in sys.argv[1:]
    )
    loss_type_cli = any(
        arg == "--diffusion_loss_type" or arg.startswith("--diffusion_loss_type=")
        for arg in sys.argv[1:]
    )
    predict_gate_cli = any(
        arg == "--predict_gate_mode" or arg.startswith("--predict_gate_mode=")
        or arg == "--no-predict_gate_mode" or arg.startswith("--no-predict_gate_mode=")
        for arg in sys.argv[1:]
    )

    is_ddp, rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer_path = args.hf_path if args.hf_path else args.ckpt
    if is_main(rank):
        logger.info(f"[1/4] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if is_main(rank):
        logger.info(f"[2/4] Loading model from {args.ckpt}...")

    ckpt_dir_for_cfg = _resolve_megatron_ckpt_dir(args.ckpt, args.iteration)
    ckpt_cfg = _read_pixel_dit_config_from_ckpt(ckpt_dir_for_cfg)
    num_diffusion_layers = args.num_diffusion_layers
    pixel_dit_hidden_size = args.pixel_dit_hidden_size
    num_diffusion_heads = args.num_diffusion_heads
    num_diffusion_kv_heads = args.num_diffusion_kv_heads
    diffusion_intermediate_size = args.diffusion_intermediate_size
    pixel_dit_gate_inject_layer = args.pixel_dit_gate_inject_layer
    pixel_dit_gate_input = args.pixel_dit_gate_input
    use_pixel_dit_pixel_residual = args.use_pixel_dit_pixel_residual
    pixel_dit_ref_gate_mode = args.pixel_dit_ref_gate_mode
    dit_head_impl = args.dit_head_impl
    dit_block_adaln = args.dit_block_adaln
    adaln_post_modulation = args.adaln_post_modulation
    predict_x0 = args.predict_x0
    diffusion_loss_type = args.diffusion_loss_type
    dynamic_noise_scale = args.dynamic_noise_scale
    dynamic_noise_ref_tokens = args.dynamic_noise_ref_tokens
    dynamic_noise_base_scale = args.dynamic_noise_base_scale
    if ckpt_cfg is not None:
        if ckpt_cfg.get("num_diffusion_layers") is not None:
            num_diffusion_layers = int(ckpt_cfg["num_diffusion_layers"])
        if ckpt_cfg.get("pixel_dit_hidden_size") is not None:
            pixel_dit_hidden_size = int(ckpt_cfg["pixel_dit_hidden_size"])
        if ckpt_cfg.get("num_diffusion_heads") is not None:
            num_diffusion_heads = int(ckpt_cfg["num_diffusion_heads"])
        if ckpt_cfg.get("num_diffusion_kv_heads") is not None:
            num_diffusion_kv_heads = int(ckpt_cfg["num_diffusion_kv_heads"])
        if ckpt_cfg.get("diffusion_intermediate_size") is not None:
            diffusion_intermediate_size = int(
                ckpt_cfg["diffusion_intermediate_size"]
            )
        if ckpt_cfg.get("pixel_dit_gate_inject_layer") is not None:
            pixel_dit_gate_inject_layer = int(ckpt_cfg["pixel_dit_gate_inject_layer"])
        if ckpt_cfg.get("pixel_dit_gate_input") is not None:
            pixel_dit_gate_input = str(ckpt_cfg["pixel_dit_gate_input"])
        if ckpt_cfg.get("use_pixel_dit_pixel_residual") is not None:
            use_pixel_dit_pixel_residual = bool(
                ckpt_cfg["use_pixel_dit_pixel_residual"]
            )
        if ckpt_cfg.get("pixel_dit_ref_gate_mode") is not None:
            pixel_dit_ref_gate_mode = str(ckpt_cfg["pixel_dit_ref_gate_mode"])
        if ckpt_cfg.get("dit_head_impl") is not None:
            dit_head_impl = ckpt_cfg["dit_head_impl"]
        if ckpt_cfg.get("dit_block_adaln") is not None:
            dit_block_adaln = bool(ckpt_cfg["dit_block_adaln"])
        if ckpt_cfg.get("adaln_post_modulation") is not None:
            adaln_post_modulation = bool(ckpt_cfg["adaln_post_modulation"])
        if ckpt_cfg.get("predict_x0") is not None and not predict_x0_cli:
            predict_x0 = bool(ckpt_cfg["predict_x0"])
        if ckpt_cfg.get("diffusion_loss_type") is not None and not loss_type_cli:
            diffusion_loss_type = ckpt_cfg["diffusion_loss_type"]
        if ckpt_cfg.get("predict_gate_mode") is not None and not predict_gate_cli:
            args.predict_gate_mode = bool(ckpt_cfg["predict_gate_mode"])
        if ckpt_cfg.get("use_dynamic_noise_scale") is not None:
            dynamic_noise_scale = bool(ckpt_cfg["use_dynamic_noise_scale"])
        if ckpt_cfg.get("dynamic_noise_ref_tokens") is not None:
            dynamic_noise_ref_tokens = int(ckpt_cfg["dynamic_noise_ref_tokens"])
        if ckpt_cfg.get("dynamic_noise_base_scale") is not None:
            dynamic_noise_base_scale = float(ckpt_cfg["dynamic_noise_base_scale"])
    if is_main(rank):
        logger.info(
            f"  num_diffusion_layers={num_diffusion_layers}, "
            f"pixel_dit_hidden_size={pixel_dit_hidden_size}, "
            f"num_diffusion_heads={num_diffusion_heads}, "
            f"num_diffusion_kv_heads={num_diffusion_kv_heads}, "
            f"diffusion_intermediate_size={diffusion_intermediate_size}, "
            f"pixel_dit_gate_inject_layer={pixel_dit_gate_inject_layer}, "
            f"pixel_dit_gate_input={pixel_dit_gate_input}, "
            f"use_pixel_dit_pixel_residual={use_pixel_dit_pixel_residual}, "
            f"pixel_dit_ref_gate_mode={pixel_dit_ref_gate_mode}, "
            f"dit_head_impl={dit_head_impl}, "
            f"dit_block_adaln={dit_block_adaln}, "
            f"adaln_post_modulation={adaln_post_modulation}, "
            f"predict_x0={predict_x0}, "
            f"diffusion_loss_type={diffusion_loss_type}, "
            f"predict_gate_mode={args.predict_gate_mode}, "
            f"dynamic_noise_scale={dynamic_noise_scale}, "
            f"dynamic_noise_ref_tokens={dynamic_noise_ref_tokens}, "
            f"dynamic_noise_base_scale={dynamic_noise_base_scale}, "
            f"drop_dit_text={args.drop_dit_text} "
            f"(train/infer consistent)"
        )

    model, ckpt_dir = build_model_pixel_dit(
        hf_path=args.hf_path if args.hf_path else args.ckpt,
        ckpt_path=args.ckpt,
        num_diffusion_layers=num_diffusion_layers,
        pixel_dit_hidden_size=pixel_dit_hidden_size,
        num_diffusion_heads=num_diffusion_heads,
        num_diffusion_kv_heads=num_diffusion_kv_heads,
        diffusion_intermediate_size=diffusion_intermediate_size,
        pixel_dit_gate_inject_layer=pixel_dit_gate_inject_layer,
        pixel_dit_gate_input=pixel_dit_gate_input,
        use_pixel_dit_pixel_residual=use_pixel_dit_pixel_residual,
        pixel_dit_ref_gate_mode=pixel_dit_ref_gate_mode,
        dit_head_impl=dit_head_impl,
        dit_block_adaln=dit_block_adaln,
        adaln_post_modulation=adaln_post_modulation,
        device=device,
        dtype=torch.bfloat16,
        iteration=args.iteration,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
    )

    from megatron.core import parallel_state
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    # TP ranks must choose the same generation instruction prefix. Different
    # prefix lengths lead to mismatched tensor-parallel collectives.
    seed_inference_rank(args.seed, dp_rank)

    if is_main(rank):
        logger.info(f"DP rank={dp_rank}, DP world_size={dp_world_size} "
                    f"(global rank={rank}, world_size={world_size})")

    inception_model = FIDFeatureExtractor(device) if args.calc_fid else None
    clip_evaluator = CLIPScoreEvaluator(device=device) if args.calc_clip else None

    benchmark_names = parse_benchmark_list(args)
    benchmark_runs = benchmark_names if benchmark_names else [None]
    pending_postprocesses = []
    total_generated_count = 0
    total_skipped_count = 0
    if is_main(rank):
        if benchmark_names:
            logger.info("[3/4] Benchmark list: %s", ", ".join(benchmark_names))
        else:
            logger.info("[3/4] Single benchmark mode")

    for benchmark_name in benchmark_runs:
        benchmark_args = build_benchmark_args(args, benchmark_name) if benchmark_name else args
        benchmark_result = run_pixel_dit_benchmark(
            benchmark_args=benchmark_args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            rank=rank,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            tp_local_rank=parallel_state.get_tensor_model_parallel_rank(),
            predict_x0=predict_x0,
            predict_gate_mode=args.predict_gate_mode,
            dynamic_noise_scale=dynamic_noise_scale,
            dynamic_noise_ref_tokens=dynamic_noise_ref_tokens,
            dynamic_noise_base_scale=dynamic_noise_base_scale,
            inception_model=inception_model,
            clip_evaluator=clip_evaluator,
        )
        total_generated_count += benchmark_result["generated_count"]
        total_skipped_count += benchmark_result["skipped_count"]
        # DP ranks can have different resume work (for example, one rank may
        # own the only missing prompt shard). Keep every rank on the same
        # benchmark before loading the next one; otherwise fast ranks enter
        # the next benchmark while slow ranks are still in sample_images.
        if is_ddp:
            dist.barrier()

        # All ranks have finished writing this benchmark before rank 0 scans
        # and stitches the repeat images.
        postprocess_proc = launch_oneig_postprocess(benchmark_args, rank)
        if postprocess_proc is not None:
            pending_postprocesses.append(postprocess_proc)

    if is_ddp:
        dist.barrier()

    if pending_postprocesses and is_main(rank):
        logger.info(
            "Queued %d OneIG postprocess job(s); batch wrapper will wait before sync.",
            len(pending_postprocesses),
        )

    if is_main(rank):
        logger.info("=" * 50)
        logger.info("Pixel-DiT Inference Completed")
        logger.info("Generated: %d images (Skipped: %d)", total_generated_count, total_skipped_count)
        logger.info("Output: %s", args.out_dir)
        logger.info("=" * 50)

    cleanup_dist(is_ddp)


if __name__ == "__main__":
    main()
