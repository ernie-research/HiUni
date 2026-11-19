"""
多模态数据清洗模块 (Data Cooker)

该模块负责将原始的多模态数据清洗为统一的 RawMultimodalSample 格式，
充当 Transfusion 训练流程中的数据处理入口。

主要组件:
    - BaseCooker: 提取公用的底层多媒体解析逻辑的基类
    - GenerationCooker: 处理图像/视频生成任务的数据清洗
    - UnderstandingCooker: 处理多模态理解/VQA/对话任务的数据清洗
    - COOKER_REGISTRY: 任务类型与 Cooker 的动态注册表
    - cook_multimodal_sample: 统一的数据清洗入口 (Dispatcher)
"""
from typing import Dict, List, Mapping, Optional, Union
from PIL import Image
import logging
import numpy as np
import pickle
import torch
import os
import io
import warnings
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.task_encoder.cooking import basic_sample_keys
from megatron.energon.wrappers import SkipSample

logger = logging.getLogger(__name__)

def _short_exception(e: Exception) -> str:
    """Format an exception into a concise, single-line diagnostic string."""
    message = getattr(e, "message", None) or str(e)
    msg = message.splitlines()[0] if message else "no details"
    code = getattr(e, "error_code", None) or getattr(e, "code", None)
    status = getattr(e, "status_code", None)
    details = [type(e).__name__]
    if code:
        details.append(f"code={code}")
    if status:
        details.append(f"status={status}")
    return f"{' '.join(details)}: {msg}"


def _check_image_size(image: Image.Image) -> None:
    """Reject images over Pillow's decompression-bomb pixel limits."""
    max_pixels = Image.MAX_IMAGE_PIXELS
    if max_pixels is None:
        return

    pixels = max(1, image.size[0]) * max(1, image.size[1])
    if pixels > 2 * max_pixels:
        raise Image.DecompressionBombError(
            f"Image size ({pixels} pixels) exceeds limit of {2 * max_pixels} pixels, "
            "could be decompression bomb DOS attack."
        )
    if pixels > max_pixels:
        raise Image.DecompressionBombWarning(
            f"Image size ({pixels} pixels) exceeds limit of {max_pixels} pixels, "
            "could be decompression bomb DOS attack."
        )


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    """Load a PIL image and flatten any transparency onto a white background."""
    _check_image_size(image)
    image.load()
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        return background.convert("RGB")
    return image.convert("RGB")


def open_image_checked(source) -> Image.Image:
    """Open an image and turn Pillow decompression-bomb warnings into errors.

    A warning is otherwise easy to miss in a dataloader worker while the
    subsequent decode can allocate hundreds of megabytes.  Raising here lets
    the caller use the existing ``SkipSample`` path for the whole record.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        image = Image.open(source)
        _check_image_size(image)
        return image


def _resize_image_to(image: Image.Image, size) -> Image.Image:
    """Resize an image to an exact ``(width, height)`` pair when needed."""
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _align_single_edit_images_to_smaller(
    ref_image: Image.Image, target_image: Image.Image
):
    """Resize the larger single-edit image to the smaller image's dimensions.

    Single-image editing needs the reference and target token grids to match,
    but keeping the smaller native resolution avoids upsampling either image.
    The resize is intentionally exact (no crop or padding), matching the
    existing single-edit alignment behavior.
    """
    ref_pixels = ref_image.size[0] * ref_image.size[1]
    target_pixels = target_image.size[0] * target_image.size[1]
    if ref_pixels >= target_pixels:
        ref_image = _resize_image_to(ref_image, target_image.size)
    elif target_pixels > ref_pixels:
        target_image = _resize_image_to(target_image, ref_image.size)
    return ref_image, target_image


def _image_from_array_checked(array: np.ndarray) -> Image.Image:
    """Convert an embedded image array while applying the same size guard."""
    image = Image.fromarray(array)
    _check_image_size(image)
    return image.convert("RGB")


def _is_remote_url(src: str) -> bool:
    """True for HTTP(S) sources, which are fetched over the network."""
    return urlsplit(src).scheme.lower() in ("http", "https")


def resolve_local_image_path(src: str) -> str:
    """Map a local image reference to a filesystem path.

    Accepts absolute paths, ``file://`` URLs and relative paths. Relative paths
    are joined with ``TRANSFUSION_IMAGE_ROOT`` when it is set, otherwise they
    are resolved against the current working directory.
    """
    if src.startswith("file://"):
        src = unquote(urlsplit(src).path)
    src = os.path.expanduser(src)
    if not os.path.isabs(src):
        root = os.environ.get("TRANSFUSION_IMAGE_ROOT")
        if root:
            src = os.path.join(os.path.expanduser(root), src)
    return src


def _read_image_bytes(src: str) -> bytes:
    """Read raw image bytes from a local path or an HTTP(S) URL."""
    if _is_remote_url(src):
        import requests

        response = requests.get(src, timeout=60)
        response.raise_for_status()
        return response.content
    with open(resolve_local_image_path(src), "rb") as f:
        return f.read()


def load_images(image_sources) -> list[Image.Image]:
    """Load images and convert them to RGB PIL images.

    Each source may be a local path (absolute, relative to
    ``TRANSFUSION_IMAGE_ROOT``, or ``file://``) or an HTTP(S) URL. Any failure
    skips the whole sample.
    """
    if isinstance(image_sources, str):
        image_sources = [image_sources]
    if image_sources is None:
        raise SkipSample()

    image_file_list = []
    for source in image_sources:
        try:
            data = _read_image_bytes(source)
            if data[:4] == b'<svg' or data[:5] == b'<?xml':
                import fitz
                doc = fitz.open(stream=data, filetype="svg")
                try:
                    pix = doc.load_page(0).get_pixmap(dpi=300, alpha=False)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    _check_image_size(image)
                finally:
                    doc.close()
            else:
                image = open_image_checked(io.BytesIO(data))
                image = _convert_image_to_rgb(image)
            image_file_list.append(image)
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as e:
            logger.warning(
                "Skip sample: image exceeds Pillow pixel safety limit for source=%s, "
                "error=%s",
                source,
                _short_exception(e),
            )
            raise SkipSample() from e
        except Exception as e:
            logger.warning(
                "Skip sample: failed to open image source=%s, error=%s",
                source,
                _short_exception(e),
            )
            raise SkipSample() from e
    return image_file_list


@dataclass
class RawMultimodalSample(Sample):
    """
    全能多模态原始数据结构。
    充当 Dispatcher（路由器）的输入载体。
    """

    __key__: str
    __subflavors__: Dict

    # ==========================================
    # 1. 媒体载荷
    # ==========================================
    imgs: List[Image.Image]
    videos: List[torch.Tensor | list[Image.Image]]

    # ==========================================
    # 2. 路由通行证
    # ==========================================
    task_type: str  # 必须字段！例如: "understanding", "generation", "editing"

    # ==========================================
    # 3. 文本载荷 (多态设计)
    # ==========================================
    # 如果是纯预训练，它是 List[str] ("这是一只猫", "<image>", "...")
    # 如果是 SFT/VQA/T2I，它是多轮对话 List[Dict] ([{"role": "user", "content": "..."}, ...])
    text: Union[List[str], List[Dict]]

    # ==========================================
    # 4. 训练掩码 (预训练专用)
    # ==========================================
    # 与 text 列表一一对应，标记每个片段是否参与训练
    # 1=参与训练, 0=跳过（图片占位符或 masked 文本）
    mask: List[int] = None

    # ==========================================
    # 5. 数据源 ID (用于 prompt 映射)
    # ==========================================
    source_id: str = None

    # ==========================================
    # 8. 原始 message dict (SFT 专用)
    # ==========================================
    # task_type="sft_text" / "sft_image_und" 时存放未渲染的原始 EB5 message JSON
    # (prompt/candidates/tools/...)；训练时由 transfusion_task_encoder._encode_sft_text /
    # _encode_sft_image_und 用 jinja 模板渲染
    raw_message: dict = None


# ==========================================
# 1. 定义基类：提取公用的底层多媒体解析逻辑
# ==========================================
class BaseCooker:
    """
    多模态数据清洗基类。
    提供所有数据类型都通用的底层解析方法。
    """

    def cook(self, sample: dict) -> RawMultimodalSample:
        """子类必须实现具体的清洗逻辑"""
        raise NotImplementedError("Subclasses must implement the cook method.")

    def _parse_images(self, sample: dict):
        """Decode embedded image arrays, local paths or HTTP URLs into RGB PIL images."""
        imgs = sample.get("jpgs", None)
        if not imgs:
            return None

        if isinstance(imgs, bytes):
            imgs = pickle.loads(imgs)
        if isinstance(imgs, str):
            imgs = [imgs]

        if isinstance(imgs, list) and len(imgs) > 0:
            if isinstance(imgs[0], np.ndarray):
                try:
                    return [_image_from_array_checked(d) for d in imgs]
                except (Image.DecompressionBombWarning, Image.DecompressionBombError) as e:
                    logger.warning(
                        "Skip sample: embedded image exceeds Pillow pixel safety limit, "
                        "error=%s",
                        _short_exception(e),
                    )
                    raise SkipSample() from e
            elif isinstance(imgs[0], str):
                return load_images(imgs)
        return None

# ==========================================
# 2. 定义具体的清洗器 (策略实现)
# ==========================================
class EditingCooker(BaseCooker):
    """
    专门负责处理【图像/视频编辑】任务的数据清洗，包含
    1. 单图生单图
    2. 单图生多图
    3. 多图生单图
    4. 多图生多图
    5. 思考式生成
    6. 交错式生成
    """
    
    def cook(self, sample: dict) -> RawMultimodalSample:
        """Convert an editing sample into the normalized multimodal representation."""
        text = sample["json"]["text"]
        mask = sample["json"]["mask"]
        imgs = self._parse_images(sample)

        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=imgs,
            videos=None,
            task_type="editing",
            text=text,
            mask=mask,
        )




class GenerationCooker(BaseCooker):
    """Normalize text-to-image samples for Dense Pixel-DiT training."""

    def cook(self, sample: dict) -> RawMultimodalSample:
        text = sample["json"]["text"]
        mask = sample["json"]["mask"]
        if (
            isinstance(text, list)
            and isinstance(mask, list)
            and mask == [0, 1]
            and "<image>" not in text
            and text
        ):
            import random
            text = [random.choice(text), "<image>"]
            mask = [0, 1]
        imgs = self._parse_images(sample)
        if not imgs:
            raise SkipSample()
        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=imgs,
            videos=None,
            task_type="generation",
            text=text,
            mask=mask,
        )


class GenerationCotCooker(GenerationCooker):
    """Clean V5 prompt-refinement samples without exposing protocol variants."""

    SHORT_PROMPT_LEVELS = ("brief", "balanced", "rich")
    PROMPT_LEVELS = (*SHORT_PROMPT_LEVELS, "long")
    DEFAULT_PROMPT_SAMPLING_WEIGHTS = {
        "brief": 0.5,
        "balanced": 0.5,
        "rich": 0.0,
        "long": 0.0,
    }

    @classmethod
    def normalize_prompt_sampling_weights(
        cls, weights: Optional[Mapping[str, float]]
    ) -> Dict[str, float]:
        """Normalize and validate prompt sampling weights across prompt levels."""
        if weights is None:
            weights = cls.DEFAULT_PROMPT_SAMPLING_WEIGHTS
        if not isinstance(weights, Mapping):
            raise ValueError("generation_cot prompt sampling weights must be a mapping")
        keys = set(weights)
        if keys == set(cls.SHORT_PROMPT_LEVELS):
            weights = {**weights, "long": 0.0}
        elif keys != set(cls.PROMPT_LEVELS):
            raise ValueError(
                "generation_cot prompt sampling weights must contain exactly "
                f"{cls.SHORT_PROMPT_LEVELS} or {cls.PROMPT_LEVELS}"
            )
        normalized = {}
        total = 0.0
        for level in cls.PROMPT_LEVELS:
            value = float(weights[level])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid generation_cot weight {level}={value!r}")
            normalized[level] = value
            total += value
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("generation_cot prompt sampling weights must sum to > 0")
        return {level: value / total for level, value in normalized.items()}

    def cook(
        self,
        sample: dict,
        prompt_sampling_weights: Optional[Mapping[str, float]] = None,
        long_template_prob: float = 0.0,
        long_template_version: str = "cpt_v5",
    ) -> RawMultimodalSample:
        """Normalize a prompt-refinement sample for the V5 CoT training route."""
        import json as _json

        json_data = sample.get("json", {}) or {}
        if isinstance(json_data, bytes):
            try:
                json_data = _json.loads(json_data.decode("utf-8"))
            except Exception:
                raise SkipSample()
        elif isinstance(json_data, str):
            try:
                json_data = _json.loads(json_data)
            except Exception:
                raise SkipSample()
        if not isinstance(json_data, dict):
            raise SkipSample()

        refined_prompt = (json_data.get("refined_prompt") or "").strip()
        language = (json_data.get("language") or "").strip().lower()
        if not refined_prompt:
            raise SkipSample()
        if not 0.0 <= float(long_template_prob) <= 1.0:
            raise ValueError("generation_cot long template probability must be in [0, 1]")
        if long_template_version != "cpt_v5":
            raise ValueError(f"unsupported generation_cot long template version: {long_template_version!r}")

        # V5 data normally contains three short prompt candidates.  Legacy
        # records are accepted as a compatibility fallback, but always emit
        # the same V5 text payload to the encoder.
        original_prompts = json_data.get("original_prompts")
        weights = self.normalize_prompt_sampling_weights(prompt_sampling_weights)
        if isinstance(original_prompts, Mapping):
            if any(
                not isinstance(original_prompts.get(level), str)
                or not original_prompts[level].strip()
                for level in self.SHORT_PROMPT_LEVELS
            ):
                raise SkipSample()
            import random
            levels = [level for level in self.PROMPT_LEVELS if weights[level] > 0]
            chosen = random.choices(levels, [weights[level] for level in levels])[0]
            if chosen == "long":
                original_prompt = refined_prompt
                if float(long_template_prob) > 0 and random.random() < float(long_template_prob):
                    noop = (
                        "信息完整，直接生成。"
                        if language.startswith(("zh", "cn"))
                        else "The image is fully specified; render it directly."
                    )
                    json_data["_long_noop_thinking"] = noop
            else:
                original_prompt = original_prompts[chosen].strip()
        elif isinstance(original_prompts, list) and original_prompts:
            original_prompt = str(original_prompts[0]).strip()
        else:
            original_prompt = (json_data.get("original_prompt") or "").strip()
        if not original_prompt:
            raise SkipSample()
        thinking = (json_data.get("thinking") or "").strip()

        imgs = self._parse_images(sample)
        if not imgs:
            raise SkipSample()

        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=imgs,
            videos=None,
            task_type="generation_cot",
            text=[original_prompt, refined_prompt, language, thinking],
            mask=[0, 1, 0, 1],
            source_id=json_data.get("source_id", None),
            raw_message=json_data,
        )


class GenerationCotTextCooker(BaseCooker):
    """Clean text-only visual-planning COT samples for the V5 route."""

    def cook(self, sample: dict) -> RawMultimodalSample:
        import json as _json

        json_data = sample.get("json", {}) or {}
        if isinstance(json_data, bytes):
            try:
                json_data = _json.loads(json_data.decode("utf-8"))
            except Exception as exc:
                raise SkipSample() from exc
        elif isinstance(json_data, str):
            try:
                json_data = _json.loads(json_data)
            except Exception as exc:
                raise SkipSample() from exc
        if not isinstance(json_data, dict):
            raise SkipSample()

        original_prompt = str(json_data.get("original_prompt") or "").strip()
        thinking = str(
            json_data.get("thinking") or json_data.get("cot") or ""
        ).strip()
        language = str(json_data.get("language") or "").strip().lower()
        if not original_prompt or not thinking:
            raise SkipSample()

        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=[],
            videos=None,
            task_type="generation_cot_text",
            text=[original_prompt, thinking, language, thinking],
            mask=[0, 1, 0, 1],
            source_id=json_data.get("source_id"),
            raw_message=json_data,
        )


class MessageCooker(BaseCooker):
    """
    专门负责处理【message 文本预训练】任务的数据清洗。

    数据格式:
    - text: List[str], len(text) = 2, 第一部分为 input，第二部分为 output
    - mask: List[int], len(mask) = 2, mask = [0, 1]，表示第一部分不参与训练，第二部分参与训练
    """

    def cook(self, sample: dict) -> RawMultimodalSample:
        """Normalize a message-style text pretraining sample."""
        json_data = sample.get("json", {})
        text = json_data.get("text", None)
        mask = json_data.get("mask", None)

        # 校验数据格式
        if text is None or not isinstance(text, list) or len(text) != 2:
            logger.warning(f"[MessageCooker] Invalid text format, expected list with len=2, got: {type(text)}")
            raise SkipSample()

        if mask is None:
            mask = [0, 1]  # 默认 mask
        elif not isinstance(mask, list) or len(mask) != 2:
            logger.warning(f"[MessageCooker] Invalid mask format, expected list with len=2, got: {type(mask)}")
            raise SkipSample()

        # 跳过空文本
        if not text[0] and not text[1]:
            raise SkipSample()

        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=None,
            videos=None,
            task_type="message",
            text=text,
            mask=mask,
        )


class TextCooker(BaseCooker):
    """
    专门负责处理【纯文本预训练】任务的数据清洗，支持流式读取。

    流式读取策略:
    - 优先从 sample["text"] 流式获取原始字节流，按需 decode
    - 回退到 sample["json"]["text"] 结构化格式
    - 逐片段处理，避免一次性加载超长文本到内存
    """

    def cook(self, sample: dict) -> RawMultimodalSample:
        """Normalize a plain text sample into the common multimodal structure."""
        # ---- 流式读取: 优先从 "text" 字段获取原始字节流 ----
        raw_text = sample.get("text", None)
        if raw_text is not None:
            # 流式获取的文本通常是 bytes 格式 (webdataset / HDFS)
            if isinstance(raw_text, bytes):
                raw_text = raw_text.decode("utf-8")
            if isinstance(raw_text, str):
                text = [raw_text]
                mask = [1]
            elif isinstance(raw_text, list):
                text = raw_text
                mask = [1] * len(text)
            else:
                logger.warning(f"[TextCooker] Unexpected text type: {type(raw_text)}, skipping")
                raise SkipSample()
        else:
            # ---- 回退到 json 结构化格式 ----
            json_data = sample.get("json", {})
            text = json_data.get("text", None)
            if text is None:
                raise SkipSample()
            if isinstance(text, str):
                text = [text]
                mask = [1]
            elif isinstance(text, list):
                mask = json_data.get("mask", [1] * len(text))
            else:
                raise SkipSample()

        # 跳过空文本
        if not text or all(not t.strip() for t in text if isinstance(t, str)):
            raise SkipSample()

        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=None,
            videos=None,
            task_type="text",
            text=text,
            mask=mask,
        )


class SftTextCooker(BaseCooker):
    """
    专门负责处理【SFT 纯文本】任务的数据清洗（原始 EB5 message 格式）。

    数据格式（来自 convert_text_sft_to_wds.py，passthrough）:
        sample["json"]     : raw JSON（含 prompt/candidates/tools/options/labels/...）
        sample["task_type"]: pickle("sft_text")

    与 MessageCooker 的区别：
        MessageCooker 期望 text=List[str](len=2) + mask=[0,1] 的预渲染格式（pretrain）
        SftTextCooker 透传原始 dict 到 raw_message，由 task encoder 用 jinja 模板渲染
    """

    def cook(self, sample: dict) -> RawMultimodalSample:
        """Normalize a supervised fine-tuning text sample."""
        import json as _json

        json_data = sample.get("json", {})
        if isinstance(json_data, bytes):
            try:
                json_data = _json.loads(json_data.decode("utf-8"))
            except Exception:
                raise SkipSample()
        elif isinstance(json_data, str):
            try:
                json_data = _json.loads(json_data)
            except Exception:
                raise SkipSample()

        if not isinstance(json_data, dict):
            raise SkipSample()

        prompt = json_data.get("prompt")
        candidates = json_data.get("candidates")
        if not prompt or not candidates or not candidates[0]:
            raise SkipSample()

        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=None,
            videos=None,
            task_type="sft_text",
            text=None,
            mask=None,
            raw_message=json_data,
        )


class SftImageUndCooker(BaseCooker):
    """
    专门负责处理【多模态理解 SFT】任务的数据清洗（图像 + 多轮对话，passthrough）。

    数据格式（来自 script/convert_image_und_sft_to_wds.py）:
        sample["json"]     : raw JSON（含 prompt/candidates/labels/tools/...，每个 user.content
                             为 list，可能混合 {type:"text"} 与 {type:"image_url"} 元素）
        sample["task_type"]: pickle("sft_image_und")
    sample["jpgs"]     : pickle(List[str])  按 prompt 内 image_url 出现顺序的图片 URL

    与 SftTextCooker 的区别：携带图像；与 UnderstandingCooker 的区别：text/mask 不预渲染，
    保留原始 message dict，由 task encoder 用 Qwen3-VL chat_template.jinja 渲染。
    """

    def cook(self, sample: dict) -> RawMultimodalSample:
        """Normalize a supervised image-understanding sample."""
        import json as _json

        json_data = sample.get("json", {})
        if isinstance(json_data, bytes):
            try:
                json_data = _json.loads(json_data.decode("utf-8"))
            except Exception:
                raise SkipSample()
        elif isinstance(json_data, str):
            try:
                json_data = _json.loads(json_data)
            except Exception:
                raise SkipSample()

        if not isinstance(json_data, dict):
            raise SkipSample()

        prompt = json_data.get("prompt")
        candidates = json_data.get("candidates")
        if not prompt or not candidates or not candidates[0]:
            raise SkipSample()

        # 校验：raw_message 内 image_url 数 == 下载得到的图片数
        n_image_urls = 0
        for msg in prompt:
            c = msg.get("content")
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        n_image_urls += 1

        imgs = self._parse_images(sample) if n_image_urls > 0 else None
        if n_image_urls > 0 and (imgs is None or len(imgs) != n_image_urls):
            # 下载失败 / URL 数不一致：跳过，避免 token 与图片错位
            raise SkipSample()

        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=imgs,
            videos=None,
            task_type="sft_image_und",
            text=None,
            mask=None,
            raw_message=json_data,
        )


class UnderstandingCooker(BaseCooker):
    """专门负责处理【多模态理解/预训练】任务的数据清洗"""

    def cook(self, sample: dict) -> RawMultimodalSample:
        """Normalize a multimodal understanding pretraining sample."""
        # 预训练数据格式: json 包含 {"text": List[str], "mask": List[int]}
        # task_type 从单独的 .task_type 文件读取
        json_data = sample.get("json", {})

        # 支持两种格式:
        # 1. 预训练格式: {"text": ["hello", "<image>", "world"], "mask": [1, 0, 1]}
        # 2. 对话格式: {"conversations": [...]} 或 {"messages": [...]}
        # 先设置默认值，避免 UnboundLocalError
        text = None
        mask = None
        if "text" in json_data and isinstance(json_data["text"], list):
            # 预训练格式
            text = json_data.get("text", [])
            mask = json_data.get("mask", None)

        if not isinstance(text, list) or not isinstance(mask, list):
            raise SkipSample()
        if not text or len(text) != len(mask):
            raise SkipSample()
        if any(value not in (0, 1) for value in mask):
            raise SkipSample()

        # 校验: 图片数量必须与 <image> 占位符数量一致
        num_image_placeholders = sum(1 for t in text if t == "<image>")
        if num_image_placeholders == 0:
            raise SkipSample()

        imgs = self._parse_images(sample)
        num_imgs = len(imgs) if imgs else 0
        if num_imgs != num_image_placeholders:
            raise SkipSample()

        # 从 .task_type 文件获取 task_type
        task_type = sample.get("task_type", "understanding")
        if isinstance(task_type, bytes):
            try:
                task_type = task_type.decode()
            except UnicodeDecodeError:
                task_type = pickle.loads(task_type)

        # 从 json 中获取 source_id
        source_id = json_data.get("source_id", None)

        # 如果该样本标记了 wobox subflavor，给 source_id 加后缀以区分 prompt
        subflavors = sample.get("__subflavors__", None) or {}
        if source_id and subflavors.get("wobox"):
            source_id = source_id + "_wobox"

        return RawMultimodalSample(
            **basic_sample_keys(sample),
            imgs=imgs,
            videos=None,
            task_type=task_type,
            text=text,
            mask=mask,
            source_id=source_id,
        )


# ==========================================
# 3. 注册表：动态路由 (Dispatcher)
# ==========================================
# 以后新增任何数据类型，只需要加在这个字典里
COOKER_REGISTRY = {
    "text": TextCooker(),
    "message": MessageCooker(),
    "sft_text": SftTextCooker(),
    "sft_image_und": SftImageUndCooker(),
    "generation": GenerationCooker(),
    "generation_cot": GenerationCotCooker(),
    "generation_cot_text": GenerationCotTextCooker(),
    "understanding": UnderstandingCooker(),
    "editing": EditingCooker(),
    # "detection": DetectionCooker(),  # 未来随时扩展
    # "llava_format": LlavaFormatCooker(),
}


def cook_multimodal_sample(
    sample: dict,
    generation_cot_prompt_sampling_weights: Optional[Mapping[str, float]] = None,
    generation_cot_long_template_prob: float = 0.0,
    generation_cot_long_template_version: str = "cpt_v5",
) -> "RawMultimodalSample":
    """Route one source record to the Dense Pixel-DiT data cooker."""
    for std_field in ("task_type", "json", "jpgs"):
        if std_field not in sample:
            for key in list(sample.keys()):
                if isinstance(key, str) and key.endswith("." + std_field):
                    sample[std_field] = sample[key]
                    break

    raw_tt = sample.get("task_type", None)
    used_default = raw_tt is None
    task_type = raw_tt if raw_tt is not None else "understanding"
    # 处理 pickle 编码的 task_type
    if type(task_type) == bytes:
        try:
            import pickle

            task_type = pickle.loads(task_type)
        except:
            # 回退到 UTF-8 解码（兼容旧数据）
            task_type = task_type.decode()
    if task_type == '["generation"]':
        task_type = "generation"
    cooker = COOKER_REGISTRY.get(task_type)
    if not cooker:
        raise ValueError(f"🚨 Unsupported task_type: '{task_type}'. Please register a new Cooker.")

    if task_type == "generation_cot":
        return cooker.cook(
            sample,
            prompt_sampling_weights=generation_cot_prompt_sampling_weights,
            long_template_prob=generation_cot_long_template_prob,
            long_template_version=generation_cot_long_template_version,
        )
    return cooker.cook(sample)
