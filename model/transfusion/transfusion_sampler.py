"""
Transfusion 采样工具 — 在 Megatron 训练环境中直接调用 TransfusionQwen3VLModel 进行推理采样。

功能:
  - predict_eps: Dense Pixel-DiT 单步速度预测
  - sample_images: raw RGB patch 去噪与重排

用法:
  在 training loop 中定期调用 sample_and_save() 即可。
"""

import math
import logging
import random
from typing import List, Optional

import torch

from model.transfusion.schedulers import FlowMatchScheduler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Token IDs (与 transfusion_task_encoder.py 一致)
# ═══════════════════════════════════════════════════════════════
IMG_PAD_ID = 151655     # <image>
BOI_ID = 151652         # <image_start>  (repurposed as BOI)
EOI_ID = 151653         # <image_end>    (repurposed as EOI)
TIME_TOKEN_ID = 151661  # <time_token>

# V5 route tokens.  Keep these IDs local to the sampler so inference does not
# need to instantiate the training task encoder just to build a sequence.
V5_TEXT_GATE_TOKEN_ID = 151669
V5_IMAGE_GATE_TOKEN_ID = 151670
V5_RECAPTION_END_TOKEN_ID = 151671

IM_START_ID = 151644    # <|im_start|>
IM_END_ID = 151645      # <|im_end|>
EOS_ID = 151645         # <|im_end|>
NL_ID = 198             # \n
USER_ID = 872           # "user"
ASSISTANT_ID = 77091    # "assistant"

GEN_INSTRUCTION_PROMPTS = [
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

NEGATIVE_PROMPT = ""


def _supports_v5_route_token(tokenizer, token: str, token_id: int) -> bool:
    """Return whether a tokenizer maps a V5 route token to exactly one ID."""
    if not hasattr(tokenizer, "encode"):
        return False
    try:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        converted = (
            tokenizer.convert_tokens_to_ids(token)
            if hasattr(tokenizer, "convert_tokens_to_ids")
            else None
        )
    except Exception:
        return False
    return encoded == [token_id] and converted == token_id


def _resolve_predict_gate_mode(tokenizer, predict_gate_mode: Optional[bool]) -> bool:
    """Resolve None=auto, True=required, False=disabled for route tokens."""
    supported = _supports_v5_route_token(
        tokenizer, "<|image_gate|>", V5_IMAGE_GATE_TOKEN_ID
    )
    if predict_gate_mode is True and not supported:
        raise ValueError(
            "predict_gate_mode=True requires <|image_gate|> to map to "
            f"the single V5 ID {V5_IMAGE_GATE_TOKEN_ID}"
        )
    return supported if predict_gate_mode is None else bool(predict_gate_mode)


def _assert_flashmask_runtime(model, mode_name: str) -> None:
    """Fail loudly when startend_row_indices would be silently ignored."""
    cfg = getattr(getattr(model, "language_model", None), "config", None)
    if cfg is None:
        cfg = getattr(model, "config", None)
    # Dist checkpoints can restore the base Qwen config after provider setup.
    # Pixel-DiT requires FlashMask regardless of that serialized base flag.
    if cfg is not None and not getattr(cfg, "use_flashmask", False):
        for owner in (model, getattr(model, "language_model", None)):
            owner_cfg = getattr(owner, "config", None)
            if owner_cfg is not None and hasattr(owner_cfg, "use_flashmask"):
                owner_cfg.use_flashmask = True
    if not getattr(cfg, "use_flashmask", False):
        # Some Megatron checkpoint loaders replace the provider config with the
        # serialized base-Qwen config.  The runtime kernel check below is the
        # authoritative guard; do not reject a valid FlashMask installation on
        # this stale boolean alone.
        logger.warning(
            "%s sampling: serialized config.use_flashmask=False; "
            "continuing because the runtime FlashMask kernel check is authoritative.",
            mode_name,
        )

    try:
        from megatron.core.transformer import attention as mcore_attention
        has_flashmask = bool(getattr(mcore_attention, "HAS_FLASHMASK", False))
    except Exception as exc:
        raise RuntimeError(
            f"{mode_name} sampling requires Megatron FlashMask, but the runtime "
            f"availability check failed: {exc}"
        ) from exc

    if not has_flashmask:
        # The backbone ignores startend_row_indices without this kernel, even
        # if the separate diffusion head has its own attention implementation.
        raise RuntimeError(
            f"{mode_name} sampling requires Megatron FlashMask "
            "(HAS_FLASHMASK=False). Refusing to generate with incorrect "
            "image attention or missing packing isolation. Install a compatible "
            "FlashMask kernel in the generation interpreter and restart."
        )
def _pixel_values_to_image(pixel_values, grid_thw, image_idx=0,
                           patch_size=16, temporal_patch_size=2,
                           merge_size=2, in_channels=3,
                           image_mean=0.5, image_std=0.5):
    """
    从 Qwen2VLImageProcessor 输出的 pixel_values 反推回 [0,1] 范围的 [1,C,H,W] 图片。

    pixel_values: [N_total, patch_dim] — processor 输出的 patchified tensor
    grid_thw: [num_images, 3] — (T, H_patches, W_patches)
    image_idx: 提取第几张图片
    """
    P = patch_size
    TP = temporal_patch_size
    C = in_channels
    MS = merge_size

    T = int(grid_thw[image_idx, 0])
    H = int(grid_thw[image_idx, 1])
    W = int(grid_thw[image_idx, 2])
    n_tokens = T * H * W

    # 算 offset
    start = 0
    for j in range(image_idx):
        start += int(grid_thw[j, 0] * grid_thw[j, 1] * grid_thw[j, 2])
    pv = pixel_values[start:start + n_tokens]  # [N, patch_dim]

    H_blocks = H // MS
    W_blocks = W // MS

    # 反转 processor 的 transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    # 从 (T, H_blocks, W_blocks, MS, MS, C, TP, P, P) 恢复到
    # (T, TP, C, H_blocks, MS, P, W_blocks, MS, P)
    inv_perm = [0, 6, 5, 1, 3, 7, 2, 4, 8]
    patches = pv.reshape(T, H_blocks, W_blocks, MS, MS, C, TP, P, P)
    patches = patches.permute(*inv_perm)
    patches = patches.reshape(T * TP, C, H * P, W * P)

    # 取第一帧，反归一化
    frame = patches[0].float()  # [C, H*P, W*P]
    frame = frame * image_std + image_mean
    return frame.unsqueeze(0).clamp(0, 1)  # [1, C, H*P, W*P]



# ═══════════════════════════════════════════════════════════════
#  构造推理输入
# ═══════════════════════════════════════════════════════════════

def build_sampling_sequence(
    tokenizer,
    prompt,
    H: int,
    W: int,
    add_instruction_prefix: bool = False,
    keep_chat_template: bool = True,
    instruction_prefix: Optional[str] = None,
    predict_gate_mode: Optional[bool] = None,
):
    """
    构造 T2I 采样的 token 序列 — 与训练时的 chat template 对齐。

    训练格式:
        <|im_start|>user\n{instruction}{prompt}<|im_end|>\n
        <|im_start|>assistant\n<BOI><TIME><IMG_PAD × H*W><EOI><|im_end|>

    Args:
        tokenizer: HF tokenizer 或 Megatron tokenizer
        prompt: 文本字符串（会被编码），或者 token id 列表（直接使用，避免 BPE 差异）
        H, W: patch grid size (merge-block 粒度)
        keep_chat_template: 为 False 时构造纯 image-only 序列，忽略 prompt。

    Returns:
        input_ids: (1, L) long tensor
    """
    num_image_tokens = H * W

    instruction_ids = []
    if add_instruction_prefix and isinstance(prompt, str):
        if instruction_prefix is None:
            instruction_prefix = random.choice(GEN_INSTRUCTION_PROMPTS)
        # Training tokenizes instruction and caption independently before
        # concatenation. Encoding the joined string can merge tokens across
        # the boundary (notably an English instruction's trailing space).
        if hasattr(tokenizer, 'encode'):
            instruction_ids = tokenizer.encode(
                instruction_prefix, add_special_tokens=False
            )
        else:
            instruction_ids = tokenizer.tokenize(instruction_prefix)

    # 如果 prompt 是 token id 列表，直接使用；否则编码文本。
    # CFG uncond 是否保留 generation instruction prefix 由调用方控制。
    if isinstance(prompt, (list, tuple)):
        text_ids = list(prompt)
    else:
        # 兼容 Megatron DefaultTokenizerText (tokenize) 和 HF tokenizer (encode)
        if hasattr(tokenizer, 'encode'):
            text_ids = tokenizer.encode(prompt, add_special_tokens=False)
        else:
            text_ids = tokenizer.tokenize(prompt)
    text_ids = instruction_ids + text_ids

    if keep_chat_template:
        if hasattr(tokenizer, 'encode'):
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)
            user_ids = tokenizer.encode("user", add_special_tokens=False)
            assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
        else:
            # NullTokenizer 不支持原始文本 tokenize，使用硬编码的 Qwen3 token ID
            nl_ids = [NL_ID]
            user_ids = [USER_ID]
            assistant_ids = [ASSISTANT_ID]

        seq = (
            [IM_START_ID] + user_ids + nl_ids       # <|im_start|>user\n
            + text_ids                                # {prompt}
            + [IM_END_ID] + nl_ids                   # <|im_end|>\n
            + [IM_START_ID] + assistant_ids + nl_ids  # <|im_start|>assistant\n
            + ([V5_IMAGE_GATE_TOKEN_ID]
               if _resolve_predict_gate_mode(tokenizer, predict_gate_mode)
               else [])                              # <|image_gate|>
            + [BOI_ID]                               # <|vision_start|>
            + [TIME_TOKEN_ID]                         # <time_token>
            + [IMG_PAD_ID] * num_image_tokens         # <|image_pad|> × H*W
            + [EOI_ID]                                # <|vision_end|>
            + [EOS_ID]                                # <|im_end|>
        )
    else:
        seq = (
            [BOI_ID]
            + [TIME_TOKEN_ID]
            + [IMG_PAD_ID] * num_image_tokens
            + [EOI_ID]
            + [EOS_ID]
        )

    # seq = (
    #     text_ids                                # {prompt}
    #     + [BOI_ID]                                # <|vision_start|>
    #     + [TIME_TOKEN_ID]                         # <time_token>
    #     + [IMG_PAD_ID] * num_image_tokens         # <|image_pad|> × H*W
    #     + [EOI_ID]                                # <|vision_end|>
    # )

    input_ids = torch.tensor([seq], dtype=torch.long)

    return input_ids


def build_sampling_startend(input_ids: torch.Tensor, device="cpu"):
    """
    构造与训练一致的 startend_row_indices: [1, 1, L, 2]

    与 yichen 推理的 dense attention mask 保持一致: 标准 causal，
    仅 IMG_PAD 连续块内部双向。

    关键区别:
      1. LTS 初始化为 max(L - 1, 0)，与训练端
         create_sample_startend_row_indices 完全一致
      2. TIME token (BOI+1) 不参与图像双向注意力，只有 IMG_PAD tokens 双向
         patch 范围: s+2 .. e-1 (跳过 BOI 和 TIME)
      3. UTE 起始设为 s+2 (第一个 IMG_PAD 位置)

    Args:
        input_ids: [1, L] 或 [L] 的 token ids
        device: 目标设备

    Returns:
        indices: [1, 1, L, 2] FlashMask 格式的 startend_row_indices
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]  # [L]

    input_ids = input_ids.to(device)
    L = len(input_ids)

    # === 1. 初始化: 与训练端 create_sample_startend_row_indices 对齐 ===
    col_idx = torch.arange(L, device=device, dtype=torch.int32)
    # LTS = L - 1 是训练端当前使用的边界；UTE = col_idx 保持 causal。
    LTS = torch.full((L,), max(L - 1, 0), device=device, dtype=torch.int32)
    UTE = col_idx.clone()

    # === 2. 找到图像区域并修改 ===
    starts = (input_ids == BOI_ID).nonzero(as_tuple=True)[0]
    ends = (input_ids == EOI_ID).nonzero(as_tuple=True)[0]

    min_len = min(len(starts), len(ends))

    for k in range(min_len):
        s, e = starts[k], ends[k]
        # s: <boi> index, e: <eoi> index
        # s+1: <time> token — 不参与图像双向注意力
        # s+2 .. e-1: IMG_PAD tokens — 参与双向注意力

        # 图像内部双向注意力: 修改 UTE
        # 与训练一致: 只对 IMG_PAD columns (s+2 .. e-1) 设置双向
        # TIME token (s+1) 保持 causal
        patch_cols_mask = (col_idx > s + 1) & (col_idx < e)
        if patch_cols_mask.any():
            patch_start_idx = (s + 2).int()
            UTE[patch_cols_mask] = patch_start_idx

    # === 3. 格式封装: [1, 1, L, 2] ===
    indices = torch.stack([LTS, UTE], dim=-1).unsqueeze(0).unsqueeze(0)

    return indices


def build_sampling_no_text_startend(input_ids: torch.Tensor, startend: torch.Tensor):
    """Mask text columns from the image rows used by the DiT head.

    This mirrors the inference-time ``nocond`` mask used by the benchmark
    encoder: the backbone still sees the normal conditional sequence, while
    the diffusion head cannot read the preceding text tokens directly.
    """
    if input_ids.dim() != 2 or startend.dim() != 4:
        raise ValueError(
            f"Expected input_ids [B,L] and startend [B,1,L,2], got "
            f"{tuple(input_ids.shape)} and {tuple(startend.shape)}"
        )
    dit_startend = startend.clone()
    col_idx = torch.arange(input_ids.shape[1], device=input_ids.device)
    for batch_idx in range(input_ids.shape[0]):
        starts = (input_ids[batch_idx] == BOI_ID).nonzero(as_tuple=True)[0]
        for start in starts:
            text_cols_mask = col_idx < start
            if text_cols_mask.any():
                current_lts = dit_startend[batch_idx, 0, text_cols_mask, 0]
                dit_startend[batch_idx, 0, text_cols_mask, 0] = torch.minimum(
                    current_lts, start.to(current_lts.dtype)
                )
    return dit_startend


def build_sampling_position_ids(input_ids: torch.Tensor, H: int, W: int, device="cpu"):
    """
    构造 3-channel M-RoPE position_ids (3, 1, L) — 与训练的 _build_position_ids 完全一致。

    通过检测 IMG_PAD 连续块来定位图像区域，而不是依赖传入的索引参数。

    Args:
        input_ids: [1, L] 或 [L] 的 token ids
        H, W: 图像 patch grid 尺寸
        device: 目标设备

    Returns:
        position_ids: (3, 1, L) long tensor
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]  # [L]

    input_ids = input_ids.to(device)
    L = len(input_ids)
    position_ids = torch.zeros(3, 1, L, dtype=torch.long, device=device)

    # 找出所有 IMG_PAD 连续块的 [start, end) 区间
    img_blocks = []
    in_block = False
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

    # 无图像: 全部线性位置
    if not img_blocks:
        pos = torch.arange(L, device=device)
        position_ids[0, 0] = pos
        position_ids[1, 0] = pos
        position_ids[2, 0] = pos
        return position_ids

    pos = 0  # 当前 text position counter
    prev_end = 0
    T = 1  # 采样时 temporal 维度固定为 1

    for blk_start, blk_end in img_blocks:
        # 1. 图像块之前的 text tokens (包括 BOI, TIME 等)
        for i in range(prev_end, blk_start):
            position_ids[:, 0, i] = pos
            pos += 1

        # 2. 图像 tokens: 2D 空间位置
        K = blk_end - blk_start
        # 对于采样，假设单个图像块使用传入的 H, W
        # 如果有多个图像块，后续可以扩展为接受 image_thw_grids 参数
        assert K == H * W, f"Image block size {K} != H*W = {H}*{W} = {H*W}"

        offset = pos
        t_index = torch.zeros(K, dtype=torch.long, device=device)
        h_index = torch.arange(H, device=device).view(-1, 1).expand(-1, W).flatten()
        w_index = torch.arange(W, device=device).view(1, -1).expand(H, -1).flatten()

        position_ids[0, 0, blk_start:blk_end] = offset + t_index
        position_ids[1, 0, blk_start:blk_end] = offset + h_index
        position_ids[2, 0, blk_start:blk_end] = offset + w_index

        pos = offset + max(T, H, W)
        prev_end = blk_end

    # 3. 最后一个图像块之后的 text tokens (EOI + EOS 等)
    for i in range(prev_end, L):
        position_ids[:, 0, i] = pos
        pos += 1

    return position_ids


def _pad_sampling_batch(
    input_ids_list: List[torch.Tensor],
    startend_list: List[torch.Tensor],
    position_ids_list: List[torch.Tensor],
    pad_token_id: int = EOS_ID,
    device=None,
):
    """Pad per-sample sampling metadata into a real batch."""
    max_len = max(ids.shape[-1] for ids in input_ids_list)

    padded_ids = []
    padded_startend = []
    padded_position_ids = []

    for input_ids, startend, position_ids in zip(
        input_ids_list, startend_list, position_ids_list
    ):
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        input_ids = input_ids.to(device) if device is not None else input_ids
        startend = startend.to(device) if device is not None else startend
        position_ids = position_ids.to(device) if device is not None else position_ids

        L = input_ids.shape[0]
        pad_len = max_len - L

        if pad_len > 0:
            pad_ids = torch.full(
                (pad_len,),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad_ids], dim=0)

            last_pos = position_ids[:, :, -1:] + 1
            pad_pos = last_pos + torch.arange(
                pad_len, dtype=position_ids.dtype, device=position_ids.device
            ).view(1, 1, -1)
            position_ids = torch.cat([position_ids, pad_pos], dim=2)

            pad_se = torch.zeros(
                (1, 1, pad_len, 2),
                dtype=startend.dtype,
                device=startend.device,
            )
            pad_se[0, 0, :, 0] = max_len
            pad_se[0, 0, :, 1] = torch.arange(
                L, max_len, dtype=startend.dtype, device=startend.device
            )
            startend = torch.cat([startend, pad_se], dim=2)

        padded_ids.append(input_ids.unsqueeze(0))
        padded_startend.append(startend)
        padded_position_ids.append(position_ids)

    return (
        torch.cat(padded_ids, dim=0),
        torch.cat(padded_startend, dim=0),
        torch.cat(padded_position_ids, dim=1),
    )


def _build_sampling_batch(
    tokenizer,
    prompts: List[str],
    H: int,
    W: int,
    device,
    add_instruction_prefix: bool = False,
    keep_chat_template: bool = True,
    instruction_prefixes: Optional[List[str]] = None,
    predict_gate_mode: Optional[bool] = None,
):
    input_ids_list = []
    startend_list = []
    position_ids_list = []

    if instruction_prefixes is not None and len(instruction_prefixes) != len(prompts):
        raise ValueError(
            f"instruction_prefixes length {len(instruction_prefixes)} "
            f"!= prompts length {len(prompts)}"
        )

    for i, prompt in enumerate(prompts):
        input_ids = build_sampling_sequence(
            tokenizer,
            prompt,
            H,
            W,
            add_instruction_prefix=add_instruction_prefix,
            keep_chat_template=keep_chat_template,
            instruction_prefix=instruction_prefixes[i] if instruction_prefixes is not None else None,
            predict_gate_mode=predict_gate_mode,
        )
        input_ids_list.append(input_ids)
        startend_list.append(build_sampling_startend(input_ids, device=device))
        position_ids_list.append(
            build_sampling_position_ids(input_ids, H, W, device=device)
        )

    return _pad_sampling_batch(
        input_ids_list,
        startend_list,
        position_ids_list,
        pad_token_id=EOS_ID,
        device=device,
    )


# ═══════════════════════════════════════════════════════════════
#  单步预测
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_eps(
    model,
    input_ids,              # [B, L]
    latents,                # [num_img_tokens, latent_ch]
    timesteps,              # [1] scalar tensor
    startend_row_indices,   # [B, 1, L, 2]
    position_ids=None,      # [3, B, L]
    image_grid_thw=None,    # [num_images, 3] raw ViT image grids
    dit_startend_row_indices=None,  # 供 diffusion head 专用的 attention mask; None = 复用 backbone 的
    return_details=False,  # diagnostic: also return backbone image-token features
):
    """Dense Pixel-DiT 的单步 velocity 预测。

    raw RGB 16x16 patches 先按 Qwen3-VL 的 2x2 merge 送入 ViT，随后经过
    Qwen3-VL backbone 和多层 Pixel-DiT head，最后还原为 raw patch 输出。

    Args:
        model: TransfusionQwen3VLModel 或 TransfusionQwen3Model (unwrapped, no DDP)
        input_ids: [B, L] 完整序列 (含 BOI/TIME/IMG_PAD/EOI/EOS)
        latents: [num_img_tokens, latent_ch] 当前噪声状态
        timesteps: [B] 或 [1] 时间步 (0~1000 范围)
        startend_row_indices: [B, 1, L, 2] FlashMask 注意力掩码
        position_ids: [3, B, L] M-RoPE position IDs
        image_grid_thw: [num_images, 3] — raw patch grid (T, H, W)
            (DiT 训练时见的是被 backbone 上下文化的 image 行).

    Returns:
        eps: [num_img_tokens, latent_ch] 预测的 velocity
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    input_ids = input_ids.to(device)
    # latents = torch.load("/path/to/external")
    latents = latents.to(device)
    timesteps = timesteps.to(device)
    startend_row_indices = startend_row_indices.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)

    B, L = input_ids.shape

    # ── FP8 alignment: pad sequence length to multiple of 8 ──
    pad_token_id = EOS_ID  # 使用 EOS 作为 pad token
    L_aligned = (L + 7) // 8 * 8
    pad_len = L_aligned - L
    # if pad_len > 0:
    #     # Pad input_ids
    #     pad_ids = torch.full((B, pad_len), pad_token_id, dtype=input_ids.dtype, device=device)
    #     input_ids = torch.cat([input_ids, pad_ids], dim=1)  # [B, L_aligned]
    #     # Pad position_ids: [3, B, L] -> [3, B, L_aligned]
    #     if position_ids is not None:
    #         last_pos = position_ids[:, :, -1:] + 1  # [3, B, 1], 继续递增
    #         pad_positions = last_pos + torch.arange(pad_len, device=device).view(1, 1, -1)
    #         position_ids = torch.cat([position_ids, pad_positions], dim=2)
    #     # Pad startend_row_indices: [B, 1, L, 2] -> [B, 1, L_aligned, 2]
    #     # 格式: [LTS, UTE]，pad token 使用 causal attention
    #     # LTS = L_aligned (下三角全可见), UTE = col_idx (causal)
    #     pad_startend = torch.zeros((B, 1, pad_len, 2), dtype=startend_row_indices.dtype, device=device)
    #     for i in range(pad_len):
    #         pad_startend[:, 0, i, 0] = L_aligned  # LTS: 下三角全可见
    #         pad_startend[:, 0, i, 1] = L + i       # UTE: causal, 只看自己及之前
    #     startend_row_indices = torch.cat([startend_row_indices, pad_startend], dim=2)
    #     L = L_aligned

    # ── 1. Text embedding ──
    if hasattr(model, 'language_model'):
        combined_embeddings = model.language_model.embedding(
            input_ids=input_ids,
            position_ids=None,
        ).clone()  # [L, B, H]
    else:
        combined_embeddings = model.embedding(
            input_ids=input_ids,
            position_ids=None,
        ).clone()  # [L, B, H]

    # ── 2. Raw pixels → merged ViT features + timestep conditioning ──
    _assert_flashmask_runtime(model, "Pixel-DiT")
    assert image_grid_thw is not None, "Pixel-DiT requires image_grid_thw"
    num_raw_patches = latents.shape[0]
    img_mask = input_ids == IMG_PAD_ID
    timesteps_expanded = timesteps.expand(num_raw_patches)
    model._gen_time_emb = model.time_embeddings(timesteps_expanded)
    model._pixel_dit_gen_time_emb = model._embed_pixel_dit_timesteps(
        timesteps_expanded
    )
    model._pixel_dit_xt_merged = None
    if (
        getattr(model, "use_pixel_dit_pixel_residual", True)
        and getattr(model, "pixel_dit_gate_input", "pixel") == "pixel"
    ):
        model._pixel_dit_xt_merged = model._merge_pixel_2x2(
            latents.to(dtype), image_grid_thw
        )

    raw_sizes = (
        image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
    ).to(device=latents.device, dtype=torch.long)
    if int(raw_sizes.sum().item()) != num_raw_patches:
        raise ValueError(
            f"latents raw token count ({num_raw_patches}) does not match "
            f"image_grid_thw product ({int(raw_sizes.sum().item())})"
        )
    vision_data_list = []
    offset = 0
    for grid, raw_size in zip(image_grid_thw, raw_sizes):
        h_grid, w_grid = int(grid[1].item()), int(grid[2].item())
        n = int(raw_size.item())
        img = latents[offset:offset + n].to(dtype).reshape(
            h_grid, w_grid, 3, 16, 16
        )
        offset += n
        img = img.permute(2, 0, 3, 1, 4).reshape(3, h_grid * 16, w_grid * 16)
        vision_data_i = img.reshape(
            1, 3, h_grid // 2, 2, 16, w_grid // 2, 2, 16
        ).permute(0, 2, 5, 3, 6, 1, 4, 7)
        vision_data_i = vision_data_i.unsqueeze(6).expand(
            -1, -1, -1, -1, -1, -1, 2, -1, -1
        )
        vision_data_list.append(vision_data_i.reshape(h_grid * w_grid, 3 * 2 * 16 * 16))
    vision_data = torch.cat(vision_data_list, dim=0)
    _, latents_proj, _, _, _ = model.vision_model(
        hidden_states=vision_data,
        grid_thw=image_grid_thw,
        patch_embed_output=True,
    )

    # ── 3. 替换图像和时间 token 的 embedding ──
    # combined_embeddings 是 [L, B, H], 转为 [B, L, H] 方便索引
    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
    combined_embeddings[img_mask] = latents_proj.to(combined_embeddings.dtype)

    # ── 4. 替换 TIME token 的 embedding ──
    time_mask = (input_ids == TIME_TOKEN_ID)  # [B, L]
    if time_mask.any():
        num_time_tokens = int(time_mask.sum().item())
        if timesteps.numel() == 1:
            time_token_steps = timesteps.expand(num_time_tokens)
        elif timesteps.numel() == B and num_time_tokens == B:
            time_token_steps = timesteps
        else:
            time_token_steps = timesteps.reshape(-1)[:num_time_tokens]
        time_token_emb = model.time_token(time_token_steps)  # [num_time_tokens, lm_hidden]
        combined_embeddings[time_mask] = time_token_emb.to(combined_embeddings.dtype)

    # 转回 [L, B, H]
    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

    # ── 5. LM backbone forward ──
    extra_block_kwargs = {
        "startend_row_indices": startend_row_indices,
        "causal": False,
    }

    if hasattr(model, 'language_model'):
        # TransfusionQwen3VLModel: 通过 language_model (GPTModel) forward
        if hasattr(model, "_captured_hidden_states"):
            model._captured_hidden_states = None
        try:
            lm_output = model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=None,
                decoder_input=combined_embeddings,
                labels=None,
                inference_params=None,
                packed_seq_params=None,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
                extra_block_kwargs=extra_block_kwargs,
            )
        finally:
            pass
        # VL 模型通过 hook 捕获 hidden_states
        hidden_states = model._captured_hidden_states
        if hidden_states is None:
            raise AssertionError(
                f"Hidden states hook did not fire.\n"
                f"  model type: {type(model).__name__}\n"
                f"  Hint: hook is only registered when post_process=True"
            )
        config = model.language_model.config
    else:
        # TransfusionQwen3Model: 直接调用内部组件，不走 training forward()
        # 1) _preprocess: 计算 rotary embeddings (M-RoPE)
        preproc_output = model._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=combined_embeddings,
        )
        (decoder_input_processed, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = (
            preproc_output[:5]
        )
        rotary_pos_cos_sin = preproc_output[5] if len(preproc_output) == 6 else None

        # 2) decoder: Transformer blocks
        try:
            hidden_states = model.decoder(
                hidden_states=decoder_input_processed,
                attention_mask=None,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                sequence_len_offset=sequence_len_offset,
                **(extra_block_kwargs or {}),
            )
        finally:
            pass
        config = model.config

    # SP 模式下 hidden_states 可能是 scattered 的，需要 all-gather
    if config.sequence_parallel:
        from megatron.core import tensor_parallel
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False
        )

    # 确保 hidden_states 为 [B, L, H]
    if hidden_states.shape[0] != img_mask.shape[0]:
        hidden_states = hidden_states.transpose(0, 1).contiguous()

    if hidden_states.shape[1] != img_mask.shape[1]:
        hidden_states = hidden_states[:, :img_mask.shape[1]]

    hidden_states = hidden_states.to(dtype=torch.float32)

    img_hidden = hidden_states[img_mask]  # [num_img_tokens, H]
    # The final-layer-normalized image-token states are the shared-backbone
    # representation consumed by the diffusion head.  Keep this optional so
    # ordinary benchmark inference has exactly the previous memory behavior.
    backbone_image_feature = img_hidden.detach() if return_details else None

    # ── Pixel-DiT: 多层 DiT head + merged pixel Gate 注入 → raw patches ──
    if True:
        gen_token_mask = img_mask
        head_dtype = next(model.diffusion_head_a[0].parameters()).dtype
        diff_hidden = model._project_pixel_dit_hidden(hidden_states).to(head_dtype)

        # per-token time_emb 需 merged 长度 (DiT 在 merged 分辨率跑)。
        # model._gen_time_emb 是 raw 长度, 需降采样: 同图同 timestep,
        # 每 2x2 block 取起始 token 的 emb, 再按 merged token 数铺开。
        MS = 2
        raw_sizes = (image_grid_thw[:, 0] * image_grid_thw[:, 1]
                     * image_grid_thw[:, 2])
        merged_sizes = raw_sizes // (MS * MS)
        pixel_dit_time_emb = model._pixel_dit_gen_time_emb
        if pixel_dit_time_emb is None:
            raise RuntimeError("Pixel-DiT timestep embedding was not initialized")
        _dev_t = pixel_dit_time_emb.device
        raw_cumsum = torch.cat([
            torch.zeros(1, device=_dev_t, dtype=torch.long),
            raw_sizes.cumsum(0)[:-1],
        ])
        per_img_time_emb = pixel_dit_time_emb[raw_cumsum]  # [N_img, H]
        merged_img_idx = torch.arange(
            len(merged_sizes), device=_dev_t
        ).repeat_interleave(merged_sizes)
        gen_time_emb = per_img_time_emb[merged_img_idx].to(head_dtype)  # [N_merged, H]
        # HiGate high-info input: merged noisy pixels or post-merger ViT feat.
        # The residual ablation keeps modules loaded but skips gate injection.
        gate_input = None
        if getattr(model, "use_pixel_dit_pixel_residual", True):
            gate_input = model._prepare_pixel_dit_gate_input(
                pixel_features=model._pixel_dit_xt_merged,
                vit_features=latents_proj,
                expected_tokens=int(gen_token_mask.sum().item()),
                name="generation",
            )
        head_startend = (
            dit_startend_row_indices
            if dit_startend_row_indices is not None
            else startend_row_indices
        )
        inject_layer = getattr(model, 'pixel_dit_gate_inject_layer', 0)
        for i, layer in enumerate(model.diffusion_head_a):
            if i == inject_layer and gate_input is not None:
                fused = model.pixel_dit_gate(
                    diff_hidden[gen_token_mask], gate_input
                )
                diff_hidden[gen_token_mask] = fused.to(diff_hidden.dtype)
            diff_hidden = layer(
                hidden_states=diff_hidden,
                gen_time_emb=gen_time_emb,
                gen_token_mask=gen_token_mask,
                attention_mask=None,
                position_ids=position_ids,
                startend_row_indices=head_startend,
            )[0]
        x0_full = model.diffusion_head_b(diff_hidden, gen_time_emb, gen_token_mask)
        eps_merged = x0_full[gen_token_mask]  # [N_gen, 3072]
        eps = model._unmerge_pixel_2x2(eps_merged, image_grid_thw)  # [N_raw, 768]


    # 清理 VL 模型的 hook 缓存 (TransfusionQwen3Model 不使用 hook)
    if hasattr(model, '_captured_hidden_states'):
        model._captured_hidden_states = None

    eps = eps.float()
    if return_details:
        return eps, {"backbone_image_feature": backbone_image_feature}
    return eps


# ═══════════════════════════════════════════════════════════════
#  完整采样
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_images(
    model,
    tokenizer,
    prompts: List[str],
    H: int = 16,
    W: int = 16,
    num_steps: int = 25,
    shift: float = 3.0,
    device=None,
    guidance_scale: float = 1.0,
    negative_prompt: str = "",
    predict_x0: bool = False,
    dynamic_noise_scale: bool = False,
    dynamic_noise_ref_tokens: int = 256,
    dynamic_noise_base_scale: float = 1.0,
    scheduler_type: str = "euler",
    timestep_sampling: str = "uniform",
    logit_norm_mu: float = 0.8,
    logit_norm_sigma: float = 0.8,
    drop_dit_text: bool = False,
    cfg_keep_chat_template: bool = False,
    predict_gate_mode: Optional[bool] = None,
    step_callback=None,
):
    """
    Args:
        model: TransfusionQwen3VLModel (unwrapped)
        tokenizer: HuggingFace tokenizer
        prompts: list of text prompts
        H, W: patch grid size (e.g., 16x16 for 256x256 images with patch_size=16)
        num_steps: denoising steps
        shift: flow matching shift parameter (与训练一致)
        device: target device
        guidance_scale: CFG guidance scale (1.0 = no guidance)
        negative_prompt: negative prompt for CFG (empty string for unconditional)
        cfg_keep_chat_template: when True, CFG uncond keeps the Qwen chat
            template and generation instruction prefix; when False, it uses a
            pure image-only uncond sequence.
        predict_gate_mode: None auto-detects the V5 image route token; True
            requires its fixed single-ID mapping; False disables the token.

    Returns:
        images: (B, 3, H*32, W*32) tensor in [0,1]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    latent_ch = 3 * 16 * 16
    grid_thw = torch.tensor([[1, 2 * H, 2 * W]], device=device, dtype=torch.long)
    num_latent_tokens = 4 * H * W

    # 初始化 scheduler
    scheduler = FlowMatchScheduler(
        num_train_timesteps=1000,
        num_inference_steps=num_steps,
        shift=shift,
        extra_one_step=True,
    )
    scheduler.set_timesteps(num_steps)
    if timestep_sampling == "shift_logit_norm":
        # Use a deterministic, descending inference grid sampled from the
        # training logit-normal timestep distribution.
        mu = float(logit_norm_mu)
        sigma = float(logit_norm_sigma)
        u = torch.randn(num_steps, device=device)
        sampled = torch.sigmoid(mu + sigma * u).clamp(
            min=float(scheduler.sigma_min), max=float(scheduler.sigma_max)
        )
        scheduler.sigmas = torch.sort(sampled, descending=True).values
        scheduler.timesteps = scheduler.sigmas * scheduler.num_train_timesteps
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    timesteps = scheduler.timesteps

    # CFG: 是否启用 classifier-free guidance
    do_cfg = guidance_scale > 1.0
    batch_size = len(prompts)
    instruction_prefixes = [random.choice(GEN_INSTRUCTION_PROMPTS) for _ in prompts]

    input_ids, startend, position_ids = _build_sampling_batch(
        tokenizer,
        prompts,
        H,
        W,
        device=device,
        add_instruction_prefix=True,
        instruction_prefixes=instruction_prefixes,
        predict_gate_mode=predict_gate_mode,
    )
    if do_cfg:
        actual_negative_prompt = negative_prompt if negative_prompt else NEGATIVE_PROMPT
        uncond_prompts = [actual_negative_prompt] * batch_size
        uncond_ids, uncond_startend, uncond_position_ids = _build_sampling_batch(
            tokenizer,
            uncond_prompts,
            H,
            W,
            device=device,
            add_instruction_prefix=cfg_keep_chat_template,
            keep_chat_template=cfg_keep_chat_template,
            instruction_prefixes=instruction_prefixes if cfg_keep_chat_template else None,
            # Image-only unconditional sequences have no assistant header and
            # therefore must not contain a route token.
            predict_gate_mode=False,
        )

    dit_startend = None
    uncond_dit_startend = None
    if drop_dit_text:
        dit_startend = build_sampling_no_text_startend(input_ids, startend)
        if do_cfg:
            uncond_dit_startend = build_sampling_no_text_startend(
                uncond_ids, uncond_startend
            )

    grid_thw = torch.tensor(
        [[1, 2 * H, 2 * W]] * batch_size,
        device=device,
        dtype=torch.long,
    )

    num_img_tokens = H * W
    noise_tokens_per_image = num_latent_tokens
    latents = torch.randn(
        batch_size * noise_tokens_per_image,
        latent_ch,
        device=device,
        dtype=torch.float32,
    )

    # Dynamic noise scale
    if dynamic_noise_scale:
        sigma_r = (
            math.sqrt(noise_tokens_per_image / dynamic_noise_ref_tokens)
            * dynamic_noise_base_scale
        )
        latents = latents * sigma_r

    # ── 去噪循环 ──
    # DPM-Solver++ multistep needs history of x0 predictions
    x0_history = []  # stores (lambda_t, x0_pred) for multistep

    for step_idx, t in enumerate(timesteps):
        t_tensor = t.to(device).unsqueeze(0)  # [1], each image shares the timestep
        step_details = None

        if do_cfg:
            # CFG: 分别预测 cond 和 uncond
            eps_cond = predict_eps(
                model=model,
                input_ids=input_ids,
                latents=latents,
                timesteps=t_tensor,
                startend_row_indices=startend,
                position_ids=position_ids,
                image_grid_thw=grid_thw,
                dit_startend_row_indices=dit_startend,
                return_details=step_callback is not None,
            )
            if step_callback is not None:
                eps_cond, step_details = eps_cond
            eps_uncond = predict_eps(
                model=model,
                input_ids=uncond_ids,
                latents=latents,
                timesteps=t_tensor,
                startend_row_indices=uncond_startend,
                position_ids=uncond_position_ids,
                image_grid_thw=grid_thw,
                dit_startend_row_indices=uncond_dit_startend,
            )
            # CFG guidance: eps = eps_uncond + guidance * (eps_cond - eps_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = predict_eps(
                model=model,
                input_ids=input_ids,
                latents=latents,
                timesteps=t_tensor,
                startend_row_indices=startend,
                position_ids=position_ids,
                image_grid_thw=grid_thw,
                dit_startend_row_indices=dit_startend,
                return_details=step_callback is not None,
            )
            if step_callback is not None:
                eps, step_details = eps

        # Convert the model output to x0 space before updating the latent.
        # This is the image that should be shown to the understanding model at
        # this denoising step, independent of Euler/DPM scheduler choice.
        timestep_id = torch.argmin(
            (scheduler.timesteps - t.to(scheduler.timesteps.device)).abs()
        )
        sigma = scheduler.sigmas[timestep_id]
        x0_pred = eps if predict_x0 else latents - sigma * eps

        if scheduler_type == "euler":
            # Euler step (1st order)
            latents = scheduler.step(
                eps,
                t,
                latents,
                predict_x0=predict_x0,
                predict_x0_sigma_min=None,
            )
        else:
            # DPM-Solver++ 2nd/3rd order
            # Convert eps to x0 prediction
            timestep_id = torch.argmin(
                (scheduler.timesteps - t.to(scheduler.timesteps.device)).abs()
            )
            sigma = scheduler.sigmas[timestep_id]
            if predict_x0:
                x0_pred = eps  # model directly predicts x0
            else:
                # eps-prediction: x0 = x_t - sigma * eps_pred (flow matching)
                x0_pred = latents - sigma * eps

            # Get sigma_next
            if timestep_id + 1 >= len(scheduler.timesteps):
                sigma_next = torch.zeros_like(sigma)
            else:
                sigma_next = scheduler.sigmas[timestep_id + 1]

            # Log-lambda parameterization: lambda = log(alpha/sigma) = log((1-sigma)/sigma)
            eps_safe = 1e-8
            lam = torch.log((1 - sigma).clamp(min=eps_safe) / sigma.clamp(min=eps_safe))
            lam_next = torch.log((1 - sigma_next).clamp(min=eps_safe) / sigma_next.clamp(min=eps_safe))
            h = lam_next - lam

            x0_history.append((lam.item(), x0_pred))

            order = 2 if scheduler_type == "dpm2" else 3
            # Effective order for this step (limited by available history)
            eff_order = min(order, len(x0_history))

            if eff_order == 1:
                # 1st order: x_{t-1} = (1-sigma_next)*x0 + sigma_next * (x_t - (1-sigma)*x0) / sigma
                latents = (1 - sigma_next) * x0_pred + \
                    sigma_next * (latents - (1 - sigma) * x0_pred) / sigma.clamp(min=eps_safe)
            elif eff_order == 2:
                # DPM-Solver++ 2nd order multistep
                lam_prev, x0_prev = x0_history[-2]
                h_prev = lam.item() - lam_prev
                r = h.item() / h_prev if h_prev != 0 else 0.0
                D0 = x0_pred
                D1 = (1.0 / (2.0 * r)) * (x0_pred - x0_prev)
                x0_corrected = D0 + D1
                latents = (1 - sigma_next) * x0_corrected + \
                    sigma_next * (latents - (1 - sigma) * x0_pred) / sigma.clamp(min=eps_safe)
            else:
                # DPM-Solver++ 3rd order multistep
                lam_2, x0_2 = x0_history[-2]
                lam_3, x0_3 = x0_history[-3]
                h_1 = lam.item() - lam_2
                h_2 = lam_2 - lam_3
                r1 = h.item() / h_1 if h_1 != 0 else 0.0
                r2 = h_1 / h_2 if h_2 != 0 else 0.0
                D0 = x0_pred
                D1_0 = (1.0 / r1) * (x0_pred - x0_2)
                D1_1 = (1.0 / r2) * (x0_2 - x0_3)
                D1 = D1_0 + (r1 / (r1 + r2)) * (D1_0 - D1_1)
                D2 = (1.0 / (r1 + r2)) * (D1_0 - D1_1)
                x0_corrected = D0 + D1 + D2
                latents = (1 - sigma_next) * x0_corrected + \
                    sigma_next * (latents - (1 - sigma) * x0_pred) / sigma.clamp(min=eps_safe)

            # Keep history bounded
            if len(x0_history) > 3:
                x0_history.pop(0)

        if step_callback is not None:
            step_callback(
                step_idx=int(step_idx),
                timestep=float(t.item() if torch.is_tensor(t) else t),
                sigma=float(sigma.item()),
                x0_pred=x0_pred.detach(),
                latent_after_step=latents.detach(),
                details=step_details,
            )

        if step_idx % 5 == 0 or step_idx == len(timesteps) - 1:
            logger.info(f"[denoise] step {step_idx}/{len(timesteps)} t={t:.1f} | "
                        f"batch={batch_size} | "
                        f"latents: mean={latents.mean():.4f} std={latents.std():.4f} | "
                        f"eps: mean={eps.mean():.4f} std={eps.std():.4f}")

    all_latents = latents  # [B*H*W, latent_ch] or [B*4*H*W, 768]

    logger.info(f"[sampling debug] raw latents: mean={all_latents.mean():.4f}, std={all_latents.std():.4f}, "
                f"min={all_latents.min():.4f}, max={all_latents.max():.4f}")

    raw_h, raw_w = 2 * H, 2 * W
    tokens_per_image = raw_h * raw_w
    images = []
    for i in range(len(prompts)):
        z = all_latents[i * tokens_per_image:(i + 1) * tokens_per_image]
        z = z.reshape(raw_h, raw_w, 3, 16, 16)
        z = z.permute(2, 0, 3, 1, 4).reshape(3, raw_h * 16, raw_w * 16)
        images.append((z * 0.5 + 0.5).unsqueeze(0))
    return torch.cat(images, dim=0).clamp(0, 1)


# ═══════════════════════════════════════════════════════════════
#  便捷入口: 训练中定期调用 (直接用训练 batch 数据)
# ═══════════════════════════════════════════════════════════════

def extract_prompts_from_batch(input_ids, tokenizer, num_vis=4):
    """
    从训练 batch 的 input_ids 中提取文本 prompt（仅用户内容部分）。

    训练格式:
        <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<BOI>...

    本函数提取 {content} 部分（不含 chat template 标记），
    避免与 build_sampling_sequence 的 template 重复嵌套。

    返回:
        prompts: List[str] — 解码后的文本（用于日志/可视化）
        prompt_token_ids: List[List[int]] — 原始 token id 列表（用于 build_sampling_sequence，
                          避免 BPE 重编码导致的 token 差异）

    对 packed 序列 (可能包含多个样本), 找到每个 BOI 前的文字段并 decode。
    返回最多 num_vis 个 prompt。
    """
    prompts = []
    prompt_token_ids = []
    # 只取第一行 (packed row)
    ids = input_ids[0].tolist()

    boi_positions = [i for i, x in enumerate(ids) if x == BOI_ID]
    seg_start = 0
    for boi_pos in boi_positions:
        if len(prompts) >= num_vis:
            break
        # 从 seg_start 到 boi_pos 之间的 token 就是文本区域
        text_region = ids[seg_start:boi_pos]

        # 精确提取用户内容: 跳过 <|im_start|>user\n 和 <|im_end|>\n<|im_start|>assistant\n
        # 找到 text_region 中第一个 IM_START + user + \n 之后的位置
        content_start = 0
        content_end = len(text_region)

        # 在 text_region 中查找 <|im_start|> (IM_START_ID=151644)
        # 第一个 <|im_start|> 后面跟 user\n, 内容从 user\n 之后开始
        for k, tok in enumerate(text_region):
            if tok == IM_START_ID:
                # 跳过 <|im_start|>, user, \n
                # 找到下一个 \n (user\n 的 \n)
                # Qwen3 tokenizer 的 \n token id
                nl_id = tokenizer.encode("\n", add_special_tokens=False)[0] if hasattr(tokenizer, 'encode') else None
                if nl_id is not None:
                    for m in range(k + 1, len(text_region)):
                        if text_region[m] == nl_id:
                            content_start = m + 1
                            break
                break

        # 内容到 <|im_end|> 结束
        for k in range(content_start, len(text_region)):
            if text_region[k] == IM_END_ID:
                content_end = k
                break

        content_ids = text_region[content_start:content_end]
        if content_ids:
            if hasattr(tokenizer, 'decode'):
                prompt = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
            else:
                prompt = tokenizer.detokenize(content_ids).strip()
            if prompt:
                prompts.append(prompt)
                prompt_token_ids.append(content_ids)
        # 跳过当前图像块: 找到 EOI 后面
        eoi_pos = None
        for j in range(boi_pos, len(ids)):
            if ids[j] == EOI_ID:
                eoi_pos = j
                break
        if eoi_pos is not None:
            seg_start = eoi_pos + 1
            # 跳过 EOS
            if seg_start < len(ids) and ids[seg_start] == EOS_ID:
                seg_start += 1
        else:
            break
    return prompts, prompt_token_ids


def extract_image_grid_info(image_grid_thw, num_vis=4):
    """从 image_grid_thw 提取前 num_vis 个图像的 (T, H, W) 信息。"""
    results = []
    for i in range(min(image_grid_thw.shape[0], num_vis)):
        T = int(image_grid_thw[i, 0])
        H = int(image_grid_thw[i, 1])
        W = int(image_grid_thw[i, 2])
        results.append((T, H, W))
    return results


@torch.no_grad()
def get_original_images(pixel_values, image_grid_thw, device, num_vis=4):
    """Recover raw input images for training-time TensorBoard previews."""
    if pixel_values is None or image_grid_thw is None:
        return None, []
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    grids = extract_image_grid_info(image_grid_thw, num_vis=num_vis)
    images = [
        _pixel_values_to_image(pixel_values, image_grid_thw, image_idx=i).to(device)
        for i in range(len(grids))
    ]
    return images, grids



def sample_and_log_from_batch(
    model,
    tokenizer,
    batch: dict,
    tb_writer,
    iteration: int,
    num_vis: int = 4,
    num_steps: int = 25,
    shift: float = 3.0,
    guidance_scale: float = 1.0,
    predict_x0: bool = False,
):
    """
    直接用训练 batch 数据进行可视化:
      1. pixel_values → raw pixel image → 原图
      2. 从 input_ids 提取 text prompt → Dense Pixel-DiT 去噪采样 → 生成图
      3. 两者一起写入 TensorBoard

    Args:
        model: TransfusionQwen3VLModel (unwrapped)
        tokenizer: tokenizer
        batch: 训练 data_batch dict
        tb_writer: TensorBoard SummaryWriter
        iteration: 当前训练步数
        num_vis: 可视化图片数量
        num_steps: 去噪步数
        shift: flow matching shift
    """
    if tb_writer is None:
        return

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    try:
        input_ids = batch["input_ids"]
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")

        # ── 1. 原图: pixel_values → ViT → clean RGB patches ──
        original_images, grids = get_original_images(
            pixel_values, image_grid_thw, device, num_vis=num_vis
        )
        if original_images is not None:
            if isinstance(original_images, list):
                # 不同分辨率的图片列表，逐张写 TensorBoard
                for i, img in enumerate(original_images):
                    tb_writer.add_image(f"samples/original_{i}", img[0] if img.dim() == 4 else img, iteration)
            else:
                for i in range(original_images.shape[0]):
                    tb_writer.add_image(f"samples/original_{i}", original_images[i], iteration)

        # ── 2. 提取 text prompt ──
        prompts, prompt_token_ids = extract_prompts_from_batch(input_ids, tokenizer, num_vis=num_vis)
        if not prompts:
            logger.warning("No prompts extracted from batch, skipping generation")
            return

        for i, prompt in enumerate(prompts):
            tb_writer.add_text(f"samples/prompt_{i}", prompt, iteration)

        # ── 3. 生成图: 用相同 prompt 进行 Dense Pixel-DiT 去噪 ──
        # 确定每个 prompt 对应的 H, W (从 grid 信息获取)
        # 使用原始 token ids (而非解码后重编码) 避免 BPE 边界差异
        for i, prompt in enumerate(prompts):
            if i < len(grids):
                T, H, W = grids[i]
            else:
                H, W = 16, 16  # fallback

            # 优先使用原始 token ids，避免 decode→re-encode 的 BPE 差异
            raw_ids = prompt_token_ids[i] if i < len(prompt_token_ids) else prompt
            gen_images = sample_images(
                model=model,
                tokenizer=tokenizer,
                prompts=[raw_ids],
                H=H, W=W,
                num_steps=num_steps,
                shift=shift,
                device=device,
                guidance_scale=guidance_scale,
                predict_x0=predict_x0,
            )
            if gen_images is not None and gen_images.dim() == 4:
                logger.info(f"[sample_and_log] generated image {i}: shape={gen_images.shape}, "
                            f"dtype={gen_images.dtype}, "
                            f"mean={gen_images[0].float().mean():.4f}, "
                            f"min={gen_images[0].min():.4f}, max={gen_images[0].max():.4f}")
                tb_writer.add_image(f"samples/generated_{i}", gen_images[0], iteration)

        tb_writer.flush()
        logger.info(f"Logged {len(prompts)} original + generated images to TensorBoard at iter {iteration}")

    except Exception as e:
        logger.error(f"Sampling from batch failed at iteration {iteration}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if was_training:
            model.train()
