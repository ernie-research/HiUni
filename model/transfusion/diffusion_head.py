"""Multi-layer Diffusion Head for Transfusion Qwen3-VL.

Ported from tuna-2/models/backbones/modules_new_mm.py and adapted for the
HiUni / Megatron-based training pipeline.

Key adaptations vs. tuna-2:
- ``adaln_input`` is **per-token** ``[N_gen, H]`` (already expanded by
  ``TimestepEmbedder`` in the model), NOT per-image ``[B*num_imgs, H]``.
  This matches ``self._gen_time_emb`` which is produced as
  ``time_embeddings(timesteps_packed)`` where ``timesteps_packed`` is
  already repeated per token.
- Modulation scatter uses a boolean ``gen_token_mask`` of shape ``[B, S]``
  (True at generation image token positions) instead of ``modality_positions``
  ``[B, num_imgs, 2]``, because the latter is never built in training_step.
- ``position_ids`` supports Qwen3-VL M-RoPE format ``[3, B, S]``
  (temporal, height, width), directly reusing the backbone's 3-D position IDs.
  Text tokens have all 3 channels equal (degenerates to 1-D); image tokens
  carry spatial position information via the H/W channels.

Public API
----------
``DiffHeadConfig``            – standalone config (no Megatron dependency).
``ModulatedAttentionBlock``   – adaLN-Zero conditioned GQA + SwiGLU block.
``DiffusionHeadFinalLayer``   – adaLN-Zero modulated RMSNorm + linear head.

Usage inside TransfusionQwen3VLModel (post_process stage)::

    # init
    self.diffusion_head_a = nn.ModuleList([
        ModulatedAttentionBlock(diff_head_cfg, i)
        for i in range(num_diffusion_layers)
    ])
    self.diffusion_head_b = DiffusionHeadFinalLayer(lm_dim, diff_out_dim)

    # forward  (hidden_states: [B, S, H], already gathered from SP)
    #   gen_token_mask : [B, S] bool  – True at generation image token positions
    #   gen_time_emb   : [total_gen_tokens, H]  – per-token timestep embedding
    #   position_ids   : [3, B, S] – backbone M-RoPE position IDs
    for layer in self.diffusion_head_a:
        hidden_states = layer(
            hidden_states=hidden_states,
            gen_time_emb=gen_time_emb,
            gen_token_mask=gen_token_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]
    x0_full = self.diffusion_head_b(hidden_states, gen_time_emb, gen_token_mask)
    image_pred = x0_full[gen_token_mask]          # [total_gen_tokens, out_channels]
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, n_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, n_kv, n_rep, slen, head_dim)
        .reshape(batch, n_kv * n_rep, slen, head_dim)
    )


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(1 + scale) * x + shift, computed in fp32 for numerical stability."""
    input_dtype = x.dtype
    return (
        x.float() * (1 + scale.float()) + shift.float()
    ).to(input_dtype)


def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
    """Merge 3-channel M-RoPE freqs into single tensor via interleaved pattern.

    Same algorithm as Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope.

    Args:
        freqs: [3, B, S, head_dim//2] – per-channel rotation frequencies.
        mrope_section: [3] – e.g. [24, 20, 20], dimensions per channel.

    Returns:
        [B, S, head_dim//2] – interleaved merged frequencies.
    """
    freqs_t = freqs[0].clone()  # start with temporal as base
    for dim, offset in enumerate((1, 2), start=1):  # H, W channels
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DiffHeadRMSNorm(nn.Module):
    """RMSNorm with learnable weight (no Megatron dependency)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """对输入做 RMS 归一化后乘以可学习权重。

        Args:
            hidden_states: 任意形状的输入张量，最后一维为 hidden_size。

        Returns:
            与输入同形状的归一化结果，dtype 与输入保持一致。
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DiffHeadConfig(PretrainedConfig):
    """Config for the multi-layer Diffusion Head.

    Mirrors tuna-2's DiffusionHeadConfig.
    Defaults are sized for Qwen3-7B (hidden_size=3584).
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        intermediate_size: int = 18944,
        max_position_embeddings: int = 131072,
    ):
        self.attn_implementation = "sdpa"
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = "silu"
        self.rms_norm_eps = 1e-05
        self.mlp_bias = False
        self.head_dim = hidden_size // num_attention_heads
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.rope_theta = 5000000.0  # 与 Qwen3-VL backbone 一致 (5M)
        self.rope_scaling = {
            "rope_type": "default",    # 与 backbone 一致，无 wavelength scaling
            "mrope_section": [24, 20, 20],
        }
        self.qk_norm = True
        super().__init__()


class _DiffHeadRotaryEmbedding(nn.Module):
    """RoPE embedding supporting both 1-D [B, S] and 3-D M-RoPE [3, B, S] position_ids."""

    # Qwen3-VL default mrope_section
    MROPE_SECTION = [24, 20, 20]

    def __init__(self, config: DiffHeadConfig):
        super().__init__()
        rope_type = config.rope_scaling.get("rope_type", "default")
        self.config = config
        if rope_type == "default":
            # Transformers 5 no longer exposes the original implementation in
            # ROPE_INIT_FUNCTIONS. Keep the standard Qwen RoPE computation
            # local so old and new Transformers releases behave identically.
            dim = config.head_dim
            inv_freq = 1.0 / (
                config.rope_theta
                ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            self.attention_scaling = 1.0
        else:
            init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = init_fn(config, None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """计算 RoPE cos/sin 旋转系数，支持 1-D 和 3-D (M-RoPE) position_ids。

        Args:
            x:            参考张量，仅用于获取目标 dtype。
            position_ids: 位置索引:
                          - ``[B, S]`` 或 ``[1, S]``: 标准 1-D RoPE
                          - ``[3, B, S]``: Qwen3-VL M-RoPE (temporal, height, width)

        Returns:
            cos: shape ``[B, S, head_dim]``
            sin: shape ``[B, S, head_dim]``
        """
        device_type = x.device.type if x.device.type != "mps" else "cpu"

        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            # M-RoPE mode: position_ids [3, B, S]
            # 按照 Qwen3VLTextRotaryEmbedding 的方式计算 interleaved mrope
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
                3, position_ids.shape[1], -1, 1
            )  # [3, B, dim, 1]
            pos_expanded = position_ids[:, :, None, :].float()  # [3, B, 1, S]
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ pos_expanded.float()).transpose(2, 3)
                # freqs: [3, B, S, dim]
                freqs = _apply_interleaved_mrope(freqs, self.MROPE_SECTION)
                # freqs: [B, S, dim]
                emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, head_dim]
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
        else:
            # Standard 1-D mode: position_ids [B, S] or [1, S]
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            )
            pos_expanded = position_ids[:, None, :].float()
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ pos_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling

        return cos.to(x.dtype), sin.to(x.dtype)


class _DiffHeadMLP(nn.Module):
    def __init__(self, config: DiffHeadConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU 前馈网络：gate_proj × up_proj 经 SiLU 激活后由 down_proj 投影回原维度。

        Args:
            x: 输入张量，shape ``[..., hidden_size]``。

        Returns:
            与输入同形状的输出张量。
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _DiffHeadSdpaAttention(nn.Module):
    """GQA attention with M-RoPE + FlashMask support (same as Qwen3-VL backbone)."""

    def __init__(self, config: DiffHeadConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        if getattr(config, "qk_norm", False):
            self.q_norm = DiffHeadRMSNorm(self.head_dim)
            self.k_norm = DiffHeadRMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rotary_emb = _DiffHeadRotaryEmbedding(config)

        # FlashMask (https://github.com/PaddlePaddle/FlashMask, FLASHMASK_FRAMEWORK=torch).
        # Without it, attention falls back to SDPA with a dense mask.
        self._flashmask_attention_fn = None
        try:
            from flashmask.frontend.torch import flashmask_attention
            self._flashmask_attention_fn = flashmask_attention
        except (ImportError, RuntimeError, OSError):
            pass

        if layer_idx == 0:
            if self._flashmask_attention_fn is not None:
                print("[DiffHead] FlashMask available, will use flashmask_attention for packing isolation")
            else:
                print("[DiffHead] FlashMask NOT available, falling back to PyTorch SDPA (no packing isolation!)")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        startend_row_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None]:
        """GQA 自注意力前向，支持 M-RoPE + FlashMask。

        Args:
            hidden_states:        输入序列，shape ``[B, S, H]``。
            attention_mask:       加性 attention mask ``[B, 1, S, S]`` 或 None (fallback)。
            position_ids:         ``[B, S]``/``[1, S]`` (1-D) 或 ``[3, B, S]`` (M-RoPE)。
            startend_row_indices: ``[B, 1, S, 2]`` FlashMask 格式 (优先使用，无需 dense mask)。

        Returns:
            Tuple: (attention output [B, S, H], None, None)
        """
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        if startend_row_indices is not None and self._flashmask_attention_fn is not None:
            # FlashMask path: O(S) 显存，直接使用 startend_row_indices
            if not hasattr(self, "_logged_flashmask"):
                print(f"[DiffHead] Using FlashMask attention (seq_len={q_len}, heads={self.num_heads})")
                self._logged_flashmask = True
            # q/k/v: [B, H, S, Dh] -> [B, S, H, Dh] (flashmask_attention 需要 BSHD layout)
            q_fm = q.transpose(1, 2).contiguous()
            k_fm = k.transpose(1, 2).contiguous()
            v_fm = v.transpose(1, 2).contiguous()
            if self.training and self.attention_dropout > 0:
                raise ValueError("FlashMask does not support attention dropout")
            out = self._flashmask_attention_fn(
                q_fm, k_fm, v_fm,
                startend_row_indices=startend_row_indices.to(torch.int32),
                causal=False,
                softmax_scale=self.head_dim ** -0.5,
            )
            # out: [B, S, H, Dh] -> [B, S, H*Dh]
            out = out.contiguous().view(bsz, q_len, -1)
        else:
            # Fallback: PyTorch SDPA (用于无 flashmask 环境或 startend 为 None)
            if not hasattr(self, "_logged_sdpa_fallback"):
                reason = "no startend_row_indices" if startend_row_indices is None else "flashmask not available"
                print(f"[DiffHead] WARNING: Falling back to SDPA ({reason}), no packing isolation!")
                self._logged_sdpa_fallback = True
            is_causal = attention_mask is None and q_len > 1
            out = F.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        return self.o_proj(out), None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ModulatedAttentionBlock(nn.Module):
    """adaLN-Zero conditioned transformer block for the diffusion head.

    Conditioning design (adapted for HiUni data flow):
    - ``gen_time_emb``  : ``[total_gen_tokens, H]`` – per-token timestep
      embedding produced by ``TimestepEmbedder(timesteps_packed)``.
    - ``gen_token_mask``: ``[B, S]`` bool – True at generation image token
      positions; used to scatter per-token modulation into the full sequence.

    Text and understanding-image positions keep identity modulation
    (zero shift/scale, unit gate), so gradients do not flow into them
    from the adaLN path.
    """

    def __init__(self, config: DiffHeadConfig, layer_idx: int, adaln_post_modulation: bool = False,
                 block_adaln: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _DiffHeadSdpaAttention(config=config, layer_idx=layer_idx)
        self.mlp = _DiffHeadMLP(config)
        self.input_layernorm = DiffHeadRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffHeadRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # block_adaln=False: transformer block 内不做任何 adaLN 调制 (纯 pre-norm
        # transformer, 无 time/gate 条件), 时间条件仅由 head 最后一层的 adaLN 注入。
        # 用于 "no adaln in transformer block" 变体; 默认 True 保持原状。
        self.block_adaln = bool(block_adaln)
        # post-modulation (NVlabs PixelDiT #6): 4-way scale/shift 施加在 attn/mlp
        # 输出上 (无 gate), 缓解训练 spike; 默认 False 保持原 6-way pre-mod。
        self.adaln_post_modulation = bool(adaln_post_modulation)
        if self.block_adaln:
            n_mod = 4 if self.adaln_post_modulation else 6
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, n_mod * config.hidden_size, bias=True),
            )
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,        # [B, S, H]
        gen_time_emb: torch.Tensor,         # [total_gen_tokens, H] per-token timestep emb
        gen_token_mask: torch.Tensor,       # [B, S] bool, True = generation image token
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        startend_row_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """adaLN-Zero 条件化的 transformer block 前向。

        Args:
            hidden_states:        完整序列的隐状态，shape ``[B, S, H]``。
            gen_time_emb:         per-token 时间步 embedding，shape ``[total_gen_tokens, H]``。
            gen_token_mask:       布尔掩码，shape ``[B, S]``，True = 生成图像 token。
            attention_mask:       加性 attention mask ``[B, 1, S, S]`` 或 None (fallback)。
            position_ids:         ``[B, S]``/``[1, S]`` (1-D) 或 ``[3, B, S]`` (M-RoPE)。
            startend_row_indices: ``[B, 1, S, 2]`` FlashMask 格式 (优先使用)。
            **kwargs:             兼容性占位。

        Returns:
            Tuple，第 0 元素为更新后的 hidden_states，shape ``[B, S, H]``。
        """
        # block_adaln=False: 纯 pre-norm transformer, 无 adaLN/gate 调制
        # (时间条件仅在 head 末层注入)。忽略 gen_time_emb / gen_token_mask。
        if not self.block_adaln:
            residual = hidden_states
            hidden_states, _, _ = self.self_attn(
                hidden_states=self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                position_ids=position_ids,
                startend_row_indices=startend_row_indices,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = residual + hidden_states
            return (hidden_states,)

        # Compute per-token adaLN modulation from per-token time embedding.
        modulation = self.adaLN_modulation(gen_time_emb)  # [N_gen, n_mod*H]

        if self.adaln_post_modulation:
            # post-mod (NVlabs PixelDiT #6): 4-way scale/shift 施加在 attn/mlp 输出上 (无 gate)。
            scale_msa, shift_msa, scale_mlp, shift_mlp = modulation.chunk(4, dim=-1)
            # 非 gen 位置默认 scale=0/shift=0 → _modulate 恒等 (等价 residual+out)。
            shift_msa_n = torch.zeros_like(hidden_states)
            scale_msa_n = torch.zeros_like(hidden_states)
            shift_mlp_n = torch.zeros_like(hidden_states)
            scale_mlp_n = torch.zeros_like(hidden_states)
            shift_msa_n[gen_token_mask] = shift_msa.to(hidden_states.dtype)
            scale_msa_n[gen_token_mask] = scale_msa.to(hidden_states.dtype)
            shift_mlp_n[gen_token_mask] = shift_mlp.to(hidden_states.dtype)
            scale_mlp_n[gen_token_mask] = scale_mlp.to(hidden_states.dtype)

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                startend_row_indices=startend_row_indices,
            )
            hidden_states = residual + _modulate(hidden_states, shift_msa_n, scale_msa_n)

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + _modulate(hidden_states, shift_mlp_n, scale_mlp_n)

            return (hidden_states,)

        # pre-mod (default): Compute per-token 6-chunk adaLN modulation.
        # [total_gen_tokens, 6*H] → 6 × [total_gen_tokens, H]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        # Scatter into full-sequence tensors; non-gen positions keep identity defaults.
        shift_msa_n = torch.zeros_like(hidden_states)
        scale_msa_n = torch.zeros_like(hidden_states)
        gate_msa_n  = torch.ones_like(hidden_states)
        shift_mlp_n = torch.zeros_like(hidden_states)
        scale_mlp_n = torch.zeros_like(hidden_states)
        gate_mlp_n  = torch.ones_like(hidden_states)

        shift_msa_n[gen_token_mask] = shift_msa.to(hidden_states.dtype)
        scale_msa_n[gen_token_mask] = scale_msa.to(hidden_states.dtype)
        gate_msa_n[gen_token_mask]  = gate_msa.to(hidden_states.dtype)
        shift_mlp_n[gen_token_mask] = shift_mlp.to(hidden_states.dtype)
        scale_mlp_n[gen_token_mask] = scale_mlp.to(hidden_states.dtype)
        gate_mlp_n[gen_token_mask]  = gate_mlp.to(hidden_states.dtype)

        residual = hidden_states
        hidden_states = _modulate(self.input_layernorm(hidden_states), shift_msa_n, scale_msa_n)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            startend_row_indices=startend_row_indices,
        )
        hidden_states = residual + gate_msa_n * hidden_states

        residual = hidden_states
        hidden_states = _modulate(self.post_attention_layernorm(hidden_states), shift_mlp_n, scale_mlp_n)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp_n * hidden_states

        return (hidden_states,)


class DiffusionHeadFinalLayer(nn.Module):
    """adaLN-modulated RMSNorm + linear output layer.

    Same conditioning design as ``ModulatedAttentionBlock``:
    uses per-token ``gen_time_emb`` and a boolean ``gen_token_mask``
    to apply modulation only at generation image token positions.

    Args:
        hidden_size:  LLM hidden dimension.
        out_channels: Output channel count (3072 for merged RGB patches).
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = DiffHeadRMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # Zero-init (standard DiT initialisation – identity at the start)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,               # [B, S, H]
        gen_time_emb: torch.Tensor,    # [total_gen_tokens, H]
        gen_token_mask: torch.Tensor,  # [B, S] bool
    ) -> torch.Tensor:
        """adaLN 调制的最终归一化与线性投影。

        流程：
          1. 从 gen_time_emb 计算 per-token 的 shift 和 scale。
          2. 将 shift/scale 按 gen_token_mask scatter 到完整序列，
             非生成位置保持 shift=0 / scale=0（恒等）。
          3. adaLN 调制后的 RMSNorm → 线性投影到输出通道数。

        输出张量保持完整序列形状 ``[B, S, out_channels]``，调用方
        用 ``x0_full[gen_token_mask]`` 提取生成 token 的预测值。

        Args:
            x:             完整序列的隐状态，shape ``[B, S, H]``。
            gen_time_emb:  生成 token 的 per-token 时间步 embedding，
                           shape ``[total_gen_tokens, H]``。
            gen_token_mask: 布尔掩码，shape ``[B, S]``，True 表示生成图像 token。

        Returns:
            shape ``[B, S, out_channels]`` 的输出张量；
            通过 ``result[gen_token_mask]`` 可得 ``[total_gen_tokens, out_channels]`` 的预测。
        """
        # [N_gen, 2*H] → shift [N_gen, H], scale [N_gen, H]
        shift, scale = self.adaLN_modulation(gen_time_emb).chunk(2, dim=-1)

        shift_new = torch.zeros_like(x)
        scale_new = torch.zeros_like(x)
        shift_new[gen_token_mask] = shift.to(x.dtype)
        scale_new[gen_token_mask] = scale.to(x.dtype)

        x = _modulate(self.norm_final(x), shift_new, scale_new)
        return self.linear(x)
