"""Diffusion Head built on Megatron's Attention/TransformerLayer infrastructure.

Directly inherits from:
- megatron.core.transformer.attention.SelfAttention  → TP/FlashMask/GQA/Recompute
- megatron.core.transformer.transformer_layer.TransformerLayer → standard layer structure

Adds:
- adaLN-Zero modulation (DiT-style conditioning)
- M-RoPE computation from position_ids [3, B, S]

Attention goes through Megatron's _flashmask_attention, which calls
flashmask.frontend.torch.flashmask_attention for training and inference.

Public API is identical to diffusion_head.py:
- ModulatedAttentionBlock(config, layer_idx)  — same forward signature
- DiffusionHeadFinalLayer (reused from diffusion_head.py)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from model.transfusion.diffusion_head import (
    DiffHeadConfig,
    DiffHeadRMSNorm,
    DiffusionHeadFinalLayer,
    _modulate,
    _apply_interleaved_mrope,
)

# Megatron core imports
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

try:
    from megatron.core.extensions.transformer_engine import TENorm
except ImportError:
    TENorm = None

try:
    from megatron.core.transformer.custom_layers.transformer_engine import TERMSNorm
except ImportError:
    TERMSNorm = None

# TE 并行 linear / attention (fp8 GEMM), 让 DiT 跟随 backbone 的 fp8 训练。
# 不可用时 (无 TE) 回退到 megatron 原生并行层 (bf16, 仍 TP 切分)。
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
        TEDotProductAttention,
    )
    _HAVE_TE_LINEAR = True
except ImportError:
    TEColumnParallelLinear = ColumnParallelLinear
    TERowParallelLinear = RowParallelLinear
    TEDotProductAttention = DotProductAttention
    _HAVE_TE_LINEAR = False


# ---------------------------------------------------------------------------
# Config helper: build a TransformerConfig from DiffHeadConfig
# ---------------------------------------------------------------------------

def _init_parallel_linear_like_torch(module: nn.Module, fan_in: int) -> None:
    """Initialize Megatron/TE parallel linear weights like torch.nn.Linear."""
    if fan_in <= 0:
        return
    bound = 1.0 / math.sqrt(fan_in)
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    with torch.no_grad():
        if weight is not None:
            nn.init.uniform_(weight, -bound, bound)
        if bias is not None:
            nn.init.uniform_(bias, -bound, bound)

def build_megatron_config_from_diff_head(
    config: DiffHeadConfig,
    tensor_model_parallel_size: int = 1,
    sequence_parallel: bool = False,
    params_dtype: Optional[torch.dtype] = None,
    bf16: bool = False,
    fp16: bool = False,
    fp8: Optional[str] = None,
    fp8_recipe: str = "tensorwise",
) -> TransformerConfig:
    """Construct a Megatron TransformerConfig suitable for the diffusion head.

    Args:
        config: The DiffHeadConfig with model architecture params.
        tensor_model_parallel_size: TP size. Pass the backbone TP so the DiT head is
            sharded together with the backbone (not replicated per rank).
        sequence_parallel: Whether to use SP (default False).
        params_dtype/bf16/fp16: 跟随 backbone 训练精度 (训练时传 bf16=True +
            params_dtype=torch.bfloat16)。
        fp8/fp8_recipe: 跟随 backbone 的 fp8 设置 (如 "hybrid")。DiT 用 TE 并行层
            (TEColumnParallelLinear 等), fp8 对其生效 → DiT 与 backbone 一同跑 fp8。
            None = 不上 fp8 (纯 bf16, 例如无 TE 环境回退)。

    Returns:
        TransformerConfig instance compatible with Megatron's SelfAttention/MLP/TransformerLayer.
    """
    fp8_kwargs = {}
    if fp8 is not None and _HAVE_TE_LINEAR:
        fp8_kwargs = dict(
            fp8=fp8,
            fp8_recipe=fp8_recipe,
            fp8_amax_history_len=1,
            fp8_amax_compute_algo="most_recent",
        )
    return TransformerConfig(
        # Parallelism
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=sequence_parallel,
        # Precision (跟随 backbone; DiT 用 TE 层, fp8 对其生效)
        params_dtype=params_dtype if params_dtype is not None else torch.float32,
        bf16=bf16,
        fp16=fp16,
        **fp8_kwargs,
        # Architecture
        num_layers=1,  # single layer, we build ModuleList externally
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
        kv_channels=config.head_dim,
        ffn_hidden_size=config.intermediate_size,
        # FlashMask
        use_flashmask=True,
        # Norm
        normalization="RMSNorm",
        layernorm_epsilon=config.rms_norm_eps,
        # Attention
        attention_dropout=config.attention_dropout,
        hidden_dropout=0.0,
        qk_layernorm=getattr(config, "qk_norm", True),
        add_bias_linear=False,
        add_qkv_bias=config.attention_bias,
        # MLP (SwiGLU)
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        # RoPE (handled externally via rotary_pos_emb)
        rotary_interleaved=False,
        apply_rope_fusion=False,
        # Softmax
        attention_softmax_in_fp32=True,
        # Init (placeholder, will be overwritten by checkpoint loading)
        init_method_std=0.02,
        # Disable features not needed
        fp32_residual_connection=False,
        apply_residual_connection_post_layernorm=False,
        bias_dropout_fusion=False,
        persist_layer_norm=False,
    )


# ---------------------------------------------------------------------------
# M-RoPE: compute rotary freqs from position_ids → Megatron format [S, B, 1, D]
# ---------------------------------------------------------------------------

class _DiffHeadMegatronRoPE(nn.Module):
    """M-RoPE that outputs raw angle freqs in Megatron format [S, B, 1, head_dim].

    Compatible with megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd.
    """

    MROPE_SECTION = [24, 20, 20]

    def __init__(self, config: DiffHeadConfig):
        super().__init__()
        rope_type = config.rope_scaling.get("rope_type", "default")
        init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, self.attention_scaling = init_fn(config, None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = config.head_dim

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute M-RoPE freqs.

        Args:
            position_ids: [3, B, S] for M-RoPE, or [B, S] for 1-D RoPE.

        Returns:
            freqs: [S, B, 1, head_dim] raw angle values.
        """
        device_type = position_ids.device.type if position_ids.device.type != "mps" else "cpu"

        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            B, S = position_ids.shape[1], position_ids.shape[2]
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, B, -1, 1)
            pos_expanded = position_ids[:, :, None, :].float()
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ pos_expanded.float()).transpose(2, 3)
                freqs = _apply_interleaved_mrope(freqs, self.MROPE_SECTION)
                freqs = torch.cat((freqs, freqs), dim=-1)
        else:
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            )
            pos_expanded = position_ids[:, None, :].float()
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ pos_expanded.float()).transpose(1, 2)
                freqs = torch.cat((freqs, freqs), dim=-1)

        # [B, S, head_dim] → [S, B, 1, head_dim] (Megatron format)
        return freqs.transpose(0, 1).unsqueeze(2)


# ---------------------------------------------------------------------------
# Layer spec helper: build standard Megatron layer spec for diffusion head
# ---------------------------------------------------------------------------

def get_diff_head_layer_spec(qk_layernorm: bool = True) -> ModuleSpec:
    """Get a Megatron TransformerLayer spec for the diffusion head.

    Uses local (non-TE) modules: ColumnParallelLinear, RowParallelLinear,
    DotProductAttention, MLP with GLU.
    """
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=WrappedTorchNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=WrappedTorchNorm if qk_layernorm else IdentityOp,
                    k_layernorm=WrappedTorchNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=ModuleSpec(module=IdentityFuncOp),
            pre_mlp_layernorm=WrappedTorchNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=ModuleSpec(module=IdentityFuncOp),
        ),
    )


# ---------------------------------------------------------------------------
# ModulatedAttentionBlock: adaLN-Zero + Megatron TransformerLayer
# ---------------------------------------------------------------------------

class ModulatedAttentionBlock(nn.Module):
    """adaLN-Zero conditioned transformer block built on Megatron's TransformerLayer.

    Directly uses Megatron's SelfAttention (with TP/FlashMask/Recompute support)
    and MLP (with TP/GLU support).

    The adaLN-Zero conditioning modulates BETWEEN layernorm and attention/MLP,
    and gates the residual connection — this is custom logic wrapping Megatron's
    standard forward path.

    Conditioning:
    - gen_time_emb  : [total_gen_tokens, H] per-token timestep embedding
    - gen_token_mask: [B, S] bool, True at generation image token positions
    """

    def __init__(
        self,
        config: DiffHeadConfig,
        layer_idx: int,
        megatron_config: Optional[TransformerConfig] = None,
        adaln_post_modulation: bool = False,
        block_adaln: bool = True,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        # block_adaln=False: transformer block 内不做任何 adaLN 调制 (纯 pre-norm
        # transformer, 无 time/gate 条件), 时间条件仅由 head 最后一层的 adaLN 注入。
        # 用于 "no adaln in transformer block" 变体; 默认 True 保持原状。
        self.block_adaln = bool(block_adaln)
        # post-modulation (NVlabs PixelDiT #6): 4-way scale/shift 施加在 attn/mlp
        # 输出上 (无 gate), 缓解训练 spike; 默认 False 保持原 6-way pre-mod。
        self.adaln_post_modulation = bool(adaln_post_modulation)

        # Build Megatron config if not provided
        if megatron_config is None:
            megatron_config = build_megatron_config_from_diff_head(config)
        self.megatron_config = megatron_config

        # Build Megatron SelfAttention (inherits TP/FlashMask/GQA)
        # 用 TE 版 linear/attention, 让 DiT 跟随 backbone 的 fp8 训练。
        from megatron.core.transformer.torch_norm import WrappedTorchNorm
        qk_layernorm = getattr(config, "qk_norm", True)
        _qk_norm = TENorm if (qk_layernorm and TENorm is not None) else (
            WrappedTorchNorm if qk_layernorm else IdentityOp
        )
        self.self_attention = SelfAttention(
            config=megatron_config,
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=_qk_norm,
                k_layernorm=_qk_norm,
            ),
            layer_number=layer_idx + 1,  # Megatron uses 1-indexed
            attn_mask_type=AttnMaskType.no_mask,
        )

        # Build Megatron MLP (SwiGLU with TP support); TE 版 fp8 linear。
        self.mlp = MLP(
            config=megatron_config,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        )
        if getattr(megatron_config, "perform_initialization", True):
            # Match diffusion_head.py's nn.Linear reset_parameters. Megatron's
            # default normal(0, 0.02) is much larger for this 2.5B hidden size
            # and made the Megatron DiT head converge on a different loss curve.
            _init_parallel_linear_like_torch(
                self.self_attention.linear_qkv, config.hidden_size
            )
            _init_parallel_linear_like_torch(
                self.self_attention.linear_proj, config.hidden_size
            )
            _init_parallel_linear_like_torch(
                self.mlp.linear_fc1, config.hidden_size
            )
            _init_parallel_linear_like_torch(
                self.mlp.linear_fc2, config.intermediate_size
            )

        # LayerNorms (standalone, not from TransformerLayer, because adaLN
        # modulates the output AFTER norm and BEFORE attention/MLP)
        self.input_layernorm = DiffHeadRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffHeadRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # adaLN-Zero modulation: 6 params pre-mod / 4 params post-mod
        # (block_adaln=False 时不建, block 内无条件调制)。
        if self.block_adaln:
            n_mod = 4 if self.adaln_post_modulation else 6
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, n_mod * config.hidden_size, bias=True),
            )
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

        # M-RoPE
        self.rotary_emb = _DiffHeadMegatronRoPE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        gen_time_emb: torch.Tensor,
        gen_token_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        startend_row_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """adaLN-Zero conditioned transformer block forward.

        Args:
            hidden_states:        [B, S, H]
            gen_time_emb:         [total_gen_tokens, H] per-token timestep embedding
            gen_token_mask:       [B, S] bool, True = generation image token
            attention_mask:       [B, 1, S, S] additive mask or None (not used with FlashMask)
            position_ids:         [3, B, S] M-RoPE or [B, S] standard
            startend_row_indices: [B, 1, S, 2] FlashMask indices

        Returns:
            Tuple with element 0 = updated hidden_states [B, S, H].
        """
        # --- block_adaln=False: 纯 pre-norm transformer (无 adaLN/gate 调制) ---
        # 时间条件仅在 head 末层注入; 忽略 gen_time_emb / gen_token_mask。
        if not self.block_adaln:
            rotary_pos_emb = None
            if position_ids is not None:
                rotary_pos_emb = self.rotary_emb(position_ids)

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = hidden_states.transpose(0, 1).contiguous()  # [B,S,H]→[S,B,H]
            attn_out_with_bias = self.self_attention(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                startend_row_indices=startend_row_indices,
                causal=False,
            )
            if isinstance(attn_out_with_bias, tuple):
                attn_output, attn_bias = attn_out_with_bias
            else:
                attn_output, attn_bias = attn_out_with_bias, None
            attn_output = attn_output.transpose(0, 1).contiguous()  # [S,B,H]→[B,S,H]
            if attn_bias is not None:
                attn_output = attn_output + attn_bias
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states.transpose(0, 1).contiguous()  # [B,S,H]→[S,B,H]
            mlp_out_with_bias = self.mlp(hidden_states)
            if isinstance(mlp_out_with_bias, tuple):
                mlp_output, mlp_bias = mlp_out_with_bias
            else:
                mlp_output, mlp_bias = mlp_out_with_bias, None
            mlp_output = mlp_output.transpose(0, 1).contiguous()  # [S,B,H]→[B,S,H]
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            hidden_states = residual + mlp_output
            return (hidden_states,)

        # --- adaLN modulation computation ---
        modulation = self.adaLN_modulation(gen_time_emb)  # [N_gen, n_mod*H]

        if self.adaln_post_modulation:
            # post-mod (NVlabs PixelDiT #6): 4-way scale/shift 施加在 attn/mlp 输出上 (无 gate)
            scale_msa, shift_msa, scale_mlp, shift_mlp = modulation.chunk(4, dim=-1)
            shift_msa_n = torch.zeros_like(hidden_states)
            scale_msa_n = torch.zeros_like(hidden_states)
            shift_mlp_n = torch.zeros_like(hidden_states)
            scale_mlp_n = torch.zeros_like(hidden_states)
            shift_msa_n[gen_token_mask] = shift_msa.to(hidden_states.dtype)
            scale_msa_n[gen_token_mask] = scale_msa.to(hidden_states.dtype)
            shift_mlp_n[gen_token_mask] = shift_mlp.to(hidden_states.dtype)
            scale_mlp_n[gen_token_mask] = scale_mlp.to(hidden_states.dtype)
            gate_msa_n = None
            gate_mlp_n = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

            # Scatter into full-sequence tensors; non-gen positions keep identity
            shift_msa_n = torch.zeros_like(hidden_states)
            scale_msa_n = torch.zeros_like(hidden_states)
            gate_msa_n = torch.ones_like(hidden_states)
            shift_mlp_n = torch.zeros_like(hidden_states)
            scale_mlp_n = torch.zeros_like(hidden_states)
            gate_mlp_n = torch.ones_like(hidden_states)

            shift_msa_n[gen_token_mask] = shift_msa.to(hidden_states.dtype)
            scale_msa_n[gen_token_mask] = scale_msa.to(hidden_states.dtype)
            gate_msa_n[gen_token_mask] = gate_msa.to(hidden_states.dtype)
            shift_mlp_n[gen_token_mask] = shift_mlp.to(hidden_states.dtype)
            scale_mlp_n[gen_token_mask] = scale_mlp.to(hidden_states.dtype)
            gate_mlp_n[gen_token_mask] = gate_mlp.to(hidden_states.dtype)

        # --- Attention block ---
        residual = hidden_states
        # adaLN: norm (pre-mod 还要 modulate 输入)
        if self.adaln_post_modulation:
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = _modulate(self.input_layernorm(hidden_states), shift_msa_n, scale_msa_n)

        # Convert [B, S, H] → [S, B, H] for Megatron's SelfAttention
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Compute rotary_pos_emb from position_ids
        rotary_pos_emb = None
        if position_ids is not None:
            rotary_pos_emb = self.rotary_emb(position_ids)  # [S, B, 1, head_dim]

        # Call Megatron SelfAttention (handles FlashMask train/infer dispatch internally)
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            startend_row_indices=startend_row_indices,
            causal=False,
        )

        # SelfAttention returns (output, bias) — unpack
        if isinstance(attention_output_with_bias, tuple):
            attn_output, attn_bias = attention_output_with_bias
        else:
            attn_output, attn_bias = attention_output_with_bias, None

        # Convert [S, B, H] → [B, S, H]
        attn_output = attn_output.transpose(0, 1).contiguous()
        if attn_bias is not None:
            attn_output = attn_output + attn_bias

        # Gated residual (pre-mod) / post-modulated residual
        if self.adaln_post_modulation:
            hidden_states = residual + _modulate(attn_output, shift_msa_n, scale_msa_n)
        else:
            hidden_states = residual + gate_msa_n * attn_output

        # --- MLP block ---
        residual = hidden_states
        # adaLN: norm (pre-mod 还要 modulate 输入)
        if self.adaln_post_modulation:
            hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            hidden_states = _modulate(self.post_attention_layernorm(hidden_states), shift_mlp_n, scale_mlp_n)

        # Convert [B, S, H] → [S, B, H] for Megatron MLP
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Call Megatron MLP (returns (output, bias))
        mlp_output_with_bias = self.mlp(hidden_states)
        if isinstance(mlp_output_with_bias, tuple):
            mlp_output, mlp_bias = mlp_output_with_bias
        else:
            mlp_output, mlp_bias = mlp_output_with_bias, None

        # Convert [S, B, H] → [B, S, H]
        mlp_output = mlp_output.transpose(0, 1).contiguous()
        if mlp_bias is not None:
            mlp_output = mlp_output + mlp_bias

        # Gated residual (pre-mod) / post-modulated residual
        if self.adaln_post_modulation:
            hidden_states = residual + _modulate(mlp_output, shift_mlp_n, scale_mlp_n)
        else:
            hidden_states = residual + gate_mlp_n * mlp_output

        return (hidden_states,)
