"""Dense Qwen3-VL model with a raw-RGB Pixel-DiT generation head."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from megatron.core import InferenceParams, mpu, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig as Qwen3VLConfigHF

# 复用原版的底层组件
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import get_rope_index, split_deepstack_embs
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync

from model.transfusion.qwen3_vl_vision import Qwen3VLVisionModel
from model.transfusion.hi_gate import HiGate
from data.transfusion.transfusion_task_encoder import TaskType
from model.transfusion.diffusion_head import (
    DiffHeadConfig,
    DiffusionHeadFinalLayer,
    ModulatedAttentionBlock as TorchModulatedAttentionBlock,
)
# DiT head 两种实现, 由 config.dit_head_impl 选择 (默认 "torch"):
#   - torch 原生版 (上面 TorchModulatedAttentionBlock): nn.Linear 分离 q/k/v + gate/up,
#     不随 TP 切分 (TP>1 每卡全量存 DiT); 与 05f9c1e 之前训练的 ckpt 命名兼容。
#   - Megatron/TE 版 (下面 MegatronModulatedAttentionBlock): 融合 qkv/gate-up, 随 backbone
#     TP 切分, TE 层支持 FP8 加速; 权重命名与 torch 版不同, 不可混用同一 ckpt。
from model.transfusion.diffusion_head_megatron import (
    ModulatedAttentionBlock as MegatronModulatedAttentionBlock,
    build_megatron_config_from_diff_head,
)


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding (matching HiUni's implementation)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal encoding + MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        """t: [B,]"""
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class TransfusionQwen3VLModel(MegatronModule):
    def __init__(
        self,
        language_transformer_config: Qwen3VLTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        vision_transformer_config: Qwen3VLConfigHF,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        # 原版初始化逻辑...
        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id

        self.share_embeddings_and_output_weights = False
        # This checkout intentionally exposes one architecture: dense Pixel-DiT.
        self.use_pixel_dit = True
        self.diffusion_head_recompute = getattr(
            language_transformer_config, "diffusion_head_recompute", False
        )
        self.pixel_dit_gate_inject_layer = getattr(
            language_transformer_config, "pixel_dit_gate_inject_layer", 0
        )
        self.pixel_dit_gate_input = str(
            getattr(language_transformer_config, "pixel_dit_gate_input", "pixel")
        ).lower()
        if self.pixel_dit_gate_input not in ("pixel", "vit"):
            raise ValueError(
                "pixel_dit_gate_input must be one of ('pixel', 'vit'), got "
                f"{self.pixel_dit_gate_input!r}"
            )
        self.use_pixel_dit_pixel_residual = getattr(
            language_transformer_config, "use_pixel_dit_pixel_residual", True
        )
        self.pixel_dit_ref_gate_mode = getattr(
            language_transformer_config, "pixel_dit_ref_gate_mode", "none"
        )
        if self.pixel_dit_ref_gate_mode not in ("none", "shared", "separate"):
            raise ValueError(
                "pixel_dit_ref_gate_mode must be one of "
                f"('none', 'shared', 'separate'), got {self.pixel_dit_ref_gate_mode!r}"
            )
        lm_hidden_size = language_transformer_config.hidden_size
        configured_pixel_dit_hidden_size = int(
            getattr(language_transformer_config, "pixel_dit_hidden_size", 0) or 0
        )
        if configured_pixel_dit_hidden_size < 0:
            raise ValueError(
                "pixel_dit_hidden_size must be >= 0, got "
                f"{configured_pixel_dit_hidden_size}"
            )
        self.pixel_dit_hidden_size = (
            configured_pixel_dit_hidden_size or lm_hidden_size
        )
        self.freeze_dit_param = bool(
            getattr(language_transformer_config, "freeze_dit_param", False)
        )
        self.freeze_pixel_proj_param = bool(
            getattr(language_transformer_config, "freeze_pixel_proj_param", False)
        )
        self.freeze_pixel_dit_gate_param = bool(
            getattr(language_transformer_config, "freeze_pixel_dit_gate_param", False)
        )
        self.freeze_time_embeddings_param = bool(
            getattr(language_transformer_config, "freeze_time_embeddings_param", False)
        )
        self.freeze_pixel_dit_last_n_layers = int(
            getattr(language_transformer_config, "freeze_pixel_dit_last_n_layers", 0)
            or 0
        )
        self.freeze_pixel_dit_final_layer = bool(
            getattr(language_transformer_config, "freeze_pixel_dit_final_layer", False)
        )
        if self.freeze_pixel_dit_last_n_layers < 0:
            raise ValueError(
                "freeze_pixel_dit_last_n_layers must be >= 0, got "
                f"{self.freeze_pixel_dit_last_n_layers}"
            )

        if self.pre_process or self.use_pixel_dit:
            self.time_embeddings = TimestepEmbedder(lm_hidden_size)
        # A separate embedder is required only when Pixel-DiT is narrower than
        # the language backbone. Keeping the legacy module for equal widths
        # preserves existing checkpoint parameter names exactly.
        self.pixel_dit_time_embeddings = None
        if self.use_pixel_dit and self.pixel_dit_hidden_size != lm_hidden_size:
            self.pixel_dit_time_embeddings = TimestepEmbedder(
                self.pixel_dit_hidden_size
            )

        if self.pre_process:
            # 完整保留 Qwen3-VL 的原版视觉编码器
            self.vision_model = Qwen3VLVisionModel._from_config(vision_transformer_config)
            if getattr(language_transformer_config, "vit_recompute", False):
                self.vision_model.gradient_checkpointing_enable()
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            if (
                torch.cuda.is_available()
                and not getattr(
                    language_transformer_config, "use_cpu_initialization", False
                )
            ):
                self.vision_model = self.vision_model.to("cuda")
            for param in self.vision_model.parameters():
                param.is_vit_parameter = True

            self.time_token = TimestepEmbedder(lm_hidden_size)  # 替换 <time> token 的 embedding
            self.time_token_id = getattr(language_transformer_config, "time_token_id", 151661)
        self.language_model = Qwen3VLGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )

        # Per-channel ViT feature normalization (running statistics)
        vit_dim = language_transformer_config.vision_config.hidden_size  # 1024
        self.register_buffer("vit_channel_mean", torch.zeros(vit_dim))
        self.register_buffer("vit_channel_std", torch.ones(vit_dim))
        self.register_buffer("vit_stats_count", torch.zeros(1))  # 已累积的 batch 数
        self._vit_stats_momentum = 0.01

        # 新增: Diffusion 输出头
        if self.post_process:
            lm_dim = language_transformer_config.hidden_size
            if self.use_pixel_dit:
                # === Dense Pixel-DiT: Qwen3-VL backbone + pixel-space DiT ===
                num_diffusion_layers = getattr(
                    language_transformer_config, "num_diffusion_layers", 0
                )
                assert num_diffusion_layers > 0, (
                    "use_pixel_dit 需要 num_diffusion_layers > 0 (DiT head 层数)"
                )
                # merged patch = 4 raw patches (2x2 spatial merge), 每 raw patch = 3*16*16 = 768
                pixel_dit_out_dim = 3 * 16 * 16 * 4  # 3072
                pixel_dit_dim = self.pixel_dit_hidden_size
                if pixel_dit_dim == lm_dim:
                    self.pixel_dit_input_projection = nn.Identity()
                else:
                    self.pixel_dit_input_projection = nn.Linear(
                        lm_dim, pixel_dit_dim, bias=True
                    )
                self.diffusion_head_a, self.diffusion_head_b = self._build_multi_layer_dit_head(
                    language_transformer_config,
                    lm_dim=pixel_dit_dim,
                    num_diffusion_layers=num_diffusion_layers,
                    out_dim=pixel_dit_out_dim,
                )
                # Gate: merged noisy pixels -> Pixel-DiT width, then inject.
                self.pixel_proj = nn.Linear(pixel_dit_out_dim, pixel_dit_dim)
                self.pixel_dit_gate = HiGate(
                    embed_dim=pixel_dit_dim,
                    config=language_transformer_config,
                )
                self.pixel_dit_ref_gate = None
                if self.pixel_dit_ref_gate_mode == "separate":
                    self.pixel_dit_ref_gate = HiGate(
                        embed_dim=pixel_dit_dim,
                        config=language_transformer_config,
                    )
                if self.pixel_dit_gate_input == "vit":
                    # Kept in the module/state dict for checkpoint compatibility,
                    # but unused when HiGate consumes ViT features.
                    self.pixel_proj.requires_grad_(False)
                if not self.use_pixel_dit_pixel_residual:
                    self.pixel_proj.requires_grad_(False)
                    self.pixel_dit_gate.requires_grad_(False)
                    if self.pixel_dit_ref_gate is not None:
                        self.pixel_dit_ref_gate.requires_grad_(False)
                self.image_diffusion_head = None

            # PP 支持: post_process stage 也需要 TimestepEmbedder（用于 dit_head 的 _gen_time_emb）
            if not self.pre_process and not hasattr(self, "time_embeddings"):
                self.time_embeddings = TimestepEmbedder(lm_hidden_size)

            # 1. 定义前向预钩子 (Pre-Hook)
            def capture_hidden_states_hook(module, args):
                # args 是个元组，args[0] 就是送到 LM Head 之前的干净 hidden_states
                self._captured_hidden_states = args[0]
                return args

            # 2. 递归寻找并挂载

            def register_hook_to_lm_head(model_module):
                if hasattr(model_module, "output_layer"):
                    model_module.output_layer.register_forward_pre_hook(capture_hidden_states_hook)
                    return True
                for child in model_module.children():
                    if register_hook_to_lm_head(child):
                        return True
                return False

            # 执行挂载！
            register_hook_to_lm_head(self.language_model)

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    @staticmethod
    def _dit_head_layer_forward(
        layer,
        hidden_states,
        gen_time_emb,
        gen_token_mask,
        position_ids,
        startend_row_indices,
    ):
        return layer(
            hidden_states=hidden_states,
            gen_time_emb=gen_time_emb,
            gen_token_mask=gen_token_mask,
            attention_mask=None,
            position_ids=position_ids,
            startend_row_indices=startend_row_indices,
        )[0]

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def _embed_pixel_dit_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed Pixel-DiT timesteps at the configured DiT width."""
        embedder = self.pixel_dit_time_embeddings
        if embedder is None:
            embedder = self.time_embeddings
        return embedder(timesteps)

    def _project_pixel_dit_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project backbone states to Pixel-DiT width without mutating them."""
        projection = self.pixel_dit_input_projection
        if isinstance(projection, nn.Identity):
            return hidden_states.clone()
        return projection(hidden_states.to(projection.weight.dtype))

    def _prepare_pixel_dit_gate_input(
        self,
        *,
        pixel_features: torch.Tensor = None,
        vit_features: torch.Tensor = None,
        expected_tokens: int = None,
        name: str = "generation",
    ) -> torch.Tensor:
        """Build HiGate high-info from merged pixels or merged ViT features."""
        if self.pixel_dit_gate_input == "pixel":
            if pixel_features is None:
                raise RuntimeError(
                    f"Pixel-DiT {name} gate requires merged pixel features"
                )
            gate_input = self.pixel_proj(
                pixel_features.to(self.pixel_proj.weight.dtype)
            )
        elif self.pixel_dit_gate_input == "vit":
            if vit_features is None:
                raise RuntimeError(
                    f"Pixel-DiT {name} gate requires merged ViT features"
                )
            gate_input = self._project_pixel_dit_hidden(vit_features)
        else:
            raise ValueError(
                "pixel_dit_gate_input must be one of ('pixel', 'vit'), got "
                f"{self.pixel_dit_gate_input!r}"
            )

        if expected_tokens is not None and gate_input.shape[0] != expected_tokens:
            raise ValueError(
                f"Pixel-DiT {name} gate token mismatch for "
                f"pixel_dit_gate_input={self.pixel_dit_gate_input!r}: "
                f"features={gate_input.shape[0]}, tokens={expected_tokens}"
            )
        return gate_input

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3VL"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    @staticmethod
    def _build_multi_layer_dit_head(
        language_transformer_config,
        lm_dim: int,
        num_diffusion_layers: int,
        out_dim: int,
    ):
        """构建多层 DiT head (diffusion_head_a + diffusion_head_b)。

        Build the dense Pixel-DiT head with either the native torch or
        Megatron/Transformer-Engine implementation.
        """
        _llm_num_heads = getattr(
            language_transformer_config, "num_attention_heads",
            getattr(language_transformer_config, "num_heads", 28),
        )
        _llm_num_kv_heads = getattr(
            language_transformer_config, "num_key_value_heads",
            getattr(language_transformer_config, "num_query_groups", 4),
        )
        num_diff_heads = (
            getattr(language_transformer_config, "num_diffusion_heads", 0)
            or _llm_num_heads
        )
        num_diff_kv_heads = (
            getattr(language_transformer_config, "num_diffusion_kv_heads", 0)
            or _llm_num_kv_heads
        )
        diff_intermediate_size = (
            getattr(language_transformer_config, "diffusion_intermediate_size", 0)
            or language_transformer_config.ffn_hidden_size
        )
        diff_head_cfg = DiffHeadConfig(
            hidden_size=lm_dim,
            num_attention_heads=num_diff_heads,
            num_key_value_heads=num_diff_kv_heads,
            intermediate_size=diff_intermediate_size,
            max_position_embeddings=(
                language_transformer_config.language_max_sequence_length
            ),
        )
        dit_head_impl = getattr(
            language_transformer_config, "dit_head_impl", "torch"
        )
        # post-modulation (NVlabs PixelDiT #6): 4-way scale/shift on attn/mlp
        # outputs, 缓解 spike; 默认 False 保持 6-way pre-mod (与旧 ckpt 兼容)。
        _post_mod = getattr(
            language_transformer_config, "adaln_post_modulation", False
        )
        # dit_block_adaln=False: transformer block 内不做 adaLN 调制 (纯 pre-norm
        # transformer), 时间条件仅由 head 末层 adaLN 注入; 默认 True 保持原状。
        _block_adaln = getattr(
            language_transformer_config, "dit_block_adaln", True
        )
        if dit_head_impl == "torch":
            # torch 原生版: nn.Linear 分离 q/k/v + gate/up, 不随 TP 切分。
            # 与 05f9c1e 之前训练的 ckpt 命名兼容 (默认)。
            diffusion_head_a = nn.ModuleList([
                TorchModulatedAttentionBlock(
                    diff_head_cfg, layer_idx=i, adaln_post_modulation=_post_mod,
                    block_adaln=_block_adaln,
                )
                for i in range(num_diffusion_layers)
            ])
        elif dit_head_impl == "megatron":
            # Megatron DiT: 用 backbone 的 TP + 精度 + fp8 构建 megatron_config,
            # 各层共享 (DiT head 随 backbone TP 切分; TE 层跟随 backbone fp8 训练)。
            _diff_mcfg = build_megatron_config_from_diff_head(
                diff_head_cfg,
                tensor_model_parallel_size=getattr(
                    language_transformer_config, "tensor_model_parallel_size", 1
                ),
                sequence_parallel=getattr(
                    language_transformer_config, "sequence_parallel", False
                ),
                params_dtype=getattr(language_transformer_config, "params_dtype", None),
                bf16=getattr(language_transformer_config, "bf16", False),
                fp16=getattr(language_transformer_config, "fp16", False),
                fp8=getattr(language_transformer_config, "fp8", None),
                fp8_recipe=getattr(language_transformer_config, "fp8_recipe", "tensorwise"),
            )
            diffusion_head_a = nn.ModuleList([
                MegatronModulatedAttentionBlock(
                    diff_head_cfg, layer_idx=i, megatron_config=_diff_mcfg,
                    adaln_post_modulation=_post_mod,
                    block_adaln=_block_adaln,
                )
                for i in range(num_diffusion_layers)
            ])
        else:
            raise ValueError(
                f"未知 dit_head_impl={dit_head_impl!r}, 仅支持 'torch' / 'megatron'"
            )
        diffusion_head_b = DiffusionHeadFinalLayer(lm_dim, out_dim)
        return diffusion_head_a, diffusion_head_b

    def _resolve_flat_gen_mask(
        self,
        image_mask: torch.Tensor,
        image_generate_mask: Optional[torch.Tensor],
        total_img_und_mask: Optional[torch.Tensor],
        edit_clean_img_mask: Optional[torch.Tensor],
    ):
        """构造 INPUT-TOKEN 空间的生成 token 掩码 (长度 == image_mask.sum())。

        ``image_generate_mask`` 在【编辑 ViT 模式】下是【ViT pre-merge 空间】(用于
        ``_prepare_vision_embeds`` 里 ``vision_embeds[image_generate_mask]`` 抽取
        gen 特征), 与 input image token 不对齐 —— 参考图在 input 空间是 spatial-merge
        后的 num_patches//4, 但在 ViT 空间是 num_patches。因此 AdaLN / diffusion head
        这类在 input-token 空间消费的逻辑【不能】直接用它, 必须从 understanding /
        clean-ref 掩码推导。

        返回 [num_image_tokens] 的 bool 张量, 无法推断时返回 None。
        """
        num_image_tokens = int(image_mask.sum().item())
        device = image_mask.device

        # 1. 优先从 input-token 空间的 understanding / clean-ref 掩码推导。
        # Pixel-DiT 理解路径的 image_generate_mask 位于 raw/ViT 特征空间；即使长度
        # 偶然与 input token 数相同，也不能覆盖 total_img_und_mask 的路由语义。
        #    gen = 非 understand 且 非 clean-ref 的 image token
        if total_img_und_mask is not None and total_img_und_mask.numel() == num_image_tokens:
            flat = ~total_img_und_mask.to(device=device).bool()
            if edit_clean_img_mask is not None and edit_clean_img_mask.numel() == num_image_tokens:
                flat = flat & (~edit_clean_img_mask.to(device=device).bool())
            return flat

        # 2. image_generate_mask is already in input-token space after collation.
        if image_generate_mask is not None and image_generate_mask.numel() == num_image_tokens:
            return image_generate_mask.to(device=device).bool()

        # 3. 无 understanding 信息且 image_generate_mask 为空: 所有 image token 都是 gen (纯 t2i)
        if image_generate_mask is None or image_generate_mask.numel() == 0:
            return torch.ones(num_image_tokens, dtype=torch.bool, device=device)

        return None

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
        freeze_dit_param: bool = False,
        freeze_pixel_proj_param: bool = False,
        freeze_pixel_dit_gate_param: bool = False,
        freeze_time_embeddings_param: bool = False,
        freeze_pixel_dit_last_n_layers: int = 0,
        freeze_pixel_dit_final_layer: bool = False,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module (patch_embed, blocks, pos_embed).
            freeze_vision_projection (bool): Freeze the vision projection modules (merger and deepstack_merger_list).
            freeze_dit_param (bool): Freeze the Pixel-DiT head, pixel gates, and timestep embeddings.
            freeze_pixel_proj_param (bool): Freeze the noisy-pixel input projection.
            freeze_pixel_dit_gate_param (bool): Freeze shared and separate Pixel-DiT gates.
            freeze_time_embeddings_param (bool): Freeze all diffusion timestep embedders,
                including the ``time_token`` input embedder.
            freeze_pixel_dit_last_n_layers (int): Freeze the last N transformer blocks
                in ``diffusion_head_a``. Each block is frozen as a whole, including
                attention, normalization/AdaLN, and MLP parameters. The final output
            projection ``diffusion_head_b`` is not included.
            freeze_pixel_dit_final_layer (bool): Freeze the final output layer
                ``diffusion_head_b`` without freezing the preceding blocks.
        """
        modules = []

        freeze_pixel_dit_last_n_layers = int(freeze_pixel_dit_last_n_layers or 0)
        if freeze_pixel_dit_last_n_layers < 0:
            raise ValueError(
                "freeze_pixel_dit_last_n_layers must be >= 0, got "
                f"{freeze_pixel_dit_last_n_layers}"
            )

        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and self.vision_model is not None:
            # Freeze vision encoder components (patch_embed, blocks, pos_embed, rotary_pos_emb)
            if hasattr(self.vision_model, "patch_embed"):
                modules.append(self.vision_model.patch_embed)
            if hasattr(self.vision_model, "blocks"):
                modules.append(self.vision_model.blocks)
            if hasattr(self.vision_model, "pos_embed"):
                modules.append(self.vision_model.pos_embed)
            if hasattr(self.vision_model, "rotary_pos_emb"):
                modules.append(self.vision_model.rotary_pos_emb)

        if freeze_vision_projection and self.vision_model is not None:
            # Freeze vision projection components (merger and deepstack_merger_list)
            if hasattr(self.vision_model, "merger"):
                modules.append(self.vision_model.merger)
            if hasattr(self.vision_model, "deepstack_merger_list"):
                modules.append(self.vision_model.deepstack_merger_list)

        if freeze_dit_param:
            for module_name in (
                "diffusion_head_a",
                "diffusion_head_b",
                "pixel_dit_input_projection",
                "pixel_proj",
                "pixel_dit_gate",
                "pixel_dit_ref_gate",
                "time_embeddings",
                "pixel_dit_time_embeddings",
                "time_token",
            ):
                module = getattr(self, module_name, None)
                if module is not None:
                    modules.append(module)

        if freeze_pixel_proj_param and getattr(self, "pixel_proj", None) is not None:
            modules.append(self.pixel_proj)

        if freeze_pixel_dit_gate_param:
            for module_name in ("pixel_dit_gate", "pixel_dit_ref_gate"):
                module = getattr(self, module_name, None)
                if module is not None:
                    modules.append(module)

        if freeze_time_embeddings_param:
            for module_name in (
                "time_embeddings",
                "pixel_dit_time_embeddings",
                "time_token",
            ):
                module = getattr(self, module_name, None)
                if module is not None:
                    modules.append(module)

        if freeze_pixel_dit_last_n_layers:
            diffusion_head_a = getattr(self, "diffusion_head_a", None)
            if diffusion_head_a is not None:
                num_layers = len(diffusion_head_a)
                if freeze_pixel_dit_last_n_layers > num_layers:
                    raise ValueError(
                        "freeze_pixel_dit_last_n_layers cannot exceed the number of "
                        f"Pixel-DiT blocks ({num_layers}), got "
                        f"{freeze_pixel_dit_last_n_layers}"
                    )
                modules.extend(diffusion_head_a[-freeze_pixel_dit_last_n_layers:])

        if freeze_pixel_dit_final_layer:
            diffusion_head_b = getattr(self, "diffusion_head_b", None)
            if diffusion_head_b is not None:
                modules.append(diffusion_head_b)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def _merge_pixel_2x2(xt_raw, image_grid_thw):
        """Merge raster raw pixel patches [N_raw, 768] into 2x2 blocks [N_merged, 3072].

        Each merged token is the channel concatenation laid out as (D, 2, 2);
        this is the exact inverse of ``_unmerge_pixel_2x2``.
        """
        MS = 2
        D = xt_raw.shape[-1]  # 768
        token_counts = (
            image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        ).tolist()
        splits = xt_raw.split(token_counts, dim=0)
        out = []
        for img_tokens, thw in zip(splits, image_grid_thw):
            H, W = int(thw[1]), int(thw[2])
            h_hat, w_hat = H // MS, W // MS
            t = img_tokens.reshape(h_hat, MS, w_hat, MS, D)
            t = t.permute(0, 2, 4, 1, 3)  # [h_hat, w_hat, D, MS, MS]
            out.append(t.reshape(h_hat * w_hat, D * MS * MS))
        return torch.cat(out, dim=0)

    @staticmethod
    def _unmerge_pixel_2x2(xt_merged, image_grid_thw, img_gen_flags=None):
        """Inverse of ``_merge_pixel_2x2``: [N_merged, 3072] -> raster [N_raw, 768].

        img_gen_flags: optional [num_images] bool; in mixed batches only the
        generation images are present in ``xt_merged`` so the grid is filtered.
        """
        MS = 2
        D = xt_merged.shape[-1] // (MS * MS)  # 768
        grid = image_grid_thw
        if img_gen_flags is not None and not bool(img_gen_flags.all()):
            grid = image_grid_thw[img_gen_flags]
        merged_counts = (
            (grid[:, 0] * grid[:, 1] * grid[:, 2]) // (MS * MS)
        ).tolist()
        splits = xt_merged.split(merged_counts, dim=0)
        out = []
        for img_tokens, thw in zip(splits, grid):
            H, W = int(thw[1]), int(thw[2])
            h_hat, w_hat = H // MS, W // MS
            t = img_tokens.reshape(h_hat, w_hat, D, MS, MS)
            t = t.permute(0, 3, 1, 4, 2)  # [h_hat, MS, w_hat, MS, D]
            out.append(t.reshape(H * W, D))
        return torch.cat(out, dim=0)

    def _visualize_vision_data(self, vision_data, image_grid_thw):
        """将 packed vision_data 从 merge-block 顺序还原为 raster 顺序的 pixel patches。

        Return the clean RGB patch target used by the dense Pixel-DiT loss.
        输出: [total_raw_tokens, C*P*P] 即 [N, 768] (C=3, P=16)。
        """
        C, TP = 3, 2
        P = 16
        MS = 2

        unique_shapes = image_grid_thw.unique(dim=0)
        if len(unique_shapes) == 1:
            T = int(unique_shapes[0, 0])
            H, W = int(unique_shapes[0, 1]), int(unique_shapes[0, 2])
            H_blocks, W_blocks = H // MS, W // MS
            n_imgs = image_grid_thw.shape[0]

            patches = vision_data.reshape(
                n_imgs, T, H_blocks, W_blocks, MS, MS, C, TP, P, P
            )
            inv_perm = [0, 1, 7, 6, 2, 4, 8, 3, 5, 9]
            patches = patches.permute(*inv_perm)
            patches = patches.reshape(n_imgs, T * TP, C, H * P, W * P)
            imgs = patches[:, 0]  # [n_imgs, C, H*P, W*P]
            imgs = imgs.reshape(n_imgs, C, H, P, W, P)
            imgs = imgs.permute(0, 2, 4, 1, 3, 5)  # [n_imgs, H, W, C, P, P]
            imgs = imgs.reshape(n_imgs * H * W, C * P * P)
            return imgs.to(vision_data.device, dtype=vision_data.dtype)
        else:
            token_counts = (
                image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
            ).tolist()
            splits = vision_data.split(token_counts, dim=0)
            img_patches_list = []

            for img_tokens, thw in zip(splits, image_grid_thw):
                T = int(thw[0])
                H, W = int(thw[1]), int(thw[2])
                H_blocks, W_blocks = H // MS, W // MS
                patches = img_tokens.reshape(
                    T, H_blocks, W_blocks, MS, MS, C, TP, P, P
                )
                inv_perm = [0, 6, 5, 1, 3, 7, 2, 4, 8]
                patches = patches.permute(*inv_perm)
                patches = patches.reshape(T * TP, C, H * P, W * P)
                img = patches[0]  # [C, H*P, W*P]
                img_patches = img.reshape(C, H, P, W, P)
                img_patches = img_patches.permute(1, 3, 0, 2, 4)
                img_patches = img_patches.reshape(H * W, C * P * P)
                img_patches_list.append(img_patches)

            return torch.cat(img_patches_list, dim=0).to(
                vision_data.device, dtype=vision_data.dtype
            )

    def _prepare_vision_embeds(
        self,
        vision_embeds: torch.Tensor,
        merged_vision_embeds: torch.Tensor,
        image_generate_mask: torch.Tensor,
        image_understand_mask: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        diff_scheduler,
        total_img_und_mask=None,
        edit_clean_img_mask=None,
        sigmas: torch.Tensor = None,
    ):
        """Assemble merged ViT features for the Dense Pixel-DiT sequence."""
        del vision_embeds, image_generate_mask, timesteps, noise, diff_scheduler
        del sigmas
        if total_img_und_mask is None:
            total_img_und_mask = torch.zeros(
                merged_vision_embeds.shape[0], dtype=torch.bool,
                device=merged_vision_embeds.device,
            )
        total_img_und_mask = total_img_und_mask.to(merged_vision_embeds.device).bool()
        if total_img_und_mask.numel() != merged_vision_embeds.shape[0]:
            raise ValueError(
                "Dense Pixel-DiT image mask does not match merged vision tokens: "
                f"mask={total_img_und_mask.numel()} tokens={merged_vision_embeds.shape[0]}"
            )
        if image_understand_mask is None:
            image_understand_mask = total_img_und_mask
        image_understand_mask = image_understand_mask.to(merged_vision_embeds.device).bool()
        if image_understand_mask.numel() != merged_vision_embeds.shape[0]:
            image_understand_mask = total_img_und_mask

        und_embeds = merged_vision_embeds[image_understand_mask]
        gen_mask = ~total_img_und_mask
        if edit_clean_img_mask is not None and edit_clean_img_mask.numel() == gen_mask.numel():
            gen_mask = gen_mask & ~edit_clean_img_mask.to(gen_mask.device).bool()
        gen_embeds = merged_vision_embeds[gen_mask]
        self._pixel_dit_gen_vit_features = gen_embeds
        self._pixel_dit_ref_vit_features = und_embeds

        output = merged_vision_embeds.new_zeros(
            (total_img_und_mask.numel(), merged_vision_embeds.shape[-1])
        )
        if und_embeds.numel() > 0:
            output[total_img_und_mask] = und_embeds
        if gen_embeds.numel() > 0:
            output[gen_mask] = gen_embeds
        return output, None

    def forward(
        self,
        # ----- 以下参数顺序完全对齐原版 Qwen3VLModel -----
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        # ----- 以下为 Transfusion 训练专属新增参数 -----
        timesteps_samples: torch.Tensor = None,  # [num_images] 每张图一个时间步
        timesteps: torch.Tensor = None,  # 每个token对应的时间步
        sigmas: torch.Tensor = None,  # [Total_Image_Tokens] 直接 sigma，shift 采样时优先使用
        noise: torch.Tensor = None,
        diff_scheduler: callable = None,  # 外部传入的加噪函数！
        image_generate_mask=None,
        image_understand_mask=None,
        image_gen_grid_thw: torch.Tensor = None,
        total_img_und_mask=None,
        task_type: torch.Tensor = None,
        edit_clean_img_mask: torch.Tensor = None,  # clean reference/feedback tokens
        predict_x0: bool = False,  # predict x0 instead of velocity (JiT style)
    ) -> dict:
        assert pixel_values_videos is None and video_grid_thw is None, "not support video now"
        assert inference_params is None, "not support inference"

        gen_time_emb = None
        needs_gen_time_emb = True
        if needs_gen_time_emb and timesteps is not None:
            gen_time_emb = self.time_embeddings(timesteps)
        self._gen_time_emb = gen_time_emb
        self._pixel_dit_gen_time_emb = None
        self._pixel_dit_gen_vit_features = None
        self._pixel_dit_ref_vit_features = None
        self._pixel_dit_image_xt = None
        if self.use_pixel_dit and timesteps is not None:
            if self.pixel_dit_time_embeddings is None:
                self._pixel_dit_gen_time_emb = gen_time_emb
            else:
                self._pixel_dit_gen_time_emb = self._embed_pixel_dit_timesteps(
                    timesteps
                )

        video_start_index = 0
        vision_grid_thw = None
        vision_data = None
        deepstack_feature_lists = None

        # 【PP 支持】image_mask 在所有 stage 上都需要，从函数参数重建
        image_mask = None
        if image_input_mask is not None:
            image_mask = image_input_mask
        elif image_grid_thw is not None and input_ids is not None:
            image_mask = (input_ids == self.image_token_id).contiguous()

        # 【新增】：用于保存给 Pipeline 算 Loss 的干净特征
        image_x0 = None

        if self.pre_process:
            if image_grid_thw is not None:
                vision_grid_thw = image_grid_thw
                vision_data = pixel_values
                video_start_index = image_mask.sum().item() if image_mask is not None else 0
                assert video_start_index > 0

            vision_embeds = None
            merged_vision_embeds = None

            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0 and vision_data is not None:
                # Dense Pixel-DiT: add noise to raw RGB patches before the ViT.
                pixel_dit_gen_grid_thw = (
                    image_gen_grid_thw.to(device=vision_grid_thw.device)
                    if image_gen_grid_thw is not None
                    else vision_grid_thw
                )
                raw_sizes = (
                    vision_grid_thw[:, 0]
                    * vision_grid_thw[:, 1]
                    * vision_grid_thw[:, 2]
                ).long()
                flat_gen_mask = self._resolve_flat_gen_mask(
                    image_mask=image_mask,
                    image_generate_mask=image_generate_mask,
                    total_img_und_mask=total_img_und_mask,
                    edit_clean_img_mask=edit_clean_img_mask,
                )
                merged_sizes = raw_sizes // 4
                if flat_gen_mask is None or flat_gen_mask.numel() == 0:
                    flat_gen_mask = torch.zeros(
                        int(merged_sizes.sum().item()), dtype=torch.bool,
                        device=vision_grid_thw.device,
                    )
                if flat_gen_mask.numel() == int(merged_sizes.sum().item()):
                    seg_sizes = merged_sizes
                elif flat_gen_mask.numel() == int(raw_sizes.sum().item()):
                    seg_sizes = raw_sizes
                else:
                    raise ValueError(
                        "Dense Pixel-DiT generation mask does not match image grids: "
                        f"mask={flat_gen_mask.numel()} merged={int(merged_sizes.sum())} "
                        f"raw={int(raw_sizes.sum())}"
                    )
                seg_ids = torch.arange(
                    len(seg_sizes), device=flat_gen_mask.device
                ).repeat_interleave(seg_sizes.to(flat_gen_mask.device))
                image_flags = torch.zeros(
                    len(seg_sizes), dtype=torch.float32, device=flat_gen_mask.device
                )
                image_flags.scatter_add_(0, seg_ids, flat_gen_mask.float())
                image_flags = image_flags.bool()
                if image_gen_grid_thw is None:
                    pixel_dit_gen_grid_thw = vision_grid_thw[image_flags.to(vision_grid_thw.device)]
                elif int(image_flags.sum().item()) != image_gen_grid_thw.shape[0]:
                    raise ValueError("image_gen_grid_thw does not match generation mask")
                pixel_dit_raw_gen_mask = image_flags.to(vision_data.device).repeat_interleave(
                    raw_sizes.to(vision_data.device)
                )
                pixel_dit_raw_ref_mask = None
                pixel_dit_ref_grid_thw = None
                if total_img_und_mask is not None and total_img_und_mask.numel() == flat_gen_mask.numel():
                    ref_flags = torch.zeros(
                        len(seg_sizes), dtype=torch.float32, device=flat_gen_mask.device
                    )
                    ref_flags.scatter_add_(0, seg_ids, total_img_und_mask.float())
                    ref_flags = ref_flags.bool()
                    if ref_flags.any():
                        pixel_dit_raw_ref_mask = ref_flags.to(vision_data.device).repeat_interleave(
                            raw_sizes.to(vision_data.device)
                        )
                        pixel_dit_ref_grid_thw = vision_grid_thw[
                            ref_flags.to(vision_grid_thw.device)
                        ]

                if pixel_dit_raw_gen_mask.any():
                    full_image_x0 = self._visualize_vision_data(
                        vision_data, vision_grid_thw
                    ).detach()
                    image_x0 = full_image_x0[pixel_dit_raw_gen_mask]
                if timesteps is not None and noise is not None and pixel_dit_raw_gen_mask.any():
                    vision_data = vision_data.clone()
                    gen_vision_data = vision_data[pixel_dit_raw_gen_mask]
                    if sigmas is not None:
                        gen_vision_data = diff_scheduler.add_noise_batch_packing_direct(
                            original_samples=gen_vision_data, noise=noise, sigmas=sigmas
                        )
                    else:
                        gen_vision_data = diff_scheduler.add_noise_batch_packing(
                            original_samples=gen_vision_data, noise=noise, timesteps=timesteps
                        )
                    vision_data[pixel_dit_raw_gen_mask] = gen_vision_data.to(vision_data.dtype)
                if pixel_dit_raw_gen_mask is not None:
                    full_image_xt = self._visualize_vision_data(
                        vision_data, vision_grid_thw
                    ).detach()
                    self._pixel_dit_image_xt = full_image_xt[pixel_dit_raw_gen_mask]
                else:
                    self._pixel_dit_image_xt = None
                self._pixel_dit_ref_image_x0 = None
                self._pixel_dit_ref_grid_thw = pixel_dit_ref_grid_thw
                if pixel_dit_raw_ref_mask is not None and pixel_dit_raw_ref_mask.any():
                    if "full_image_x0" not in locals():
                        full_image_x0 = self._visualize_vision_data(
                            vision_data, vision_grid_thw
                        ).detach()
                    self._pixel_dit_ref_image_x0 = full_image_x0[pixel_dit_raw_ref_mask]
                self._pixel_dit_gen_grid_thw = pixel_dit_gen_grid_thw

                (
                    vision_embeds,
                    merged_vision_embeds,
                    deepstack_feature_lists,
                    _,
                    _,
                ) = self.vision_model(
                    hidden_states=vision_data,
                    grid_thw=vision_grid_thw,
                    patch_embed_output=True,
                )
            # Assemble the merged ViT states consumed by the language backbone.
            if vision_embeds is not None:
                video_start_index = image_mask.sum().item() if image_mask is not None else 0
                if image_generate_mask is None:
                    image_generate_mask = ~total_img_und_mask if total_img_und_mask is not None else None
                if image_understand_mask is None:
                    image_understand_mask = total_img_und_mask
                vision_embeds, _ = self._prepare_vision_embeds(
                    vision_embeds=vision_embeds,
                    merged_vision_embeds=merged_vision_embeds,
                    image_generate_mask=image_generate_mask,
                    image_understand_mask=image_understand_mask,
                    timesteps=timesteps,
                    noise=noise,
                    diff_scheduler=diff_scheduler,
                    total_img_und_mask=total_img_und_mask,
                    edit_clean_img_mask=edit_clean_img_mask,
                )

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            # 将 <time> token 的 embedding 替换为时间步编码
            if timesteps_samples is not None and self.pre_process:
                time_mask = input_ids == self.time_token_id  # [B, SeqLen]
                if time_mask.any():
                    # timesteps: [num_images], time_mask 中 True 的数量通常 == num_images
                    time_token_emb = self.time_token(timesteps_samples)  # [num_images, lm_hidden]
                    # 边界情况：truncation 导致某个样本的 <time> token 保留但图像 token 被截掉，
                    # 此时 time_mask.sum() > timesteps_samples.shape[0]，取二者最小值对齐
                    num_time_in_seq = int(time_mask.sum().item())
                    num_time_emb = time_token_emb.shape[0]
                    if num_time_in_seq != num_time_emb:
                        # 只替换有对应时间步的前 N 个 <time> token，多余的保留原始 embedding
                        count = min(num_time_in_seq, num_time_emb)
                        time_positions = time_mask.nonzero(as_tuple=False)[:count]  # [count, 2]
                        combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                        combined_embeddings[time_positions[:, 0], time_positions[:, 1]] = \
                            time_token_emb[:count].to(combined_embeddings.dtype)
                        combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                    else:
                        # combined_embeddings 是 [SeqLen, B, Hidden]，需要转为 [B, SeqLen, Hidden] 来索引
                        combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                        combined_embeddings[time_mask] = time_token_emb.to(combined_embeddings.dtype)
                        combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if vision_embeds is not None:
                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                else:
                    raise ValueError(
                        f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got "
                        f"{video_start_index}"
                    )
                if video_embeds is not None:
                    print(f"video_embeds shape: {video_embeds.shape}")
                    print(f"video_start_index: {video_start_index}")
                assert video_embeds is None, "not support video now"

                if image_embeds is not None:
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                    combined_embeddings[image_mask] = image_embeds
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
            
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None

        # visual_pos_masks: 只标记理解任务的图像位置 (生成任务不使用 deepstack)
        # image_mask: [B, SeqLen] 所有图像 token
        # total_img_und_mask: [total_image_tokens] 理解图像标记
        if image_mask is not None and total_img_und_mask is not None and total_img_und_mask.numel() > 0:
            # 将 total_img_und_mask 展开到 image_mask 的形状
            # image_mask 中 True 的位置与 total_img_und_mask 一一对应
            visual_pos_masks = image_mask.clone()
            visual_pos_masks[image_mask] = total_img_und_mask.to(visual_pos_masks.device)
        else:
            visual_pos_masks = image_mask
        # deepstack 只用于理解任务，过滤掉非 und token。
        # 注意: deepstack_feature 在【merged-ViT 空间】(每图 num_patches//4，spatial-merge 后)，
        # 必须用 image_understand_mask (同样是 merged-ViT 空间) 过滤；
        # total_img_und_mask is in input-token space (including reference,
        # generation, and feedback padding), so it cannot index merged ViT
        # features directly.
        # 纯理解任务两者长度相同(都=merged token 数)，此处行为不变。
        deepstack_visual_embeds = deepstack_feature_lists
        if deepstack_visual_embeds is not None and image_understand_mask is not None \
                and image_understand_mask.numel() > 0:
            if image_understand_mask.any():
                deepstack_visual_embeds = [feat[image_understand_mask] for feat in deepstack_visual_embeds]
            else:
                deepstack_visual_embeds = None
        if self.config.sequence_parallel and deepstack_visual_embeds is not None:
            visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                visual_pos_masks,
                deepstack_visual_embeds,
                tp_size=mpu.get_tensor_model_parallel_world_size(),
                tp_rank=mpu.get_tensor_model_parallel_rank(),
            )

        # 4. 过主干语言网络
        lm_output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=None,
            extra_block_kwargs=extra_block_kwargs,
        )

        # 如果不是最后一张卡，直接把中间状态抛给下一张卡
        if not self.post_process:
            return lm_output

        text_loss = lm_output

        # Pixel-DiT image targets are produced on the vision stage.
        if not self.pre_process:
            if (
                image_mask is not None
                and image_mask.any()
                and total_img_und_mask is not None
                and (~total_img_und_mask).any()
            ):
                raise RuntimeError(
                    "Pipeline-parallel Dense Pixel-DiT generation requires the vision "
                    "and diffusion heads on the same stage."
                )
            else:
                image_x0 = None

        image_pred = None
        assert self._captured_hidden_states is not None
        hidden_states = self._captured_hidden_states
        # SP 模式下 hook 截获的 hidden_states 是 scattered 的 [seq_len/TP, B, H]，
        # 需要 all-gather 恢复完整 sequence 才能用 image_mask 索引。
        if self.config.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False
            )

        image_target = None
        if image_x0 is not None and image_mask is not None:
            # Megatron 默认形状通常是 [Seq_Len, Batch, Hidden_Size]
            # 为了用 image_mask (形状为 [Batch, Seq_Len]) 进行 boolean 切片，我们需要 transpose
            if hidden_states.shape[0] != image_mask.shape[0]:
                hidden_states = hidden_states.transpose(0, 1).contiguous()

            if self.use_pixel_dit:
                # === Pixel-DiT: 完整序列过多层 DiT head + merged 加噪 pixel Gate 注入 ===
                gen_token_mask = image_mask.clone()
                flat_gen_mask = self._resolve_flat_gen_mask(
                    image_mask=image_mask,
                    image_generate_mask=image_generate_mask,
                    total_img_und_mask=total_img_und_mask,
                    edit_clean_img_mask=edit_clean_img_mask,
                )
                gen_token_mask[image_mask] = flat_gen_mask
                ref_token_mask = None  # 完整 input_ids 掩码; True = hidden_states 中的 ref/feedback 图像 token
                ref_gate_input = None  # clean ref/feedback gate input, 与 ref_token_mask 对齐
                if (
                    self.use_pixel_dit_pixel_residual
                    and self.pixel_dit_ref_gate_mode != "none"
                    and total_img_und_mask is not None
                    and total_img_und_mask.numel() == int(image_mask.sum().item())
                ):
                    # input-token 空间; True = image token 中的 clean ref/feedback token。
                    flat_ref_mask = total_img_und_mask.to(
                        device=image_mask.device
                    ).bool()
                    if flat_ref_mask.any():
                        ref_token_mask = image_mask.clone()
                        ref_token_mask[image_mask] = flat_ref_mask
                        ref_pixels = None
                        if self.pixel_dit_gate_input == "pixel":
                            ref_image_x0 = getattr(
                                self, "_pixel_dit_ref_image_x0", None
                            )
                            ref_grid_thw = getattr(
                                self, "_pixel_dit_ref_grid_thw", None
                            )
                            if ref_image_x0 is not None and ref_grid_thw is not None:
                                ref_pixels = self._merge_pixel_2x2(
                                    ref_image_x0, ref_grid_thw
                                )
                        ref_vit_features = getattr(
                            self, "_pixel_dit_ref_vit_features", None
                        )
                        if ref_pixels is not None or ref_vit_features is not None:
                            ref_gate_input = self._prepare_pixel_dit_gate_input(
                                pixel_features=ref_pixels,
                                vit_features=ref_vit_features,
                                expected_tokens=int(ref_token_mask.sum().item()),
                                name="reference",
                            )

                diff_hidden = self._project_pixel_dit_hidden(hidden_states)
                pixel_dit_grid_thw = getattr(
                    self, "_pixel_dit_gen_grid_thw", image_grid_thw
                )

                # HiGate high-info: merged noisy pixels or post-merger ViT feat.
                gate_input = None
                if self.use_pixel_dit_pixel_residual:
                    gen_pixels = None
                    if self.pixel_dit_gate_input == "pixel":
                        gen_pixels = self._merge_pixel_2x2(
                            self._pixel_dit_image_xt, pixel_dit_grid_thw
                        )
                    gate_input = self._prepare_pixel_dit_gate_input(
                        pixel_features=gen_pixels,
                        vit_features=getattr(
                            self, "_pixel_dit_gen_vit_features", None
                        ),
                        expected_tokens=int(gen_token_mask.sum().item()),
                        name="generation",
                    )

                # per-token time_emb 需 merged 长度 (DiT 在 merged 分辨率跑)。
                # self._gen_time_emb 是 raw 长度 (image_gen_seq_lens=raw), 需降采样:
                # 同图同 timestep, 每张图取起始 token 的 emb, 再按 merged 数铺开。
                _raw_sizes = (
                    pixel_dit_grid_thw[:, 0]
                    * pixel_dit_grid_thw[:, 1]
                    * pixel_dit_grid_thw[:, 2]
                )
                _merged_sizes = _raw_sizes // 4
                pixel_dit_time_emb = self._pixel_dit_gen_time_emb
                if pixel_dit_time_emb is None:
                    raise RuntimeError("Pixel-DiT timestep embedding was not initialized")
                _dev_t = pixel_dit_time_emb.device
                _raw_cumsum = torch.cat([
                    torch.zeros(1, device=_dev_t, dtype=torch.long),
                    _raw_sizes.cumsum(0)[:-1],
                ])
                _per_img_time_emb = pixel_dit_time_emb[_raw_cumsum]  # [N_img, H]
                _merged_img_idx = torch.arange(
                    len(_merged_sizes), device=_dev_t
                ).repeat_interleave(_merged_sizes)
                merged_gen_time_emb = _per_img_time_emb[_merged_img_idx]  # [N_merged, H]

                diff_position_ids = position_ids
                diff_startend = None
                if extra_block_kwargs is not None:
                    diff_startend = extra_block_kwargs.get("startend_row_indices")

                inject_layer = self.pixel_dit_gate_inject_layer
                num_layers = len(self.diffusion_head_a)
                if self.use_pixel_dit_pixel_residual and inject_layer >= num_layers:
                    import logging
                    logging.getLogger(__name__).warning(
                        "pixel_dit_gate_inject_layer=%d >= num_layers=%d, "
                        "Gate 不会注入 (纯 backbone hidden 进 DiT)",
                        inject_layer, num_layers,
                    )
                for i, layer in enumerate(self.diffusion_head_a):
                    if i == inject_layer and self.use_pixel_dit_pixel_residual:
                        has_ref = (
                            ref_gate_input is not None and ref_token_mask is not None
                        )
                        if has_ref and self.pixel_dit_ref_gate_mode == "shared":
                            # shared: ref/gen 复用同一个 pixel_dit_gate, 且 ref/gen token
                            # 互不相交。HiGate 为逐 token 运算 (Linear/SiLU/LayerNorm),
                            # 故拼成一次前向与分别两次调用数学等价, 省一次 gate 前向 +
                            # 一组 mask gather/scatter。
                            n_ref = ref_gate_input.shape[0]
                            fused = self.pixel_dit_gate(
                                torch.cat(
                                    [diff_hidden[ref_token_mask], diff_hidden[gen_token_mask]],
                                    dim=0,
                                ),
                                torch.cat([ref_gate_input, gate_input], dim=0),
                            ).to(diff_hidden.dtype)
                            diff_hidden[ref_token_mask] = fused[:n_ref]
                            diff_hidden[gen_token_mask] = fused[n_ref:]
                        else:
                            if has_ref:
                                # separate: ref 走独立 gate
                                ref_gate = self.pixel_dit_ref_gate
                                if ref_gate is None:
                                    raise RuntimeError(
                                        "pixel_dit_ref_gate_mode='separate' requires "
                                        "self.pixel_dit_ref_gate to be initialized."
                                    )
                                fused_ref = ref_gate(
                                    diff_hidden[ref_token_mask], ref_gate_input
                                )
                                diff_hidden[ref_token_mask] = fused_ref.to(diff_hidden.dtype)
                            fused = self.pixel_dit_gate(
                                diff_hidden[gen_token_mask], gate_input
                            )
                            diff_hidden[gen_token_mask] = fused.to(diff_hidden.dtype)
                    if self.diffusion_head_recompute and self.training:
                        diff_hidden = torch.utils.checkpoint.checkpoint(
                            self._dit_head_layer_forward,
                            layer,
                            diff_hidden,
                            merged_gen_time_emb,
                            gen_token_mask,
                            diff_position_ids,
                            diff_startend,
                            use_reentrant=False,
                        )
                    else:
                        diff_hidden = self._dit_head_layer_forward(
                            layer,
                            diff_hidden,
                            merged_gen_time_emb,
                            gen_token_mask,
                            diff_position_ids,
                            diff_startend,
                        )
                x0_full = self.diffusion_head_b(
                    diff_hidden, merged_gen_time_emb, gen_token_mask
                )
                image_pred_merged = x0_full[gen_token_mask]  # [N_merged, 3072]

                # unmerge merged [N_merged, 3072] → raw [N_raw, 768]
                image_pred = self._unmerge_pixel_2x2(
                    image_pred_merged, pixel_dit_grid_thw
                )

                # Pixel-DiT supports both velocity and clean-x0 targets.  Keep
                # this branch aligned with the generic diffusion heads below.
                self._captured_hidden_states = None
                if predict_x0:
                    image_target = diff_scheduler.training_target_x0(
                        image_x0, noise
                    )
                else:
                    image_target = diff_scheduler.training_target(
                        image_x0, noise
                    )
                image_target = image_target.to(dtype=image_pred.dtype)

        self._captured_hidden_states = None
        if hasattr(self, '_pixel_dit_image_xt'):
            self._pixel_dit_image_xt = None
        if hasattr(self, '_pixel_dit_gen_grid_thw'):
            self._pixel_dit_gen_grid_thw = None

        return {
            "text_loss": text_loss,
            "image_pred": image_pred,
            "image_target": image_target,
            "timesteps_packed": timesteps,
        }
