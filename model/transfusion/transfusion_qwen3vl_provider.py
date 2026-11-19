"""Megatron provider for the dense Qwen3-VL Pixel-DiT model."""

from dataclasses import dataclass

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider

from model.transfusion.transfusion_qwen3vl_model import TransfusionQwen3VLModel


def _transfusion_provide(self, pre_process=None, post_process=None, vp_stage=None):
    """Construct the dense Pixel-DiT model and apply configured freezes."""
    language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=self.qk_layernorm,
        normalization="RMSNorm",
        fp8=False,
    )
    model = TransfusionQwen3VLModel(
        language_transformer_config=self,
        language_transformer_layer_spec=language_transformer_layer_spec,
        vision_transformer_config=self.vision_config,
        pre_process=pre_process,
        post_process=post_process,
    )

    freeze_flags = {
        "freeze_language_model": self.freeze_language_model,
        "freeze_vision_model": self.freeze_vision_model,
        "freeze_vision_projection": self.freeze_vision_projection,
        "freeze_dit_param": getattr(self, "freeze_dit_param", False),
        "freeze_pixel_proj_param": getattr(self, "freeze_pixel_proj_param", False),
        "freeze_pixel_dit_gate_param": getattr(
            self, "freeze_pixel_dit_gate_param", False
        ),
        "freeze_time_embeddings_param": getattr(
            self, "freeze_time_embeddings_param", False
        ),
        "freeze_pixel_dit_last_n_layers": getattr(
            self, "freeze_pixel_dit_last_n_layers", 0
        ),
        "freeze_pixel_dit_final_layer": getattr(
            self, "freeze_pixel_dit_final_layer", False
        ),
    }
    if any(value for value in freeze_flags.values()):
        model.freeze(**freeze_flags)

    for parameter in model.parameters():
        if not hasattr(parameter, "main_grad"):
            parameter.main_grad = None
    return model


@dataclass
class TransfusionQwen3VLProvider(Qwen3VLModelProvider):
    """Dense Qwen3-VL provider with the Pixel-DiT head."""

    vit_recompute: bool = False
    diffusion_head_recompute: bool = False
    predict_x0: bool = False
    diffusion_loss_type: str = None
    diffusion_loss_reduction: str = None

    pixel_dit_hidden_size: int = 0
    pixel_dit_gate_inject_layer: int = 0
    pixel_dit_gate_input: str = "pixel"
    use_pixel_dit_pixel_residual: bool = True
    pixel_dit_ref_gate_mode: str = "none"

    freeze_dit_param: bool = False
    freeze_pixel_proj_param: bool = False
    freeze_pixel_dit_gate_param: bool = False
    freeze_time_embeddings_param: bool = False
    freeze_pixel_dit_last_n_layers: int = 0
    freeze_pixel_dit_final_layer: bool = False

    adaln_post_modulation: bool = False
    dit_block_adaln: bool = True
    num_diffusion_layers: int = 0
    num_diffusion_heads: int = 0
    num_diffusion_kv_heads: int = 0
    diffusion_intermediate_size: int = 0
    dit_head_impl: str = "torch"

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        return _transfusion_provide(self, pre_process, post_process, vp_stage)
