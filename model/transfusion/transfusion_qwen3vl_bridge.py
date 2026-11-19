"""
Transfusion Qwen3-VL Bridge: Megatron <-> HuggingFace checkpoint conversion.

Extends the standard dense Qwen3VLBridge with mappings for the
Pixel-DiT-specific diffusion heads and pixel conditioning modules.

Usage:
    # Register this bridge before using AutoBridge
    import model.transfusion.transfusion_qwen3vl_bridge  # noqa: F401

    from megatron.bridge import AutoBridge
    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)
    bridge.export_ckpt(megatron_path, hf_path)
"""

import dataclasses

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLBridge

from model.transfusion.transfusion_qwen3vl_model import TransfusionQwen3VLModel
from model.transfusion.transfusion_qwen3vl_provider import TransfusionQwen3VLProvider


def _build_transfusion_mappings() -> list:
    """Build mappings for parameters added by dense Pixel-DiT."""
    mappings = [
        # ---- time_embeddings (TimestepEmbedder: nn.Sequential MLP) ----
        # mlp.0 = nn.Linear (frequency_embedding_size -> hidden_size)
        ReplicatedMapping(
            megatron_param="time_embeddings.mlp.0.weight",
            hf_param="transfusion.time_embeddings.mlp.0.weight",
        ),
        ReplicatedMapping(
            megatron_param="time_embeddings.mlp.0.bias",
            hf_param="transfusion.time_embeddings.mlp.0.bias",
        ),
        # mlp.2 = nn.Linear (hidden_size -> hidden_size)
        ReplicatedMapping(
            megatron_param="time_embeddings.mlp.2.weight",
            hf_param="transfusion.time_embeddings.mlp.2.weight",
        ),
        ReplicatedMapping(
            megatron_param="time_embeddings.mlp.2.bias",
            hf_param="transfusion.time_embeddings.mlp.2.bias",
        ),

        # ---- time_token (TimestepEmbedder: nn.Sequential MLP) ----
        ReplicatedMapping(
            megatron_param="time_token.mlp.0.weight",
            hf_param="transfusion.time_token.mlp.0.weight",
        ),
        ReplicatedMapping(
            megatron_param="time_token.mlp.0.bias",
            hf_param="transfusion.time_token.mlp.0.bias",
        ),
        ReplicatedMapping(
            megatron_param="time_token.mlp.2.weight",
            hf_param="transfusion.time_token.mlp.2.weight",
        ),
        ReplicatedMapping(
            megatron_param="time_token.mlp.2.bias",
            hf_param="transfusion.time_token.mlp.2.bias",
        ),

        # ---- ViT channel normalization buffers ----
        ReplicatedMapping(
            megatron_param="vit_channel_mean",
            hf_param="transfusion.vit_channel_mean",
        ),
        ReplicatedMapping(
            megatron_param="vit_channel_std",
            hf_param="transfusion.vit_channel_std",
        ),
        ReplicatedMapping(
            megatron_param="vit_stats_count",
            hf_param="transfusion.vit_stats_count",
        ),

        # ---- Pixel-DiT gates ----
        # gate_fc1: ColumnParallelLinear(embed_dim -> embed_dim), has weight + bias
        # gate_fc2: RowParallelLinear(embed_dim -> 1), has weight + bias
        # layer_norm: nn.LayerNorm(embed_dim), has weight + bias
        ReplicatedMapping(
            megatron_param="hi_gate.gate_fc1.weight",
            hf_param="transfusion.hi_gate.gate_fc1.weight",
        ),
        ReplicatedMapping(
            megatron_param="hi_gate.gate_fc1.bias",
            hf_param="transfusion.hi_gate.gate_fc1.bias",
        ),
        ReplicatedMapping(
            megatron_param="hi_gate.gate_fc2.weight",
            hf_param="transfusion.hi_gate.gate_fc2.weight",
        ),
        ReplicatedMapping(
            megatron_param="hi_gate.gate_fc2.bias",
            hf_param="transfusion.hi_gate.gate_fc2.bias",
        ),
        ReplicatedMapping(
            megatron_param="hi_gate.layer_norm.weight",
            hf_param="transfusion.hi_gate.layer_norm.weight",
        ),
        ReplicatedMapping(
            megatron_param="hi_gate.layer_norm.bias",
            hf_param="transfusion.hi_gate.layer_norm.bias",
        ),
        # ---- Multi-layer diffusion head: diffusion_head_a (N x ModulatedAttentionBlock) ----
        # Each layer i has the following parameters:
        #   self_attn.{q,k,v,o}_proj.{weight,bias}
        #   mlp.{gate,up,down}_proj.weight
        #   input_layernorm.weight
        #   post_attention_layernorm.weight
        #   adaLN_modulation.1.{weight,bias}
        #   self_attn.{q,k}_norm.weight  (qk_norm=True)
        # All are replicated (no TP sharding) since this head runs outside Megatron TP.
        *[
            ReplicatedMapping(
                megatron_param=f"diffusion_head_a.{i}.{param}",
                hf_param=f"transfusion.diffusion_head_a.{i}.{param}",
            )
            for i in range(32)  # upper bound; unused mappings are silently ignored
            for param in [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "adaLN_modulation.1.weight",
                "adaLN_modulation.1.bias",
            ]
        ],

        # ---- Multi-layer diffusion head: diffusion_head_b (DiffusionHeadFinalLayer) ----
        ReplicatedMapping(
            megatron_param="diffusion_head_b.norm_final.weight",
            hf_param="transfusion.diffusion_head_b.norm_final.weight",
        ),
        ReplicatedMapping(
            megatron_param="diffusion_head_b.linear.weight",
            hf_param="transfusion.diffusion_head_b.linear.weight",
        ),
        ReplicatedMapping(
            megatron_param="diffusion_head_b.linear.bias",
            hf_param="transfusion.diffusion_head_b.linear.bias",
        ),
        ReplicatedMapping(
            megatron_param="diffusion_head_b.adaLN_modulation.1.weight",
            hf_param="transfusion.diffusion_head_b.adaLN_modulation.1.weight",
        ),
        ReplicatedMapping(
            megatron_param="diffusion_head_b.adaLN_modulation.1.bias",
            hf_param="transfusion.diffusion_head_b.adaLN_modulation.1.bias",
        ),
    ]

    # Pixel-DiT parameters are also extracted to the exporter's sidecar. They
    # still need mappings here: an unmapped parameter produces a None task and
    # interrupts the Bridge stream before sidecar extraction can run.
    pixel_linear_modules = (
        "pixel_proj",
        "pixel_dit_input_projection",
        "pixel_dit_time_embeddings.mlp.0",
        "pixel_dit_time_embeddings.mlp.2",
    )
    pixel_gate_modules = tuple(
        f"{gate}.{layer}"
        for gate in ("pixel_dit_gate", "pixel_dit_ref_gate")
        for layer in ("gate_fc1", "gate_fc2", "layer_norm", "hi_info_norm")
    )
    for module_name in pixel_linear_modules + pixel_gate_modules:
        for parameter_name in ("weight", "bias"):
            name = f"{module_name}.{parameter_name}"
            mappings.append(
                ReplicatedMapping(megatron_param=name, hf_param=f"transfusion.{name}")
            )

    # These HF keys don't exist in the original Qwen3-VL model —
    # they are new parameters added by transfusion training.
    # Set allow_hf_name_mismatch so the bridge doesn't skip them.
    for m in mappings:
        m.allow_hf_name_mismatch = True

    return mappings


@MegatronModelBridge.register_bridge(
    source="TransfusionQwen3VLForConditionalGeneration",
    target=TransfusionQwen3VLModel,
)
class TransfusionQwen3VLBridge(Qwen3VLBridge):
    """
    Bridge for converting between HuggingFace and Megatron formats
    for the Transfusion Qwen3-VL (Dense) model.

    Inherits all standard Qwen3-VL language + vision mappings from Qwen3VLBridge,
    and adds mappings for transfusion-specific parameters.
    """

    def provider_bridge(self, hf_pretrained) -> TransfusionQwen3VLProvider:
        """Create a TransfusionQwen3VLProvider from a HuggingFace pretrained config."""
        base_provider = super().provider_bridge(hf_pretrained)
        kwargs = {f.name: getattr(base_provider, f.name) for f in dataclasses.fields(base_provider)}
        return TransfusionQwen3VLProvider(**kwargs)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Merge base Qwen3-VL mappings with transfusion-specific mappings."""
        base_registry = super().mapping_registry()
        all_mappings = list(base_registry.mappings) + _build_transfusion_mappings()
        return MegatronMappingRegistry(*all_mappings)
