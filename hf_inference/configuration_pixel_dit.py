"""Configuration shared by HF export and standalone Pixel-DiT inference."""

import math

from transformers import Qwen3VLConfig


PIXEL_DIT_DEFAULTS = {
    "use_pixel_dit": True,
    "num_diffusion_layers": 10,
    "pixel_dit_hidden_size": 0,
    "pixel_dit_use_text_tokens": True,
    "num_diffusion_heads": 0,
    "num_diffusion_kv_heads": 0,
    "diffusion_intermediate_size": 0,
    "pixel_dit_gate_inject_layer": 3,
    "pixel_dit_gate_input": "pixel",
    "pixel_dit_ref_gate_mode": "none",
    "use_pixel_dit_pixel_residual": True,
    "dit_block_adaln": True,
    "adaln_post_modulation": False,
    "dit_head_impl": "torch",
    "predict_x0": True,
    "predict_gate_mode": False,
    "use_dynamic_noise_scale": True,
    "dynamic_noise_ref_tokens": 256,
    "dynamic_noise_base_scale": 1.0,
    "time_token_id": 151661,
    "image_gate_token_id": 151670,
}


def reject_enabled_refiner(values):
    """Accept the disabled post-DiT refiner keys older run configs still carry.

    The refiner was removed from this release; a checkpoint that enabled it
    has weights this code cannot load, so fail instead of dropping them.
    """
    if (
        values.get("pit_depth")
        or values.get("pixel_dit_post_upsample_attention_depth")
        or values.get("use_pixel_dit_post_upsample_attention")
        or values.get("train_pit_only")
        or str(values.get("pixel_dit_refiner_type") or "").lower() == "sensenova_u1"
    ):
        raise ValueError("refiner checkpoints are not supported by this release")


def validate_architecture(values):
    """Reject training variants that this inference implementation cannot run."""
    if not values.get("use_pixel_dit", False):
        raise ValueError("This release supports use_pixel_dit=true checkpoints only")
    unsupported = (
        "use_mot", "use_mot_backbone_pixel", "use_cheers", "use_vae_server",
        "use_rae", "use_adaln_backbone",
    )
    for name in unsupported:
        if values.get(name, False):
            raise ValueError(f"Unsupported architecture flag: {name}=true")
    if "mot" in str(values.get("_target_", "")).lower():
        raise ValueError("MoT checkpoints need a separate HF model implementation")
    reject_enabled_refiner(values)
    if values.get("dit_head_impl", "torch") != "torch":
        raise ValueError("This release requires dit_head_impl=torch")
    if int(values.get("num_diffusion_layers", 0)) <= 0:
        raise ValueError("num_diffusion_layers must be positive")
    if values.get("pixel_dit_gate_input", "pixel") not in ("pixel", "vit"):
        raise ValueError("pixel_dit_gate_input must be pixel or vit")
    if values.get("pixel_dit_ref_gate_mode", "none") not in ("none", "shared", "separate"):
        raise ValueError("Invalid pixel_dit_ref_gate_mode")
    if values.get("dynamic_noise_ref_tokens", 256) <= 0:
        raise ValueError("dynamic_noise_ref_tokens must be positive")
    noise_base = values.get("dynamic_noise_base_scale", 1.0)
    if not math.isfinite(noise_base) or noise_base <= 0:
        raise ValueError("dynamic_noise_base_scale must be finite and positive")


class PixelDiTConfig(Qwen3VLConfig):
    """Qwen3-VL configuration plus the generation architecture and noise settings."""

    model_type = "pixel_dit_qwen3_vl"

    def __init__(self, **kwargs):
        """Load generation settings alongside standard Qwen3-VL subconfigs."""
        values = {**PIXEL_DIT_DEFAULTS, **kwargs}
        validate_architecture(values)
        super().__init__(**values)
        # Transformers uses the text subconfig when tying multimodal LM weights.
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
