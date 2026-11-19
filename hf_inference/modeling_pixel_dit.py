"""Qwen3-VL Pixel-DiT inference using only PyTorch and Transformers."""

import math

import torch
from torch import nn
from transformers import Qwen3VLForConditionalGeneration

from .configuration_pixel_dit import PixelDiTConfig
from .diffusion_head import DiffHeadConfig, DiffusionHeadFinalLayer, ModulatedAttentionBlock
from .sampling import generate_image, merge_patches, unmerge_patches, vision_patches


class TimestepEmbedder(nn.Module):
    """Training-compatible sinusoidal time embedding followed by a GELU MLP."""

    def __init__(self, hidden_size):
        """Create the checkpoint's 256-channel time projection."""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, timesteps):
        """Embed timesteps in the training scale of zero to one thousand."""
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=timesteps.device).float() / 128)
        phase = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class HiGate(nn.Module):
    """Fuse backbone information with normalized noisy pixel or ViT features."""

    def __init__(self, dim):
        """Match all gate and normalization parameters in the training checkpoint."""
        super().__init__()
        self.gate_fc1 = nn.Linear(dim, dim)
        self.gate_fc2 = nn.Linear(dim, 1)
        self.layer_norm = nn.LayerNorm(dim)
        self.hi_info_norm = nn.LayerNorm(dim)

    def forward(self, low, high):
        """Apply the unbounded learned gate used during training."""
        gate = self.gate_fc2(torch.nn.functional.silu(self.gate_fc1(low)))
        return self.layer_norm(low + gate * self.hi_info_norm(high))


class PixelDiTForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Standard HF Qwen3-VL backbone with a checkpoint-compatible Pixel-DiT head."""

    config_class = PixelDiTConfig
    _supports_flash_attn = False

    def __init__(self, config):
        """Build the backbone and the generation modules described by config.json."""
        vision = config.vision_config
        if (vision.patch_size, vision.spatial_merge_size, vision.temporal_patch_size) != (16, 2, 2):
            raise ValueError("Pixel-DiT requires vision patch_size=16, merge_size=2, temporal_patch_size=2")
        super().__init__(config)
        text = config.text_config
        dim = config.pixel_dit_hidden_size or text.hidden_size
        self.time_token = TimestepEmbedder(text.hidden_size)
        self.time_embeddings = TimestepEmbedder(text.hidden_size)
        self.pixel_dit_input_projection = nn.Identity() if dim == text.hidden_size else nn.Linear(text.hidden_size, dim)
        self.pixel_dit_time_embeddings = None if dim == text.hidden_size else TimestepEmbedder(dim)
        head_config = DiffHeadConfig(
            hidden_size=dim,
            num_attention_heads=config.num_diffusion_heads or text.num_attention_heads,
            num_key_value_heads=config.num_diffusion_kv_heads or text.num_key_value_heads,
            intermediate_size=config.diffusion_intermediate_size or text.intermediate_size,
            max_position_embeddings=text.max_position_embeddings,
        )
        self.diffusion_head_a = nn.ModuleList([
            ModulatedAttentionBlock(head_config, i, config.adaln_post_modulation, config.dit_block_adaln)
            for i in range(config.num_diffusion_layers)
        ])
        self.diffusion_head_b = DiffusionHeadFinalLayer(dim, 3072)
        self.pixel_proj = nn.Linear(3072, dim)
        self.pixel_dit_gate = HiGate(dim)
        self.pixel_dit_ref_gate = HiGate(dim) if config.pixel_dit_ref_gate_mode == "separate" else None
        self.post_init()

    @torch.inference_mode()
    def predict_image(self, ids, latents, timesteps, height, width, positions, attention_mask,
                      image_mask, drop_dit_text=False):
        """Predict raw raster patches from one conditional or unconditional sequence."""
        grid = torch.tensor([[1, height * 2, width * 2]], device=latents.device)
        visual_output = self.model.visual(vision_patches(latents.to(self.dtype), height, width), grid_thw=grid)
        vision_features = visual_output[0]
        embeddings = self.get_input_embeddings()(ids).clone()
        embeddings[image_mask] = vision_features.to(embeddings.dtype)
        embeddings[ids.eq(self.config.time_token_id)] = self.time_token(timesteps).to(embeddings.dtype)
        # Generation uses no DeepStack injection, matching transfusion_sampler.py.
        hidden = self.model.language_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                                           position_ids=positions, use_cache=False).last_hidden_state
        hidden = self.pixel_dit_input_projection(hidden)
        embedder = self.pixel_dit_time_embeddings or self.time_embeddings
        time = embedder(timesteps).expand(height * width, -1)
        gate_input = None
        if self.config.use_pixel_dit_pixel_residual:
            if self.config.pixel_dit_gate_input == "pixel":
                gate_input = self.pixel_proj(merge_patches(latents.to(self.dtype), height, width))
            else:
                gate_input = self.pixel_dit_input_projection(vision_features)
        head_mask = attention_mask
        if drop_dit_text:
            head_mask = attention_mask.clone()
            start = int(ids[0].eq(self.config.vision_start_token_id).nonzero()[0])
            head_mask[:, :, start:, :start] = torch.finfo(head_mask.dtype).min
        if not self.config.pixel_dit_use_text_tokens:
            # Standalone sampling uses one sequence per call. Keep original
            # M-RoPE coordinates and the attention submatrix, as in training.
            keep = (image_mask | ids.eq(self.config.time_token_id))[0]
            hidden = hidden[:, keep]
            positions = positions[..., keep]
            image_mask = image_mask[:, keep]
            head_mask = head_mask[:, :, keep, :][:, :, :, keep]
        for i, layer in enumerate(self.diffusion_head_a):
            if i == self.config.pixel_dit_gate_inject_layer and gate_input is not None:
                hidden[image_mask] = self.pixel_dit_gate(hidden[image_mask], gate_input).to(hidden.dtype)
            hidden = layer(hidden, time, image_mask, attention_mask=head_mask, position_ids=positions)[0]
        output = self.diffusion_head_b(hidden, time, image_mask)[image_mask]
        return unmerge_patches(output, height, width).float()

    def generate_image(self, prompt, tokenizer=None, **kwargs):
        """Return a PIL image; height and width are specified in pixels."""
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        return generate_image(self, tokenizer, prompt, **kwargs)
