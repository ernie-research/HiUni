"""Pixel layout, attention masks and flow matching used by HF image generation."""

import math
import random

import torch


INSTRUCTIONS = [
    "\u8bf7\u6839\u636e\u4ee5\u4e0b\u63cf\u8ff0\u751f\u6210\u4e00\u5f20\u56fe\u7247\uff1a",
    "\u6839\u636e\u4e0b\u9762\u7684\u63cf\u8ff0\u751f\u6210\u56fe\u7247\uff1a",
    "\u751f\u6210\u56fe\u7247\uff1a", "\u8bf7\u753b\u4e00\u5f20\u56fe\uff1a",
    "\u5e2e\u6211\u751f\u6210\u4e00\u5f20\u56fe\u7247\uff1a",
    "\u6839\u636e\u63cf\u8ff0\u521b\u5efa\u56fe\u50cf\uff1a",
    "\u8bf7\u6839\u636e\u6587\u5b57\u63cf\u8ff0\u7ed8\u5236\u56fe\u7247\uff1a",
    "\u6309\u7167\u4ee5\u4e0b\u5185\u5bb9\u751f\u6210\u56fe\u7247\uff1a",
    "\u751f\u6210\u4e00\u5f20\u7b26\u5408\u63cf\u8ff0\u7684\u56fe\u7247\uff1a",
    "\u8bf7\u4e3a\u4ee5\u4e0b\u5185\u5bb9\u521b\u4f5c\u4e00\u5f20\u56fe\u7247\uff1a",
    "\u6839\u636e\u63d0\u793a\u751f\u6210\u56fe\u50cf\uff1a", "\u56fe\u7247\u751f\u6210\uff1a",
    "Generate an image based on the following description: ", "Create an image: ",
    "Please generate an image: ", "Draw an image based on: ", "Create a picture: ",
    "Generate: ", "Please draw: ", "Create an image from the description: ",
    "Generate a picture: ", "Please create an image based on: ",
]


def merge_patches(raw, height, width):
    """Pack raster patches as [patch_channel, block_row, block_column]."""
    dim = raw.shape[-1]
    return raw.reshape(height, 2, width, 2, dim).permute(0, 2, 4, 1, 3).reshape(-1, dim * 4)


def unmerge_patches(merged, height, width):
    """Invert the checkpoint's 2x2 pixel packing."""
    dim = merged.shape[-1] // 4
    return merged.reshape(height, width, dim, 2, 2).permute(0, 3, 1, 4, 2).reshape(-1, dim)


def vision_patches(raw, height, width, patch_size=16):
    """Convert raster RGB patches to Qwen's block-major, duplicated-frame layout."""
    patches = raw.reshape(height, 2, width, 2, 3, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4, 5, 6)
    return patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1).reshape(
        4 * height * width, 6 * patch_size * patch_size
    )


def build_sequence(tokenizer, config, prompt, height, width, instruction="", chat=True, route=False):
    """Build the training chat template without retokenizing instruction boundaries."""
    def encode(text):
        """Encode text without injecting tokenizer-level special tokens."""
        return tokenizer.encode(text, add_special_tokens=False)

    image = [config.vision_start_token_id, config.time_token_id]
    image += [config.image_token_id] * (height * width)
    image += [config.vision_end_token_id, 151645]
    if not chat:
        return torch.tensor([image], dtype=torch.long)
    prefix = [151644] + encode("user") + encode("\n")
    prefix += encode(instruction) + encode(prompt) + [151645] + encode("\n")
    prefix += [151644] + encode("assistant") + encode("\n")
    if route:
        if encode("<|image_gate|>") != [config.image_gate_token_id]:
            raise ValueError("Tokenizer does not contain the configured image gate token")
        prefix.append(config.image_gate_token_id)
    return torch.tensor([prefix + image], dtype=torch.long)


def sequence_layout(ids, config, height, width, dtype):
    """Return M-RoPE positions and causal attention with bidirectional image tokens."""
    image_mask = ids.eq(config.image_token_id)
    indices = image_mask[0].nonzero(as_tuple=True)[0]
    if indices.numel() != height * width:
        raise ValueError("Prompt must not contain image special tokens")
    start, end = int(indices[0]), int(indices[-1]) + 1
    if end - start != height * width or ids.eq(config.time_token_id).sum() != 1:
        raise ValueError("Expected one contiguous image block and one time token")
    length = ids.shape[1]
    positions = torch.arange(length, device=ids.device).view(1, 1, -1).repeat(3, 1, 1)
    positions[0, 0, start:end] = start
    positions[1, 0, start:end] = start + torch.arange(height, device=ids.device).repeat_interleave(width)
    positions[2, 0, start:end] = start + torch.arange(width, device=ids.device).repeat(height)
    positions[:, 0, end:] = start + max(height, width) + torch.arange(length - end, device=ids.device)
    allowed = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril()
    allowed[start:end, start:end] = True
    # The Megatron startend mask has LTS=L-1, masking the final EOS row.
    # Its output is unused by the image head; keep that row valid for SDPA.
    mask = torch.zeros(length, length, device=ids.device, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return positions, mask[None, None], image_mask


def flow_sigmas(steps, shift, device):
    """Reproduce FlowMatchScheduler(extra_one_step=True), including its sigma floor."""
    if steps <= 0 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("steps and shift must be positive")
    sigma = torch.linspace(1.0, 0.003 / 1.002, steps + 1, device=device)[:-1]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return torch.cat((sigma, sigma.new_zeros(1)))


def flow_step(prediction, latents, sigma, next_sigma, predict_x0):
    """Match Pixel-DiT Euler sampling, which disables the training loss sigma clamp."""
    velocity = (latents - prediction) / sigma if predict_x0 else prediction
    return latents + velocity * (next_sigma - sigma)


@torch.inference_mode()
def generate_image(model, tokenizer, prompt, height=1024, width=1024, num_steps=50,
                   guidance_scale=4.0, shift=4.0, seed=42, negative_prompt="",
                   cfg_keep_chat_template=False, instruction=None, predict_x0=None,
                   drop_dit_text=False, dynamic_noise_scale=None, dynamic_noise_ref_tokens=None,
                   dynamic_noise_base_scale=None):
    """Generate one PIL image; dimensions are pixels and seeds are local to the call."""
    from PIL import Image

    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("height and width must be positive multiples of 32 pixels")
    if not math.isfinite(guidance_scale) or guidance_scale < 1:
        raise ValueError("guidance_scale must be finite and at least 1")
    model.eval()
    config = model.config
    device, dtype = model.device, model.dtype
    h, w = height // 32, width // 32
    prefix = random.Random(seed).choice(INSTRUCTIONS) if instruction is None else instruction
    cond = build_sequence(tokenizer, config, prompt, h, w, prefix, route=config.predict_gate_mode).to(device)
    sequences = [cond]
    if guidance_scale > 1:
        sequences.append(build_sequence(tokenizer, config, negative_prompt, h, w,
                                        prefix if cfg_keep_chat_template else "",
                                        chat=cfg_keep_chat_template).to(device))
    layouts = [sequence_layout(ids, config, h, w, dtype) for ids in sequences]
    max_length = config.text_config.max_position_embeddings
    if max(ids.shape[1] for ids in sequences) > max_length:
        raise ValueError(f"Prompt and image exceed max_position_embeddings={max_length}")
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(4 * h * w, 768, generator=generator, device=device, dtype=torch.float32)
    dynamic_noise_scale = config.use_dynamic_noise_scale if dynamic_noise_scale is None else dynamic_noise_scale
    noise_ref = config.dynamic_noise_ref_tokens if dynamic_noise_ref_tokens is None else dynamic_noise_ref_tokens
    noise_base = config.dynamic_noise_base_scale if dynamic_noise_base_scale is None else dynamic_noise_base_scale
    if noise_ref <= 0 or not math.isfinite(noise_base) or noise_base <= 0:
        raise ValueError("Dynamic noise reference tokens and base scale must be positive and finite")
    if dynamic_noise_scale:
        latents *= math.sqrt(4 * h * w / noise_ref) * noise_base
    sigmas = flow_sigmas(num_steps, shift, device)
    predict_x0 = config.predict_x0 if predict_x0 is None else predict_x0
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        predictions = [model.predict_image(ids, latents, sigma.reshape(1) * 1000,
                                          h, w, *layout, drop_dit_text=drop_dit_text)
                       for ids, layout in zip(sequences, layouts)]
        prediction = predictions[0]
        if len(predictions) == 2:
            prediction = predictions[1] + guidance_scale * (prediction - predictions[1])
        latents = flow_step(prediction, latents, sigma, next_sigma, predict_x0)
    if not torch.isfinite(latents).all():
        raise RuntimeError("Non-finite pixels produced during denoising")
    pixels = latents.reshape(2 * h, 2 * w, 3, 16, 16).permute(0, 3, 1, 4, 2)
    pixels = pixels.reshape(height, width, 3).add(1).div(2).clamp(0, 1)
    return Image.fromarray(pixels.mul(255).add(0.5).to(torch.uint8).cpu().numpy())
