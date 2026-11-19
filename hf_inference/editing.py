"""Instruction-based image editing (I2I) for the HF Pixel-DiT model.

Mirrors ``recipe/transfusion/eval/inference_transfusion_qwen3vl_pixel_dit_i2i.py``:

* user turn ``[BOI, ref pads, EOI] + instruction``, assistant turn
  ``[BOI, TIME, gen pads, EOI, EOS]``;
* image blocks are bidirectional; the noisy generated block is self-closed so
  the trailing EOS cannot see it (FlashMask ``startend`` rows, dense here);
* the clean reference goes through the ViT once, the noisy latents once per
  step, and both feed the HiGate at ``pixel_dit_gate_inject_layer``
  (``pixel_dit_ref_gate_mode`` selects shared / separate / no reference gate);
* CFG branches ``none`` (image + text), ``text`` (image only) and ``full``
  (unconditional), combined as ``text`` CFG or the three-way ``edit3`` CFG.
"""

import math

import torch
import torch.nn.functional as F

if __package__:
    from .sampling import flow_sigmas, flow_step, merge_patches, unmerge_patches, vision_patches
    from .text_decode import build_positions
else:
    from sampling import flow_sigmas, flow_step, merge_patches, unmerge_patches, vision_patches
    from text_decode import build_positions

IMG_PAD_ID, BOI_ID, EOI_ID, TIME_TOKEN_ID = 151655, 151652, 151653, 151661
IM_START_ID, IM_END_ID, NL_ID, USER_ID, ASSISTANT_ID = 151644, 151645, 198, 872, 77091
EOS_ID = IM_END_ID
IMAGE_GATE_TOKEN_ID = 151670
CFG_MODES = ("none", "text", "edit3")
SCHEDULERS = ("euler", "dpm2", "dpm3")


def resolve_image_route(tokenizer, mode=None):
    """None=auto (use <|image_gate|> when the tokenizer has it), True=required, False=off."""
    token = "<|image_gate|>"
    supported = (tokenizer.encode(token, add_special_tokens=False) == [IMAGE_GATE_TOKEN_ID]
                 and tokenizer.convert_tokens_to_ids(token) == IMAGE_GATE_TOKEN_ID)
    if mode is True and not supported:
        raise ValueError(f"route token requires {token} to encode as the single ID {IMAGE_GATE_TOKEN_ID}")
    return supported if mode is None else bool(mode)


def build_edit_sequence(tokenizer, prompt, ref_tokens, gen_tokens, branch, keep_chat_template=False, route=False):
    """Token ids of one CFG branch: ``none`` (ref + text), ``text`` (ref only) or ``full`` (neither)."""
    if branch not in ("none", "text", "full"):
        raise ValueError(f"invalid CFG branch: {branch}")
    user = [BOI_ID] + [IMG_PAD_ID] * ref_tokens + [EOI_ID] if branch != "full" and ref_tokens else []
    if branch == "none" and prompt:
        user += tokenizer.encode(prompt, add_special_tokens=False)
    image = ([IMAGE_GATE_TOKEN_ID] if branch == "none" and route else [])
    image += [BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * gen_tokens + [EOI_ID, EOS_ID]
    if branch == "full" and not keep_chat_template:
        return image
    return [IM_START_ID, USER_ID, NL_ID] + user + [IM_END_ID, NL_ID, IM_START_ID, ASSISTANT_ID, NL_ID] + image


def edit_startend(ids):
    """Per-column (first visible row, first hidden row) of the training FlashMask.

    Image patches are visible from their first patch row; a block containing the
    time token is hidden from every row after its EOI. Megatron hides all
    columns from the final EOS row (LTS=L-1); that row is unused, so it stays
    valid here to avoid an all-masked SDPA row.
    """
    ids = torch.as_tensor(ids)
    length = ids.numel()
    ute, lts = torch.arange(length), torch.full((length,), length)
    starts = ids.eq(BOI_ID).nonzero(as_tuple=True)[0].tolist()
    ends = ids.eq(EOI_ID).nonzero(as_tuple=True)[0].tolist()
    for start, end in zip(starts, ends):
        if start >= end:
            continue
        times = ids[start + 1:end].eq(TIME_TOKEN_ID).nonzero(as_tuple=True)[0]
        patch_start = start + 2 + int(times[0]) if times.numel() else start + 1
        ute[patch_start:end] = patch_start
        if times.numel():
            lts[start:end + 1] = lts[start:end + 1].clamp(max=end + 1)
    return ute, lts


def edit_mask(ute, lts, dtype, device):
    """Dense [1, 1, L, L] SDPA mask: row i sees column j iff ute[j] <= i < lts[j]."""
    rows = torch.arange(ute.numel(), device=device)[:, None]
    allowed = (ute.to(device)[None, :] <= rows) & (rows < lts.to(device)[None, :])
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


def select_branches(cfg_mode, cfg_scale, w_img):
    """Branches to run and whether edit3 reduces to two branches (w_img == 1 cancels ``full``)."""
    if cfg_mode == "none" or (cfg_mode == "text" and cfg_scale <= 1.0):
        return ["none"], False
    if cfg_mode == "text":
        return ["none", "text"], False
    if cfg_mode == "edit3":
        two = abs(float(w_img) - 1.0) < 1e-6
        return (["none", "text"] if two else ["none", "text", "full"]), two
    raise ValueError(f"unsupported cfg_mode: {cfg_mode}")


def combine_branches(preds, cfg_mode, cfg_scale, w_text, w_img, two_branch):
    """``text``: text + s(none - text); ``edit3``: full + w_text(none - text) + w_img(text - full)."""
    if cfg_mode == "none" or (cfg_mode == "text" and cfg_scale <= 1.0):
        return preds["none"]
    if cfg_mode == "text":
        return preds["text"] + cfg_scale * (preds["none"] - preds["text"])
    if two_branch:
        return preds["text"] + w_text * (preds["none"] - preds["text"])
    return preds["full"] + w_text * (preds["none"] - preds["text"]) + w_img * (preds["text"] - preds["full"])


def reference_raster(pixel_values, grid):
    """Recover raster RGB patches [H*W, 768] from Qwen processor pixel values (first frame)."""
    t, h, w = (int(value) for value in grid)
    if t != 1:
        raise ValueError("Reference images must have a single temporal patch")
    patches = pixel_values.reshape(h // 2, w // 2, 2, 2, 3, 2, 16, 16)[:, :, :, :, :, 0]
    return patches.permute(0, 2, 1, 3, 4, 5, 6).reshape(h * w, 768)


def dpm_step(order, prediction, latents, sigma, next_sigma, history, predict_x0=True):
    """Multistep DPM-Solver(2/3) in x0 space; ``history`` keeps (lambda, x0) across steps."""
    x0 = prediction if predict_x0 else latents - sigma * prediction
    eps = 1e-8
    lam = torch.log((1 - sigma).clamp(min=eps) / sigma.clamp(min=eps))
    lam_next = torch.log((1 - next_sigma).clamp(min=eps) / next_sigma.clamp(min=eps))
    h = lam_next - lam
    history.append((lam.item(), x0))
    effective = min(order, len(history))
    if effective == 2:
        lam_prev, x0_prev = history[-2]
        h_prev = lam.item() - lam_prev
        r = h.item() / h_prev if h_prev != 0 else 0.0
        corrected = x0 + (1.0 / (2.0 * r)) * (x0 - x0_prev)
    elif effective == 3:
        (lam_2, x0_2), (lam_3, x0_3) = history[-2], history[-3]
        h_1, h_2 = lam.item() - lam_2, lam_2 - lam_3
        r1 = h.item() / h_1 if h_1 != 0 else 0.0
        r2 = h_1 / h_2 if h_2 != 0 else 0.0
        d1_0, d1_1 = (1.0 / r1) * (x0 - x0_2), (1.0 / r2) * (x0_2 - x0_3)
        corrected = x0 + d1_0 + (r1 / (r1 + r2)) * (d1_0 - d1_1) + (1.0 / (r1 + r2)) * (d1_0 - d1_1)
    else:
        corrected = x0
    latents = (1 - next_sigma) * corrected + next_sigma * (latents - (1 - sigma) * x0) / sigma.clamp(min=eps)
    if len(history) > 3:
        history.pop(0)
    return latents


class EditReference:
    """Clean reference image prepared once per request: ViT features, raster patches and grids."""

    def __init__(self, model, processor, image, min_pixels=None, max_pixels=None):
        """Run the Qwen processor and the ViT on the reference image."""
        kwargs = {key: int(value) for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels))
                  if value is not None and value > 0}
        inputs = processor(images=[image], return_tensors="pt", **kwargs)
        self.grid = inputs["image_grid_thw"].long()
        _, h, w = (int(value) for value in self.grid[0])
        self.raw_h, self.raw_w = h, w
        self.merged_h, self.merged_w = h // 2, w // 2
        self.tokens = self.merged_h * self.merged_w
        pixels = inputs["pixel_values"].to(model.device)
        with torch.inference_mode():
            self.features = model.model.visual(pixels.to(model.dtype), grid_thw=self.grid.to(model.device))[0]
        self.raster = reference_raster(pixels.float(), self.grid[0])
        self.size = image.size


def _gate_high(model, raster, features, height, width):
    """HiGate high-information input: merged clean/noisy pixels or projected ViT features."""
    if model.config.pixel_dit_gate_input == "pixel":
        return model.pixel_proj(merge_patches(raster.to(model.dtype), height, width))
    return model.pixel_dit_input_projection(features)


@torch.inference_mode()
def predict_edit(model, ids, ref, latents, timesteps, gen_h, gen_w):
    """Predict raw generated patches [4*gen_h*gen_w, 768] for one CFG branch sequence.

    ``ids`` is a 1D id list from :func:`build_edit_sequence`; ``ref`` is an
    :class:`EditReference` (its block is present unless this is the ``full`` branch).
    """
    config, device, dtype = model.config, model.device, model.dtype
    ids_t = torch.as_tensor(ids, device=device)
    pad = ids_t.eq(IMG_PAD_ID)
    gen_tokens = gen_h * gen_w
    pad_index = pad.nonzero(as_tuple=True)[0]
    has_ref = pad_index.numel() == gen_tokens + ref.tokens
    if not has_ref and pad_index.numel() != gen_tokens:
        raise ValueError("Edit sequence image pads do not match the reference/generation grids")
    ref_mask = torch.zeros_like(pad)
    if has_ref:
        ref_mask[pad_index[:ref.tokens]] = True
    gen_mask = pad & ~ref_mask
    grids = ([[1, ref.merged_h, ref.merged_w]] if has_ref else []) + [[1, gen_h, gen_w]]
    positions = build_positions(ids_t.tolist(), grids)[:, None].to(device)
    ute, lts = edit_startend(ids_t.cpu())
    mask = edit_mask(ute, lts, dtype, device)

    gen_grid = torch.tensor([[1, gen_h * 2, gen_w * 2]], device=device)
    gen_features = model.model.visual(vision_patches(latents.to(dtype), gen_h, gen_w), grid_thw=gen_grid)[0]
    embeddings = model.get_input_embeddings()(ids_t[None]).clone()
    if has_ref:
        embeddings[0, ref_mask] = ref.features.to(embeddings.dtype)
    embeddings[0, gen_mask] = gen_features.to(embeddings.dtype)
    embeddings[0, ids_t.eq(TIME_TOKEN_ID)] = model.time_token(timesteps).to(embeddings.dtype)
    # As in training, generation runs without DeepStack injection.
    hidden = model.model.language_model(inputs_embeds=embeddings, attention_mask=mask,
                                        position_ids=positions, use_cache=False).last_hidden_state
    hidden = model.pixel_dit_input_projection(hidden)
    embedder = model.pixel_dit_time_embeddings or model.time_embeddings
    time = embedder(timesteps).expand(gen_tokens, -1)

    gen_high = ref_high = None
    if config.use_pixel_dit_pixel_residual:
        gen_high = _gate_high(model, latents, gen_features, gen_h, gen_w)
        if has_ref and config.pixel_dit_ref_gate_mode != "none":
            ref_high = _gate_high(model, ref.raster, ref.features, ref.merged_h, ref.merged_w)
    head_mask, gen_flag, ref_flag = mask, gen_mask[None], ref_mask[None]
    if not config.pixel_dit_use_text_tokens:
        keep = pad | ids_t.eq(TIME_TOKEN_ID)
        hidden, positions = hidden[:, keep], positions[..., keep]
        gen_flag, ref_flag = gen_flag[:, keep], ref_flag[:, keep]
        head_mask = mask[:, :, keep][:, :, :, keep]
    for index, layer in enumerate(model.diffusion_head_a):
        if index == config.pixel_dit_gate_inject_layer and gen_high is not None:
            if ref_high is not None:
                ref_gate = model.pixel_dit_gate if config.pixel_dit_ref_gate_mode == "shared" else model.pixel_dit_ref_gate
                hidden[ref_flag] = ref_gate(hidden[ref_flag], ref_high).to(hidden.dtype)
            hidden[gen_flag] = model.pixel_dit_gate(hidden[gen_flag], gen_high).to(hidden.dtype)
        hidden = layer(hidden, time, gen_flag, attention_mask=head_mask, position_ids=positions)[0]
    output = model.diffusion_head_b(hidden, time, gen_flag)[gen_flag]
    return unmerge_patches(output, gen_h, gen_w).float()


def resolve_gen_grid(ref, height=None, width=None, match_ref_resolution=True, resolution=512):
    """Merged generation grid (h, w); ``height``/``width``/``resolution`` are pixels (multiples of 32)."""
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("height and width must be set together")
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("height and width must be positive multiples of 32 pixels")
        return height // 32, width // 32
    if match_ref_resolution:
        return ref.merged_h, ref.merged_w
    if resolution <= 0 or resolution % 32:
        raise ValueError("resolution must be a positive multiple of 32")
    return resolution // 32, resolution // 32


def decode_latents(latents, gen_h, gen_w):
    """Raster patches [4*h*w, 768] in [-1, 1] -> image tensor [3, 32h, 32w] in [0, 1]."""
    z = latents.float().reshape(2 * gen_h, 2 * gen_w, 3, 16, 16).permute(2, 0, 3, 1, 4)
    return z.reshape(3, 32 * gen_h, 32 * gen_w).mul(0.5).add(0.5).clamp(0, 1)


def to_pil(image, size=None):
    """[3, H, W] in [0, 1] -> PIL, optionally resized (bicubic + antialias) to ``size`` = (W, H)."""
    from PIL import Image

    if size is not None and tuple(image.shape[-2:]) != (size[1], size[0]):
        image = F.interpolate(image[None].float(), size=(size[1], size[0]), mode="bicubic",
                              align_corners=False, antialias=True)[0]
    array = image.clamp(0, 1).mul(255).add(0.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


@torch.inference_mode()
def edit_image(model, tokenizer, processor, prompt, ref_image, steps=25, shift=4.0, scheduler="euler",
               cfg_mode="edit3", cfg_scale=2.0, w_text=4.0, w_img=1.0, seed=42, match_ref_resolution=True,
               resolution=512, height=None, width=None, min_pixels=None, max_pixels=None,
               keep_chat_template=False, route=None, dynamic_noise_scale=None, dynamic_noise_ref_tokens=None,
               dynamic_noise_base_scale=None, resize_to_ref=True, latents=None, return_tensor=False):
    """Edit ``ref_image`` (PIL) following ``prompt``; returns a PIL image.

    ``latents`` optionally supplies the initial raw noise [4*h*w, 768] before
    dynamic scaling, e.g. to reproduce a Megatron run bit-for-bit in noise.
    """
    if cfg_mode not in CFG_MODES:
        raise ValueError(f"cfg_mode must be one of {CFG_MODES}")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}")
    model.eval()
    config, device = model.config, model.device
    ref = EditReference(model, processor, ref_image.convert("RGB"), min_pixels, max_pixels)
    gen_h, gen_w = resolve_gen_grid(ref, height, width, match_ref_resolution, resolution)
    route = resolve_image_route(tokenizer, route)
    branches, two_branch = select_branches(cfg_mode, cfg_scale, w_img)
    sequences = {branch: build_edit_sequence(tokenizer, prompt, ref.tokens, gen_h * gen_w, branch,
                                             keep_chat_template, route) for branch in branches}
    max_length = config.text_config.max_position_embeddings
    if max(len(ids) for ids in sequences.values()) > max_length:
        raise ValueError(f"Reference, prompt and output exceed max_position_embeddings={max_length}")

    raw_tokens = 4 * gen_h * gen_w
    if latents is None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn(raw_tokens, 768, generator=generator, device=device, dtype=torch.float32)
    else:
        latents = latents.to(device=device, dtype=torch.float32).clone()
        if latents.shape != (raw_tokens, 768):
            raise ValueError(f"latents must have shape ({raw_tokens}, 768)")
    use_dynamic = config.use_dynamic_noise_scale if dynamic_noise_scale is None else dynamic_noise_scale
    noise_ref = config.dynamic_noise_ref_tokens if dynamic_noise_ref_tokens is None else dynamic_noise_ref_tokens
    noise_base = config.dynamic_noise_base_scale if dynamic_noise_base_scale is None else dynamic_noise_base_scale
    if use_dynamic:
        if noise_ref <= 0 or not math.isfinite(noise_base) or noise_base <= 0:
            raise ValueError("Dynamic noise reference tokens and base scale must be positive and finite")
        latents *= math.sqrt(raw_tokens / noise_ref) * noise_base

    sigmas = flow_sigmas(steps, shift, device)
    history = []
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        timesteps = sigma.reshape(1) * 1000
        preds = {branch: predict_edit(model, ids, ref, latents, timesteps, gen_h, gen_w)
                 for branch, ids in sequences.items()}
        prediction = combine_branches(preds, cfg_mode, cfg_scale, w_text, w_img, two_branch)
        # The Megatron edit sampler always treats the head output as clean x0.
        if scheduler == "euler":
            latents = flow_step(prediction, latents, sigma, next_sigma, True)
        else:
            latents = dpm_step(2 if scheduler == "dpm2" else 3, prediction, latents, sigma, next_sigma, history)
    if not torch.isfinite(latents).all():
        raise RuntimeError("Non-finite pixels produced during denoising")
    image = decode_latents(latents, gen_h, gen_w)
    if return_tensor:
        return image
    return to_pil(image, ref.size if resize_to_ref else None)
