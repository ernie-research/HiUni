"""Autoregressive text decoding (understanding and pure text) for the HF Pixel-DiT model.

Mirrors ``recipe/transfusion/eval/pixel_dit_text_decode.py``, i.e. the SFT
``sft_image_und`` / ``sft_text`` layout:

* Qwen3-VL chat template; each ``<|image_pad|>`` becomes ``<time_token>`` plus
  one pad per merged ViT token, and the time token embeds t=0;
* 3D M-RoPE ramps for image pads, linear positions elsewhere;
* causal attention, with bidirectional image blocks when images are present;
* no DeepStack injection, as in training.

There is no KV cache: each new token recomputes the full sequence so the
numerics match the Megatron entrypoint. The ViT runs once per request.
"""

import re

import torch

IMG_PAD_ID, BOI_ID, EOI_ID, TIME_TOKEN_ID, IM_END_ID = 151655, 151652, 151653, 151661, 151645
TEXT_GATE_TOKEN_ID = 151669
MERGE_TOKENS = 4
ASSISTANT_HEADER = "<|im_start|>assistant\n"
THINK_END = "</think>"
# SFT never supervises the opening <think>, so thinking is entered by prefilling it.
THINK_PREFILLS = {"think": "<think>\n", "empty": "<think>\n\n</think>\n\n", "none": ""}
FORCED_THINK_CLOSE = "\n</think>\n\n"


def resolve_route_token(tokenizer, mode=None):
    """None=auto (use <|text_gate|> when the tokenizer has it), True=required, False=off."""
    token = "<|text_gate|>"
    supported = (tokenizer.encode(token, add_special_tokens=False) == [TEXT_GATE_TOKEN_ID]
                 and tokenizer.convert_tokens_to_ids(token) == TEXT_GATE_TOKEN_ID)
    if mode is True and not supported:
        raise ValueError(f"--route-token requires {token} to encode as the single ID {TEXT_GATE_TOKEN_ID}")
    return supported if mode is None else bool(mode)


def render_prompt(tokenizer, prompt, num_images, system="", prefill=""):
    """Return (ids including prefill, number of ids up to the assistant header)."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or "<|vision_start|>" not in template:
        raise ValueError("Tokenizer lacks the Qwen3-VL chat template (chat_template.jinja)")
    content = [{"type": "image"} for _ in range(num_images)] + [{"type": "text", "text": prompt}]
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content if num_images else prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         add_vision_id=False)
    if not text.endswith(ASSISTANT_HEADER):
        raise ValueError("Chat template must end with the assistant header")
    header = len(tokenizer.encode(text, add_special_tokens=False))
    return tokenizer.encode(text + prefill, add_special_tokens=False), header


def expand_image_pads(ids, merged_counts):
    """Replace each template ``<|image_pad|>`` with the time token and merged pads."""
    if sum(token == IMG_PAD_ID for token in ids) != len(merged_counts):
        raise ValueError("Prompt text must not contain image placeholder tokens")
    counts, expanded = iter(merged_counts), []
    for token in ids:
        expanded += [TIME_TOKEN_ID] + [IMG_PAD_ID] * next(counts) if token == IMG_PAD_ID else [token]
    return expanded


def build_positions(ids, merged_grids):
    """Training M-RoPE layout as a [3, L] tensor (one grid per contiguous pad block)."""
    positions = torch.zeros(3, len(ids), dtype=torch.long)
    grids, pos, index = iter(merged_grids), 0, 0
    while index < len(ids):
        if ids[index] != IMG_PAD_ID:
            positions[:, index] = pos
            pos, index = pos + 1, index + 1
            continue
        t, h, w = next(grids)
        count = t * h * w
        if any(token != IMG_PAD_ID for token in ids[index:index + count]):
            raise ValueError("Image pad block does not match its grid")
        grid = torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"))
        positions[:, index:index + count] = pos + grid.reshape(3, -1)
        pos, index = pos + max(t, h, w), index + count
    return positions


def build_ute(ids, bidirectional):
    """Per-column first visible row: causal, image patch columns visible from the first patch row."""
    ids = torch.as_tensor(ids)
    ute = torch.arange(ids.numel())
    if bidirectional:
        starts = ids.eq(BOI_ID).nonzero(as_tuple=True)[0].tolist()
        ends = ids.eq(EOI_ID).nonzero(as_tuple=True)[0].tolist()
        for start, end in zip(starts, ends):
            # BOI at start, time token at start+1, patches in [start+2, end).
            ute[start + 2:end] = start + 2
    return ute


def additive_mask(ute, dtype, device):
    """Dense [1, 1, L, L] SDPA mask for column j visible to row i iff ute[j] <= i."""
    length = ute.numel()
    allowed = ute.to(device)[None, :] <= torch.arange(length, device=device)[:, None]
    mask = torch.zeros(length, length, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


def filter_logits(logits, temperature, top_k, top_p, generator):
    """Greedy for temperature<=0, otherwise top-k / top-p sampling."""
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(-1)
        logits[order[probs.cumsum(-1) - probs > top_p]] = float("-inf")
    return int(torch.multinomial(logits.softmax(-1), 1, generator=generator))


def split_answer(text):
    """Keep only the text after the last ``</think>``; keep the text when it is unclosed."""
    if THINK_END in text:
        text = text.rsplit(THINK_END, 1)[1]
    return re.sub(r"^\s+", "", text).strip()


class TextDecoder:
    """Greedy / sampled decoding over the HF Pixel-DiT backbone."""

    def __init__(self, model, tokenizer, processor=None, route_token=None, min_pixels=0, max_pixels=0):
        """Bind the model, tokenizer and optional image processor."""
        self.model, self.tokenizer, self.processor = model, tokenizer, processor
        self.route_token = resolve_route_token(tokenizer, route_token)
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.device, self.dtype = model.device, model.dtype
        self.stop_ids = {IM_END_ID} | ({int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set())
        self.think_end_ids = tokenizer.encode(THINK_END, add_special_tokens=False)
        self.close_ids = tokenizer.encode(FORCED_THINK_CLOSE, add_special_tokens=False)

    @torch.inference_mode()
    def _vision(self, images):
        """Return merged ViT features and merged [t, h, w] grids for clean input images."""
        kwargs = {key: int(value) for key, value in
                  (("min_pixels", self.min_pixels), ("max_pixels", self.max_pixels)) if value > 0}
        inputs = self.processor(images=images, return_tensors="pt", **kwargs)
        grid = inputs["image_grid_thw"].long()
        pixels = inputs["pixel_values"].to(self.device, self.dtype)
        features = self.model.model.visual(pixels, grid_thw=grid.to(self.device))[0]
        return features, [[t, h // 2, w // 2] for t, h, w in grid.tolist()]

    @torch.inference_mode()
    def prepare(self, prompt, images=(), system="", prefill=""):
        """Return ids, M-RoPE positions, mask columns and the image embeddings."""
        images = list(images or [])
        if images and self.processor is None:
            raise ValueError("An image processor is required for understanding")
        ids, header = render_prompt(self.tokenizer, prompt, len(images), system, prefill)
        if self.route_token:
            ids.insert(header, TEXT_GATE_TOKEN_ID)
        vision, grids = None, []
        if images:
            features, grids = self._vision(images)
            ids = expand_image_pads(ids, [t * h * w for t, h, w in grids])
            # Understanding images are clean: their time tokens embed t=0, as in training.
            times = self.model.time_token(torch.zeros(len(grids), device=self.device))
            vision = (features, times)
        return torch.tensor(ids), build_positions(ids, grids), build_ute(ids, bool(images)), vision

    @torch.inference_mode()
    def next_logits(self, ids, positions, ute, vision):
        """Logits of the token after ``ids``, recomputing the whole sequence."""
        input_ids = ids[None].to(self.device)
        embeddings = self.model.get_input_embeddings()(input_ids)
        if vision is not None:
            features, times = vision
            embeddings = embeddings.clone()
            embeddings[input_ids.eq(IMG_PAD_ID)] = features.to(embeddings.dtype)
            embeddings[input_ids.eq(TIME_TOKEN_ID)] = times.to(embeddings.dtype)
        hidden = self.model.model.language_model(
            inputs_embeds=embeddings, attention_mask=additive_mask(ute, self.dtype, self.device),
            position_ids=positions[:, None].to(self.device), use_cache=False,
        ).last_hidden_state
        return self.model.lm_head(hidden[:, -1])[0]

    def generate(self, prompt, images=(), system="", prefill="", max_new_tokens=1024, temperature=0.0,
                 top_k=0, top_p=1.0, seed=42, max_length=0, max_think_tokens=0):
        """Decode until a stop token or a budget runs out.

        With ``max_think_tokens > 0`` and a prefill that opens ``<think>``, an
        unclosed think block is closed after that many tokens and the answer
        gets the full ``max_new_tokens`` budget.
        """
        ids, positions, ute, vision = self.prepare(prompt, images, system, prefill)
        prompt_tokens = ids.numel()
        generator = torch.Generator(device=self.device).manual_seed(seed) if temperature > 0 else None
        thinking = max_think_tokens > 0 and prefill.rstrip().endswith("<think>")
        new_tokens, answer_tokens, forced_close, finish = [], 0, False, "length"

        def append(token):
            """Extend the sequence by one causal text token."""
            nonlocal ids, positions, ute
            positions = torch.cat([positions, positions[:, -1:] + 1], dim=1)
            ute = torch.cat([ute, ute.new_tensor([ids.numel()])])
            ids = torch.cat([ids, ids.new_tensor([token])])

        while answer_tokens < max_new_tokens:
            if max_length and ids.numel() >= max_length:
                break
            if thinking and len(new_tokens) >= max_think_tokens:
                for token in self.close_ids:
                    append(token)
                new_tokens += self.close_ids
                thinking, forced_close = False, True
                continue
            token = filter_logits(self.next_logits(ids, positions, ute, vision), temperature, top_k, top_p, generator)
            if token in self.stop_ids:
                finish = "stop"
                break
            new_tokens.append(token)
            append(token)
            if thinking and new_tokens[-len(self.think_end_ids):] == self.think_end_ids:
                thinking = False
            elif not thinking:
                answer_tokens += 1
        raw = prefill + self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return {"raw": raw, "answer": split_answer(raw), "prompt_tokens": prompt_tokens,
                "new_tokens": len(new_tokens), "finish_reason": finish, "forced_think_close": forced_close}


# ---------------------------------------------------------------------------
# Command-line driver shared by understand.py and chat.py.
# ---------------------------------------------------------------------------

def add_decode_args(parser, with_images):
    """Register the arguments shared by the understanding and pure-text CLIs."""
    import argparse

    parser.add_argument("--model", required=True, help="Exported HF Pixel-DiT directory")
    parser.add_argument("--tokenizer", help="Optional tokenizer override")
    parser.add_argument("--prompt", default="", help="Single prompt; otherwise use --data")
    parser.add_argument("--data", default="", help="jsonl: {id, prompt|question, system?"
                        + (", image|images?}" if with_images else "}"))
    parser.add_argument("--output", default="", help="Output jsonl; print to stdout when empty")
    parser.add_argument("--resume", action="store_true", help="Skip ids already in --output")
    parser.add_argument("--system", default="")
    parser.add_argument("--think-mode", choices=sorted(THINK_PREFILLS), default="think",
                        help="think: prefill '<think>\\n'; empty: an empty think block; none: no prefill")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Answer budget; also counts thinking unless --max-think-tokens is set")
    parser.add_argument("--max-think-tokens", type=int, default=0,
                        help="Close an unclosed think block after this many tokens (0 = off)")
    parser.add_argument("--max-length", type=int, default=9216, help="Cap on prompt + generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--route-token", action=argparse.BooleanOptionalAction, default=None,
                        help="<|text_gate|> after the assistant header: auto (default), force or disable")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    if with_images:
        parser.add_argument("--image", action="append", default=[],
                            help="Image path for --prompt; repeat for several images")
        parser.add_argument("--image-root", default="", help="Prefix for relative jsonl image paths")
        parser.add_argument("--min-pixels", type=int, default=262144)
        parser.add_argument("--max-pixels", type=int, default=4194304)


def load_items(args, with_images, read_jsonl, resolve_image_paths):
    """Return [{id, prompt, images, system}] from --prompt or --data."""
    if args.prompt:
        return [{"id": "0", "prompt": args.prompt, "images": list(args.image) if with_images else [],
                 "system": args.system}]
    if not args.data:
        raise SystemExit("Pass --prompt or --data")
    items = []
    for index, record in enumerate(read_jsonl(args.data)):
        prompt = record.get("prompt", record.get("question"))
        if prompt is None:
            raise ValueError(f"{args.data} record {index} has no prompt/question")
        images = record.get("images") or ([record["image"]] if record.get("image") else [])
        if images and not with_images:
            raise ValueError(f"{args.data} record {index} has images; use understand.py")
        items.append({"id": str(record.get("id", index)), "prompt": prompt,
                      "images": resolve_image_paths(images, getattr(args, "image_root", "")),
                      "system": record.get("system", args.system)})
    return items


def run(args, with_images):
    """Load the model, shard items over torchrun workers and write jsonl."""
    import json
    import os

    from PIL import Image

    if __package__:
        from . import runtime
    else:
        import runtime

    rank, world, local_rank = runtime.dist_info()
    items = load_items(args, with_images, runtime.read_jsonl, runtime.resolve_image_paths)
    runtime.init_dist(world)
    try:
        device = runtime.resolve_device(args.device, local_rank)
        model = runtime.load_model(args.model, args.dtype, device)
        tokenizer = runtime.load_tokenizer(args.model, args.tokenizer)
        processor = runtime.load_processor(args.model) if with_images else None
        decoder = TextDecoder(model, tokenizer, processor, args.route_token,
                              getattr(args, "min_pixels", 0), getattr(args, "max_pixels", 0))
        prefill = THINK_PREFILLS[args.think_mode]
        if rank == 0:
            print(f"[decode] model={args.model} items={len(items)} think_mode={args.think_mode} "
                  f"route_token={decoder.route_token}", flush=True)
        shard = runtime.shard_path(args.output, rank, world) if args.output else ""
        done = runtime.done_ids(args.output) | runtime.done_ids(shard) if args.output and args.resume else set()
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        handle = open(shard, "a" if args.resume else "w", encoding="utf-8") if args.output else None
        try:
            for item in items[rank::world]:
                if item["id"] in done:
                    continue
                images = [Image.open(path).convert("RGB") for path in item["images"]]
                result = decoder.generate(
                    item["prompt"], images, item["system"], prefill, args.max_new_tokens, args.temperature,
                    args.top_k, args.top_p, args.seed, args.max_length, args.max_think_tokens,
                )
                record = {"id": item["id"], "prompt": item["prompt"], "images": item["images"], **result}
                if handle is None:
                    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"rank={rank} id={item['id']} new_tokens={result['new_tokens']} "
                      f"finish={result['finish_reason']}", flush=True)
        finally:
            if handle is not None:
                handle.close()
        runtime.barrier()
        if args.output and world > 1 and rank == 0:
            runtime.merge_shards(args.output, world, [item["id"] for item in items])
    finally:
        runtime.close_dist()
