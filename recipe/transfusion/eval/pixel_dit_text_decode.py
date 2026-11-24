"""Autoregressive text decoding for the dense Pixel-DiT Megatron model.

Shared by the understanding (images + text -> text) and pure-text
entrypoints. Inputs are laid out exactly as the SFT task encoder lays out
``sft_image_und`` / ``sft_text`` samples:

* Qwen3-VL chat template, each ``<|image_pad|>`` expanded to
  ``<time_token>`` + one pad per merged ViT token;
* 3D M-RoPE positions for image blocks, linear positions elsewhere;
* FlashMask ``startend_row_indices`` with bidirectional image blocks
  (understanding) or a plain causal mask (text).

The Megatron forward has no KV cache, so every generated token recomputes the
whole sequence, ViT included. This reproduces training numerics and is meant
for validation and small evaluations, not serving.
"""

import argparse
import json
import os
import re
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

from data.transfusion.transfusion_task_encoder import (
    V5_TEXT_GATE_TOKEN_ID,
    TaskType,
    TransfusionTaskEncoder,
    create_sample_startend_row_indices,
)
from model.transfusion.transfusion_sampler import (
    BOI_ID,
    EOI_ID,
    IM_END_ID,
    IMG_PAD_ID,
    TIME_TOKEN_ID,
    _assert_flashmask_runtime,
    _supports_v5_route_token,
)

MERGE_TOKENS = 4  # 2x2 spatial merge
ASSISTANT_HEADER = "<|im_start|>assistant\n"
THINK_END = "</think>"
# SFT never supervises the opening <think> marker, so thinking is entered by
# prefilling it, as the Qwen3 thinking template does.
THINK_PREFILLS = {
    "think": "<think>\n",
    "empty": "<think>\n\n</think>\n\n",
    "none": "",
}
FORCED_THINK_CLOSE = "\n</think>\n\n"


def load_vl_chat_template(tokenizer, hf_path: str) -> str:
    """Resolve the Qwen3-VL template the same way the SFT task encoder does."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and "<|vision_start|>" in template:
        return template
    path = os.path.join(hf_path, "chat_template.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            template = json.load(handle).get("chat_template")
        if template:
            return template
    raise FileNotFoundError(
        f"No Qwen3-VL chat template in the tokenizer or {path}"
    )


def resolve_route_token(tokenizer, mode: Optional[bool]) -> bool:
    """None=auto (use it when the tokenizer has it), True=required, False=off."""
    supported = _supports_v5_route_token(
        tokenizer, "<|text_gate|>", V5_TEXT_GATE_TOKEN_ID
    )
    if mode is True and not supported:
        raise ValueError(
            "--route_token requires <|text_gate|> to encode as the single ID "
            f"{V5_TEXT_GATE_TOKEN_ID}; the base Qwen3-VL tokenizer lacks it"
        )
    return supported if mode is None else bool(mode)


def render_prompt(tokenizer, template: str, prompt: str, num_images: int,
                  system: str = "", prefill: str = "") -> Tuple[List[int], int]:
    """Return (ids including prefill, length of the ids up to the assistant header)."""
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content if num_images else prompt})
    text = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False,
        add_generation_prompt=True, add_vision_id=False,
    )
    if not text.endswith(ASSISTANT_HEADER):
        raise ValueError("Chat template must end with the assistant header")
    return tokenizer.encode(text + prefill, add_special_tokens=False), len(
        tokenizer.encode(text, add_special_tokens=False)
    )


def expand_image_pads(ids: List[int], merged_counts: List[int]) -> List[int]:
    """Replace each template ``<|image_pad|>`` with time token + merged pads."""
    if sum(1 for tok in ids if tok == IMG_PAD_ID) != len(merged_counts):
        raise ValueError("Prompt text must not contain image placeholder tokens")
    counts = iter(merged_counts)
    expanded = []
    for tok in ids:
        if tok == IMG_PAD_ID:
            expanded.append(TIME_TOKEN_ID)
            expanded.extend([IMG_PAD_ID] * next(counts))
        else:
            expanded.append(tok)
    return expanded


def build_position_ids(ids: List[int], merged_grids: List[List[int]]) -> torch.Tensor:
    """Training M-RoPE layout (3, L); the encoder method only reads image_token_id."""
    owner = SimpleNamespace(image_token_id=IMG_PAD_ID)
    return TransfusionTaskEncoder._build_position_ids(owner, ids, merged_grids or None)


def build_startend(ids: torch.Tensor, bidirectional: bool) -> torch.Tensor:
    """Return the per-column upper-triangle ends (UTE) of the training mask.

    Training uses LTS = L - 1, which hides the last row because its target is
    padding. Decoding needs that row, so the caller sets LTS = L, i.e. the
    exact causal mask the training rows saw.
    """
    cond, _ = create_sample_startend_row_indices(
        input_ids=ids, img_start_id=BOI_ID, img_end_id=EOI_ID,
        bidirectional=bidirectional,
    )
    return cond[0, :, 1].clone()


def _filter_logits(logits: torch.Tensor, temperature: float, top_k: int,
                   top_p: float, generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, order = torch.sort(logits, descending=True)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        drop = cumulative - sorted_logits.softmax(-1) > top_p
        logits[order[drop]] = float("-inf")
    probs = logits.softmax(-1)
    return int(torch.multinomial(probs, 1, generator=generator))


class PixelDiTTextDecoder:
    """Greedy / sampled decoding over the Megatron Pixel-DiT backbone."""

    def __init__(self, model, tokenizer, hf_path: str, processor=None,
                 route_token: Optional[bool] = None, min_pixels: int = 0,
                 max_pixels: int = 0):
        _assert_flashmask_runtime(model, "Pixel-DiT text decoding")
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.template = load_vl_chat_template(tokenizer, hf_path)
        self.route_token = resolve_route_token(tokenizer, route_token)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.stop_ids = {IM_END_ID}
        if tokenizer.eos_token_id is not None:
            self.stop_ids.add(int(tokenizer.eos_token_id))
        self.think_end_ids = tokenizer.encode(THINK_END, add_special_tokens=False)
        self.close_ids = tokenizer.encode(FORCED_THINK_CLOSE, add_special_tokens=False)

    def _encode_images(self, images):
        kwargs = {}
        if self.min_pixels > 0:
            kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels > 0:
            kwargs["max_pixels"] = int(self.max_pixels)
        inputs = self.processor(images=images, videos=None, return_tensors="pt", **kwargs)
        grid = inputs["image_grid_thw"].long()
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        return pixel_values, grid

    def _prepare(self, prompt: str, images, system: str, prefill: str):
        images = list(images or [])
        if images and self.processor is None:
            raise ValueError("An image processor is required for understanding")
        ids, header_len = render_prompt(
            self.tokenizer, self.template, prompt, len(images), system, prefill
        )
        if self.route_token:
            ids.insert(header_len, V5_TEXT_GATE_TOKEN_ID)
        vision = None
        merged_grids = []
        if images:
            pixel_values, grid = self._encode_images(images)
            merged_grids = [[int(t), int(h) // 2, int(w) // 2] for t, h, w in grid.tolist()]
            ids = expand_image_pads(ids, [t * h * w for t, h, w in merged_grids])
            vision = (pixel_values, grid.to(self.device))
        prompt_ids = torch.tensor(ids, dtype=torch.long)
        positions = build_position_ids(ids, merged_grids)
        ute = build_startend(prompt_ids, bidirectional=bool(images))
        return prompt_ids, positions, ute, vision

    @torch.no_grad()
    def _next_logits(self, ids, positions, ute, vision) -> torch.Tensor:
        length = ids.numel()
        lts = torch.full((length,), length, dtype=ute.dtype)
        startend = torch.stack([lts, ute], dim=-1)[None, None].to(self.device)
        input_ids = ids[None].to(self.device)
        kwargs = {"task_type": torch.tensor([int(TaskType.SFT_TEXT)], device=self.device)}
        if vision is not None:
            pixel_values, grid = vision
            num_tokens = int((grid.prod(-1) // MERGE_TOKENS).sum())
            kwargs = {
                "pixel_values": pixel_values,
                "image_grid_thw": grid,
                "image_input_mask": input_ids.eq(IMG_PAD_ID),
                "image_generate_mask": torch.zeros(
                    num_tokens * MERGE_TOKENS, dtype=torch.bool, device=self.device
                ),
                "image_understand_mask": torch.ones(num_tokens, dtype=torch.bool, device=self.device),
                "total_img_und_mask": torch.ones(num_tokens, dtype=torch.bool, device=self.device),
                # Understanding images are clean: time token embeds t=0, as in training.
                "timesteps_samples": torch.zeros(grid.shape[0], dtype=torch.long, device=self.device),
                "timesteps": torch.zeros(num_tokens, device=self.device),
                "task_type": torch.tensor([int(TaskType.SFT_IMAGE_UND)], device=self.device),
            }
        output = self.model(
            input_ids=input_ids,
            position_ids=positions[:, None, :].to(self.device),
            attention_mask=None,
            labels=None,
            extra_block_kwargs={"startend_row_indices": startend, "causal": False},
            **kwargs,
        )
        logits = output["text_loss"]
        if logits.shape[0] != 1:  # Megatron may return [S, B, V]
            logits = logits.transpose(0, 1)
        return logits[0, -1]

    def generate(self, prompt: str, images=None, system: str = "", prefill: str = "",
                 max_new_tokens: int = 1024, temperature: float = 0.0, top_k: int = 0,
                 top_p: float = 1.0, seed: int = 42, max_length: int = 0,
                 max_think_tokens: int = 0) -> Dict:
        """Decode until a stop token or a budget runs out.

        With ``max_think_tokens > 0`` and a prefill that opens ``<think>``, an
        unclosed think block is closed after that many tokens and the answer
        gets the remaining ``max_new_tokens`` budget.
        """
        state = list(self._prepare(prompt, images, system, prefill))
        vision = state.pop()
        prompt_len = state[0].numel()
        generator = None
        if temperature > 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        thinking = max_think_tokens > 0 and prefill.rstrip().endswith("<think>")
        new_tokens: List[int] = []
        answer_tokens = 0
        forced_close = False
        finish = "length"
        while answer_tokens < max_new_tokens:
            if max_length and state[0].numel() >= max_length:
                break
            if thinking and len(new_tokens) >= max_think_tokens:
                for tok in self.close_ids:
                    self._append(state, tok)
                new_tokens.extend(self.close_ids)
                thinking, forced_close = False, True
                continue
            logits = self._next_logits(*state, vision)
            token = _filter_logits(logits, temperature, top_k, top_p, generator)
            if token in self.stop_ids:
                finish = "stop"
                break
            new_tokens.append(token)
            self._append(state, token)
            if thinking and new_tokens[-len(self.think_end_ids):] == self.think_end_ids:
                thinking = False
            elif not thinking:
                answer_tokens += 1
        raw = prefill + self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return {
            "raw": raw,
            "answer": split_answer(raw),
            "prompt_tokens": prompt_len,
            "new_tokens": len(new_tokens),
            "finish_reason": finish,
            "forced_think_close": forced_close,
        }

    @staticmethod
    def _append(state, token: int) -> None:
        """Extend ids / positions / UTE by one causal text token."""
        ids, positions, ute = state
        # Generated text continues linearly after the last position.
        state[1] = torch.cat([positions, positions[:, -1:] + 1], dim=1)
        state[2] = torch.cat([ute, ute.new_tensor([ids.numel()])])
        state[0] = torch.cat([ids, ids.new_tensor([token])])


def split_answer(text: str) -> str:
    """Keep only the text after the last ``</think>``; keep text when it is unclosed."""
    if THINK_END in text:
        text = text.rsplit(THINK_END, 1)[1]
    return re.sub(r"^\s+", "", text).strip()


# ---------------------------------------------------------------------------
# Command-line driver shared by the understanding and text entrypoints.
# ---------------------------------------------------------------------------

def add_decode_args(parser, with_images: bool) -> None:
    parser.add_argument("--ckpt", required=True, help="Megatron torch_dist checkpoint root or iter_* dir")
    parser.add_argument("--hf_path", required=True, help="Qwen3-VL HF dir (tokenizer, processor, config)")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--prompt", default="", help="single prompt; otherwise use --data_path")
    parser.add_argument(
        "--data_path", default="",
        help="jsonl, one record per line: {id, prompt|question, system?"
             + (", image|images?}" if with_images else "}"),
    )
    parser.add_argument("--output", default="", help="output jsonl; print to stdout when empty")
    parser.add_argument("--resume", action="store_true", help="skip ids already in --output")
    parser.add_argument("--system", default="")
    parser.add_argument(
        "--think_mode", choices=sorted(THINK_PREFILLS), default="think",
        help="think: prefill '<think>\\n'; empty: prefill an empty think block; none: no prefill",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="answer budget; also counts thinking unless --max_think_tokens is set")
    parser.add_argument("--max_think_tokens", type=int, default=0,
                        help="close an unclosed think block after this many tokens (0 = off)")
    parser.add_argument("--max_length", type=int, default=9216,
                        help="hard cap on prompt + generated tokens (training seq_length)")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--route_token", action=argparse.BooleanOptionalAction, default=None,
        help="<|text_gate|> after the assistant header: auto (default), force, or disable",
    )
    if with_images:
        parser.add_argument("--image", action="append", default=[],
                            help="image path for --prompt; repeat for multiple images")
        parser.add_argument("--image_root", default="", help="prefix for relative jsonl image paths")
        parser.add_argument("--min_pixels", type=int, default=262144)
        parser.add_argument("--max_pixels", type=int, default=4194304)


def load_items(args, with_images: bool) -> List[Dict]:
    if args.prompt:
        images = list(args.image) if with_images else []
        return [{"id": "0", "prompt": args.prompt, "images": images, "system": args.system}]
    if not args.data_path:
        raise SystemExit("Pass --prompt or --data_path")
    items = []
    with open(args.data_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt", record.get("question"))
            if prompt is None:
                raise ValueError(f"{args.data_path}:{index + 1} has no prompt/question")
            images = record.get("images") or ([record["image"]] if record.get("image") else [])
            if images and not with_images:
                raise ValueError(f"{args.data_path}:{index + 1} has images; use the understanding entrypoint")
            if args.__dict__.get("image_root"):
                images = [p if os.path.isabs(p) else os.path.join(args.image_root, p) for p in images]
            items.append({
                "id": str(record.get("id", index)),
                "prompt": prompt,
                "images": images,
                "system": record.get("system", args.system),
            })
    return items


def _done_ids(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def _read_records(path: str) -> Dict[str, Dict]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        records = (json.loads(line) for line in handle if line.strip())
        return {record["id"]: record for record in records}


def _merge_shards(output: str, shards: List[str], order: List[str]) -> None:
    records = _read_records(output)
    for shard in shards:
        records.update(_read_records(shard))
    with open(output + ".tmp", "w", encoding="utf-8") as handle:
        for key in order:
            if key in records:
                handle.write(json.dumps(records[key], ensure_ascii=False) + "\n")
    os.replace(output + ".tmp", output)
    for shard in shards:
        os.remove(shard)


@torch.no_grad()
def run(args, with_images: bool) -> None:
    """Build the model, shard items over data-parallel ranks and write jsonl."""
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoTokenizer, Qwen2VLImageProcessor

    from megatron.core import parallel_state
    from recipe.transfusion.eval.inference_transfusion_qwen3vl_pixel_dit_i2i import (
        build_pixel_dit_from_args,
    )
    from recipe.transfusion.eval.pixel_dit_runtime import cleanup_dist, setup_dist

    _, rank, _, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    items = load_items(args, with_images)
    model, ckpt_dir = build_pixel_dit_from_args(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
    processor = Qwen2VLImageProcessor.from_pretrained(args.hf_path) if with_images else None
    decoder = PixelDiTTextDecoder(
        model, tokenizer, args.hf_path, processor=processor, route_token=args.route_token,
        min_pixels=getattr(args, "min_pixels", 0), max_pixels=getattr(args, "max_pixels", 0),
    )
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world = parallel_state.get_data_parallel_world_size()
    writer = parallel_state.get_tensor_model_parallel_rank() == 0
    prefill = THINK_PREFILLS[args.think_mode]
    if rank == 0:
        print(f"[decode] ckpt={ckpt_dir} items={len(items)} think_mode={args.think_mode} "
              f"route_token={decoder.route_token}", flush=True)

    shard = args.output
    if args.output and dp_world > 1:
        shard = f"{args.output}.rank{dp_rank}-of-{dp_world}"
    done = _done_ids(args.output) | _done_ids(shard) if args.output and args.resume else set()
    if args.output and writer:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    handle = open(shard, "a" if args.resume else "w", encoding="utf-8") if args.output and writer else None
    try:
        for item in tqdm(items[dp_rank::dp_world], disable=rank != 0 or not args.output):
            if item["id"] in done:
                continue
            images = [Image.open(path).convert("RGB") for path in item["images"]]
            result = decoder.generate(
                item["prompt"], images=images, system=item["system"], prefill=prefill,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p, seed=args.seed,
                max_length=args.max_length, max_think_tokens=args.max_think_tokens,
            )
            record = {"id": item["id"], "prompt": item["prompt"], "images": item["images"], **result}
            if handle is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            elif writer:
                print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    finally:
        if handle is not None:
            handle.close()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if args.output and dp_world > 1 and rank == 0:
        shards = [f"{args.output}.rank{r}-of-{dp_world}" for r in range(dp_world)]
        _merge_shards(args.output, [s for s in shards if os.path.isfile(s)],
                      [item["id"] for item in items])
    # Megatron initialises a process group even for a single process.
    cleanup_dist(torch.distributed.is_initialized())
