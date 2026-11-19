"""Benchmark input and output contracts, independent of the model runtime."""

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


ONEIG_CATEGORIES = {
    "Anime_Stylization": "anime", "Portrait": "human", "General_Object": "object",
    "Text_Rendering": "text", "Knowledge_Reasoning": "reasoning", "Multilingualism": "multilingualism",
}


@dataclass(frozen=True)
class Prompt:
    """A stable benchmark identifier and its language-specific generation prompt."""

    item_id: int
    text: str
    category: str = ""


def numeric_id(value):
    """Normalize zero-padded IDs without silently truncating fractional values."""
    text = str(value).strip()
    if not text.isascii() or not text.isdigit():
        raise ValueError(f"Expected a nonnegative integer benchmark ID, got {value!r}")
    return int(text)


def read_rows(path, csv_format):
    """Stream rows so Qwen benchmark judge responses are not retained in memory."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        if csv_format:
            yield from csv.DictReader(handle)
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_prompts(path, benchmark, language="EN", prompt_column=None):
    """Read OneIG CSV, LongText TXT/JSONL, or bilingual Qwen-Image-Bench JSONL."""
    path = Path(path)
    records = []
    if benchmark == "LongText" and path.suffix.lower() == ".txt":
        with path.open(encoding="utf-8-sig") as handle:
            texts = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
        return [Prompt(i, text) for i, text in enumerate(texts)]
    seen = set()
    language_column = "prompt_en" if language == "EN" else "prompt_cn"
    for index, row in enumerate(read_rows(path, benchmark == "OneIG")):
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at record {index + 1}")
        # Local OneIG EN CSV stores English text under the historical prompt_cn header.
        columns = [prompt_column] if prompt_column else [language_column, "prompt", "text"]
        if benchmark == "OneIG" and language == "EN" and "prompt_en" not in row:
            columns.append("prompt_cn")
        text = next((row.get(key) for key in columns if isinstance(row.get(key), str) and row[key].strip()), None)
        if text is None:
            raise ValueError(f"Missing {language} prompt at record {index + 1}; expected one of {columns}")
        item_id = next((row[key] for key in ("ID", "id", "item_id", "prompt_id") if key in row), None)
        if item_id is None:
            raise ValueError(f"Missing benchmark ID at record {index + 1}")
        item_id = numeric_id(item_id)
        category = ""
        if benchmark == "OneIG":
            category = ONEIG_CATEGORIES.get(row.get("category"), row.get("category"))
            if category not in ONEIG_CATEGORIES.values():
                raise ValueError(f"Unknown OneIG category at record {index + 1}: {category!r}")
        identity = (category, item_id)
        if identity in seen:
            raise ValueError(f"Duplicate benchmark ID: {identity}")
        seen.add(identity)
        records.append(Prompt(item_id, text.strip(), category))
    if not records:
        raise ValueError(f"No prompts found in {path}")
    return records


def validate_model_name(name):
    """Keep the evaluator's model name a single nonempty directory component."""
    if not name or name in (".", "..") or any(c in name for c in ("/", "\\", "\0")):
        raise ValueError("model-name must be a single directory component")


def sample_path(prompt, benchmark, model_name, repeat, repeats):
    """Return an evaluator-compatible relative image filename."""
    if benchmark == "OneIG":
        root = "images" if repeats == 1 else "samples"
        suffix = "" if repeats == 1 else f"_{repeat}"
        return f"{root}/{prompt.category}/{model_name}/{prompt.item_id}{suffix}.webp"
    return f"images/{prompt.item_id:05d}_{repeat}.jpg"


def gallery_path(prompt, model_name):
    """Return the official OneIG gallery location."""
    return f"images/{prompt.category}/{model_name}/{prompt.item_id}.webp"


def valid_image(path, size):
    """Check decoding and dimensions before reusing an image during resume."""
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == size and image.mode == "RGB"
    except (OSError, ValueError):
        return False


def save_image_atomic(image, path):
    """Avoid exposing partially written images to resume or evaluators."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        image.save(temporary, quality=95)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path, value, jsonl=False):
    """Publish manifests and evaluator input only after writing them completely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            if jsonl:
                for row in value:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prompt_digest(prompts):
    """Fingerprint the selected ordered prompts for resume validation."""
    payload = json.dumps([vars(item) for item in prompts], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finalize_outputs(prompts, output, benchmark, model_name, repeats, size):
    """Validate completeness and emit metadata, OneIG galleries, and Qwen judge inputs."""
    from PIL import Image

    output = Path(output)

    def finalize_prompt(prompt):
        paths = [sample_path(prompt, benchmark, model_name, repeat, repeats) for repeat in range(repeats)]
        for path in paths:
            if not valid_image(output / path, size):
                raise ValueError(f"Missing, corrupt, or wrong-size image: {output / path}")
        entry = {"prompt_id": prompt.item_id, "prompt": prompt.text, "images": paths}
        if benchmark == "OneIG":
            entry["category"] = prompt.category
            entry["gallery"] = gallery_path(prompt, model_name)
            if repeats == 4:
                with Image.new("RGB", (size[0] * 2, size[1] * 2)) as gallery:
                    for i, path in enumerate(paths):
                        with Image.open(output / path) as image:
                            gallery.paste(image, ((i % 2) * size[0], (i // 2) * size[1]))
                    save_image_atomic(gallery, output / entry["gallery"])
        return entry

    if benchmark == "OneIG" and repeats == 4:
        workers = os.cpu_count() or 1
        print(f"Finalizing OneIG galleries: prompts={len(prompts)} threads={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # map preserves prompt order; exceptions prevent publishing metadata.
            metadata = list(executor.map(finalize_prompt, prompts))
    else:
        metadata = [finalize_prompt(prompt) for prompt in prompts]

    judge_inputs = [[] for _ in range(repeats)]
    if benchmark == "QwenImageBench":
        for entry in metadata:
            for repeat, path in enumerate(entry["images"]):
                judge_inputs[repeat].append({"ID": entry["prompt_id"], "prompt": entry["prompt"],
                                             "image_path": str((output / path).resolve())})
    write_json_atomic(output / "metadata.jsonl", metadata, jsonl=True)
    if benchmark == "QwenImageBench":
        for repeat, records in enumerate(judge_inputs):
            name = "judge_input.jsonl" if repeat == 0 else f"judge_input_{repeat}.jsonl"
            write_json_atomic(output / name, records, jsonl=True)
    return metadata
