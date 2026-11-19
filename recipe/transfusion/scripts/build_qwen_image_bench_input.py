#!/usr/bin/env python3
"""Build the JSONL input expected by Qwen-Image-Bench ``judge.py``.

Pixel-DiT writes ``metadata.jsonl`` with paths relative to its output
directory.  The judge instead expects one record per benchmark item with the
numeric ``ID``, prompt text, and an absolute image path.
"""

import argparse
import json
from pathlib import Path


def _numeric_id(value):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Qwen-Image-Bench ID must be numeric, got {value!r}") from exc


def build_input(metadata_path: Path, output_dir: Path, output_path: Path, repeat_index: int):
    records = []
    seen_ids = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = _numeric_id(item.get("prompt_id", item.get("ID", item.get("id"))))
            if item_id in seen_ids:
                raise ValueError(f"Duplicate Qwen-Image-Bench ID {item_id} at line {line_number}")
            seen_ids.add(item_id)

            prompt = item.get("prompt")
            images = item.get("images") or []
            if prompt is None:
                raise ValueError(f"Missing prompt at metadata line {line_number}")
            if not images:
                raise ValueError(f"Missing generated image at metadata line {line_number}")
            if repeat_index < 0 or repeat_index >= len(images):
                raise ValueError(
                    f"repeat index {repeat_index} is out of range for ID {item_id} "
                    f"({len(images)} generated images)"
                )

            image_path = Path(images[repeat_index])
            if not image_path.is_absolute():
                image_path = output_dir / image_path
            records.append(
                {
                    "ID": item_id,
                    "prompt": prompt,
                    "image_path": str(image_path.resolve()),
                }
            )

    if not records:
        raise ValueError(f"No records found in {metadata_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat-index", type=int, default=0)
    args = parser.parse_args()
    count = build_input(args.metadata, args.output_dir, args.output, args.repeat_index)
    print(f"Wrote {count} Qwen-Image-Bench records to {args.output}")


if __name__ == "__main__":
    main()
