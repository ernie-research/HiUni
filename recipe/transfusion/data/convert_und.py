# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert JSONL data to WebDataset format for Transfusion pretraining.

Input JSONL format:
{
    "data_id": "xxx",
    "image_info": [
        {"image": "images/xxx_0.jpg", "matched_text_index": 18, ...},
        {"url": "https://...", "matched_text_index": 2, ...},
        ...
    ],
    "text_list": [
        {"text": "...", "tag": "no_mask", "is_valid": true},
        ...
    ]
}

The image field may be named image / image_path / path / url and may be
a local path (absolute, relative, file://) or an http(s):// URL. Relative
local paths are resolved against --image-root (default: the JSONL's directory)
and stored as absolute paths; --keep-relative-paths stores them unchanged for
TRANSFUSION_IMAGE_ROOT at training time. Samples with a missing local file are
skipped.

Output WebDataset format (tar):
- Each sample contains:
    - __key__: data_id
    - task_type: task_type (pickled), separate file (e.g., understanding.task_type)
    - jpgs: list of image paths / URLs (pickled), ordered by matched_text_index
    - json: {"text": [...], "mask": [...]}

For pretraining:
- text: List of text fragments with <image> placeholders inserted at matched_text_index positions
- mask: List of mask values (1=train, 0=skip) for each fragment in text list
        Text fragments with tag="no_mask" have mask=1
        Text fragments with tag="mask" have mask=0
        <image> placeholders always have mask=0

Example Usage:
python convert_und.py \\
    --input-jsonl /path/to/data.jsonl \\
    --output-dir /path/to/output_wds \\
    --max-samples-per-tar 1000
"""

import argparse
import json
import os
import pickle
import sys
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import webdataset as wds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_source import (  # noqa: E402
    MissingImageError,
    add_image_args,
    default_image_root,
    get_image_source,
    normalize_image_source,
)


TEXTUAL_DATA_MARKERS = (
    "<image>",
    "<|image_pad|>",
    "<image_pad>",
    "<|endoftext|>",
)


def point_to_refbox(content: str, points: List) -> str:
    """Convert OCR quadrilaterals to the Qwen ref/box token format."""
    boxes = []
    for point in points:
        box = "".join(f"({x},{y})" for x, y in point)
        boxes.append(f"<|box_start|>{box}<|box_end|>")
    return f"<|object_ref_start|>{content}<|object_ref_end|>{''.join(boxes)}"


def clean_understanding_text(text_item: Dict[str, Any]) -> str:
    text = text_item.get("text", "")
    if (
        text_item.get("text_type", "").lower() == "ocr"
        and text_item.get("points")
    ):
        text = point_to_refbox(text, text_item["points"])
    for marker in TEXTUAL_DATA_MARKERS:
        text = text.replace(marker, "")
    return text.strip()


def parse_jsonl_line(line: str) -> Dict[str, Any]:
    """Parse a single JSONL line."""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSONL line: {e}")
        return None


def extract_images_with_order(
    image_info: List[Dict],
    image_root: Optional[str] = None,
    keep_relative: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Extract image references and their matched_text_index from image_info.

    Each entry names its image under image / image_path / path / url
    (local path or http(s):// URL); entries without one are dropped.

    Args:
        image_info: List of image dictionaries
        image_root: Directory relative local paths are resolved against
        keep_relative: Store relative local paths unchanged

    Returns:
        Tuple of (image_sources, matched_text_indices), both ordered by matched_text_index

    Raises:
        MissingImageError: a resolved local image file does not exist
    """
    if not image_info:
        return [], []

    # Collect (matched_text_index, image source) pairs
    image_data = []
    for img in image_info:
        image_source = get_image_source(img)
        matched_index = img.get("matched_text_index", 0)
        if image_source:
            image_source = normalize_image_source(image_source, image_root, keep_relative)
            try:
                matched_index = max(0, int(matched_index))
            except (TypeError, ValueError):
                matched_index = 0
            image_data.append((matched_index, image_source))

    # Sort by matched_text_index to get the correct order
    image_data.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))

    # Extract ordered URLs and indices
    image_urls = [url for _, url in image_data]
    matched_indices = [idx for idx, _ in image_data]

    return image_urls, matched_indices


def build_pretraining_text_with_mask(
    text_list: List[Dict],
    image_text_order: List[int]
) -> Tuple[List[str], List[int]]:
    """
    Build pretraining text with <image> placeholders and training mask.

    The text and mask are returned as lists, where each element is a fragment:
    - For text: the text string
    - For image: "<image>" placeholder
    The mask indicates which fragments participate in training (1=train, 0=skip).

    Args:
        text_list: List of text dictionaries with 'text', 'tag', 'is_valid' fields
        image_text_order: List of matched_text_index values indicating where to insert <image>

    Returns:
        Tuple of (text_fragments, mask_list)
        text_fragments: List of text/image fragments
        mask_list: List of mask values (1=train, 0=skip) for each fragment

    Example:
        text_list = [
            {"text": "Hello", "tag": "no_mask"},    # index 0
            {"text": "world", "tag": "no_mask"},    # index 1
            {"text": "!", "tag": "no_mask"}         # index 2
        ]
        image_text_order = [1]

        Result:
            text = ["Hello", "<image>", "world", "!"]
            mask = [1, 0, 1, 1]
    """
    # Preserve multiplicity: several images may share one matched_text_index.
    image_counts = {}
    for position in image_text_order:
        image_counts[position] = image_counts.get(position, 0) + 1

    text_fragments = []
    mask_list = []

    for i, text_item in enumerate(text_list):
        # Check if an image should be inserted before this text
        for _ in range(image_counts.get(i, 0)):
            text_fragments.append("<image>")
            mask_list.append(0)  # Images don't participate in training

        # Add text if valid
        is_valid = text_item.get("is_valid", True)
        tag = text_item.get("tag", "no_mask")
        text = clean_understanding_text(text_item)

        if text:
            text_fragments.append(text)
            # Mask=1 for training (no_mask tag), Mask=0 for skip (mask tag)
            mask_list.append(1 if is_valid and tag == "no_mask" else 0)
        else:
            # Empty text also doesn't participate
            text_fragments.append("")
            mask_list.append(0)

    # Check if there are images at the end (after all text)
    for position in sorted(image_counts):
        if position >= len(text_list):
            for _ in range(image_counts[position]):
                text_fragments.append("<image>")
                mask_list.append(0)

    return text_fragments, mask_list


def convert_sample(data: Dict[str, Any], source_id: str = None) -> Dict[str, Any]:
    """
    Convert a single JSONL data sample to WebDataset format.

    Args:
        data: Raw JSONL data

    Returns:
        WebDataset sample dict with __key__, jpgs, task_type, and json fields.
        json contains: {"text": List[str], "mask": List[int]}
        text is a list of text/image fragments with <image> placeholders
        mask corresponds 1:1 with text list elements
    """
    data_id = data.get("data_id", "")

    # Extract images with their order (by matched_text_index)
    image_info = data.get("image_info", [])
    image_urls, image_text_order = extract_images_with_order(image_info)

    # Build pretraining text with <image> placeholders and mask
    text_list = data.get("text_list", [])
    text, mask = build_pretraining_text_with_mask(text_list, image_text_order)

    # Build WebDataset sample
    json_data = {"text": text, "mask": mask}
    resolved_source_id = source_id or data.get("source_id") or data.get("release_no")
    if resolved_source_id:
        json_data["source_id"] = resolved_source_id

    sample = {
        "__key__": data_id,
        "task_type": "understanding".encode("utf-8"),
        "json": json.dumps(json_data).encode("utf-8"),
    }

    # Add images (if any)
    if image_urls:
        sample["jpgs"] = pickle.dumps(image_urls)

    return sample


def convert_jsonl_to_wds(
    input_jsonl: str,
    output_dir: str,
    max_samples_per_tar: int = 1000,
    task_type: str = "understanding",
    source_id: str = None,
    image_root: Optional[str] = None,
    keep_relative: bool = False,
) -> str:
    """
    Convert JSONL data to WebDataset format.

    Args:
        input_jsonl: Path to input JSONL file
        output_dir: Directory to save WebDataset tar files
        max_samples_per_tar: Maximum samples per tar file
        task_type: Task type ("understanding" or "generation")
        source_id: Optional prompt mapping id applied to every output sample
        image_root: Directory relative local image paths are resolved against
            (default: the directory of input_jsonl)
        keep_relative: Store relative local paths unchanged

    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    image_root = default_image_root(input_jsonl, image_root)

    # Output pattern for tar files
    output_pattern = os.path.join(output_dir, f"{task_type}-%d.tar")

    samples_count = 0
    valid_samples = []

    # First pass: validate and prepare samples
    print(f"Reading JSONL file: {input_jsonl}")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing JSONL"):
            line = line.strip()
            if not line:
                continue

            data = parse_jsonl_line(line)
            if not data:
                continue

            data_id = data.get("data_id", "")
            if not data_id:
                continue

            # Extract images with order
            image_info = data.get("image_info", [])
            try:
                image_urls, image_text_order = extract_images_with_order(
                    image_info, image_root, keep_relative
                )
            except MissingImageError as e:
                print(f"Skip sample {data_id}: {e}")
                continue
            text_list = data.get("text_list", [])
            has_valid_text = any(
                item.get("is_valid", True) and item.get("tag") == "no_mask" and item.get("text", "").strip()
                for item in text_list
            )

            if task_type == "understanding":
                is_valid_sample = bool(image_urls) and has_valid_text
            else:
                is_valid_sample = bool(image_urls) or has_valid_text

            if not is_valid_sample:
                continue

            sample_source_id = source_id or data.get("source_id") or data.get("release_no")
            valid_samples.append(
                (data_id, image_urls, image_text_order, text_list, sample_source_id)
            )
            samples_count += 1

    print(f"Total valid samples: {samples_count}")

    if samples_count == 0:
        print("No valid samples found!")
        return output_dir

    # Second pass: write to WebDataset
    print(f"Writing WebDataset to: {output_dir}")
    with wds.ShardWriter(output_pattern, maxcount=max_samples_per_tar) as shard_writer:
        for data_id, image_urls, image_text_order, text_list, sample_source_id in tqdm(
            valid_samples, desc="Writing WebDataset"
        ):
            # Build pretraining text with <image> placeholders and mask
            text, mask = build_pretraining_text_with_mask(text_list, image_text_order)

            # Build sample
            json_data = {"text": text, "mask": mask}
            if sample_source_id:
                json_data["source_id"] = sample_source_id
            sample = {
                "__key__": data_id,
                "task_type": task_type.encode("utf-8"),
                "json": json.dumps(json_data).encode("utf-8"),
            }

            # Add images (if any)
            if image_urls:
                sample["jpgs"] = pickle.dumps(image_urls)

            # Write sample to shard
            shard_writer.write(sample)

    print(f"Conversion complete! Output directory: {output_dir}")
    return output_dir


def main():
    """
    Convert JSONL data to WebDataset format for Transfusion pretraining.
    """
    parser = argparse.ArgumentParser(
        description="Convert JSONL data to WebDataset format for Transfusion pretraining"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save WebDataset tar files"
    )
    parser.add_argument(
        "--max-samples-per-tar",
        type=int,
        default=1000,
        help="Maximum samples per tar file (default: 1000)"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="understanding",
        choices=["understanding", "generation", "editing"],
        help="Task type (default: understanding)"
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default=None,
        help="Prompt mapping source id; overrides source_id/release_no in each record",
    )
    add_image_args(parser)

    args = parser.parse_args()

    convert_jsonl_to_wds(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        max_samples_per_tar=args.max_samples_per_tar,
        task_type=args.task_type,
        source_id=args.source_id,
        image_root=args.image_root,
        keep_relative=args.keep_relative_paths,
    )


if __name__ == "__main__":
    main()
