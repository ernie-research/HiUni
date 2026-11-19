#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert JSON/JSONL to Megatron Energon WebDataset format (Editing task).

基于 datafactory/mapper/mapper.py 和 datafactory/reducer/reducer.py 的模式
实现单条 JSON 转换为 Energon 格式 (tar + tar.idx + .nv-meta)

Input JSON format (Editing task):
{
    "data_id": "xxx",
    "image_info": [
        {"image": "images/xxx_src.jpg", "matched_text_index": 0, ...},
        {"image": "images/xxx_dst.jpg", "matched_text_index": 1, ...},
        ...
    ],
    "text_list": [
        {"text": "...", "tag": "no_mask", "is_valid": true},
        ...
    ],
    "order": {
        "type": ["image", "text", "image", "image", "image"],
        "index": [0, 0, 1, 2, 3],
        "mask": [1, 1, 0, 0, 0]   # 0=需要训练, 1=不需要训练
    }
}

当 order={} 为空时，默认仅最后一张图需要训练。

The image field may be named image / image_path / path / url and may be
a local path (absolute, relative, file://) or an http(s):// URL. Relative
local paths are resolved against --image-root (default: the input file's
directory) and stored as absolute paths; --keep-relative-paths stores them
unchanged for TRANSFUSION_IMAGE_ROOT at training time. A sample with an image
entry that has no image field, or whose local file is missing, is skipped.

Output WebDataset format:
- tar: WebDataset 格式，包含 __key__, task_type, jpgs, json 字段
- tar.idx: Energon 索引文件，记录每个样本的偏移量
- .nv-meta: Energon 元数据文件，描述数据集信息

Example Usage:
    python convert_i2i.py --input-json /path/to/data.jsonl --output-dir /path/to/output --task-type editing
"""

import argparse
import json
import os
import pickle
import struct
import sys
import tarfile
from typing import Dict, Any, List, Tuple, Optional
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_source import (  # noqa: E402
    IMAGE_KEYS,
    add_image_args,
    default_image_root,
    get_image_source,
    normalize_image_source,
)

# 日志配置
log = logging.getLogger(__name__)
stream_hdl = logging.StreamHandler()
stream_hdl.setFormatter(logging.Formatter(
    fmt="[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d]: %(message)s"
))
log.addHandler(stream_hdl)
log.setLevel(logging.INFO)


def ensure_id(data: Dict[str, Any]) -> str:
    """确保数据有唯一 ID，无则自动生成"""
    import uuid
    data_id = data.get("data_id", "") or data.get("id", "")
    if not data_id:
        data_id = str(uuid.uuid4())[:8]
        data["data_id"] = data_id
    return data_id


def extract_images_with_order(
    image_info: List[Dict],
    image_root: Optional[str] = None,
    keep_relative: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Extract image references and their matched_text_index from image_info.

    Each entry names its image under image / image_path / path / url
    (local path or http(s):// URL). Every entry must have one, because
    the <image> placeholders are built from image_info / order as well.

    Args:
        image_info: List of image dictionaries
        image_root: Directory relative local paths are resolved against
        keep_relative: Store relative local paths unchanged

    Returns:
        Tuple of (image_sources, matched_text_indices), both ordered by matched_text_index

    Raises:
        ValueError: an entry has no image reference, or a local file is missing
    """
    if not image_info:
        return [], []

    image_data = []
    for img in image_info:
        src = get_image_source(img)
        if not src:
            raise ValueError(f"image_info entry has no image field ({'/'.join(IMAGE_KEYS)})")
        src = normalize_image_source(src, image_root, keep_relative)
        image_data.append((img.get("matched_text_index"), src))

    image_data.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))

    image_urls = [url for _, url in image_data]
    matched_indices = [idx for idx, _ in image_data]
    return image_urls, matched_indices


def build_editing_text_with_mask(
    text_list: List[Dict],
    image_info: List[Dict],
    order: Dict[str, List]
) -> Tuple[List[str], List[int]]:
    """
    构建图片编辑格式的 text 和 mask

    order 格式:
        type: ["image", "text", "image", "image", "image"]
        index: [0, 0, 1, 2, 3]  # image_info 中的索引
        mask: [1, 1, 0, 0, 0]   # 0=需要训练, 1=不需要训练

    当 order={} 为空时，根据 matched_text_index 确认 <image> 加在哪条 text 之前

    Args:
        text_list: List of text dictionaries
        image_info: List of image info
        order: Order 字典

    Returns:
        Tuple of (text_fragments, mask_list)
    """
    text_fragments = []
    mask_list = []

    if not order:
        # order 为空，根据 matched_text_index 确认 <image> 加在哪条 text 之前
        # 收集每个图像的 matched_text_index
        image_positions = {}
        for idx, img in enumerate(image_info):
            matched_index = img.get("matched_text_index")
            if matched_index is not None:
                image_positions[matched_index] = idx

        # 遍历 text_list，在对应位置插入 <image>
        for i, text_item in enumerate(text_list):
            # 检查是否有图像需要插入在这条 text 之前
            if i in image_positions:
                img_idx = image_positions[i]
                text_fragments.append("<image>")
                # 默认只有最后一张图需要训练
                is_last_image = (img_idx == len(image_info) - 1)
                mask_list.append(0 if is_last_image else 1)

            # 添加文本
            text = text_item.get("text", "").strip()
            if text:
                text_fragments.append(text)
            else:
                text_fragments.append("")
            # 文本不参与训练
            mask_list.append(1)

        # 处理文本之后可能还有的图像
        remaining_images = [idx for idx, img in enumerate(image_info)
                          if img.get("matched_text_index") >= len(text_list)]
        for img_idx in remaining_images:
            text_fragments.append("<image>")
            is_last_image = (img_idx == len(image_info) - 1)
            mask_list.append(0 if is_last_image else 1)

        return text_fragments, mask_list

    order_type = order.get("type", [])
    order_index = order.get("index", [])
    order_mask = order.get("mask", [])

    # 按 order 遍历构建序列
    for i, (item_type, idx, mask_val) in enumerate(zip(order_type, order_index, order_mask)):
        if item_type == "image":
            # 图像
            text_fragments.append("<image>")
            # mask_val: 0=需要训练, 1=不需要训练
            mask_list.append(mask_val)
        elif item_type == "text":
            # 文本，从 text_list 中获取
            if idx < len(text_list):
                text = text_list[idx].get("text", "").strip()
                if text:
                    text_fragments.append(text)
                else:
                    text_fragments.append("")
                # 编辑模式下，文本不参与训练，mask=1
                mask_list.append(1)
            else:
                # 超出范围的文本
                text_fragments.append("")
                mask_list.append(1)

    return text_fragments, mask_list


def create_tar_index(tar_path: str) -> str:
    """
    为 tar 文件生成 Energon 格式的索引文件 (.tar.idx)

    Energon 索引格式：连续的 64位无符号整数(Q)，记录每个样本在 tar 包中的起始字节偏移量，
    并在最后记录文件总大小作为哨兵。

    Args:
        tar_path: tar 文件路径

    Returns:
        索引文件路径
    """
    idx_path = tar_path + ".idx"
    sample_count = 0

    log.info(f"生成索引文件: {idx_path}")

    try:
        with open(tar_path, "rb") as f_in, open(idx_path, "wb") as f_out:
            tar = tarfile.open(fileobj=f_in, mode="r|")

            last_base_name = None

            for member in tar:
                if not member.isfile():
                    continue

                if member.name.startswith("__") or member.name.startswith("."):
                    continue

                if "." in member.name:
                    base_name = member.name.rsplit(".", 1)[0]
                else:
                    base_name = member.name

                if base_name != last_base_name:
                    f_out.write(struct.pack("Q", member.offset))
                    last_base_name = base_name
                    sample_count += 1

            f_out.write(struct.pack("Q", f_in.tell()))

        log.info(f"索引生成完成: {sample_count} 个样本")
        return idx_path

    except Exception as e:
        log.error(f"索引生成失败: {e}")
        raise


def create_nv_meta(
    output_dir: str,
    tar_name: str,
    sample_count: int,
    split_ratio: Tuple[int, int, int] = (10, 0, 0),
    sample_type: str = "crude",
    field_map: Optional[str] = None
) -> str:
    """
    创建 Energon 元数据目录 (.nv-meta)

    Args:
        output_dir: 输出目录
        tar_name: tar 文件名（不含扩展名）
        sample_count: 样本总数
        split_ratio: (train, val, test) 划分比例
        sample_type: 样本类型 (crude/text/interleaved 等)
        field_map: 字段映射（某些 sample_type 需要）

    Returns:
        元数据目录路径
    """
    nv_meta_dir = os.path.join(output_dir, ".nv-meta")
    os.makedirs(nv_meta_dir, exist_ok=True)

    # .info.yaml - shard 信息
    info_yaml = os.path.join(nv_meta_dir, ".info.yaml")
    with open(info_yaml, "w", encoding="utf-8") as f:
        f.write(f"shard_counts:\n")
        f.write(f"  {tar_name}.tar: {sample_count}\n")

    # split.yaml - 数据划分
    split_yaml = os.path.join(nv_meta_dir, "split.yaml")
    total_ratio = sum(split_ratio)
    train_ratio = split_ratio[0] / total_ratio

    with open(split_yaml, "w", encoding="utf-8") as f:
        f.write(f"split_parts:\n")
        f.write(f"  train:\n")
        f.write(f"    - {tar_name}.tar\n")
        f.write(f"  val: []\n")
        f.write(f"  test: []\n")

    # dataset.yaml - 数据集类型配置
    # Sample type 映射表
    SAMPLE_TYPES = {
        "text": {
            "module": "megatron.energon",
            "class": "TextSample",
            "needs_field_map": True,
        },
        "caption": {
            "module": "megatron.energon",
            "class": "CaptioningSample",
            "needs_field_map": True,
        },
        "interleaved": {
            "module": "megatron.energon",
            "class": "InterleavedSample",
            "needs_field_map": True,
        },
        "vqa": {
            "module": "megatron.energon",
            "class": "VQASample",
            "needs_field_map": True,
        },
        "crude": {
            "module": "megatron.energon",
            "class": "CrudeWebdataset",
            "needs_field_map": False,
        },
    }

    type_info = SAMPLE_TYPES.get(sample_type, SAMPLE_TYPES["crude"])

    dataset_yaml = os.path.join(nv_meta_dir, "dataset.yaml")
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        f.write(f"__module__: {type_info['module']}\n")
        f.write(f"__class__: {type_info['class']}\n")
        if type_info["needs_field_map"] and field_map:
            f.write(f"field_map:\n")
            f.write(f"  text: {field_map}\n")

    log.info(f"生成元数据目录: {nv_meta_dir}")
    log.info(f"  - .info.yaml")
    log.info(f"  - split.yaml")
    log.info(f"  - dataset.yaml (sample_type={sample_type})")

    return nv_meta_dir


def convert_sample(
    data: Dict[str, Any],
    task_type: str = "understanding",
    image_root: Optional[str] = None,
    keep_relative: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    转换单条 JSON 数据为 WebDataset 样本格式（适配图片编辑格式）

    Args:
        data: 原始 JSON 数据
        task_type: 任务类型
        image_root: 相对本地路径的解析根目录
        keep_relative: 保留相对路径，训练时按 TRANSFUSION_IMAGE_ROOT 解析

    Returns:
        WebDataset 样本字典；图像缺失时返回 None
    """
    data_id = ensure_id(data)

    image_info = data.get("image_info", [])
    try:
        image_urls, _ = extract_images_with_order(image_info, image_root, keep_relative)
    except ValueError as e:
        log.warning(f"样本 {data_id}: {e}，跳过")
        return None

    text_list = data.get("text_list", [])
    order = data.get("order", {})

    # 使用编辑格式构建 text 和 mask
    text, mask = build_editing_text_with_mask(text_list, image_info, order)

    sample = {
        "__key__": data_id,
        "task_type": task_type,
        "json": {
            "text": text,
            "mask": [1 - x for x in mask],
        },
    }

    if image_urls:
        sample["jpgs"] = image_urls

    return sample


def write_webdataset(
    samples: List[Dict[str, Any]],
    tar_path: str,
) -> int:
    """
    将样本写入 tar 文件

    Args:
        samples: 样本列表
        tar_path: 输出 tar 文件路径

    Returns:
        写入的样本数
    """
    sample_count = 0

    with tarfile.open(tar_path, "w") as tar:
        for sample in samples:
            key = sample["__key__"]

            # 写入 task_type (pickled)
            task_type_bytes = pickle.dumps(sample["task_type"])
            task_type_info = tarfile.TarInfo(name=f"{key}.task_type")
            task_type_info.size = len(task_type_bytes)
            tar.addfile(task_type_info, __import__('io').BytesIO(task_type_bytes))

            # 写入 json
            json_bytes = json.dumps(sample["json"], ensure_ascii=False).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, __import__('io').BytesIO(json_bytes))

            # 写入 jpgs (如果有)
            if "jpgs" in sample and sample["jpgs"]:
                jpgs_bytes = pickle.dumps(sample["jpgs"])
                jpgs_info = tarfile.TarInfo(name=f"{key}.jpgs")
                jpgs_info.size = len(jpgs_bytes)
                tar.addfile(jpgs_info, __import__('io').BytesIO(jpgs_bytes))

            sample_count += 1
            log.info(f"已写入样本: {key}")

    return sample_count


def parse_jsonl_line(line: str) -> Dict[str, Any]:
    """解析单行 JSONL 数据"""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError as e:
        log.warning(f"JSONL 解析失败: {e}")
        return None


def convert_json_to_wds(
    input_json: str,
    output_dir: str,
    output_name: str = "data",
    task_type: str = "understanding",
    sample_type: str = "crude",
    field_map: Optional[str] = None,
    image_root: Optional[str] = None,
    keep_relative: bool = False,
) -> Tuple[str, str, str]:
    """
    将 JSON/JSONL 文件转换为 Energon WebDataset 格式

    Args:
        input_json: 输入 JSON/JSONL 文件路径
        output_dir: 输出目录
        output_name: 输出文件名（不含扩展名）
        task_type: 任务类型（用于样本中的 task_type 字段）
        sample_type: Energon 样本类型 (crude/text/interleaved 等)
        field_map: 字段映射（某些 sample_type 需要）
        image_root: 相对本地路径的解析根目录（默认输入文件所在目录）
        keep_relative: 保留相对路径，训练时按 TRANSFUSION_IMAGE_ROOT 解析

    Returns:
        Tuple of (tar_path, idx_path, meta_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    image_root = default_image_root(input_json, image_root)

    def convert(item):
        return convert_sample(item, task_type, image_root, keep_relative)

    tar_path = os.path.join(output_dir, f"{output_name}.tar")

    # 读取数据（支持 JSON 和 JSONL 格式）
    log.info(f"读取数据文件: {input_json}")
    samples = []

    with open(input_json, "r", encoding="utf-8") as f:
        # 尝试判断是 JSON 还是 JSONL
        first_line = f.readline().strip()
        f.seek(0)

        # 如果第一行是完整 JSON 对象且有后续数据，则为 JSONL
        try:
            json.loads(first_line)
            # 尝试读取第二行判断
            second_line = f.readline()
            f.seek(0)
            is_jsonl = len(second_line.strip()) > 0
        except:
            is_jsonl = True

        if is_jsonl:
            # JSONL 格式：每行一个 JSON
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = parse_jsonl_line(line)
                if data:
                    samples.append(convert(data))
        else:
            # 单个 JSON 对象格式
            data = json.load(f)
            if isinstance(data, list):
                samples = [convert(item) for item in data]
            else:
                samples = [convert(data)]

    samples = [s for s in samples if s is not None]
    log.info(f"处理了 {len(samples)} 个有效样本")

    if not samples:
        log.warning("没有有效的样本，跳过写入")
        return tar_path, "", ""

    # 写入 tar 文件
    log.info(f"写入 tar 文件: {tar_path}")
    count = write_webdataset(samples, tar_path)
    log.info(f"写入完成: {count} 个样本")

    # 生成索引文件
    idx_path = create_tar_index(tar_path)

    # 生成元数据目录
    meta_dir = create_nv_meta(output_dir, output_name, count, (10, 0, 0), sample_type, field_map)

    log.info("转换完成!")
    return tar_path, idx_path, meta_dir


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert single JSON/JSONL to Megatron Energon WebDataset format"
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to input JSON/JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for WebDataset files"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="data",
        help="Output file name without extension (default: data)"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="understanding",
        choices=["understanding", "generation", "editing"],
        help="Task type field in sample (default: understanding)"
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        default="crude",
        choices=["crude", "text", "interleaved", "caption", "vqa"],
        help="Energon sample type (default: crude)"
    )
    parser.add_argument(
        "--field-map",
        type=str,
        default=None,
        help="Field mapping for sample types that require it (e.g., 'json' for TextSample)"
    )
    add_image_args(parser)

    args = parser.parse_args()

    convert_json_to_wds(
        input_json=args.input_json,
        output_dir=args.output_dir,
        output_name=args.output_name,
        task_type=args.task_type,
        sample_type=args.sample_type,
        field_map=args.field_map,
        image_root=args.image_root,
        keep_relative=args.keep_relative_paths,
    )


if __name__ == "__main__":
    main()
