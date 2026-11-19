#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert JSONL to Megatron Energon WebDataset format (Text-to-Image task).

基于 datafactory/mapper/mapper.py 和 datafactory/reducer/reducer.py 的模式
实现单条 JSONL 转换为 Energon 格式 (tar + tar.idx + .nv-meta)

Input JSONL format (T2I task):
{
    "data_id": "000001",
    "image": "images/000001.jpg",
    "caption": "This is a watercolor illustration...",
    "tag": "watercolor painting, illustration...",   # optional
    "status": "success"                              # optional
}

The image field may be named image / image_path / path / url and may be
a local path (absolute, relative, file://) or an http(s):// URL. Relative
local paths are resolved against --image-root (default: the JSONL's directory) and
stored as absolute paths; pass --keep-relative-paths to store them unchanged and
resolve them at training time against TRANSFUSION_IMAGE_ROOT. Records whose
"status" is present and not "success" are skipped.

默认顺序: text -> image
- text: caption + tag (可选)
- image: 本地路径或 HTTP URL，需要训练

Output WebDataset format:
- tar: WebDataset 格式，包含 __key__, task_type, jpgs, json 字段
- tar.idx: Energon 索引文件，记录每个样本的偏移量
- .nv-meta: Energon 元数据文件，描述数据集信息

Example Usage:
    python convert_t2i.py --input-jsonl /path/to/data.jsonl --output-dir /path/to/output
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
    MissingImageError,
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


def build_t2i_text_with_mask(
    caption: str,
    tag: Optional[str] = None,
    use_tag: bool = False
) -> Tuple[List[str], List[int]]:
    """
    构建 T2I 格式的 text 和 mask

    顺序: text -> image
    - text: caption (可选 + tag)
    - image: 需要训练

    Args:
        caption: 图像描述文本
        tag: 标签（可选）
        use_tag: 是否使用标签

    Returns:
        Tuple of (text_fragments, mask_list)
        text_fragments: [caption, "<image>"]
        mask_list: [1, 0]  # 0=需要训练, 1=不需要训练
    """
    text_fragments = []
    mask_list = []

    # 构建文本
    if use_tag and tag:
        # 使用 caption + tag
        combined_text = f"{caption} {tag}".strip()
    else:
        # 只使用 caption
        combined_text = caption.strip()

    if combined_text:
        text_fragments.append(combined_text)
        # 文本不参与训练 (mask=1)
        mask_list.append(1)

    # 添加图像，需要训练 (mask=0)
    text_fragments.append("<image>")
    mask_list.append(0)

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
    task_type: str = "generation",
    use_tag: bool = False,
    image_root: Optional[str] = None,
    keep_relative: bool = False,
) -> Dict[str, Any]:
    """
    转换单条 JSON 数据为 WebDataset 样本格式（适配 T2I 格式）

    Args:
        data: 原始 JSON 数据
        task_type: 任务类型
        use_tag: 是否使用标签
        image_root: 相对本地路径的解析根目录
        keep_relative: 保留相对路径，训练时按 TRANSFUSION_IMAGE_ROOT 解析

    Returns:
        WebDataset 样本字典
    """
    data_id = ensure_id(data)

    # 获取图像（本地路径或 HTTP URL）
    image_src = get_image_source(data)
    if not image_src:
        log.warning(f"样本 {data_id} 缺少图像字段，跳过")
        return None
    try:
        image_src = normalize_image_source(image_src, image_root, keep_relative)
    except MissingImageError as e:
        log.warning(f"样本 {data_id}: {e}，跳过")
        return None

    # 获取文本
    caption = data.get("caption", "")
    if not caption:
        log.warning(f"样本 {data_id} 缺少 caption，跳过")
        return None

    tag = data.get("tag", "")

    # 构建 T2I 格式的 text 和 mask
    text, mask = build_t2i_text_with_mask(caption, tag, use_tag)

    sample = {
        "__key__": data_id,
        "task_type": task_type,
        "json": {
            "text": text,
            "mask": [1 - x for x in mask],  # 反转 mask: 0->1 (train), 1->0 (skip)
        },
        "jpgs": [image_src],
    }

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
            if sample is None:
                continue

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

            # 写入 jpgs
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


def convert_jsonl_to_wds(
    input_jsonl: str,
    output_dir: str,
    output_name: str = "data",
    task_type: str = "generation",
    sample_type: str = "crude",
    field_map: Optional[str] = None,
    use_tag: bool = False,
    image_root: Optional[str] = None,
    keep_relative: bool = False,
) -> Tuple[str, str, str]:
    """
    将 JSONL 文件转换为 Energon WebDataset 格式

    Args:
        input_jsonl: 输入 JSONL 文件路径
        output_dir: 输出目录
        output_name: 输出文件名（不含扩展名）
        task_type: 任务类型（用于样本中的 task_type 字段）
        sample_type: Energon 样本类型 (crude/text/interleaved 等)
        field_map: 字段映射（某些 sample_type 需要）
        use_tag: 是否使用标签
        image_root: 相对本地路径的解析根目录（默认 JSONL 所在目录）
        keep_relative: 保留相对路径，训练时按 TRANSFUSION_IMAGE_ROOT 解析

    Returns:
        Tuple of (tar_path, idx_path, meta_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    image_root = default_image_root(input_jsonl, image_root)

    tar_path = os.path.join(output_dir, f"{output_name}.tar")

    # 读取 JSONL 数据
    log.info(f"读取 JSONL 文件: {input_jsonl}")
    samples = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = parse_jsonl_line(line)
            if data and data.get("status", "success") == "success":
                sample = convert_sample(data, task_type, use_tag, image_root, keep_relative)
                if sample:
                    samples.append(sample)

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
        description="Convert JSONL to Megatron Energon WebDataset format (Text-to-Image)"
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
        default="generation",
        choices=["understanding", "generation", "editing"],
        help="Task type field in sample (default: generation)"
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
    parser.add_argument(
        "--use-tag",
        action="store_true",
        help="Use tag in addition to caption (default: caption only)"
    )
    add_image_args(parser)

    args = parser.parse_args()

    convert_jsonl_to_wds(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        output_name=args.output_name,
        task_type=args.task_type,
        sample_type=args.sample_type,
        field_map=args.field_map,
        use_tag=args.use_tag,
        image_root=args.image_root,
        keep_relative=args.keep_relative_paths,
    )


if __name__ == "__main__":
    main()
