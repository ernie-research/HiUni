#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from data.transfusion.transfusion_data_cooker import open_image_checked

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SOURCE_REPEAT_RE = re.compile(r"^(.+)_([0-9]+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel OneIG stitching helper."
    )
    parser.add_argument("--input_root", required=True, type=str)
    parser.add_argument("--output_root", default="images", type=str)
    parser.add_argument("--resolution", default=1024, type=int)
    parser.add_argument("--grid_size", default=2, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--metadata_path", default="", type=str)
    parser.add_argument("--status_path", default="", type=str)
    parser.add_argument("--keep_source", action="store_true")
    return parser.parse_args()


def scan_all_tasks(input_root):
    tasks = []
    for root, _, files in os.walk(input_root):
        current_folder_images = [
            os.path.join(root, name)
            for name in files
            if name.lower().endswith(VALID_EXTS)
        ]
        if not current_folder_images:
            continue

        groups = defaultdict(list)
        for file_path in current_folder_images:
            basename = os.path.basename(file_path)
            stem = os.path.splitext(basename)[0]
            match = SOURCE_REPEAT_RE.match(stem)
            if not match:
                continue
            img_id = match.group(1)
            groups[img_id].append(file_path)

        rel_path = os.path.relpath(root, input_root)
        for img_id, file_list in groups.items():
            tasks.append(
                {
                    "id": img_id,
                    "files": sorted(file_list),
                    "rel_path": rel_path,
                }
            )

    return tasks


def get_task_target_dir(task, output_root_abs):
    rel_path = task["rel_path"]
    if rel_path == ".":
        return output_root_abs
    return os.path.join(output_root_abs, rel_path)


def cleanup_source_images_for_finished_tasks(input_root_abs, output_root_abs):
    """兜底删除已成功拼接任务残留的 *_0/*_1 源散图。"""
    deleted_count = 0
    for task in scan_all_tasks(input_root_abs):
        target_dir = get_task_target_dir(task, output_root_abs)
        stitched_path = os.path.join(target_dir, f"{task['id']}.png")
        if not os.path.isfile(stitched_path):
            continue
        for file_path in task["files"]:
            try:
                os.remove(file_path)
                deleted_count += 1
            except FileNotFoundError:
                pass
    return deleted_count


def create_image_gallery(images, rows=2, cols=2):
    if not images:
        return None

    img_width, img_height = images[0].size
    gallery = Image.new("RGB", (cols * img_width, rows * img_height))

    for idx, img in enumerate(images[: rows * cols]):
        row = idx // cols
        col = idx % cols
        gallery.paste(img, (col * img_width, row * img_height))

    return gallery


def process_single_task(task, output_root_abs, resolution, grid_size, keep_source):
    total_slots = grid_size * grid_size
    img_id = task["id"]
    file_list = task["files"]
    target_dir = get_task_target_dir(task, output_root_abs)

    images = []
    for file_path in file_list:
        try:
            with open_image_checked(file_path) as img:
                images.append(img.convert("RGB"))
        except Exception:
            continue

    if not images:
        return False

    try:
        if len(images) != total_slots:
            print(
                f"[错误] {img_id}: expected {total_slots} repeat images, "
                f"found {len(images)}; refusing to create a partial grid"
            )
            return False
        if len(images) > total_slots:
            for img in images[total_slots:]:
                img.close()
            images = images[:total_slots]

        gallery = create_image_gallery(images, grid_size, grid_size)
        if gallery is None:
            return False

        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, f"{img_id}.png")
        gallery.save(output_path)

        if not keep_source:
            for file_path in file_list:
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    pass

        return True
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        try:
            gallery.close()  # type: ignore[name-defined]
        except Exception:
            pass


def rewrite_metadata(meta_path):
    if not meta_path.is_file():
        return

    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with meta_path.open("r", encoding="utf-8") as src, tmp_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            entry = json.loads(line)
            images = entry.get("images") or []
            if images:
                first = images[0]
                stem, ext = os.path.splitext(first)
                entry["images"] = [re.sub(r"_\d+$", "", stem) + ext]
            dst.write(json.dumps(entry, ensure_ascii=False) + "\n")

    os.replace(tmp_path, meta_path)


def write_status(status_path, ok, message=""):
    if not status_path:
        return
    status = {"ok": bool(ok), "message": message}
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def run(args):
    input_root_abs = os.path.abspath(args.input_root)
    output_root_abs = os.path.abspath(args.output_root)
    metadata_path = (
        Path(args.metadata_path)
        if args.metadata_path
        else Path(output_root_abs) / "metadata.jsonl"
    )

    workers = args.workers
    if workers <= 0:
        workers = min(8, os.cpu_count() or 1)
    workers = max(1, workers)

    print(f"正在扫描目录: {input_root_abs} ...")
    all_tasks = scan_all_tasks(input_root_abs)
    total_tasks = len(all_tasks)
    print(f"扫描完成，共找到 {total_tasks} 组图片需要拼接。")
    if not args.keep_source:
        print("注意：拼接成功后将【自动删除】原始散图。")
    else:
        print("注意：已启用 --keep_source，将【保留】原始散图。")
    print(f"并行 worker 数: {workers}")
    print("-" * 30)

    if total_tasks == 0:
        print("未找到符合规则 (ID_序号) 的图片组，程序退出。")
        rewrite_metadata(metadata_path)
        return

    success_count = 0
    failure_count = 0

    if workers == 1 or total_tasks == 1:
        iterator = all_tasks
        if tqdm is not None:
            iterator = tqdm(all_tasks, total=total_tasks, desc="处理进度", unit="组", ncols=100)
        for task in iterator:
            ok = process_single_task(
                task,
                output_root_abs,
                args.resolution,
                args.grid_size,
                args.keep_source,
            )
            if ok:
                success_count += 1
            else:
                failure_count += 1
    else:
        with cf.ProcessPoolExecutor(max_workers=min(workers, total_tasks)) as pool:
            futures = [
                pool.submit(
                    process_single_task,
                    task,
                    output_root_abs,
                    args.resolution,
                    args.grid_size,
                    args.keep_source,
                )
                for task in all_tasks
            ]
            iterator = cf.as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="处理进度", unit="组", ncols=100)
            for future in iterator:
                try:
                    ok = future.result()
                except Exception as exc:  # pragma: no cover
                    print(f"\n[错误] 处理任务失败: {exc}")
                    ok = False
                if ok:
                    success_count += 1
                else:
                    failure_count += 1

    rewrite_metadata(metadata_path)
    if not args.keep_source:
        cleanup_count = cleanup_source_images_for_finished_tasks(
            input_root_abs,
            output_root_abs,
        )
        if cleanup_count > 0:
            print(f"兜底清理残留源散图: {cleanup_count} 张")

    print("-" * 30)
    print(
        f"所有任务完成！成功 {success_count} 组, 失败 {failure_count} 组, 结果已保存至: {output_root_abs}"
    )
    if failure_count > 0:
        raise RuntimeError(f"{failure_count} OneIG stitch task(s) failed")


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        write_status(args.status_path, False, repr(exc))
        traceback.print_exc()
        return 1

    write_status(args.status_path, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
