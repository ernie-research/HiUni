# -*- coding: utf-8 -*-
"""Image reference handling shared by the JSONL -> Energon converters.

An image entry may name its file under any of ``IMAGE_KEYS``. The value can be
a local path (absolute, relative, ``~`` or ``file://``), or an HTTP(S) URL.
Remote URLs are stored unchanged and fetched at training time.

Local paths are resolved at conversion time against ``image_root`` (default: the
directory of the input JSONL) and written as absolute paths, so training does not
depend on the working directory. With ``keep_relative=True`` relative paths are
written as-is and resolved at training time against ``TRANSFUSION_IMAGE_ROOT``.
"""

import os
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlsplit

IMAGE_KEYS = ("image", "image_path", "path", "url")


class MissingImageError(ValueError):
    """A local image referenced by a sample does not exist."""


def is_remote_url(src: str) -> bool:
    return urlsplit(src).scheme.lower() in ("http", "https")


def get_image_source(item: Dict[str, Any]) -> Optional[str]:
    """Return the first non-empty image reference of a record, or None."""
    for key in IMAGE_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_image_source(
    src: str,
    image_root: Optional[str] = None,
    keep_relative: bool = False,
    check_exists: bool = True,
) -> str:
    """Return the reference to store in ``jpgs``.

    Raises MissingImageError when a resolved local file does not exist.
    """
    if is_remote_url(src):
        return src
    if src.startswith("file://"):
        src = unquote(urlsplit(src).path)
    src = os.path.expanduser(src)
    if not os.path.isabs(src):
        if keep_relative:
            return src
        src = os.path.join(os.path.expanduser(image_root or os.getcwd()), src)
    src = os.path.abspath(src)
    if check_exists and not os.path.isfile(src):
        raise MissingImageError(f"image file not found: {src}")
    return src


def default_image_root(input_path: str, image_root: Optional[str]) -> str:
    """``--image-root`` if given, otherwise the directory holding the input file."""
    if image_root:
        return image_root
    return os.path.dirname(os.path.abspath(input_path))


def add_image_args(parser) -> None:
    """Register the shared ``--image-root`` / ``--keep-relative-paths`` options."""
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Directory that relative local image paths are resolved against "
        "(default: the directory of the input file)",
    )
    parser.add_argument(
        "--keep-relative-paths",
        action="store_true",
        help="Store relative local paths unchanged; training then resolves them "
        "against TRANSFUSION_IMAGE_ROOT",
    )
