#!/usr/bin/env bash
set -euo pipefail

# Small local wrapper around the self-contained JSONL converter.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INPUT_FILE=""
OUTPUT_DIR="${ROOT_DIR}/output/local_test_t2i"
OUTPUT_NAME="data"
IMAGE_ARGS=()
USE_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_FILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --use-tag) USE_TAG="--use-tag"; shift ;;
    --image-root) IMAGE_ARGS+=(--image-root "$(cd "$2" && pwd)"); shift 2 ;;
    --keep-relative-paths) IMAGE_ARGS+=(--keep-relative-paths); shift ;;
    -h|--help)
      echo "Usage: $0 --input INPUT.jsonl [--output-dir DIR] [--output-name NAME] [--use-tag] [--image-root DIR] [--keep-relative-paths]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${INPUT_FILE}" ]] || { echo "--input is required" >&2; exit 2; }
[[ -f "${INPUT_FILE}" ]] || { echo "Input not found: ${INPUT_FILE}" >&2; exit 1; }
INPUT_FILE="$(cd "$(dirname "${INPUT_FILE}")" && pwd)/$(basename "${INPUT_FILE}")"
mkdir -p "${OUTPUT_DIR}" && OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
cd "${ROOT_DIR}"
"${PYTHON_BIN:-python}" recipe/transfusion/data/convert_t2i.py \
  --input-jsonl "${INPUT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-name "${OUTPUT_NAME}" \
  --task-type generation \
  --sample-type crude \
  "${IMAGE_ARGS[@]}" \
  ${USE_TAG}
