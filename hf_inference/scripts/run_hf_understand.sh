#!/bin/bash
# Image understanding (images + prompt -> text) with an exported HF Pixel-DiT model.
#
#   IMAGE=cat.jpg PROMPT="Describe the image." bash hf_inference/scripts/run_hf_understand.sh
#   DATA_PATH=vqa.jsonl IMAGE_ROOT=images/ OUTPUT=answers.jsonl NPROC=8 bash hf_inference/scripts/run_hf_understand.sh
#
# Extra CLI flags are forwarded, e.g. --temperature 0.7 --top-p 0.9.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-./checkpoints/pixel_dit_hf}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29521}"
IMAGE="${IMAGE:-}"            # single-sample mode, comma-separated for several images
PROMPT="${PROMPT:-}"
DATA_PATH="${DATA_PATH:-}"    # batch mode: jsonl {id, prompt|question, image|images}
IMAGE_ROOT="${IMAGE_ROOT:-}"
OUTPUT="${OUTPUT:-}"          # empty prints to stdout

ARGS=(--model "${MODEL}" --think-mode "${THINK_MODE:-think}"
      --max-think-tokens "${MAX_THINK_TOKENS:-2048}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}"
      --min-pixels "${MIN_PIXELS:-262144}" --max-pixels "${MAX_PIXELS:-4194304}")
if [ -n "${DATA_PATH}" ]; then
    ARGS+=(--data "${DATA_PATH}")
    [ -n "${IMAGE_ROOT}" ] && ARGS+=(--image-root "${IMAGE_ROOT}")
else
    [ -n "${IMAGE}" ] && [ -n "${PROMPT}" ] || { echo "[ERROR] set IMAGE and PROMPT, or DATA_PATH"; exit 1; }
    IFS=',' read -r -a IMAGES <<< "${IMAGE}"
    for img in "${IMAGES[@]}"; do ARGS+=(--image "${img}"); done
    ARGS+=(--prompt "${PROMPT}")
fi
[ -n "${OUTPUT}" ] && ARGS+=(--output "${OUTPUT}")

${PYTHON_BIN:-python} -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    hf_inference/understand.py "${ARGS[@]}" "$@"
