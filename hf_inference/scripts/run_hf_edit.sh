#!/bin/bash
# Instruction-based image editing (reference image + instruction -> image) with an HF Pixel-DiT export.
#
#   IMAGE=input.jpg PROMPT="Make the sky purple." bash hf_inference/scripts/run_hf_edit.sh
#   DATA_PATH=edits.jsonl IMAGE_ROOT=images/ OUTPUT=edits/ NPROC=8 bash hf_inference/scripts/run_hf_edit.sh --resume
#
# Defaults (512x512-area reference, 25 steps, shift 4, edit3 w_text=4 w_img=1) match the
# setting verified against the Megatron edit entrypoint. Extra CLI flags are
# forwarded, e.g. --scheduler dpm2 --w-img 1.5 --no-resize-to-ref.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-./checkpoints/pixel_dit_hf}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29524}"
IMAGE="${IMAGE:-}"
PROMPT="${PROMPT:-}"
DATA_PATH="${DATA_PATH:-}"    # batch mode: jsonl {id, image, prompt|instruction}
IMAGE_ROOT="${IMAGE_ROOT:-}"
OUTPUT="${OUTPUT:-./outputs/hf_edit}"

ARGS=(--model "${MODEL}" --output "${OUTPUT}" --steps "${STEPS:-25}" --shift "${SHIFT:-4.0}"
      --cfg-mode "${CFG_MODE:-edit3}" --w-text "${W_TEXT:-4.0}" --w-img "${W_IMG:-1.0}"
      --seed "${SEED:-42}" --min-pixels "${MIN_PIXELS:-262144}" --max-pixels "${MAX_PIXELS:-262144}")
if [ -n "${DATA_PATH}" ]; then
    ARGS+=(--data "${DATA_PATH}")
    [ -n "${IMAGE_ROOT}" ] && ARGS+=(--image-root "${IMAGE_ROOT}")
else
    [ -n "${IMAGE}" ] && [ -n "${PROMPT}" ] || { echo "[ERROR] set IMAGE and PROMPT, or DATA_PATH"; exit 1; }
    ARGS+=(--image "${IMAGE}" --prompt "${PROMPT}")
fi

${PYTHON_BIN:-python} -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    hf_inference/edit.py "${ARGS[@]}" "$@"
