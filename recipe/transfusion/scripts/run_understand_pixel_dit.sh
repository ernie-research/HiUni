#!/bin/bash
# Image understanding (images + prompt -> text) with the dense Pixel-DiT Megatron model.
#
#   IMAGE=cat.jpg PROMPT="Describe the image." bash run_understand_pixel_dit.sh
#   DATA_PATH=vqa.jsonl OUTPUT=answers.jsonl NPROC=8 bash run_understand_pixel_dit.sh
#
# Structural hyperparameters are read from the checkpoint's run_config.yaml.
# Extra CLI flags are forwarded, e.g. --temperature 0.7 --top_p 0.9.

set -euo pipefail

export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${WORKSPACE_DIR}"
export PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/3rdparty/Megatron-Bridge/src:${WORKSPACE_DIR}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

HF_PATH="${HF_PATH:-./checkpoints/Qwen3-VL-4B-Instruct}"
CKPT="${CKPT:-./checkpoints/pixel_dit_megatron}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"

IMAGE="${IMAGE:-}"            # single-sample mode, comma-separated for several images
PROMPT="${PROMPT:-}"
DATA_PATH="${DATA_PATH:-}"    # batch mode: jsonl {id, prompt, image|images}
IMAGE_ROOT="${IMAGE_ROOT:-}"
OUTPUT="${OUTPUT:-}"          # empty prints to stdout

THINK_MODE="${THINK_MODE:-think}"          # think | empty | none
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MIN_PIXELS="${MIN_PIXELS:-262144}"
MAX_PIXELS="${MAX_PIXELS:-4194304}"

ARGS=(--ckpt "${CKPT}" --hf_path "${HF_PATH}"
      --think_mode "${THINK_MODE}" --max_think_tokens "${MAX_THINK_TOKENS}"
      --max_new_tokens "${MAX_NEW_TOKENS}"
      --min_pixels "${MIN_PIXELS}" --max_pixels "${MAX_PIXELS}")
if [ -n "${DATA_PATH}" ]; then
    ARGS+=(--data_path "${DATA_PATH}")
    [ -n "${IMAGE_ROOT}" ] && ARGS+=(--image_root "${IMAGE_ROOT}")
else
    [ -n "${IMAGE}" ] && [ -n "${PROMPT}" ] || { echo "[ERROR] set IMAGE and PROMPT, or DATA_PATH"; exit 1; }
    IFS=',' read -r -a IMAGES <<< "${IMAGE}"
    for img in "${IMAGES[@]}"; do ARGS+=(--image "${img}"); done
    ARGS+=(--prompt "${PROMPT}")
fi
[ -n "${OUTPUT}" ] && ARGS+=(--output "${OUTPUT}")
[ -n "${ITERATION:-}" ] && ARGS+=(--iteration "${ITERATION}")

${PYTHON_BIN:-python} -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    recipe/transfusion/eval/inference_transfusion_qwen3vl_pixel_dit_understand.py "${ARGS[@]}" "$@"
