#!/bin/bash
# Pure-text chat (prompt -> text) with the dense Pixel-DiT Megatron model.
#
#   PROMPT="Explain flow matching briefly." bash run_text_pixel_dit.sh
#   DATA_PATH=prompts.jsonl OUTPUT=answers.jsonl NPROC=8 bash run_text_pixel_dit.sh
#
# Extra CLI flags are forwarded, e.g. --system "You are a helpful assistant."

set -euo pipefail

export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${WORKSPACE_DIR}"
export PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/3rdparty/Megatron-Bridge/src:${WORKSPACE_DIR}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

HF_PATH="${HF_PATH:-./checkpoints/Qwen3-VL-4B-Instruct}"
CKPT="${CKPT:-./checkpoints/pixel_dit_megatron}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29512}"

PROMPT="${PROMPT:-}"
DATA_PATH="${DATA_PATH:-}"    # batch mode: jsonl {id, prompt, system?}
OUTPUT="${OUTPUT:-}"          # empty prints to stdout

THINK_MODE="${THINK_MODE:-think}"          # think | empty | none
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

ARGS=(--ckpt "${CKPT}" --hf_path "${HF_PATH}"
      --think_mode "${THINK_MODE}" --max_think_tokens "${MAX_THINK_TOKENS}"
      --max_new_tokens "${MAX_NEW_TOKENS}")
if [ -n "${DATA_PATH}" ]; then
    ARGS+=(--data_path "${DATA_PATH}")
else
    [ -n "${PROMPT}" ] || { echo "[ERROR] set PROMPT or DATA_PATH"; exit 1; }
    ARGS+=(--prompt "${PROMPT}")
fi
[ -n "${OUTPUT}" ] && ARGS+=(--output "${OUTPUT}")
[ -n "${ITERATION:-}" ] && ARGS+=(--iteration "${ITERATION}")

${PYTHON_BIN:-python} -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    recipe/transfusion/eval/inference_transfusion_qwen3vl_pixel_dit_text.py "${ARGS[@]}" "$@"
