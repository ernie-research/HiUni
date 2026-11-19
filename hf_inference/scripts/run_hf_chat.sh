#!/bin/bash
# Pure-text chat (prompt -> text) with an exported HF Pixel-DiT model.
#
#   PROMPT="Explain flow matching briefly." bash hf_inference/scripts/run_hf_chat.sh
#   DATA_PATH=prompts.jsonl OUTPUT=answers.jsonl NPROC=8 bash hf_inference/scripts/run_hf_chat.sh
#
# Extra CLI flags are forwarded, e.g. --system "You are a helpful assistant."

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-./checkpoints/pixel_dit_hf}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29522}"
PROMPT="${PROMPT:-}"
DATA_PATH="${DATA_PATH:-}"    # batch mode: jsonl {id, prompt|question, system?}
OUTPUT="${OUTPUT:-}"          # empty prints to stdout

ARGS=(--model "${MODEL}" --think-mode "${THINK_MODE:-think}"
      --max-think-tokens "${MAX_THINK_TOKENS:-2048}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}")
if [ -n "${DATA_PATH}" ]; then
    ARGS+=(--data "${DATA_PATH}")
else
    [ -n "${PROMPT}" ] || { echo "[ERROR] set PROMPT or DATA_PATH"; exit 1; }
    ARGS+=(--prompt "${PROMPT}")
fi
[ -n "${OUTPUT}" ] && ARGS+=(--output "${OUTPUT}")

${PYTHON_BIN:-python} -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    hf_inference/chat.py "${ARGS[@]}" "$@"
