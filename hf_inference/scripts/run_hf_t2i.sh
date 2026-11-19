#!/bin/bash
# Text-to-image (prompt -> image) with an exported HF Pixel-DiT model.
#
#   PROMPT="A red fox in fresh snow." bash hf_inference/scripts/run_hf_t2i.sh
#   DATA_PATH=prompts.jsonl OUTPUT=t2i/ NPROC=8 bash hf_inference/scripts/run_hf_t2i.sh --resume
#
# Benchmark layouts (OneIG / LongText / QwenImageBench) use hf_inference/generate.py.
# Extra CLI flags are forwarded, e.g. --negative-prompt "blurry" --repeats 4.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-./checkpoints/pixel_dit_hf}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29523}"
PROMPT="${PROMPT:-}"
DATA_PATH="${DATA_PATH:-}"    # batch mode: jsonl {id, prompt|caption, height?, width?}
OUTPUT="${OUTPUT:-./outputs/hf_t2i}"

ARGS=(--model "${MODEL}" --output "${OUTPUT}"
      --height "${HEIGHT:-2048}" --width "${WIDTH:-2048}" --steps "${STEPS:-25}"
      --cfg "${CFG:-4.0}" --shift "${SHIFT:-4.0}" --seed "${SEED:-42}")
if [ -n "${DATA_PATH}" ]; then
    ARGS+=(--data "${DATA_PATH}")
else
    [ -n "${PROMPT}" ] || { echo "[ERROR] set PROMPT or DATA_PATH"; exit 1; }
    ARGS+=(--prompt "${PROMPT}")
fi

${PYTHON_BIN:-python} -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    hf_inference/t2i.py "${ARGS[@]}" "$@"
