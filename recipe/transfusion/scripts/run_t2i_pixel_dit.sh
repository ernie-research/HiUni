#!/bin/bash
# Text-to-image generation (prompt -> image) with the dense Pixel-DiT Megatron model.
#
#   PROMPT="A red fox in the snow." bash run_t2i_pixel_dit.sh
#   DATA_PATH=prompts.jsonl DATA_FORMAT=jsonl NPROC=8 bash run_t2i_pixel_dit.sh
#
# Images are written under OUT_DIR. RESOLUTION is the output side in pixels (multiple of 32);
# aspect ratios written in the prompt ("16:9") are honoured unless --no-respect_prompt_aspect_ratio.
# Structural hyperparameters are read from the checkpoint's run_config.yaml.
# Extra CLI flags are forwarded, e.g. --negative_prompt "blurry" --repeat_times 4.

set -euo pipefail

export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${WORKSPACE_DIR}"
export PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/3rdparty/Megatron-Bridge/src:${WORKSPACE_DIR}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

HF_PATH="${HF_PATH:-./checkpoints/Qwen3-VL-4B-Instruct}"
CKPT="${CKPT:-./checkpoints/pixel_dit_megatron}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29513}"

PROMPT="${PROMPT:-}"          # single prompt
DATA_PATH="${DATA_PATH:-}"    # batch mode: txt (one prompt per line) or jsonl {id?, prompt|text}
DATA_FORMAT="${DATA_FORMAT:-}"
OUT_DIR="${OUT_DIR:-./output/t2i_pixel_dit}"

RESOLUTION="${RESOLUTION:-2048}"
STEPS="${STEPS:-25}"
CFG_SCALE="${CFG_SCALE:-4.0}"
SHIFT="${SHIFT:-4.0}"
SCHEDULER="${SCHEDULER:-euler}"   # euler | dpm2 | dpm3
SEED="${SEED:-42}"

if (( RESOLUTION % 32 != 0 )); then
    echo "[ERROR] RESOLUTION must be a multiple of 32, got ${RESOLUTION}"; exit 1
fi
GRID=$(( RESOLUTION / 32 ))   # merged grid side: pixels = 2 * GRID * 16

ARGS=(--ckpt "${CKPT}" --hf_path "${HF_PATH}" --out_dir "${OUT_DIR}"
      --height "${GRID}" --width "${GRID}"
      --steps "${STEPS}" --cfg_scale "${CFG_SCALE}" --shift "${SHIFT}" --scheduler "${SCHEDULER}"
      --seed "${SEED}" --save_images)
if [ -n "${DATA_PATH}" ]; then
    if [ -z "${DATA_FORMAT}" ]; then
        case "${DATA_PATH}" in
            *.jsonl) DATA_FORMAT=jsonl ;;
            *.csv) DATA_FORMAT=csv ;;
            *) DATA_FORMAT=txt ;;
        esac
    fi
    ARGS+=(--data_format "${DATA_FORMAT}" --data_file "${DATA_PATH}")
else
    [ -n "${PROMPT}" ] || { echo "[ERROR] set PROMPT or DATA_PATH"; exit 1; }
    ARGS+=(--prompt "${PROMPT}")
fi
[ -n "${ITERATION:-}" ] && ARGS+=(--iteration "${ITERATION}")

${PYTHON_BIN:-python} -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    recipe/transfusion/eval/inference_transfusion_qwen3vl_pixel_dit.py "${ARGS[@]}" "$@"
