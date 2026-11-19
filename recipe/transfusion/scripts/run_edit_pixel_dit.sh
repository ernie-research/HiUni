#!/bin/bash
# Image editing (reference image + instruction -> image) with the dense Pixel-DiT Megatron model.
#
#   IMAGE=cat.jpg PROMPT="Turn the cat purple." bash run_edit_pixel_dit.sh
#   DATA_PATH=GEdit-Bench/stepfun-ai/GEdit-Bench NPROC=8 bash run_edit_pixel_dit.sh
#
# Batch mode reads a HF `datasets.load_from_disk` directory in the GEdit-Bench layout
# (key, instruction, input_image). The output matches the reference resolution by default.
# The defaults are the setting verified against hf_inference/scripts/run_hf_edit.sh.
# Structural hyperparameters are read from the checkpoint's run_config.yaml.
# Extra CLI flags are forwarded, e.g. --repeat_times 2 --resume.

set -euo pipefail

export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${WORKSPACE_DIR}"
export PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/3rdparty/Megatron-Bridge/src:${WORKSPACE_DIR}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

HF_PATH="${HF_PATH:-./checkpoints/Qwen3-VL-4B-Instruct}"
CKPT="${CKPT:-./checkpoints/pixel_dit_megatron}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29514}"

IMAGE="${IMAGE:-}"            # single-sample mode
PROMPT="${PROMPT:-}"
DATA_PATH="${DATA_PATH:-}"    # batch mode: load_from_disk directory
OUT_DIR="${OUT_DIR:-./output/edit_pixel_dit}"

STEPS="${STEPS:-25}"
SHIFT="${SHIFT:-4.0}"
SCHEDULER="${SCHEDULER:-euler}"    # euler | dpm2 | dpm3
CFG_MODE="${CFG_MODE:-edit3}"      # none | text | edit3
CFG_SCALE="${CFG_SCALE:-2.0}"      # text mode
CFG_W_TEXT="${CFG_W_TEXT:-4.0}"    # edit3 mode
CFG_W_IMG="${CFG_W_IMG:-1.0}"      # edit3 mode; 1.0 drops the image-only branch
SEED="${SEED:-42}"
REF_MIN_PIXELS="${REF_MIN_PIXELS:-262144}"   # 512x512-area reference, same as hf_inference/edit.py
REF_MAX_PIXELS="${REF_MAX_PIXELS:-262144}"

ARGS=(--ckpt "${CKPT}" --hf_path "${HF_PATH}" --out_dir "${OUT_DIR}"
      --steps "${STEPS}" --shift "${SHIFT}" --scheduler "${SCHEDULER}"
      --cfg_mode "${CFG_MODE}" --cfg_scale "${CFG_SCALE}"
      --cfg_w_text "${CFG_W_TEXT}" --cfg_w_img "${CFG_W_IMG}"
      --seed "${SEED}" --ref_min_pixels "${REF_MIN_PIXELS}" --ref_max_pixels "${REF_MAX_PIXELS}"
      --save_images)
if [ -n "${DATA_PATH}" ]; then
    ARGS+=(--data_path "${DATA_PATH}")
else
    [ -n "${IMAGE}" ] && [ -n "${PROMPT}" ] || { echo "[ERROR] set IMAGE and PROMPT, or DATA_PATH"; exit 1; }
    ARGS+=(--ref_image "${IMAGE}" --prompt "${PROMPT}")
fi
[ -n "${ITERATION:-}" ] && ARGS+=(--iteration "${ITERATION}")

${PYTHON_BIN:-python} -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
    recipe/transfusion/eval/inference_transfusion_qwen3vl_pixel_dit_i2i.py "${ARGS[@]}" "$@"
