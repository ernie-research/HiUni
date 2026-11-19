#!/bin/bash
# Batch GEdit-Bench inference for Qwen3-VL Pixel-DiT edit checkpoints.

set -e

export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${WORKSPACE_DIR}"

export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"
export PYTHONPATH=./3rdparty/Megatron-Bridge/src:./3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

HF_PATH="${HF_PATH:-./checkpoints/Qwen3-VL-4B-Instruct}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23514}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

DATA_PATH="${DATA_PATH:-GEdit-Bench/stepfun-ai/GEdit-Bench}"
OUT_ROOT="${OUT_ROOT:-eval_results_i2i_pixel_dit}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
REPEAT_TIMES="${REPEAT_TIMES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-1}"
SAVE_REF="${SAVE_REF:-0}"
USE_RAW_IMAGE="${USE_RAW_IMAGE:-0}"

STEPS="${STEPS:-25}"
SHIFT="${SHIFT:-1.0}"
SCHEDULER="${SCHEDULER:-euler}"
CFG_MODE="${CFG_MODE:-text}"       # none | text | edit3
CFG_SCALE="${CFG_SCALE:-4.0}"      # text 模式; 等价于 edit3: 4*(none-text)+1*(text-full)+full
CFG_W_TEXT="${CFG_W_TEXT:-4.0}"    # edit3 模式使用
CFG_W_IMG="${CFG_W_IMG:-1.0}"      # edit3 模式使用; =1 时推理代码自动省掉 full 分支
CFG_KEEP_CHAT_TEMPLATE="${CFG_KEEP_CHAT_TEMPLATE:-0}"  # edit full: 0=image-only, 1=保留空 chat template
SEED="${SEED:-42}"

MATCH_REF_RESOLUTION="${MATCH_REF_RESOLUTION:-1}"
RESOLUTION="${RESOLUTION:-512}"
REF_MIN_PIXELS="${REF_MIN_PIXELS:-262144}"
REF_MAX_PIXELS="${REF_MAX_PIXELS:-1048576}"

NUM_DIFFUSION_LAYERS="${NUM_DIFFUSION_LAYERS:-10}"
PIXEL_DIT_GATE_INJECT_LAYER="${PIXEL_DIT_GATE_INJECT_LAYER:-3}"
PIXEL_DIT_REF_GATE_MODE="${PIXEL_DIT_REF_GATE_MODE:-none}"
DIT_HEAD_IMPL="${DIT_HEAD_IMPL:-torch}"

CKPT_LIST=()

if [ -n "${CKPT_LIST_FILE:-}" ]; then
    mapfile -t CKPT_LIST < <(grep -v '^[[:space:]]*#' "${CKPT_LIST_FILE}" | sed '/^[[:space:]]*$/d')
fi
if [ -n "${CKPT_PATH_OVERRIDE:-}" ]; then
    if [ -z "${MODEL_NAME_OVERRIDE:-}" ]; then
        echo "[ERROR] MODEL_NAME_OVERRIDE is required with CKPT_PATH_OVERRIDE" >&2
        exit 2
    fi
    CKPT_LIST=("${CKPT_PATH_OVERRIDE} ${MODEL_NAME_OVERRIDE}")
fi
if [ ${#CKPT_LIST[@]} -eq 0 ]; then
    echo "[ERROR] No checkpoints configured. Set CKPT_LIST_FILE or CKPT_PATH_OVERRIDE + MODEL_NAME_OVERRIDE." >&2
    exit 2
fi

resolve_ckpt_args() {
    local ckpt_path="$1"
    local ckpt_arg="${ckpt_path}"
    local iter_arg=""
    local base_name

    base_name=$(basename "${ckpt_path}")
    if [[ "${base_name}" =~ ^iter_([0-9]+)$ ]]; then
        ckpt_arg=$(dirname "${ckpt_path}")
        iter_arg="--iteration $((10#${BASH_REMATCH[1]}))"
    fi

    echo "${ckpt_arg}|${iter_arg}"
}

build_extra_args() {
    local extra=""

    if [ "${RESUME}" = "1" ]; then
        extra="${extra} --resume"
    fi
    if [ "${SAVE_REF}" = "1" ]; then
        extra="${extra} --save_ref"
    fi
    if [ "${USE_RAW_IMAGE}" = "1" ]; then
        extra="${extra} --use_raw_image"
    fi
    if [ -n "${HEIGHT:-}" ] || [ -n "${WIDTH:-}" ]; then
        extra="${extra} --height ${HEIGHT} --width ${WIDTH}"
    fi
    case "${MATCH_REF_RESOLUTION}" in
        1|true|TRUE|True)
            extra="${extra} --match_ref_resolution"
            ;;
        0|false|FALSE|False)
            extra="${extra} --no-match_ref_resolution --resolution ${RESOLUTION}"
            ;;
        *)
            echo "[ERROR] MATCH_REF_RESOLUTION must be 0/1/true/false, got ${MATCH_REF_RESOLUTION}" >&2
            return 1
            ;;
    esac
    case "${CFG_KEEP_CHAT_TEMPLATE}" in
        1|true|TRUE|True)
            extra="${extra} --cfg_keep_chat_template"
            ;;
        0|false|FALSE|False)
            extra="${extra} --no-cfg_keep_chat_template"
            ;;
        *)
            echo "[ERROR] CFG_KEEP_CHAT_TEMPLATE must be 0/1/true/false, got ${CFG_KEEP_CHAT_TEMPLATE}" >&2
            return 1
            ;;
    esac

    echo "${extra}"
}

run_ckpt() {
    local ckpt_path="$1"
    local model_name="$2"
    local resolved
    local ckpt_arg
    local iter_arg
    local extra_args
    local out_dir

    resolved=$(resolve_ckpt_args "${ckpt_path}")
    ckpt_arg="${resolved%%|*}"
    iter_arg="${resolved#*|}"
    extra_args=$(build_extra_args)
    out_dir="${OUT_ROOT}/${model_name}"

    echo ""
    echo "========== Pixel-DiT I2I Eval =========="
    echo "CKPT=${ckpt_path}"
    echo "MODEL=${model_name}"
    echo "DATA=${DATA_PATH}"
    echo "OUT=${out_dir}"
    echo "MATCH_REF_RESOLUTION=${MATCH_REF_RESOLUTION} RESOLUTION=${RESOLUTION} REF_PIXELS=${REF_MIN_PIXELS}-${REF_MAX_PIXELS}"
    echo "steps=${STEPS} shift=${SHIFT} scheduler=${SCHEDULER} cfg=${CFG_MODE} cfg_keep_chat_template=${CFG_KEEP_CHAT_TEMPLATE} batch_size=${BATCH_SIZE}"
    echo ""

    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        recipe/transfusion/eval/inference_transfusion_qwen3vl_pixel_dit_i2i.py \
        --model_mode pixel_dit \
        --ckpt "${ckpt_arg}" \
        --hf_path "${HF_PATH}" \
        --out_dir "${out_dir}" \
        --data_path "${DATA_PATH}" \
        --num_samples "${NUM_SAMPLES}" \
        --repeat_times "${REPEAT_TIMES}" \
        --batch_size "${BATCH_SIZE}" \
        --steps "${STEPS}" \
        --shift "${SHIFT}" \
        --scheduler "${SCHEDULER}" \
        --cfg_mode "${CFG_MODE}" \
        --cfg_scale "${CFG_SCALE}" \
        --cfg_w_text "${CFG_W_TEXT}" \
        --cfg_w_img "${CFG_W_IMG}" \
        --seed "${SEED}" \
        --ref_min_pixels "${REF_MIN_PIXELS}" \
        --ref_max_pixels "${REF_MAX_PIXELS}" \
        --num_diffusion_layers "${NUM_DIFFUSION_LAYERS}" \
        --pixel_dit_gate_inject_layer "${PIXEL_DIT_GATE_INJECT_LAYER}" \
        --pixel_dit_ref_gate_mode "${PIXEL_DIT_REF_GATE_MODE}" \
        --dit_head_impl "${DIT_HEAD_IMPL}" \
        --model_name "${model_name}" \
        --save_images \
        ${iter_arg} \
        ${extra_args}
}

FAIL_COUNT=0
for ckpt_entry in "${CKPT_LIST[@]}"; do
    CKPT_PATH=$(echo "${ckpt_entry}" | awk '{print $1}')
    MODEL_NAME=$(echo "${ckpt_entry}" | awk '{print $2}')
    if [ -z "${CKPT_PATH}" ] || [ -z "${MODEL_NAME}" ]; then
        echo "[WARN] skipping invalid entry: ${ckpt_entry}"
        continue
    fi
    if [ ! -d "${CKPT_PATH}" ]; then
        echo "[WARN] checkpoint not found, skipping: ${CKPT_PATH}"
        continue
    fi
    if ! run_ckpt "${CKPT_PATH}" "${MODEL_NAME}"; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "========== Pixel-DiT I2I Batch Eval Complete =========="
if [ "${FAIL_COUNT}" -gt 0 ]; then
    echo "[WARNING] ${FAIL_COUNT} task(s) failed."
    exit 1
fi
echo "All tasks succeeded."
