#!/bin/bash
# ===================================================================
# batch_eval_pixel_dit.sh
# 批量评估多个 Pixel-DiT checkpoint 的上层脚本
#
# 与 batch_eval_qwen3vl.sh 的区别:
#   - 驱动 inference_transfusion_qwen3vl_pixel_dit.py (不是 inference_transfusion_qwen3vl.py)
#   - pixel_dit 直接在 pixel 空间扩散 (latent = raw pixel patch)
#   - 用 --height/--width (merged grid patch 数) 替代 --resolution
#   - 不支持 AdaLN/HiGate/bypass_image_backbone/drop_dit_text (与 pixel_dit 代码路径无关)
#   - 新增 num_diffusion_layers/pixel_dit_gate_inject_layer/dit_head_impl 结构超参
#     (若 checkpoint 有 run_config.yaml, 脚本内部会自动读取覆盖这里的默认值)
#
# 用法:
#   bash recipe/transfusion/scripts/batch_eval_pixel_dit.sh
#   RESOLUTION=1024 bash recipe/transfusion/scripts/batch_eval_pixel_dit.sh
#   RESOLUTION=1024,2048 bash recipe/transfusion/scripts/batch_eval_pixel_dit.sh
#   RESOLUTION=256 CFG_SCALE=2.0 CFG_KEEP_CHAT_TEMPLATE=1 bash recipe/transfusion/scripts/batch_eval_pixel_dit.sh
#   BENCHMARK_LIST=QwenImageBench QWEN_IMAGE_BENCH_PROMPTS=/path/qwen_image_bench_hf_v0518.jsonl \
#     EVAL_BENCHMARKS=QwenImageBench bash recipe/transfusion/scripts/batch_eval_pixel_dit.sh
#   RESUME=1 bash recipe/transfusion/scripts/batch_eval_pixel_dit.sh
# ===================================================================

set -e

# ═══════════════════════════════════════════════════════════════
#  全局配置 (可通过环境变量覆盖)
# ═══════════════════════════════════════════════════════════════

export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${WORKSPACE_DIR}"
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"
export PYTHONPATH=./3rdparty/Megatron-Bridge/src:./3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:$PYTHONPATH

# Use the selected Python interpreter consistently for all distributed workers.
PYTHON_BIN="${PYTHON_BIN:-python}"
torchrun() { "${PYTHON_BIN}" -m torch.distributed.run "$@"; }

# Run the dense Pixel-DiT generation, post-processing, sync, and evaluation pipeline.
INFER_SCRIPT="${INFER_SCRIPT:-recipe/transfusion/eval/inference_transfusion_qwen3vl_pixel_dit.py}"

HF_PATH="${HF_PATH:-./checkpoints/Qwen3-VL-4B-Instruct}"

# 分布式
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# 推理参数 (可通过环境变量覆盖)
# Pixel-DiT: --resolution 用于计算 merged grid (image_token_merge_size=2 时, merged H=W=resolution/32).
# 默认 1024 (对应 merged 32x32，即无比例 prompt 的 1024x1024 px)；
# 有比例要求的 prompt 会在 Python 推理脚本中按面积保持该基准。
RESOLUTION_RAW="${RESOLUTION:-2048}"
IFS=',' read -r -a RESOLUTIONS <<< "${RESOLUTION_RAW}"
VALID_RESOLUTIONS=()
for resolution_item in "${RESOLUTIONS[@]}"; do
    # 允许 RESOLUTION=1024, 2048 这种带空格的写法。
    resolution_item="${resolution_item//[[:space:]]/}"
    if [[ ! "${resolution_item}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] RESOLUTION must contain integer values, got: ${resolution_item}"
        exit 1
    fi
    resolution_value=$((10#${resolution_item}))
    if (( resolution_value <= 0 || resolution_value % 32 != 0 )); then
        echo "[ERROR] each RESOLUTION must be positive and divisible by 32, got: ${resolution_value}"
        exit 1
    fi
    VALID_RESOLUTIONS+=("${resolution_value}")
done

if [ ${#VALID_RESOLUTIONS[@]} -eq 0 ]; then
    echo "[ERROR] RESOLUTION must contain at least one value"
    exit 1
fi

# HEIGHT/WIDTH 仍支持旧的单分辨率矩形网格用法。多分辨率模式下必须由
# RESOLUTION 自动计算，避免一个 override 同时错误地应用到多个分辨率。
if [ ${#VALID_RESOLUTIONS[@]} -eq 1 ]; then
    SINGLE_HEIGHT="${HEIGHT:-$((VALID_RESOLUTIONS[0] / 32))}"
    SINGLE_WIDTH="${WIDTH:-$((VALID_RESOLUTIONS[0] / 32))}"
else
    if [ -n "${HEIGHT:-}" ] || [ -n "${WIDTH:-}" ]; then
        echo "[ERROR] HEIGHT/WIDTH overrides are only supported with one RESOLUTION value"
        exit 1
    fi
fi
STEPS="${STEPS:-25}"
CFG_SCALE="${CFG_SCALE:-4.0}"
CFG_KEEP_CHAT_TEMPLATE="${CFG_KEEP_CHAT_TEMPLATE:-0}"  # 0: 默认纯 image-only; 1: 保留 chat template + 生成指令
SHIFT="${SHIFT:-4.0}"
RESPECT_PROMPT_ASPECT_RATIO="${RESPECT_PROMPT_ASPECT_RATIO:-1}"  # 1: 按 prompt 宽高比/基准面积生成
REPEAT_TIMES="${REPEAT_TIMES:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"   # 每个 DP rank 内一次 sample_images 的 prompt 数量
SCHEDULER="${SCHEDULER:-euler}"  # euler | dpm2 | dpm3
RESUME="${RESUME:-1}"            # 1: 跳过 OUT_DIR/images 下已经生成的图片
RUN_EVAL_AFTER_SYNC="${RUN_EVAL_AFTER_SYNC:-1}"
EVAL_BENCHMARKS_WAS_SET=0
if [ -n "${EVAL_BENCHMARKS+x}" ]; then
    EVAL_BENCHMARKS_WAS_SET=1
fi
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-QwenImageBench}"
HIUNI_EVAL_ROOT="${HIUNI_EVAL_ROOT:-/path/to/external}"
ONEIG_BENCH_ROOT="${ONEIG_BENCH_ROOT:-${HIUNI_EVAL_ROOT}/OneIG-Benchmark}"
ONEIG_ALIGNMENT_SCRIPT="${ONEIG_ALIGNMENT_SCRIPT:-${ONEIG_BENCH_ROOT}/vllm_alignment/run_alignment_vllm.sh}"
ONEIG_OTHER_DIMS_SCRIPT="${ONEIG_OTHER_DIMS_SCRIPT:-${ONEIG_BENCH_ROOT}/vllm_alignment/run_other_dims.sh}"
ONEIG_ALIGNMENT_GPUS="${ONEIG_ALIGNMENT_GPUS:-0,1,2,3,4,5,6}"
ONEIG_OTHER_DIMS_GPUS="${ONEIG_OTHER_DIMS_GPUS:-7}"
ONEIG_ALIGNMENT_CONCURRENCY="${ONEIG_ALIGNMENT_CONCURRENCY:-1024}"
ONEIG_ALIGNMENT_TILE_WORKERS="${ONEIG_ALIGNMENT_TILE_WORKERS:-$(nproc)}"
ONEIG_ALIGNMENT_API_SERVER_COUNT="${ONEIG_ALIGNMENT_API_SERVER_COUNT:-1}"
ONEIG_ALIGNMENT_MAX_NUM_SEQS="${ONEIG_ALIGNMENT_MAX_NUM_SEQS:-256}"
LONGTEXT_EVAL_SCRIPT="${LONGTEXT_EVAL_SCRIPT:-${HIUNI_EVAL_ROOT}/X-Omni-LongTextEval/textbench/eval.sh}"
LONGTEXT_GPUS="${LONGTEXT_GPUS:-0,1,2,3,4,5,6,7}"
LONGTEXT_MASTER_PORT="${LONGTEXT_MASTER_PORT:-29506}"
ONEIG_STITCH_WORKERS="${ONEIG_STITCH_WORKERS:-8}"
ONEIG_STITCH_SCRIPT="${ONEIG_STITCH_SCRIPT:-recipe/transfusion/scripts/oneig_postprocess_parallel.py}"

# Qwen-Image-Bench: generation consumes a local JSONL prompt file containing
# ``ID``/``prompt``; judging uses the local Q-Judger checkpoint.
QWEN_IMAGE_BENCH_ROOT="${QWEN_IMAGE_BENCH_ROOT:-./external/Qwen-Image-Bench}"
QWEN_IMAGE_BENCH_DATASET_ROOT="${QWEN_IMAGE_BENCH_DATASET_ROOT:-${QWEN_IMAGE_BENCH_ROOT}/Qwen/Qwen-Image-Bench-Dataset}"
QWEN_IMAGE_BENCH_PYTHON="${QWEN_IMAGE_BENCH_PYTHON:-python}"
QWEN_IMAGE_BENCH_MODEL="${QWEN_IMAGE_BENCH_MODEL:-${QWEN_IMAGE_BENCH_ROOT}/Qwen/Qwen-Image-Bench}"
QWEN_IMAGE_BENCH_PROMPTS="${QWEN_IMAGE_BENCH_PROMPTS:-${QWEN_IMAGE_BENCH_DATASET_ROOT}/qwen_image_bench_hf_v0518.jsonl}"
QWEN_IMAGE_BENCH_METADATA="${QWEN_IMAGE_BENCH_METADATA:-${QWEN_IMAGE_BENCH_DATASET_ROOT}/metadata/bench_metadata.json}"
QWEN_IMAGE_BENCH_OUTPUT_ROOT="${QWEN_IMAGE_BENCH_OUTPUT_ROOT:-${WORKSPACE_DIR}/Qwen-Image-Bench_results}"
QWEN_IMAGE_BENCH_JUDGE_GPUS="${QWEN_IMAGE_BENCH_JUDGE_GPUS:-}"
QWEN_IMAGE_BENCH_JUDGE_BATCH_SIZE="${QWEN_IMAGE_BENCH_JUDGE_BATCH_SIZE:-24}"
QWEN_IMAGE_BENCH_MAX_NEW_TOKENS="${QWEN_IMAGE_BENCH_MAX_NEW_TOKENS:-4096}"
QWEN_IMAGE_BENCH_REPEAT_INDEX="${QWEN_IMAGE_BENCH_REPEAT_INDEX:-0}"
QWEN_IMAGE_BENCH_INPUT_BUILDER="${QWEN_IMAGE_BENCH_INPUT_BUILDER:-${WORKSPACE_DIR}/recipe/transfusion/scripts/build_qwen_image_bench_input.py}"
QWEN_IMAGE_BENCH_RESOLUTION="${QWEN_IMAGE_BENCH_RESOLUTION:-1024}"

if [[ ! "${QWEN_IMAGE_BENCH_RESOLUTION}" =~ ^[0-9]+$ ]] || (( 10#${QWEN_IMAGE_BENCH_RESOLUTION} <= 0 || 10#${QWEN_IMAGE_BENCH_RESOLUTION} % 32 != 0 )); then
    echo "[ERROR] QWEN_IMAGE_BENCH_RESOLUTION must be a positive integer divisible by 32, got: ${QWEN_IMAGE_BENCH_RESOLUTION}" >&2
    exit 1
fi
QWEN_IMAGE_BENCH_HEIGHT=$((10#${QWEN_IMAGE_BENCH_RESOLUTION} / 32))
QWEN_IMAGE_BENCH_WIDTH="${QWEN_IMAGE_BENCH_HEIGHT}"
if [ ! -f "${QWEN_IMAGE_BENCH_METADATA}" ] && [ -f "${QWEN_IMAGE_BENCH_ROOT}/metadata/bench_metadata.json" ]; then
    QWEN_IMAGE_BENCH_METADATA="${QWEN_IMAGE_BENCH_ROOT}/metadata/bench_metadata.json"
fi

# Pixel-DiT 结构超参 (若 checkpoint 有 run_config.yaml, 脚本会自动读取覆盖以下默认值)
NUM_DIFFUSION_LAYERS="${NUM_DIFFUSION_LAYERS:-10}"
PIXEL_DIT_GATE_INJECT_LAYER="${PIXEL_DIT_GATE_INJECT_LAYER:-3}"
PIXEL_DIT_REF_GATE_MODE="${PIXEL_DIT_REF_GATE_MODE:-none}"
DIT_HEAD_IMPL="${DIT_HEAD_IMPL:-torch}"
PREDICT_GATE_MODE="${PREDICT_GATE_MODE:-auto}"  # auto | 1 | 0

case "${PREDICT_GATE_MODE}" in
    auto|AUTO|Auto)
        PREDICT_GATE_ARG=""
        ;;
    1|true|TRUE|True)
        PREDICT_GATE_ARG="--predict_gate_mode"
        ;;
    0|false|FALSE|False)
        PREDICT_GATE_ARG="--no-predict_gate_mode"
        ;;
    *)
        echo "[ERROR] PREDICT_GATE_MODE must be auto/1/0/true/false, got: ${PREDICT_GATE_MODE}"
        exit 1
        ;;
esac

case "${RESPECT_PROMPT_ASPECT_RATIO}" in
    1|true|TRUE|True)
        PROMPT_ASPECT_RATIO_ARG="--respect_prompt_aspect_ratio"
        ;;
    0|false|FALSE|False)
        PROMPT_ASPECT_RATIO_ARG="--no-respect_prompt_aspect_ratio"
        ;;
    *)
        echo "[ERROR] RESPECT_PROMPT_ASPECT_RATIO must be 0/1/true/false, got: ${RESPECT_PROMPT_ASPECT_RATIO}" >&2
        exit 1
        ;;
esac

case "${CFG_KEEP_CHAT_TEMPLATE}" in
    1|true|TRUE|True)
        CFG_TEMPLATE_ARG="--cfg_keep_chat_template"
        ;;
    0|false|FALSE|False)
        CFG_TEMPLATE_ARG="--no-cfg_keep_chat_template"
        ;;
    *)
        echo "[ERROR] CFG_KEEP_CHAT_TEMPLATE must be 0/1/true/false, got: ${CFG_KEEP_CHAT_TEMPLATE}"
        exit 1
        ;;
esac

case "${RESUME}" in
    1|true|TRUE|True)
        RESUME_ARG="--resume"
        ;;
    0|false|FALSE|False)
        RESUME_ARG=""
        ;;
    *)
        echo "[ERROR] RESUME must be 0/1/true/false, got: ${RESUME}"
        exit 1
        ;;
esac

# ═══════════════════════════════════════════════════════════════
#  Checkpoint 列表: CKPT_PATH MODEL_NAME
#  每行一个, 格式: "checkpoint路径 模型名称"
# ═══════════════════════════════════════════════════════════════

CKPT_LIST=(
    # 尚无正式训完的 pixel_dit checkpoint, 先用过拟合验证配置的产出占位:
    # "output/pixel_dit_overfit/iter_0005000  pixel_dit-overfit-5k"
    # "<COMPANY>/Qwen3-VL-4B-umm-v0-7-Pro-256-stage2-dit-0710/iter_0110000 Qwen3-VL-4B-umm-v0-7-Pro-256-stage2-0710-110k"
    # "<COMPANY>/Qwen3-VL-4B-umm-v0-7-Pro-512-stage3-dit-0710/iter_0030000 Qwen3-VL-4B-umm-v0-7-Pro-512-stage3-0710-30k-cfg4-shift4"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-512-stage4-dit-0710/iter_0040000 Qwen3-VL-4B-umm-v0-7-Pro-512-stage4-0710-40k-cfg4-shift4"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-512-stage4-dit-0710-RoPE/iter_0005000 Qwen3-VL-4B-umm-v0-7-Pro-512-stage4-0710-5k-RoPE-cfg4-shift4-fixed-cfg"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-stage5-dit-0710/iter_0015000 Qwen3-VL-4B-umm-v0-7-Pro-1024-stage5-0710-15k-cfg4-shift4-fixed-cfg"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-stage5-dit-0710-1024/iter_0005000 Qwen3-VL-4B-umm-v0-7-Pro-1024-stage5-0710-1024-5k-cfg4-shift4-fixed-cfg"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-stage5-dit-0723-1024-RoPE/iter_0005000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-stage5-dit-0723-1024-RoPE-5k-cfg4-shift4-fixed-cfg"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-512-1024-stage5-decay-0723-RoPE/iter_0015000/ Qwen3-VL-4B-umm-v0-7-Pro-512-1024-stage5-decay-0723-RoPE-15k-cfg4-shift4-cfg-chat"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-512-1024-stage5-decay-0724-RoPE/iter_0060000/ Qwen3-VL-4B-umm-v0-7-Pro-512-1024-stage5-decay-0724-RoPE-60k-cfg4-shift4"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-stage6-SFT-0724-RoPE/iter_0030000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-stage6-SFT-0724-RoPE-30k-cfg4-shift4"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-stage6-SFT-0724-RoPE/iter_0020000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-stage6-SFT-0724-RoPE-20k-cfg4-shift4"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-stage6-SFT-0724-RoPE-uniform/iter_0015000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-stage6-SFT-0724-RoPE-uniform-15k-cfg4-shift4"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724/iter_0005000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-5k-2048"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v1/iter_0010000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v1-10k-2048"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v1/iter_0015000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v1-15k-2048"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v1/iter_0020000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v1-20k-2048"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v2/iter_0020000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v2-20k-2048"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v2/iter_0025000/ Qwen3-VL-4B-umm-v0-7-Pro-1024-2048-stage7-0724-v2-25k-2048"
    # Add entries here, or provide CKPT_LIST_FILE/CKPT_PATH_OVERRIDE.
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-512-t2i-0731/iter_0200000/ Qwen3-VL-4B-umm-v0-7-Pro-512-t2i-0731-200k-cfg4-shift4"
    # "output/Qwen3-VL-4B-umm-v0-7-Pro-512-t2i-0731/iter_0150000/ Qwen3-VL-4B-umm-v0-7-Pro-512-t2i-0731-150k-cfg4-shift4"
)   

# 可通过文件或环境变量临时评测 checkpoint，而不修改上面的默认列表。
if [ -n "${CKPT_LIST_FILE:-}" ]; then
    mapfile -t CKPT_LIST < <(grep -v '^[[:space:]]*#' "${CKPT_LIST_FILE}" | sed '/^[[:space:]]*$/d')
fi
if [ -n "${CKPT_PATH_OVERRIDE:-}" ]; then
    if [ -z "${MODEL_NAME_OVERRIDE:-}" ]; then
        echo "[ERROR] MODEL_NAME_OVERRIDE is required with CKPT_PATH_OVERRIDE"
        exit 1
    fi
    CKPT_LIST=("${CKPT_PATH_OVERRIDE} ${MODEL_NAME_OVERRIDE}")
fi
if [ -n "${CKPT_PATH:-}" ] && [ -n "${MODEL_NAME:-}" ]; then
    CKPT_LIST=("${CKPT_PATH} ${MODEL_NAME}")
fi
if [ ${#CKPT_LIST[@]} -eq 0 ]; then
    echo "[ERROR] No checkpoints configured. Set CKPT_LIST_FILE or CKPT_PATH_OVERRIDE + MODEL_NAME_OVERRIDE." >&2
    exit 2
fi

# ═══════════════════════════════════════════════════════════════
#  Benchmark 列表: 通过 Python 端 benchmark_list 顺序执行
# ═══════════════════════════════════════════════════════════════

BENCHMARK_LIST_WAS_SET=0
if [ -n "${BENCHMARK_LIST+x}" ]; then
    BENCHMARK_LIST_WAS_SET=1
fi

BENCHMARK_LIST="${BENCHMARK_LIST:-OneIG_ZH,OneIG_EN,LongText_ZH,LongText_EN}"
IFS=',' read -r -a BENCHMARKS <<< "${BENCHMARK_LIST}"

benchmark_list_contains_qwen_image_bench() {
    local benchmark
    for benchmark in "$@"; do
        case "${benchmark//[[:space:]]/}" in
            QwenImageBench|Qwen-Image-Bench|QWEN_IMAGE_BENCH)
                return 0
                ;;
        esac
    done
    return 1
}

QWEN_IMAGE_BENCH_REQUESTED=0
if benchmark_list_contains_qwen_image_bench "${BENCHMARKS[@]}"; then
    QWEN_IMAGE_BENCH_REQUESTED=1
fi

IFS=',' read -r -a EVAL_BENCHMARK_ARRAY <<< "${EVAL_BENCHMARKS}"
if benchmark_list_contains_qwen_image_bench "${EVAL_BENCHMARK_ARRAY[@]}"; then
    QWEN_IMAGE_BENCH_REQUESTED=1
fi

# If the caller selected only the Qwen-Image-Bench evaluator, avoid silently
# running the legacy OneIG/LongText generation defaults as well.
if [ "${QWEN_IMAGE_BENCH_REQUESTED}" -eq 1 ] && [ "${BENCHMARK_LIST_WAS_SET}" -eq 0 ]; then
    BENCHMARKS=(QwenImageBench)
fi
if [ "${QWEN_IMAGE_BENCH_REQUESTED}" -eq 1 ] && [ "${EVAL_BENCHMARKS_WAS_SET}" -eq 0 ]; then
    EVAL_BENCHMARKS="QwenImageBench"
    EVAL_BENCHMARK_ARRAY=(QwenImageBench)
fi
BENCHMARK_LIST="$(IFS=,; echo "${BENCHMARKS[*]}")"

# The inference entry point only knows the legacy benchmark names. Remove the
# Qwen-Image-Bench token before passing the list through, while retaining it in
# the wrapper so the judge can run after generation.
NATIVE_BENCHMARKS=()
for benchmark in "${BENCHMARKS[@]}"; do
    case "${benchmark//[[:space:]]/}" in
        QwenImageBench|Qwen-Image-Bench|QWEN_IMAGE_BENCH) ;;
        *) NATIVE_BENCHMARKS+=("${benchmark}") ;;
    esac
done
NATIVE_BENCHMARK_LIST="$(IFS=,; echo "${NATIVE_BENCHMARKS[*]}")"
if [ "${QWEN_IMAGE_BENCH_REQUESTED}" -eq 1 ] && ! benchmark_list_contains_qwen_image_bench "${EVAL_BENCHMARK_ARRAY[@]}"; then
    EVAL_BENCHMARKS="${EVAL_BENCHMARKS},QwenImageBench"
fi

POSTPROCESS_QUEUE_FILE="${POSTPROCESS_QUEUE_FILE:-/tmp/oneig_postprocess_queue_${USER:-user}_$$.tsv}"
: > "${POSTPROCESS_QUEUE_FILE}"

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

run_qwen_image_bench_inference() {
    local ckpt_path="$1"
    local model_name="$2"
    local run_height="$3"
    local run_width="$4"
    local iter_arg="$5"
    local infer_master_port="$6"
    local output_dir="${QWEN_IMAGE_BENCH_OUTPUT_ROOT}/${model_name}"

    if [ ! -f "${QWEN_IMAGE_BENCH_PROMPTS}" ]; then
        echo "[ERROR] Qwen-Image-Bench prompt JSONL not found: ${QWEN_IMAGE_BENCH_PROMPTS}" >&2
        echo "        Set QWEN_IMAGE_BENCH_PROMPTS to qwen_image_bench_hf_v0518.jsonl (or equivalent)." >&2
        return 1
    fi
    if [ ! -f "${QWEN_IMAGE_BENCH_INPUT_BUILDER}" ]; then
        echo "[ERROR] Qwen-Image-Bench input builder not found: ${QWEN_IMAGE_BENCH_INPUT_BUILDER}" >&2
        return 1
    fi

    mkdir -p "${output_dir}"
    echo "[QWEN-BENCH] Generating ${model_name} -> ${output_dir}"
    local infer_rc=0
    torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${QWEN_IMAGE_BENCH_INFER_MASTER_PORT:-${infer_master_port}} \
        "${INFER_SCRIPT}" \
        --ckpt "${ckpt_path}" \
        --hf_path "${HF_PATH}" \
        --out_dir "${output_dir}" \
        --data_format jsonl \
        --data_file "${QWEN_IMAGE_BENCH_PROMPTS}" \
        --height "${run_height}" \
        --width "${run_width}" \
        ${PROMPT_ASPECT_RATIO_ARG} \
        --steps "${STEPS}" \
        --cfg_scale "${CFG_SCALE}" \
        --shift "${SHIFT}" \
        --scheduler "${SCHEDULER}" \
        --repeat_times "${REPEAT_TIMES}" \
        --num_diffusion_layers "${NUM_DIFFUSION_LAYERS}" \
        --pixel_dit_gate_inject_layer "${PIXEL_DIT_GATE_INJECT_LAYER}" \
        --pixel_dit_ref_gate_mode "${PIXEL_DIT_REF_GATE_MODE}" \
        --dit_head_impl "${DIT_HEAD_IMPL}" \
        --num_samples "${QWEN_IMAGE_BENCH_NUM_SAMPLES:--1}" \
        --save_images \
        --batch_size "${BATCH_SIZE}" \
        --model_name "${model_name}" \
        ${iter_arg:-} \
        ${CFG_TEMPLATE_ARG} \
        ${PREDICT_GATE_ARG} \
        ${RESUME_ARG} || infer_rc=$?

    if [ "${infer_rc}" -ne 0 ]; then
        echo "[ERROR] Qwen-Image-Bench generation failed with exit code ${infer_rc}: ${model_name}" >&2
        return "${infer_rc}"
    fi

    echo "[QWEN-BENCH] Generated ${model_name}"
}

run_qwen_image_bench_evaluation() {
    local model="$1"
    local result_dir="${QWEN_IMAGE_BENCH_OUTPUT_ROOT}/${model}"
    local metadata_path="${result_dir}/metadata.jsonl"
    local input_path="${result_dir}/qwen_image_bench_input.jsonl"
    local log_path="${result_dir}/qwen_image_bench_judge.log"

    if [ ! -x "${QWEN_IMAGE_BENCH_PYTHON}" ]; then
        echo "[ERROR] Qwen-Image-Bench Python not executable: ${QWEN_IMAGE_BENCH_PYTHON}" >&2
        return 1
    fi
    if [ ! -f "${QWEN_IMAGE_BENCH_MODEL}/config.json" ]; then
        echo "[ERROR] Qwen-Image-Bench judge model not found: ${QWEN_IMAGE_BENCH_MODEL}" >&2
        return 1
    fi
    if [ ! -f "${QWEN_IMAGE_BENCH_METADATA}" ]; then
        echo "[ERROR] Qwen-Image-Bench metadata not found: ${QWEN_IMAGE_BENCH_METADATA}" >&2
        return 1
    fi
    if [ ! -f "${metadata_path}" ]; then
        echo "[ERROR] Pixel-DiT metadata not found: ${metadata_path}" >&2
        return 1
    fi
    if ! (
        cd "${QWEN_IMAGE_BENCH_ROOT}"
        env -u PYTHONPATH "${QWEN_IMAGE_BENCH_PYTHON}" -c 'import numpy, pandas, swift'
    ) >/dev/null 2>&1; then
        echo "[ERROR] Qwen-Image-Bench environment cannot import numpy/pandas/swift." >&2
        echo "        Run the judge with a clean PYTHONPATH and install requirements.txt in ${QWEN_IMAGE_BENCH_PYTHON}." >&2
        return 1
    fi

    echo "[QWEN-BENCH] Building judge input for ${model}"
    "${QWEN_IMAGE_BENCH_PYTHON}" "${QWEN_IMAGE_BENCH_INPUT_BUILDER}" \
        --metadata "${metadata_path}" \
        --output-dir "${result_dir}" \
        --output "${input_path}" \
        --repeat-index "${QWEN_IMAGE_BENCH_REPEAT_INDEX}"

    echo "[QWEN-BENCH] Judging ${model} with ${QWEN_IMAGE_BENCH_MODEL}"
    local -a judge_cmd=(
        "${QWEN_IMAGE_BENCH_PYTHON}"
        "${QWEN_IMAGE_BENCH_ROOT}/judge.py"
        --input "${input_path}"
        --model "${QWEN_IMAGE_BENCH_MODEL}"
        --local-metadata "${QWEN_IMAGE_BENCH_METADATA}"
        --max-batch-size "${QWEN_IMAGE_BENCH_JUDGE_BATCH_SIZE}"
        --max-new-tokens "${QWEN_IMAGE_BENCH_MAX_NEW_TOKENS}"
    )
    if [ -n "${QWEN_IMAGE_BENCH_JUDGE_GPUS}" ]; then
        (
            cd "${QWEN_IMAGE_BENCH_ROOT}"
            env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${QWEN_IMAGE_BENCH_JUDGE_GPUS}" \
                "${judge_cmd[@]}"
        ) >"${log_path}" 2>&1
    else
        (
            cd "${QWEN_IMAGE_BENCH_ROOT}"
            env -u PYTHONPATH "${judge_cmd[@]}"
        ) >"${log_path}" 2>&1
    fi
    echo "[QWEN-BENCH] Scores written under ${result_dir}"
}

# ═══════════════════════════════════════════════════════════════
#  运行函数
# ═══════════════════════════════════════════════════════════════

run_model() {
    local CKPT_PATH="$1"
    local MODEL_NAME="$2"
    local RUN_RESOLUTION="$3"
    local RUN_HEIGHT="$4"
    local RUN_WIDTH="$5"

    local RUN_MASTER_PORT="${MASTER_PORT}"
    if [ -z "${RUN_MASTER_PORT}" ]; then
        # 必须按 0.0.0.0 探测：torchrun 的 TCPStore 绑在 0.0.0.0，而只占在外部
        # 地址上的端口用 bind(("127.0.0.1", 0)) 是探不出来的 —— 探测成功、真正
        # 启动时 EADDRINUSE。实测 bind(127.0.0.1,18003) 成功而
        # bind(0.0.0.0,18003) 报 errno 98，两个 LongText 评测点就是这样在启动
        # 30 秒内连续挂掉的。选完再 bind+listen 复验一次。
        # 残留 TOCTOU：探测释放端口到 torchrun 占用之间仍有窗口，所以调用方
        # （如 workbench/scripts/.tmp_longtext_*.sh）保留一次换端口重试。
        RUN_MASTER_PORT="$(python - <<'PY'
import os, random, socket, sys

# 只在 ip_local_reserved_ports 里挑。内核不会把保留段里的端口自动分配给别人的
# 出向连接，而临时段（ip_local_port_range，本机 10000-61000）里的端口随时会被抢。
# 挂掉的那个 18003 正好落在保留段 6676-18000 的上沿之外 —— 这才是它被占的原因。
try:
    reserved = open("/proc/sys/net/ipv4/ip_local_reserved_ports").read().strip()
except OSError:
    reserved = ""
segs = []
for seg in reserved.split(","):
    if "-" not in seg:
        continue
    lo, _, hi = seg.partition("-")
    try:
        lo, hi = int(lo), int(hi)
    except ValueError:
        continue
    if hi - lo >= 64:          # 短段多为系统组件自用
        segs.append((lo, hi))
if not segs:
    segs = [(20000, 20999)]    # 内核没配保留段时的兜底

def usable(port):
    # 必须按 0.0.0.0 试：torchrun 的 TCPStore 绑 0.0.0.0，只占在外部地址上的端口
    # 用 bind(("127.0.0.1", ...)) 探不出来，也不能只用 connect 探 —— connect 只
    # 看得见 listen 中的 socket，看不见 bind 了但没 listen 的。
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port))
        s.listen(1)
        return True
    except OSError:
        return False
    finally:
        s.close()

rng = random.SystemRandom()
for _ in range(400):
    lo, hi = rng.choice(segs)
    if usable(rng.randrange(lo, hi + 1)):
        # 复挑一次并立即打印：上面那个已被 close，这里重取是为了把"可用"这件事
        # 尽量贴近 torchrun 真正 bind 的时刻（TOCTOU 窗口只能缩小、消不掉）。
        for _ in range(64):
            p = rng.randrange(lo, hi + 1)
            if usable(p):
                print(p)
                sys.exit(0)
sys.exit(1)
PY
)"
        if [ -z "${RUN_MASTER_PORT}" ]; then
            echo "[ERROR] ${MODEL_NAME}: 找不到可用的 MASTER_PORT" >&2
            return 1
        fi
    fi

    local resolved
    local ckpt_arg
    local iter_arg
    resolved=$(resolve_ckpt_args "${CKPT_PATH}")
    ckpt_arg="${resolved%%|*}"
    iter_arg="${resolved#*|}"

    local rc=0
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "  CKPT:      ${CKPT_PATH}"
    echo "  MODEL:     ${MODEL_NAME}"
    echo "  RESOLUTION:${RUN_RESOLUTION} (merged grid ${RUN_HEIGHT}x${RUN_WIDTH})"
    echo "  BENCHMARKS:${BENCHMARK_LIST}"
    echo "  PREDICT_GATE_MODE:${PREDICT_GATE_MODE}"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    echo "[INFO] ${MODEL_NAME}: MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${RUN_MASTER_PORT}"
    if [ -n "${NATIVE_BENCHMARK_LIST}" ]; then
        torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${RUN_MASTER_PORT} \
        "${INFER_SCRIPT}" \
        --ckpt ${ckpt_arg} \
        --hf_path ${HF_PATH} \
        --benchmark_list "${NATIVE_BENCHMARK_LIST}" \
        --height ${RUN_HEIGHT} \
        --width ${RUN_WIDTH} \
        ${PROMPT_ASPECT_RATIO_ARG} \
        --steps ${STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --shift ${SHIFT} \
        --scheduler ${SCHEDULER} \
        --repeat_times ${REPEAT_TIMES} \
        --num_diffusion_layers ${NUM_DIFFUSION_LAYERS} \
        --pixel_dit_gate_inject_layer ${PIXEL_DIT_GATE_INJECT_LAYER} \
        --pixel_dit_ref_gate_mode ${PIXEL_DIT_REF_GATE_MODE} \
        --dit_head_impl ${DIT_HEAD_IMPL} \
        --num_samples -1 \
        --save_images \
        --batch_size ${BATCH_SIZE} \
        --model_name "${MODEL_NAME}" \
        --oneig_postprocess \
        --oneig_postprocess_queue_file "${POSTPROCESS_QUEUE_FILE}" \
        --oneig_postprocess_script "${ONEIG_STITCH_SCRIPT}" \
        --oneig_postprocess_workers "${ONEIG_STITCH_WORKERS}" \
        ${iter_arg} \
        ${CFG_TEMPLATE_ARG} \
        ${PREDICT_GATE_ARG} \
        ${RESUME_ARG} || rc=$?
    fi

    if [ ${rc} -eq 0 ] && [ "${QWEN_IMAGE_BENCH_REQUESTED}" -eq 1 ]; then
        if run_qwen_image_bench_inference "${ckpt_arg}" "${MODEL_NAME}" "${QWEN_IMAGE_BENCH_HEIGHT}" "${QWEN_IMAGE_BENCH_WIDTH}" "${iter_arg}" "${RUN_MASTER_PORT}"; then
            :
        else
            rc=$?
        fi
    fi

    if [ ${rc} -ne 0 ]; then
        echo "[ERROR] torchrun failed with exit code ${rc}: ${MODEL_NAME}"
        return ${rc}
    fi

    echo "[DONE] ${MODEL_NAME}"
}

wait_for_oneig_postprocess_queue() {
    local queue_file="$1"
    if [ ! -s "${queue_file}" ]; then
        echo "[INFO] No OneIG postprocess jobs queued."
        return 0
    fi

    echo ""
    echo "[INFO] Waiting for queued OneIG postprocess jobs: ${queue_file}"
    local rc=0
    while IFS=$'\t' read -r pid status_path model_name benchmark_name; do
        [ -z "${pid}" ] && continue
        echo "[WAIT] ${model_name} / ${benchmark_name} (pid=${pid})"
        while kill -0 "${pid}" 2>/dev/null; do
            sleep 2
        done
        if [ ! -f "${status_path}" ]; then
            echo "[FAIL] Missing postprocess status file: ${status_path}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            model_mark_failed "${model_name}"
            rc=1
            continue
        fi
        if ! python - "${status_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    status = json.load(f)

sys.exit(0 if status.get("ok") else 1)
PY
        then
            echo "[FAIL] OneIG postprocess failed: ${model_name} / ${benchmark_name}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            model_mark_failed "${model_name}"
            rc=1
        else
            echo "[DONE] OneIG postprocess: ${model_name} / ${benchmark_name}"
        fi
    done < "${queue_file}"

    return ${rc}
}

model_mark_failed() {
    local model="$1"
    FAILED_MODEL_NAMES+=("${model}")
}

model_has_failed() {
    local model="$1"
    local failed
    for failed in "${FAILED_MODEL_NAMES[@]}"; do
        if [ "${failed}" = "${model}" ]; then
            return 0
        fi
    done
    return 1
}

run_oneig_vllm_evaluation() {
    local model="$1"
    local result_root="${HIUNI_EVAL_ROOT}/OneIG_results/${model}"
    local alignment_log="/tmp/oneig_alignment_${model}.log"
    local other_dims_log="/tmp/oneig_other_dims_${model}.log"
    local alignment_pid
    local other_dims_pid
    local rc=0

    if [ ! -x "${ONEIG_ALIGNMENT_SCRIPT}" ]; then
        echo "[ERROR] Missing OneIG vLLM alignment runner: ${ONEIG_ALIGNMENT_SCRIPT}" >&2
        return 1
    fi
    if [ ! -x "${ONEIG_OTHER_DIMS_SCRIPT}" ]; then
        echo "[ERROR] Missing OneIG other-dim runner: ${ONEIG_OTHER_DIMS_SCRIPT}" >&2
        return 1
    fi

    echo "[EVAL] OneIG alignment via vLLM (${ONEIG_ALIGNMENT_GPUS})"
    MODEL_NAME="${model}" \
    RESULT_ROOT="${result_root}" \
    ALIGNMENT_GPUS="${ONEIG_ALIGNMENT_GPUS}" \
    CONCURRENCY="${ONEIG_ALIGNMENT_CONCURRENCY}" \
    TILE_WORKERS="${ONEIG_ALIGNMENT_TILE_WORKERS}" \
    API_SERVER_COUNT="${ONEIG_ALIGNMENT_API_SERVER_COUNT}" \
    MAX_NUM_SEQS="${ONEIG_ALIGNMENT_MAX_NUM_SEQS}" \
    bash "${ONEIG_ALIGNMENT_SCRIPT}" >"${alignment_log}" 2>&1 &
    alignment_pid=$!

    echo "[EVAL] OneIG Text/Reasoning/Style/Diversity (${ONEIG_OTHER_DIMS_GPUS})"
    MODEL_NAME="${model}" \
    RESULT_ROOT="${result_root}" \
    OTHER_DIMS_GPUS="${ONEIG_OTHER_DIMS_GPUS}" \
    MODES="${ONEIG_OTHER_DIMS_MODES:-EN ZH}" \
    bash "${ONEIG_OTHER_DIMS_SCRIPT}" >"${other_dims_log}" 2>&1 &
    other_dims_pid=$!

    if [ -z "${alignment_pid}" ] || ! wait "${alignment_pid}"; then
        echo "[FAIL] OneIG vLLM alignment; see ${alignment_log}" >&2
        rc=1
    fi
    if [ -z "${other_dims_pid}" ] || ! wait "${other_dims_pid}"; then
        echo "[FAIL] OneIG other dimensions; see ${other_dims_log}" >&2
        rc=1
    fi
    return "${rc}"
}

run_longtext_evaluation() {
    local model="$1"
    local result_root="${HIUNI_EVAL_ROOT}/LongText_results/${model}"
    local rc=0

    if [ ! -f "${LONGTEXT_EVAL_SCRIPT}" ]; then
        echo "[ERROR] Missing LongText evaluator: ${LONGTEXT_EVAL_SCRIPT}" >&2
        return 1
    fi

    # The evaluator owns all 8 visible GPUs and rewrites results.jsonl, so
    # evaluate EN/ZH serially and give each language its own output directory.
    for mode in EN ZH; do
        local mode_lower="${mode,,}"
        local sample_dir="${result_root}/${mode}/images"
        local output_dir="${result_root}/${mode}/textbench"
        local log_file="/tmp/longtext_${model}_${mode_lower}.log"
        mkdir -p "${output_dir}"
        echo "[EVAL] LongText ${mode} on GPUs ${LONGTEXT_GPUS}"
        if ! CUDA_VISIBLE_DEVICES="${LONGTEXT_GPUS}" \
            bash "${LONGTEXT_EVAL_SCRIPT}" \
                --sample_dir "${sample_dir}" \
                --output_dir "${output_dir}" \
                --mode "${mode_lower}" \
                --master_port "${LONGTEXT_MASTER_PORT}" >"${log_file}" 2>&1; then
            echo "[FAIL] LongText ${mode}; see ${log_file}" >&2
            rc=1
        fi
    done
    return "${rc}"
}

run_post_sync_evaluation() {
    [[ "${RUN_EVAL_AFTER_SYNC}" =~ ^(1|true|TRUE|True)$ ]] || return 0
    local eval_fail_count=0
    for model in "${SYNCED_MODEL_NAMES[@]}"; do
        echo ""
        echo "========== Evaluating ${model} =========="
        case ",${EVAL_BENCHMARKS}," in
            *,OneIG,*)
                if ! run_oneig_vllm_evaluation "${model}"; then
                    eval_fail_count=$((eval_fail_count + 1))
                fi
                ;;
        esac
        case ",${EVAL_BENCHMARKS}," in
            *,LongText,*)
                if ! run_longtext_evaluation "${model}"; then
                    eval_fail_count=$((eval_fail_count + 1))
                fi
                ;;
        esac
        case ",${EVAL_BENCHMARKS}," in
            *,QwenImageBench,*|*,Qwen-Image-Bench,*|*,QWEN_IMAGE_BENCH,*)
                if ! run_qwen_image_bench_evaluation "${model}"; then
                    eval_fail_count=$((eval_fail_count + 1))
                fi
                ;;
        esac
    done
    if [ "${eval_fail_count}" -gt 0 ]; then
        echo "[WARNING] ${eval_fail_count} evaluation task(s) failed."
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════════

TOTAL_CKPTS=${#CKPT_LIST[@]}
TOTAL_BENCHMARKS=${#BENCHMARKS[@]}
TOTAL_RESOLUTIONS=${#VALID_RESOLUTIONS[@]}
echo "========== Pixel-DiT Batch Eval: ${TOTAL_CKPTS} checkpoints x ${TOTAL_RESOLUTIONS} resolutions x ${TOTAL_BENCHMARKS} benchmarks =========="
echo "Resolutions=${VALID_RESOLUTIONS[*]}  Steps=${STEPS}  CFG=${CFG_SCALE}  CFGKeepChatTemplate=${CFG_KEEP_CHAT_TEMPLATE}  PromptAspectRatio=${RESPECT_PROMPT_ASPECT_RATIO}  Shift=${SHIFT}  Scheduler=${SCHEDULER}"
echo "RepeatTimes=${REPEAT_TIMES}"
echo "NumDiffusionLayers=${NUM_DIFFUSION_LAYERS}  GateInjectLayer=${PIXEL_DIT_GATE_INJECT_LAYER}  DitHeadImpl=${DIT_HEAD_IMPL}"
echo "PixelDitRefGateMode=${PIXEL_DIT_REF_GATE_MODE}"
echo ""

FAIL_COUNT=0
INFER_SUCCEEDED_MODEL_NAMES=()
SYNCED_MODEL_NAMES=()
FAILED_MODEL_NAMES=()

for ckpt_entry in "${CKPT_LIST[@]}"; do
    # 解析 "CKPT_PATH  MODEL_NAME"
    CKPT_PATH=$(echo "${ckpt_entry}" | awk '{print $1}')
    MODEL_NAME=$(echo "${ckpt_entry}" | awk '{print $2}')

    if [ -z "${CKPT_PATH}" ] || [ -z "${MODEL_NAME}" ]; then
        echo "[WARN] Skipping invalid entry: ${ckpt_entry}"
        continue
    fi

    # 检查 checkpoint 是否存在
    if [ ! -d "${CKPT_PATH}" ]; then
        echo "[WARN] Checkpoint not found, skipping: ${CKPT_PATH}"
        continue
    fi

    for resolution_value in "${VALID_RESOLUTIONS[@]}"; do
        if [ "${TOTAL_RESOLUTIONS}" -eq 1 ]; then
            RUN_MODEL_NAME="${MODEL_NAME}"
            RUN_HEIGHT="${SINGLE_HEIGHT}"
            RUN_WIDTH="${SINGLE_WIDTH}"
        else
            RUN_MODEL_NAME="${MODEL_NAME}-res${resolution_value}"
            RUN_HEIGHT=$((resolution_value / 32))
            RUN_WIDTH=$((resolution_value / 32))
        fi

        if ! run_model "${CKPT_PATH}" "${RUN_MODEL_NAME}" "${resolution_value}" "${RUN_HEIGHT}" "${RUN_WIDTH}"; then
            echo "[FAIL] ${RUN_MODEL_NAME}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            model_mark_failed "${RUN_MODEL_NAME}"
        else
            INFER_SUCCEEDED_MODEL_NAMES+=("${RUN_MODEL_NAME}")
        fi
    done
done

if ! wait_for_oneig_postprocess_queue "${POSTPROCESS_QUEUE_FILE}"; then
    echo "[WARNING] OneIG postprocess had failures."
fi

for MODEL_NAME in "${INFER_SUCCEEDED_MODEL_NAMES[@]}"; do
    if ! model_has_failed "${MODEL_NAME}"; then
        SYNCED_MODEL_NAMES+=("${MODEL_NAME}")
    fi
done

if ! run_post_sync_evaluation; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""
echo "========== Pixel-DiT Batch Eval Complete =========="
if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "[WARNING] ${FAIL_COUNT} task(s) failed."
    exit 1
else
    echo "All tasks succeeded."
fi
