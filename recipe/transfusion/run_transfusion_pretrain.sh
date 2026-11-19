#!/bin/bash

# ===================================================================
# Transfusion 多模态架构 - 单机预训练启动脚本
# ===================================================================

# 1. 设置工作目录和环境变量
# 把当前目录加入 PYTHONPATH，这样系统就能认识 data, model, recipe 这些包了
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${WORKSPACE_DIR}"
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"
export PYTHONPATH=./3rdparty/Megatron-Bridge/src:./3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 缓解训练末尾 barrier 处 NCCL 申请通信 buffer 时的显存碎片导致的 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 分布式训练参数配置 (单机默认 8 卡; 多机时设置 NNODES / NODE_RANK / MASTER_ADDR)
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 3. 指定入口脚本和配置文件
# 指向我们刚刚写好的高层架构代码
TRAIN_SCRIPT="recipe/transfusion/pretrain_transfusion_model.py"

# Dense Pixel-DiT training configuration.
CONFIG_FILE="${CONFIG_FILE:-recipe/transfusion/conf/transfusion_qwen3_vl_4b_pixel_dit.yaml}"

# 训练数据在 config 的 transfusion.data_streams 中配置 (每个模态一个 energon MetadatasetV2 yaml)。
# 从发布的 Megatron ckpt 热启 SFT:
#   CONFIG_FILE=recipe/transfusion/conf/transfusion_qwen3_vl_4b_pixel_dit_sft_hotstart.yaml \
#       bash recipe/transfusion/run_transfusion_pretrain.sh

# ===================================================================
# 启动引擎
# ===================================================================
echo "🔥 开始点火！正在启动 Transfusion 预训练引擎..."
echo "🔹 使用 GPU 数量: ${GPUS_PER_NODE}"
echo "🔹 配置文件: ${CONFIG_FILE}"

torchrun() { ${PYTHON_BIN:-python} -m torch.distributed.run "$@"; }
# 额外参数原样透传给训练入口, 例如 Hydra 覆盖: train.train_iters=100 checkpoint.save=./output/x
torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --config-file "${CONFIG_FILE}" \
    --debug \
    --recipe qwen3_vl_2b \
    "$@"
