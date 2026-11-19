# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL model recipe integration for Transfusion training."""

import os
import dataclasses
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, Qwen2VLImageProcessor
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.data.energon.energon_provider import EnergonProvider
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig

from data.transfusion.transfusion_task_encoder import TransfusionTaskEncoder
from model.transfusion.schedulers import (
    normalize_diffusion_loss_type,
    normalize_diffusion_loss_reduction,
)
from model.transfusion.transfusion_qwen3vl_provider import TransfusionQwen3VLProvider


class Qwen3VLCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3-VL recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str
    # Dataset configuration
    train_data_path: Optional[List[str]]
    valid_data_path: Optional[List[str]]
    test_data_path: Optional[List[str]]
    dataset_type: Optional[str]
    image_folder: Optional[str]
    tokenizer_model: Optional[str]
    prompt_mapping_file: Optional[str]
    gen_min_pixels: Optional[int]
    gen_max_pixels: Optional[int]
    gen_max_input_pixels: Optional[int]
    edit_min_pixels: Optional[int]
    edit_max_pixels: Optional[int]
    edit_max_input_pixels: Optional[int]
    gen_resolution_buckets: Optional[List[int]]
    edit_resolution_buckets: Optional[List[int]]
    resolution_bucket_threshold: float
    # Model configuration
    tensor_parallelism: int
    pipeline_parallelism: int
    expert_parallelism: int
    pipeline_parallelism_dtype: Optional[torch.dtype]
    virtual_pipeline_parallelism: Optional[int]
    context_parallelism: int
    sequence_parallelism: bool
    use_megatron_fsdp: bool
    optimizer_cpu_offload: bool
    optimizer_offload_fraction: float
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    vit_lr: Optional[float]
    vit_min_lr: Optional[float]
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]
    # Freeze options
    freeze_language_model: bool
    freeze_vision_model: bool
    freeze_vision_projection: bool
    freeze_dit_param: bool
    freeze_pixel_proj_param: bool
    freeze_pixel_dit_gate_param: bool
    freeze_time_embeddings_param: bool
    freeze_pixel_dit_last_n_layers: int
    freeze_pixel_dit_final_layer: bool
    # Checkpoint options
    pretrained_checkpoint: Optional[str]
    # DDP options
    check_for_nan_in_grad: bool
    grad_reduce_in_fp32: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    data_parallel_sharding_strategy: str
    # TransfusionForwardStep options (consumed by pretrain entry, passed here to avoid error)
    predict_x0: bool
    diffusion_loss_type: Optional[str]
    diffusion_loss_reduction: Optional[str]
    log_per_loss_grad_norm: bool
    grad_norm_interval: int
    diff_loss_weight: float
    edit_diff_loss_weight: float
    # Deprecated compatibility field; COT CE uses the regular CE weight.
    generation_cot_ce_loss_weight: float
    generation_cot_diff_loss_weight: float
    generation_empty_think: bool
    generation_cot_prompt_sampling_weights: Optional[Dict[str, float]]
    generation_cot_long_template_prob: float
    generation_cot_long_template_version: str
    predict_gate_mode: Optional[bool]
    # Batch schedule (round-robin)
    batch_schedule: Optional[Dict[str, int]]
    batch_schedule_strict: bool
    # Multi-stream
    data_streams: Optional[Dict[str, str]]
    dataloader_load: Optional[str]
    dataloader_load_iteration: Optional[int]
    dataloader_reset_streams: Optional[Union[List[str], str]]
    # Dense Pixel-DiT DiT head 调制变体开关
    pixel_dit_hidden_size: int
    pixel_dit_gate_inject_layer: int
    pixel_dit_gate_input: str
    pixel_dit_ref_gate_mode: str
    diffusion_head_recompute: bool
    num_diffusion_layers: int
    num_diffusion_heads: int
    num_diffusion_kv_heads: int
    diffusion_intermediate_size: int
    adaln_post_modulation: bool
    dit_block_adaln: bool
    # DiT head 实现: "torch" 原生 / "megatron" TE+FP8
    dit_head_impl: str


def qwen3_vl_2b_finetune_config(**user_kwargs: Unpack[Qwen3VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3-VL 2B Instruct."""
    recommended_kwargs: Qwen3VLCommonKwargs = {
        "hf_path": "./checkpoints/Qwen3-VL-2B-Instruct",
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
        "min_lr": 1e-6,
        "lr": 1e-5,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen3VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _transfusion_qwen3_vl_common(**combined_kwargs)


def _transfusion_qwen3_vl_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "qwen3_vl_finetune",
    pretrained_checkpoint: Optional[str] = None,
    # Dataset configuration
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    dataset_type: Optional[str] = None,
    image_folder: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    prompt_mapping_file: Optional[str] = "data/image_und_midtrain_v0214_prompt_v4.list",
    num_workers: int = 2,
    packing_buffer_size: Optional[int] = None,
    dataloader_load: Optional[str] = None,
    dataloader_save: Optional[str] = None,
    dataloader_reset_streams: Optional[Union[List[str], str]] = None,
    dataloader_load_iteration: Optional[int] = None,
    # Model configuration
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    expert_parallelism: Optional[int] = None,
    expert_tensor_parallelism: int = 1,
    sequence_parallelism: bool = False,
    use_megatron_fsdp: bool = False,
    optimizer_cpu_offload: bool = False,
    optimizer_offload_fraction: float = 1.0,
    # Training hyperparameters
    train_iters: int = 300000,
    global_batch_size: int = 32,
    micro_batch_size: int = 2,
    seq_length: int = 512,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    vit_lr: Optional[float] = None,
    vit_min_lr: Optional[float] = None,
    lr_warmup_iters: int = 500,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 500,
    save_interval: int = 500,
    # Precision and comm overlap
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # Freeze options
    freeze_language_model: bool = True,
    freeze_vision_model: bool = True,
    freeze_vision_projection: bool = False,
    freeze_dit_param: bool = False,
    freeze_pixel_proj_param: bool = False,
    freeze_pixel_dit_gate_param: bool = False,
    freeze_time_embeddings_param: bool = False,
    freeze_pixel_dit_last_n_layers: int = 0,
    freeze_pixel_dit_final_layer: bool = False,
    # Vision pixel range (per task type)
    gen_min_pixels: Optional[int] = None,
    gen_max_pixels: Optional[int] = None,
    gen_max_input_pixels: Optional[int] = None,
    edit_min_pixels: Optional[int] = None,
    edit_max_pixels: Optional[int] = None,
    edit_max_input_pixels: Optional[int] = None,
    gen_resolution_buckets: Optional[List[int]] = None,
    edit_resolution_buckets: Optional[List[int]] = None,
    resolution_bucket_threshold: float = 0.95,
    und_min_pixels: Optional[int] = None,
    und_max_pixels: Optional[int] = None,
    cfg_drop_rate: float = 0.1,
    cfg_keep_chat_template: bool = False,
    square_crop: bool = False,
    enable_und_bidirection: bool = False,
    enable_think: bool = False,  # 是否在 SFT 训练时注入 <think> 块
    generation_empty_think: bool = False,
    generation_cot_prompt_sampling_weights: Optional[Dict[str, float]] = None,
    generation_cot_long_template_prob: float = 0.0,
    generation_cot_long_template_version: str = "cpt_v5",
    predict_gate_mode: Optional[bool] = None,
    editing_encode_type: str = "vit",
    # ViT recompute (gradient checkpointing for vision model)
    vit_recompute: bool = False,
    diffusion_head_recompute: bool = False,
    # Dense Pixel-DiT configuration
    pixel_dit_hidden_size: int = 0,
    pixel_dit_gate_inject_layer: int = 0,
    pixel_dit_gate_input: str = "pixel",
    use_pixel_dit_pixel_residual: bool = True,
    pixel_dit_ref_gate_mode: str = "none",
    # post-modulation (NVlabs PixelDiT #6): 4-way scale/shift on attn/mlp outputs,
    # 缓解 spike; 默认 False 保持 6-way pre-mod。作用于 pixel_dit 多层 DiT head。
    adaln_post_modulation: bool = False,
    # dit_block_adaln=False: pixel_dit DiT head 的 transformer block 内不做 adaLN 调制
    # (纯 pre-norm transformer), 时间条件仅由 head 末层 adaLN 注入; 默认 True 保持原状。
    dit_block_adaln: bool = True,
    # DiT head 实现: "torch" 原生 (默认) / "megatron" TE+FP8。见 provider 同名字段。
    dit_head_impl: str = "torch",
    # Multi-layer diffusion head (tuna-2 style)
    num_diffusion_layers: int = 0,
    num_diffusion_heads: int = 0,      # 0 = 自动对齐 LLM 的 num_attention_heads
    num_diffusion_kv_heads: int = 0,   # 0 = 自动对齐 LLM 的 num_key_value_heads
    diffusion_intermediate_size: int = 0,
    # DDP options
    check_for_nan_in_grad: bool = True,
    grad_reduce_in_fp32: bool = True,
    overlap_grad_reduce: bool = False,
    overlap_param_gather: bool = False,
    average_in_collective: bool = True,
    data_parallel_sharding_strategy: str = "optim_grads_params",
    use_distributed_optimizer: bool = True,
    # diffusion_loss_type / diffusion_loss_reduction / predict_x0 /
    # log_per_loss_grad_norm / grad_norm_interval /
    # diff_loss_weight / edit_diff_loss_weight:
    # handled by pretrain entry (TransfusionForwardStep), accepted here to avoid **kwargs error
    predict_x0: bool = False,
    diffusion_loss_type: Optional[str] = None,
    diffusion_loss_reduction: Optional[str] = None,
    log_per_loss_grad_norm: bool = False,
    grad_norm_interval: int = 100,
    diff_loss_weight: float = 1.0,
    edit_diff_loss_weight: float = 1.0,
    # Deprecated compatibility argument; COT CE uses the regular CE weight.
    generation_cot_ce_loss_weight: float = 1.0,
    generation_cot_diff_loss_weight: float = 1.0,
    # Batch schedule: 确定性 round-robin 调度，按模态比例出 batch
    batch_schedule: Optional[Dict[str, int]] = None,
    batch_schedule_strict: bool = True,
    # 多流模式: 每个模态独立 pipeline，配合 batch_schedule 使用
    data_streams: Optional[Dict[str, str]] = None,
) -> ConfigContainer:
    """
    The dataset pipeline is conversation-based. To train multimodal tokens, ensure your
    preprocessed data includes placeholders (e.g., <image>) as needed.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    base_model_cfg = bridge.to_megatron_provider(load_weights=False)
    normalized_loss_type = normalize_diffusion_loss_type(diffusion_loss_type)
    normalized_loss_type = normalized_loss_type or "v-loss"
    normalized_loss_reduction = normalize_diffusion_loss_reduction(diffusion_loss_reduction)
    normalized_loss_reduction = normalized_loss_reduction or "auto"
    kwargs = {f.name: getattr(base_model_cfg, f.name) for f in dataclasses.fields(base_model_cfg)}
    kwargs.update({
        "tensor_model_parallel_size": tensor_parallelism,
        "pipeline_model_parallel_size": pipeline_parallelism,
        "pipeline_dtype": pipeline_parallelism_dtype,
        "virtual_pipeline_model_parallel_size": virtual_pipeline_parallelism,
        "context_parallel_size": context_parallelism,
        "sequence_parallel": sequence_parallelism,
        "freeze_language_model": freeze_language_model,
        "freeze_vision_model": freeze_vision_model,
        "freeze_vision_projection": freeze_vision_projection,
        "freeze_dit_param": freeze_dit_param,
        "freeze_pixel_proj_param": freeze_pixel_proj_param,
        "freeze_pixel_dit_gate_param": freeze_pixel_dit_gate_param,
        "freeze_time_embeddings_param": freeze_time_embeddings_param,
        "freeze_pixel_dit_last_n_layers": freeze_pixel_dit_last_n_layers,
        "freeze_pixel_dit_final_layer": freeze_pixel_dit_final_layer,
        "seq_length": seq_length,
        "predict_x0": bool(predict_x0),
        "diffusion_loss_type": normalized_loss_type,
        "diffusion_loss_reduction": normalized_loss_reduction,
        "vit_recompute": vit_recompute,
        "diffusion_head_recompute": diffusion_head_recompute,
        "pixel_dit_hidden_size": pixel_dit_hidden_size,
        "pixel_dit_gate_inject_layer": pixel_dit_gate_inject_layer,
        "pixel_dit_gate_input": pixel_dit_gate_input,
        "use_pixel_dit_pixel_residual": use_pixel_dit_pixel_residual,
        "pixel_dit_ref_gate_mode": pixel_dit_ref_gate_mode,
        "adaln_post_modulation": adaln_post_modulation,
        "dit_block_adaln": dit_block_adaln,
        "dit_head_impl": dit_head_impl,
        "num_diffusion_layers": num_diffusion_layers,
        "num_diffusion_heads": num_diffusion_heads,
        "num_diffusion_kv_heads": num_diffusion_kv_heads,
        "diffusion_intermediate_size": diffusion_intermediate_size,
    })

    # This distribution intentionally supports only the dense Qwen3-VL provider.
    ProviderCls = TransfusionQwen3VLProvider
    valid_fields = {f.name for f in dataclasses.fields(ProviderCls)}
    kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
    model_cfg = ProviderCls(**kwargs)

    # TE fused cross-entropy: 避免具现化 (seq_len, vocab/TP) 的巨型 logits 张量，大幅省显存
    model_cfg.cross_entropy_fusion_impl = "te"

    if expert_parallelism is not None:
        model_cfg.expert_model_parallel_size = expert_parallelism
    if expert_tensor_parallelism is not None:
        model_cfg.expert_tensor_parallel_size = expert_tensor_parallelism

    # Optimizer and scheduler
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=lr,
        min_lr=min_lr,
    )
    opt_config.vit_lr = vit_lr
    opt_config.vit_min_lr = vit_min_lr

    # ── CPU offload (HybridDeviceOptimizer): 把优化器状态(fp32 Adam, TP=1 显存大头)+梯度
    #    offload 到 CPU, GPU 只留参数+激活，减少训练峰值显存。
    #    控制流的 FSDP collective 死锁 (与 FSDP 路径正交)。仅 optimizer_cpu_offload=True 时启用。
    #    约束: 需 use_precision_aware_optimizer + decoupled_weight_decay(AdamW); 见 optimizer_config.py。
    #    最终值仍可被 yaml optimizer: 段 override。
    if optimizer_cpu_offload:
        opt_config.optimizer_cpu_offload = True
        opt_config.optimizer_offload_fraction = optimizer_offload_fraction
        opt_config.use_precision_aware_optimizer = True
        opt_config.decoupled_weight_decay = True
        opt_config.overlap_cpu_optimizer_d2h_h2d = True
        opt_config.pin_cpu_grads = True
        opt_config.pin_cpu_params = True
        # 注: fp8 + offload caveat (fp8_param_gather 与 offload 不兼容) 在 pretrain 主流程里
        #   于 mixed_precision 字符串解析后处理 (recipe 这里改 precision_config 会被 yaml 顶层
        #   mixed_precision 覆盖, 见 config.py:1496)。此处不动 precision_config。

    # ==========================================
    # 数据集 Provider 替换
    # ==========================================
    _dataset_choice = dataset_type or "energon"
    _processor_model = tokenizer_model or hf_path

    if _dataset_choice == "energon":
        tokenizer = AutoTokenizer.from_pretrained(_processor_model)
        image_processor = Qwen2VLImageProcessor.from_pretrained(_processor_model)

        # 如果不能满足需求，需要自定义TransfusionEnergonProvider
        def _make_task_encoder():
            """工厂函数：创建独立的 TaskEncoder 实例（多流模式下每条流一个）。"""
            return TransfusionTaskEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                gen_min_pixels=gen_min_pixels,
                gen_max_pixels=gen_max_pixels,
                gen_max_input_pixels=gen_max_input_pixels,
                edit_min_pixels=edit_min_pixels,
                edit_max_pixels=edit_max_pixels,
                edit_max_input_pixels=edit_max_input_pixels,
                gen_resolution_buckets=gen_resolution_buckets,
                edit_resolution_buckets=edit_resolution_buckets,
                resolution_bucket_threshold=resolution_bucket_threshold,
                und_min_pixels=und_min_pixels,
                und_max_pixels=und_max_pixels,
                cfg_drop_rate=cfg_drop_rate,
                cfg_keep_chat_template=cfg_keep_chat_template,
                packing_seq_length=seq_length,
                square_crop=square_crop,
                enable_und_bidirection=enable_und_bidirection,
                prompt_mapping_file=prompt_mapping_file,
                editing_encode_type=editing_encode_type,
                enable_think=enable_think,
                generation_empty_think=generation_empty_think,
                generation_cot_prompt_sampling_weights=generation_cot_prompt_sampling_weights,
                generation_cot_long_template_prob=generation_cot_long_template_prob,
                generation_cot_long_template_version=generation_cot_long_template_version,
                predict_gate_mode=predict_gate_mode,
            )

        if data_streams and batch_schedule:
            from data.transfusion.multi_stream_provider import (
                MultiStreamEnergonProvider,
            )
            dataset_cfg = MultiStreamEnergonProvider(
                streams=data_streams,
                schedule=batch_schedule,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                num_workers=num_workers,
                packing_buffer_size=packing_buffer_size,
                task_encoder_factory=_make_task_encoder,
                dataloader_load=dataloader_load,
                dataloader_load_iteration=dataloader_load_iteration,
                dataloader_save=dataloader_save,
                dataloader_reset_streams=dataloader_reset_streams,
            )
        else:
            dataset_cfg = EnergonProvider(
                seq_length=seq_length,
                path=(
                    train_data_path[0]
                    if isinstance(train_data_path, list)
                    else train_data_path
                ),
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                num_workers=num_workers,
                dataloader_type="external",
                task_encoder=TransfusionTaskEncoder(
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    gen_min_pixels=gen_min_pixels,
                    gen_max_pixels=gen_max_pixels,
                    gen_max_input_pixels=gen_max_input_pixels,
                    edit_min_pixels=edit_min_pixels,
                    edit_max_pixels=edit_max_pixels,
                    edit_max_input_pixels=edit_max_input_pixels,
                    gen_resolution_buckets=gen_resolution_buckets,
                    edit_resolution_buckets=edit_resolution_buckets,
                    resolution_bucket_threshold=resolution_bucket_threshold,
                    und_min_pixels=und_min_pixels,
                    und_max_pixels=und_max_pixels,
                    cfg_drop_rate=cfg_drop_rate,
                    cfg_keep_chat_template=cfg_keep_chat_template,
                    packing_seq_length=seq_length,
                    square_crop=square_crop,
                    enable_und_bidirection=enable_und_bidirection,
                    prompt_mapping_file=prompt_mapping_file,
                    editing_encode_type=editing_encode_type,
                    enable_think=enable_think,
                    generation_empty_think=generation_empty_think,
                    generation_cot_prompt_sampling_weights=generation_cot_prompt_sampling_weights,
                    generation_cot_long_template_prob=generation_cot_long_template_prob,
                    generation_cot_long_template_version=generation_cot_long_template_version,
                    predict_gate_mode=predict_gate_mode,
                    batch_schedule=batch_schedule,
                    batch_schedule_strict=batch_schedule_strict,
                ),
            )
    else:
        # TODO: 写个mock provider 用于快速调试
        raise NotImplementedError

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=check_for_nan_in_grad,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            average_in_collective=average_in_collective,
            data_parallel_sharding_strategy=data_parallel_sharding_strategy,
            use_distributed_optimizer=use_distributed_optimizer,
            use_megatron_fsdp=use_megatron_fsdp,
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        # TODO: 这行可能需要换成真正的tokenizer，方便获取词表大小
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            pretrained_checkpoint=pretrained_checkpoint,
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            # Megatron FSDP 强制 fsdp_dtensor 格式 (config 校验 + load/save 路径均要求);
            # 普通 DDP/TP 用 torch_dist。注: 切到 fsdp_dtensor 后, pretrained_checkpoint 若是
            # torch_dist 格式会读不进 (需先转格式), 见 convert_ckpt_to_fsdp_dtensor.py。
            ckpt_format="fsdp_dtensor" if use_megatron_fsdp else "torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg
