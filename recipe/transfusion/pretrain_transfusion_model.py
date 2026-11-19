#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Transfusion 架构的预训练入口脚本。
基于 Megatron Bridge 与 YAML 动态配置。
"""

import argparse
import copy
import logging
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Tuple

import torch

# 确保 multiprocessing.spawn 子进程也能找到项目模块
# (Megatron checkpoint save 使用 spawn context 创建 Manager,
#  子进程会重新导入此文件, 需要项目根目录在 sys.path 中)
_project_root = str(Path(__file__).resolve().parents[2])
# PYTHONPATH 里可能同时残留其他 checkout。即使当前根目录已经存在，也必须
# 将它提升到首位，否则会出现入口来自新 checkout、recipe 却从旧 checkout 导入。
sys.path = [path for path in sys.path if path != _project_root]
sys.path.insert(0, _project_root)

from omegaconf import OmegaConf
from megatron.core.optimizer import ParamKey

# 1. 导入基础的配方 (使用原版 Qwen 作为打底配置)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)

# 2. 💥 导入咱们辛辛苦苦写好的 "三位一体" 核心组件！
# (请根据你的实际目录结构调整 import 路径)
from data.transfusion.transfusion_task_encoder import TransfusionTaskEncoder
from model.transfusion.transfusion_qwen3vl_model import TransfusionQwen3VLModel
from model.transfusion.transfusion_step import TransfusionForwardStep
from model.transfusion.schedulers import (
    normalize_diffusion_loss_type,
    normalize_diffusion_loss_reduction,
)
from model.transfusion.offline_ema import install_offline_ema
from model.transfusion.model_only_checkpoint import (
    install_model_only_checkpointing,
)
from recipe.transfusion.transfusion_qwen3vl_model import (
    qwen3_vl_2b_finetune_config,
)

RECIPE_MAP = {
    "qwen3_vl_2b": qwen3_vl_2b_finetune_config,
}

logger = logging.getLogger(__name__)

# 指向咱们专属的 Transfusion YAML 配置文件
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "conf" / "transfusion_qwen3_vl_2b_pixel_dit_dense_ablation.yaml"


def _vit_lr_group_marker(backbone_lr: float, vit_lr: float) -> float:
    """Return an lr_mult value used only to distinguish ViT optimizer groups."""
    if backbone_lr:
        marker = float(vit_lr) / float(backbone_lr)
        if math.isfinite(marker) and marker != 1.0:
            return marker
    return math.nextafter(1.0, 2.0)


def _vit_param_group_status(param_group, vit_lr: float):
    """Classify an optimizer group as ViT, backbone, or not yet identifiable."""
    if param_group.get("default_config") is False:
        return True
    max_lr = param_group.get("max_lr")
    if param_group.get("default_config") is True and max_lr is not None:
        return False
    if max_lr is not None:
        return float(max_lr) == float(vit_lr)
    if not param_group.get("params"):
        return None
    return False


def _is_vit_param_group(param_group, vit_lr: float) -> bool:
    """Return whether an optimizer parameter group belongs to the ViT branch."""
    return _vit_param_group_status(param_group, vit_lr) is True


def _infer_local_vit_group_statuses(param_groups, vit_lr: float):
    """Infer missing ViT group flags from paired optimizer-group metadata."""
    statuses = [_vit_param_group_status(group, vit_lr) for group in param_groups]
    group_buckets = {}
    for index, group in enumerate(param_groups):
        key = (
            group.get("wd_mult", group.get("pre_wd_mult")),
            group.get("is_expert_parallel"),
            group.get("is_decoupled_lr"),
        )
        group_buckets.setdefault(key, []).append(index)

    for indices in group_buckets.values():
        if len(indices) != 2:
            continue
        known_indices = [index for index in indices if statuses[index] is not None]
        unknown_indices = [index for index in indices if statuses[index] is None]
        if len(known_indices) == 1 and len(unknown_indices) == 1:
            statuses[unknown_indices[0]] = not statuses[known_indices[0]]
    return statuses


def _synchronize_vit_group_flags(param_groups, vit_lr: float) -> list[bool]:
    """Identify empty ViT groups using metadata from ranks that own their parameters."""
    local_statuses = _infer_local_vit_group_statuses(param_groups, vit_lr)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [status is True for status in local_statuses]

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return [status is True for status in local_statuses]
    gathered_statuses = [None] * world_size
    torch.distributed.all_gather_object(gathered_statuses, local_statuses)
    if any(len(statuses) != len(local_statuses) for statuses in gathered_statuses):
        raise RuntimeError("Optimizer param groups are inconsistent across distributed ranks")

    group_flags = []
    for index, local_status in enumerate(local_statuses):
        if local_status is not None:
            group_flags.append(local_status)
            continue
        known_statuses = {
            statuses[index] for statuses in gathered_statuses if statuses[index] is not None
        }
        if len(known_statuses) > 1:
            raise RuntimeError(f"Cannot identify empty optimizer param group {index} across ranks")
        group_flags.append(known_statuses == {True})
    return group_flags


def _mark_vit_param_groups(
    param_groups,
    marker: float,
    vit_lr: float,
    group_flags=None,
) -> None:
    """Give ViT groups a unique key for Megatron's checkpoint group matching."""
    for index, param_group in enumerate(param_groups):
        is_vit_group = (
            bool(group_flags[index])
            if group_flags is not None
            else _is_vit_param_group(param_group, vit_lr)
        )
        if is_vit_group:
            param_group["lr_mult"] = marker


def _migrate_vit_lr_checkpoint_groups(
    state_dict,
    marker: float,
    vit_lr: float,
    group_flags=None,
) -> None:
    """Add the ViT group identity to checkpoints written before this fix."""
    param_group_lists = []

    def collect_param_group_lists(value):
        """Collect nested optimizer parameter-group lists from checkpoint data."""
        if isinstance(value, Mapping):
            param_groups = value.get("param_groups")
            if isinstance(param_groups, list):
                param_group_lists.append(param_groups)
            for key, nested_value in value.items():
                if key not in {"param_groups", "state"}:
                    collect_param_group_lists(nested_value)
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                collect_param_group_lists(nested_value)

    collect_param_group_lists(state_dict)
    use_group_flags = group_flags is not None and sum(map(len, param_group_lists)) == len(group_flags)
    offset = 0
    for param_groups in param_group_lists:
        flags = group_flags[offset : offset + len(param_groups)] if use_group_flags else None
        _mark_vit_param_groups(param_groups, marker, vit_lr, flags)
        offset += len(param_groups)


def _prepare_optimizer_for_vit_lr_resume(
    optimizer,
    backbone_lr: float,
    vit_lr: float,
) -> None:
    """Keep backbone and ViT groups distinct in new and legacy checkpoints."""
    marker = _vit_lr_group_marker(backbone_lr, vit_lr)
    group_flags = _synchronize_vit_group_flags(optimizer.param_groups, vit_lr)
    _mark_vit_param_groups(optimizer.param_groups, marker, vit_lr, group_flags)

    original_load_state_dict = optimizer.load_state_dict

    def load_state_dict_with_vit_lr_groups(state_dict):
        """Migrate legacy groups before delegating to the optimizer loader."""
        _migrate_vit_lr_checkpoint_groups(state_dict, marker, vit_lr, group_flags)
        return original_load_state_dict(state_dict)

    optimizer.load_state_dict = load_state_dict_with_vit_lr_groups


def _enable_vit_lr_override() -> None:
    """Patch Bridge optimizer setup to honor config.optimizer.vit_lr."""
    import megatron.bridge.training.optim as optim_mod
    import megatron.bridge.training.setup as setup_mod

    original_setup_optimizer = getattr(
        optim_mod,
        "_transfusion_original_setup_optimizer",
        optim_mod.setup_optimizer,
    )

    def setup_optimizer_with_vit_lr(
        optimizer_config,
        scheduler_config,
        model,
        use_gloo_process_groups=False,
    ):
        """Build the optimizer while preserving the configured ViT learning rate."""
        vit_lr = getattr(optimizer_config, "vit_lr", None)
        if vit_lr is None:
            return original_setup_optimizer(
                optimizer_config,
                scheduler_config,
                model,
                use_gloo_process_groups=use_gloo_process_groups,
            )

        from megatron.core.optimizer import get_megatron_optimizer
        from megatron.core.optimizer.muon import get_megatron_muon_optimizer

        vit_optimizer_config = copy.deepcopy(optimizer_config)
        vit_optimizer_config.lr = vit_lr
        vit_min_lr = getattr(optimizer_config, "vit_min_lr", None)
        if vit_min_lr is not None:
            vit_optimizer_config.min_lr = vit_min_lr
        elif optimizer_config.lr and optimizer_config.min_lr is not None:
            vit_optimizer_config.min_lr = vit_lr * (optimizer_config.min_lr / optimizer_config.lr)
        else:
            vit_optimizer_config.min_lr = optimizer_config.min_lr

        config_overrides = {ParamKey(attr="is_vit_parameter"): vit_optimizer_config}
        if "muon" not in optimizer_config.optimizer and "soap" not in optimizer_config.optimizer:
            optimizer = get_megatron_optimizer(
                optimizer_config,
                model,
                config_overrides=config_overrides,
                use_gloo_process_groups=use_gloo_process_groups,
            )
        else:
            optimizer = get_megatron_muon_optimizer(
                optimizer_config,
                model,
                config_overrides=config_overrides,
                use_gloo_process_groups=use_gloo_process_groups,
                layer_wise_distributed_optimizer="dist" in optimizer_config.optimizer,
            )
        _prepare_optimizer_for_vit_lr_resume(
            optimizer,
            backbone_lr=optimizer_config.lr,
            vit_lr=vit_lr,
        )
        scheduler = optim_mod._get_scheduler(optimizer_config, scheduler_config, optimizer)
        return optimizer, scheduler

    optim_mod._transfusion_original_setup_optimizer = original_setup_optimizer
    optim_mod.setup_optimizer = setup_optimizer_with_vit_lr
    setup_mod.setup_optimizer = setup_optimizer_with_vit_lr


def _enable_sharded_lr_logging() -> None:
    """Gather split optimizer-group LRs across DP ranks before logging."""
    import megatron.bridge.training.train as train_mod
    from megatron.bridge.training.utils.pg_utils import get_pg_collection
    from megatron.bridge.training.utils.train_utils import (
        reduce_max_stat_across_model_parallel_group,
    )

    original_training_log = getattr(
        train_mod,
        "_transfusion_original_training_log",
        train_mod.training_log,
    )

    def training_log_with_sharded_lrs(
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        vit_learning_rate,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
        config,
        global_state,
        history_wct,
        model,
        log_max_attention_logit=None,
    ):
        """Log learning rates after gathering split optimizer groups across ranks."""
        pg_collection = get_pg_collection(model)
        learning_rate = reduce_max_stat_across_model_parallel_group(
            learning_rate, mp_group=pg_collection.dp
        )
        vit_learning_rate = reduce_max_stat_across_model_parallel_group(
            vit_learning_rate, mp_group=pg_collection.dp
        )
        # A valid training setup has at least one LR group. This fallback also
        # keeps ViT-only training compatible with Bridge's mandatory base-LR log.
        if learning_rate is None:
            learning_rate = vit_learning_rate

        return original_training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            vit_learning_rate,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            config,
            global_state,
            history_wct,
            model,
            log_max_attention_logit,
        )

    train_mod._transfusion_original_training_log = original_training_log
    train_mod.training_log = training_log_with_sharded_lrs


def parse_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse the Transfusion pretraining configuration and CLI overrides."""
    parser = argparse.ArgumentParser(
        description="Pretrain Transfusion Model with YAML configuration",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE),
        help=f"Path to YAML config file. Default: {DEFAULT_CONFIG_FILE}",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--recipe",
        type=str,
        default="qwen3_vl_2b",
        choices=list(RECIPE_MAP.keys()),
        help="Model recipe to use (default: qwen3_vl_2b)",
    )
    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def main() -> None:
    """Load configuration, construct the recipe, and launch pretraining."""
    # Monkey-patch: 用 fork context Manager 代替 spawn context，
    # 避免 spawn 重新导入 __main__ 时找不到项目模块
    import multiprocessing as _mp
    from megatron.core.dist_checkpointing.strategies import filesystem_async as _fs_async
    _fork_ctx = _mp.get_context("fork")
    _manager = _fork_ctx.Manager()
    _fs_async._results_queue = _manager.Queue()

    args, cli_overrides = parse_args()

    # ==========================================
    # 步骤 1: 预加载 YAML，提取 transfusion 专属配置
    # ==========================================
    transfusion_cfg = {}
    offline_ema_cfg = {}
    model_only_checkpoint_cfg = {}
    lora_cfg = {}
    if args.config_file:
        config_file_path = Path(args.config_file)
        if not config_file_path.exists():
            logger.error(f"Config file not found: {config_file_path}")
            sys.exit(1)
        yaml_conf = OmegaConf.load(config_file_path)
        offline_ema_cfg = OmegaConf.to_container(
            yaml_conf.get("offline_ema", OmegaConf.create({})), resolve=True
        )
        model_only_checkpoint_cfg = OmegaConf.to_container(
            yaml_conf.get("model_only_checkpoint", OmegaConf.create({})),
            resolve=True,
        )
        lora_cfg = OmegaConf.to_container(yaml_conf.get("lora", OmegaConf.create({})), resolve=True)
        transfusion_cfg = OmegaConf.to_container(yaml_conf.get("transfusion", OmegaConf.create({})), resolve=True)
        # 把 dataset 中需要在构建阶段就生效的参数也提取出来
        dataset_cfg = OmegaConf.to_container(yaml_conf.get("dataset", OmegaConf.create({})), resolve=True)
        if "seq_length" in dataset_cfg:
            transfusion_cfg.setdefault("seq_length", dataset_cfg["seq_length"])
        if "dataloader_load" in dataset_cfg:
            transfusion_cfg.setdefault("dataloader_load", dataset_cfg["dataloader_load"])
        if "dataloader_load_iteration" in dataset_cfg:
            transfusion_cfg.setdefault(
                "dataloader_load_iteration", dataset_cfg["dataloader_load_iteration"]
            )
        if "dataloader_save" in dataset_cfg:
            transfusion_cfg.setdefault("dataloader_save", dataset_cfg["dataloader_save"])
        if "dataloader_reset_streams" in dataset_cfg:
            transfusion_cfg.setdefault("dataloader_reset_streams", dataset_cfg["dataloader_reset_streams"])
        # 把 train.train_iters 提取出来，用于 lr_decay_iters 默认值
        train_cfg = OmegaConf.to_container(yaml_conf.get("train", OmegaConf.create({})), resolve=True)
        if "train_iters" in train_cfg:
            transfusion_cfg.setdefault("train_iters", train_cfg["train_iters"])

    # ==========================================
    # 步骤 2: 创建基础配置，直接传入 transfusion 参数
    # ==========================================
    _forward_step_keys = (
        "diff_loss_weight",
        "edit_diff_loss_weight",
        # Kept for old YAML compatibility; the forward step ignores this
        # deprecated knob and uses the regular CE weight for COT CE.
        "generation_cot_ce_loss_weight",
        "generation_cot_diff_loss_weight",
    )
    forward_step_cfg = {
        k: transfusion_cfg.pop(k)
        for k in _forward_step_keys
        if k in transfusion_cfg
    }
    recipe_fn = RECIPE_MAP[args.recipe]
    config: ConfigContainer = recipe_fn(**transfusion_cfg)

    omega_conf, excluded_fields = create_omegaconf_dict_config(config)

    # ==========================================
    # 步骤 3: 合并 YAML 与 CLI 覆盖参数（标准 Megatron Bridge 字段）
    # ==========================================
    if args.config_file:
        yaml_conf = OmegaConf.load(config_file_path)
        # offline_ema is a Transfusion runtime extension rather than a Bridge
        # ConfigContainer field.  Parse it above and keep it out of the strict
        # structured-config merge.
        yaml_conf = OmegaConf.create(OmegaConf.to_container(yaml_conf, resolve=False))
        yaml_conf.pop("offline_ema", None)
        yaml_conf.pop("model_only_checkpoint", None)
        yaml_conf.pop("lora", None)
        omega_conf = OmegaConf.merge(omega_conf, yaml_conf)

    # `lora.<component>.<key>=value` overrides go to the lora section, not to Bridge.
    lora_overrides = [o for o in cli_overrides if o.lstrip("+").startswith("lora.")]
    cli_overrides = [o for o in cli_overrides if o not in lora_overrides]
    if lora_overrides:
        lora_conf = OmegaConf.create({"lora": lora_cfg})
        lora_conf = OmegaConf.merge(lora_conf, OmegaConf.from_dotlist([o.lstrip("+") for o in lora_overrides]))
        lora_cfg = OmegaConf.to_container(lora_conf, resolve=True)["lora"]

    if cli_overrides:
        omega_conf = parse_hydra_overrides(omega_conf, cli_overrides)

    final_config_dict = OmegaConf.to_container(omega_conf, resolve=True)
    final_transfusion_cfg = final_config_dict.get("transfusion", {})
    final_config_dict.pop("transfusion", None)  # 已在步骤2处理，移除避免 warning
    final_config_dict.pop("scheduler_config", None)
    final_config_dict.pop("sampling", None)
    apply_overrides(config, final_config_dict, excluded_fields)
    if "vit_lr" in final_transfusion_cfg:
        config.optimizer.vit_lr = final_transfusion_cfg["vit_lr"]
    if "vit_min_lr" in final_transfusion_cfg:
        config.optimizer.vit_min_lr = final_transfusion_cfg["vit_min_lr"]
    _enable_vit_lr_override()
    _enable_sharded_lr_logging()

    # ── LoRA (per component: backbone / pixel_dit / vit) ───────────────────────
    # Bridge applies config.peft after loading checkpoint.pretrained_checkpoint and
    # saves adapter-only checkpoints; merge them with scripts/merge_lora_checkpoint.py.
    if any(isinstance(v, dict) and v.get("enabled") for v in (lora_cfg or {}).values()):
        from model.transfusion.lora import TransfusionLoRA

        config.peft = TransfusionLoRA.from_dict(lora_cfg)
        if not config.checkpoint.pretrained_checkpoint:
            logger.error("lora needs checkpoint.pretrained_checkpoint (the base weights)")
            sys.exit(1)
        logger.info(f"LoRA enabled for {config.peft.enabled_components()}")

    # ── fp8 + CPU offload caveat 处理 ──────────────────────────────────────────
    # bf16_with_fp8_current_scaling_mixed 默认 fp8_param_gather=True, 与 optimizer_cpu_offload
    # 不兼容。offload 下参数全量留 GPU 不需 fp8 param gather → 解析 mixed_precision 字符串后强制
    # 关掉它, 保留 fp8 GEMM 加速。须在 framework validate/setup (config.py:1496 解析字符串) 之前做,
    # 故在此把字符串提前解析成 config 并改字段。
    if getattr(config.optimizer, "optimizer_cpu_offload", False) and config.mixed_precision is not None:
        from megatron.bridge.training.mixed_precision import get_mixed_precision_config, MixedPrecisionConfig
        if isinstance(config.mixed_precision, str):
            config.mixed_precision = get_mixed_precision_config(config.mixed_precision)
        if isinstance(config.mixed_precision, MixedPrecisionConfig) and getattr(config.mixed_precision, "fp8", None):
            config.mixed_precision.fp8_param_gather = False
            logger.info("CPU offload + fp8: 已关闭 fp8_param_gather (offload 不兼容, 参数全量留 GPU 不需要)")

    # ==========================================
    # 步骤 4: 核心注入
    # ==========================================
    if hasattr(config, "data") and hasattr(config.data, "task_encoder_cls"):
        config.data.task_encoder_cls = TransfusionTaskEncoder

    install_offline_ema(config, offline_ema_cfg)
    install_model_only_checkpointing(config, model_only_checkpoint_cfg)

    # ==========================================
    # 步骤 5: 点火发射
    # ==========================================
    logger.info("启动 Transfusion 预训练主循环...")

    # 从 YAML 提取采样配置 (可选)
    sample_cfg = {}
    scheduler_config = {}
    if args.config_file:
        yaml_conf_raw = OmegaConf.load(config_file_path)
        sample_cfg = OmegaConf.to_container(
            yaml_conf_raw.get("sampling", OmegaConf.create({})), resolve=True
        )
        scheduler_config = OmegaConf.to_container(
            yaml_conf_raw.get("scheduler_config", OmegaConf.create({})), resolve=True
        )

    configured_loss_type = final_transfusion_cfg.get(
        "diffusion_loss_type", transfusion_cfg.get("diffusion_loss_type")
    )
    configured_loss_reduction = final_transfusion_cfg.get(
        "diffusion_loss_reduction", transfusion_cfg.get("diffusion_loss_reduction")
    )
    configured_predict_x0 = final_transfusion_cfg.get(
        "predict_x0", transfusion_cfg.get("predict_x0", False)
    )
    normalized_loss_type = normalize_diffusion_loss_type(configured_loss_type) or "v-loss"
    normalized_loss_reduction = (
        normalize_diffusion_loss_reduction(configured_loss_reduction) or "auto"
    )
    logger.info(
        "Diffusion loss type: %s",
        normalized_loss_type,
    )
    logger.info("Diffusion loss reduction: %s", normalized_loss_reduction)
    forward_step_instance = TransfusionForwardStep(
        sample_cfg=sample_cfg,
        scheduler_config=scheduler_config,
        predict_x0=bool(configured_predict_x0),
        diffusion_loss_type=normalized_loss_type,
        diffusion_loss_reduction=normalized_loss_reduction,
        log_per_loss_grad_norm=final_transfusion_cfg.get(
            "log_per_loss_grad_norm",
            transfusion_cfg.get("log_per_loss_grad_norm", False),
        ),
        grad_norm_interval=final_transfusion_cfg.get(
            "grad_norm_interval",
            transfusion_cfg.get("grad_norm_interval", 100),
        ),
        diff_loss_weight=final_transfusion_cfg.get(
            "diff_loss_weight",
            forward_step_cfg.get("diff_loss_weight", 1.0),
        ),
        edit_diff_loss_weight=final_transfusion_cfg.get(
            "edit_diff_loss_weight",
            forward_step_cfg.get("edit_diff_loss_weight", 1.0),
        ),
        # Deprecated compatibility argument; COT CE is merged into regular CE.
        generation_cot_ce_loss_weight=final_transfusion_cfg.get(
            "generation_cot_ce_loss_weight",
            forward_step_cfg.get("generation_cot_ce_loss_weight", 1.0),
        ),
        generation_cot_diff_loss_weight=final_transfusion_cfg.get(
            "generation_cot_diff_loss_weight",
            forward_step_cfg.get("generation_cot_diff_loss_weight", 1.0),
        ),
    )
    pretrain(
        config=config,
        forward_step_func=forward_step_instance
    )


if __name__ == "__main__":
    main()
