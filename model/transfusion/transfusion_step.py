"""Training-step orchestration and loss aggregation for Transfusion models."""

import logging
import re
from functools import partial
from typing import Dict, Iterable, Callable, Optional, List, Tuple

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.bridge.training.state import GlobalState

from model.transfusion.schedulers import (
    diffusion_parameterization_weight,
    FlowMatchScheduler,
    normalize_diffusion_loss_type,
    normalize_diffusion_loss_reduction,
)
from model.transfusion.schedulers import compute_dynamic_noise_scale
from model.transfusion.transfusion_qwen3vl_model import TransfusionQwen3VLModel


logger = logging.getLogger(__name__)

TASK_UNDERSTANDING = 0
TASK_GENERATION = 1
TASK_EDITING = 2
TASK_TEXT = 3
TASK_MESSAGE = 4
TASK_SFT_TEXT = 5
TASK_SFT_IMAGE_UND = 6
TASK_GENERATION_COT = 7
TASK_GENERATION_COT_TEXT = 8


def _split_ce_loss_by_task(
    text_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    task_type: torch.Tensor,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Split token-level CE sums/counts by task without detaching gradients."""
    if text_loss.shape != loss_mask.shape:
        raise ValueError(
            "text_loss and loss_mask must have the same shape: "
            f"{tuple(text_loss.shape)} != {tuple(loss_mask.shape)}"
        )
    if text_loss.dim() != 2 or task_type.numel() != text_loss.shape[0]:
        raise ValueError(
            "expected [B, S] CE tensors and one task id per row: "
            f"text_loss={tuple(text_loss.shape)}, task_type={tuple(task_type.shape)}"
        )

    task_groups = {
        # Ordinary generation/edit rows only expose their route gate in
        # loss_mask, so their gate CE belongs to the regular text bucket.
        # generation_cot is aggregated into that same text CE by the caller;
        # keep a separate diagnostic bucket below for per-task reporting.
        "text": (
            TASK_TEXT,
            TASK_MESSAGE,
            TASK_SFT_TEXT,
            TASK_GENERATION,
            TASK_EDITING,
        ),
        "understanding": (TASK_UNDERSTANDING, TASK_SFT_IMAGE_UND),
        "generation_cot": (TASK_GENERATION_COT,),
        "generation_cot_text": (TASK_GENERATION_COT_TEXT,),
    }
    result = {}
    for name, task_ids in task_groups.items():
        row_mask = torch.zeros_like(task_type, dtype=torch.bool)
        for task_id in task_ids:
            row_mask |= task_type == task_id
        token_mask = loss_mask * row_mask.unsqueeze(-1).to(
            device=loss_mask.device, dtype=loss_mask.dtype
        )
        result[name] = ((text_loss * token_mask).sum(), token_mask.sum())
    return result


def _split_diffusion_token_counts_by_task(
    valid_token_mask: torch.Tensor,
    task_type: torch.Tensor,
    num_gen_images_per_sample: torch.Tensor,
    image_gen_seq_lens: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Count valid packed diffusion tokens for each detailed task."""
    valid_token_mask = torch.as_tensor(valid_token_mask, dtype=torch.bool).reshape(-1)
    task_type = torch.as_tensor(task_type).reshape(-1)
    num_gen_images_per_sample = torch.as_tensor(
        num_gen_images_per_sample, dtype=torch.long
    ).reshape(-1)
    image_gen_seq_lens = torch.as_tensor(
        image_gen_seq_lens, dtype=torch.long
    ).reshape(-1)
    if task_type.numel() != num_gen_images_per_sample.numel():
        raise ValueError("task_type and num_gen_images_per_sample must have the same length")
    if (num_gen_images_per_sample < 0).any() or (image_gen_seq_lens < 0).any():
        raise ValueError("image counts and sequence lengths cannot be negative")
    if int(num_gen_images_per_sample.sum().item()) != image_gen_seq_lens.numel():
        raise ValueError(
            "num_gen_images_per_sample does not cover generated images: "
            f"{int(num_gen_images_per_sample.sum().item())} != {image_gen_seq_lens.numel()}"
        )
    if int(image_gen_seq_lens.sum().item()) != valid_token_mask.numel():
        raise ValueError(
            "image_gen_seq_lens does not cover valid diffusion tokens: "
            f"{int(image_gen_seq_lens.sum().item())} != {valid_token_mask.numel()}"
        )

    counts = {
        "generation": 0,
        "generation_cot": 0,
        "editing": 0,
    }
    image_offset = 0
    token_offset = 0
    for sample_index, task_id in enumerate(task_type.tolist()):
        image_count = int(num_gen_images_per_sample[sample_index].item())
        sample_token_count = int(
            image_gen_seq_lens[image_offset:image_offset + image_count].sum().item()
        )
        sample_valid_count = int(
            valid_token_mask[token_offset:token_offset + sample_token_count].sum().item()
        )
        if task_id == TASK_GENERATION:
            counts["generation"] += sample_valid_count
        elif task_id == TASK_GENERATION_COT:
            counts["generation_cot"] += sample_valid_count
        elif task_id == TASK_EDITING:
            counts["editing"] += sample_valid_count
        image_offset += image_count
        token_offset += sample_token_count

    device = valid_token_mask.device
    result = {
        name: torch.tensor(float(count), dtype=torch.float32, device=device)
        for name, count in counts.items()
    }
    result["t2i"] = result["generation"] + result["generation_cot"]
    result["total"] = result["t2i"] + result["editing"]
    return result


def _split_diff_loss_by_task(
    diff_loss_per_token: torch.Tensor,
    task_type: torch.Tensor,
    num_gen_images_per_sample: torch.Tensor,
    image_gen_seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed diffusion tokens into T2I-like and editing losses."""
    zero_loss = diff_loss_per_token.sum() * 0.0
    t2i_diff_loss_sum = zero_loss
    edit_diff_loss_sum = zero_loss
    t2i_count = 0
    edit_count = 0
    image_idx = 0
    token_offset = 0

    for sample_idx in range(task_type.shape[0]):
        num_images = int(num_gen_images_per_sample[sample_idx].item())
        num_tokens = 0
        if num_images > 0:
            num_tokens = int(
                image_gen_seq_lens[image_idx:image_idx + num_images].sum().item()
            )
        image_idx += num_images

        if num_tokens == 0:
            continue

        sample_loss = diff_loss_per_token[
            token_offset:token_offset + num_tokens
        ].sum()
        sample_task = int(task_type[sample_idx].item())
        if sample_task == TASK_EDITING:
            edit_diff_loss_sum = edit_diff_loss_sum + sample_loss
            edit_count += num_tokens
        elif sample_task in (TASK_GENERATION, TASK_GENERATION_COT):
            t2i_diff_loss_sum = t2i_diff_loss_sum + sample_loss
            t2i_count += num_tokens

        # Understanding rows can carry dummy diffusion tokens in Pixel-DiT mode.
        # They are skipped above, but still occupy their segment in the flat tensor.
        token_offset += num_tokens

    device = diff_loss_per_token.device
    return (
        t2i_diff_loss_sum,
        torch.tensor(float(t2i_count), device=device),
        edit_diff_loss_sum,
        torch.tensor(float(edit_count), device=device),
    )


def _split_diff_loss_by_detailed_task(
    diff_loss_units: torch.Tensor,
    task_type: torch.Tensor,
    units_per_sample: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed diffusion loss units into generation, COT-generation and editing.

    ``diff_loss_units`` may contain individual valid tokens or one mean per
    image. ``units_per_sample`` describes the corresponding packed segment for
    each sample, keeping task metrics consistent with the selected reduction.
    """
    if task_type.numel() != units_per_sample.numel():
        raise ValueError(
            "task_type and units_per_sample must have the same length: "
            f"{task_type.numel()} != {units_per_sample.numel()}"
        )
    zero = diff_loss_units.sum() * 0.0
    generation_sum = zero
    cot_sum = zero
    edit_sum = zero
    generation_count = cot_count = edit_count = 0
    unit_offset = 0
    for sample_index in range(task_type.shape[0]):
        unit_count = int(units_per_sample.reshape(-1)[sample_index].item())
        if unit_count < 0:
            raise ValueError(f"units_per_sample cannot be negative: {unit_count}")
        next_offset = unit_offset + unit_count
        if next_offset > diff_loss_units.numel():
            raise ValueError(
                "units_per_sample exceeds diffusion loss unit count: "
                f"{next_offset} > {diff_loss_units.numel()}"
            )
        sample_loss = diff_loss_units[unit_offset:next_offset].sum()
        task_id = int(task_type[sample_index].item())
        if task_id == TASK_GENERATION:
            generation_sum = generation_sum + sample_loss
            generation_count += unit_count
        elif task_id == TASK_GENERATION_COT:
            cot_sum = cot_sum + sample_loss
            cot_count += unit_count
        elif task_id == TASK_EDITING:
            edit_sum = edit_sum + sample_loss
            edit_count += unit_count
        unit_offset = next_offset
    if unit_offset != diff_loss_units.numel():
        raise ValueError(
            "units_per_sample does not cover diffusion loss unit count: "
            f"{unit_offset} != {diff_loss_units.numel()}"
        )
    device = diff_loss_units.device
    return (
        generation_sum,
        torch.tensor(float(generation_count), device=device),
        cot_sum,
        torch.tensor(float(cot_count), device=device),
        edit_sum,
        torch.tensor(float(edit_count), device=device),
    )


def _reduce_diffusion_loss(
    diff_loss_per_token: torch.Tensor,
    image_gen_seq_lens: torch.Tensor,
    valid_token_mask: torch.Tensor,
    reduction: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce packed diffusion losses into token- or image-level units.

    Returns ``(loss_units, units_per_image, loss_sum, loss_count)``.  Invalid
    images contribute zero units and are omitted from ``loss_units``.
    """
    normalized = normalize_diffusion_loss_reduction(reduction) or "per_token"
    if normalized == "auto":
        raise ValueError("_reduce_diffusion_loss requires a resolved reduction, not 'auto'")
    if diff_loss_per_token.dim() != 1:
        raise ValueError(
            "diff_loss_per_token must be one-dimensional, got "
            f"{tuple(diff_loss_per_token.shape)}"
        )
    lengths = torch.as_tensor(
        image_gen_seq_lens, device=diff_loss_per_token.device, dtype=torch.long
    ).reshape(-1)
    if (lengths < 0).any():
        raise ValueError("image_gen_seq_lens cannot contain negative values")
    total_tokens = int(lengths.sum().item())
    if total_tokens != diff_loss_per_token.numel():
        raise ValueError(
            "image_gen_seq_lens does not cover diffusion loss tokens: "
            f"{total_tokens} != {diff_loss_per_token.numel()}"
        )
    token_mask = torch.as_tensor(
        valid_token_mask, device=diff_loss_per_token.device, dtype=torch.bool
    ).reshape(-1)
    if token_mask.numel() != diff_loss_per_token.numel():
        raise ValueError(
            "valid_token_mask must match diffusion loss tokens: "
            f"{token_mask.numel()} != {diff_loss_per_token.numel()}"
        )

    if normalized == "per_token":
        loss_units = diff_loss_per_token[token_mask]
        if lengths.numel() > 0:
            units_per_image = torch.stack(
                [mask.sum() for mask in torch.split(token_mask, lengths.tolist())]
            ).to(dtype=torch.long)
        else:
            units_per_image = torch.zeros(
                0, device=diff_loss_per_token.device, dtype=torch.long
            )
    else:
        image_losses = []
        image_unit_counts = []
        for image_loss, image_mask in zip(
            torch.split(diff_loss_per_token, lengths.tolist()),
            torch.split(token_mask, lengths.tolist()),
        ):
            token_count = image_mask.sum()
            if int(token_count.item()) > 0:
                image_losses.append(image_loss[image_mask].mean())
                image_unit_counts.append(1)
            else:
                image_unit_counts.append(0)
        loss_units = (
            torch.stack(image_losses)
            if image_losses
            else diff_loss_per_token[:0]
        )
        units_per_image = torch.tensor(
            image_unit_counts, device=diff_loss_per_token.device, dtype=torch.long
        )

    loss_sum = loss_units.sum()
    # Keep counts in fp32 even when the model loss is bf16/fp16.
    loss_count = units_per_image.sum().to(dtype=torch.float32)
    return loss_units, units_per_image, loss_sum, loss_count


def _build_loss_metric_pair(
    loss_sum: torch.Tensor,
    loss_count: torch.Tensor,
    count_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Build a sum/count metric that remains finite when a task is absent."""
    loss_sum = torch.as_tensor(loss_sum, dtype=torch.float)
    loss_count = torch.as_tensor(
        loss_count, dtype=torch.float, device=loss_sum.device
    )
    return torch.stack((loss_sum, loss_count + count_epsilon))


def _diffusion_mse_loss(
    image_pred: torch.Tensor,
    image_target: torch.Tensor,
) -> torch.Tensor:
    """Compute diffusion MSE after aligning target dtype with the prediction."""
    if image_target.dtype != image_pred.dtype:
        image_target = image_target.to(dtype=image_pred.dtype)
    return F.mse_loss(image_pred, image_target, reduction="none")


class MultiLossTensors(list):
    """list 子类，用于在 multi_loss_tensors 上附加额外属性。

    backward_step 收到 output_tensor 后可以通过 isinstance 判断是否为多模态模式，
    并读取附加的模型引用、grad norm tensor 引用等信息。
    """
    pass


class TransfusionPipeline:
    """
    TransfusionPipeline
    负责生成时间步和纯噪声，将加噪任务下放给底层模型，并接收联合 Loss。
    """

    def __init__(
        self,
        scheduler_config: Optional[Dict] = None,
        seed=1234,
        time_token_id: int = -1,
        ce_loss_weight: float = 1.0,
        # Deprecated compatibility argument. COT CE now uses ce_loss_weight.
        generation_cot_ce_loss_weight: float = 1.0,
        diff_loss_weight: float = 1.0,
        generation_cot_diff_loss_weight: float = 1.0,
        edit_diff_loss_weight: float = 1.0,
        predict_x0: bool = False,
        diffusion_loss_type: Optional[str] = None,
        log_per_loss_grad_norm: bool = False,
        grad_norm_interval: int = 100,
        diffusion_loss_reduction: Optional[str] = None,
    ):
        """
        Initializes the Flow Matching pipeline.
        """
        self.seed = seed
        self._noise_generator = None
        self._log_counter = 0

        # 初始化调度器，用于管理时间步和插值逻辑
        # 训练 scheduler 必须与 HiUni 保持一致:
        #   shift=1.0 (均匀覆盖所有噪声等级), extra_one_step=True
        #   num_inference_steps=1000 (训练用 1000 步细网格)
        scheduler_config = scheduler_config or {}
        scheduler_config.setdefault("num_inference_steps", 1000)
        scheduler_config.setdefault("shift", 1.0)
        scheduler_config.setdefault("extra_one_step", True)
        self.scheduler = FlowMatchScheduler(**scheduler_config)

        # 核心参数注册
        self.time_token_id = time_token_id
        self.ce_loss_weight = ce_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.generation_cot_diff_loss_weight = float(generation_cot_diff_loss_weight)
        self.edit_diff_loss_weight = edit_diff_loss_weight
        self.predict_x0 = bool(predict_x0)
        self.diffusion_loss_type = (
            normalize_diffusion_loss_type(diffusion_loss_type) or "v-loss"
        )
        self.diffusion_loss_reduction = (
            normalize_diffusion_loss_reduction(diffusion_loss_reduction) or "auto"
        )
        self.log_per_loss_grad_norm = log_per_loss_grad_norm
        self.grad_norm_interval = grad_norm_interval

    def _initialize_generators(self, rank_offset: int = 0):
        """
        Initializes the random number generators for noise generation.
        Ensures proper seeding across different data parallel ranks.
        """
        noise_seed = self.seed + 100 * rank_offset
        current_device = torch.cuda.current_device()
        self._noise_generator = torch.Generator(device=current_device)
        self._noise_generator.manual_seed(noise_seed)

    # ------------------------------------------------------------------
    # Post-backward CE replay: 计算 per-layer CE vs Diff 梯度符号一致率 + grad norm
    # ------------------------------------------------------------------
    # 原理:
    #   training backward → main_grad = total_grad (CE + diff)
    #   clone total_grad (bf16) → zero main_grad → replay CE-only forward+backward
    #   → main_grad = ce_grad → diff_grad = total_grad - ce_grad
    #   → per-layer sign agreement, grad norms, inter-mbs sign agreement
    #   → restore main_grad = total_grad
    # ------------------------------------------------------------------

    def _save_replay_data(self, data_batch: dict, extra: dict = None):
        """保存当前 microbatch 的输入数据 (detached) 供 CE-only replay 使用。"""
        if not hasattr(self, '_replay_data_list'):
            self._replay_data_list = []
        replay = {}
        for key in [
            "input_ids", "labels", "attention_mask", "position_ids",
            "startend_row_indices", "pixel_values", "image_grid_thw",
            "image_input_mask", "image_generate_mask", "image_understand_mask",
            "total_img_und_mask", "image_gen_seq_lens", "image_gen_grid_thw",
            "task_type",
            "num_gen_images_per_sample",
            "loss_mask",
        ]:
            val = data_batch.get(key)
            if val is not None:
                replay[key] = val.detach() if isinstance(val, torch.Tensor) else val
        # 附加训练时计算的 timesteps/noise 等，确保 replay forward 与训练 forward 一致
        if extra:
            for key, val in extra.items():
                if val is not None:
                    replay[key] = val.detach() if isinstance(val, torch.Tensor) else val
        self._replay_data_list.append(replay)

    def _collect_replay_data(self):
        """收集并清空所有 microbatch 的 replay 数据。"""
        data = getattr(self, '_replay_data_list', [])
        self._replay_data_list = []
        return data

    @staticmethod
    def _post_backward_grad_analysis(output_tensor):
        """在所有 microbatch backward 完成后调用，计算 per-layer CE vs Diff 梯度符号一致率。

        此方法作为 callback 附在 MultiLossTensors 上，由 backward_step 在最后一个
        microbatch 的 backward 结束后调用。

        步骤:
          1. clone ALL 参数的 total_grad (bf16)，同时记录 backbone 层归属
          2. zero ALL main_grad
          3. 对每个 microbatch 做 CE-only replay forward+backward (累积 ce_grad)
          4. per-layer: diff_grad = total_grad - ce_grad, sign agreement + grad norm
          5. restore ALL main_grad = total_grad
        """
        model_ref = output_tensor.model_ref
        replay_data_list = getattr(output_tensor, '_replay_data_list', [])
        ce_loss_weight = getattr(output_tensor, '_ce_loss_weight', 1.0)
        is_last_microbatch = getattr(output_tensor, 'is_last_microbatch', True)
        pipeline_ref = getattr(output_tensor, '_pipeline_ref', None)
        total_n_mbs = get_num_microbatches()  # Megatron 级别的总 microbatch 数
        compute_inter_mbs = total_n_mbs > 1 and pipeline_ref is not None

        if not replay_data_list:
            return

        # 解包 DDP / Float16Module
        unwrapped = model_ref
        while hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module

        # ---- 1. Clone ALL 参数的 total_grad (bf16)，标记 backbone 层归属 ----
        layer_pattern = re.compile(r'language_model\.decoder\.layers\.(\d+)\.')
        # all_grads: [(name, total_grad_bf16, param_ref, layer_idx_or_None), ...]
        all_grads = []
        # layer_params: {layer_idx: [index_in_all_grads, ...]}
        layer_params = {}

        for name, p in unwrapped.named_parameters():
            if not p.requires_grad:
                continue
            g = getattr(p, 'main_grad', None)
            if g is None:
                g = p.grad
            if g is None:
                continue
            total_g_bf16 = g.detach().to(torch.bfloat16).clone()
            m = layer_pattern.search(name)
            layer_idx = int(m.group(1)) if m else None
            idx = len(all_grads)
            all_grads.append((name, total_g_bf16, p, layer_idx))
            if layer_idx is not None:
                if layer_idx not in layer_params:
                    layer_params[layer_idx] = []
                layer_params[layer_idx].append(idx)

        # head_groups: 非 backbone 参数按模块分组
        # e.g. "output_layer", "image_diffusion_head", "time_embeddings", "vision_model"
        head_groups = {}  # group_name -> [index_in_all_grads]
        for idx, (name, total_g_bf16, p, layer_idx) in enumerate(all_grads):
            if layer_idx is not None:
                continue  # backbone layer, already in layer_params
            parts = name.split('.')
            if parts[0] == 'language_model':
                group = parts[1] if len(parts) > 1 and parts[1] != 'decoder' else None
            else:
                group = parts[0]
            if group is None:
                continue
            if group not in head_groups:
                head_groups[group] = []
            head_groups[group].append(idx)

        if not layer_params and not head_groups:
            # 没有可分析的参数，跳过
            return

        # ---- 2. Zero ALL main_grad ----
        for name, total_g_bf16, p, _ in all_grads:
            g = getattr(p, 'main_grad', None)
            if g is not None:
                g.zero_()
            elif p.grad is not None:
                p.grad.zero_()

        # ---- 3. CE-only replay forward+backward (每个 microbatch) ----
        dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        world_size = torch.distributed.get_world_size(dp_group)
        grad_scale_func = getattr(output_tensor, '_grad_scale_func', None)

        # 获取 DDP no_sync 上下文: 防止 replay backward 触发 DDP 梯度同步
        import contextlib
        no_sync_ctx = getattr(model_ref, 'no_sync', contextlib.nullcontext)

        # ---- 3a. Inter-mbs tracking: 不同 microbatch CE 梯度的符号一致率 ----
        # 在 1F1B 调度下，每个 microbatch 的 callback 只处理 1 条 replay data,
        # 因此将 param_sign_acc 挂在 pipeline_ref 上跨 microbatch 持久累积。
        # S_j = Σ_k sign(ce_grad_k_j), rate = ((N²-2N)*M + Σ S_j²) / (2*N*(N-1)*M)
        # baseline=0.5

        with no_sync_ctx():
            for mb_idx, mb_data in enumerate(replay_data_list):
                # 完整 forward (与训练时完全一致的输入，包括 timesteps/noise)
                # 但只 backward 通过 CE loss，不 backward diff loss
                with torch.enable_grad():
                    output = unwrapped(
                        input_ids=mb_data["input_ids"],
                        position_ids=mb_data.get("position_ids"),
                        labels=mb_data["labels"],
                        attention_mask=mb_data.get("attention_mask"),
                        extra_block_kwargs={
                            "startend_row_indices": mb_data.get("startend_row_indices"),
                            "causal": False,
                        } if mb_data.get("startend_row_indices") is not None else None,
                        pixel_values=mb_data.get("pixel_values"),
                        image_grid_thw=mb_data.get("image_grid_thw"),
                        image_input_mask=mb_data.get("image_input_mask"),
                        image_generate_mask=mb_data.get("image_generate_mask"),
                        image_understand_mask=mb_data.get("image_understand_mask"),
                        image_gen_grid_thw=mb_data.get("image_gen_grid_thw"),
                        total_img_und_mask=mb_data.get("total_img_und_mask"),
                        timesteps_samples=mb_data.get("timesteps_samples"),
                        timesteps=mb_data.get("timesteps_packed"),
                        noise=mb_data.get("noise"),
                        diff_scheduler=getattr(
                            getattr(output_tensor, '_pipeline_ref', None),
                            'scheduler', None),
                        task_type=mb_data.get("task_type"),
                        predict_x0=getattr(output_tensor, '_predict_x0', False),
                    )

                text_loss = output.get("text_loss")
                if text_loss is None or (isinstance(text_loss, (int, float)) and text_loss == 0.0):
                    continue

                loss_mask = mb_data.get("loss_mask")
                if loss_mask is not None and text_loss.dim() == 2:
                    ce_loss_sum = (text_loss * loss_mask).sum()
                    ce_count = loss_mask.sum()
                else:
                    ce_loss_sum = text_loss.sum() if text_loss.dim() >= 1 else text_loss
                    ce_count = torch.tensor(text_loss.numel(), dtype=torch.float, device="cuda")

                # all_reduce count (与训练 backward 保持一致的 scale)
                global_count = torch.tensor([ce_count.item()], dtype=torch.float, device=ce_loss_sum.device)
                torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM, group=dp_group)
                if global_count.item() > 0:
                    scaled_ce_loss = ce_loss_weight * (1.0 / global_count) * ce_loss_sum * world_size
                    # 应用与训练 backward 相同的 AMP scaler，确保 replay grad 与 total grad scale 一致
                    if grad_scale_func is not None:
                        scaled_ce_loss = grad_scale_func(scaled_ce_loss)
                    torch.autograd.backward(scaled_ce_loss)

                # ---- 3b. 累积本 microbatch CE / Diff 梯度的 sign 到持久 buffer ----
                # 1F1B 下每次 replay 只有 1 条数据，ce_grad 即本 microbatch 的 CE 梯度
                # diff_grad = total_grad - ce_grad
                if compute_inter_mbs:
                    if not hasattr(pipeline_ref, '_param_sign_acc'):
                        pipeline_ref._param_sign_acc = {}
                    if not hasattr(pipeline_ref, '_param_sign_acc_diff'):
                        pipeline_ref._param_sign_acc_diff = {}
                    for idx, (name, total_g_bf16_item, p, _) in enumerate(all_grads):
                        g = getattr(p, 'main_grad', p.grad)
                        if g is None:
                            continue
                        ce_grad = g.detach().flatten().float()
                        cur_sign_ce = torch.sign(ce_grad).to(torch.int8)
                        if name in pipeline_ref._param_sign_acc:
                            pipeline_ref._param_sign_acc[name].add_(cur_sign_ce)
                        else:
                            pipeline_ref._param_sign_acc[name] = cur_sign_ce
                        # diff sign
                        diff_grad = total_g_bf16_item.flatten().float() - ce_grad
                        cur_sign_diff = torch.sign(diff_grad).to(torch.int8)
                        if name in pipeline_ref._param_sign_acc_diff:
                            pipeline_ref._param_sign_acc_diff[name].add_(cur_sign_diff)
                        else:
                            pipeline_ref._param_sign_acc_diff[name] = cur_sign_diff

        # ---- 非最后 microbatch: 累积 per-group 统计量，恢复梯度后返回 ----
        if not is_last_microbatch:
            if pipeline_ref is not None:
                # 累积 per-group CE vs Diff 统计量 (只存标量，不存全量梯度)
                if not hasattr(pipeline_ref, '_acc_group_stats'):
                    pipeline_ref._acc_group_stats = {}  # key -> [sign_agree, elems, ce_sq, diff_sq]
                for idx, (name, total_g_bf16, p, layer_idx) in enumerate(all_grads):
                    ce_g = getattr(p, 'main_grad', p.grad)
                    if ce_g is None:
                        continue
                    ce_flat = ce_g.detach().flatten().float()
                    total_flat = total_g_bf16.flatten().float()
                    diff_flat = total_flat - ce_flat
                    sa = (torch.sign(ce_flat) == torch.sign(diff_flat)).sum().item()
                    ne = ce_flat.numel()
                    cs = torch.dot(ce_flat, ce_flat).item()
                    ds = torch.dot(diff_flat, diff_flat).item()
                    if idx not in pipeline_ref._acc_group_stats:
                        pipeline_ref._acc_group_stats[idx] = [sa, ne, cs, ds]
                    else:
                        s = pipeline_ref._acc_group_stats[idx]
                        s[0] += sa; s[1] += ne; s[2] += cs; s[3] += ds
            # 恢复 main_grad = total_grad
            for name, total_g_bf16, p, _ in all_grads:
                g = getattr(p, 'main_grad', None)
                if g is not None:
                    g.copy_(total_g_bf16)
                elif p.grad is not None:
                    p.grad.copy_(total_g_bf16)
            return

        # ---- 最后 microbatch: 取出之前累积的 per-group stats ----
        _acc_stats = getattr(pipeline_ref, '_acc_group_stats', None) if pipeline_ref else None

        # ---- 从 pipeline_ref 取出累积的 param_sign_acc (CE & Diff) ----
        param_sign_acc = getattr(pipeline_ref, '_param_sign_acc', {}) if compute_inter_mbs else {}
        param_sign_acc_diff = getattr(pipeline_ref, '_param_sign_acc_diff', {}) if compute_inter_mbs else {}
        n_mbs = total_n_mbs  # 用于 inter-mbs sign agree 公式

        # ---- 4. 计算 per-group sign agree + ce/diff grad norm ----
        # groups = backbone layers (keyed by "layer_N") + head groups (keyed by group name)
        # 在 TP > 1 时，先收集 raw dot/norm^2 再 all_reduce
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        # 统一收集: group_key -> (sign_agree, total_elements, ce_sq, diff_sq)
        # 包含当前 microbatch (最后一个) + 之前累积的 per-group stats
        def _compute_group_stats(indices):
            """Aggregate gradient sign agreement and norm statistics for indices."""
            sign_agree_count, total_elements = 0, 0
            ce_sq_val, diff_sq_val = 0.0, 0.0
            for idx in indices:
                name, total_g_bf16, p, _ = all_grads[idx]
                ce_g = getattr(p, 'main_grad', None)
                if ce_g is None:
                    ce_g = p.grad
                if ce_g is None:
                    continue
                ce_flat = ce_g.detach().flatten().float()
                total_flat = total_g_bf16.flatten().float()
                diff_flat = total_flat - ce_flat
                sign_agree_count += (torch.sign(ce_flat) == torch.sign(diff_flat)).sum().item()
                total_elements += ce_flat.numel()
                ce_sq_val += torch.dot(ce_flat, ce_flat).item()
                diff_sq_val += torch.dot(diff_flat, diff_flat).item()
                # 加上之前 microbatch 累积的 stats
                if _acc_stats and idx in _acc_stats:
                    s = _acc_stats[idx]
                    sign_agree_count += s[0]
                    total_elements += s[1]
                    ce_sq_val += s[2]
                    diff_sq_val += s[3]
            return sign_agree_count, total_elements, ce_sq_val, diff_sq_val

        # 按固定顺序排列所有 group: backbone layers first, then head groups
        ordered_keys = [f"layer_{i}" for i in sorted(layer_params.keys())]
        head_keys = sorted(head_groups.keys())
        ordered_keys.extend(head_keys)

        # buf layout per group: [sign_agree, total_elems, ce_sq, diff_sq, S_sq_sum_ce, S_sq_sum_diff]
        # S_sq_sum = Σ_j S_j² where S_j = Σ_k sign(g_k_j) for inter-mbs sign agree
        stride = 6 if compute_inter_mbs else 4
        buf = torch.zeros(len(ordered_keys) * stride, device='cuda')
        for ki, key in enumerate(ordered_keys):
            if key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                indices = layer_params[layer_idx]
            else:
                indices = head_groups[key]
            sign_agree, total_elems, ce_sq_val, diff_sq_val = _compute_group_stats(indices)
            buf[ki * stride] = sign_agree
            buf[ki * stride + 1] = total_elems
            buf[ki * stride + 2] = ce_sq_val
            buf[ki * stride + 3] = diff_sq_val
            # 聚合 per-param inter-mbs sign acc 到 per-group: Σ_j S_j² (CE & Diff)
            if compute_inter_mbs:
                grp_S_sq_ce = 0.0
                grp_S_sq_diff = 0.0
                for idx in indices:
                    name = all_grads[idx][0]
                    if name in param_sign_acc:
                        grp_S_sq_ce += (param_sign_acc[name].float() ** 2).sum().item()
                    if name in param_sign_acc_diff:
                        grp_S_sq_diff += (param_sign_acc_diff[name].float() ** 2).sum().item()
                buf[ki * stride + 4] = grp_S_sq_ce
                buf[ki * stride + 5] = grp_S_sq_diff

        if tp_size > 1:
            torch.distributed.all_reduce(buf, group=tp_group)

        per_layer_sign_agree = {}  # layer_idx -> sign agreement rate (0~1, baseline=0.5)
        per_group_ce_norm = {}   # group_key -> ce_grad_norm
        per_group_diff_norm = {} # group_key -> diff_grad_norm
        per_group_inter_mbs_sign = {}  # group_key -> inter-mbs avg pairwise sign agree rate (CE)
        per_group_inter_mbs_sign_diff = {}  # group_key -> inter-mbs avg pairwise sign agree rate (Diff)
        ce_norm_sq_total = 0.0
        diff_norm_sq_total = 0.0
        for ki, key in enumerate(ordered_keys):
            sign_agree = buf[ki * stride].item()
            total_elems = buf[ki * stride + 1].item()
            ce_sq_val = buf[ki * stride + 2].item()
            diff_sq_val = buf[ki * stride + 3].item()
            ce_norm_sq_total += ce_sq_val
            diff_norm_sq_total += diff_sq_val
            per_group_ce_norm[key] = ce_sq_val ** 0.5
            per_group_diff_norm[key] = diff_sq_val ** 0.5
            # CE vs Diff 梯度符号一致率 (0.5=随机, >0.5=同向, <0.5=反向)
            if key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                per_layer_sign_agree[layer_idx] = sign_agree / total_elems if total_elems > 0 else 0.5
            # inter-mbs CE / Diff 梯度符号一致率
            # rate = ((N²-2N)*M + Σ_j S_j²) / (2*N*(N-1)*M)
            # S_j = Σ_k sign(g_k_j), M=total_elems, N=n_mbs
            # = 0.5 表示随机, >0.5 表示一致, <0.5 表示冲突
            if compute_inter_mbs:
                N = n_mbs
                M = total_elems
                denom = 2.0 * N * (N - 1) * M
                # CE
                S_sq_sum_ce = buf[ki * stride + 4].item()
                if denom > 0:
                    per_group_inter_mbs_sign[key] = ((N * N - 2 * N) * M + S_sq_sum_ce) / denom
                else:
                    per_group_inter_mbs_sign[key] = 0.5
                # Diff
                S_sq_sum_diff = buf[ki * stride + 5].item()
                if denom > 0:
                    per_group_inter_mbs_sign_diff[key] = ((N * N - 2 * N) * M + S_sq_sum_diff) / denom
                else:
                    per_group_inter_mbs_sign_diff[key] = 0.5
        # 释放 param_sign_acc / param_sign_acc_diff 和 acc_group_stats 内存
        if compute_inter_mbs and hasattr(pipeline_ref, '_param_sign_acc'):
            pipeline_ref._param_sign_acc.clear()
            del pipeline_ref._param_sign_acc
        if compute_inter_mbs and hasattr(pipeline_ref, '_param_sign_acc_diff'):
            pipeline_ref._param_sign_acc_diff.clear()
            del pipeline_ref._param_sign_acc_diff
        if _acc_stats is not None:
            _acc_stats.clear()
            if hasattr(pipeline_ref, '_acc_group_stats'):
                del pipeline_ref._acc_group_stats

        # debug: inter-mbs sign agree 计算结果
        if compute_inter_mbs:
            _sample_keys = list(per_group_inter_mbs_sign.keys())[:3]
            _sample_vals = [per_group_inter_mbs_sign[k] for k in _sample_keys]
            _sample_vals_diff = [per_group_inter_mbs_sign_diff[k] for k in _sample_keys]
            logger.info(f"[grad_analysis] inter_mbs_sign: n_mbs={n_mbs}, "
                        f"n_groups={len(per_group_inter_mbs_sign)}, "
                        f"ce_sample={dict(zip(_sample_keys, _sample_vals))}, "
                        f"diff_sample={dict(zip(_sample_keys, _sample_vals_diff))}")
        else:
            logger.info(f"[grad_analysis] inter_mbs_sign SKIPPED: compute_inter_mbs=False, "
                        f"total_n_mbs={total_n_mbs}")

        # ---- 5. 写入 placeholder ----
        ce_grad_norm = ce_norm_sq_total ** 0.5
        diff_grad_norm = diff_norm_sq_total ** 0.5

        # 判断本 DP rank 是否有有效的 CE / Diff 梯度
        # 没有 text token 的 rank CE replay 被 skip，ce_grad ≈ 0，
        # placeholder count 设为 0 以排除该 rank，避免稀释 DP 平均值
        _has_valid_ce = ce_grad_norm > 1e-10
        _has_valid_diff = diff_grad_norm > 1e-10
        _ce_count = 1.0 if _has_valid_ce else 0.0
        _diff_count = 1.0 if _has_valid_diff else 0.0

        ph_ce = getattr(output_tensor, '_grad_norm_placeholder_ce', None)
        ph_diff = getattr(output_tensor, '_grad_norm_placeholder_diff', None)
        if ph_ce is not None:
            ph_ce[0] = ce_grad_norm
            ph_ce[1] = _ce_count
        if ph_diff is not None:
            ph_diff[0] = diff_grad_norm
            ph_diff[1] = _diff_count

        # per-layer sign agreement rate: CE vs Diff (仅两者都有效时才计入)
        _both_valid_count = 1.0 if (_has_valid_ce and _has_valid_diff) else 0.0
        ph_sign = getattr(output_tensor, '_sign_agree_ce_vs_diff_placeholders', None)
        if ph_sign is not None:
            for layer_idx, sa_val in per_layer_sign_agree.items():
                key = f"sign_agree_ce_vs_diff/layer_{layer_idx}"
                if key in ph_sign:
                    ph_sign[key][0] = sa_val
                    ph_sign[key][1] = _both_valid_count

        ph_avg_sign = getattr(output_tensor, '_avg_sign_agree_ce_vs_diff_placeholder', None)
        if ph_avg_sign is not None and per_layer_sign_agree:
            avg_sa = sum(per_layer_sign_agree.values()) / len(per_layer_sign_agree)
            ph_avg_sign[0] = avg_sa
            ph_avg_sign[1] = _both_valid_count

        # per-group ce/diff grad norms
        ph_group_norms = getattr(output_tensor, '_per_group_norm_placeholders', None)
        if ph_group_norms is not None:
            for gkey, ce_n in per_group_ce_norm.items():
                ph_key = f"grad_ce_norm/{gkey}"
                if ph_key in ph_group_norms:
                    ph_group_norms[ph_key][0] = ce_n
                    ph_group_norms[ph_key][1] = _ce_count
            for gkey, diff_n in per_group_diff_norm.items():
                ph_key = f"grad_diff_norm/{gkey}"
                if ph_key in ph_group_norms:
                    ph_group_norms[ph_key][0] = diff_n
                    ph_group_norms[ph_key][1] = _diff_count

        # inter-mbs CE sign agreement rate placeholders
        ph_inter_mbs = getattr(output_tensor, '_inter_mbs_sign_placeholders', None)
        logger.info(f"[grad_analysis] inter_mbs placeholder write: "
                    f"ph_inter_mbs={ph_inter_mbs is not None}, "
                    f"per_group_inter_mbs_sign={len(per_group_inter_mbs_sign)}, "
                    f"_ce_count={_ce_count}")
        if ph_inter_mbs is not None and per_group_inter_mbs_sign:
            for gkey, ims_val in per_group_inter_mbs_sign.items():
                ph_key = f"sign_agree_inter_mbs_ce/{gkey}"
                if ph_key in ph_inter_mbs:
                    ph_inter_mbs[ph_key][0] = ims_val
                    ph_inter_mbs[ph_key][1] = _ce_count  # 排除无 CE 的 rank
        ph_avg_ims = getattr(output_tensor, '_avg_inter_mbs_sign_placeholder', None)
        if ph_avg_ims is not None and per_group_inter_mbs_sign:
            # 只对 backbone layers 求平均 (head groups 可能语义不同)
            backbone_ims = [v for k, v in per_group_inter_mbs_sign.items() if k.startswith("layer_")]
            if backbone_ims:
                ph_avg_ims[0] = sum(backbone_ims) / len(backbone_ims)
                ph_avg_ims[1] = _ce_count

        # inter-mbs Diff sign agreement rate placeholders
        ph_inter_mbs_diff = getattr(output_tensor, '_inter_mbs_diff_sign_placeholders', None)
        if ph_inter_mbs_diff is not None and per_group_inter_mbs_sign_diff:
            for gkey, ims_val in per_group_inter_mbs_sign_diff.items():
                ph_key = f"sign_agree_inter_mbs_diff/{gkey}"
                if ph_key in ph_inter_mbs_diff:
                    ph_inter_mbs_diff[ph_key][0] = ims_val
                    ph_inter_mbs_diff[ph_key][1] = _diff_count  # 排除无 Diff 的 rank
        ph_avg_ims_diff = getattr(output_tensor, '_avg_inter_mbs_diff_sign_placeholder', None)
        if ph_avg_ims_diff is not None and per_group_inter_mbs_sign_diff:
            backbone_ims_diff = [v for k, v in per_group_inter_mbs_sign_diff.items() if k.startswith("layer_")]
            if backbone_ims_diff:
                ph_avg_ims_diff[0] = sum(backbone_ims_diff) / len(backbone_ims_diff)
                ph_avg_ims_diff[1] = _diff_count

        # 保存 per_layer_sign_agree 供外部使用
        output_tensor._per_layer_sign_agree_result = per_layer_sign_agree

        # ---- 5b. 直接写 TensorBoard scalar + matplotlib 图表 ----
        # 绕过 loss_dict -> training_log 流程，直接写 TensorBoard，
        # 避免 tensorboard_log_interval 与 grad_norm_interval 不对齐导致丢失数据。
        tb_writer = getattr(output_tensor, '_tb_writer', None)
        iteration = getattr(output_tensor, '_iteration', None)

        # --- DP all-reduce norms / sign_agree / inter_mbs_sign，使图表反映全局平均值 ---
        # 注意: all_reduce 是集合通信，所有 DP rank 都必须参与，
        # 因此必须放在 tb_writer guard 外面，否则会导致 NCCL 死锁。
        dp_size_val = torch.distributed.get_world_size(dp_group)
        n_ims = len(per_group_inter_mbs_sign)  # inter-mbs sign agree 条目数 (CE)
        n_ims_diff = len(per_group_inter_mbs_sign_diff)  # inter-mbs sign agree 条目数 (Diff)
        if dp_size_val > 1:
            n_keys = len(ordered_keys)
            n_backbone = len(per_layer_sign_agree)
            # buf: [ce_0, diff_0, ..., sa_0, ..., ims_ce_0, ..., ims_diff_0, ...,
            #       ce_cnt, diff_cnt, sa_cnt]
            buf_len = n_keys * 2 + n_backbone + n_ims + n_ims_diff + 3
            _dp_buf = torch.zeros(buf_len, device='cuda')
            for ki, key in enumerate(ordered_keys):
                _dp_buf[ki * 2] = per_group_ce_norm[key]
                _dp_buf[ki * 2 + 1] = per_group_diff_norm[key]
            sorted_sa_keys = sorted(per_layer_sign_agree.keys())
            for ci, lidx in enumerate(sorted_sa_keys):
                _dp_buf[n_keys * 2 + ci] = per_layer_sign_agree[lidx]
            # inter-mbs CE sign agree (same order as ordered_keys, only if computed)
            ims_offset = n_keys * 2 + n_backbone
            if per_group_inter_mbs_sign:
                for ki, key in enumerate(ordered_keys):
                    _dp_buf[ims_offset + ki] = per_group_inter_mbs_sign.get(key, 0.5)
            # inter-mbs Diff sign agree
            ims_diff_offset = ims_offset + n_ims
            if per_group_inter_mbs_sign_diff:
                for ki, key in enumerate(ordered_keys):
                    _dp_buf[ims_diff_offset + ki] = per_group_inter_mbs_sign_diff.get(key, 0.5)
            # valid counts
            _dp_buf[-3] = 1.0 if _has_valid_ce else 0.0
            _dp_buf[-2] = 1.0 if _has_valid_diff else 0.0
            _dp_buf[-1] = 1.0 if (_has_valid_ce and _has_valid_diff) else 0.0
            torch.distributed.all_reduce(_dp_buf, group=dp_group)
            _ce_cnt = max(_dp_buf[-3].item(), 1.0)
            _diff_cnt = max(_dp_buf[-2].item(), 1.0)
            _sa_cnt = max(_dp_buf[-1].item(), 1.0)
            fig_ce_norms = {k: _dp_buf[i * 2].item() / _ce_cnt
                            for i, k in enumerate(ordered_keys)}
            fig_diff_norms = {k: _dp_buf[i * 2 + 1].item() / _diff_cnt
                              for i, k in enumerate(ordered_keys)}
            fig_sign_agree = {lidx: _dp_buf[n_keys * 2 + ci].item() / _sa_cnt
                              for ci, lidx in enumerate(sorted_sa_keys)}
            fig_inter_mbs_sign = {k: _dp_buf[ims_offset + i].item() / _ce_cnt
                                  for i, k in enumerate(ordered_keys)} if n_ims > 0 else {}
            fig_inter_mbs_sign_diff = {k: _dp_buf[ims_diff_offset + i].item() / _diff_cnt
                                       for i, k in enumerate(ordered_keys)} if n_ims_diff > 0 else {}
            fig_ce_total = sum(v ** 2 for v in fig_ce_norms.values()) ** 0.5
            fig_diff_total = sum(v ** 2 for v in fig_diff_norms.values()) ** 0.5
        else:
            fig_ce_norms = per_group_ce_norm
            fig_diff_norms = per_group_diff_norm
            fig_sign_agree = per_layer_sign_agree
            fig_inter_mbs_sign = per_group_inter_mbs_sign
            fig_inter_mbs_sign_diff = per_group_inter_mbs_sign_diff
            fig_ce_total = ce_grad_norm
            fig_diff_total = diff_grad_norm

        # --- 只有 tb_writer rank 才写 TensorBoard ---
        if tb_writer is not None and iteration is not None:
            # --- 直接写 scalar: 使用 DP 平均后的值 ---
            try:
                tb_writer.add_scalar("grad_norm/ce_loss", fig_ce_total, iteration)
                tb_writer.add_scalar("grad_norm/diff_loss", fig_diff_total, iteration)
                for gkey in ordered_keys:
                    tb_writer.add_scalar(f"grad_ce_norm/{gkey}", fig_ce_norms[gkey], iteration)
                    tb_writer.add_scalar(f"grad_diff_norm/{gkey}", fig_diff_norms[gkey], iteration)
                for layer_idx, sa_val in fig_sign_agree.items():
                    tb_writer.add_scalar(f"sign_agree_ce_vs_diff/layer_{layer_idx}", sa_val, iteration)
                if fig_sign_agree:
                    avg_sa = sum(fig_sign_agree.values()) / len(fig_sign_agree)
                    tb_writer.add_scalar("sign_agree_ce_vs_diff/avg", avg_sa, iteration)
                # inter-mbs CE sign agreement rate
                if fig_inter_mbs_sign:
                    for gkey in ordered_keys:
                        if gkey in fig_inter_mbs_sign:
                            tb_writer.add_scalar(f"sign_agree_inter_mbs_ce/{gkey}",
                                                 fig_inter_mbs_sign[gkey], iteration)
                    backbone_ims = [fig_inter_mbs_sign[k] for k in ordered_keys
                                    if k.startswith("layer_") and k in fig_inter_mbs_sign]
                    if backbone_ims:
                        tb_writer.add_scalar("sign_agree_inter_mbs_ce/avg",
                                             sum(backbone_ims) / len(backbone_ims), iteration)
                # inter-mbs Diff sign agreement rate
                if fig_inter_mbs_sign_diff:
                    for gkey in ordered_keys:
                        if gkey in fig_inter_mbs_sign_diff:
                            tb_writer.add_scalar(f"sign_agree_inter_mbs_diff/{gkey}",
                                                 fig_inter_mbs_sign_diff[gkey], iteration)
                    backbone_ims_d = [fig_inter_mbs_sign_diff[k] for k in ordered_keys
                                      if k.startswith("layer_") and k in fig_inter_mbs_sign_diff]
                    if backbone_ims_d:
                        tb_writer.add_scalar("sign_agree_inter_mbs_diff/avg",
                                             sum(backbone_ims_d) / len(backbone_ims_d), iteration)
                tb_writer.flush()
            except Exception as e:
                logger.warning(f"[grad_analysis] TensorBoard scalar write failed: {e}")

            # --- matplotlib 图表 (使用 DP 平均后的值) ---
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np

                # --- Figure 1: per-group CE vs Diff grad norm (分组柱状图) ---
                groups = ordered_keys
                ce_vals = np.array([fig_ce_norms[k] for k in groups])
                diff_vals = np.array([fig_diff_norms[k] for k in groups])

                x = np.arange(len(groups))
                width = 0.35

                fig, ax = plt.subplots(figsize=(max(len(groups) * 0.45, 8), 4))
                ax.bar(x - width / 2, np.maximum(ce_vals, 1e-12), width,
                       label='CE grad', color='#3498db', alpha=0.85)
                ax.bar(x + width / 2, np.maximum(diff_vals, 1e-12), width,
                       label='Diff grad', color='#e74c3c', alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(groups, fontsize=6, rotation=45, ha='right')
                ax.set_ylabel('Grad Norm')
                ax.set_title(f'Per-Layer/Head CE vs Diff Grad Norm  (iter {iteration})')
                ax.set_yscale('log')
                ax.legend(fontsize=8)
                fig.tight_layout()
                tb_writer.add_figure("grad_analysis/per_layer_norm", fig, iteration)
                plt.close(fig)

                # --- Figure 2: CE vs Diff sign agreement rate (backbone layers) ---
                if fig_sign_agree:
                    layers = sorted(fig_sign_agree.keys())
                    sa_vals = [fig_sign_agree[l] for l in layers]
                    colors = ["#2ecc71" if v > 0.5 else "#e74c3c" for v in sa_vals]

                    fig, ax = plt.subplots(figsize=(max(len(layers) * 0.35, 6), 3.5))
                    ax.bar(range(len(layers)), sa_vals, color=colors, width=0.7)
                    ax.set_xticks(range(len(layers)))
                    ax.set_xticklabels(layers, fontsize=7)
                    ax.set_xlabel("Layer Index")
                    ax.set_ylabel("Sign Agreement Rate")
                    ax.set_title(f"CE vs Diff Sign Agreement  (iter {iteration})")
                    ax.axhline(y=0.5, color="black", linewidth=0.5, linestyle="--")
                    ax.set_ylim(0, 1.05)
                    fig.tight_layout()
                    tb_writer.add_figure("grad_analysis/sign_agree_ce_vs_diff", fig, iteration)
                    plt.close(fig)

                # --- Figure 3: inter-mbs CE sign agreement rate (backbone layers) ---
                if fig_inter_mbs_sign:
                    ims_layers = sorted([int(k.split("_")[1]) for k in fig_inter_mbs_sign
                                         if k.startswith("layer_")])
                    if ims_layers:
                        ims_vals = [fig_inter_mbs_sign[f"layer_{l}"] for l in ims_layers]
                        ims_colors = ["#2ecc71" if v > 0.5 else "#e74c3c" for v in ims_vals]

                        fig, ax = plt.subplots(figsize=(max(len(ims_layers) * 0.35, 6), 3.5))
                        ax.bar(range(len(ims_layers)), ims_vals, color=ims_colors, width=0.7)
                        ax.set_xticks(range(len(ims_layers)))
                        ax.set_xticklabels(ims_layers, fontsize=7)
                        ax.set_xlabel("Layer Index")
                        ax.set_ylabel("Sign Agreement Rate")
                        ax.set_title(f"Inter-Mbs CE Sign Agreement  (iter {iteration}, N={n_mbs})")
                        ax.axhline(y=0.5, color="black", linewidth=0.5, linestyle="--")
                        ax.set_ylim(0, 1.05)
                        fig.tight_layout()
                        tb_writer.add_figure("grad_analysis/sign_agree_inter_mbs_ce", fig, iteration)
                        plt.close(fig)

                # --- Figure 4: inter-mbs Diff sign agreement rate (backbone layers) ---
                if fig_inter_mbs_sign_diff:
                    ims_d_layers = sorted([int(k.split("_")[1]) for k in fig_inter_mbs_sign_diff
                                           if k.startswith("layer_")])
                    if ims_d_layers:
                        ims_d_vals = [fig_inter_mbs_sign_diff[f"layer_{l}"] for l in ims_d_layers]
                        ims_d_colors = ["#2ecc71" if v > 0.5 else "#e74c3c" for v in ims_d_vals]

                        fig, ax = plt.subplots(figsize=(max(len(ims_d_layers) * 0.35, 6), 3.5))
                        ax.bar(range(len(ims_d_layers)), ims_d_vals, color=ims_d_colors, width=0.7)
                        ax.set_xticks(range(len(ims_d_layers)))
                        ax.set_xticklabels(ims_d_layers, fontsize=7)
                        ax.set_xlabel("Layer Index")
                        ax.set_ylabel("Sign Agreement Rate")
                        ax.set_title(f"Inter-Mbs Diff Sign Agreement  (iter {iteration}, N={n_mbs})")
                        ax.axhline(y=0.5, color="black", linewidth=0.5, linestyle="--")
                        ax.set_ylim(0, 1.05)
                        fig.tight_layout()
                        tb_writer.add_figure("grad_analysis/sign_agree_inter_mbs_diff", fig, iteration)
                        plt.close(fig)
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"[grad_analysis] matplotlib figure write failed: {e}")

        # ---- 6. 恢复 ALL main_grad = total_grad ----
        for name, total_g_bf16, p, _ in all_grads:
            g = getattr(p, 'main_grad', None)
            if g is not None:
                g.copy_(total_g_bf16)
            elif p.grad is not None:
                p.grad.copy_(total_g_bf16)

        return per_layer_sign_agree

    @staticmethod
    def _build_ce_loss_reweight(packed_sample_lens: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """
        构造 sample-level loss reweight 张量，使每个样本对 CE loss 的贡献相等。

        每个 token 的 reweight = 1 / n_k
        其中 n_k = 该样本的有效 token 数 (loss_mask == 1 的 token 数)

        效果:
        - 每个样本对 loss_sum 的贡献 = avg_loss_k (该样本的平均 loss)
        - 每个样本对 loss_count 的贡献 = 1
        - loss = sum / count = Σ(avg_loss_k) / total_samples = 精确的样本平均 loss

        Args:
            packed_sample_lens: [B, max_num_samples], 每行内各子样本的 token 长度, padding 用 0
            loss_mask: [B, S], token 级别的 loss mask (1.0 = 有效, 0.0 = 无效)

        Returns:
            ce_loss_reweight: [B, S], 每个 token 的 reweight 系数
        """
        B, S = loss_mask.shape
        device = loss_mask.device
        ce_loss_reweight = torch.zeros(B, S, device=device, dtype=loss_mask.dtype)

        for b in range(B):
            sample_lens = packed_sample_lens[b]  # [max_num_samples]
            # 过滤掉 padding 的 0 值
            valid_lens = sample_lens[sample_lens > 0]
            num_samples = valid_lens.shape[0]
            if num_samples == 0:
                continue

            offset = 0
            for i in range(num_samples):
                slen = valid_lens[i].item()
                end = min(offset + slen, S)
                # 该样本区间内的有效 token 数 (loss_mask == 1 的)
                sample_loss_mask = loss_mask[b, offset:end]
                valid_tokens = sample_loss_mask.sum()
                if valid_tokens > 0:
                    # reweight = 1 / n_k
                    # 每个样本对 count 贡献 1, 对 sum 贡献该样本的平均 loss
                    ce_loss_reweight[b, offset:end] = sample_loss_mask / valid_tokens
                offset = end

        return ce_loss_reweight


    def training_step(
        self, model, data_batch: dict[str, torch.Tensor], compute_loss: bool = True, rank_offset: int = 0,
        iteration: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the Transfusion model.

        Args:
            model: The neural network model (e.g., Transfusion)
            data_batch (dict): Raw data batch drawn from the training dataloader.
            iteration (int): Current iteration number.

        Returns:
            A tuple containing the output dictionary (metrics/tensors) and the computed loss.
        """
        self.net = model

        if self._noise_generator is None:
            self._initialize_generators(rank_offset)

        current_device = torch.cuda.current_device()
        for key, val in data_batch.items():
            if isinstance(val, torch.Tensor) and val.device.type != "cuda":
                # 非阻塞传输可以稍微加速
                data_batch[key] = val.to(device=current_device, non_blocking=True)
        # ==========================================
        # 1. 提取文本数据 [B, S] (这里的 B 只是物理行数，不是逻辑样本数)
        # ==========================================
        input_ids = data_batch["input_ids"]
        labels = data_batch["labels"]
        attention_mask = data_batch.get("attention_mask")
        position_ids = data_batch.get("position_ids")  # (3, B, S) 由数据侧构造的 M-RoPE

        # FlashMask 参数
        startend_row_indices = data_batch.get("startend_row_indices")

        pixel_values = data_batch.get("pixel_values")          # 干净的图像像素
        image_grid_thw = data_batch.get("image_grid_thw")
        image_input_mask = data_batch.get("image_input_mask")
        # 提取生成掩码和每个生成图像的长度
        image_generate_mask = data_batch.get("image_generate_mask")  # [Total_Image_Tokens] (布尔值)
        image_understand_mask = data_batch.get("image_understand_mask")
        image_gen_seq_lens = data_batch.get("image_gen_seq_lens")   # [num_gen_images], 每张生成图的 raw token 数
        image_gen_grid_thw = data_batch.get("image_gen_grid_thw")
        # [num_gen_images, 3], pixel-space noise 使用的 raw grid

        total_img_und_mask = data_batch.get("total_img_und_mask")

        timesteps_samples = None
        timesteps_packed = None
        sigmas_packed = None
        image_x1 = None
        num_gen_tokens = 0
        has_gen_images = image_gen_seq_lens is not None and image_gen_seq_lens.shape[0] > 0
        has_gen_data = has_gen_images and pixel_values is not None
        if has_gen_data:
            num_gen_images = image_gen_seq_lens.shape[0]
            num_gen_tokens = image_gen_seq_lens.sum().item()

            # 1. 根据任务类型决定 sigma/timestep 采样策略
            # 生成图才携带 image_gen_seq_lens；理解图在无生成数据时走下面的 clean 分支。
            config = getattr(model, "config", getattr(
                getattr(model, "module", None), "config", None
            ))
            _is_pixel_dit = True
            task_type_val = data_batch.get("task_type")
            _first_task = int(task_type_val[0]) if task_type_val is not None else -1

            if _is_pixel_dit and _first_task == 0:
                # UNDERSTANDING: clean image condition for Pixel-DiT as well.
                sigmas_samples = torch.full(
                    (num_gen_images,), 0.0,
                    device=current_device, dtype=torch.float32
                )
                timesteps_samples = (sigmas_samples * 1000).long()
            else:
                # Generation/editing uses the regular flow-matching schedule.
                sigmas_samples, timesteps_samples = self.scheduler.sample_timesteps(
                    num_gen_images,
                    generator=self._noise_generator,
                    device=current_device,
                )

            # 2. 将时间步/sigma 按每张图的 token 长度展开
            timesteps_packed = torch.repeat_interleave(timesteps_samples, image_gen_seq_lens)
            sigmas_packed = torch.repeat_interleave(sigmas_samples, image_gen_seq_lens)

            # Pixel-DiT noise lives in raw RGB patch space and follows the
            # same patchification order as Qwen3-VL's image processor.
            if _is_pixel_dit:
                # 流程，确保 token 排列顺序与 pixel_values 一致。
                _patch_size = config.vision_config.patch_size  # 16
                _merge_size = config.vision_config.spatial_merge_size  # 2
                _temporal_patch_size = config.vision_config.temporal_patch_size  # 2
                # Qwen-VL duplicates each spatial RGB patch across the temporal
                # patch dimension, hence its packed pixel_values/noise width is
                # 3 * temporal_patch_size * patch_size**2 (1536 for 2x16x16).
                _channel = config.vision_config.in_channels  # 3

                gen_grid_thw = (
                    image_gen_grid_thw
                    if image_gen_grid_thw is not None
                    else image_grid_thw[-num_gen_images:]
                )
                noise_tokens_list = []
                for i in range(num_gen_images):
                    T_i = int(gen_grid_thw[i, 0])
                    H_i = int(gen_grid_thw[i, 1])
                    W_i = int(gen_grid_thw[i, 2])

                    noise_img = torch.randn(
                        T_i, _channel, H_i * _patch_size, W_i * _patch_size,
                        generator=self._noise_generator,
                        device=current_device,
                        dtype=torch.float32,
                    )
                    noise_patches = noise_img.reshape(
                        T_i, _channel,
                        H_i // _merge_size, _merge_size, _patch_size,
                        W_i // _merge_size, _merge_size, _patch_size,
                    )
                    noise_patches = noise_patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
                    noise_patches = (
                        noise_patches.unsqueeze(6)
                        .expand(-1, -1, -1, -1, -1, -1, _temporal_patch_size, -1, -1)
                        .reshape(
                            T_i * H_i * W_i,
                            _channel * _temporal_patch_size * _patch_size * _patch_size,
                        )
                    )
                    noise_tokens_list.append(noise_patches)

                image_x1 = torch.cat(noise_tokens_list, dim=0).to(torch.bfloat16)
            else:
                noise_dim = config.vision_config.hidden_size
            # 4. 按总生成 token 数生成纯噪声 X_1
            # 先用 float32 生成噪声以保证数值稳定性，再转换为模型精度
            if image_x1 is None:
                image_x1 = torch.randn(
                    (num_gen_tokens, noise_dim),
                    generator=self._noise_generator,
                    device=current_device,
                    dtype=torch.float32
                ).to(torch.bfloat16)

            # Dynamic noise scale: 按分辨率缩放噪声
            if self.scheduler.use_dynamic_noise_scale:
                noise_scales = compute_dynamic_noise_scale(
                    image_gen_seq_lens,
                    ref_token_count=self.scheduler.dynamic_noise_ref_tokens,
                    base_scale=self.scheduler.dynamic_noise_base_scale,
                )
                noise_scales_packed = torch.repeat_interleave(
                    noise_scales, image_gen_seq_lens
                ).unsqueeze(-1).to(image_x1.device, image_x1.dtype)
                image_x1 = image_x1 * noise_scales_packed

        # 提取其他条件
        # condition = {k: v for k, v in data_batch.items() if k not in [
        #     "input_ids", "labels", "attention_mask", "pixel_values",
        #     "image_grid_thw", "image_input_mask",
        #     "image_generate_mask", "image_gen_seq_lens" # 记得在这里过滤掉
        # ]}

        # Understanding uses sigma=0/timestep=0 for clean Pixel-DiT conditions.
        # 同一 micro-batch 不会同时有理解和生成数据
        # Dense Pixel-DiT always uses the clean-condition path.
        if not has_gen_data:
            config = getattr(model, "config", getattr(
                getattr(model, "module", None), "config", None
            ))
            _is_pixel_dit = True
            has_und_images = (
                _is_pixel_dit
                and total_img_und_mask is not None
                and total_img_und_mask.numel() > 0
                and total_img_und_mask.any()
            )
            if has_und_images:
                num_und_tokens = int(total_img_und_mask.sum().item())
                sigmas_packed = torch.zeros(
                    num_und_tokens, device=current_device, dtype=torch.float32
                )
                timestep_value = 0
                timesteps_packed = torch.full(
                    (num_und_tokens,), timestep_value, device=current_device, dtype=torch.long
                )
                num_und_images = (
                    image_grid_thw.shape[0] if image_grid_thw is not None else 0
                )
                if num_und_images > 0:
                    timesteps_samples = torch.full(
                        (num_und_images,), timestep_value,
                        device=current_device, dtype=torch.long
                    )

        # ==========================================
        # 5. 判断是否需要在 backward 后计算 per-loss 梯度范数
        #    (实际计算延迟到 backward_step 中, 零额外 forward, 不会 OOM)
        # ==========================================
        should_log_grad = (
            self.log_per_loss_grad_norm
            and iteration > 0
            and iteration % self.grad_norm_interval == 0
        )

        # ==========================================
        # 6. 统一模型前向传播
        # ==========================================
        output = self.net(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            attention_mask=attention_mask,
            extra_block_kwargs={
                "startend_row_indices": startend_row_indices,
                "causal": False,  # 图片内部双向 attention，由 startend_row_indices 手动编码
            } if startend_row_indices is not None else None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_input_mask=image_input_mask,
            image_generate_mask=image_generate_mask,
            image_understand_mask=image_understand_mask,
            image_gen_grid_thw=image_gen_grid_thw,
            total_img_und_mask=total_img_und_mask,
            timesteps_samples=timesteps_samples,
            timesteps=timesteps_packed,
            sigmas=sigmas_packed if self.scheduler.timestep_sampling != "uniform" else None,
            noise=image_x1,
            diff_scheduler=self.scheduler,
            task_type=data_batch.get("task_type"),
            predict_x0=self.predict_x0,
        )

        # 如果不是最后一个 Stage (不需要算 Loss)，直接返回模型的隐藏状态
        if not compute_loss:
            return output

        text_loss = output["text_loss"]  # [B, seq_len] 或标量 0.0 (纯生成 batch 跳过 output_layer)
        image_pred = output["image_pred"]     # [Total_Image_Tokens, D]

        # ==========================================
        # 7. 计算 Text CE Loss (区分理解 / 纯文)
        # ==========================================
        # 普通 generation/edit 样本在 gate 协议下只保留 route-gate target；
        # 图像 token 和其它视觉边界仍为 pad，不进入 CE。关闭 gate 时则无普通
        # generation/edit CE。generation_cot 的 gate + 规划文本单独统计。
        #
        # 为支持 global batch loss 统计，保存 loss_sum 和 token_count，
        # Megatron 会跨 DP/mbs 聚合后得到正确的全局平均 loss。

        # 按 task_type 拆分 CE loss。普通 generation/edit 的 gate CE，以及
        # generation_cot 的 gate + think/recaption 文本，都归入同一个 text CE
        # 反传分支；generation_cot 子集仍单独保留用于日志观测。
        task_type = data_batch.get("task_type")
        loss_mask = data_batch.get("loss_mask")
        num_gen_per_sample = data_batch.get("num_gen_images_per_sample")

        if task_type is not None and text_loss.dim() == 2:
            ce_by_task = _split_ce_loss_by_task(text_loss, loss_mask, task_type)
            regular_text_ce_sum, regular_text_ce_count = ce_by_task["text"]
            und_ce_loss_sum, und_ce_loss_count = ce_by_task["understanding"]
            gen_cot_ce_loss_sum, gen_cot_ce_loss_count = ce_by_task["generation_cot"]
            (
                gen_cot_text_ce_loss_sum,
                gen_cot_text_ce_loss_count,
            ) = ce_by_task["generation_cot_text"]
            text_ce_loss_sum = (
                regular_text_ce_sum
                + gen_cot_ce_loss_sum
                + gen_cot_text_ce_loss_sum
            )
            text_ce_loss_count = (
                regular_text_ce_count
                + gen_cot_ce_loss_count
                + gen_cot_text_ce_loss_count
            )

        else:
            text_ce_loss_sum = (text_loss * loss_mask).sum()
            text_ce_loss_count = data_batch["loss_mask"].sum()
            zero_ce_loss = text_ce_loss_sum * 0.0
            zero_ce_count = text_ce_loss_count * 0.0
            und_ce_loss_sum = zero_ce_loss
            und_ce_loss_count = zero_ce_count
            gen_cot_ce_loss_sum = zero_ce_loss
            gen_cot_ce_loss_count = zero_ce_count
            gen_cot_text_ce_loss_sum = zero_ce_loss
            gen_cot_text_ce_loss_count = zero_ce_count

        # ==========================================
        # 8. 计算 Image Flow Match Loss
        # ==========================================
        # 只有当本批次有生成任务时才算 diff_loss，否则给个 0 梯度
        # 同样保存 loss_sum 和 count 以支持 global batch 统计
        has_diffusion_target = (
            num_gen_tokens > 0
            and "image_target" in output
            and output["image_target"] is not None
        )
        metric_device = text_loss.device if isinstance(text_loss, torch.Tensor) else (
            image_gen_seq_lens.device if isinstance(image_gen_seq_lens, torch.Tensor)
            else torch.device("cuda")
        )
        diffusion_token_counts = {
            name: torch.tensor(0.0, device=metric_device)
            for name in ("generation", "generation_cot", "editing", "t2i", "total")
        }
        if has_diffusion_target:
            image_pred = output["image_pred"]     # [Total_Image_Tokens, D]
            image_target = output["image_target"]  # [Total_Image_Tokens, D]

            # Pixel-DiT image predictions are BF16, while raw pixel x0 can
            # originate from FP32 processor tensors.  Subtraction in the
            # velocity target then promotes it to FP32; mixed-dtype MSE can
            # pass forward but fails in the BF16 backward kernel.
            diff_loss_per_element = _diffusion_mse_loss(image_pred, image_target)
            diff_loss_per_element = diff_loss_per_element.reshape(-1, diff_loss_per_element.shape[-1])

            sigmas = self.scheduler.get_sigmas_batch_packing(
                output["timesteps_packed"], device=current_device
            )
            loss_reweight = diffusion_parameterization_weight(
                sigmas,
                predict_x0=self.predict_x0,
                loss_type=self.diffusion_loss_type,
            )
            if not self.predict_x0:
                scheduler_weight = self.scheduler.training_weight_packing(
                    timesteps_packed
                ).to(device=current_device).unsqueeze(-1)
                loss_reweight = loss_reweight * scheduler_weight
            loss_reweight = loss_reweight.expand(-1, image_pred.shape[-1])
            diff_loss_per_element = diff_loss_per_element * loss_reweight

            # 先对 D 维度求平均，得到 per-token loss
            diff_loss_per_token = diff_loss_per_element.mean(dim=-1)  # [Total_Image_Tokens]

            # Dense Pixel-DiT filters understanding/text dummy tokens. Then
            # normalize at token or image granularity according to the config;
            # 归一化，返回的 loss units 同时供任务拆分使用。
            config = getattr(model, "config", getattr(
                getattr(model, "module", None), "config", None
            ))
            _filters_dummy_diff_tokens = True
            if _filters_dummy_diff_tokens:
                valid_token_mask = (sigmas_packed > 0.001) & (sigmas_packed < 0.999)
            else:
                valid_token_mask = torch.ones_like(diff_loss_per_token, dtype=torch.bool)
            valid_token_mask = valid_token_mask.to(diff_loss_per_token.device)
            if task_type is not None and num_gen_per_sample is not None:
                diffusion_token_counts = _split_diffusion_token_counts_by_task(
                    valid_token_mask,
                    task_type,
                    num_gen_per_sample,
                    image_gen_seq_lens,
                )
            else:
                diffusion_token_counts["generation"] = valid_token_mask.sum().to(
                    dtype=torch.float32
                )
                diffusion_token_counts["t2i"] = diffusion_token_counts["generation"]
                diffusion_token_counts["total"] = diffusion_token_counts["generation"]

            configured_reduction = self.diffusion_loss_reduction
            if configured_reduction == "auto":
                configured_reduction = (
                    "per_image_token" if _filters_dummy_diff_tokens else "per_token"
                )
            (
                diff_loss_units,
                units_per_image,
                diff_loss_sum,
                diff_loss_count,
            ) = _reduce_diffusion_loss(
                diff_loss_per_token,
                image_gen_seq_lens,
                valid_token_mask,
                configured_reduction,
            )
            if diff_loss_count.item() == 0:
                # Pure understanding/text batches can still carry dummy image
                # predictions. Keep a tiny autograd edge without affecting loss.
                diff_loss_sum = diff_loss_per_token.sum() * 1e-8
        else:
            diff_loss_sum = torch.tensor(0.0, device="cuda", requires_grad=True)
            diff_loss_count = torch.tensor(0.0, device="cuda")
            diff_loss_units = diff_loss_sum.unsqueeze(0)[:0]
            units_per_image = torch.zeros(0, device=diff_loss_sum.device, dtype=torch.long)

        # 按 task_type 拆分 diff loss，单独保留 generation-COT 以支持专属权重。
        # 每个 seq 内模态相同，不同 seq 模态可能不同，按 sample 级别拆分。
        if has_diffusion_target and task_type is not None and num_gen_per_sample is not None:
            # 每个 sample 的 loss unit 数由选定 reduction 决定：有效 token 数，
            # 或有效图片数。图片顺序与 packed diffusion units 保持一致。
            units_per_sample = torch.zeros(
                task_type.shape[0], device=diff_loss_units.device, dtype=torch.long
            )
            image_index = 0
            for sample_index in range(task_type.shape[0]):
                image_count = int(num_gen_per_sample[sample_index].item())
                if image_count > 0:
                    units_per_sample[sample_index] = units_per_image[
                        image_index:image_index + image_count
                    ].sum()
                image_index += image_count
            if image_index != units_per_image.numel():
                raise ValueError(
                    "num_gen_images_per_sample does not cover generated images: "
                    f"{image_index} != {units_per_image.numel()}"
                )
            (
                generation_diff_loss_sum,
                generation_diff_loss_count,
                generation_cot_diff_loss_sum,
                generation_cot_diff_loss_count,
                edit_diff_loss_sum,
                edit_diff_loss_count,
            ) = _split_diff_loss_by_detailed_task(
                diff_loss_units, task_type, units_per_sample
            )
        elif has_diffusion_target and task_type is not None:
            # fallback: 无 num_gen_images_per_sample，整 batch 只有一种模态
            has_editing = (task_type == TASK_EDITING).any().item()
            has_generation = (task_type == TASK_GENERATION).any().item()
            has_generation_cot = (task_type == TASK_GENERATION_COT).any().item()
            active_tasks = sum(int(value) for value in (
                has_editing, has_generation, has_generation_cot
            ))
            if active_tasks > 1:
                raise ValueError(
                    "Cannot split mixed diffusion tasks without "
                    "num_gen_images_per_sample"
                )
            zero_diff_loss = diff_loss_sum * 0.0
            zero_count = torch.tensor(0.0, device=diff_loss_per_token.device)
            if has_editing:
                edit_diff_loss_sum = diff_loss_sum
                edit_diff_loss_count = diff_loss_count
                generation_diff_loss_sum = zero_diff_loss
                generation_diff_loss_count = zero_count
                generation_cot_diff_loss_sum = zero_diff_loss
                generation_cot_diff_loss_count = zero_count
            elif has_generation:
                generation_diff_loss_sum = diff_loss_sum
                generation_diff_loss_count = diff_loss_count
                generation_cot_diff_loss_sum = zero_diff_loss
                generation_cot_diff_loss_count = zero_count
                edit_diff_loss_sum = zero_diff_loss
                edit_diff_loss_count = zero_count
            elif has_generation_cot:
                generation_diff_loss_sum = zero_diff_loss
                generation_diff_loss_count = zero_count
                generation_cot_diff_loss_sum = diff_loss_sum
                generation_cot_diff_loss_count = diff_loss_count
                edit_diff_loss_sum = zero_diff_loss
                edit_diff_loss_count = zero_count
            else:
                generation_diff_loss_sum = zero_diff_loss
                generation_diff_loss_count = zero_count
                generation_cot_diff_loss_sum = zero_diff_loss
                generation_cot_diff_loss_count = zero_count
                edit_diff_loss_sum = zero_diff_loss
                edit_diff_loss_count = zero_count
        else:
            generation_diff_loss_sum = diff_loss_sum * 0.0
            generation_diff_loss_count = torch.tensor(0.0, device=diff_loss_sum.device)
            generation_cot_diff_loss_sum = diff_loss_sum * 0.0
            generation_cot_diff_loss_count = torch.tensor(0.0, device=diff_loss_sum.device)
            edit_diff_loss_sum = diff_loss_sum * 0.0
            edit_diff_loss_count = torch.tensor(0.0, device=diff_loss_sum.device)

        t2i_diff_loss_sum = generation_diff_loss_sum + generation_cot_diff_loss_sum
        t2i_diff_loss_count = generation_diff_loss_count + generation_cot_diff_loss_count
        weighted_t2i_diff_loss_sum = (
            generation_diff_loss_sum
            + self.generation_cot_diff_loss_weight * generation_cot_diff_loss_sum
        )

        # ==========================================
        # 9. 返回 loss_sum 和 count（保留计算图）
        # ==========================================
        output_batch = {
            "text_ce_loss": (text_ce_loss_sum / (text_ce_loss_count + 1e-5)).detach().clone(),
            "und_ce_loss": (und_ce_loss_sum / (und_ce_loss_count + 1e-5)).detach().clone(),
            "gen_cot_ce_loss": (gen_cot_ce_loss_sum / (gen_cot_ce_loss_count + 1e-5)).detach().clone(),
            "gen_cot_text_ce_loss": (gen_cot_text_ce_loss_sum / (gen_cot_text_ce_loss_count + 1e-5)).detach().clone(),
            "diff_loss": (diff_loss_sum / (diff_loss_count + 1e-5)).detach().clone(),
            "text_ce_loss_sum": text_ce_loss_sum.detach().clone(),
            "text_ce_loss_count": text_ce_loss_count.detach().clone(),
            "und_ce_loss_sum": und_ce_loss_sum.detach().clone(),
            "und_ce_loss_count": und_ce_loss_count.detach().clone(),
            "gen_cot_ce_loss_sum": gen_cot_ce_loss_sum.detach().clone(),
            "gen_cot_ce_loss_count": gen_cot_ce_loss_count.detach().clone(),
            "gen_cot_text_ce_loss_sum": gen_cot_text_ce_loss_sum.detach().clone(),
            "gen_cot_text_ce_loss_count": gen_cot_text_ce_loss_count.detach().clone(),
            "text_ce_token_count": text_ce_loss_count.detach().clone(),
            "und_ce_token_count": und_ce_loss_count.detach().clone(),
            "gen_cot_ce_token_count": gen_cot_ce_loss_count.detach().clone(),
            "gen_cot_text_ce_token_count": gen_cot_text_ce_loss_count.detach().clone(),
            "diff_loss_sum": diff_loss_sum.detach().clone(),
            "diff_loss_count": diff_loss_count.detach().clone(),
            "t2i_diff_loss_sum": t2i_diff_loss_sum.detach().clone(),
            "t2i_diff_loss_count": t2i_diff_loss_count.detach().clone(),
            "generation_diff_loss_sum": generation_diff_loss_sum.detach().clone(),
            "generation_diff_loss_count": generation_diff_loss_count.detach().clone(),
            "generation_cot_diff_loss_sum": generation_cot_diff_loss_sum.detach().clone(),
            "generation_cot_diff_loss_count": generation_cot_diff_loss_count.detach().clone(),
            "weighted_t2i_diff_loss_sum": weighted_t2i_diff_loss_sum.detach().clone(),
            "edit_diff_loss_sum": edit_diff_loss_sum.detach().clone(),
            "edit_diff_loss_count": edit_diff_loss_count.detach().clone(),
            "generation_diff_token_count": diffusion_token_counts["generation"].detach().clone(),
            "generation_cot_diff_token_count": diffusion_token_counts["generation_cot"].detach().clone(),
            "edit_diff_token_count": diffusion_token_counts["editing"].detach().clone(),
            "t2i_diff_token_count": diffusion_token_counts["t2i"].detach().clone(),
            "diff_token_count": diffusion_token_counts["total"].detach().clone(),
            "_grad_norm_computed": should_log_grad,
        }

        # 返回独立的 (loss_sum, count) 对，用于独立 all_reduce + backward
        # 格式: (modality_name, loss_sum, count, loss_weight)
        # - loss_sum: 保留计算图的 loss 总和
        # - count: 按 diffusion_loss_reduction 统计的 loss units（不需要梯度）
        # - loss_weight: 用于调节不同模态的相对重要性
        multi_loss_tensors = MultiLossTensors([
            ("ce", text_ce_loss_sum + und_ce_loss_sum,
             text_ce_loss_count + und_ce_loss_count,
             self.ce_loss_weight),
            # ("und_ce", und_ce_loss_sum, und_ce_loss_count, self.ce_loss_weight),
            ("diff", weighted_t2i_diff_loss_sum, t2i_diff_loss_count, self.diff_loss_weight),
            ("edit_diff", edit_diff_loss_sum, edit_diff_loss_count, self.edit_diff_loss_weight),
        ])
        # 附加信息给 backward_step: 模型引用 + grad norm 控制
        multi_loss_tensors.should_compute_grad_norm = should_log_grad
        multi_loss_tensors.model_ref = self.net

        # 保存 replay 数据供 CE-only replay 使用
        if should_log_grad:
            self._save_replay_data(data_batch, extra={
                "timesteps_samples": timesteps_samples,
                "timesteps_packed": timesteps_packed,
                "noise": image_x1,
            })
            # 每次都把当前累积的 replay 数据列表挂到 multi_loss_tensors 上
            # (backward_step 只在 is_last_microbatch 时使用)
            multi_loss_tensors._replay_data_list = self._replay_data_list
            multi_loss_tensors._ce_loss_weight = self.ce_loss_weight
            multi_loss_tensors._pipeline_ref = self
            multi_loss_tensors._predict_x0 = self.predict_x0
            multi_loss_tensors._grad_analysis_callback = TransfusionPipeline._post_backward_grad_analysis
        else:
            # 非 grad norm step，清空之前的 replay 数据
            self._replay_data_list = []

        return output_batch, multi_loss_tensors


class TransfusionForwardStep:
    """
    Megatron Forward Step 包装器 (纯训练版)。
    负责对接数据迭代器、判断流水线 Stage，并返回标准的 loss_function。
    """

    def __init__(
            self,
            sample_cfg: Optional[Dict] = None,
            scheduler_config: Optional[Dict] = None,
            predict_x0: bool = False,
            log_per_loss_grad_norm: bool = False,
            grad_norm_interval: int = 100,
            diff_loss_weight: float = 1.0,
            edit_diff_loss_weight: float = 1.0,
            # Deprecated compatibility argument. COT CE now uses ce_loss_weight.
            generation_cot_ce_loss_weight: float = 1.0,
            generation_cot_diff_loss_weight: float = 1.0,
            diffusion_loss_type: Optional[str] = None,
            diffusion_loss_reduction: Optional[str] = None,
        ):
        """Initialize the forward-step wrapper and its Transfusion pipeline."""
        # 实例化 Pipeline 引擎
        self.pipeline = TransfusionPipeline(
            scheduler_config=scheduler_config or {},
            seed=1234,
            time_token_id=-1,  # 替换为你实际的 token ID
            ce_loss_weight=1.0,
            diff_loss_weight=diff_loss_weight,
            generation_cot_diff_loss_weight=generation_cot_diff_loss_weight,
            edit_diff_loss_weight=edit_diff_loss_weight,
            predict_x0=predict_x0,
            diffusion_loss_type=diffusion_loss_type,
            diffusion_loss_reduction=diffusion_loss_reduction,
            log_per_loss_grad_norm=log_per_loss_grad_norm,
            grad_norm_interval=grad_norm_interval,
        )

        # 采样配置 (用于训练中定期生成图片)
        self.sample_cfg = sample_cfg or {}

    def __call__(
        self, 
        state: GlobalState, 
        data_iterator: Iterable, 
        model: TransfusionQwen3VLModel, 
        return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, partial]:
        """Fetch a batch, run the forward step, and perform optional sampling."""

        # 1. 抓取数据
        batch = self.data_process(state, data_iterator)

        # 2. 执行前向步
        result = self.forward_step(state, batch, model)

        # 3. 定期采样 (用当前 batch 数据做可视化)
        self._maybe_sample(state, model, batch)

        return result

    def data_process(self, state: GlobalState, data_iterator: Iterable) -> dict:
        """Fetch one batch from the iterator while recording loader timings."""
        timers = state.timers
        straggler_timer = state.straggler_timer

        timers("batch-generator", log_level=2).start()
        with straggler_timer(bdata=True):
            batch = next(data_iterator)
        timers("batch-generator").stop()

        return batch

    def forward_step(self, state, batch, model):
        """Run the pipeline step and return its output tensor and loss closure."""
        straggler_timer = state.straggler_timer

        with straggler_timer:
            # 核心判断：是不是最后一个 Stage？
            is_last_stage = parallel_state.is_pipeline_last_stage()
            rank_offset = parallel_state.get_data_parallel_rank(with_context_parallel=True)
            iteration = state.train_state.step

            # 交给 Pipeline 干活
            pipeline_result = self.pipeline.training_step(
                model=model,
                data_batch=batch,
                compute_loss=is_last_stage,
                rank_offset=rank_offset,
                iteration=iteration,
            )

            # 根据返回结果拆包
            if is_last_stage:
                output_batch, multi_loss_tensors = pipeline_result
                # multi_loss_tensors: MultiLossTensors[(name, loss_tensor, count, weight), ...]
                # 实际的多 loss 反传和 grad norm 计算在 backward_step 中处理
            else:
                output_batch = None
                multi_loss_tensors = None

        # 提取当前 micro batch 的图片统计信息
        image_stats = {
            "num_images": batch.get("num_images", 0),
            "num_gen_images": batch.get("num_gen_images", 0),
            "num_gen_cot_images": batch.get("num_gen_cot_images", 0),
            "num_und_images": batch.get("num_und_images", 0),
            "num_edit_images": batch.get("num_edit_images", 0),
            # 不同模态的训练行数统计
            "num_text_rows": batch.get("num_text_rows", 0),
            "num_und_rows": batch.get("num_und_rows", 0),
            "num_gen_rows": batch.get("num_gen_rows", 0),
            "num_gen_cot_rows": batch.get("num_gen_cot_rows", 0),
            "num_gen_cot_text_rows": batch.get("num_gen_cot_text_rows", 0),
            "num_edit_rows": batch.get("num_edit_rows", 0),
        }

        # 构建 Megatron 需要的回调函数
        # 获取 backbone 层数 + head group 名称，用于 per-layer/group grad norm placeholder
        num_backbone_layers = 0
        head_group_names = []
        if is_last_stage and isinstance(multi_loss_tensors, MultiLossTensors) and \
                getattr(multi_loss_tensors, 'should_compute_grad_norm', False):
            _unwrapped = model
            while hasattr(_unwrapped, 'module'):
                _unwrapped = _unwrapped.module
            _decoder = getattr(getattr(_unwrapped, 'language_model', None), 'decoder', None)
            if _decoder is not None:
                _layers = getattr(_decoder, 'layers', None)
                if _layers is not None:
                    num_backbone_layers = len(_layers)
            # 检测 head group 名称 (非 backbone 参数按顶层模块分组)
            _layer_pat = re.compile(r'language_model\.decoder\.layers\.\d+\.')
            _seen = set()
            for _name, _p in _unwrapped.named_parameters():
                if not _p.requires_grad or _layer_pat.search(_name):
                    continue
                _parts = _name.split('.')
                if _parts[0] == 'language_model':
                    _grp = _parts[1] if len(_parts) > 1 and _parts[1] != 'decoder' else None
                else:
                    _grp = _parts[0]
                if _grp and _grp not in _seen:
                    _seen.add(_grp)
                    head_group_names.append(_grp)
            head_group_names.sort()
            # 传递 TensorBoard writer 和 iteration 给 callback, 用于画图
            multi_loss_tensors._tb_writer = getattr(state, 'tensorboard_logger', None)
            multi_loss_tensors._iteration = iteration
        loss_function = self._create_loss_function(
            output_batch, multi_loss_tensors, image_stats, num_backbone_layers, head_group_names
        )

        # 标记最后一个 microbatch (用于 backward_step 中的 grad analysis)
        if is_last_stage and isinstance(multi_loss_tensors, MultiLossTensors):
            num_mbs = get_num_microbatches()
            self._mbs_counter = getattr(self, '_mbs_counter', 0) + 1
            multi_loss_tensors.is_last_microbatch = (self._mbs_counter >= num_mbs)
            if self._mbs_counter >= num_mbs:
                self._mbs_counter = 0
                # 最后一个 microbatch 后清空 replay 数据
                # (replay_data_list 已经通过引用挂到了 multi_loss_tensors 上)
                self.pipeline._replay_data_list = []

        # 返回一个 placeholder tensor（用于 pipeline 通信），实际 loss 在 loss_func 中处理
        if is_last_stage and multi_loss_tensors is not None:
            # 创建一个带梯度的 placeholder，实际反传在 loss_func 中处理
            output_tensor = multi_loss_tensors[0][1]  # 使用第一个 loss 作为 placeholder
        else:
            output_tensor = pipeline_result if not is_last_stage else None

        return output_tensor, loss_function

    def _create_loss_function(
        self,
        output_batch: Optional[dict],
        multi_loss_tensors: Optional[List[Tuple[str, torch.Tensor, float]]],
        image_stats: Optional[dict] = None,
        num_backbone_layers: int = 0,
        head_group_names: Optional[List[str]] = None,
    ) -> Callable:
        """
        闭包函数：Megatron 会在反向传播前调用它拿 Loss。

        多模态独立 loss 反传：
        - multi_loss_tensors: [(name, loss_tensor, weight), ...]
        - 每个 loss 独立 all_reduce 后独立 backward
        - 这样每个模态的梯度自然按各自的样本数平均，不会被稀释

        对于需要跨 DP/microbatch 聚合的统计量，使用 [sum, count] 格式的 tensor:
        - loss metric 的 ``samples`` 是对应模态的训练行数，由 encoder row counters 提供；
        - ``loss/token_nums`` 的 count 是有效、可训练的 token 数；
        - diffusion backward 仍使用 diffusion reduction units（token 或 image），不受日志
          sample/token denominator 改变影响；
        - Megatron 跨 mbs 和 DP 做 all_reduce sum: [sum(sum_i), sum(count_i)]
        - 最终: total_sum / total_count = 正确的全局平均值

        对于计数类统计 (num_images 等)，使用 [count, 1/N] 格式的 tensor:
        - N = num_microbatches * dp_size (参与 reduce 的总 micro batch 数)
        - Megatron 跨 mbs 和 DP sum: [sum(count_i), sum(1/N)] = [total_count, 1]
        - 最终: total_count / 1 = total_count (即全局总数)
        """
        # 计算 N = num_microbatches * dp_size
        num_mbs = get_num_microbatches()
        dp_size = parallel_state.get_data_parallel_world_size()
        N = num_mbs * dp_size
        denominator = 1.0 / N if N > 0 else 1.0

        def loss_func(output_tensor):
            """Compute and aggregate losses for the current microbatch."""
            if output_batch is not None:
                # Loss 统计使用 [sum, count] 格式，Megatron 会正确聚合为 global_sum / global_count
                #
                # 【修复 resume CE=INF】train.py 的聚合是 val[0]/val[1] (无 epsilon)。续训首步
                # 某个模态全局 count 可能为 0 (例如该 step 的 global batch 不含 understanding 行,
                # 或恢复/调度异常使该模态行被全部 mask)。0/0 → NaN, sum≠0/0 → INF, 都会污染
                # 日志且骗过 nan-detection (它只在 skipped_iter 分支跑)。
                # 给 count 分量加极小 eps: count 聚合后仍 >0 时 sum/(count+N*eps) ≈ 真值;
                # 全局 count==0 时退化为 0/(N*eps)=0, 不再 INF/NaN。eps 远小于单 token (>=1) 量级,
                # 对正常步的指标无可见影响。
                # ``samples`` are authoritative encoder rows; diffusion reduction
                # units remain in output_batch for backward normalization and raw
                # loss semantics. ``token_nums`` are valid trainable tokens.
                rows = image_stats or {}
                text_rows = rows.get("num_text_rows", 0)
                understanding_rows = rows.get("num_und_rows", 0)
                generation_rows = rows.get("num_gen_rows", 0)
                generation_cot_rows = rows.get("num_gen_cot_rows", 0)
                generation_cot_text_rows = rows.get("num_gen_cot_text_rows", 0)
                editing_rows = rows.get("num_edit_rows", 0)
                diffusion_rows = generation_rows + generation_cot_rows + editing_rows
                t2i_rows = generation_rows + generation_cot_rows
                loss_dict = {
                    "ce_loss_by_modality/text": _build_loss_metric_pair(
                        output_batch["text_ce_loss_sum"], text_rows
                    ),
                    "ce_loss_by_modality/understanding": _build_loss_metric_pair(
                        output_batch["und_ce_loss_sum"], understanding_rows
                    ),
                    "ce_loss_by_modality/generation_cot": _build_loss_metric_pair(
                        output_batch["gen_cot_ce_loss_sum"], generation_cot_rows
                    ),
                    "ce_loss_by_modality/generation_cot_text": _build_loss_metric_pair(
                        output_batch["gen_cot_text_ce_loss_sum"], generation_cot_text_rows
                    ),
                    "diff_loss": _build_loss_metric_pair(
                        output_batch["diff_loss_sum"], diffusion_rows
                    ),
                    "diff_loss_by_task/t2i": _build_loss_metric_pair(
                        output_batch["t2i_diff_loss_sum"], t2i_rows
                    ),
                    "diff_loss_by_task/generation": _build_loss_metric_pair(
                        output_batch["generation_diff_loss_sum"], generation_rows
                    ),
                    "diff_loss_by_task/generation_cot": _build_loss_metric_pair(
                        output_batch["generation_cot_diff_loss_sum"], generation_cot_rows
                    ),
                    "diff_loss_by_task/editing": _build_loss_metric_pair(
                        output_batch["edit_diff_loss_sum"], editing_rows
                    ),
                    # Weighted metrics use the same row denominator as raw metrics.
                    "weighted_diff_loss_by_task/t2i": _build_loss_metric_pair(
                        output_batch["weighted_t2i_diff_loss_sum"]
                        * self.pipeline.diff_loss_weight,
                        t2i_rows,
                    ),
                    "weighted_diff_loss_by_task/generation": _build_loss_metric_pair(
                        output_batch["generation_diff_loss_sum"]
                        * self.pipeline.diff_loss_weight,
                        generation_rows,
                    ),
                    "weighted_diff_loss_by_task/generation_cot": _build_loss_metric_pair(
                        output_batch["generation_cot_diff_loss_sum"]
                        * self.pipeline.diff_loss_weight
                        * self.pipeline.generation_cot_diff_loss_weight,
                        generation_cot_rows,
                    ),
                    "weighted_diff_loss_by_task/editing": _build_loss_metric_pair(
                        output_batch["edit_diff_loss_sum"] * self.pipeline.edit_diff_loss_weight,
                        editing_rows,
                    ),
                    "loss/token_nums/ce": _build_loss_metric_pair(
                        output_batch["text_ce_loss_sum"] + output_batch["und_ce_loss_sum"],
                        output_batch["text_ce_token_count"] + output_batch["und_ce_token_count"],
                    ),
                    "loss/token_nums/text": _build_loss_metric_pair(
                        output_batch["text_ce_loss_sum"], output_batch["text_ce_token_count"]
                    ),
                    "loss/token_nums/understanding": _build_loss_metric_pair(
                        output_batch["und_ce_loss_sum"], output_batch["und_ce_token_count"]
                    ),
                    "loss/token_nums/generation_cot": _build_loss_metric_pair(
                        output_batch["gen_cot_ce_loss_sum"], output_batch["gen_cot_ce_token_count"]
                    ),
                    "loss/token_nums/generation_cot_text": _build_loss_metric_pair(
                        output_batch["gen_cot_text_ce_loss_sum"],
                        output_batch["gen_cot_text_ce_token_count"],
                    ),
                    "loss/token_nums/diffusion": _build_loss_metric_pair(
                        output_batch["diff_loss_sum"], output_batch["diff_token_count"]
                    ),
                    "loss/token_nums/t2i": _build_loss_metric_pair(
                        output_batch["t2i_diff_loss_sum"], output_batch["t2i_diff_token_count"]
                    ),
                    "loss/token_nums/generation": _build_loss_metric_pair(
                        output_batch["generation_diff_loss_sum"],
                        output_batch["generation_diff_token_count"],
                    ),
                    "loss/token_nums/generation_cot": _build_loss_metric_pair(
                        output_batch["generation_cot_diff_loss_sum"],
                        output_batch["generation_cot_diff_token_count"],
                    ),
                    "loss/token_nums/editing": _build_loss_metric_pair(
                        output_batch["edit_diff_loss_sum"], output_batch["edit_diff_token_count"]
                    ),
                }
                # 梯度范数 (在 backward_step 中通过 in-place 修改 placeholder 填入)
                if output_batch.get("_grad_norm_computed", False):
                    # 创建 placeholder tensor, backward_step 中 in-place 更新 [0] 位
                    # 使用 [value, count] 格式: count=0 的 microbatch 不影响聚合结果
                    # 只有最后一个 microbatch 由 backward_step 设置为 [norm, 1.0]
                    ce_grad_placeholder = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                    diff_grad_placeholder = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                    loss_dict["grad_norm/ce_loss"] = ce_grad_placeholder
                    loss_dict["grad_norm/diff_loss"] = diff_grad_placeholder
                    # 把 placeholder 引用附到 multi_loss_tensors 上
                    # backward_step 在 backward 完成后 in-place 写入真实值
                    if isinstance(multi_loss_tensors, MultiLossTensors):
                        multi_loss_tensors._grad_norm_placeholder_ce = ce_grad_placeholder
                        multi_loss_tensors._grad_norm_placeholder_diff = diff_grad_placeholder

                    # Per-layer sign agreement rate placeholders (CE vs Diff)
                    if num_backbone_layers > 0:
                        sign_agree_placeholders = {}
                        for layer_idx in range(num_backbone_layers):
                            key = f"sign_agree_ce_vs_diff/layer_{layer_idx}"
                            ph = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                            loss_dict[key] = ph
                            sign_agree_placeholders[key] = ph
                        # 全局平均 sign agreement rate
                        avg_sign_ph = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                        loss_dict["sign_agree_ce_vs_diff/avg"] = avg_sign_ph
                        if isinstance(multi_loss_tensors, MultiLossTensors):
                            multi_loss_tensors._sign_agree_ce_vs_diff_placeholders = sign_agree_placeholders
                            multi_loss_tensors._avg_sign_agree_ce_vs_diff_placeholder = avg_sign_ph

                    # Per-layer/group CE 和 Diff grad norm placeholders
                    all_group_keys = [f"layer_{i}" for i in range(num_backbone_layers)]
                    if head_group_names:
                        all_group_keys.extend(head_group_names)
                    if all_group_keys:
                        group_norm_placeholders = {}
                        for gkey in all_group_keys:
                            for prefix in ("grad_ce_norm", "grad_diff_norm"):
                                ph_key = f"{prefix}/{gkey}"
                                ph = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                                loss_dict[ph_key] = ph
                                group_norm_placeholders[ph_key] = ph
                        if isinstance(multi_loss_tensors, MultiLossTensors):
                            multi_loss_tensors._per_group_norm_placeholders = group_norm_placeholders

                    # Inter-mbs CE sign agreement rate placeholders
                    # (只在 num_microbatches > 1 时有意义)
                    if num_backbone_layers > 0 and get_num_microbatches() > 1:
                        inter_mbs_placeholders = {}
                        all_ims_keys = [f"layer_{i}" for i in range(num_backbone_layers)]
                        if head_group_names:
                            all_ims_keys.extend(head_group_names)
                        for gkey in all_ims_keys:
                            ph_key = f"sign_agree_inter_mbs_ce/{gkey}"
                            ph = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                            loss_dict[ph_key] = ph
                            inter_mbs_placeholders[ph_key] = ph
                        avg_ims_ph = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                        loss_dict["sign_agree_inter_mbs_ce/avg"] = avg_ims_ph
                        if isinstance(multi_loss_tensors, MultiLossTensors):
                            multi_loss_tensors._inter_mbs_sign_placeholders = inter_mbs_placeholders
                            multi_loss_tensors._avg_inter_mbs_sign_placeholder = avg_ims_ph

                        # Inter-mbs Diff sign agreement rate placeholders
                        inter_mbs_diff_placeholders = {}
                        for gkey in all_ims_keys:
                            ph_key = f"sign_agree_inter_mbs_diff/{gkey}"
                            ph = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                            loss_dict[ph_key] = ph
                            inter_mbs_diff_placeholders[ph_key] = ph
                        avg_ims_diff_ph = torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")
                        loss_dict["sign_agree_inter_mbs_diff/avg"] = avg_ims_diff_ph
                        if isinstance(multi_loss_tensors, MultiLossTensors):
                            multi_loss_tensors._inter_mbs_diff_sign_placeholders = inter_mbs_diff_placeholders
                            multi_loss_tensors._avg_inter_mbs_diff_sign_placeholder = avg_ims_diff_ph
                # 添加图片统计信息 (当前 micro batch 的统计)
                # 使用 [count, 1/N] 格式:
                # - Megatron 跨 mbs 和 DP sum: [sum(count_i), N * (1/N)] = [total, 1]
                # - 最终: total / 1 = total (全局总数)
                if image_stats:
                    loss_dict["num_images"] = torch.tensor(
                        [image_stats.get("num_images", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["num_gen_images"] = torch.tensor(
                        [image_stats.get("num_gen_images", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["num_gen_cot_images"] = torch.tensor(
                        [image_stats.get("num_gen_cot_images", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["num_und_images"] = torch.tensor(
                        [image_stats.get("num_und_images", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["num_edit_images"] = torch.tensor(
                        [image_stats.get("num_edit_images", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    # 不同模态的训练行数统计 (同一 "/" 前缀 → TensorBoard 归为一栏)
                    loss_dict["modality_rows/text"] = torch.tensor(
                        [image_stats.get("num_text_rows", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["modality_rows/understanding"] = torch.tensor(
                        [image_stats.get("num_und_rows", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["modality_rows/generation"] = torch.tensor(
                        [image_stats.get("num_gen_rows", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["modality_rows/generation_cot"] = torch.tensor(
                        [image_stats.get("num_gen_cot_rows", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["modality_rows/generation_cot_text"] = torch.tensor(
                        [image_stats.get("num_gen_cot_text_rows", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )
                    loss_dict["modality_rows/editing"] = torch.tensor(
                        [image_stats.get("num_edit_rows", 0), denominator],
                        dtype=torch.float, device="cuda"
                    )

                # 返回多模态 loss 列表，用于独立 all_reduce + 独立 backward
                # 格式: (multi_loss_tensors, num_tokens, loss_dict)
                # num_tokens 设为 0（我们自己做 all_reduce，不使用 Megatron 的 token 统计）
                num_tokens = torch.tensor(0, dtype=torch.int, device="cuda")
                return multi_loss_tensors, num_tokens, loss_dict
            else:
                return output_tensor, {}

        return loss_func

    def _maybe_sample(self, state: GlobalState, model, batch: dict):
        """训练中定期采样: 用当前 batch 的数据做原图 vs 生成图对比，写入 TensorBoard。"""
        sample_every = self.sample_cfg.get("sample_every", 0)
        if sample_every <= 0:
            return

        iteration = state.train_state.step
        if iteration == 0 or iteration % sample_every != 0:
            return

        # 仅在最后一个 pipeline stage 执行采样
        if not parallel_state.is_pipeline_last_stage():
            return

        # TensorBoard writer 仅在特定 rank 上存在
        tb_writer = state.tensorboard_logger
        if tb_writer is None:
            return

        from model.transfusion.transfusion_sampler import sample_and_log_from_batch

        # Unwrap DDP + Float16Module (递归解包所有 .module 层)
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        tokenizer = state.tokenizer
        # Megatron NullTokenizer 不支持真正的 text tokenize/detokenize，用 HF tokenizer 替代
        model_id = self.sample_cfg.get("model_id")
        if model_id and not hasattr(tokenizer, 'encode'):
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Failed to load HF tokenizer from {model_id}: {e}")

        sample_and_log_from_batch(
            model=unwrapped,
            tokenizer=tokenizer,
            batch=batch,
            tb_writer=tb_writer,
            iteration=iteration,
            num_vis=self.sample_cfg.get("num_vis", 4),
            num_steps=self.sample_cfg.get("num_steps", 25),
            shift=self.sample_cfg.get("shift", 3.0),
            guidance_scale=self.sample_cfg.get("guidance_scale", 1.0),
            predict_x0=self.pipeline.predict_x0,
        )


# ============================================================================
# 注意：模态感知的梯度 Rescale 机制已移除
# ============================================================================
# 新方案：在 schedules.py 中实现多模态独立 loss 反传
# - training_step 返回 multi_loss_tensors: [(name, loss_tensor, weight), ...]
# - loss_func 返回 (multi_loss_tensors, None, loss_dict)
# - backward_step 对每个 loss 独立 backward，梯度自动累加
# - 每个模态的梯度自然按各自的样本数平均，不需要 rescale
# ============================================================================
