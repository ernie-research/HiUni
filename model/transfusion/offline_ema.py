"""Checkpoint-sampled offline EMA for dense Pixel-DiT training.

This module deliberately does not keep shadow weights during training.  It
writes model-parameter snapshots after successful optimizer updates, computes
EMA from those snapshots once training is complete, overlays the result on the
live distributed model, and asks Megatron Bridge to save a normal full model
checkpoint.  Frozen parameters are not duplicated in snapshots by default.

The result is an EMA over sampled checkpoints.  With ``decay_mode=per_step``
the transition between snapshots at steps ``a`` and ``b`` uses
``decay ** (b - a)``.  This preserves the requested time constant but cannot be
identical to an EMA updated from every unseen intermediate optimizer step.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.utils import unwrap_model


FORMAT_VERSION = 1


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _barrier() -> None:
    if _dist_ready():
        dist.barrier()


def _safe_parallel_rank(name: str) -> int:
    getter = getattr(mpu, name, None)
    if getter is None:
        return 0
    try:
        value = getter()
        return 0 if value is None else int(value)
    except (AssertionError, RuntimeError, TypeError):
        return 0


def _model_coordinate() -> tuple[int, int, int, int]:
    """Return ranks that can own different local model parameter shards."""
    return (
        _safe_parallel_rank("get_pipeline_model_parallel_rank"),
        _safe_parallel_rank("get_tensor_model_parallel_rank"),
        _safe_parallel_rank("get_expert_model_parallel_rank"),
        _safe_parallel_rank("get_expert_tensor_parallel_rank"),
    )


def _coordinate_label(coordinate: Sequence[int]) -> str:
    pp, tp, ep, etp = coordinate
    return f"pp{pp:03d}-tp{tp:03d}-ep{ep:03d}-etp{etp:03d}"


def _model_chunks(model: Any) -> list[torch.nn.Module]:
    chunks = unwrap_model(model)
    if isinstance(chunks, torch.nn.Module):
        return [chunks]
    return list(chunks)


def _parameter_key(chunk_index: int, chunk_count: int, name: str) -> str:
    prefix = "model" if chunk_count == 1 else f"model{chunk_index}"
    return f"{prefix}.{name}"


def _parameter_lookup_key(chunk_index: int, chunk_count: int, name: str) -> str:
    """Match module sharded-state-dict keys (one chunk has no ``model.``)."""
    if chunk_count == 1:
        return name
    return f"model{chunk_index}.{name}"


def _selected_parameters(
    model: Any, *, trainable_only: bool
) -> dict[str, torch.nn.Parameter]:
    chunks = _model_chunks(model)
    selected: dict[str, torch.nn.Parameter] = {}
    for chunk_index, chunk in enumerate(chunks):
        for name, parameter in chunk.named_parameters():
            if trainable_only and not parameter.requires_grad:
                continue
            key = _parameter_key(chunk_index, len(chunks), name)
            if key in selected:
                raise RuntimeError(f"duplicate offline EMA parameter key: {key}")
            selected[key] = parameter
    if not selected:
        scope = "trainable" if trainable_only else "model"
        raise RuntimeError(f"offline EMA selected no {scope} parameters")
    return selected


@dataclass(frozen=True)
class OfflineEMAConfig:
    """Configuration for checkpoint-sampled offline EMA."""

    enabled: bool = False
    snapshot_interval: int = 50
    decay: float = 0.9999
    decay_mode: str = "per_step"
    trainable_only: bool = True
    snapshot_dtype: str = "float32"
    snapshot_dir: str | None = None
    output_dir: str | None = None
    keep_snapshots: bool = True
    auto_finalize: bool = True

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None, *, checkpoint_save: str | None
    ) -> "OfflineEMAConfig":
        """Build and validate EMA settings from a mapping."""
        values = dict(raw or {})
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown offline_ema fields: {', '.join(unknown)}")
        config = cls(**values)
        if not config.enabled:
            return config
        if checkpoint_save is None:
            raise ValueError("offline_ema.enabled=true requires checkpoint.save")
        if config.snapshot_interval <= 0:
            raise ValueError("offline_ema.snapshot_interval must be positive")
        if not 0.0 <= config.decay < 1.0:
            raise ValueError("offline_ema.decay must be in [0, 1)")
        if config.decay_mode not in {"per_step", "per_snapshot"}:
            raise ValueError(
                "offline_ema.decay_mode must be 'per_step' or 'per_snapshot'"
            )
        if config.snapshot_dtype not in {"float32", "bfloat16"}:
            raise ValueError(
                "offline_ema.snapshot_dtype must be 'float32' or 'bfloat16'"
            )

        default_root = Path(checkpoint_save).with_name(
            Path(checkpoint_save).name + "-offline-ema"
        )
        return replace(
            config,
            snapshot_dir=config.snapshot_dir or str(default_root / "snapshots"),
            output_dir=config.output_dir or str(default_root / "checkpoint"),
        )


class OfflineEMAManager:
    """Save sparse parameter snapshots and finalize them into a full EMA ckpt."""

    def __init__(self, config: OfflineEMAConfig):
        if not config.enabled:
            raise ValueError("OfflineEMAManager requires enabled=true")
        self.config = config
        self.snapshot_root = Path(str(config.snapshot_dir))
        self.output_root = Path(str(config.output_dir))
        self.coordinate = _model_coordinate()
        self.coordinate_label = _coordinate_label(self.coordinate)
        self.coordinate_leader = self._resolve_coordinate_leader()
        self._coordinate_initialized = not _dist_ready()

    def _refresh_coordinate(self) -> None:
        """Resolve parallel coordinates after Megatron initializes process groups."""
        if self._coordinate_initialized:
            return
        self.coordinate = _model_coordinate()
        self.coordinate_label = _coordinate_label(self.coordinate)
        self.coordinate_leader = self._resolve_coordinate_leader()
        self._coordinate_initialized = True

    def _resolve_coordinate_leader(self) -> int:
        if not _dist_ready():
            return 0
        rows: list[tuple[tuple[int, int, int, int], int] | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(rows, (self.coordinate, _rank()))
        return min(rank for coordinate, rank in rows if coordinate == self.coordinate)

    @property
    def is_coordinate_leader(self) -> bool:
        """Return whether this rank writes the local model-coordinate snapshot."""
        return _rank() == self.coordinate_leader

    def should_snapshot(self, step: int) -> bool:
        """Return whether ``step`` reaches the configured snapshot cadence."""
        return step > 0 and step % self.config.snapshot_interval == 0

    def _step_dir(self, step: int) -> Path:
        return self.snapshot_root / f"step_{step:08d}"

    def _coordinate_file(self, step: int) -> Path:
        return self._step_dir(step) / f"{self.coordinate_label}.pt"

    def _snapshot_complete(self, step: int) -> bool:
        marker = self._step_dir(step) / "COMPLETE"
        if _dist_ready():
            complete = False
            if _rank() == 0:
                complete = marker.is_file() and marker.stat().st_size > 0
            status = torch.tensor(
                [int(complete)], device="cuda" if torch.cuda.is_available() else "cpu"
            )
            dist.broadcast(status, src=0)
            return bool(status.item())
        return marker.is_file() and marker.stat().st_size > 0

    def save_snapshot(self, model: Any, step: int) -> None:
        """Persist selected model weights after a successful optimizer step."""
        if step <= 0:
            raise ValueError(f"offline EMA snapshot step must be positive, got {step}")
        self._refresh_coordinate()
        if self._snapshot_complete(step):
            return

        step_dir = self._step_dir(step)
        if _rank() == 0:
            step_dir.mkdir(parents=True, exist_ok=True)
        _barrier()

        parameter_count = None
        if self.is_coordinate_leader:
            parameters = _selected_parameters(
                model, trainable_only=self.config.trainable_only
            )
            dtype = (
                torch.float32
                if self.config.snapshot_dtype == "float32"
                else torch.bfloat16
            )
            tensors = {
                key: parameter.detach().to(device="cpu", dtype=dtype).clone()
                for key, parameter in parameters.items()
            }
            parameter_metadata = {
                key: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                }
                for key, tensor in tensors.items()
            }
            parameter_count = len(parameter_metadata)
            destination = self._coordinate_file(step)
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(
                {
                    "format_version": FORMAT_VERSION,
                    "step": step,
                    "coordinate": list(self.coordinate),
                    "trainable_only": self.config.trainable_only,
                    "parameters": tensors,
                },
                temporary,
            )
            os.replace(temporary, destination)
        _barrier()

        if _rank() == 0:
            files = sorted(path.name for path in step_dir.glob("*.pt"))
            if not files:
                raise RuntimeError(f"offline EMA snapshot {step} has no rank files")
            manifest = {
                "format_version": FORMAT_VERSION,
                "step": step,
                "snapshot_interval": self.config.snapshot_interval,
                "snapshot_dtype": self.config.snapshot_dtype,
                "trainable_only": self.config.trainable_only,
                "coordinate_files": files,
                "rank0_parameter_count": parameter_count,
            }
            (step_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            (step_dir / "COMPLETE").write_text(f"step={step}\n")
        _barrier()

    def snapshot_steps(self) -> list[int]:
        """Return completed snapshot step numbers in ascending order."""
        steps = []
        if self.snapshot_root.is_dir():
            for path in self.snapshot_root.glob("step_*"):
                try:
                    step = int(path.name.removeprefix("step_"))
                except ValueError:
                    continue
                marker = path / "COMPLETE"
                if marker.is_file() and marker.stat().st_size > 0:
                    steps.append(step)
        return sorted(set(steps))

    def compute_local_ema(self) -> tuple[dict[str, torch.Tensor], list[int]]:
        """Compute this model-coordinate's EMA in CPU float32."""
        self._refresh_coordinate()
        steps = self.snapshot_steps()
        if not steps:
            raise RuntimeError(f"no complete offline EMA snapshots under {self.snapshot_root}")

        accumulator: dict[str, torch.Tensor] = {}
        expected_keys: set[str] | None = None
        previous_step: int | None = None
        for step in steps:
            source = self._coordinate_file(step)
            if not source.is_file():
                raise RuntimeError(
                    f"snapshot step {step} is missing coordinate file {source.name}"
                )
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if payload.get("format_version") != FORMAT_VERSION:
                raise RuntimeError(f"unsupported offline EMA snapshot: {source}")
            if int(payload.get("step", -1)) != step:
                raise RuntimeError(f"offline EMA snapshot step mismatch: {source}")
            tensors = payload.get("parameters", {})
            keys = set(tensors)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                missing = sorted(expected_keys - keys)
                extra = sorted(keys - expected_keys)
                raise RuntimeError(
                    f"offline EMA parameter set changed at step {step}; "
                    f"missing={missing[:8]} extra={extra[:8]}"
                )

            if previous_step is None:
                accumulator = {
                    key: tensor.float().clone() for key, tensor in tensors.items()
                }
            else:
                delta_steps = step - previous_step
                if delta_steps <= 0:
                    raise RuntimeError("offline EMA snapshot steps must increase")
                transition_decay = self.config.decay
                if self.config.decay_mode == "per_step":
                    transition_decay = self.config.decay**delta_steps
                for key, tensor in tensors.items():
                    accumulator[key].mul_(transition_decay).add_(
                        tensor.float(), alpha=1.0 - transition_decay
                    )
            previous_step = step
        return accumulator, steps

    def apply_local_ema(
        self, model: Any, ema: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply EMA weights and return CPU copies of the final live weights."""
        self._refresh_coordinate()
        parameters = _selected_parameters(
            model, trainable_only=self.config.trainable_only
        )
        shard_lookup = self._parameter_shard_lookup(model)
        if set(parameters) != set(ema):
            missing = sorted(set(parameters) - set(ema))
            extra = sorted(set(ema) - set(parameters))
            raise RuntimeError(
                "offline EMA/live model parameter mismatch; "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        originals = {
            key: parameter.detach().to(device="cpu").clone()
            for key, parameter in parameters.items()
        }
        with torch.no_grad():
            for key, parameter in parameters.items():
                value = ema[key]
                target = value
                shard = shard_lookup.get(key)
                if (
                    shard is not None
                    and shard.flattened_range is not None
                    and tuple(value.shape) != tuple(parameter.shape)
                ):
                    start = int(shard.flattened_range.start or 0)
                    stop = int(shard.flattened_range.stop)
                    target = value.flatten()[start:stop].reshape(parameter.shape)
                if tuple(target.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"offline EMA shape mismatch for {key}: "
                        f"{tuple(target.shape)} != {tuple(parameter.shape)}"
                    )
                parameter.copy_(
                    target.to(device=parameter.device, dtype=parameter.dtype)
                )
        return originals

    def restore_local_parameters(
        self, model: Any, originals: Mapping[str, torch.Tensor]
    ) -> None:
        """Restore the live model parameters saved before EMA application."""
        parameters = _selected_parameters(
            model, trainable_only=self.config.trainable_only
        )
        with torch.no_grad():
            for key, parameter in parameters.items():
                parameter.copy_(
                    originals[key].to(device=parameter.device, dtype=parameter.dtype)
                )

    def _parameter_shard_lookup(self, model: Any) -> dict[str, Any]:
        """Map snapshot names to exact MCore sharding descriptors."""
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        chunks = _model_chunks(model)
        lookup = {}
        for chunk_index, chunk in enumerate(chunks):
            if not hasattr(chunk, "sharded_state_dict"):
                continue
            sharded = chunk.sharded_state_dict()
            for name, _parameter in chunk.named_parameters():
                lookup_key = _parameter_lookup_key(
                    chunk_index, len(chunks), name
                )
                shard = sharded.get(lookup_key)
                if isinstance(shard, ShardedTensor):
                    lookup[_parameter_key(chunk_index, len(chunks), name)] = shard
        return lookup

    @contextlib.contextmanager
    def _ema_checkpoint_config(self, state: Any):
        ckpt = state.cfg.checkpoint
        dataset = state.cfg.dataset
        original = {
            "save": ckpt.save,
            "save_optim": ckpt.save_optim,
            "save_rng": ckpt.save_rng,
            "async_save": ckpt.async_save,
            "most_recent_k": ckpt.most_recent_k,
            "dataloader_save": dataset.dataloader_save,
        }
        ckpt.save = str(self.output_root)
        ckpt.save_optim = False
        ckpt.save_rng = False
        ckpt.async_save = False
        ckpt.most_recent_k = -1
        dataset.dataloader_save = None
        try:
            yield
        finally:
            for name, value in original.items():
                if name == "dataloader_save":
                    dataset.dataloader_save = value
                else:
                    setattr(ckpt, name, value)

    def finalize_and_save_full_checkpoint(
        self,
        *,
        state: Any,
        model: Any,
        num_floating_point_operations_so_far: float,
    ) -> None:
        """Compute sampled EMA and save it as a normal full Megatron ckpt."""
        final_step = int(state.train_state.step)
        if not self._snapshot_complete(final_step):
            self.save_snapshot(model, final_step)

        ema, steps = self.compute_local_ema()
        originals = self.apply_local_ema(model, ema)
        _barrier()
        try:
            checkpointing = importlib.import_module(
                "megatron.bridge.training.checkpointing"
            )
            with self._ema_checkpoint_config(state):
                checkpointing.save_checkpoint(
                    state,
                    model,
                    optimizer=None,
                    opt_param_scheduler=None,
                    num_floating_point_operations_so_far=(
                        num_floating_point_operations_so_far
                    ),
                    checkpointing_context={},
                    train_data_iterator=None,
                )
            if _rank() == 0:
                self.output_root.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "format_version": FORMAT_VERSION,
                    "kind": "checkpoint_sampled_offline_ema",
                    "final_step": final_step,
                    "snapshot_steps": steps,
                    "snapshot_interval": self.config.snapshot_interval,
                    "decay": self.config.decay,
                    "decay_mode": self.config.decay_mode,
                    "trainable_only": self.config.trainable_only,
                    "snapshot_dtype": self.config.snapshot_dtype,
                }
                (self.output_root / "offline_ema_manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
            _barrier()
            if not self.config.keep_snapshots:
                if _rank() == 0:
                    shutil.rmtree(self.snapshot_root)
                _barrier()
        finally:
            self.restore_local_parameters(model, originals)
            _barrier()


def install_offline_ema(
    config: Any, raw_config: Mapping[str, Any] | None
) -> OfflineEMAManager | None:
    """Install snapshot/finalization hooks without modifying Bridge sources."""
    ema_config = OfflineEMAConfig.from_mapping(
        raw_config, checkpoint_save=config.checkpoint.save
    )
    if not ema_config.enabled:
        return None

    manager = OfflineEMAManager(ema_config)
    pretrain_module = importlib.import_module("megatron.bridge.training.pretrain")
    train_module = importlib.import_module("megatron.bridge.training.train")
    if hasattr(pretrain_module, "_transfusion_offline_ema_original_train"):
        raise RuntimeError("offline EMA hooks are already installed")

    original_train = pretrain_module.train

    def _argument(name: str, position: int, args, kwargs):
        if name in kwargs:
            return kwargs[name]
        if len(args) <= position:
            raise TypeError(f"missing required train_step argument: {name}")
        return args[position]

    def make_train_step_with_snapshots(inner_train_step):
        def train_step_with_snapshots(*args, **kwargs):
            result = inner_train_step(*args, **kwargs)
            model = _argument("model", 2, args, kwargs)
            state = _argument("global_state", 5, args, kwargs)
            skipped_iter = int(result[1])
            next_step = int(state.train_state.step) + 1
            if skipped_iter == 0 and manager.should_snapshot(next_step):
                if train_module.should_disable_forward_pre_hook(
                    state.cfg.ddp.use_megatron_fsdp,
                    state.cfg.optimizer.use_distributed_optimizer,
                    state.cfg.ddp.overlap_param_gather,
                ):
                    train_module.force_param_sync(model)
                manager.save_snapshot(model, next_step)
            return result

        return train_step_with_snapshots

    def train_with_offline_ema(
        forward_step_func,
        model,
        optimizer,
        scheduler,
        train_data_iterator,
        valid_data_iterator,
        state,
        checkpointing_context,
        pg_collection,
    ):
        # Chain onto whatever train_step is installed right now so that other
        # Transfusion extensions (e.g. model-only checkpoints) keep working.
        previous_train_step = train_module.train_step
        train_module.train_step = make_train_step_with_snapshots(previous_train_step)
        completed = False
        try:
            result = original_train(
                forward_step_func,
                model,
                optimizer,
                scheduler,
                train_data_iterator,
                valid_data_iterator,
                state,
                checkpointing_context,
                pg_collection,
            )
            completed = True
        finally:
            train_module.train_step = previous_train_step
        if completed and ema_config.auto_finalize:
            if train_module.should_disable_forward_pre_hook(
                state.cfg.ddp.use_megatron_fsdp,
                state.cfg.optimizer.use_distributed_optimizer,
                state.cfg.ddp.overlap_param_gather,
            ):
                train_module.force_param_sync(model)
            manager.finalize_and_save_full_checkpoint(
                state=state,
                model=model,
                num_floating_point_operations_so_far=(
                    state.train_state.floating_point_operations_so_far
                ),
            )
        return result

    pretrain_module._transfusion_offline_ema_original_train = original_train
    pretrain_module.train = train_with_offline_ema
    return manager
