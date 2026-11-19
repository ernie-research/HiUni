"""Independent model-only checkpoint stream for Transfusion training.

The normal Megatron checkpoint cadence remains unchanged.  This extension adds
another cadence that saves standard, loadable distributed checkpoints without
optimizer, scheduler, RNG, or dataloader state.  It is intentionally unrelated
to offline EMA.
"""

from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from megatron.core.rerun_state_machine import get_rerun_state_machine


@dataclass(frozen=True)
class ModelOnlyCheckpointConfig:
    """Configuration for periodic checkpoints containing model weights only."""

    enabled: bool = False
    interval: int = 50
    output_dir: str | None = None
    most_recent_k: int = -1

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None, *, checkpoint_save: str | None
    ) -> "ModelOnlyCheckpointConfig":
        """Build and validate model-only checkpoint settings from a mapping."""
        values = dict(raw or {})
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "unknown model_only_checkpoint fields: " + ", ".join(unknown)
            )
        config = cls(**values)
        if not config.enabled:
            return config
        if checkpoint_save is None:
            raise ValueError(
                "model_only_checkpoint.enabled=true requires checkpoint.save"
            )
        if config.interval <= 0:
            raise ValueError("model_only_checkpoint.interval must be positive")
        if config.most_recent_k == 0 or config.most_recent_k < -1:
            raise ValueError(
                "model_only_checkpoint.most_recent_k must be -1 or positive"
            )
        default = str(Path(checkpoint_save).with_name(
            Path(checkpoint_save).name + "-model-only"
        ))
        return replace(config, output_dir=config.output_dir or default)


@contextlib.contextmanager
def _model_only_config(state: Any, config: ModelOnlyCheckpointConfig):
    """Temporarily configure Bridge to save model weights without runtime state."""
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
    ckpt.save = str(config.output_dir)
    ckpt.save_optim = False
    ckpt.save_rng = False
    ckpt.async_save = False
    ckpt.most_recent_k = config.most_recent_k
    dataset.dataloader_save = None
    try:
        yield
    finally:
        ckpt.save = original["save"]
        ckpt.save_optim = original["save_optim"]
        ckpt.save_rng = original["save_rng"]
        ckpt.async_save = original["async_save"]
        ckpt.most_recent_k = original["most_recent_k"]
        dataset.dataloader_save = original["dataloader_save"]


@contextlib.contextmanager
def _omit_non_model_state(checkpointing: Any):
    """Keep the standard checkpoint format while emitting model tensors only."""
    original_get_rng_state = checkpointing.get_rng_state
    original_rerun_state_dict = get_rerun_state_machine().state_dict
    checkpointing.get_rng_state = lambda **_kwargs: None
    get_rerun_state_machine().state_dict = lambda **_kwargs: None
    try:
        yield
    finally:
        checkpointing.get_rng_state = original_get_rng_state
        get_rerun_state_machine().state_dict = original_rerun_state_dict


def install_model_only_checkpointing(
    config: Any, raw_config: Mapping[str, Any] | None
) -> ModelOnlyCheckpointConfig | None:
    """Install a 50-step model-only save hook without editing Bridge sources."""
    extension = ModelOnlyCheckpointConfig.from_mapping(
        raw_config, checkpoint_save=config.checkpoint.save
    )
    if not extension.enabled:
        return None

    pretrain_module = importlib.import_module("megatron.bridge.training.pretrain")
    train_module = importlib.import_module("megatron.bridge.training.train")
    if hasattr(pretrain_module, "_transfusion_model_only_original_train"):
        raise RuntimeError("model-only checkpoint hooks are already installed")

    original_train = pretrain_module.train

    def argument(name: str, position: int, args, kwargs):
        """Read a train-step argument from keywords or its positional slot."""
        if name in kwargs:
            return kwargs[name]
        if len(args) <= position:
            raise TypeError(f"missing required train_step argument: {name}")
        return args[position]

    def make_train_step_with_model_only_save(inner_train_step):
        """Wrap a train step with periodic model-only checkpoint saving."""
        def train_step_with_model_only_save(*args, **kwargs):
            """Run one train step and save at the configured interval."""
            result = inner_train_step(*args, **kwargs)
            model = argument("model", 2, args, kwargs)
            state = argument("global_state", 5, args, kwargs)
            skipped_iter = int(result[1])
            next_step = int(state.train_state.step) + 1
            if skipped_iter == 0 and next_step % extension.interval == 0:
                if train_module.should_disable_forward_pre_hook(
                    state.cfg.ddp.use_megatron_fsdp,
                    state.cfg.optimizer.use_distributed_optimizer,
                    state.cfg.ddp.overlap_param_gather,
                ):
                    train_module.force_param_sync(model)
                checkpointing = importlib.import_module(
                    "megatron.bridge.training.checkpointing"
                )
                # Bridge increments TrainState immediately after train_step
                # returns.  Temporarily expose the completed step to the
                # checkpoint writer.
                current_step = state.train_state.step
                state.train_state.step = next_step
                try:
                    with (
                        _model_only_config(state, extension),
                        _omit_non_model_state(checkpointing),
                    ):
                        checkpointing.save_checkpoint(
                            state,
                            model,
                            optimizer=None,
                            opt_param_scheduler=None,
                            num_floating_point_operations_so_far=(
                                state.train_state.floating_point_operations_so_far
                            ),
                            checkpointing_context={},
                            train_data_iterator=None,
                        )
                finally:
                    state.train_state.step = current_step
            return result

        return train_step_with_model_only_save

    def train_with_model_only_checkpoints(
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
        """Run training with the model-only train-step wrapper installed."""
        previous_train_step = train_module.train_step
        train_module.train_step = make_train_step_with_model_only_save(
            previous_train_step
        )
        try:
            return original_train(
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
        finally:
            train_module.train_step = previous_train_step

    pretrain_module._transfusion_model_only_original_train = original_train
    pretrain_module.train = train_with_model_only_checkpoints
    return extension
