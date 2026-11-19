"""Per-component LoRA for the dense Pixel-DiT.

The model has three trunks that are adapted independently:

* ``backbone``  - the Qwen3-VL language model (``language_model.*``, ``time_token``)
* ``pixel_dit`` - the Pixel-DiT head (``diffusion_head_a/b``, gates, pixel/time projections)
* ``vit``       - the vision encoder (``vision_model.*``)

Each component has its own rank, alpha, dropout, target modules, layer range and
list of modules to train in full.  ``TransfusionLoRA`` is a Megatron-Bridge
``PEFT``: assign it to ``ConfigContainer.peft`` and Bridge loads the base
weights, applies the adapters before DDP wrapping and saves adapter-only
checkpoints.  ``merge_lora_into_model`` folds trained adapters back into the
base weights (see ``recipe/transfusion/scripts/merge_lora_checkpoint.py``).
"""

import logging
import re
import types
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.lora_layers import LinearAdapter, LoRALinear, TELinearAdapter
from megatron.bridge.peft.utils import wildcard_match
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

logger = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )

    _TE_LINEARS = (te.Linear, TEColumnParallelLinear, TELayerNormColumnParallelLinear, TERowParallelLinear)
except ImportError:  # pragma: no cover - TE is part of the training container
    _TE_LINEARS = ()

LINEAR_TYPES = (nn.Linear, ColumnParallelLinear, RowParallelLinear) + _TE_LINEARS
ADAPTER_TYPES = (LinearAdapter, LoRALinear, TELinearAdapter)

# Top-level submodules of TransfusionQwen3VLModel owned by each component.
COMPONENT_ROOTS = {
    "backbone": ("language_model", "time_token"),
    "pixel_dit": (
        "diffusion_head_a", "diffusion_head_b", "pixel_dit_input_projection", "pixel_proj",
        "pixel_dit_gate", "pixel_dit_ref_gate", "pixel_dit_time_embeddings", "time_embeddings",
    ),
    "vit": ("vision_model",),
}
# Module path of one layer of each component's layer stack; group 1 is the local index.
LAYER_PATTERNS = {
    "backbone": re.compile(r"^language_model\.decoder\.layers\.(\d+)$"),
    "pixel_dit": re.compile(r"^diffusion_head_a\.(\d+)$"),
    "vit": re.compile(r"^vision_model\.blocks\.(\d+)$"),
}
# Parameter attributes the optimizer / grad sync read; copied onto adapter weights.
INHERITED_PARAM_ATTRS = ("is_vit_parameter", "average_gradients_across_tp_domain")


@dataclass
class LoRAComponent:
    """LoRA settings for one component.

    ``target_modules`` entries match a linear module by its leaf name
    (``q_proj``), a dotted suffix (``self_attn.q_proj``) or a ``*`` wildcard over
    the full module path (``diffusion_head_a.*.mlp.*``).  ``layers`` restricts
    adapters to these layer indices (global, 0-based) of the component's layer
    stack; modules outside the stack are then skipped.  ``trainable_modules``
    are trained in full (same matching rules, prefixes included); a module
    cannot be both a LoRA target and fully trained.
    """

    enabled: bool = False
    dim: int = 16
    alpha: float = 32
    dropout: float = 0.0
    dropout_position: str = "pre"
    target_modules: List[str] = field(default_factory=list)
    layers: Optional[Union[List[int], str]] = None
    trainable_modules: List[str] = field(default_factory=list)

    def layer_set(self):
        """Return the selected layer indices, or None for all layers."""
        if self.layers is None:
            return None
        if isinstance(self.layers, int):
            return {self.layers}
        if isinstance(self.layers, str):
            selected = set()
            for part in filter(None, (p.strip() for p in self.layers.split(","))):
                lo, _, hi = part.partition("-")
                selected.update(range(int(lo), int(hi or lo) + 1))
            return selected
        return {int(i) for i in self.layers}


DEFAULT_TARGETS = {
    "backbone": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    # torch DiT head names, then the Megatron/TE head names (dit_head_impl: megatron).
    "pixel_dit": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    "vit": ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"],
}


def matches(pattern: str, path: str) -> bool:
    """Leaf name, dotted suffix or wildcard match of ``pattern`` against ``path``."""
    return path == pattern or path.endswith("." + pattern) or bool(wildcard_match(pattern, path))


def _under(pattern: str, path: str) -> bool:
    """Like ``matches`` but also true for everything inside a matched module."""
    parts = path.split(".")
    return any(matches(pattern, ".".join(parts[:i])) for i in range(1, len(parts) + 1))


def _component_of(path: str):
    root = path.split(".", 1)[0]
    for name, roots in COMPONENT_ROOTS.items():
        if root in roots:
            return name
    return None


def _copy_param_attrs(src: nn.Parameter, dst: nn.Parameter, names=None):
    for key, value in vars(src).items():
        if (names is None and not key.startswith("_")) or (names is not None and key in names):
            setattr(dst, key, value)


def _component_field(name):
    return field(default_factory=lambda: LoRAComponent(target_modules=list(DEFAULT_TARGETS[name])))


@dataclass
class TransfusionLoRA(PEFT):
    """Megatron-Bridge PEFT with separate backbone / Pixel-DiT / ViT LoRA settings."""

    backbone: LoRAComponent = _component_field("backbone")
    pixel_dit: LoRAComponent = _component_field("pixel_dit")
    vit: LoRAComponent = _component_field("vit")

    COMPONENTS = ("backbone", "pixel_dit", "vit")

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "TransfusionLoRA":
        """Build from the YAML ``lora:`` section (or a serialized ``peft:`` section)."""
        cfg = {k: v for k, v in dict(cfg or {}).items() if k not in ("_target_", "params_to_save")}
        unknown = sorted(set(cfg) - set(cls.COMPONENTS))
        if unknown:
            raise ValueError(f"Unknown lora components {unknown}; expected {list(cls.COMPONENTS)}")
        valid = {f.name for f in fields(LoRAComponent)}
        defaults = cls()
        kwargs = {}
        for name, sub in cfg.items():
            sub = {k: v for k, v in dict(sub or {}).items() if k != "_target_"}
            bad = sorted(set(sub) - valid)
            if bad:
                raise ValueError(f"Unknown keys {bad} in lora.{name}; expected {sorted(valid)}")
            base = getattr(defaults, name)
            kwargs[name] = LoRAComponent(**{**{f.name: getattr(base, f.name) for f in fields(base)}, **sub})
        peft = cls(**kwargs)
        peft.validate()
        return peft

    def enabled_components(self):
        return [name for name in self.COMPONENTS if getattr(self, name).enabled]

    def validate(self):
        if not self.enabled_components():
            raise ValueError("lora: enable at least one of backbone / pixel_dit / vit")
        for name in self.enabled_components():
            comp = getattr(self, name)
            if comp.target_modules and comp.dim <= 0:
                raise ValueError(f"lora.{name}.dim must be > 0, got {comp.dim}")
            if comp.dropout_position not in ("pre", "post"):
                raise ValueError(f"lora.{name}.dropout_position must be 'pre' or 'post'")
            if not comp.target_modules and not comp.trainable_modules:
                raise ValueError(f"lora.{name} is enabled but has no target_modules or trainable_modules")
            comp.layer_set()  # parse errors surface here

    def transform(self, module, name=None, prefix=None):  # pragma: no cover - __call__ does the work
        return module

    def __call__(self, model, training: bool = True):
        chunks = model if isinstance(model, list) else [model]
        self.stats = {}
        for chunk in chunks:
            inner = chunk.module if isinstance(chunk, torch.nn.parallel.DistributedDataParallel) else chunk
            for comp, values in self._apply(inner, training).items():
                total = self.stats.setdefault(comp, dict.fromkeys(values, 0))
                for key, value in values.items():
                    total[key] += value
            inner.train(mode=training)
        _log_rank0("TransfusionLoRA: " + "; ".join(
            f"{comp}: {s['lora_modules']} adapted modules, {s['lora_params']:,} LoRA params, "
            f"{s['full_params']:,} fully trained params" for comp, s in self.stats.items()))
        return model

    @staticmethod
    def _layer_index(path, modules):
        """(component, global layer index) of the layer containing ``path``, if any."""
        parts = path.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            for comp, pattern in LAYER_PATTERNS.items():
                found = pattern.match(prefix)
                if found:
                    number = getattr(modules[prefix], "layer_number", None) if comp == "backbone" else None
                    return comp, (number - 1 if isinstance(number, int) else int(found.group(1)))
        return None, None

    def _select(self, model):
        modules = dict(model.named_modules())
        params = dict(model.named_parameters())
        targets, full = [], {}
        for comp_name in self.enabled_components():
            comp = getattr(self, comp_name)
            layers = comp.layer_set()
            found = []
            for path, module in modules.items():
                if _component_of(path) != comp_name or not any(matches(p, path) for p in comp.target_modules):
                    continue
                if not isinstance(module, LINEAR_TYPES) or isinstance(module, ADAPTER_TYPES):
                    continue
                index = self._layer_index(path, modules)[1]
                if layers is not None and index not in layers:
                    continue
                found.append((comp_name, path, index))
            if comp.target_modules and not found:
                raise ValueError(f"lora.{comp_name}.target_modules {comp.target_modules} "
                                 f"(layers={comp.layers}) matched no linear module")
            targets += found
            for pattern in comp.trainable_modules:
                hit = [n for n in params if _component_of(n) == comp_name and _under(pattern, n)]
                if not hit:
                    raise ValueError(f"lora.{comp_name}.trainable_modules entry {pattern!r} matched no parameter")
                full.update(dict.fromkeys(hit, comp_name))
        overlap = sorted(n for n in full for _, path, _ in targets if n.startswith(path + "."))
        if overlap:
            raise ValueError(f"Modules cannot be both LoRA targets and trainable_modules: {overlap[:5]}")
        return targets, full

    def _apply(self, model, training):
        trainable_before = {n for n, p in model.named_parameters() if p.requires_grad}
        targets, full = self._select(model)
        stats = {c: {"lora_modules": 0, "lora_params": 0, "full_params": 0} for c in self.enabled_components()}
        for param in model.parameters():
            param.requires_grad = False
        params = dict(model.named_parameters())
        for name, comp_name in full.items():
            # Parameters the model or the freeze_* flags froze by design stay frozen.
            if training and name in trainable_before:
                params[name].requires_grad = True
                stats[comp_name]["full_params"] += params[name].numel()
        wrappers = {}
        for comp_name, path, layer_index in targets:
            comp = getattr(self, comp_name)
            if comp_name not in wrappers:
                wrappers[comp_name] = LoRA(
                    target_modules=["*"], dim=comp.dim, alpha=comp.alpha, dropout=comp.dropout,
                    dropout_position=comp.dropout_position,
                    # Bridge's "xavier" fills nn.Linear adapters with U(0, 1); kaiming is the usual LoRA init.
                    lora_A_init_method="kaiming", lora_B_init_method="zero",
                )
            parent_path, _, leaf = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            old = getattr(parent, leaf)
            old_params = dict(old.named_parameters(recurse=False))
            new = wrappers[comp_name].transform(old, leaf, parent_path)
            if isinstance(new, LoRALinear):
                lora_params = list(new.adapter.parameters())
                if comp.layers is not None and layer_index is not None:
                    new.lora_layer_index = layer_index
                    new.sharded_state_dict = types.MethodType(_per_layer_adapter_sharded_state_dict, new)
            else:
                new.dropout_position = comp.dropout_position  # Bridge does not forward it for nn.Linear
                lora_params = list(new.lora_a.parameters()) + list(new.lora_b.parameters())
                for key, param in new.named_parameters(recurse=False):
                    if key in old_params:
                        _copy_param_attrs(old_params[key], param)
            for param in lora_params:
                if "weight" in old_params:
                    _copy_param_attrs(old_params["weight"], param, INHERITED_PARAM_ATTRS)
                param.requires_grad = training
            setattr(parent, leaf, new)
            stats[comp_name]["lora_modules"] += 1
            stats[comp_name]["lora_params"] += sum(p.numel() for p in lora_params)
        return stats


def _per_layer_adapter_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
    """``LoRALinear.sharded_state_dict`` that stores the adapter under a per-layer key.

    Megatron checkpoints a decoder layer's tensors as slices of one tensor stacked over all
    layers (leading offset ``(0, layer, num_layers)``); adapters on a subset of the layers
    would leave that stack incomplete.  The wrapped linear keeps the stacked layout; the
    adapter drops the layer axis and gets ``adapter_layer<N>`` in its checkpoint key (the
    state-dict key is unchanged).
    """
    index = self.lora_layer_index
    sharded = self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata)
    offsets = tuple(sharded_offsets)
    if offsets and offsets[0][0] == 0 and offsets[0][1] == index:
        offsets = offsets[1:]
    real, marked = f"{prefix}adapter.", f"{prefix}adapter_layer{index}."
    for key, value in self.adapter.sharded_state_dict(marked, offsets, metadata).items():
        sharded[real + key[len(marked):]] = value
    return sharded


@torch.no_grad()
def merge_lora_into_model(model: nn.Module) -> int:
    """Fold every LoRA adapter into its base weight and restore the plain module.

    ``LinearAdapter`` / ``TELinearAdapter`` keep the base weight at the original
    FQN, so the merged weight is copied into a plain ``nn.Linear`` / ``te.Linear``;
    ``LoRALinear`` is replaced by its wrapped Megatron linear.  The resulting state
    dict has exactly the base model's keys.  Returns the number of merged modules.
    """
    merged = 0
    for path, module in list(model.named_modules()):
        if not isinstance(module, ADAPTER_TYPES):
            continue
        parent_path, _, leaf = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        if isinstance(module, LoRALinear):
            base, adapter = module.to_wrap, module.adapter
            weight = base.weight
            delta = (adapter.alpha / adapter.dim) * (
                adapter.linear_out.weight.float() @ adapter.linear_in.weight.float())
        else:
            if isinstance(module, LinearAdapter):
                base = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None,
                                 device=module.weight.device, dtype=module.weight.dtype)
            else:
                has_bias = module.bias is not None and module.bias.shape[0] != 0
                base = te.Linear(module.in_features, module.out_features, bias=has_bias,
                                 device=module.weight.device, params_dtype=module.weight.dtype)
            for key, param in base.named_parameters(recurse=False):
                param.copy_(getattr(module, key))
                _copy_param_attrs(getattr(module, key), param)
            weight = base.weight
            delta = module.scale * (module.lora_b.weight.float() @ module.lora_a.weight.float())
        weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
        weight.requires_grad = False
        setattr(parent, leaf, base)
        merged += 1
    return merged


def _log_rank0(message):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message, flush=True)
