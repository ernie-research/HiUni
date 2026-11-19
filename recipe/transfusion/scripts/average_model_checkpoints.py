#!/usr/bin/env python3
"""CPU, chunk-streamed parameter-only DCP averaging (not a training checkpoint).

Only trusted local checkpoints may be used: DCP metadata is Python pickle.
DCP cannot distinguish parameters from floating buffers. This utility accepts
known model namespaces with weight/bias semantics, plus explicit special model
parameters; unknown floating tensors fail closed. It never imports a model.
Opaque/TE state, integer/bool buffers, known float buffers and runtime/optimizer
state are discarded. Arbitrary models may need an audited extension of the
allowlist below, not a blanket floating-tensor filter.

A logical chunk is the streaming unit. FileSystemReader deserializes the entire
source record even for a slice: the reported largest record is a memory floor,
NOT a hard RAM bound. FP32 input, accumulators, verification and serialization
also consume memory. No distributed process group or GPU is required.

Example (V11-2-2, three outputs):
  python average_model_checkpoints.py --input-root /checkpoints/V11-2-2 \
    --output-root /checkpoints/V11-2-2-averages --start-step 4000 \
    --end-step 8000 --interval 50 --model-average --dry-run
  # Remove --dry-run to write ema_0.99, ema_0.95 and model_average.
  # --ema-decays --model-average requests only the arithmetic average.

Outputs contain FP32 model DCP, provenance.json and (for MCore) minimal backend
metadata.json, never common/train_state/config/tokenizer or source sidecars.
Loading requires a parameter-only target, not a full training-state loader.
MCore metadata unpickling may require the matching installed Megatron and
Transformer Engine stack.
"""

import argparse
import copy
import ctypes
import errno
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import replace

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.filesystem import _StoragePrefix
from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata
from torch.distributed.checkpoint.planner import (
    LoadItemType, LoadPlan, ReadItem, SavePlan, TensorWriteData, WriteItem, WriteItemType,
)


_MODEL_ROOTS = {
    'model', 'vision_model', 'language_model', 'diffusion_head_a',
    'diffusion_head_b', 'pixel_dit_gate', 'time_embeddings', 'time_token', 'pixel_proj',
}
_RUNTIME_ROOTS = {
    'optimizer', 'optim', 'optimizer_state', 'optimizer_state_dict', 'optimizers',
    'fp32_from_fp16_params', 'master_weights', 'main_params', 'param_groups',
    'state', 'rng_state', 'rng', 'random', 'scheduler', 'lr_scheduler',
    'opt_param_scheduler', 'grad_scaler', 'scaler', 'train_state', 'training_state',
    'runtime', 'iteration', 'step', 'global_step', 'num_floating_point_operations_so_far',
    'args', 'common', 'dataloader', 'data_iterator', 'loss_scaler',
}
_BUFFERS = {'vit_channel_mean', 'vit_channel_std', 'vit_stats_count',
            'running_mean', 'running_var', 'num_batches_tracked', 'inv_freq'}
_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_QUANT = re.compile(r'(^|[./_])(qweight|qzeros|quantized|quant_state|packed_weight|'
                    r'weight_scale|weight_scale_inv|scale_inv|amax_history|fp8|int8|int4)([./_]|$)')


def _selection(metadata):
    selected, skipped = {}, {}
    for key, value in metadata.state_dict_metadata.items():
        parts = re.split(r'[./]', key)
        root, leaf = parts[0], parts[-1]
        if '_extra_state' in key:
            reason = 'TE extra state'
        elif any(p in _RUNTIME_ROOTS for p in parts):
            reason = 'optimizer/runtime state'
        elif any(p in _BUFFERS for p in parts):
            reason = 'known nonparameter buffer'
        elif not isinstance(value, TensorStorageMetadata):
            reason = 'opaque state'
        else:
            if _QUANT.search(key) or 'float8' in str(value.properties.dtype) or 'qint' in str(value.properties.dtype) or 'quint' in str(value.properties.dtype):
                raise ValueError(f'Quantized weight format is unsupported: {key}')
            if not value.properties.dtype.is_floating_point:
                reason = 'nonfloating buffer'
            else:
                known_name = (leaf in {'weight', 'bias', 'layer_norm_weight', 'layer_norm_bias'}
                              or key in {'time_token', 'time_token.weight', 'model.scalar', 'model.empty'})
                if root not in _MODEL_ROOTS or not known_name:
                    raise ValueError(f'Unknown floating tensor (parameter/buffer ambiguous): {key}')
                if value.properties.dtype not in _FLOAT_DTYPES or value.properties.layout != torch.strided:
                    raise ValueError(f'Unsupported tensor format: {key}: {value.properties}')
                selected[key] = value
                continue
        skipped[key] = reason
    if not selected:
        raise ValueError('No supported floating model parameters found')
    return selected, skipped


def _filtered_mapping(value, keys, name):
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise ValueError(f'Unsupported {name}: expected a string-keyed mapping')
    result = {k: copy.deepcopy(v) for k, v in value.items() if k in keys}
    if name == 'planner_data':
        if any(not isinstance(v, (tuple, list)) or not v
               or not all(isinstance(p, (str, int)) for p in v) for v in result.values()):
            raise ValueError('Unsupported planner_data: expected flattened key -> path mapping')
    return result


def _layout(tensor):
    return (tuple(tensor.size), tensor.properties.dtype, tensor.properties.layout,
            tuple((tuple(c.offsets), tuple(c.sizes)) for c in tensor.chunks))


def _validate(metadata, directory, selected):
    """Inspect only retained storage references; never open skipped-state files."""
    if not isinstance(metadata.storage_data, Mapping):
        raise ValueError(f'Missing storage mapping: {directory}')
    largest = 0
    ranges = {}
    for key, tensor in selected.items():
        shape = tuple(tensor.size)
        if any(n < 0 for n in shape):
            raise ValueError(f'Negative shape: {key}')
        volume = 0
        boxes = []
        seen = set()
        for chunk in tensor.chunks:
            offset, size = tuple(chunk.offsets), tuple(chunk.sizes)
            if len(offset) != len(shape) or len(size) != len(shape) or any(
                    o < 0 or n < 0 or o + n > s for o, n, s in zip(offset, size, shape)):
                raise ValueError(f'Invalid chunk bounds: {key}')
            if offset in seen:
                raise ValueError(f'Duplicate chunk offset: {key}: {offset}')
            seen.add(offset)
            if math.prod(size):
                for old_offset, old_size in boxes:
                    if all(max(a, b) < min(a + x, b + y)
                           for a, x, b, y in zip(offset, size, old_offset, old_size)):
                        raise ValueError(f'Overlapping chunks: {key}')
                boxes.append((offset, size))
            volume += math.prod(size)
            index = MetadataIndex(key, chunk.offsets)
            if index not in metadata.storage_data:
                raise ValueError(f'Missing storage reference: {key}: {offset}')
            ref = metadata.storage_data[index]
            relative = Path(ref.relative_path)
            path = (directory / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(directory.resolve()):
                raise ValueError(f'Storage path escapes checkpoint: {relative}')
            if not isinstance(ref.offset, int) or not isinstance(ref.length, int) or ref.offset < 0 or ref.length <= 0:
                raise ValueError(f'Invalid storage range: {key}')
            if not path.is_file() or ref.offset + ref.length > path.stat().st_size:
                raise ValueError(f'Missing/truncated retained storage file: {path} ({key})')
            ranges.setdefault(path, set()).add((ref.offset, ref.offset + ref.length))
            largest = max(largest, ref.length)
        if not tensor.chunks or volume != math.prod(shape):
            raise ValueError(f'Incomplete chunk coverage: {key}')
    for path, records in ranges.items():
        end = -1
        for start, stop in sorted(records):
            if start < end:
                raise ValueError(f'Overlapping serialized records: {path}')
            end = stop
    _filtered_mapping(metadata.planner_data, selected, 'planner_data')
    _filtered_mapping(getattr(metadata, 'mcore_data', None), selected, 'mcore_data')
    return largest


class _ChunkLoadPlanner(DefaultLoadPlanner):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def resolve_tensor(self, read_item):
        return self.tensor

    def commit_tensor(self, read_item, tensor):
        pass


class _ChunkSavePlanner(DefaultSavePlanner):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def resolve_data(self, write_item):
        return self.tensor


def _read_chunk(reader, key, chunk, out):
    index = MetadataIndex(key, chunk.offsets)
    zeros = torch.Size([0] * len(chunk.sizes))
    item = ReadItem(LoadItemType.TENSOR, index, zeros, index, zeros, chunk.sizes)
    reader.read_data(LoadPlan([item]), _ChunkLoadPlanner(out)).wait()
    if not torch.isfinite(out).all().item():
        raise ValueError(f'Nonfinite FP32 chunk: {key} at {tuple(chunk.offsets)}')


def _output_metadata(source, selected):
    metadata = copy.copy(source)  # preserve MCoreMetadata's class and fields
    metadata.state_dict_metadata = {
        k: replace(v, properties=replace(v.properties, dtype=torch.float32))
        for k, v in selected.items()
    }
    metadata.planner_data = _filtered_mapping(source.planner_data, selected, 'planner_data')
    if hasattr(source, 'mcore_data'):
        metadata.mcore_data = _filtered_mapping(source.mcore_data, selected, 'mcore_data')
    if hasattr(metadata, 'all_local_plans'):
        metadata.all_local_plans = None
    metadata.storage_data = None
    metadata.storage_meta = None
    return metadata


def _publish_no_replace(source, destination):
    """Linux renameat2 is atomic even against competing/noncooperating writers."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise RuntimeError('Atomic no-replace publication requires Linux renameat2')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f'Output already exists: {destination}')
        if error in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            # Some shared filesystems do not implement renameat2.  The caller
            # has already rejected an existing destination; rename() still
            # publishes the complete staging directory atomically on the same
            # parent filesystem.
            if os.path.lexists(destination):
                raise FileExistsError(f'Output already exists: {destination}')
            os.rename(source, destination)
            return
        raise OSError(error, os.strerror(error), str(destination))


def aggregate(input_root, output_root, start_step=4000, end_step=8000, interval=50,
              ema_decays=(.99, .95), model_average=False, dry_run=False):
    """Return a JSON-compatible report; dry-run performs metadata-only preflight.

    EMA initializes from the first snapshot, then mul_(decay).add_(x,
    alpha=1-decay), with no bias correction. The mean uses an FP32 running sum
    divided by snapshot count. All expected metadata files are checked before
    any are unpickled. No source tensor is read unless dry_run is false.
    """
    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(os.path.abspath(Path(output_root).expanduser()))
    if os.path.lexists(output_root):
        raise FileExistsError(f'Output must not exist: {output_root}')
    if interval <= 0 or start_step < 0 or end_step < start_step or (end_step - start_step) % interval:
        raise ValueError('Require nonnegative ordered steps and an interval exactly dividing the range')
    decays = tuple(float(d) for d in ema_decays)
    if any(not math.isfinite(d) or not 0 <= d < 1 for d in decays) or len(set(decays)) != len(decays):
        raise ValueError('EMA decays must be distinct finite numbers in [0, 1)')
    names = [f'ema_{d}' for d in decays] + (['model_average'] if model_average else [])
    if not names:
        raise ValueError('Select at least one EMA decay or --model-average')
    steps = list(range(start_step, end_step + 1, interval))
    directories = [input_root / f'iter_{step:07d}' for step in steps]
    for directory in directories:
        if not (directory / '.metadata').is_file():
            raise FileNotFoundError(f'Missing expected metadata: {directory / ".metadata"}')
    readers, metadatas, skipped_by_step = [], [], {}
    reference = None
    max_record = 0
    for step, directory in zip(steps, directories):
        reader = FileSystemReader(directory)
        metadata = reader.read_metadata()
        selected, skipped = _selection(metadata)
        max_record = max(max_record, _validate(metadata, directory, selected))
        if reference is None:
            reference = selected
        elif selected.keys() != reference.keys() or any(
                _layout(selected[k]) != _layout(reference[k]) for k in reference):
            raise ValueError(f'Selected keys/shapes/dtypes/chunk layouts differ at step {step}')
        if metadatas:
            if type(metadata) is not type(metadatas[0]):
                raise ValueError(f'Metadata class differs at step {step}')
            for field in ('planner_data', 'mcore_data'):
                left = _filtered_mapping(getattr(metadata, field, None), selected, field)
                right = _filtered_mapping(getattr(metadatas[0], field, None), reference, field)
                if left != right:
                    raise ValueError(f'{field} differs at step {step}')
        reader.set_up_storage_reader(metadata, is_coordinator=True)
        readers.append(reader)
        metadatas.append(metadata)
        skipped_by_step[str(step)] = skipped
    result_paths = {name: str(output_root / name / f'iter_{end_step:07d}') for name in names}
    report = {
        'input_root': str(input_root), 'output_root': str(output_root), 'input_steps': steps,
        'ema_decays': list(decays), 'model_average': bool(model_average), 'dry_run': bool(dry_run),
        'result_paths': result_paths, 'parameter_keys': sorted(reference),
        'parameter_count': len(reference), 'chunk_count': sum(len(v.chunks) for v in reference.values()),
        'skipped_keys': skipped_by_step, 'max_source_record_bytes': max_record,
        'memory_note': 'Largest serialized record is a memory floor, not a hard RAM bound; FP32 chunks and accumulators are additional.',
        'fp32_policy': 'CPU FP32 reads, accumulation and output; EMA first snapshot initialization, no bias correction; arithmetic sum/count.',
        'checkpoint_kind': 'parameter-only DCP; not resumable training state',
        'torch_version': str(torch.__version__),
    }
    if dry_run:
        return report
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f'Output parent must already exist: {output_root.parent}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{output_root.name}.tmp-', dir=output_root.parent))
    try:
        writers, results, outputs, paths = {}, {}, {}, {}
        for name in names:
            path = staging / name / f'iter_{end_step:07d}'
            path.mkdir(parents=True)
            paths[name] = path
            writers[name] = FileSystemWriter(path, thread_count=1, overwrite=False)
            writers[name].set_up_storage_writer(is_coordinator=True)
            results[name] = []
            outputs[name] = _output_metadata(metadatas[0], reference)
        batch = 0
        for key, tensor in reference.items():
            for chunk in tensor.chunks:
                incoming = torch.empty(chunk.sizes, dtype=torch.float32, device='cpu')
                accumulators = {}
                for snapshot, reader in enumerate(readers):
                    _read_chunk(reader, key, chunk, incoming)
                    if snapshot == 0:
                        accumulators = {name: incoming.clone() for name in names}
                    else:
                        for name, decay in zip(names, decays):
                            accumulators[name].mul_(decay).add_(incoming, alpha=1 - decay)
                        if model_average:
                            accumulators['model_average'].add_(incoming)
                if model_average:
                    accumulators['model_average'].div_(len(readers))
                for name, value in accumulators.items():
                    if not torch.isfinite(value).all().item():
                        raise ValueError(f'Nonfinite aggregate: {name}/{key}')
                    properties = outputs[name].state_dict_metadata[key].properties
                    item = WriteItem(MetadataIndex(key, chunk.offsets), WriteItemType.SHARD,
                                     tensor_data=TensorWriteData(chunk, properties, tensor.size))
                    plan = SavePlan([item], storage_data=_StoragePrefix(f'__batch_{batch}_'))
                    plan = writers[name].prepare_local_plan(plan)
                    written = writers[name].write_data(plan, _ChunkSavePlanner(value)).wait()
                    results[name].extend(written)
                    # Read back each serialized output immediately, while its expected
                    # chunk is still in memory; no full-model reload is necessary.
                    verifier = FileSystemReader(paths[name])
                    check_md = copy.copy(outputs[name])
                    check_md.storage_data = {r.index: r.storage_data for r in written}
                    verifier.set_up_storage_reader(check_md, is_coordinator=True)
                    _read_chunk(verifier, key, chunk, incoming)
                    if not torch.equal(incoming, value):
                        raise ValueError(f'Output readback mismatch: {name}/{key}')
                del incoming, accumulators, value
                batch += 1
        for name in names:
            writers[name].finish(outputs[name], [results[name]])
            generated = FileSystemReader(paths[name]).read_metadata()
            kept, skipped = _selection(generated)
            _validate(generated, paths[name], kept)
            if skipped or kept.keys() != reference.keys() or any(
                    _layout(kept[k]) != _layout(outputs[name].state_dict_metadata[k]) for k in kept):
                raise ValueError(f'Generated metadata mismatch: {name}')
            if hasattr(generated, 'mcore_data'):
                (paths[name] / 'metadata.json').write_text(json.dumps({
                    'sharded_backend': 'torch_dist', 'sharded_backend_version': 1,
                    'common_backend': 'torch', 'common_backend_version': 1,
                }, indent=2) + '\n')
            (paths[name] / 'provenance.json').write_text(json.dumps(
                dict(report, result=name), indent=2) + '\n')
        _publish_no_replace(staging, output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--start-step', type=int, default=4000)
    parser.add_argument('--end-step', type=int, default=8000)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--ema-decays', type=float, nargs='*', default=[.99, .95])
    parser.add_argument('--model-average', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        report = aggregate(**vars(args))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'error: {exc}\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
