"""多流 Energon 数据 Provider：每个模态独立 pipeline，调度器按比例轮转输出。"""

import logging
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple

import torch

# 单条子流 save_state 的超时上限 (秒)。energon 的 _worker_command 用无超时的
# result_queue.get() 逐个等 worker 子进程; 多机下某节点 GPFS/AFS 抖动会让某个
# worker 卡在 get_checkpoint, 该 dp_rank 永远到不了 save_checkpoint 里的 DP-group
# barrier → 整个 DP group hang。超时后按「缺 checkpoint」同等处理 (记 None +
# save_complete=False), 让 barrier 得以推进, restore 时所有流一致从头重填。
# 可用环境变量 MULTISTREAM_SAVE_TIMEOUT 覆盖。
_STREAM_SAVE_TIMEOUT_SEC = float(os.environ.get("MULTISTREAM_SAVE_TIMEOUT", "120"))

from megatron.bridge.data.energon.base_energon_datamodule import (
    EnergonMultiModalDataModule,
)
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider

logger = logging.getLogger(__name__)


def _normalize_stream_names(stream_names: Optional[Iterable[str]]) -> Set[str]:
    """归一化需要按流跳过恢复的 stream 名称。"""
    if stream_names is None:
        return set()
    if isinstance(stream_names, str):
        candidates = stream_names.split(",")
    else:
        candidates = stream_names
    return {str(name).strip() for name in candidates if str(name).strip()}


class ScheduledDataloader:
    """包装多条 EnergonDataloader，按 deficit 确定性调度。"""

    def __init__(
        self,
        streams: Dict[str, Any],
        schedule: Dict[str, int],
        rank_offset: int = 0,
        reset_streams_on_restore: Optional[Iterable[str]] = None,
    ):
        """初始化调度器。

        Args:
            streams: {modality_name: EnergonDataloader} 各模态的独立 dataloader
            schedule: {modality_name: weight} 各模态的目标比例
            rank_offset: 当前 rank 的调度偏移步数，使各 rank 覆盖周期的不同位置
        """
        self.streams = streams
        total = sum(schedule.values())
        self.weights = {n: w / total for n, w in schedule.items()}
        self.deficit = {n: 0.0 for n in schedule}
        self.step = 0
        self.counts: Dict[str, int] = defaultdict(int)
        self.reset_streams_on_restore = _normalize_stream_names(reset_streams_on_restore)

        # 按 rank 偏移 deficit 状态，使各 rank 错开调度相位
        for _ in range(rank_offset):
            for name, w in self.weights.items():
                self.deficit[name] += w
            selected = max(self.deficit, key=lambda n: self.deficit[n])
            self.deficit[selected] -= 1.0

    def __iter__(self):
        """返回自身作为迭代器。"""
        return self

    def __next__(self):
        """按 deficit 调度返回下一个 batch。"""
        for name, w in self.weights.items():
            self.deficit[name] += w

        candidates = sorted(self.deficit, key=lambda n: -self.deficit[n])
        for name in candidates:
            self.deficit[name] -= 1.0
            batch = next(self.streams[name])
            self.step += 1
            self.counts[name] += 1
            if self.step % 100 == 0:
                total = self.step
                ratios = {
                    k: f"{v}/{total}({v / total * 100:.1f}%)"
                    for k, v in self.counts.items()
                }
                # logger.info(
                #     f"[MultiStream] step {total}, 模态比例: {ratios}"
                # )
            return batch
        raise StopIteration

    def save_state(self):
        """保存所有子流状态和调度器状态。

        兜底: 某条子流的 energon savable dataloader 在某些 rank/worker 上可能尚无可保存的
        checkpoint (worker 还没消费过样本 / packing buffer 未攒够 → energon 抛
        "ValueError: No checkpoint found")。混训 round-robin 下稀疏 stream 在该 rank 很常见。
        此时记 None, 不让单条流的缺失把整个 save_checkpoint 拖崩。

        【多机 hang 兜底】energon 的 _worker_command 用无超时的 result_queue.get() 逐个等
        worker 子进程返回 get_checkpoint 结果。多机下某节点 GPFS/AFS 慢读 / flaky 节点上
        worker 死掉, 会让该 dp_rank 的 save_state 永久阻塞, 到不了 save_checkpoint 里
        get_data_parallel_group() 上的 barrier → 整个 DP group (进而所有 rank) hang。
        故对每条子流的 save_state 加 _STREAM_SAVE_TIMEOUT_SEC 超时: 超时与「缺 checkpoint」
        同等处理 (记 None + save_complete=False), 保证 barrier 能推进。

        【一致性约束 (修复 resume CE=INF)】调度器是确定性 round-robin: deficit/step/counts
        共同决定下一个 batch 取自哪条流、以及每条流被消费到的相对位置。若 restore 时
        部分流跳到 checkpoint 位置、另一部分流 (None) 从数据源头重填, 则恢复出的首个
        global batch 与各流真实位置错位 —— 稀疏流 (常是 understanding) 拿到的是数据源
        开头的样本而非续训点的样本, 该模态 CE 在第一步即异常 (→ und_ce_loss / count 退化,
        经 train.py 无 epsilon 除法放大为 INF)。
        因此采取「全有或全无」: 只要有任一流缺 checkpoint, 标记 incomplete; restore 时
        丢弃所有流状态、调度器计数清零回到 rank_offset 基线, 全部流从数据源头一致重启
        (牺牲极小的数据位置精度, 换取首 batch 与调度相位的严格一致, 不再 desync)。
        """
        stream_states = {}
        save_complete = True
        for n, s in self.streams.items():
            # energon _worker_command 的 result_queue.get() 无超时, 用独立线程跑 + 超时
            # 兜底, 避免单条流卡死把整个 DP-group barrier 拖挂 (多机 FS 抖动场景)。
            # 注意: 超时只能让本 rank 跳过该流的状态保存, 后台线程仍可能滞留在
            # 阻塞的 .get() 上 (无法强杀), 但已不再阻塞 save_checkpoint 主流程。
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(s.save_state)
                    stream_states[n] = fut.result(timeout=_STREAM_SAVE_TIMEOUT_SEC)
            except FuturesTimeout:
                logger.warning(
                    f"[MultiStream] 流 '{n}' save_state 超时 (>{_STREAM_SAVE_TIMEOUT_SEC}s); "
                    f"疑似 energon worker 卡在 get_checkpoint (多机 FS 慢读/worker 死)。"
                    f"→ 标记 incomplete, restore 时所有流一致从头重填, 保证 barrier 能推进。"
                )
                stream_states[n] = None
                save_complete = False
            except Exception as e:
                logger.warning(
                    f"[MultiStream] 流 '{n}' save_state 失败 ({type(e).__name__}: {e}); "
                    f"该流暂无 checkpoint → 整个调度器状态标记 incomplete, "
                    f"restore 时所有流将一致从头重填 (不影响模型权重)。"
                )
                stream_states[n] = None
                save_complete = False
        return {
            "stream_states": stream_states,
            "save_complete": save_complete,
            "weights": dict(self.weights),
            "deficit": dict(self.deficit),
            "step": self.step,
            "counts": dict(self.counts),
        }

    def restore_state(self, state_dict: dict) -> None:
        """从保存的状态恢复调度器和所有子流。

        必须在训练第一次 next() 调用之前执行（build_datasets 阶段）。
        子流的 EnergonDataloader 内部 cyclic_iter generator 在首次 next() 前
        不会触发 dataloader.__iter__()，因此此时调用 restore_state_rank() 安全。

        注意：SavableSampleBuffer（packing pre-buffer）在 checkpoint 时可能存储了
        None restore_key（因为 packed 中间序列不携带 __restore_key__），
        直接 restore_state_rank() 会导致 worker 进程在 worker_start() 时崩溃。
        _clear_buffer_states() 先将这些 buffer 状态清空，让其自然从数据源重新填充，
        底层数据集位置仍正确恢复。

        【一致性约束 (修复 resume CE=INF)】见 save_state docstring。若本次存档不完整
        (save_complete=False, 即至少一条流当时无 checkpoint), 任何「部分流跳位 + 部分流
        重填」的混合恢复都会让 round-robin 首 batch 与各流真实位置错位, 使稀疏模态 (常是
        understanding) CE 在第一步炸成 INF。此时一律放弃恢复子流状态: 调度器计数保持
        build 阶段已按 rank_offset 设好的基线 (不覆盖 deficit/step/counts), 所有流统一从
        数据源头开始, 保证首 batch 一致。仅当 save_complete=True (所有流都有 checkpoint)
        才执行精确的逐流位置恢复。
        """
        stream_states = state_dict.get("stream_states", {})
        saved_deficit = state_dict.get("deficit")
        current_names = set(self.weights)
        saved_stream_names = (
            set(stream_states) if isinstance(stream_states, dict) else set()
        )
        saved_deficit_names = (
            set(saved_deficit) if isinstance(saved_deficit, dict) else set()
        )

        # 调度状态只能与完全相同的流集合配套恢复。否则旧 deficit 会覆盖当前初始化
        # 状态，例如旧 checkpoint 没有新加入的 editing 流，下一次调度就会触发
        # KeyError。流集合变化时放弃整个 dataloader restore，避免半恢复导致各流
        # 数据位置与 round-robin 相位错乱；模型 checkpoint 的恢复不受影响。
        if (
            set(self.streams) != current_names
            or saved_stream_names != current_names
            or saved_deficit_names != current_names
        ):
            logger.warning(
                "[MultiStream] dataloader checkpoint 与当前流集合不兼容; "
                f"current={sorted(current_names)}, "
                f"saved_streams={sorted(saved_stream_names)}, "
                f"saved_deficit={sorted(saved_deficit_names)}. "
                "放弃恢复全部 dataloader 状态，所有流从当前初始化位置开始。"
            )
            return

        # 新 checkpoint 会保存调度权重。即使流名称相同，只要比例变化，旧 deficit
        # 也不再代表当前调度相位，因此同样整套跳过。兼容未保存 weights 的旧存档：
        # 这类存档只能校验流集合，无法判断同名流的比例是否发生过变化。
        saved_weights = state_dict.get("weights")
        if saved_weights is not None:
            weights_compatible = (
                isinstance(saved_weights, dict) and set(saved_weights) == current_names
            )
            if weights_compatible:
                try:
                    weights_compatible = all(
                        math.isclose(
                            float(saved_weights[name]),
                            float(self.weights[name]),
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        for name in current_names
                    )
                except (TypeError, ValueError):
                    weights_compatible = False
            if not weights_compatible:
                logger.warning(
                    "[MultiStream] dataloader checkpoint 的调度权重与当前配置不一致; "
                    f"current={self.weights}, saved={saved_weights}. "
                    "放弃恢复全部 dataloader 状态，所有流从当前初始化位置开始。"
                )
                return

        # 旧存档无 save_complete 字段时, 用 stream_states 是否全部非 None 兜底判定
        reset_streams = self.reset_streams_on_restore
        save_complete = state_dict.get(
            "save_complete",
            all(v is not None for v in stream_states.values()),
        )
        if reset_streams:
            unknown = sorted(reset_streams - set(self.streams))
            if unknown:
                logger.warning(
                    f"[MultiStream] dataloader_reset_streams 包含未知流 {unknown}; "
                    f"当前可用流: {list(self.streams.keys())}"
                )
            required_streams = set(self.streams) - reset_streams
            missing_required = sorted(
                n for n in required_streams if stream_states.get(n) is None
            )
            if missing_required:
                logger.warning(
                    f"[MultiStream] 仍有必须恢复的流缺 checkpoint: {missing_required}; "
                    f"无法仅重置部分流, 因此全部流从头重填。"
                )
                return
            if not save_complete:
                logger.info(
                    f"[MultiStream] 存档标记为 incomplete, 但缺失的流都在 "
                    f"dataloader_reset_streams 中, 仍将恢复其余流并重置 "
                    f"{sorted(reset_streams & set(self.streams))}"
                )
            else:
                logger.info(
                    f"[MultiStream] 以下流跳过 dataloader 恢复并从当前数据源头开始: "
                    f"{sorted(reset_streams & set(self.streams))}"
                )
        elif not save_complete:
            missing = [n for n, v in stream_states.items() if v is None]
            logger.warning(
                f"[MultiStream] 存档不完整 (缺 checkpoint 的流: {missing}); "
                f"为避免 round-robin 调度错位 (会导致稀疏模态 CE 第一步 INF), "
                f"放弃恢复所有子流状态, 调度器保持 rank_offset 基线, 全部流一致从头重填。"
            )
            return

        self.deficit = dict(saved_deficit)
        self.step = state_dict["step"]
        self.counts = defaultdict(int, state_dict["counts"])
        for name, stream_state in stream_states.items():
            if name not in self.streams:
                logger.warning(f"[MultiStream] 恢复状态中包含未知流 '{name}'，跳过")
                continue
            if name in reset_streams:
                logger.info(
                    f"[MultiStream] 流 '{name}' 已按 dataloader_reset_streams 跳过恢复"
                )
                continue
            if stream_state is None:
                # save_complete=True 路径下理论上不会出现 None; 防御性跳过。
                logger.warning(f"[MultiStream] 流 '{name}' 无保存状态，跳过恢复 (从头重填)")
                continue
            _clear_buffer_states(stream_state)
            # streams[name] 是 EnergonDataloader；_dataloader 是 SavableDataLoader
            self.streams[name]._dataloader.restore_state_rank(stream_state)
            logger.info(f"[MultiStream] 流 '{name}' 状态已恢复")


def _clear_buffer_states(obj: Any, depth: int = 0) -> None:
    """递归清空 packing buffer 中含 None restore_key 的状态。

    Energon PackingDataset 的 pre_packing_buffer 在 checkpoint 时可能包含 None 的
    restore_key（packed 中间序列不携带 __restore_key__），导致 worker_start() 中
    restore_sample(None) 崩溃。

    pre_packing_buffer.buffer[i] 与 pre_packing_lengths[i] 是配对结构，
    必须同步清空：单独清空 buffer 而保留 lengths 会导致 next_final_pack()
    以空 buffer 取切片得到空 pack，触发 final_packer([]) 崩溃（IndexError）。
    """
    if depth > 20 or obj is None:
        return

    # 检测 PackingDatasetMergedState（同时持有 pre_packing_buffer + pre_packing_lengths）
    pb = getattr(obj, "pre_packing_buffer", None)
    pl = getattr(obj, "pre_packing_lengths", None)
    if (
        pb is not None
        and type(pb).__name__ == "SampleBufferMergedState"
        and hasattr(pb, "buffer")
        and isinstance(pl, list)
    ):
        for i, worker_buf in enumerate(pb.buffer):
            if any(key is None for key in worker_buf):
                pb.buffer[i] = []
                if i < len(pl):
                    pl[i] = []
        # 继续递归其他字段（如 reading_buffer），跳过已处理的 pb/pl
        for v in vars(obj).values():
            if v is not pb and v is not pl:
                _clear_buffer_states(v, depth + 1)
        return

    type_name = type(obj).__name__
    if type_name == "SampleBufferMergedState" and hasattr(obj, "buffer"):
        for i, worker_buf in enumerate(obj.buffer):
            if any(key is None for key in worker_buf):
                obj.buffer[i] = []
        return
    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            _clear_buffer_states(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _clear_buffer_states(item, depth + 1)


@dataclass(kw_only=True)
class MultiStreamEnergonProvider(DatasetProvider):
    """多流 Energon Provider：每个模态一条独立 pipeline，调度器统一输出。"""

    streams: Dict[str, str]
    schedule: Dict[str, int]
    seq_length: int
    micro_batch_size: int = 1
    global_batch_size: int = 32
    num_workers: int = 8
    packing_buffer_size: Optional[int] = None
    task_encoder_factory: Callable = None
    dataloader_load: Optional[str] = None
    dataloader_load_iteration: Optional[int] = None
    dataloader_save: Optional[str] = None
    dataloader_reset_streams: Optional[Iterable[str]] = None
    dataloader_type: str = "external"

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """构建多条独立 Energon pipeline 并包装为 ScheduledDataloader。"""
        import megatron.core.parallel_state as parallel_state

        train_loaders = {}
        for name, path in self.streams.items():
            task_encoder = self.task_encoder_factory()
            # 子流不传 dataloader_save：续训状态由顶层 ScheduledDataloader 统一管理，
            # 避免 EnergonMultiModalDataModule._maybe_restore_dataloader_state 因找不到
            # 各子目录的 tracker file 而产生误导性日志或错误恢复行为。
            module = EnergonMultiModalDataModule(
                path=path,
                tokenizer=None,
                image_processor=None,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                global_batch_size=self.global_batch_size,
                num_workers=self.num_workers,
                packing_buffer_size=self.packing_buffer_size,
                task_encoder=task_encoder,
                dataloader_save=None,
            )
            train_loaders[name] = module.train_dataloader()
            logger.info(
                f"[MultiStream] 流 '{name}' 初始化完成, path={path}"
            )

        # 计算 rank 偏移，使各 rank 覆盖调度周期的不同区段
        dp_rank = 0
        steps_per_iter = self.global_batch_size // self.micro_batch_size
        steps_per_rank = steps_per_iter
        if parallel_state.is_initialized():
            dp_rank = parallel_state.get_data_parallel_rank()
            dp_world = parallel_state.get_data_parallel_world_size()
            steps_per_rank = steps_per_iter // dp_world

        # 用户配置 schedule sum = gbs，内部除以 mbs 得到 scheduler 实际周期
        schedule_sum = sum(self.schedule.values())
        mbs = self.micro_batch_size
        if schedule_sum != self.global_batch_size:
            raise ValueError(
                f"batch_schedule 权重之和 ({schedule_sum}) "
                f"必须等于 global_batch_size ({self.global_batch_size})"
            )
        # 每个 scheduler step 产出 mbs 条 packed seq，所以周期 = sum/mbs
        scaled_schedule = {}
        for name, w in self.schedule.items():
            if w % mbs != 0:
                logger.warning(
                    f"batch_schedule['{name}']={w} 不能被 "
                    f"micro_batch_size={mbs} 整除，"
                    f"每 iter 比例会有轻微波动"
                )
            scaled_schedule[name] = w / mbs

        rank_offset = dp_rank * steps_per_rank

        composite = ScheduledDataloader(
            train_loaders,
            scaled_schedule,
            rank_offset=rank_offset,
            reset_streams_on_restore=self.dataloader_reset_streams,
        )
        logger.info(
            f"[MultiStream] 调度器就绪, "
            f"流: {list(self.streams.keys())}, "
            f"用户配比: {self.schedule}, "
            f"实际调度周期: {scaled_schedule} (mbs={mbs})"
        )

        # 续训恢复：默认从 dataloader_save 读；配置 dataloader_load 时可读旧目录、
        # 写新目录，避免新任务继续污染旧 dataloader checkpoint。
        dataloader_restore_path = self.dataloader_load or self.dataloader_save
        if self.dataloader_load and self.dataloader_save:
            logger.info(
                f"[MultiStream] dataloader_load={self.dataloader_load}, "
                f"dataloader_save={self.dataloader_save}"
            )
        _maybe_restore_scheduled_dataloader(
            composite,
            dataloader_restore_path,
            iteration=self.dataloader_load_iteration,
        )

        return (composite, None, None)


def _maybe_restore_scheduled_dataloader(
    loader: ScheduledDataloader,
    dataloader_load: Optional[str],
    iteration: Optional[int] = None,
) -> None:
    """从 checkpoint 恢复 ScheduledDataloader 状态（对齐单流续训行为）。

    与单流的 _maybe_restore_dataloader_state 逻辑完全相同：
    读取 {dataloader_load}/latest_checkpointed_iteration.txt 确定 iter，
    再加载对应的 train_dataloader_dprank{rank}.pt。

    必须在训练循环第一次调用 next() 之前执行，此时各子流
    EnergonDataloader 内的 cyclic_iter generator 尚未触发
    dataloader.__iter__()，restore_state_rank() 可以安全调用。
    """
    if not dataloader_load:
        return

    if iteration is None:
        tracker_file = os.path.join(dataloader_load, "latest_checkpointed_iteration.txt")
        if not os.path.isfile(tracker_file):
            logger.info(f"[MultiStream] 未找到 tracker file {tracker_file}，从头开始")
            return
        try:
            with open(tracker_file) as f:
                iteration = int(f.read().strip())
        except (ValueError, IOError) as e:
            logger.warning(f"[MultiStream] 读取 tracker file 失败: {e}，从头开始")
            return
    elif iteration < 0:
        logger.warning(
            f"[MultiStream] dataloader_load_iteration={iteration} 非法，从头开始"
        )
        return

    import megatron.core.parallel_state as parallel_state
    dp_rank = (
        parallel_state.get_data_parallel_rank()
        if parallel_state.is_initialized()
        else 0
    )
    state_file = os.path.join(
        dataloader_load,
        f"iter_{iteration:07d}",
        f"train_dataloader_dprank{dp_rank:03d}.pt",
    )

    if not os.path.isfile(state_file):
        logger.info(
            f"[MultiStream] 未找到状态文件 {state_file}，从头开始"
        )
        return

    try:
        state_dict = torch.load(state_file, map_location="cpu", weights_only=False)
        loader.restore_state(state_dict["dataloader_state_dict"])
        logger.info(
            f"[MultiStream] 续训恢复成功: iter={iteration}, "
            f"dp_rank={dp_rank}, file={state_file}"
        )
    except Exception as e:
        logger.warning(
            f"[MultiStream] 恢复状态失败: {e}，从头开始"
        )
