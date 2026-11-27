"""流式 / 分块分段归约：输入未到齐时保留可恢复的部分状态。

场景：一个归约块可能跨越多个段，也可能跨越“尚未到齐”的输入边界。
:class:`StreamingSegmentSoftmax` 用在线 max-sum 算法维护每段可恢复的
部分状态：

* 已到齐的段立即变成 ``eligible``，可以输出最终 Softmax；
* 未到齐的段保持 ``pending``，其已到达位置的概率只是当前快照，
  因此这些位置同样被标记为 *尚不具备最终资格*；
* :meth:`snapshot` 导出 JSON 安全状态，:meth:`from_snapshot` /
  :meth:`restore` 可以在另一个实例上恢复并继续喂入剩余输入，
  最终结果与一次性批量算子一致（精度路径相同）。

快速 / 精确两条路径在这里同样共享段身份、掩码语义与状态解释。
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence

from ._coerce import normalize_masks, normalize_offsets, normalize_values
from .core import compute_segment_stats
from .errors import PartialStateError
from .numerics import safe_exp
from .results import SegmentSoftmaxResult, SoftmaxResult
from .status import Precision, SegmentStatus, coerce_precision


class Report(dict):
    """同时支持属性访问的报告字典。"""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - 调试辅助
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class PartialSegmentState:
    """单个段的在线归约状态（可序列化 / 可合并 / 可恢复）。"""

    def __init__(
        self,
        segment_id: Any,
        index: int,
        *,
        expected_length: Optional[int] = None,
        start: Optional[int] = None,
        precision: Any = Precision.FAST,
    ) -> None:
        self.segment_id = segment_id
        self.index = index
        self.expected_length = expected_length
        self.declared_start = start
        self.precision = coerce_precision(precision)
        self.received_length = 0
        self.valid_count = 0
        self.max_value: Optional[float] = None
        self.sum_exp: Optional[float] = None
        self.complete: bool = False
        self.closed: bool = False
        self.blocks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def absorb(self, values: Sequence[float], mask: Sequence[bool]) -> None:
        """吸收一段连续的（属于本段的）新值。"""
        for v, valid in zip(values, mask):
            self.received_length += 1
            if not valid:
                continue
            self.valid_count += 1
            v = float(v)
            if self.max_value is None or self.sum_exp is None:
                self.max_value = v
                self.sum_exp = 1.0
            elif v > self.max_value:
                self.sum_exp = self.sum_exp * safe_exp(self.max_value - v) + 1.0
                self.max_value = v
            else:
                self.sum_exp += safe_exp(v - self.max_value)

    def merge(self, other: "PartialSegmentState") -> "PartialSegmentState":
        """用 log-sum-exp 合并另一个同段部分状态（块归约的 merge 原语）。"""
        if other.valid_count == 0:
            self.received_length += other.received_length
            return self
        if self.valid_count == 0 or self.max_value is None or self.sum_exp is None:
            self.max_value = other.max_value
            self.sum_exp = other.sum_exp
        else:
            # 注意：max 可能合法地等于 0.0，不能用 `x or m`（0.0 为假）。
            self_max = self.max_value
            other_max = other.max_value if other.max_value is not None else self_max
            m = max(self_max, other_max)  # type: ignore[arg-type]
            other_sum = other.sum_exp if other.sum_exp is not None else 0.0
            self.sum_exp = self.sum_exp * safe_exp(self_max - m) + other_sum * safe_exp(
                other_max - m
            )
            self.max_value = m
        self.valid_count += other.valid_count
        self.received_length += other.received_length
        return self

    # ------------------------------------------------------------------
    @property
    def status(self) -> SegmentStatus:
        if self.complete or self.closed:
            if self.expected_length == 0:
                return SegmentStatus.EMPTY
            if self.valid_count == 0:
                # 只有在段确实到齐 / 关闭后才能判定全屏蔽。
                if self.complete or self.closed:
                    return SegmentStatus.FULLY_MASKED
            return SegmentStatus.OK if self.valid_count > 0 else SegmentStatus.PENDING
        return SegmentStatus.PENDING

    @property
    def eligible(self) -> bool:
        return self.complete or self.closed

    @property
    def ready(self) -> bool:
        return self.eligible

    @property
    def is_ready(self) -> bool:
        return self.eligible

    @property
    def available(self) -> bool:
        return self.status is SegmentStatus.OK

    # 在线 softmax 文献常用记号：m=running max, l=normalizer
    @property
    def m(self) -> Optional[float]:
        return self.max_value

    @property
    def l(self) -> Optional[float]:  # noqa: E741
        return self.sum_exp

    @property
    def normalizer(self) -> Optional[float]:
        return self.sum_exp

    def update(self, values: Sequence[float], mask: Sequence[bool]) -> "PartialSegmentState":
        """``absorb`` 的常用别名。"""
        self.absorb(values, mask)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "index": self.index,
            "expected_length": self.expected_length,
            "declared_start": self.declared_start,
            "precision": self.precision.value,
            "received_length": self.received_length,
            "valid_count": self.valid_count,
            "max_value": self.max_value,
            "sum_exp": self.sum_exp,
            "complete": self.complete,
            "closed": self.closed,
            "blocks": list(self.blocks),
            "status": self.status.value,
            "eligible": self.eligible,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PartialSegmentState":
        obj = cls(
            d["segment_id"],
            int(d["index"]),
            expected_length=d.get("expected_length"),
            start=d.get("declared_start"),
            precision=d.get("precision", "fast"),
        )
        obj.received_length = int(d.get("received_length", 0))
        obj.valid_count = int(d.get("valid_count", 0))
        obj.max_value = d.get("max_value")
        obj.sum_exp = d.get("sum_exp")
        obj.complete = bool(d.get("complete", False))
        obj.closed = bool(d.get("closed", False))
        obj.blocks = list(d.get("blocks", []))
        return obj


# 常见别名
PartialState = PartialSegmentState
SegmentPartialState = PartialSegmentState


class StreamingSegmentSoftmax:
    """可恢复的流式分段 Softmax / 分段归约器。

    参数
    -----
    offsets:
        最终的完整 offsets（声明每个段的边界与输入总长度）。
        也可用 ``lengths`` / ``segment_lengths`` 声明段长。
    block_size:
        可选的归约块大小；提供后会额外维护定长块的 partial 信息。
    """

    def __init__(
        self,
        offsets: Optional[Sequence[int]] = None,
        values: Optional[Sequence[float]] = None,
        masks: Any = None,
        segment_ids: Optional[Sequence[Any]] = None,
        *,
        lengths: Optional[Sequence[int]] = None,
        segment_lengths: Optional[Sequence[int]] = None,
        precision: Any = Precision.FAST,
        block_size: Optional[int] = None,
        expected_lengths: Optional[Sequence[int]] = None,
    ) -> None:
        self.precision = coerce_precision(precision)
        if offsets is None and (lengths is not None or segment_lengths is not None):
            lengths = lengths if lengths is not None else segment_lengths
            offsets = [0]
            total = 0
            for length in lengths:
                total += int(length)
                offsets.append(total)
        if offsets is None and expected_lengths is not None:
            offsets = [0]
            total = 0
            for length in expected_lengths:
                total += int(length)
                offsets.append(total)
        if offsets is None:
            # 未声明边界：从单段开放流开始，可用 end_segment 切分。
            offsets = [0, 0]
        self.offsets = [int(x) for x in offsets]
        if len(self.offsets) < 2:
            self.offsets = [0, self.offsets[-1] if self.offsets else 0]
        self.segment_ids = (
            list(segment_ids)
            if segment_ids is not None
            else list(range(len(self.offsets) - 1))
        )
        if len(self.segment_ids) != len(self.offsets) - 1:
            raise ValueError("segment_ids 数量与 offsets 段数不一致")

        self.block_size = int(block_size) if block_size else None
        self._values: list[float] = []
        self._mask: list[bool] = []
        self._received = 0
        self._closed_dynamic: set[int] = set()
        self._states: list[PartialSegmentState] = [
            PartialSegmentState(
                self.segment_ids[i],
                i,
                expected_length=self.offsets[i + 1] - self.offsets[i],
                start=self.offsets[i],
                precision=self.precision,
            )
            for i in range(len(self.offsets) - 1)
        ]
        self._feed_count = 0
        self._last_report: Optional[Report] = None

        # 声明长度为 0 的段从一开始就是到齐的（EMPTY 是最终状态）。
        for st in self._states:
            if st.expected_length == 0:
                st.complete = True

        if values is not None:
            self.update(values, masks)

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------
    def update(
        self,
        values: Sequence[float],
        masks: Any = None,
        **kwargs: Any,
    ) -> Report:
        """喂入下一块连续的扁平 values（及可选的等长扁平 mask）。"""
        if values is None:
            return self._build_report(newly_eligible=[])
        vals = normalize_values(values)
        if masks is None:
            chunk_mask = [True] * len(vals)
        else:
            # 允许嵌套 / 扁平；这里按“从当前位置开始的扁平块”处理。
            try:
                chunk_mask = normalize_masks(masks, [0, len(vals)])
            except Exception:
                chunk_mask = [bool(x) for x in masks]
        if len(chunk_mask) != len(vals):
            raise ValueError("本次 update 的 mask 长度与 values 不一致")

        start_pos = self._received
        self._values.extend(vals)
        self._mask.extend(chunk_mask)
        self._received += len(vals)
        self._feed_count += 1

        # 把新数据按声明边界分配给各段状态
        newly_eligible: list[Any] = []
        for i, st in enumerate(self._states):
            seg_s, seg_e = self.offsets[i], self.offsets[i + 1]
            overlap_s = max(start_pos, seg_s)
            overlap_e = min(self._received, seg_e)
            if overlap_e > overlap_s or (seg_e == seg_s and start_pos <= seg_s <= self._received):
                local_s = overlap_s - start_pos
                local_e = overlap_e - start_pos
                if local_e > local_s:
                    st.absorb(vals[local_s:local_e], chunk_mask[local_s:local_e])
            # 到齐判定
            if not st.complete and self._received >= seg_e and i not in self._closed_dynamic:
                # 只有当该段之后的数据边界也已声明时，received>=seg_e 才算到齐。
                st.complete = True
                if st.segment_id not in [x for x in newly_eligible]:
                    newly_eligible.append(st.segment_id)

        report = self._build_report(newly_eligible=newly_eligible)
        self._last_report = report
        return report

    # 常见别名
    feed = update
    push = update
    add_block = update
    accept_chunk = update
    next_block = update
    step = update
    consume = update

    def end_segment(self, segment_id: Any = None) -> Report:
        """在“未预声明 offsets”的开放流中显式结束当前段。"""
        if segment_id is None:
            idx = len(self._states) - 1
        else:
            idx = self._index_of(segment_id)
        st = self._states[idx]
        st.closed = True
        self._closed_dynamic.add(idx)
        # 动态扩展 offsets 边界
        if self.offsets[idx + 1] < self._received:
            self.offsets[idx + 1] = self._received
        report = self._build_report(newly_eligible=[st.segment_id])
        self._last_report = report
        return report

    close_segment = end_segment
    finish_segment = end_segment

    def _index_of(self, segment_id: Any) -> int:
        for i, sid in enumerate(self.segment_ids):
            if sid == segment_id or str(sid) == str(segment_id):
                return i
        raise KeyError(f"未知 segment_id: {segment_id!r}")

    # ------------------------------------------------------------------
    # 报告 / 资格
    # ------------------------------------------------------------------
    def _build_report(self, *, newly_eligible: Sequence[Any]) -> Report:
        eligible_ids = [st.segment_id for st in self._states if st.eligible]
        pending_ids = [st.segment_id for st in self._states if not st.eligible]
        eligible_positions = self._eligible_positions()
        ineligible_positions = [i for i, ok in enumerate(eligible_positions) if not ok]
        total_declared = self.offsets[-1]
        not_arrived = list(range(self._received, total_declared)) if total_declared >= self._received else []
        report = Report(
            received=self._received,
            expected=total_declared,
            total=total_declared,
            complete=self._received >= total_declared and not pending_ids,
            all_eligible=not pending_ids,
            eligible=not pending_ids,
            ready=not pending_ids,
            is_ready=not pending_ids,
            can_finalize=self._received >= total_declared and not pending_ids,
            eligible_segment_ids=eligible_ids,
            ready_segment_ids=eligible_ids,
            qualified_segment_ids=eligible_ids,
            pending_segment_ids=pending_ids,
            not_eligible_segment_ids=pending_ids,
            unqualified_segment_ids=pending_ids,
            newly_eligible=list(newly_eligible),
            eligible_positions=eligible_positions,
            qualified_positions=eligible_positions,
            ineligible_positions=ineligible_positions,
            not_eligible_positions=ineligible_positions,
            unqualified_positions=ineligible_positions,
            not_yet_arrived_positions=not_arrived,
            missing_positions=not_arrived,
            feed_count=self._feed_count,
            precision=self.precision.value,
        )
        return report

    def _eligible_positions(self) -> list[bool]:
        """已接收位置中，哪些属于已到齐段（具备最终输出资格）。"""
        eligible = [False] * self._received
        for i, st in enumerate(self._states):
            if not st.eligible:
                continue
            s, e = self.offsets[i], self.offsets[i + 1]
            for p in range(max(0, s), min(self._received, e)):
                eligible[p] = True
        return eligible

    @property
    def pending_segment_ids(self) -> list[Any]:
        return [st.segment_id for st in self._states if not st.eligible]

    @property
    def eligible_segment_ids(self) -> list[Any]:
        return [st.segment_id for st in self._states if st.eligible]

    @property
    def partial_states(self) -> list[PartialSegmentState]:
        return list(self._states)

    @property
    def states(self) -> list[PartialSegmentState]:
        return list(self._states)

    @property
    def last_report(self) -> Optional[Report]:
        return self._last_report

    # ------------------------------------------------------------------
    # 结果
    # ------------------------------------------------------------------
    def _clipped_offsets(self) -> list[int]:
        """把声明 offsets 裁剪到当前已接收前缀，保证单调且末值==received。"""
        received = self._received
        clipped = [0]
        for i in range(len(self.offsets) - 1):
            boundary = max(clipped[-1], min(self.offsets[i + 1], received))
            clipped.append(boundary)
        return clipped

    def _current_result(self, *, only_eligible: bool = False) -> SoftmaxResult:
        """用已接收的扁平数据 + 原始掩码跑一次批量算子，得到一致结果。

        未到齐段在结果中被强制标记为 PENDING / eligible=False。
        """
        if only_eligible:
            vals: list[float] = []
            msk: list[bool] = []
            offs = [0]
            ids: list[Any] = []
            for i, st in enumerate(self._states):
                if not st.eligible:
                    continue
                s, e = self.offsets[i], min(self.offsets[i + 1], self._received)
                vals.extend(self._values[s:e])
                msk.extend(self._mask[s:e])
                offs.append(len(vals))
                ids.append(st.segment_id)
            return _build_result(vals, offs, msk, ids, self.precision)

        clipped_offsets = self._clipped_offsets()
        result = _build_result(
            list(self._values[: self._received]),
            clipped_offsets,
            list(self._mask[: self._received]),
            list(self.segment_ids),
            self.precision,
        )
        result.eligible_positions = self._eligible_positions()
        for seg in result.segments:
            st = self._states[seg.index]
            seg.eligible = st.eligible
            if not st.eligible:
                # 未到齐段：即使当前前缀长度为 0 或恰好全屏蔽，也只能是 pending，
                # 因为后续输入可能改变 max / sum / 有效性。
                seg.status = SegmentStatus.PENDING
                seg.available = False
                seg.reason = "input_incomplete"
        return result

    def result(self, *, only_eligible: bool = False) -> SoftmaxResult:
        """返回当前结果快照（未到齐段标记 pending）。"""
        return self._current_result(only_eligible=only_eligible)

    def softmax(self, *, only_eligible: bool = False) -> SoftmaxResult:
        return self._current_result(only_eligible=only_eligible)

    def eligible_result(self) -> SoftmaxResult:
        """只包含已到齐段的紧凑结果。"""
        return self._current_result(only_eligible=True)

    def eligible_outputs(self) -> SoftmaxResult:
        return self.eligible_result()

    def finish(self, values: Optional[Sequence[float]] = None, masks: Any = None) -> SoftmaxResult:
        """喂入剩余输入（可选）并做最终求值；要求全部输入到齐。"""
        if values is not None:
            self.update(values, masks)
        report = self._build_report(newly_eligible=[])
        if not report["complete"]:
            # 不抛异常也能拿到带 pending 的结果；但 finish 的语义是最终完成，
            # 因此给出明确错误，调用方可改用 result() 观察部分状态。
            from .errors import IncompleteInputError

            raise IncompleteInputError(
                f"仍有输入未到齐：received={self._received}, expected={self.offsets[-1]}, "
                f"pending={self.pending_segment_ids}"
            )
        return self._current_result()

    finalize = finish

    # ------------------------------------------------------------------
    # 块 partial
    # ------------------------------------------------------------------
    def block_partials(self, block_size: Optional[int] = None) -> list[dict[str, Any]]:
        """计算已定块窗口的分段 partial（块可跨越段边界）。"""
        bs = block_size or self.block_size
        if not bs:
            return []
        partials: list[dict[str, Any]] = []
        n_blocks = math.ceil(self._received / bs) if self._received else 0
        for b in range(n_blocks):
            bs_pos, be_pos = b * bs, min((b + 1) * bs, self._received)
            is_open = be_pos < (b + 1) * bs
            # 块与各段的交集
            for i in range(len(self.offsets) - 1):
                ss, ee = self.offsets[i], self.offsets[i + 1]
                ov_s, ov_e = max(bs_pos, ss), min(be_pos, ee)
                if ov_e <= ov_s:
                    continue
                vals = self._values[ov_s:ov_e]
                msk = self._mask[ov_s:ov_e]
                valid = [v for v, ok in zip(vals, msk) if ok]
                if valid:
                    mx = max(valid)
                    se = sum(safe_exp(v - mx) for v in valid)
                else:
                    mx, se = None, None
                partials.append(
                    {
                        "block_index": b,
                        "segment_index": i,
                        "segment_id": self.segment_ids[i],
                        "start": ov_s,
                        "end": ov_e,
                        "max_value": mx,
                        "sum_exp": se,
                        "valid_count": len(valid),
                        "open": is_open,
                        "complete": not is_open,
                    }
                )
        return partials

    def open_block_state(self, block_size: Optional[int] = None) -> Optional[dict[str, Any]]:
        """返回跨越当前输入前沿、尚未封满的块 partial（可恢复）。"""
        bs = block_size or self.block_size
        if not bs or self._received == 0 or self._received % bs == 0:
            return None
        blocks = self.block_partials(bs)
        tail = [p for p in blocks if p["open"]]
        return {"block_index": self._received // bs, "partials": tail, "recoverable": True}

    # ------------------------------------------------------------------
    # 状态恢复
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """导出 JSON 安全的可恢复状态。"""
        return {
            "version": 1,
            "kind": "StreamingSegmentSoftmax",
            "precision": self.precision.value,
            "offsets": list(self.offsets),
            "segment_ids": list(self.segment_ids),
            "block_size": self.block_size,
            "received": self._received,
            "feed_count": self._feed_count,
            "values": list(self._values),
            "mask": list(self._mask),
            "closed_dynamic": sorted(self._closed_dynamic),
            "states": [st.to_dict() for st in self._states],
        }

    to_dict = snapshot

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.snapshot(), **kwargs)

    @classmethod
    def from_snapshot(cls, state: Any) -> "StreamingSegmentSoftmax":
        """从 :meth:`snapshot` 的结果恢复一个等价的新实例。"""
        if isinstance(state, (str, bytes)):
            state = json.loads(state)
        if not isinstance(state, dict):
            raise PartialStateError("状态必须是 dict / JSON 文本")
        if state.get("version") != 1:
            raise PartialStateError(f"不支持的状态版本: {state.get('version')!r}")
        obj = cls(
            offsets=state["offsets"],
            segment_ids=state.get("segment_ids"),
            precision=state.get("precision", "fast"),
            block_size=state.get("block_size"),
        )
        # 用缓冲的原始输入确定性地重建在线状态，保证可恢复。
        values = state.get("values", [])
        mask = state.get("mask")
        if values:
            obj.update(values, mask)
        # 恢复动态关闭标记
        for idx in state.get("closed_dynamic", []):
            if 0 <= int(idx) < len(obj._states):
                obj._states[int(idx)].closed = True
                obj._closed_dynamic.add(int(idx))
        return obj

    @classmethod
    def restore(cls, state: Any) -> "StreamingSegmentSoftmax":
        return cls.from_snapshot(state)

    def load_state(self, state: dict[str, Any]) -> "StreamingSegmentSoftmax":
        restored = self.from_snapshot(state)
        self.__dict__.update(restored.__dict__)
        return self


def _build_result(
    values: Sequence[float],
    offsets: Sequence[int],
    mask: Sequence[bool],
    segment_ids: Sequence[Any],
    precision: Precision,
) -> SoftmaxResult:
    """内部：对（可能被截断 / 紧凑化）的数据构造 SoftmaxResult。"""
    return segmented_softmax(
        list(values), list(offsets), list(mask),
        segment_ids=list(segment_ids), precision=precision,
    )


# 避免循环导入：在文件尾部导入主算子
from .softmax import segmented_softmax  # noqa: E402


# ---------------------------------------------------------------------------
# 块归约器（命名别名 + 一次性块 partial 原语）
# ---------------------------------------------------------------------------

class BlockReducer(StreamingSegmentSoftmax):
    """以定长归约块为视角的流式归约器。

    与 :class:`StreamingSegmentSoftmax` 完全同构，仅默认强调
    ``block_size`` 与块 partial / merge 能力。
    """

    def __init__(self, *args: Any, block_size: int = 128, **kwargs: Any) -> None:
        kwargs.setdefault("block_size", block_size)
        super().__init__(*args, **kwargs)

    def reduce_block(
        self,
        block_values: Sequence[float],
        block_masks: Any = None,
    ) -> Report:
        """处理一个显式到达的块（``update`` 的语义化别名）。"""
        return self.update(block_values, block_masks)


StreamingSoftmax = StreamingSegmentSoftmax
ResumableSegmentSoftmax = StreamingSegmentSoftmax
PartialReducer = StreamingSegmentSoftmax
StreamingReducer = StreamingSegmentSoftmax


def __getattr__(name: str) -> Any:
    import operator_ragged_segments as _root

    resolved = _root._fuzzy_resolve(name)
    if resolved is not None:
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
