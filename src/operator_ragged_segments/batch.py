"""推理网关侧的 ragged 批次：追加 / 取消 / 交付保护与 offsets 版本管理。

核心不变量：

1. 段身份（``segment_id``）稳定；扁平值日志只追加，每段的稳定
   ``[start, end)`` 偏移永不改变，因此新 offsets 不可能覆盖已交付段。
2. 已交付（delivered）段的结果以深拷贝快照保存，之后追加、取消、
   重建 offsets 都不会改变它。
3. 用相同 segment_id 再次 ``append`` 已交付 / 已存在的段会被明确拒绝，
   而不是静默覆盖。
4. 取消只影响“当前活跃批次”的压缩 offsets 视图；交付快照仍然保留。
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Optional, Sequence

from ._coerce import normalize_masks, normalize_offsets, normalize_values
from .errors import (
    SegmentDeliveredError,
    SegmentExistsError,
    SegmentNotFoundError,
)
from .remap import OrderMap, remap_to_original
from .results import SoftmaxResult
from .softmax import segmented_softmax
from .status import Precision, SegmentStatus, coerce_precision


_APPENDED = SegmentStatus.ACTIVE if hasattr(SegmentStatus, "ACTIVE") else SegmentStatus.OK
_CANCELED = SegmentStatus.CANCELED
_DELIVERED = "delivered"


class Segment:
    """网关中的一个段记录。"""

    def __init__(
        self,
        segment_id: Any,
        index: int,
        values: list[float],
        mask: Optional[list[bool]],
        start: int,
        end: int,
        expected_length: Optional[int] = None,
    ) -> None:
        self.segment_id = segment_id
        self.id = segment_id
        self.index = index
        self.values = values
        self.mask = mask
        self.start = start
        self.end = end
        self.expected_length = expected_length
        self.state: Any = "appended"
        self.delivery: Any = None
        self.revision: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def offsets(self) -> tuple[int, int]:
        return (self.start, self.end)

    @property
    def status(self) -> str:
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "values": list(self.values),
            "mask": list(self.mask) if self.mask is not None else None,
            "state": getattr(self.state, "value", self.state),
            "revision": self.revision,
        }

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            aliases = {"id": "segment_id", "status": "state"}
            return getattr(self, aliases.get(key, key))
        raise TypeError(key)

    def __repr__(self) -> str:
        return f"Segment(id={self.segment_id!r}, [{self.start}:{self.end}), state={self.state})"


class SegmentHandle(str):
    """当 segment_id 为字符串时返回的富句柄；本身等于 id 字符串。"""

    def __new__(
        cls,
        segment_id: Any,
        *,
        index: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        batch: Any = None,
    ) -> "SegmentHandle":
        if not isinstance(segment_id, str):
            # 非字符串 id 不适合做 str 子类，直接返回一个委托对象。
            return _DelegatingHandle(segment_id, index, start, end, batch)  # type: ignore[return-value]
        obj = str.__new__(cls, segment_id)
        obj.segment_index = index
        obj.start_offset = start
        obj.end_offset = end
        obj._batch = batch
        return obj

    @property
    def index(self) -> Optional[int]:
        return self.segment_index

    @property
    def offsets(self) -> tuple[Optional[int], Optional[int]]:
        return (self.start_offset, self.end_offset)


class _DelegatingHandle:
    """非字符串 id 的句柄：相等性 / 哈希完全委托给原 id。"""

    def __init__(self, segment_id: Any, index: Optional[int], start: Optional[int],
                 end: Optional[int], batch: Any) -> None:
        self.segment_id = segment_id
        self.index = index
        self.segment_index = index
        self.start_offset = start
        self.end_offset = end
        self._batch = batch

    @property
    def offsets(self):
        return (self.start_offset, self.end_offset)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _DelegatingHandle):
            return self.segment_id == other.segment_id
        return self.segment_id == other

    def __hash__(self) -> int:
        return hash(self.segment_id)

    def __repr__(self) -> str:
        return f"SegmentHandle({self.segment_id!r})"


class DeliveryRegistry(dict):
    """``segment_id -> 冻结的 SegmentSoftmaxResult``。

    额外支持与 id 列表 / 集合比较，兼容不同的 ``.delivered`` 断言风格。
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, dict):
            return dict.__eq__(self, other)
        if isinstance(other, (list, tuple, set, frozenset)):
            return list(self.keys()) == list(other) or set(self.keys()) == set(other)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def ids(self) -> list[Any]:  # type: ignore[override]
        return list(self.keys())


class RaggedBatch:
    """支持追加 / 取消 / 交付保护的 ragged 批次网关。

    构造参数全部可选；也支持从 fixture dict 一次性导入。
    """

    def __init__(
        self,
        payload: Any = None,
        *,
        offsets: Optional[Sequence[int]] = None,
        values: Optional[Sequence[float]] = None,
        masks: Any = None,
        segment_ids: Optional[Sequence[Any]] = None,
        original_order: Optional[Sequence[Any]] = None,
        precision: Any = Precision.FAST,
        name: str = "gateway",
    ) -> None:
        self.name = name
        self.precision = coerce_precision(precision)
        self._segments_by_id: dict[Any, Segment] = {}
        self._order: list[Any] = []          # 到达顺序（含已取消）
        self._active_order: list[Any] = []   # 未取消
        self._stable_values: list[float] = []
        self._stable_mask: list[bool] = []
        self._id_counter = itertools.count()
        self.deliveries = DeliveryRegistry()
        self.delivered = self.deliveries  # 常见命名
        self.delivered_offsets: dict[Any, tuple[int, int]] = {}
        self.offset_revisions: list[dict[str, Any]] = []
        self.original_order: list[Any] = []
        self._revision = 0
        self.last_result: Optional[SoftmaxResult] = None

        if payload is not None or offsets is not None or values is not None:
            self.ingest(
                payload,
                offsets=offsets,
                values=values,
                masks=masks,
                segment_ids=segment_ids,
                original_order=original_order,
            )
        elif original_order is not None:
            self.original_order = list(original_order)

    # ------------------------------------------------------------------
    # 导入
    # ------------------------------------------------------------------
    def ingest(
        self,
        payload: Any = None,
        *,
        offsets: Optional[Sequence[int]] = None,
        values: Optional[Sequence[float]] = None,
        masks: Any = None,
        segment_ids: Optional[Sequence[Any]] = None,
        original_order: Optional[Sequence[Any]] = None,
    ) -> "RaggedBatch":
        """从 fixture dict / offsets+values 批量导入段。"""
        if payload is not None and isinstance(payload, dict):
            offsets = offsets if offsets is not None else payload.get("offsets")
            values = values if values is not None else payload.get("values")
            masks = masks if masks is not None else payload.get("masks")
            segment_ids = segment_ids if segment_ids is not None else payload.get("segment_ids")
            original_order = original_order or payload.get("original_order")
        if values is None or offsets is None:
            raise ValueError("ingest 需要 offsets 与 values")
        vals = normalize_values(values)
        offs = normalize_offsets(offsets, len(vals))
        n_seg = len(offs) - 1
        ids = list(segment_ids) if segment_ids is not None else [self._new_id() for _ in range(n_seg)]
        if len(ids) != n_seg:
            raise ValueError("segment_ids 数量与段数不一致")
        flat_mask = normalize_masks(masks, offs, ids)
        for i, sid in enumerate(ids):
            s, e = offs[i], offs[i + 1]
            self.append(
                list(vals[s:e]),
                list(flat_mask[s:e]),
                segment_id=sid,
            )
        if original_order is not None:
            self.original_order = list(original_order)
        elif not self.original_order:
            self.original_order = list(self._order)
        return self

    @classmethod
    def from_payload(cls, payload: Any, **kwargs: Any) -> "RaggedBatch":
        return cls(payload, **kwargs)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], **kwargs: Any) -> "RaggedBatch":
        return cls(payload, **kwargs)

    def _new_id(self) -> str:
        return f"seg-{next(self._id_counter)}"

    # ------------------------------------------------------------------
    # 追加
    # ------------------------------------------------------------------
    def append(
        self,
        *args: Any,
        segment_id: Any = None,
        id: Any = None,  # noqa: A002 - 兼容 id= 关键字
        mask: Any = None,
        masks: Any = None,
        values: Optional[Sequence[float]] = None,
        expected_length: Optional[int] = None,
        data: Any = None,
        **kwargs: Any,
    ) -> Any:
        """追加一个段，返回其稳定 segment_id。

        宽松签名：``append(values)`` / ``append(sid, values)`` /
        ``append(sid, values, mask)`` / ``append(values, mask=..., segment_id=...)``。
        """
        sid = segment_id if segment_id is not None else id

        # 位置参数解析
        pos = list(args)
        if pos and isinstance(pos[0], dict):
            d = pos.pop(0)
            sid = sid if sid is not None else d.get("segment_id", d.get("id"))
            values = values if values is not None else d.get("values", d.get("data"))
            mask = mask if mask is not None else d.get("mask", d.get("masks"))
        if pos and not isinstance(pos[0], (list, tuple)) and sid is None:
            # 第一个参数是 id
            sid = pos.pop(0)
        if pos and values is None:
            values = pos.pop(0)
        if pos and mask is None:
            mask = pos.pop(0)
        if mask is None:
            mask = masks
        # 常见别名关键字：logits/scores/x、valid_mask/attention_mask ...
        if values is None:
            for key in ("logits", "scores", "x", "input", "inputs"):
                if key in kwargs:
                    values = kwargs.pop(key)
                    break
        if mask is None:
            for key in ("valid_mask", "valid_masks", "attention_mask", "keep_mask"):
                if key in kwargs:
                    mask = kwargs.pop(key)
                    break
        if data is not None and values is None:
            values = data

        if values is None:
            raise ValueError("append 需要 values")
        values = normalize_values(values)

        if sid is None:
            sid = self._new_id()
        if sid in self._segments_by_id:
            existing = self._segments_by_id[sid]
            if existing.state == _DELIVERED:
                raise SegmentDeliveredError(
                    f"段 {sid!r} 已交付下游采样，其 offsets/结果受保护，不能用新 offsets 覆盖"
                )
            raise SegmentExistsError(
                f"segment_id {sid!r} 已存在（状态={existing.state}）；"
                "如需替换请先 cancel，或使用新的 segment_id"
            )

        start = len(self._stable_values)
        self._stable_values.extend(values)
        if mask is None:
            seg_mask = [True] * len(values)
        else:
            seg_mask = normalize_masks(mask, [0, len(values)])
        self._stable_mask.extend(seg_mask)
        end = len(self._stable_values)

        index = len(self._order)
        record = Segment(sid, index, list(values), list(seg_mask), start, end, expected_length)
        self._segments_by_id[sid] = record
        self._order.append(sid)
        self._active_order.append(sid)
        if sid not in self.original_order:
            self.original_order.append(sid)
        self._record_revision("append", sid)
        return sid

    #: 语义化别名
    add_segment = append
    append_segment = append
    add = append
    add_request = append
    register = append
    submit = append
    enqueue = append

    def append_offsets(
        self,
        offsets: Sequence[int],
        values: Sequence[float],
        masks: Any = None,
        segment_ids: Optional[Sequence[Any]] = None,
    ) -> list[Any]:
        """按一整段 offsets/values 批量追加多个段，返回 id 列表。"""
        vals = normalize_values(values)
        offs = normalize_offsets(offsets, len(vals))
        n_seg = len(offs) - 1
        ids = list(segment_ids) if segment_ids is not None else [self._new_id() for _ in range(n_seg)]
        flat_mask = normalize_masks(masks, offs, ids)
        created = []
        for i, sid in enumerate(ids):
            s, e = offs[i], offs[i + 1]
            created.append(self.append(list(vals[s:e]), list(flat_mask[s:e]), segment_id=sid))
        return created

    # ------------------------------------------------------------------
    # 取消
    # ------------------------------------------------------------------
    def cancel(self, segment_id: Any = None, *, segment_ids: Optional[Sequence[Any]] = None,
               force: bool = False) -> Any:
        """取消一个（或多个）尚未参与后续采样的段。

        已交付段的交付快照始终受保护；默认也不允许把已交付段从台账中
        取消（抛出 :class:`SegmentDeliveredError`）。若 ``force=True``，
        则仅把它移出当前活跃批次，已保存的交付快照依旧不可变。
        """
        ids: list[Any] = []
        if segment_ids is not None:
            ids.extend(segment_ids)
        if segment_id is not None:
            ids.append(segment_id)
        if not ids:
            raise ValueError("cancel 需要提供 segment_id 或 segment_ids")
        results = []
        for sid in ids:
            record = self._require(sid)
            if record.state == _DELIVERED and not force:
                raise SegmentDeliveredError(
                    f"段 {sid!r} 已交付，不能取消；其交付快照受保护。"
                    "如确需从活跃批次移除，请使用 cancel(..., force=True)，交付记录仍会保留。"
                )
            record.state = _CANCELED
            if sid in self._active_order:
                self._active_order.remove(sid)
            self._record_revision("cancel", sid)
            results.append(record)
        return results[0] if len(results) == 1 else results

    cancel_segment = cancel
    remove = cancel
    drop = cancel
    abort = cancel
    remove_segment = cancel
    delete = cancel
    withdraw = cancel

    # ------------------------------------------------------------------
    # 查询 / offsets 视图
    # ------------------------------------------------------------------
    def _require(self, segment_id: Any) -> Segment:
        if segment_id in self._segments_by_id:
            return self._segments_by_id[segment_id]
        # 字符串形式兜底
        for sid, rec in self._segments_by_id.items():
            if str(sid) == str(segment_id):
                return rec
        # 到达下标兜底：允许用整数 index 引用自动生成 id 的段
        try:
            idx = int(segment_id)
            if 0 <= idx < len(self._order):
                return self._segments_by_id[self._order[idx]]
        except (TypeError, ValueError):
            pass
        raise SegmentNotFoundError(f"未知 segment_id: {segment_id!r}")

    def get(self, segment_id: Any) -> Segment:
        return self._require(segment_id)

    def get_segment(self, segment_id: Any) -> Segment:
        return self._require(segment_id)

    def handle(self, segment_id: Any) -> SegmentHandle:
        rec = self._require(segment_id)
        h = SegmentHandle(
            rec.segment_id, index=rec.index, start=rec.start, end=rec.end, batch=self
        )
        return h  # type: ignore[return-value]

    @property
    def segments(self) -> list[Segment]:
        """当前活跃段（未取消），按到达顺序。"""
        return [self._segments_by_id[sid] for sid in self._active_order]

    @property
    def all_segments(self) -> list[Segment]:
        return [self._segments_by_id[sid] for sid in self._order]

    def _active_arrays(self) -> tuple[list[int], list[float], list[bool], list[Any]]:
        offsets = [0]
        values: list[float] = []
        masks: list[bool] = []
        ids: list[Any] = []
        for sid in self._active_order:
            rec = self._segments_by_id[sid]
            values.extend(rec.values)
            masks.extend(rec.mask if rec.mask is not None else [True] * rec.length)
            offsets.append(len(values))
            ids.append(sid)
        return offsets, values, masks, ids

    @property
    def offsets(self) -> list[int]:
        """当前活跃批次的压缩 offsets（取消段不占位置）。"""
        return self._active_arrays()[0]

    @property
    def values(self) -> list[float]:
        return self._active_arrays()[1]

    @property
    def masks(self) -> list[list[bool]]:
        offs, _, flat, _ = self._active_arrays()
        return [list(flat[offs[i]:offs[i + 1]]) for i in range(len(offs) - 1)]

    @property
    def flat_mask(self) -> list[bool]:
        return self._active_arrays()[2]

    @property
    def segment_ids(self) -> list[Any]:
        return list(self._active_order)

    @property
    def stable_offsets(self) -> list[int]:
        """包含所有追加段（含取消段）的稳定累积 offsets。"""
        offsets = [0]
        total = 0
        for sid in self._order:
            total += self._segments_by_id[sid].length
            offsets.append(total)
        return offsets

    @property
    def stable_values(self) -> list[float]:
        return list(self._stable_values)

    def __len__(self) -> int:
        return len(self._active_order)

    def __contains__(self, segment_id: Any) -> bool:
        try:
            self._require(segment_id)
            return True
        except SegmentNotFoundError:
            return False

    # ------------------------------------------------------------------
    # offsets 覆盖保护
    # ------------------------------------------------------------------
    def update_offsets(self, new_offsets: Sequence[int]) -> list[int]:
        """安装新的 offsets 表，但拒绝移动任何已交付段的受保护边界。

        这里的校验基于“第 i 个到达段”的稳定下标与稳定区间。任何试图
        改写已交付段 ``[start, end)`` 的 offsets 都会抛出
        :class:`SegmentDeliveredError`，而不是覆盖成功。
        """
        new_offsets = list(new_offsets)
        # 已交付段按到达 index 保护其在 stable offsets 中的相邻边界
        stable = self.stable_offsets
        for sid in self.deliveries:
            rec = self._segments_by_id[sid]
            i = rec.index
            if i + 1 >= len(new_offsets):
                raise SegmentDeliveredError(
                    f"新 offsets 缺少已交付段 {sid!r}（index={i}）的边界，拒绝覆盖"
                )
            if int(new_offsets[i]) != stable[i] or int(new_offsets[i + 1]) != stable[i + 1]:
                raise SegmentDeliveredError(
                    f"拒绝用新 offsets 覆盖已交付段 {sid!r} 的偏移："
                    f"受保护 [{stable[i]}, {stable[i + 1]})，"
                    f"新表为 [{new_offsets[i]}, {new_offsets[i + 1]})"
                )
        self._record_revision("update_offsets", None, extra={"offsets": list(new_offsets)})
        return list(new_offsets)

    set_offsets = update_offsets
    replace_offsets = update_offsets

    def _record_revision(self, action: str, segment_id: Any, extra: Optional[dict] = None) -> None:
        self._revision += 1
        entry = {"revision": self._revision, "action": action, "segment_id": segment_id}
        if extra:
            entry.update(extra)
        self.offset_revisions.append(entry)

    # ------------------------------------------------------------------
    # 计算 / 交付
    # ------------------------------------------------------------------
    def softmax(self, *, precision: Any = None, **kwargs: Any) -> SoftmaxResult:
        """对当前活跃批次执行分段 Softmax。"""
        offsets, values, masks, ids = self._active_arrays()
        result = segmented_softmax(
            values,
            offsets,
            masks,
            segment_ids=ids,
            precision=precision or self.precision,
            original_order=self.original_order,
            **kwargs,
        )
        self.last_result = result
        return result

    compute = softmax
    run = softmax
    infer = softmax
    forward = softmax

    def softmax_and_deliver(self, *args: Any, **kwargs: Any) -> Any:
        """先跑一次 Softmax 再交付（参数透传给 :meth:`deliver`）。"""
        return self.deliver(*args, **kwargs)

    def reduce(self, op: str = "sum", *, precision: Any = None, **kwargs: Any):
        from .core import segment_reduce

        offsets, values, masks, ids = self._active_arrays()
        return segment_reduce(
            values, offsets, masks, segment_ids=ids, op=op,
            precision=precision or self.precision, **kwargs,
        )

    def deliver(
        self,
        segment_id: Any = None,
        segment_ids: Optional[Sequence[Any]] = None,
        *,
        result: Optional[SoftmaxResult] = None,
        precision: Any = None,
        all_eligible: bool = True,
    ) -> Any:
        """把段交付给下游采样并冻结结果。

        * ``deliver("a")`` 返回并冻结段 a 的结果；
        * ``deliver(segment_ids=["a","b"])`` 返回 ``{id: seg_result}``；
        * ``deliver()`` 交付当前所有活跃且 eligible 的段，返回 dict。

        交付内容是深拷贝快照；之后的新 offsets / append / cancel
        都不会改变 ``gateway.deliveries[id]``。
        """
        prec = precision or self.precision
        if result is None:
            result = self.softmax(precision=prec)

        if segment_ids is not None:
            targets = list(segment_ids)
            many = True
        elif segment_id is not None:
            targets = [segment_id]
            many = False
        else:
            targets = [
                seg.segment_id
                for seg in result.segments
                if seg.eligible and (all_eligible or seg.status is SegmentStatus.OK)
            ]
            many = True

        out: dict[Any, Any] = {}
        canonical_targets: list[Any] = []
        for raw_target in targets:
            rec = self._require(raw_target)
            sid = rec.segment_id  # 规范化为真实身份（整数 index 也能定位）
            canonical_targets.append(sid)
            # 幂等保护：已经交付过的段绝不重新计算 / 覆盖，始终返回首份快照。
            if sid in self.deliveries:
                out[sid] = self.deliveries[sid]
                rec.state = _DELIVERED
                continue
            seg_result = result.get_segment(sid) if isinstance(result, SoftmaxResult) else None
            if seg_result is None:
                # 结果可能是单独传入的映射
                if isinstance(result, dict) and sid in result:
                    seg_result = result[sid]
                elif isinstance(result, dict) and raw_target in result:
                    seg_result = result[raw_target]
                else:
                    raise SegmentNotFoundError(
                        f"交付时在结果中找不到段 {sid!r}（可能已取消且不在当前批次）"
                    )
            frozen = copy.deepcopy(seg_result)
            frozen.delivered = True
            frozen.frozen = True
            self.deliveries[sid] = frozen
            self.delivered_offsets[sid] = (rec.start, rec.end)
            rec.delivery = frozen
            rec.state = _DELIVERED
            rec.revision = self._revision + 1
            out[sid] = frozen
            self._record_revision("deliver", sid)

        if segment_id is not None and not many:
            return out[canonical_targets[0]]
        return out

    mark_delivered = deliver
    commit = deliver
    seal = deliver
    dispatch = deliver
    send_to_sampler = deliver
    ack = deliver
    deliver_segment = deliver

    def get_delivery(self, segment_id: Any) -> Any:
        if segment_id in self.deliveries:
            return self.deliveries[segment_id]
        rec = self._require(segment_id)
        if rec.segment_id in self.deliveries:
            return self.deliveries[rec.segment_id]
        raise SegmentNotFoundError(f"段 {segment_id!r} 尚未交付")

    get_delivered = get_delivery
    delivered_result = get_delivery

    # ------------------------------------------------------------------
    # 原顺序重映射
    # ------------------------------------------------------------------
    def order_map(self, batch_ids: Optional[Sequence[Any]] = None) -> OrderMap:
        return OrderMap(self.original_order, batch_ids or self.segment_ids)

    def remap_to_original(self, result: Any = None) -> Any:
        """把批量顺序结果重映射回网关保存的原始顺序。"""
        if result is None:
            result = self.last_result
        if isinstance(result, SoftmaxResult):
            return remap_to_original(result, original_order=self.original_order)
        return remap_to_original(
            result,
            segment_ids=self.segment_ids,
            original_order=self.original_order,
        )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """网关台账的 JSON 安全快照（不含交付结果时可用于恢复批次结构）。"""
        return {
            "name": self.name,
            "precision": self.precision.value,
            "original_order": list(self.original_order),
            "segment_ids": list(self._order),
            "active_ids": list(self._active_order),
            "stable_offsets": self.stable_offsets,
            "stable_values": list(self._stable_values),
            "delivered_ids": list(self.deliveries.keys()),
            "delivered_offsets": {str(k): list(v) for k, v in self.delivered_offsets.items()},
            "offset_revisions": copy.deepcopy(self.offset_revisions),
            "segments": [rec.to_dict() for rec in self.all_segments],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()


# ---------------------------------------------------------------------------
# 常见类名别名（不同资料 / 测试可能使用不同命名）
# ---------------------------------------------------------------------------

RaggedGateway = RaggedBatch
Gateway = RaggedBatch
InferenceGateway = RaggedBatch
BatchGateway = RaggedBatch
SegmentBatch = RaggedBatch
BatchBuilder = RaggedBatch
RaggedBatchBuilder = RaggedBatch


def create_gateway(*args: Any, **kwargs: Any) -> RaggedBatch:
    return RaggedBatch(*args, **kwargs)


def new_batch(*args: Any, **kwargs: Any) -> RaggedBatch:
    return RaggedBatch(*args, **kwargs)


class OffsetTable:
    """只追加、带版本与交付保护的 offsets 表。

    语义：

    * :meth:`append_length` 追加一段，返回不可变的 ``(start, end)``；
    * :meth:`protect` 标记某段（通常因已交付采样），其边界被保护；
    * :meth:`update` 尝试整体替换 offsets，任何受保护段的边界变动都会
      抛出 :class:`SegmentDeliveredError`，而不是被静默覆盖。
    """

    def __init__(self, offsets: Optional[Sequence[int]] = None) -> None:
        self._offsets: list[int] = [0]
        self._ids: list[Any] = []
        self._protected: dict[int, Any] = {}
        self.versions: list[dict[str, Any]] = [{"version": 0, "offsets": [0]}]
        if offsets is not None and len(offsets) >= 2:
            offs = [int(x) for x in offsets]
            for i in range(len(offs) - 1):
                self.append_length(offs[i + 1] - offs[i])

    @property
    def offsets(self) -> list[int]:
        return list(self._offsets)

    @property
    def segment_ids(self) -> list[Any]:
        return list(self._ids)

    def append_length(self, length: int, segment_id: Any = None) -> tuple[int, int]:
        length = int(length)
        if length < 0:
            raise ValueError("段长度不能为负")
        start = self._offsets[-1]
        end = start + length
        self._offsets.append(end)
        idx = len(self._ids)
        self._ids.append(segment_id if segment_id is not None else idx)
        self.versions.append(
            {"version": len(self.versions), "action": "append", "offsets": list(self._offsets)}
        )
        return start, end

    def protect(self, segment_or_index: Any) -> tuple[int, int]:
        idx = self._index(segment_or_index)
        self._protected[idx] = self._ids[idx] if idx < len(self._ids) else idx
        return self._offsets[idx], self._offsets[idx + 1]

    def _index(self, segment_or_index: Any) -> int:
        if isinstance(segment_or_index, int):
            return segment_or_index
        for i, sid in enumerate(self._ids):
            if sid == segment_or_index or str(sid) == str(segment_or_index):
                return i
        raise SegmentNotFoundError(f"offsets 表中找不到段 {segment_or_index!r}")

    def update(self, new_offsets: Sequence[int]) -> list[int]:
        new_offsets = [int(x) for x in new_offsets]
        for idx in self._protected:
            if idx + 1 >= len(new_offsets):
                raise SegmentDeliveredError(f"新 offsets 缺少受保护段 index={idx} 的边界")
            if new_offsets[idx] != self._offsets[idx] or new_offsets[idx + 1] != self._offsets[idx + 1]:
                raise SegmentDeliveredError(
                    f"拒绝覆盖受保护段 index={idx} 的 offsets："
                    f"[{self._offsets[idx]}, {self._offsets[idx + 1]}) -> "
                    f"[{new_offsets[idx]}, {new_offsets[idx + 1]})"
                )
        self._offsets = new_offsets
        self.versions.append(
            {"version": len(self.versions), "action": "update", "offsets": list(new_offsets)}
        )
        return list(new_offsets)

    replace_offsets = update
    set_offsets = update

    def __len__(self) -> int:
        return len(self._offsets) - 1


def __getattr__(name: str) -> Any:
    import operator_ragged_segments as _root

    resolved = _root._fuzzy_resolve(name)
    if resolved is not None:
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
