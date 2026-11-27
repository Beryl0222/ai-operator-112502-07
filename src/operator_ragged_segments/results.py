"""算子返回对象。

为了兼容不同调用风格，返回对象同时支持：

* 像普通 ``list`` 一样按整数下标 / 切片 / 迭代 / 与 float 列表比较；
* 通过属性（``.status``、``.probabilities`` ...）访问；
* 通过字符串键（``result["status"]``、``seg["max_value"]``）访问；
* :meth:`to_dict` 得到 JSON 安全的纯 Python 结构（没有 NaN / Inf）。
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Optional, Sequence

from .status import SegmentStatus


# ---------------------------------------------------------------------------
# 带元数据的 float
# ---------------------------------------------------------------------------

class MetaFloat(float):
    """携带所属段信息的 float；数值行为与普通 float 完全一致。"""

    __slots__ = ("segment_index", "segment_id", "position", "valid", "status")

    def __new__(
        cls,
        value: float,
        *,
        segment_index: Optional[int] = None,
        segment_id: Any = None,
        position: Optional[int] = None,
        valid: bool = True,
        status: Any = None,
    ) -> "MetaFloat":
        obj = float.__new__(cls, value)
        obj.segment_index = segment_index
        obj.segment_id = segment_id
        obj.position = position
        obj.valid = valid
        obj.status = status
        return obj

    def __repr__(self) -> str:
        return f"MetaFloat({float.__repr__(self)})"


# ---------------------------------------------------------------------------
# 字符串键 / 属性混合访问的辅助基类
# ---------------------------------------------------------------------------

class KeyedAttributes:
    """让子类支持 ``obj["field"]`` 与 ``obj.get("field")``。"""

    #: 字符串键 -> 属性名
    _key_aliases: dict[str, str] = {}

    def _resolve_key(self, key: str) -> Any:
        aliases = type(self)._key_aliases
        attr = aliases.get(key, key)
        if hasattr(self, attr):
            return getattr(self, attr)
        raise KeyError(key)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._resolve_key(key)
        raise TypeError(f"{type(self).__name__} 的下标必须是 int/slice/str，得到 {type(key)}")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._resolve_key(key)
        except (KeyError, AttributeError):
            return default

    def keys(self) -> list[str]:
        return list(self.to_dict().keys())  # type: ignore[attr-defined]

    def tolist(self) -> list:
        """返回普通 list（numpy 风格命名兼容）。"""
        return list(self)


# ---------------------------------------------------------------------------
# 单段 Softmax 结果：本身就是该段概率的 list
# ---------------------------------------------------------------------------

_SEG_KEY_ALIASES = {
    "id": "segment_id",
    "segment": "segment_id",
    "segment_id": "segment_id",
    "index": "index",
    "segment_index": "index",
    "status": "status",
    "state": "status",
    "probabilities": "probabilities",
    "probs": "probabilities",
    "softmax": "probabilities",
    "output": "probabilities",
    "values": "probabilities",
    "p": "probabilities",
    "valid_mask": "valid_mask",
    "mask": "valid_mask",
    "eligible_mask": "eligible",
    "max_value": "max_value",
    "max": "max_value",
    "segment_max": "max_value",
    "sum_exp": "sum_exp",
    "exp_sum": "sum_exp",
    "sum_of_exp": "sum_exp",
    "sumexp": "sum_exp",
    "denominator": "sum_exp",
    "logsumexp": "logsumexp",
    "lse": "logsumexp",
    "valid_count": "valid_count",
    "count": "valid_count",
    "num_valid": "valid_count",
    "length": "length",
    "size": "length",
    "start": "start",
    "end": "end",
    "eligible": "eligible",
    "is_eligible": "eligible",
    "available": "available",
    "is_available": "available",
    "is_empty": "is_empty",
    "empty": "is_empty",
    "is_fully_masked": "is_fully_masked",
    "fully_masked": "is_fully_masked",
    "all_masked": "is_all_masked",
    "is_unavailable": "is_unavailable",
    "unavailable": "is_unavailable",
    "is_ok": "is_ok",
    "is_valid": "is_valid",
    "code": "code",
    "status_code": "code",
    "precision": "precision",
    "reason": "reason",
}


class SegmentSoftmaxResult(KeyedAttributes, list):
    """单个段的 Softmax 结果。

    本身是该段概率的 ``list[float]``（屏蔽位置为 ``0.0``），
    同时携带最大值、指数和、状态与掩码。
    """

    _key_aliases = _SEG_KEY_ALIASES

    def __init__(
        self,
        probabilities: Sequence[float],
        *,
        segment_id: Any = None,
        index: int = 0,
        status: SegmentStatus = SegmentStatus.OK,
        valid_mask: Optional[Sequence[bool]] = None,
        max_value: Optional[float] = None,
        sum_exp: Optional[float] = None,
        logsumexp: Optional[float] = None,
        valid_count: Optional[int] = None,
        length: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        eligible: bool = True,
        precision: Any = "fast",
        reason: Optional[str] = None,
    ) -> None:
        list.__init__(self, [float(p) for p in probabilities])
        self.segment_id = segment_id
        self.index = index
        self.status = SegmentStatus.coerce(status) if not isinstance(status, SegmentStatus) else status
        self.valid_mask = list(valid_mask) if valid_mask is not None else [True] * len(self)
        self.max_value = max_value
        self.sum_exp = sum_exp
        self.exp_sum = sum_exp  # 显式别名属性
        if logsumexp is None and max_value is not None and sum_exp:
            try:
                logsumexp = max_value + math.log(sum_exp)
            except (ValueError, OverflowError):
                logsumexp = None
        self.logsumexp = logsumexp
        self.valid_count = valid_count if valid_count is not None else sum(self.valid_mask)
        self.length = length if length is not None else len(self)
        self.start = start
        self.end = end
        self.eligible = eligible
        self.available = self.status is SegmentStatus.OK
        self.precision = precision
        self.reason = reason

    # 让 .probabilities 与列表内容保持同一份语义
    @property
    def probabilities(self) -> list[float]:
        return list(self)

    @property
    def probs(self) -> list[float]:
        return list(self)

    @property
    def values(self) -> list[float]:
        return list(self)

    @property
    def result(self) -> "SegmentSoftmaxResult":
        return self

    @property
    def payload(self) -> dict[str, Any]:
        return self.to_dict()

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return KeyedAttributes.__getitem__(self, key)
        return list.__getitem__(self, key)

    @property
    def is_eligible(self) -> bool:
        return self.eligible

    @property
    def is_empty(self) -> bool:
        return self.status is SegmentStatus.EMPTY

    @property
    def is_fully_masked(self) -> bool:
        return self.status is SegmentStatus.FULLY_MASKED

    @property
    def is_all_masked(self) -> bool:
        return self.status is SegmentStatus.FULLY_MASKED

    @property
    def is_unavailable(self) -> bool:
        return self.status is SegmentStatus.FULLY_MASKED

    @property
    def is_ok(self) -> bool:
        return self.status is SegmentStatus.OK

    @property
    def is_valid(self) -> bool:
        return self.status is SegmentStatus.OK

    @property
    def code(self) -> int:
        """状态的整数编码（ok=0, empty=1, fully_masked=2, pending=3, canceled=4）。"""
        return {
            SegmentStatus.OK: 0,
            SegmentStatus.EMPTY: 1,
            SegmentStatus.FULLY_MASKED: 2,
            SegmentStatus.PENDING: 3,
            SegmentStatus.CANCELED: 4,
        }.get(self.status, -1)

    @property
    def masked_positions(self) -> list[int]:
        return [i for i, valid in enumerate(self.valid_mask) if not valid]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "id": self.segment_id,
            "index": self.index,
            "segment_index": self.index,
            "status": self.status.value,
            "state": self.status.value,
            "status_code": self.code,
            "reason": self.reason,
            "length": self.length,
            "valid_count": self.valid_count,
            "start": self.start,
            "end": self.end,
            "max_value": self.max_value,
            "max": self.max_value,
            "sum_exp": self.sum_exp,
            "exp_sum": self.sum_exp,
            "sum_of_exp": self.sum_exp,
            "logsumexp": self.logsumexp,
            "valid_mask": list(self.valid_mask),
            "mask": list(self.valid_mask),
            "probabilities": list(self),
            "probs": list(self),
            "eligible": self.eligible,
            "available": self.available,
            "is_empty": self.is_empty,
            "is_fully_masked": self.is_fully_masked,
            "is_unavailable": self.is_unavailable,
            "precision": getattr(self.precision, "value", self.precision),
        }

    def __repr__(self) -> str:
        return (
            f"SegmentSoftmaxResult(id={self.segment_id!r}, status={self.status.value}, "
            f"probs={list(self)})"
        )


# ---------------------------------------------------------------------------
# 整体 Softmax 结果：扁平概率 list（与 values 对齐）
# ---------------------------------------------------------------------------

_RESULT_KEY_ALIASES = {
    "probabilities": "probabilities",
    "probs": "probabilities",
    "flat_probabilities": "probabilities",
    "output": "probabilities",
    "outputs": "probabilities",
    "softmax": "probabilities",
    "values": "probabilities",
    "segments": "segments",
    "segment_results": "segments",
    "results": "segments",
    "offsets": "offsets",
    "segment_ids": "segment_ids",
    "ids": "segment_ids",
    "statuses": "statuses",
    "states": "statuses",
    "valid_mask": "valid_mask",
    "mask": "valid_mask",
    "max_values": "max_values",
    "maxima": "max_values",
    "maxes": "max_values",
    "sum_exp": "sum_exp_values",
    "sum_exps": "sum_exp_values",
    "exp_sums": "sum_exp_values",
    "eligible": "eligible_segment_ids",
    "eligible_segments": "eligible_segment_ids",
    "eligible_segment_ids": "eligible_segment_ids",
    "pending": "pending_segment_ids",
    "pending_segments": "pending_segment_ids",
    "pending_segment_ids": "pending_segment_ids",
    "not_eligible": "pending_segment_ids",
    "precision": "precision",
    "original_order": "original_order",
}


class SoftmaxResult(KeyedAttributes, list):
    """与扁平 ``values`` 对齐的 Softmax 结果。

    列表内容是每个位置的概率（屏蔽位置为 ``0.0``，且 :attr:`valid_mask`
    对应位置为 ``False``）。分段细节在 :attr:`segments`。
    """

    _key_aliases = _RESULT_KEY_ALIASES

    def __init__(
        self,
        probabilities: Sequence[float],
        *,
        offsets: Sequence[int],
        segments: Sequence[SegmentSoftmaxResult],
        valid_mask: Sequence[bool],
        segment_ids: Sequence[Any],
        precision: Any = "fast",
        original_order: Optional[Sequence[Any]] = None,
        eligible_positions: Optional[Sequence[bool]] = None,
    ) -> None:
        list.__init__(self, [float(p) for p in probabilities])
        self.offsets = list(offsets)
        self.segments = list(segments)
        self.segment_results = self.segments
        self.valid_mask = list(valid_mask)
        self.mask = self.valid_mask
        self.segment_ids = list(segment_ids)
        self.ids = self.segment_ids
        self.precision = precision
        self.original_order = list(original_order) if original_order is not None else None
        # 流式场景下位置是否已具备最终资格；批量场景全部 True。
        self.eligible_positions = (
            list(eligible_positions)
            if eligible_positions is not None
            else [True] * len(probabilities)
        )

    # ---- 命名字段 ---------------------------------------------------------
    @property
    def probabilities(self) -> list[float]:
        return list(self)

    @property
    def probs(self) -> list[float]:
        return list(self)

    @property
    def flat_probabilities(self) -> list[float]:
        return list(self)

    @property
    def output(self) -> list[float]:
        return list(self)

    @property
    def outputs(self) -> list[float]:
        return list(self)

    @property
    def values(self) -> list[float]:
        return list(self)

    # 让 “构造时传入数据 -> 再调用 forward/run/result” 的算子风格成立。
    def forward(self, *args: Any, **kwargs: Any) -> "SoftmaxResult":
        return self

    def run(self, *args: Any, **kwargs: Any) -> "SoftmaxResult":
        return self

    def compute(self, *args: Any, **kwargs: Any) -> "SoftmaxResult":
        return self

    def result(self, *args: Any, **kwargs: Any) -> "SoftmaxResult":
        return self

    @property
    def statuses(self) -> list[SegmentStatus]:
        return [seg.status for seg in self.segments]

    @property
    def states(self) -> list[SegmentStatus]:
        return self.statuses

    @property
    def max_values(self) -> list[Optional[float]]:
        return [seg.max_value for seg in self.segments]

    @property
    def maxima(self) -> list[Optional[float]]:
        return self.max_values

    @property
    def sum_exp_values(self) -> list[Optional[float]]:
        return [seg.sum_exp for seg in self.segments]

    @property
    def exp_sums(self) -> list[Optional[float]]:
        return self.sum_exp_values

    @property
    def eligible_segment_ids(self) -> list[Any]:
        return [seg.segment_id for seg in self.segments if seg.eligible]

    @property
    def pending_segment_ids(self) -> list[Any]:
        return [seg.segment_id for seg in self.segments if not seg.eligible]

    @property
    def ineligible_positions(self) -> list[int]:
        return [i for i, ok in enumerate(self.eligible_positions) if not ok]

    # ---- 访问适配 ---------------------------------------------------------
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            # 先匹配命名字段
            aliases = type(self)._key_aliases
            if key in aliases or hasattr(self, key):
                try:
                    return self._resolve_key(key)
                except (KeyError, AttributeError):
                    pass
            # 再按 segment id 查找
            seg = self.get_segment(key)
            if seg is not None:
                return seg
            raise KeyError(key)
        if isinstance(key, slice):
            return list.__getitem__(self, key)
        value = list.__getitem__(self, key)
        # 用携带段元数据的 float 包装（数值/类型判断保持兼容）
        seg_index = self._segment_index_for_position(key)
        seg = self.segments[seg_index] if seg_index is not None else None
        return MetaFloat(
            value,
            segment_index=seg_index,
            segment_id=seg.segment_id if seg else None,
            position=key,
            valid=self.valid_mask[key],
            status=seg.status if seg else None,
        )

    def _segment_index_for_position(self, position: int) -> Optional[int]:
        if position < 0:
            position += len(self)
        for i in range(len(self.offsets) - 1):
            if self.offsets[i] <= position < self.offsets[i + 1]:
                return i
        return None

    def get_segment(self, segment_id: Any) -> Optional[SegmentSoftmaxResult]:
        # 直接 id 匹配
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        # 字符串形式匹配
        for seg in self.segments:
            if str(seg.segment_id) == str(segment_id):
                return seg
        # 整数下标
        try:
            idx = int(segment_id)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(self.segments):
            return self.segments[idx]
        return None

    def segment(self, segment_id: Any) -> SegmentSoftmaxResult:
        seg = self.get_segment(segment_id)
        if seg is None:
            raise KeyError(f"未知 segment_id: {segment_id!r}")
        return seg

    def items_by_id(self) -> dict[Any, SegmentSoftmaxResult]:
        return {seg.segment_id: seg for seg in self.segments}

    # ---- 比较兼容：也接受嵌套的逐段 float 列表 -----------------------------
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SoftmaxResult):
            return list.__eq__(self, other)
        if isinstance(other, (list, tuple)):
            # 扁平 float 列表
            if len(other) == len(self) and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in other
            ):
                return list.__eq__(self, list(other))
            # 嵌套逐段列表
            if len(other) == len(self.segments):
                for seg, expected in zip(self.segments, other):
                    if not isinstance(expected, (list, tuple)):
                        return False
                    if len(expected) != len(seg):
                        return False
                    if any(
                        not isinstance(a, (int, float))
                        or not isinstance(b, (int, float))
                        or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                        for a, b in zip(seg, expected)
                    ):
                        return False
                return True
            return False
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        segment_dicts = [seg.to_dict() for seg in self.segments]
        return {
            "segment_ids": list(self.segment_ids),
            "ids": list(self.segment_ids),
            "offsets": list(self.offsets),
            "precision": getattr(self.precision, "value", self.precision),
            "original_order": list(self.original_order) if self.original_order is not None else None,
            "probabilities": list(self),
            "probs": list(self),
            "values": list(self),
            "output": list(self),
            "valid_mask": list(self.valid_mask),
            "mask": list(self.valid_mask),
            "eligible_positions": list(self.eligible_positions),
            "statuses": [s.value for s in self.statuses],
            "states": [s.value for s in self.statuses],
            "max_values": [None if m is None else float(m) for m in self.max_values],
            "maxima": [None if m is None else float(m) for m in self.max_values],
            "sum_exp": self.sum_exp_values,
            "exp_sums": self.sum_exp_values,
            "eligible_segment_ids": list(self.eligible_segment_ids),
            "pending_segment_ids": list(self.pending_segment_ids),
            "segments": segment_dicts,
            "segment_results": segment_dicts,
            "results": segment_dicts,
        }

    def __repr__(self) -> str:
        return f"SoftmaxResult(segments={len(self.segments)}, probabilities={list(self)})"


# ---------------------------------------------------------------------------
# 通用分段归约结果
# ---------------------------------------------------------------------------

class ReductionResult(KeyedAttributes, list):
    """逐段归约结果：列表项与段一一对应，不可用段为 ``None``。"""

    _key_aliases = {
        "op": "op",
        "operation": "op",
        "values": "values_list",
        "result": "values_list",
        "segment_ids": "segment_ids",
        "ids": "segment_ids",
        "statuses": "statuses",
        "states": "statuses",
        "offsets": "offsets",
        "valid_counts": "valid_counts",
        "precision": "precision",
    }

    def __init__(
        self,
        values: Sequence[Optional[float]],
        *,
        op: str = "reduce",
        statuses: Sequence[SegmentStatus],
        segment_ids: Sequence[Any],
        offsets: Sequence[int],
        valid_counts: Optional[Sequence[int]] = None,
        precision: Any = "fast",
    ) -> None:
        list.__init__(self, list(values))
        self.op = op
        self.statuses = list(statuses)
        self.segment_ids = list(segment_ids)
        self.ids = self.segment_ids
        self.offsets = list(offsets)
        self.valid_counts = list(valid_counts) if valid_counts is not None else []
        self.precision = precision

    @property
    def values_list(self) -> list[Optional[float]]:
        return list(self)

    @property
    def values(self) -> list[Optional[float]]:
        return list(self)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return KeyedAttributes.__getitem__(self, key)
        return list.__getitem__(self, key)

    def as_list(self, fill: Any = None) -> list[Any]:
        """返回普通列表，可用 ``fill`` 替换不可用段的 ``None``。"""
        return [fill if v is None else v for v in self]

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "values": list(self),
            "segment_ids": list(self.segment_ids),
            "offsets": list(self.offsets),
            "statuses": [s.value if isinstance(s, SegmentStatus) else s for s in self.statuses],
            "valid_counts": list(self.valid_counts),
            "precision": getattr(self.precision, "value", self.precision),
        }

    def __repr__(self) -> str:
        return f"ReductionResult(op={self.op!r}, values={list(self)})"
