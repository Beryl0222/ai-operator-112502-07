"""核心分段统计与通用分段归约（纯标准库实现）。

所有计算都严格基于 offsets 区间，不需要把 ragged batch 补齐成矩阵。
掩码在任何归约（包括 max）之前生效：被屏蔽位置不会进入最大值，
因此“屏蔽项恰好是最大/最小值”不会污染结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ._coerce import coerce_input
from .numerics import safe_exp
from .results import ReductionResult
from .status import Precision, SegmentStatus, coerce_precision


@dataclass
class SegmentStats:
    """单个段的归约统计（内部结构，同时对外可读）。"""

    index: int
    start: int
    end: int
    length: int
    valid_count: int
    status: SegmentStatus
    max_value: Optional[float]
    sum_exp: Optional[float]
    logsumexp: Optional[float]
    eligible: bool = True

    @property
    def available(self) -> bool:
        return self.status is SegmentStatus.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "valid_count": self.valid_count,
            "status": self.status.value,
            "max_value": self.max_value,
            "sum_exp": self.sum_exp,
            "logsumexp": self.logsumexp,
            "eligible": self.eligible,
            "available": self.available,
        }


def _exp_terms(
    values: Sequence[float],
    mask: Sequence[bool],
    start: int,
    end: int,
    max_value: float,
) -> list[float]:
    terms: list[float] = []
    for j in range(start, end):
        if mask[j]:
            # max 一定来自某个有效元素，因此 max_value 有限；
            # 极小差值会安全下溢为 0.0，不会产生 NaN。
            terms.append(safe_exp(values[j] - max_value))
    return terms


def compute_segment_stats(
    values: Sequence[float],
    offsets: Sequence[int],
    mask: Sequence[bool],
    *,
    precision: Any = Precision.FAST,
    eligible: Optional[Sequence[bool]] = None,
) -> list[SegmentStats]:
    """计算每段的有效元素数、状态、最大值、指数和与 logsumexp。"""
    mode = coerce_precision(precision)
    stats: list[SegmentStats] = []
    n_segments = len(offsets) - 1
    for i in range(n_segments):
        start, end = int(offsets[i]), int(offsets[i + 1])
        length = end - start
        valid_idx = [j for j in range(start, end) if mask[j]]
        valid_count = len(valid_idx)
        is_eligible = True if eligible is None else bool(eligible[i])
        status = SegmentStatus.classify(length, valid_count, complete=is_eligible)

        if not is_eligible:
            # 流式未到齐：仍尽量给出可恢复的部分统计。
            if valid_count == 0:
                stats.append(
                    SegmentStats(i, start, end, length, 0, SegmentStatus.PENDING,
                                 None, None, None, eligible=False)
                )
                continue
            max_value = max(values[j] for j in valid_idx)
            terms = _exp_terms(values, mask, start, end, max_value)
            sum_exp = math.fsum(terms) if mode.uses_fsum else sum(terms)
            stats.append(
                SegmentStats(
                    i, start, end, length, valid_count, SegmentStatus.PENDING,
                    max_value, sum_exp, max_value + math.log(sum_exp), eligible=False,
                )
            )
            continue

        if length == 0:
            stats.append(
                SegmentStats(i, start, end, 0, 0, SegmentStatus.EMPTY,
                             None, None, None, eligible=True)
            )
            continue
        if valid_count == 0:
            # 明确的全屏蔽 / 不可用状态：不制造 NaN。
            stats.append(
                SegmentStats(i, start, end, length, 0, SegmentStatus.FULLY_MASKED,
                             None, None, None, eligible=True)
            )
            continue

        max_value = max(values[j] for j in valid_idx)
        terms = _exp_terms(values, mask, start, end, max_value)
        # 至少最大值对应项 exp(0)=1，所以 sum_exp >= 1，不会出现 0/0。
        sum_exp = math.fsum(terms) if mode.uses_fsum else sum(terms)
        logsumexp = max_value + math.log(sum_exp)
        stats.append(
            SegmentStats(
                i, start, end, length, valid_count, SegmentStatus.OK,
                max_value, sum_exp, logsumexp, eligible=True,
            )
        )
    return stats


# ---------------------------------------------------------------------------
# 通用分段归约
# ---------------------------------------------------------------------------

def _reduce_max(vals: list[float]) -> Optional[float]:
    return max(vals) if vals else None


def _reduce_min(vals: list[float]) -> Optional[float]:
    return min(vals) if vals else None


def _reduce_mean(vals: list[float]) -> Optional[float]:
    return math.fsum(vals) / len(vals) if vals else None


def _reduce_sum(vals: list[float]) -> float:
    return math.fsum(vals) if vals else 0.0


def _reduce_prod(vals: list[float]) -> float:
    result = 1.0
    for v in vals:
        result *= v
    return result


def _reduce_any(vals: list[float]) -> bool:
    return any(bool(v) for v in vals)


def _reduce_all(vals: list[float]) -> bool:
    return all(bool(v) for v in vals) if vals else True


def _reduce_first(vals: list[float]) -> Optional[float]:
    return vals[0] if vals else None


def _reduce_last(vals: list[float]) -> Optional[float]:
    return vals[-1] if vals else None


_REDUCERS: dict[str, Callable[[list[float]], Any]] = {
    "max": _reduce_max,
    "amax": _reduce_max,
    "min": _reduce_min,
    "amin": _reduce_min,
    "sum": _reduce_sum,
    "add": _reduce_sum,
    "mean": _reduce_mean,
    "avg": _reduce_mean,
    "average": _reduce_mean,
    "prod": _reduce_prod,
    "product": _reduce_prod,
    "any": _reduce_any,
    "or": _reduce_any,
    "all": _reduce_all,
    "and": _reduce_all,
    "first": _reduce_first,
    "last": _reduce_last,
}

#: 这些 op 在“没有有效元素”时有代数单位元；其余返回 None（明确不可用）。
_IDENTITY_OPS = {"sum": 0.0, "add": 0.0, "prod": 1.0, "product": 1.0,
                 "any": False, "or": False, "all": True, "and": True,
                 "count": 0, "valid_count": 0, "len": 0}


def segment_reduce(*args: Any, op: str = "sum", **kwargs: Any) -> ReductionResult:
    """按 offsets 分段应用归约。

    调用形态与 :func:`segmented_softmax` 一样宽松。``op`` 支持
    ``max/min/sum/mean/prod/any/all/first/last/count/valid_count``。
    掩码先于归约生效。不可用段（空 / 全屏蔽）的数值默认是 ``None``，
    可通过 :meth:`ReductionResult.as_list` 替换填充值。
    """
    ci = coerce_input(*args, **kwargs)
    op_name = str(op).lower()
    if op_name in ("count", "valid_count", "num_valid"):
        values_out: list[Any] = []
        for i in range(len(ci.offsets) - 1):
            s, e = ci.offsets[i], ci.offsets[i + 1]
            values_out.append(sum(1 for j in range(s, e) if ci.mask[j]))
        stats = compute_segment_stats(ci.values, ci.offsets, ci.mask)
        return ReductionResult(
            values_out, op="valid_count", statuses=[st.status for st in stats],
            segment_ids=ci.segment_ids, offsets=ci.offsets,
            valid_counts=[v for v in values_out],
            precision=kwargs.get("precision", "fast"),
        )
    if op_name in ("len", "length", "size"):
        values_out = [ci.offsets[i + 1] - ci.offsets[i] for i in range(len(ci.offsets) - 1)]
        stats = compute_segment_stats(ci.values, ci.offsets, ci.mask)
        return ReductionResult(
            values_out, op="length", statuses=[st.status for st in stats],
            segment_ids=ci.segment_ids, offsets=ci.offsets,
            precision=kwargs.get("precision", "fast"),
        )

    reducer = _REDUCERS.get(op_name)
    if reducer is None:
        raise ValueError(f"不支持的归约 op: {op!r}，可选: {sorted(set(_REDUCERS))}")

    mode = coerce_precision(kwargs.get("precision", Precision.FAST))
    stats = compute_segment_stats(ci.values, ci.offsets, ci.mask, precision=mode)
    out: list[Any] = []
    for st in stats:
        if st.status in (SegmentStatus.EMPTY, SegmentStatus.FULLY_MASKED):
            if op_name in _IDENTITY_OPS:
                out.append(_IDENTITY_OPS[op_name])
            else:
                out.append(None)
            continue
        vals = [ci.values[j] for j in range(st.start, st.end) if ci.mask[j]]
        if op_name in ("mean", "avg", "average") and mode is Precision.PRECISE:
            out.append(math.fsum(vals) / len(vals))
        elif op_name in ("sum", "add") and mode is Precision.PRECISE:
            out.append(math.fsum(vals))
        else:
            out.append(reducer(vals))
    return ReductionResult(
        out, op=op_name, statuses=[st.status for st in stats],
        segment_ids=ci.segment_ids, offsets=ci.offsets,
        valid_counts=[st.valid_count for st in stats], precision=mode,
    )


# ---- 命名快捷算子 ---------------------------------------------------------

def segment_max(*args: Any, **kwargs: Any) -> ReductionResult:
    return segment_reduce(*args, op="max", **kwargs)


def segment_min(*args: Any, **kwargs: Any) -> ReductionResult:
    return segment_reduce(*args, op="min", **kwargs)


def segment_sum(*args: Any, **kwargs: Any) -> ReductionResult:
    return segment_reduce(*args, op="sum", **kwargs)


def segment_mean(*args: Any, **kwargs: Any) -> ReductionResult:
    return segment_reduce(*args, op="mean", **kwargs)


def segment_count(*args: Any, **kwargs: Any) -> ReductionResult:
    return segment_reduce(*args, op="valid_count", **kwargs)


def segment_logsumexp(*args: Any, **kwargs: Any) -> ReductionResult:
    ci = coerce_input(*args, **kwargs)
    mode = coerce_precision(kwargs.get("precision", Precision.FAST))
    stats = compute_segment_stats(ci.values, ci.offsets, ci.mask, precision=mode)
    return ReductionResult(
        [st.logsumexp for st in stats], op="logsumexp",
        statuses=[st.status for st in stats], segment_ids=ci.segment_ids,
        offsets=ci.offsets, valid_counts=[st.valid_count for st in stats], precision=mode,
    )


def segment_sum_exp(*args: Any, **kwargs: Any) -> ReductionResult:
    """每段有效元素的指数和（sum of exp(x - max)，平移不影响归一化比值）。"""
    ci = coerce_input(*args, **kwargs)
    mode = coerce_precision(kwargs.get("precision", Precision.FAST))
    stats = compute_segment_stats(ci.values, ci.offsets, ci.mask, precision=mode)
    return ReductionResult(
        [st.sum_exp for st in stats], op="sum_exp",
        statuses=[st.status for st in stats], segment_ids=ci.segment_ids,
        offsets=ci.offsets, valid_counts=[st.valid_count for st in stats], precision=mode,
    )


def __getattr__(name: str) -> Any:
    import operator_ragged_segments as _root

    resolved = _root._fuzzy_resolve(name)
    if resolved is not None:
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
