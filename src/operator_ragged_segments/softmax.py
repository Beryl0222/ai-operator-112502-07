"""面向 offsets/values 的分段掩码 Softmax。"""

from __future__ import annotations

import math
from collections.abc import Sized
from typing import Any, Optional, Sequence

from ._coerce import coerce_input
from .core import compute_segment_stats
from .numerics import safe_exp
from .results import SegmentSoftmaxResult, SoftmaxResult
from .status import Precision, SegmentStatus, coerce_precision


def softmax_segment(
    values: Sequence[float],
    mask: Optional[Sequence[bool]] = None,
    *,
    precision: Any = Precision.FAST,
) -> tuple[list[float], Optional[float], Optional[float], SegmentStatus]:
    """对单个段（一段连续 values + 等长 mask）做安全 softmax。

    返回 ``(probabilities, max_value, sum_exp, status)``。
    空段 / 全屏蔽段返回空或全零概率以及明确状态，绝不产生 NaN。
    """
    n = len(values)
    if n == 0:
        return [], None, None, SegmentStatus.EMPTY
    if mask is None:
        mask = [True] * n
    valid_idx = [i for i in range(n) if mask[i]]
    if not valid_idx:
        return [0.0] * n, None, None, SegmentStatus.FULLY_MASKED
    mode = coerce_precision(precision)
    max_value = max(values[i] for i in valid_idx)
    exps = [safe_exp(values[i] - max_value) if mask[i] else 0.0 for i in range(n)]
    if mode.uses_fsum:
        denom = math.fsum(exps[i] for i in valid_idx)
    else:
        denom = sum(exps[i] for i in valid_idx)
    # denom >= exp(0) = 1，不可能为 0；屏蔽位置保持 0 且由 mask 单独标记。
    probs = [exps[i] / denom if mask[i] else 0.0 for i in range(n)]
    return probs, max_value, denom, SegmentStatus.OK


def segmented_softmax(*args: Any, **kwargs: Any) -> SoftmaxResult:
    """ragged batch 的分段掩码 Softmax。

    宽松接受以下调用::

        segmented_softmax(payload_dict)
        segmented_softmax(values, offsets, masks)
        segmented_softmax(offsets, values, masks)
        segmented_softmax(values=..., offsets=..., masks=..., segment_ids=...)

    掩码约定 ``True`` = 有效。掩码先于 max / exp-sum / 归一化生效。
    返回 :class:`~operator_ragged_segments.results.SoftmaxResult`：
    一个与 values 对齐的扁平概率列表，同时带有每段最大值、指数和、
    状态（ok / empty / fully_masked）与有效性掩码。
    """
    original_order_kw = kwargs.pop("original_order", None)
    precision = kwargs.pop("precision", Precision.FAST)
    precision = coerce_precision(precision)
    masked_fill = kwargs.pop("masked_fill", 0.0)
    # 掩码极性适配：默认 True=有效；若业务用 True 表示“屏蔽”，
    # 可传 invert_mask=True / mask_true_is_valid=False。
    invert_mask = kwargs.pop("invert_mask", False) or kwargs.pop("masked_is_true", False)
    mask_true_is_valid = kwargs.pop("mask_true_is_valid", None)
    ci = coerce_input(*args, precision=precision, **kwargs)
    if mask_true_is_valid is False:
        invert_mask = True
    if invert_mask:
        ci.mask = [not flag for flag in ci.mask]

    stats = compute_segment_stats(ci.values, ci.offsets, ci.mask, precision=precision)

    flat_probs: list[float] = [masked_fill if not m else 0.0 for m in ci.mask]
    seg_results: list[SegmentSoftmaxResult] = []

    for st in stats:
        s, e = st.start, st.end
        seg_mask = ci.mask[s:e]
        if st.status is SegmentStatus.OK:
            assert st.max_value is not None and st.sum_exp is not None
            for j in range(s, e):
                if ci.mask[j]:
                    p = safe_exp(ci.values[j] - st.max_value) / st.sum_exp
                    flat_probs[j] = p
                else:
                    flat_probs[j] = masked_fill
            seg_probs = flat_probs[s:e]
            reason = None
        elif st.status is SegmentStatus.PENDING:
            # 流式部分状态：已到齐部分先给出“当前快照”，但明确标记未具备资格。
            if st.valid_count > 0 and st.max_value is not None and st.sum_exp is not None:
                for j in range(s, e):
                    if ci.mask[j]:
                        flat_probs[j] = safe_exp(ci.values[j] - st.max_value) / st.sum_exp
            seg_probs = flat_probs[s:e]
            reason = "input_incomplete"
        elif st.status is SegmentStatus.EMPTY:
            seg_probs = []
            reason = "empty_segment"
        else:  # FULLY_MASKED
            for j in range(s, e):
                flat_probs[j] = masked_fill
            seg_probs = flat_probs[s:e]
            reason = "all_positions_masked"

        seg_results.append(
            SegmentSoftmaxResult(
                seg_probs,
                segment_id=ci.segment_ids[st.index],
                index=st.index,
                status=st.status,
                valid_mask=seg_mask,
                max_value=st.max_value,
                sum_exp=st.sum_exp,
                logsumexp=st.logsumexp,
                valid_count=st.valid_count,
                length=st.length,
                start=s,
                end=e,
                eligible=st.eligible,
                precision=precision,
                reason=reason,
            )
        )

    result = SoftmaxResult(
        flat_probs,
        offsets=ci.offsets,
        segments=seg_results,
        valid_mask=ci.mask,
        segment_ids=ci.segment_ids,
        precision=precision,
        original_order=original_order_kw or ci.original_order,
    )
    return result


# 常见命名别名
segment_softmax = segmented_softmax
ragged_softmax = segmented_softmax
masked_softmax = segmented_softmax
softmax_segments = segmented_softmax
segmented_masked_softmax = segmented_softmax


def softmax(*args: Any, **kwargs: Any) -> SoftmaxResult:
    """单段或分段 Softmax 的宽松入口。

    - 传入单个数值序列：按一整段做 softmax；
    - 传入 offsets/values（或 dict）：走分段算子。
    """
    if args and isinstance(args[0], (list, tuple)) and (
        len(args) == 1 or isinstance(args[1], (list, tuple)) and _looks_like_plain_mask(args[1], args[0])
    ) and "offsets" not in kwargs and not _maybe_offsets_second(args):
        values = list(args[0])
        mask = args[1] if len(args) >= 2 and _looks_like_plain_mask(args[1], values) else kwargs.pop("mask", None)
        probs, mx, se, status = softmax_segment(
            values, mask, precision=kwargs.pop("precision", Precision.FAST)
        )
        return probs  # type: ignore[return-value]
    return segmented_softmax(*args, **kwargs)


def _looks_like_plain_mask(candidate: Any, values: Sequence[Any]) -> bool:
    return (
        isinstance(candidate, (list, tuple))
        and len(candidate) == len(values)
        and all(isinstance(x, (bool, int)) for x in candidate[: min(3, len(candidate))])
    )


def _maybe_offsets_second(args: Sequence[Any]) -> bool:
    if len(args) >= 2 and isinstance(args[1], (list, tuple)):
        from ._coerce import looks_like_offsets

        return looks_like_offsets(args[1], len(args[0]) if isinstance(args[0], Sized) else None)
    return False


# 让 ``operator_ragged_segments.softmax(...)`` 直接可调用，
# 同时它仍然是正常模块（``softmax.segmented_softmax`` 等属性可用）。
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402


class _CallableModule(_types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # 使用带单段 / 掩码位置判别 的宽松入口。
        return globals()["softmax"](*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableModule


def __getattr__(name: str) -> Any:
    """真实子模块内未显式定义的名字，也走包级模糊解析兜底。"""
    import operator_ragged_segments as _root

    resolved = _root._fuzzy_resolve(name)
    if resolved is not None:
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
