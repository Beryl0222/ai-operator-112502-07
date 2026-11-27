"""高层函数式入口：一次调用完成解析、归约、Softmax 与（可选）原顺序重映射。"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ._coerce import coerce_input
from .core import (
    compute_segment_stats,
    segment_logsumexp,
    segment_max,
    segment_mean,
    segment_reduce,
    segment_sum,
    segment_sum_exp,
)
from .remap import remap_to_original
from .softmax import segmented_softmax
from .status import PrecisionPolicy, coerce_precision, precision_for_tier


def compute(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """执行完整的分段 Softmax 契约，返回 JSON 友好的字典。

    额外支持：

    * ``tier``：请求等级，自动选择 fast/precise 精度路径；
    * ``remap=True``：按 ``original_order`` 重映射；
    * ``return_stats=True``：附带每段 max / sum_exp / logsumexp。
    """
    tier = kwargs.pop("tier", None)
    if tier is not None and "precision" not in kwargs:
        policy = PrecisionPolicy.from_tier(tier)
        kwargs["precision"] = policy.mode
    do_remap = kwargs.pop("remap", False)
    original_order = kwargs.pop("original_order", None)
    with_stats = kwargs.pop("return_stats", True)

    result = segmented_softmax(*args, original_order=original_order, **kwargs)
    if do_remap and (original_order or result.original_order):
        result = remap_to_original(result, original_order=original_order)

    payload = result.to_dict()
    if with_stats:
        payload["stats"] = [
            {
                "segment_id": seg.segment_id,
                "status": seg.status.value,
                "max_value": seg.max_value,
                "sum_exp": seg.sum_exp,
                "logsumexp": seg.logsumexp,
                "valid_count": seg.valid_count,
                "eligible": seg.eligible,
                "available": seg.available,
            }
            for seg in result.segments
        ]
    return payload


# 语义化别名
run_operator = compute
operator = compute
apply_operator = compute


def precision_policy_for(tier: Any) -> PrecisionPolicy:
    return PrecisionPolicy.from_tier(tier)


__all__ = [
    "compute",
    "run_operator",
    "operator",
    "apply_operator",
    "precision_policy_for",
    "precision_for_tier",
    "coerce_precision",
    "segmented_softmax",
    "segment_reduce",
    "segment_max",
    "segment_sum",
    "segment_mean",
    "segment_logsumexp",
    "segment_sum_exp",
    "compute_segment_stats",
    "remap_to_original",
]
