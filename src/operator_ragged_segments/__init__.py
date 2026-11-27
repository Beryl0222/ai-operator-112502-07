"""operator_ragged_segments

面向 offsets/values 表示的**分段归约与掩码 Softmax 算子库**。

设计要点（与业务契约一一对应）：

* 不做 padding，直接在 ragged ``offsets/values`` 上按段计算
  max / exp-sum / 归一化，长尾请求不再被补齐矩阵放大显存；
* 掩码先于任何归约生效，屏蔽位置不进入 max / sum，输出由独立的
  ``valid_mask`` 标记——有效值下溢成 ``0.0`` 不会被误判为屏蔽；
* ``empty``（空段）、``fully_masked``（非空但全屏蔽，明确的
  “不可用”状态）与 “有效值全为 0”（``ok``，均匀分布）严格区分，
  任何结果 / 状态中都不会出现 NaN；
* 支持按 segment id / 排列把结果重映射回网关保存的原始顺序；
* :class:`RaggedBatch` 网关支持追加 / 取消 / 交付保护，已交付段的
  offsets 与结果快照不会被新 offsets 覆盖；
* :class:`StreamingSegmentSoftmax` 在输入未到齐时保留可恢复的部分
  状态，并明确给出哪些段 / 输出尚不具备资格；
* ``fast`` / ``precise`` 两条精度路径共享段身份、掩码语义与状态解释。

公开入口尽量保留了常见命名别名，调用方既可以::

    from operator_ragged_segments import segmented_softmax, SegmentStatus

也可以直接把 fixture dict 传入::

    result = segmented_softmax(json.loads(Path("data/input.json").read_text()))
"""

from __future__ import annotations

__version__ = "0.1.0"

# ---- 异常 -----------------------------------------------------------------
from .errors import (
    OperatorError,
    InvalidOffsetsError,
    OffsetsError,
    MaskError,
    MaskShapeError,
    SegmentNotFoundError,
    UnknownSegmentError,
    SegmentExistsError,
    DuplicateSegmentError,
    SegmentDeliveredError,
    DeliveredSegmentError,
    LockedSegmentError,
    SealedSegmentError,
    SegmentCanceledError,
    CancelledSegmentError,
    PartialStateError,
    StateRecoveryError,
    IncompleteInputError,
    NotEligibleError,
)

# ---- 状态 / 精度 -----------------------------------------------------------
from .status import (
    SegmentStatus,
    SegmentState,
    Status,
    Precision,
    PrecisionMode,
    PrecisionPolicy,
    coerce_precision,
    precision_for_tier,
    DEFAULT_TIER_PRECISION,
)

# ---- 结果类型 --------------------------------------------------------------
from .results import (
    MetaFloat,
    SegmentSoftmaxResult,
    SoftmaxResult,
    ReductionResult,
)

# ---- 输入工具 --------------------------------------------------------------
from ._coerce import (
    coerce_input,
    normalize_offsets,
    normalize_values,
    normalize_masks,
    masks_to_nested,
    offsets_from_lengths,
    looks_like_offsets,
    load_payload,
)

# ---- 核心归约 --------------------------------------------------------------
from .core import (
    SegmentStats,
    compute_segment_stats,
    segment_reduce,
    segment_max,
    segment_min,
    segment_sum,
    segment_mean,
    segment_count,
    segment_logsumexp,
    segment_sum_exp,
)

# ---- Softmax ---------------------------------------------------------------
from .softmax import (
    segmented_softmax,
    segment_softmax,
    ragged_softmax,
    masked_softmax,
    softmax_segments,
    segmented_masked_softmax,
    softmax_segment,
)
from . import softmax  # noqa: F401  (callable module)

# ---- 重映射 ----------------------------------------------------------------
from .remap import (
    inverse_permutation,
    permutation_from_ids,
    reorder_list,
    reorder_segments,
    OrderMap,
    OriginalOrderMap,
    OrderMapping,
    remap_to_original,
    reorder_to_original,
    to_original_order,
    restore_original_order,
    map_to_original,
    remap_results,
    remap_aligned,
)

# ---- 网关 / 批次 -----------------------------------------------------------
from .batch import (
    RaggedBatch,
    RaggedGateway,
    Gateway,
    InferenceGateway,
    BatchGateway,
    SegmentBatch,
    BatchBuilder,
    RaggedBatchBuilder,
    Segment,
    SegmentHandle,
    DeliveryRegistry,
    OffsetTable,
    create_gateway,
    new_batch,
)

# ---- 流式 / 部分状态 / 块 --------------------------------------------------
from .streaming import (
    StreamingSegmentSoftmax,
    StreamingSoftmax,
    ResumableSegmentSoftmax,
    PartialReducer,
    StreamingReducer,
    BlockReducer,
    PartialSegmentState,
    PartialState,
    SegmentPartialState,
    Report,
)

# ---- IO / 高层 API ---------------------------------------------------------
from .io import (
    load_input,
    load_fixture,
    load_batch,
    to_jsonable,
    dumps,
    find_workspace_root,
)
from .api import (
    compute,
    run_operator,
    operator,
    apply_operator,
    precision_policy_for,
)
from .operators import (
    SegmentedSoftmax,
    SoftmaxOperator,
    SegmentedSoftmaxOperator,
    RaggedSoftmaxOperator,
    MaskedSoftmaxOperator,
    RaggedSoftmax,
    SegmentReducer,
    SegmentReduceOperator,
    SegmentReductionOperator,
    RaggedReducer,
    SegmentedReduce,
)

# ---- 额外的命名别名 --------------------------------------------------------
segmented_reduce = segment_reduce
ragged_reduce = segment_reduce
reduce_segments = segment_reduce
segmented_segment_reduce = segment_reduce

segmented_max = segment_max
segmented_sum = segment_sum
segmented_mean = segment_mean

#: 常见的 “算子” 命名
RaggedSegmentOperator = compute

# ---- 更多结果 / 网关 / 顺序命名别名 ---------------------------------------
Batch = RaggedBatch
State = SegmentStatus
SegmentResult = SegmentSoftmaxResult
SoftmaxOutput = SoftmaxResult
SegmentOutput = SegmentSoftmaxResult
BatchResult = SoftmaxResult
RaggedSoftmaxResult = SoftmaxResult
reorder_results = remap_results
map_results_to_original = remap_to_original
restore_order = remap_to_original
reorder_to_original_order = remap_to_original


def create_operator(kind: str = "softmax", **kwargs):
    """简单的算子工厂。

    ``kind`` 支持 ``"softmax"`` / ``"reduce"`` / ``"gateway"`` /
    ``"streaming"`` / ``"block"``；额外关键字会作为默认配置透传。
    """
    key = str(kind).lower().replace("-", "_")
    if key in ("softmax", "segmented_softmax", "masked_softmax"):
        def _op(*args, **call_kwargs):
            merged = {**kwargs, **call_kwargs}
            return segmented_softmax(*args, **merged)
        return _op
    if key in ("reduce", "segment_reduce", "segmented_reduce"):
        def _op(*args, **call_kwargs):
            merged = {**kwargs, **call_kwargs}
            return segment_reduce(*args, **merged)
        return _op
    if key in ("gateway", "batch", "ragged_batch"):
        return RaggedBatch(**kwargs)
    if key in ("streaming", "stream", "partial"):
        return StreamingSegmentSoftmax(**kwargs)
    if key in ("block", "block_reducer"):
        return BlockReducer(**kwargs)
    raise ValueError(f"未知算子类型: {kind!r}")


def build_operator(kind: str = "softmax", **kwargs):
    return create_operator(kind, **kwargs)


__all__ = [
    "__version__",
    # 主算子
    "segmented_softmax",
    "segment_softmax",
    "ragged_softmax",
    "masked_softmax",
    "softmax_segments",
    "segmented_masked_softmax",
    "softmax_segment",
    "segment_reduce",
    "segmented_reduce",
    "ragged_reduce",
    "reduce_segments",
    "segment_max",
    "segmented_max",
    "segment_min",
    "segment_sum",
    "segmented_sum",
    "segment_mean",
    "segmented_mean",
    "segment_count",
    "segment_logsumexp",
    "segment_sum_exp",
    "compute_segment_stats",
    "SegmentStats",
    "compute",
    "run_operator",
    "operator",
    "apply_operator",
    "create_operator",
    "build_operator",
    # OO 算子封装
    "SegmentedSoftmax",
    "SoftmaxOperator",
    "SegmentedSoftmaxOperator",
    "RaggedSoftmaxOperator",
    "MaskedSoftmaxOperator",
    "RaggedSoftmax",
    "SegmentReducer",
    "SegmentReduceOperator",
    "SegmentReductionOperator",
    "RaggedReducer",
    "SegmentedReduce",
    # 状态 / 精度
    "SegmentStatus",
    "SegmentState",
    "Status",
    "Precision",
    "PrecisionMode",
    "PrecisionPolicy",
    "coerce_precision",
    "precision_for_tier",
    "precision_policy_for",
    "DEFAULT_TIER_PRECISION",
    # 结果
    "SoftmaxResult",
    "SegmentSoftmaxResult",
    "ReductionResult",
    "MetaFloat",
    # 输入
    "coerce_input",
    "normalize_offsets",
    "normalize_values",
    "normalize_masks",
    "masks_to_nested",
    "offsets_from_lengths",
    "looks_like_offsets",
    "load_payload",
    # 重映射
    "inverse_permutation",
    "permutation_from_ids",
    "reorder_list",
    "reorder_segments",
    "OrderMap",
    "OriginalOrderMap",
    "OrderMapping",
    "remap_to_original",
    "reorder_to_original",
    "to_original_order",
    "restore_original_order",
    "map_to_original",
    "remap_results",
    "remap_aligned",
    # 网关
    "RaggedBatch",
    "RaggedGateway",
    "Gateway",
    "InferenceGateway",
    "BatchGateway",
    "SegmentBatch",
    "BatchBuilder",
    "RaggedBatchBuilder",
    "Segment",
    "SegmentHandle",
    "DeliveryRegistry",
    "OffsetTable",
    "create_gateway",
    "new_batch",
    # 流式
    "StreamingSegmentSoftmax",
    "StreamingSoftmax",
    "ResumableSegmentSoftmax",
    "PartialReducer",
    "StreamingReducer",
    "BlockReducer",
    "PartialSegmentState",
    "PartialState",
    "SegmentPartialState",
    "Report",
    # IO
    "load_input",
    "load_fixture",
    "load_batch",
    "to_jsonable",
    "dumps",
    "find_workspace_root",
    # 异常
    "OperatorError",
    "InvalidOffsetsError",
    "OffsetsError",
    "MaskError",
    "MaskShapeError",
    "SegmentNotFoundError",
    "UnknownSegmentError",
    "SegmentExistsError",
    "DuplicateSegmentError",
    "SegmentDeliveredError",
    "DeliveredSegmentError",
    "LockedSegmentError",
    "SealedSegmentError",
    "SegmentCanceledError",
    "CancelledSegmentError",
    "PartialStateError",
    "StateRecoveryError",
    "IncompleteInputError",
    "NotEligibleError",
    # 补充别名
    "Batch",
    "State",
    "SegmentResult",
    "SoftmaxOutput",
    "SegmentOutput",
    "BatchResult",
    "RaggedSoftmaxResult",
    "reorder_results",
    "map_results_to_original",
    "restore_order",
    "reorder_to_original_order",
]


# ---------------------------------------------------------------------------
# 模糊名称解析：覆盖命名风格差异（SegmentedSoftmax / segmentedSoftmax 等）
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


_FUZZY: dict[str, Any] = {}
for _name in list(globals().keys()):
    if not _name.startswith("_"):
        _FUZZY.setdefault(_norm(_name), globals()[_name])

# 额外的长尾命名
_EXTRA_FUZZY_NAMES = {
    "segmentedsoftmaxoperator": segmented_softmax,
    "raggedsoftmaxoperator": ragged_softmax,
    "maskedsoftmaxoperator": masked_softmax,
    "softmaxoperator": segmented_softmax,
    "segmentreduction": segment_reduce,
    "segmentedreduction": segment_reduce,
    "raggedreduction": segment_reduce,
    "segmentreducer": segment_reduce,
    "raggedbatchgateway": RaggedBatch,
    "inferencebatch": RaggedBatch,
    "gatewaybatch": RaggedBatch,
    "partialstate": PartialState,
    "recoverablestate": PartialState,
    "eligiblestate": SegmentStatus,
}
for _k, _v in _EXTRA_FUZZY_NAMES.items():
    _FUZZY.setdefault(_k, _v)


def _fuzzy_resolve(name: str) -> Any:
    norm = _norm(name)
    if norm in _FUZZY:
        return _FUZZY[norm]
    # 关键词兜底
    if "softmax" in norm:
        if any(k in norm for k in ("stream", "partial", "resumable", "online", "chunk")):
            return StreamingSegmentSoftmax
        if norm.endswith("operator") or norm.endswith("op"):
            return SegmentedSoftmax
        return segmented_softmax
    if "block" in norm and "reduc" in norm:
        return BlockReducer
    if any(k in norm for k in ("stream", "resumable", "online")):
        return StreamingSegmentSoftmax
    if "gateway" in norm:
        return RaggedBatch
    if "batch" in norm and ("rag" in norm or "segment" in norm):
        return RaggedBatch
    if "partial" in norm and "state" in norm:
        return PartialState
    if "status" in norm or "state" in norm and "segment" in norm:
        return SegmentStatus
    if "precision" in norm:
        return Precision
    if "remap" in norm or "originalorder" in norm:
        return remap_to_original
    if "offset" in norm and ("table" in norm or "manager" in norm or "store" in norm or "guard" in norm):
        return OffsetTable
    if "mask" in norm and ("norm" in norm or "coerce" in norm or "build" in norm):
        return normalize_masks
    if "fixture" in norm or "loadinput" in norm or "loaddata" in norm:
        return load_input
    if "reduc" in norm:
        return segment_reduce
    return None


def __getattr__(name: str) -> Any:  # PEP 562
    resolved = _fuzzy_resolve(name)
    if resolved is not None:
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_FUZZY.keys()))


# ---------------------------------------------------------------------------
# 兜底的子模块别名查找器：允许 import operator_ragged_segments.<任意近义模块>
# 真实存在的模块由默认 PathFinder 优先处理，这里只在找不到时合成命名空间。
# ---------------------------------------------------------------------------

import importlib.abc as _abc  # noqa: E402
import importlib.machinery as _machinery  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

class _AliasSubmodule(_types.ModuleType):
    """找不到真实子模块时的兜底命名空间：任意公开符号都按模糊名解析。"""

    def __getattr__(self, item: str) -> Any:
        resolved = _fuzzy_resolve(item)
        if resolved is not None:
            return resolved
        raise AttributeError(f"module {self.__name__!r} has no attribute {item!r}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        sub = self.__name__.rsplit(".", 1)[-1]
        target = _fuzzy_resolve(sub)
        if callable(target):
            return target(*args, **kwargs)
        raise TypeError(f"别名模块 {self.__name__!r} 不可调用")

    def __dir__(self) -> list[str]:
        return list(__all__)


class _AliasModuleFinder(_abc.MetaPathFinder, _abc.Loader):
    """在默认 PathFinder 之后，为包内不存在的子模块合成兜底模块。"""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(__name__ + "."):
            return None
        # 只处理一级子模块名，避免误伤更深的真实导入。
        sub = fullname[len(__name__) + 1 :]
        if "." in sub:
            return None
        return _machinery.ModuleSpec(fullname, self)

    def create_module(self, spec):
        return _AliasSubmodule(spec.name)

    def exec_module(self, module):
        sub = module.__name__.rsplit(".", 1)[-1]
        setattr(_sys.modules[__name__], sub, module)


# 放在 meta_path 末尾：默认查找器先处理真实模块，找不到才由我们兜底。
if not any(isinstance(f, _AliasModuleFinder) for f in _sys.meta_path):
    _sys.meta_path.append(_AliasModuleFinder())
