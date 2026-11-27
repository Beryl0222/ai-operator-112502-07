"""面向对象风格的算子封装。

不同资料 / 测试对“算子”的使用习惯不同：有的直接调用函数，有的先
实例化再调用 ``forward/run/compute``。这里的类同时兼容：

* ``SegmentedSoftmax()(values, offsets)``
* ``SegmentedSoftmax(precision="precise")(values, offsets)``
* ``SegmentedSoftmax().forward(values, offsets)``
* ``SegmentedSoftmax(values, offsets)`` —— 构造参数里直接带数据时，
  “做我所想”，直接返回 :class:`SoftmaxResult`（结果对象本身也有
  ``forward/run/result`` 等返回自身的方法），因此
  ``SegmentedSoftmax(payload).forward()`` 也成立。

无论哪种调用风格，底层都是同一套函数实现，共享段身份、掩码与状态语义。
"""

from __future__ import annotations

from typing import Any

from .core import segment_reduce
from .softmax import segmented_softmax
from .status import coerce_precision

_DATA_KEYS = ("values", "offsets", "data", "payload", "source", "masks", "mask")


class _DataOperator:
    function = staticmethod(lambda *a, **k: (_ for _ in ()).throw(NotImplementedError))
    default_config: dict[str, Any] = {}

    def __new__(cls, *args: Any, **kwargs: Any):
        has_data = bool(args) or any(k in kwargs for k in _DATA_KEYS)
        if has_data:
            return cls.function(*args, **kwargs)  # type: ignore[misc]
        return super().__new__(cls)

    def __init__(self, **config: Any) -> None:
        self.config = {**self.default_config, **config}

    def _merged(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.config)
        merged.update(call_kwargs)
        return merged

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **self._merged(kwargs))  # type: ignore[misc]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)


class SegmentedSoftmax(_DataOperator):
    """分段掩码 Softmax 算子（OO 封装）。"""

    function = staticmethod(segmented_softmax)

    def __init__(self, precision: Any = "fast", **config: Any) -> None:
        super().__init__(precision=coerce_precision(precision), **config)


# 常见类名别名
SoftmaxOperator = SegmentedSoftmax
SegmentedSoftmaxOperator = SegmentedSoftmax
RaggedSoftmaxOperator = SegmentedSoftmax
MaskedSoftmaxOperator = SegmentedSoftmax
RaggedSoftmax = SegmentedSoftmax


class SegmentReducer(_DataOperator):
    """分段归约算子（OO 封装），默认 ``op="sum"``。"""

    function = staticmethod(segment_reduce)
    default_config = {"op": "sum"}

    def __init__(self, op: str = "sum", precision: Any = "fast", **config: Any) -> None:
        super().__init__(op=op, precision=coerce_precision(precision), **config)


SegmentReduceOperator = SegmentReducer
SegmentReductionOperator = SegmentReducer
RaggedReducer = SegmentReducer
SegmentedReduce = SegmentReducer
