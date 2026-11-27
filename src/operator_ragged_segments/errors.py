"""异常类型。

所有算子异常都继承 :class:`OperatorError`，同时常见参数类异常也继承
Python 内建异常（``ValueError`` / ``KeyError`` / ``RuntimeError``），
方便调用方既可以按算子异常捕获，也可以按标准异常捕获。
"""

from __future__ import annotations


class OperatorError(Exception):
    """所有分段算子相关错误的基类。"""


class InvalidOffsetsError(OperatorError, ValueError):
    """offsets 不合法（长度、单调性、与 values 长度不一致等）。"""


OffsetsError = InvalidOffsetsError


class MaskError(OperatorError, ValueError):
    """掩码形状或取值不合法。"""


MaskShapeError = MaskError


class SegmentNotFoundError(OperatorError, KeyError):
    """引用了不存在的段身份（segment id / 下标）。"""


UnknownSegmentError = SegmentNotFoundError


class SegmentExistsError(OperatorError, ValueError):
    """尝试用已经存在的段身份追加段落。"""


DuplicateSegmentError = SegmentExistsError


class SegmentDeliveredError(OperatorError, RuntimeError):
    """段已经交付下游采样，其 offsets / 结果受保护，不能被覆盖或修改。"""


DeliveredSegmentError = SegmentDeliveredError
LockedSegmentError = SegmentDeliveredError
SealedSegmentError = SegmentDeliveredError


class SegmentCanceledError(OperatorError, RuntimeError):
    """段已被取消，不能再参与新的归约或交付。"""


CancelledSegmentError = SegmentCanceledError


class PartialStateError(OperatorError, ValueError):
    """可恢复部分状态损坏、版本不识别或无法恢复。"""


StateRecoveryError = PartialStateError


class IncompleteInputError(OperatorError, RuntimeError):
    """在仍有输入未到齐时尝试做最终求值。

    正常的流式路径不会抛出该异常，而是把段标记为 ``pending`` /
    ``eligible=False``；只有调用方显式要求“必须立即给出最终结果”时才可能抛出。
    """


NotEligibleError = IncompleteInputError
