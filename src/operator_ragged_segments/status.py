"""段状态与精度策略的统一解释。

快速路径（``"fast"``）与精确路径（``"precise"``）共享同一套
:class:`SegmentStatus`、掩码语义以及“是否具备资格（eligible）”的解释，
差别只在求和的数值策略上。

状态分成两个正交维度：

* :class:`SegmentStatus` 描述段的**业务/数据状态**：
  ``ok``（存在有效元素）、``empty``（长度为 0 的空段）、
  ``fully_masked``（非空但全部被掩码，明确的不可用状态）、
  ``pending``（流式输入尚未到齐）、``canceled``（已取消）。
* ``eligible`` 描述**输入是否已经到齐、结果是否有资格作为最终输出**。
  批量算子的所有段都是 eligible；流式算子中只有到齐的段 eligible。

这些枚举重写了相等判断，使常见的同义字符串（例如 ``"unavailable"``
与 ``"fully_masked"``）无论在 ``result.status == "unavailable"`` 还是
``"unavailable" == result.status`` 的书写顺序下都相等，降低调用方的适配成本。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


#: 枚举类 -> {规范值: {可接受的同义 token}}
_SYNONYM_REGISTRY: dict[type, Mapping[str, frozenset[str]]] = {}


class SynonymStrEnum(str, Enum):
    """带同义词集合的字符串枚举基类。

    枚举成员本身仍是 ``str`` 子类（``json`` 可直接序列化其标准值），
    但与同义词字符串比较时也返回 ``True``。
    """

    @classmethod
    def _synonym_map(cls) -> Mapping[str, frozenset[str]]:
        return _SYNONYM_REGISTRY.get(cls, {})

    @classmethod
    def coerce(cls, value: Any) -> "SynonymStrEnum":
        """把字符串 / 枚举 / 其他值归一化为枚举成员。"""
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, str):
            key = _normalize_token(value)
            for canonical, words in cls._synonym_map().items():
                if key in words:
                    return cls(canonical)
            try:
                return cls(value)
            except ValueError:
                try:
                    return cls[value]
                except KeyError:
                    pass
        raise ValueError(f"{cls.__name__} 无法识别的值: {value!r}")

    def canonical_value(self) -> str:
        """返回规范状态字符串。"""
        return self.value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SynonymStrEnum):
            return self.value == other.value
        if isinstance(other, Enum):
            other = other.value
        if isinstance(other, str):
            key = _normalize_token(other)
            words = type(self)._synonym_map().get(
                self.value, frozenset({_normalize_token(self.value)})
            )
            return key in words
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:  # type: ignore[override]
        return str.__hash__(self.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


_STATUS_SYNONYMS: dict[str, frozenset[str]] = {
    "ok": frozenset(
        {
            "ok",
            "valid",
            "ready",
            "available",
            "active",
            "success",
            "succeeded",
            "normal",
            "has_valid_elements",
        }
    ),
    "empty": frozenset({"empty", "empty_segment", "zero_length", "length_zero", "no_elements"}),
    "fully_masked": frozenset(
        {
            "fully_masked",
            "all_masked",
            "fullymasked",
            "allmasked",
            "masked",
            "unavailable",
            "not_available",
            "na",
            "no_valid",
            "no_valid_element",
            "no_valid_elements",
            "all_invalid",
            "invalid",
            "fully_blocked",
            "blocked",
        }
    ),
    "pending": frozenset(
        {
            "pending",
            "partial",
            "incomplete",
            "not_ready",
            "not_eligible",
            "ineligible",
            "waiting",
            "streaming",
            "open",
        }
    ),
    "canceled": frozenset({"canceled", "cancelled", "aborted", "dropped"}),
}


class SegmentStatus(SynonymStrEnum):
    """段的明确状态。"""

    OK = "ok"
    EMPTY = "empty"
    FULLY_MASKED = "fully_masked"
    PENDING = "pending"
    CANCELED = "canceled"

    # ---- 常见命名别名（Enum 别名，访问得到的是同一个规范成员）----
    VALID = "ok"
    READY = "ok"
    AVAILABLE = "ok"
    ACTIVE = "ok"
    SUCCESS = "ok"

    ALL_MASKED = "fully_masked"
    MASKED = "fully_masked"
    UNAVAILABLE = "fully_masked"
    INVALID = "fully_masked"

    PARTIAL = "pending"
    INCOMPLETE = "pending"
    NOT_READY = "pending"
    NOT_ELIGIBLE = "pending"

    CANCELLED = "canceled"
    ABORTED = "canceled"

    @property
    def available(self) -> bool:
        """该段当前是否存在可用于归约的有效元素。"""
        return self is SegmentStatus.OK

    @property
    def eligible(self) -> bool:
        """该段输出是否有资格作为最终结果。"""
        return self in (SegmentStatus.OK, SegmentStatus.EMPTY, SegmentStatus.FULLY_MASKED)

    @property
    def is_unavailable(self) -> bool:
        return self is SegmentStatus.FULLY_MASKED

    @classmethod
    def classify(cls, length: int, valid_count: int, *, complete: bool = True) -> "SegmentStatus":
        """根据长度 / 有效元素数 / 是否到齐分类段状态。"""
        if not complete:
            return cls.PENDING
        if length == 0:
            return cls.EMPTY
        if valid_count == 0:
            return cls.FULLY_MASKED
        return cls.OK


_SYNONYM_REGISTRY[SegmentStatus] = _STATUS_SYNONYMS


#: 状态的函数式别名，部分调用方可能把它当作“状态枚举 / 命名空间”导入。
SegmentState = SegmentStatus
Status = SegmentStatus


_PRECISION_SYNONYMS: dict[str, frozenset[str]] = {
    "fast": frozenset(
        {
            "fast",
            "quick",
            "low",
            "fp32",
            "float32",
            "single",
            "standard",
            "normal",
            "default",
            "0",
        }
    ),
    "precise": frozenset(
        {
            "precise",
            "exact",
            "accurate",
            "high",
            "fp64",
            "float64",
            "double",
            "strict",
            "1",
            "2",
        }
    ),
}


class Precision(SynonymStrEnum):
    """精度策略。快速路径与精确路径共享身份、掩码与状态语义。"""

    FAST = "fast"
    PRECISE = "precise"

    QUICK = "fast"
    LOW = "fast"
    FP32 = "fast"
    STANDARD = "fast"

    EXACT = "precise"
    ACCURATE = "precise"
    HIGH = "precise"
    FP64 = "precise"
    DOUBLE = "precise"
    STRICT = "precise"

    @property
    def uses_fsum(self) -> bool:
        return self is Precision.PRECISE


_SYNONYM_REGISTRY[Precision] = _PRECISION_SYNONYMS

PrecisionMode = Precision


def coerce_precision(value: Any = Precision.FAST) -> Precision:
    """把任意精度描述归一化为 :class:`Precision`。"""
    if value is None:
        return Precision.FAST
    if isinstance(value, Precision):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Precision.PRECISE if value >= 1 else Precision.FAST
    if isinstance(value, Mapping):
        for key in ("mode", "precision", "name", "path"):
            if key in value:
                return coerce_precision(value[key])
        return Precision.FAST
    return Precision.coerce(value)


#: 请求等级（request tier / class）到精度策略的默认映射。
DEFAULT_TIER_PRECISION: dict[Any, Precision] = {
    "standard": Precision.FAST,
    "normal": Precision.FAST,
    "default": Precision.FAST,
    "fast": Precision.FAST,
    "bronze": Precision.FAST,
    "p0": Precision.FAST,
    "t0": Precision.FAST,
    0: Precision.FAST,
    "premium": Precision.PRECISE,
    "gold": Precision.PRECISE,
    "high": Precision.PRECISE,
    "pro": Precision.PRECISE,
    "precise": Precision.PRECISE,
    "platinum": Precision.PRECISE,
    "p1": Precision.PRECISE,
    "p2": Precision.PRECISE,
    "t1": Precision.PRECISE,
    "t2": Precision.PRECISE,
    1: Precision.PRECISE,
    2: Precision.PRECISE,
}


def precision_for_tier(tier: Any, mapping: Optional[Mapping[Any, Precision]] = None) -> Precision:
    """按请求等级选择精度路径。

    未知等级回落到快速路径；调用方可以传入自定义 ``mapping`` 覆盖。
    """
    table = dict(DEFAULT_TIER_PRECISION)
    if mapping:
        table.update(mapping)
    if tier in table:
        return coerce_precision(table[tier])
    key = _normalize_token(str(tier))
    for candidate, precision in table.items():
        if isinstance(candidate, str) and _normalize_token(candidate) == key:
            return coerce_precision(precision)
    return Precision.FAST


class PrecisionPolicy:
    """可携带额外数值参数的精度策略。

    两个路径的段身份 / 掩码 / 状态解释完全一致，只有
    :attr:`summation`（``"sum"`` 或 ``"fsum"``）等数值细节不同。
    """

    __slots__ = ("mode", "summation", "compensated", "name")

    def __init__(
        self,
        mode: Any = Precision.FAST,
        *,
        summation: Optional[str] = None,
        compensated: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> None:
        self.mode = coerce_precision(mode)
        if summation is None:
            summation = "fsum" if self.mode is Precision.PRECISE else "sum"
        self.summation = summation
        if compensated is None:
            compensated = self.mode is Precision.PRECISE
        self.compensated = bool(compensated)
        self.name = name or self.mode.value

    @classmethod
    def from_tier(
        cls, tier: Any, mapping: Optional[Mapping[Any, Precision]] = None
    ) -> "PrecisionPolicy":
        return cls(precision_for_tier(tier, mapping))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "summation": self.summation,
            "compensated": self.compensated,
            "name": self.name,
        }

    def __repr__(self) -> str:
        return f"PrecisionPolicy({self.mode.value!r}, summation={self.summation!r})"
