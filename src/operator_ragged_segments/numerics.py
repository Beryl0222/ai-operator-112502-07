"""数值辅助：在极端取值 / 下溢情况下也不抛异常、不产生 NaN。"""

from __future__ import annotations

import math
from typing import Any


def safe_exp(x: float) -> float:
    """平移后的指数。

    正常调用点都保证 ``x <= 0``（减去了段最大值），因此不会上溢出；
    极端负数下溢为 ``0.0``。这里再包一层异常 / 非有限保护，确保任何
    边界输入都不会把 NaN / 异常扩散到归约结果里。
    """
    try:
        y = math.exp(x)
    except OverflowError:
        return 0.0 if x < 0 else math.inf
    except ValueError:
        return 0.0
    if math.isnan(y):
        return 0.0
    return y


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
