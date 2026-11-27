"""资料 / fixture 读取与 JSON 安全序列化辅助。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional, Union


def find_workspace_root(start: Optional[Union[str, Path]] = None) -> Path:
    """从给定位置向上查找包含 ``data/input.json`` 的工程根目录。"""
    here = Path(start or os.getcwd()).resolve()
    candidates = [here, *here.parents]
    # 也检查包所在仓库的常见层级（src/operator_ragged_segments -> 上两级）
    pkg_root = Path(__file__).resolve().parents[2]
    candidates.append(pkg_root)
    for base in candidates:
        if (base / "data" / "input.json").is_file():
            return base
    return here


def load_input(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """读取合成输入 fixture。

    未指定路径时依次尝试：当前目录 ``data/input.json``、
    工程根 ``data/input.json``、包上级仓库的 ``data/input.json``。
    """
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    root = find_workspace_root()
    candidate = root / "data" / "input.json"
    if not candidate.is_file():
        # 最后兜底：相对当前工作目录
        candidate = Path("data/input.json")
    return json.loads(candidate.read_text(encoding="utf-8"))


#: 语义化别名
load_fixture = load_input
load_batch = load_input


def to_jsonable(obj: Any) -> Any:
    """把结果 / 状态对象转成 JSON 安全的纯 Python 结构。

    规则：枚举 -> 规范字符串；结果对象 -> ``to_dict()``；
    ``NaN/Inf``（理论上算子不会产生）-> ``None``；其余递归处理。
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "value"):
        return to_jsonable(obj.value)
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    """:func:`json.dumps` 的安全版本（默认 ``ensure_ascii=False``）。"""
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(to_jsonable(obj), **kwargs)
