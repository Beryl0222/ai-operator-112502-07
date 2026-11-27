"""测试包初始化。

在没有执行 ``pip install -e .`` 的环境里，也把仓库的 ``src`` 目录加入
``sys.path``，保证 ``python -m unittest discover -s tests`` 能直接导入
``operator_ragged_segments``。已经安装（site-packages / editable finder）
时这里不会产生重复路径。
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
