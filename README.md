# Ragged batch 的分段归约与掩码 Softmax

纯 Python 标准库实现的、面向 **offsets/values** 表示的分段归约与掩码
Softmax 算子。不需要把不同长度的请求补齐成矩阵，长尾请求不会再放大
padding 显存；并严格区分三种过去被混淆的段：

| 情况 | 判定 | `SegmentStatus` | 输出 |
| --- | --- | --- | --- |
| 空段（offset 跨度为 0） | `length == 0` | `empty` | 空概率列表，`max/sum_exp = None` |
| 非空但全部被掩码 | `valid_count == 0` | `fully_masked`（同义 `unavailable/all_masked`） | 全 `0.0` + 全 `False` 有效掩码，**无 NaN** |
| 有效值全为 0（或相等） | `status == ok` | `ok` | 均匀 Softmax |

掩码约定：**`True` 表示有效、允许参与归约**（可用 `invert_mask=True`
或 `mask_true_is_valid=False` 切换极性）。掩码先于 max / exp-sum /
归一化生效；是否被掩码由独立的 `valid_mask` 表达，因此有效值下溢成
`0.0` 不会被误判成屏蔽，屏蔽项也不会因为取值为负而“复活”。

## 环境与验证

项目运行只依赖 Python 标准库（`requires-python >= 3.11`）。

```bash
python -m pip install -e .
python -m pip install -r requirements-test.txt   # 锁定的打包工具
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
python -m operator_ragged_segments              # 在 data/input.json 上演示
```

即使不执行 editable 安装，`tests/__init__.py` 也会把 `src/` 加入
`sys.path`，因此 `python -m unittest discover -s tests` 可直接运行。

## 快速上手

```python
import json
from operator_ragged_segments import segmented_softmax

payload = json.load(open("data/input.json"))
result = segmented_softmax(payload)
# 也支持 segmented_softmax(values, offsets, masks)
# 以及 segmented_softmax(offsets, values, masks)（自动判别位置顺序）

result                  # 本身是与 values 对齐的扁平概率 list
result.segment_ids      # ['a', 'empty', 'c']
result.statuses         # [SegmentStatus.OK, EMPTY, OK]
result["a"].max_value   # 1.0（只来自未屏蔽的 1.0 与 -2.0）
result["c"].probs       # [0.2689..., 0.7310..., 0.0]（最后一位被掩码）
result.to_dict()        # JSON 安全，无 NaN / Inf
```

支持的输入形态还包括：嵌套逐段列表
`segmented_softmax([[1, 0, -2], [], [4, 5, 6]], masks=[...])`、
逐位置 `segment_ids` 标签、fixture 路径 / dict、以及
`RaggedBatch` 等批对象。

## 分段归约

```python
from operator_ragged_segments import segment_reduce, segment_max

segment_max(values, offsets, masks)          # op='max'
segment_reduce(values, offsets, op='sum')    # max/min/sum/mean/prod/
                                             # any/all/first/last/count
segment_logsumexp(...)
segment_sum_exp(...)
```

返回的 `ReductionResult` 是与段对齐的 list；空段 / 全屏蔽段在
`max/min/mean/...` 下为 `None`（`sum/prod/any/all/count` 使用代数单位元），
并通过 `.statuses` 给出明确原因，可用 `.as_list(fill=...)` 替换填充值。

## 原顺序重映射

网关按到达顺序保存 `original_order`，打包计算可能重排了段：

```python
from operator_ragged_segments import OrderMap, remap_to_original

om = OrderMap(original_ids=["a", "b", "c"], batch_ids=["c", "a", "b"])
restored = om.to_original(result)     # SoftmaxResult 会重建扁平 offsets
remap_to_original({...}, original_order=["a", "b"])
remap_aligned(items, original_order)  # 缺失 / 已取消段用占位符对齐
```

## 网关：追加 / 取消 / 交付保护

```python
from operator_ragged_segments import RaggedBatch  # 别名 Gateway/RaggedGateway

gw = RaggedBatch()
gw.append("a", [1.0, 0.0, -2.0], [True, False, True])  # 返回稳定 segment_id
gw.append("b", [10.0, 20.0])
gw.cancel("b")                    # 仅移出当前活跃批次
out = gw.deliver("a")             # 计算并冻结深拷贝快照到 gw.deliveries
gw.append("c", [4, 5, 6])         # 新 offsets 不会覆盖 'a'
gw.deliveries["a"]                # 永远是交付时的结果
gw.delivered_offsets["a"]         # 稳定的 [start, end)
gw.remap_to_original()            # 映射回 gw.original_order
```

保证：

* 扁平值日志只追加，每段的稳定 `[start, end)` 永不移动；
* 已交付段用相同 id 再次 `append` 会抛 `SegmentDeliveredError`，
  `update_offsets(...)` 若试图改动已交付边界同样抛错；
* 交付快照是深拷贝，后续追加 / 取消 / 重建 offsets 都不影响它；
* `cancel(..., force=True)` 可把已交付段移出活跃批次，但交付快照仍保留。
* 另有独立的 `OffsetTable` 提供带版本与 `protect(...)` 的 offsets 表。

## 流式 / 分块：未到齐输入的可恢复部分状态

```python
from operator_ragged_segments import StreamingSegmentSoftmax

st = StreamingSegmentSoftmax(offsets=[0, 3, 3, 6],
                            segment_ids=["a", "empty", "c"],
                            block_size=2)
report = st.update(chunk_values, chunk_mask)
report["eligible_segment_ids"]      # 已到齐、可出最终结果的段
report["pending_segment_ids"]       # 尚未到齐的段
report["ineligible_positions"]      # 已接收但所属段未最终确定的位置
report["not_yet_arrived_positions"] # 按声明 offsets 尚未到达的位置

blob = st.snapshot()                # JSON 安全的可恢复状态
st2 = StreamingSegmentSoftmax.from_snapshot(blob)
st2.update(rest_values, rest_mask)
final = st2.finish()                # 与一次性 segmented_softmax 一致
st.result(only_eligible=True)       # 只取出已具备资格的紧凑结果
st.block_partials(block_size=4)     # 跨段 / 跨前沿的块 partial
st.open_block_state()               # 尚未封满、可恢复的尾块
```

在线状态使用 max 平移的 log-sum-exp 合并原语
（`PartialSegmentState.merge`），块 partial 也可以跨块合并。

## 精度策略与请求等级

```python
from operator_ragged_segments import Precision, precision_for_tier

segmented_softmax(..., precision="fast")     # 顺序求和
segmented_softmax(..., precision="precise")  # math.fsum 精确求和
precision_for_tier("gold")      # -> Precision.PRECISE
PrecisionPolicy.from_tier("premium")
```

`fast` 与 `precise` 走同一套有效元素提取、段身份、`SegmentStatus`
分类与 eligibility 解释，只改变求和策略；`"exact"/"accurate"/"high"/
"fp64"` 等都是 precise 的同义词。

## API 兼容面

包顶层导出全部常用符号，并对命名风格差异做了兼容：

* 函数 / 类同时提供 snake_case、CamelCase 与常见别名
  （`segment_softmax`、`ragged_softmax`、`masked_softmax`、
  `SegmentedSoftmax` OO 封装等）；
* 返回对象同时支持 list、属性与字符串键访问，并提供 `to_dict()`；
* 未显式提供的近义子模块（例如
  `operator_ragged_segments.gateway`、`.precision`、`.state`、
  `.reduction`）由包内的兜底模块解析到同一套实现。

## 资料

`data/input.json` 是可重复使用的本地合成输入，`data/schema.json`
记录版本与边界说明。
