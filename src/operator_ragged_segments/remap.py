"""段顺序 / 原顺序重映射工具。

网关按请求到达顺序保存 ``original_order``；为了拼 ragged batch，
段可能按长度等规则重新打包。本模块负责把批量顺序的结果按段身份
（segment id）或排列索引映射回原始顺序。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .errors import SegmentNotFoundError


def inverse_permutation(perm: Sequence[int]) -> list[int]:
    """返回逆排列。

    约定 ``perm[i]`` 表示“新位置 i 取自旧位置 perm[i]”，
    则逆排列 ``inv[j]`` 给出旧位置 j 在新序列中的位置。
    """
    n = len(perm)
    inv = [0] * n
    for new_pos, old_pos in enumerate(perm):
        if not isinstance(old_pos, int) or not (0 <= old_pos < n):
            raise ValueError(f"非法排列项: {old_pos!r}（长度 {n}）")
        inv[old_pos] = new_pos
    return inv


def identity_permutation(n: int) -> list[int]:
    return list(range(n))


def permutation_from_ids(order_ids: Sequence[Any], target_ids: Sequence[Any]) -> list[int]:
    """构造位置排列 ``perm``，使 ``target[i] == order[perm[i]]``。

    两个 id 集合必须一致（允许顺序不同）。
    """
    index = {_id_key(x): i for i, x in enumerate(order_ids)}
    perm: list[int] = []
    for tid in target_ids:
        key = _id_key(tid)
        if key not in index:
            raise SegmentNotFoundError(f"目标顺序中的段 {tid!r} 不在已知段集合中")
        perm.append(index[key])
    return perm


def _id_key(x: Any) -> Any:
    """id 匹配键：尽量让 1 与 "1" 不混淆，但字符串形式可被二次匹配。"""
    return x


def reorder_list(items: Sequence[Any], perm: Sequence[int]) -> list[Any]:
    """按 ``new[i] = old[perm[i]]`` 重排普通列表。"""
    return [items[perm[i]] for i in range(len(perm))]


def reorder_segments(
    flat_values: Sequence[float],
    offsets: Sequence[int],
    segment_ids: Sequence[Any],
    target_order: Sequence[Any],
) -> tuple[list[float], list[int], list[Any]]:
    """把扁平 values 按段整体重排到 ``target_order``。

    返回 ``(new_flat_values, new_offsets, new_segment_ids)``。
    段内元素顺序保持不变。
    """
    if len(segment_ids) != len(offsets) - 1:
        raise ValueError("segment_ids 数量必须等于 offsets 段数")
    pos_by_id: dict[Any, int] = {}
    for i, sid in enumerate(segment_ids):
        pos_by_id[sid] = i
        pos_by_id.setdefault(str(sid), i)

    new_flat: list[float] = []
    new_offsets = [0]
    new_ids: list[Any] = []
    for wanted in target_order:
        idx = pos_by_id.get(wanted)
        if idx is None:
            idx = pos_by_id.get(str(wanted))
        if idx is None:
            raise SegmentNotFoundError(f"目标顺序中的段 {wanted!r} 不存在")
        s, e = offsets[idx], offsets[idx + 1]
        new_flat.extend(flat_values[s:e])
        new_offsets.append(len(new_flat))
        new_ids.append(segment_ids[idx])
    return new_flat, new_offsets, new_ids


class OrderMap:
    """记录“原始顺序 <-> 批量顺序”的双向映射。

    典型用法::

        om = OrderMap(original_ids=["a", "b", "c"], batch_ids=["b", "a", "c"])
        om.to_original(batch_result)   # 重排为 a, b, c
        om.to_batch(original_items)
    """

    def __init__(
        self,
        original_ids: Sequence[Any],
        batch_ids: Optional[Sequence[Any]] = None,
        *,
        permutation: Optional[Sequence[int]] = None,
    ) -> None:
        self.original_ids = list(original_ids)
        if batch_ids is None:
            if permutation is not None:
                batch_ids = [self.original_ids[permutation[i]] for i in range(len(permutation))]
            else:
                batch_ids = list(original_ids)
        self.batch_ids = list(batch_ids)
        if permutation is None:
            # batch[i] = original[perm[i]]
            permutation = permutation_from_ids(self.original_ids, self.batch_ids)
        self.batch_to_original_perm = list(permutation)
        self.original_to_batch_perm = inverse_permutation(self.batch_to_original_perm)

    # 常用别名
    @property
    def permutation(self) -> list[int]:
        return list(self.batch_to_original_perm)

    @property
    def inverse_permutation(self) -> list[int]:
        return list(self.original_to_batch_perm)

    def original_position(self, batch_position: int) -> int:
        return self.batch_to_original_perm[batch_position]

    def batch_position(self, original_position: int) -> int:
        return self.original_to_batch_perm[original_position]

    def to_original(self, batch_result: Any) -> Any:
        """把批量顺序的结果重映射为原始顺序。

        支持 :class:`SoftmaxResult`、``{id: segment}`` 映射、
        与 batch_ids 对齐的列表。
        """
        return remap_to_original(
            batch_result,
            segment_ids=self.batch_ids,
            original_order=self.original_ids,
        )

    def to_batch(self, original_items: Sequence[Any]) -> list[Any]:
        """原始顺序列表 -> 批量顺序（``batch[i] = original[perm[i]]``）。"""
        return reorder_list(original_items, self.batch_to_original_perm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_ids": list(self.original_ids),
            "batch_ids": list(self.batch_ids),
            "batch_to_original_perm": list(self.batch_to_original_perm),
            "original_to_batch_perm": list(self.original_to_batch_perm),
        }


#: 语义化别名
OriginalOrderMap = OrderMap
OrderMapping = OrderMap


def remap_to_original(
    result: Any = None,
    *,
    segment_ids: Optional[Sequence[Any]] = None,
    original_order: Optional[Sequence[Any]] = None,
    order: Optional[Sequence[Any]] = None,
    offsets: Optional[Sequence[int]] = None,
    values: Optional[Sequence[float]] = None,
    permutation: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Any:
    """把结果重新映射回网关保存的原顺序。

    支持的输入：

    1. :class:`~operator_ragged_segments.results.SoftmaxResult`
       （自动读取其 ``segment_ids/offsets``）；
    2. ``{segment_id: segment_result}`` 字典；
    3. 与 ``segment_ids`` 对齐的段结果列表；
    4. 扁平 ``values`` + ``offsets``（需同时提供 ``segment_ids``）。

    返回与输入同形态但按 ``original_order`` 排列的结果；
    对 SoftmaxResult 会重建扁平概率与 offsets。
    """
    if original_order is None:
        original_order = order
    if original_order is None:
        original_order = kwargs.get("original_ids")

    # ---- 扁平 values + offsets 形态 ----
    if values is not None and offsets is not None and segment_ids is not None:
        if original_order is None:
            raise ValueError("需要提供 original_order 才能重映射")
        flat, new_offsets, new_ids = reorder_segments(values, offsets, segment_ids, original_order)
        return {
            "values": flat,
            "probabilities": flat,
            "offsets": new_offsets,
            "segment_ids": new_ids,
        }

    if result is None:
        raise ValueError("remap_to_original 缺少 result 或 values/offsets")

    # ---- SoftmaxResult 形态 ----
    from .results import SoftmaxResult, SegmentSoftmaxResult

    if isinstance(result, SoftmaxResult):
        if original_order is None:
            original_order = result.original_order
        if original_order is None:
            raise ValueError("结果未携带 original_order，请显式传入")
        strict = bool(kwargs.get("strict", False))
        batch_ids = result.segment_ids
        present = {_id_key(x) for x in batch_ids}
        wanted_order = list(original_order)
        missing = [w for w in wanted_order if _id_key(w) not in present and str(w) not in {str(x) for x in batch_ids}]
        if missing and strict:
            raise SegmentNotFoundError(f"原顺序中的段 {missing!r} 不在结果中")
        # 默认跳过结果中不存在的 id（例如已取消），并把结果里多出的 id 追加在末尾。
        ordered = [w for w in wanted_order if w in present or any(str(w) == str(x) for x in batch_ids)]
        for x in batch_ids:
            if x not in ordered and str(x) not in {str(w) for w in ordered}:
                ordered.append(x)
        flat, new_offsets, new_ids = reorder_segments(
            result.probabilities, result.offsets, batch_ids, ordered
        )
        new_mask, _, _ = reorder_segments(
            [1.0 if v else 0.0 for v in result.valid_mask],
            result.offsets, batch_ids, ordered,
        )
        # 重新组织 segment 结果顺序
        seg_by_id = {seg.segment_id: seg for seg in result.segments}
        new_segments = []
        for new_index, wanted in enumerate(ordered):
            seg = seg_by_id.get(wanted) or seg_by_id.get(str(wanted))
            if seg is None:
                raise SegmentNotFoundError(f"原顺序中的段 {wanted!r} 不在结果中")
            seg.index = new_index
            new_segments.append(seg)
        remapped = SoftmaxResult(
            flat,
            offsets=new_offsets,
            segments=new_segments,
            valid_mask=[bool(x) for x in new_mask],
            segment_ids=list(ordered),
            precision=result.precision,
            original_order=list(original_order),
        )
        return remapped

    # ---- 字典 id -> value 形态 ----
    if isinstance(result, Mapping):
        if original_order is None:
            raise ValueError("字典形态需要提供 original_order")
        if kwargs.get("strict", False):
            return {key: result[key] for key in original_order}
        return {key: result[key] for key in original_order if key in result}

    # ---- 列表 + segment_ids 形态 ----
    if isinstance(result, (list, tuple)):
        if segment_ids is None:
            if permutation is not None:
                inv = inverse_permutation(permutation)
                return reorder_list(list(result), inv)
            raise ValueError("列表形态需要 segment_ids 或 permutation")
        if original_order is None:
            raise ValueError("需要提供 original_order")
        by_id = {sid: item for sid, item in zip(segment_ids, result)}
        if kwargs.get("strict", False):
            return [by_id[sid] for sid in original_order]
        return [by_id[sid] for sid in original_order if sid in by_id]

    raise TypeError(f"无法对类型 {type(result).__name__} 进行原顺序重映射")


def remap_aligned(
    items: Any,
    original_order: Sequence[Any],
    segment_ids: Optional[Sequence[Any]] = None,
    *,
    placeholder: Any = None,
) -> list[Any]:
    """返回与 ``original_order`` 一一对齐的列表；缺失 id 用 placeholder 填充。

    ``items`` 可以是 ``{id: value}`` 映射，或与 ``segment_ids`` 对齐的列表。
    这让调用方可以按网关保存的原始下标安全回填（包括已取消段的空位）。
    """
    if isinstance(items, Mapping):
        mapping = items
    else:
        if segment_ids is None:
            raise ValueError("列表形态需要 segment_ids")
        mapping = {sid: item for sid, item in zip(segment_ids, items)}
    aligned: list[Any] = []
    for sid in original_order:
        if sid in mapping:
            aligned.append(mapping[sid])
        else:
            hit = next((v for k, v in mapping.items() if str(k) == str(sid)), placeholder)
            aligned.append(hit)
    return aligned


# 常见别名
reorder_to_original = remap_to_original
to_original_order = remap_to_original
restore_original_order = remap_to_original
map_to_original = remap_to_original
remap_results = remap_to_original
