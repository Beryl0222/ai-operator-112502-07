"""输入归一化与校验。

内部统一使用 ``offsets / values / flat mask / segment_ids`` 表示；
对外的宽松入口可以接受：

* 已解析的 fixture ``dict``（键名兼容 ``offsets/indptr/cu_seqlens`` 等）；
* ``(values, offsets, masks)`` 或 ``(offsets, values, masks)`` 两种位置顺序
  （根据 offsets 的结构特征自动判别）；
* 嵌套的逐段掩码（与 ``data/input.json`` 一致）或与 values 对齐的扁平掩码；
* :class:`~operator_ragged_segments.batch.RaggedBatch` 等带
  ``offsets/values/masks`` 属性的对象。
"""

from __future__ import annotations

import json
import os
from collections.abc import Sized
from pathlib import Path
from typing import Any, Optional, Sequence

from .errors import InvalidOffsetsError, MaskShapeError


# ---------------------------------------------------------------------------
# 字典 / 对象取值
# ---------------------------------------------------------------------------

_OFFSET_KEYS = ("offsets", "segment_offsets", "boundaries", "indptr", "cu_seqlens", "offset")
_VALUE_KEYS = ("values", "data", "x", "scores", "logits", "input", "inputs")
_MASK_KEYS = ("masks", "mask", "valid_mask", "valid_masks", "attention_mask", "keep_mask")
_ID_KEYS = ("segment_ids", "segment_id", "ids", "id", "names")
_ORDER_KEYS = ("original_order", "order", "original_ids", "gateway_order")


def _pick(container: Any, keys: Sequence[str]) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        for key in keys:
            if key in container and container[key] is not None:
                return container[key]
        return None
    for key in keys:
        if hasattr(container, key):
            value = getattr(container, key)
            if value is not None:
                return value
    return None


def load_payload(source: Any = None) -> Any:
    """把路径 / JSON 文本 / dict / 文件对象统一成 dict（或原样返回对象）。"""
    if source is None:
        return None
    if isinstance(source, (str, os.PathLike)):
        text = Path(source).read_text(encoding="utf-8")
        return json.loads(text)
    if hasattr(source, "read"):
        return json.loads(source.read())
    return source


# ---------------------------------------------------------------------------
# offsets / values 判别
# ---------------------------------------------------------------------------

def _is_int_number(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _len_or_zero(x: Any) -> int:
    try:
        return len(x)
    except TypeError:
        return 0


def looks_like_offsets(seq: Any, expected_values_len: Optional[int] = None) -> bool:
    """启发式判断一个序列是否像 ragged offsets。"""
    if not isinstance(seq, (list, tuple)) or len(seq) < 1:
        return False
    for x in seq:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return False
        if isinstance(x, float) and not x.is_integer():
            return False
    if any(seq[i] > seq[i + 1] for i in range(len(seq) - 1)):
        return False
    if seq[0] < 0:
        return False
    if expected_values_len is not None and len(seq) >= 2:
        # 标准 offsets 的最后一项等于扁平 values 长度。
        if int(seq[-1]) != expected_values_len:
            return False
    return True


def _offset_score(seq: Any, other_len: int, candidate_mask: Any = None) -> int:
    """给“该序列作为 offsets、另一个序列长度为 other_len”的可信度打分。"""
    if not looks_like_offsets(seq):
        return -100
    score = 0
    if int(seq[0]) == 0:
        score += 4
    if int(seq[-1]) == other_len:
        score += 5
    else:
        score -= 5
    # 真正的 offsets 通常是 Python int；全 float（尤其 1.0/2.0）更像 values。
    if all(_is_int_number(x) for x in seq):
        score += 2
    if len(seq) >= 2:
        score += 1
    # 掩码形态一致性
    if candidate_mask is not None:
        n_seg = len(seq) - 1
        if isinstance(candidate_mask, (list, tuple)):
            if len(candidate_mask) == other_len:
                score += 1  # 扁平掩码与 values 长度一致
            if len(candidate_mask) == n_seg and all(
                x is None or isinstance(x, (list, tuple, bool, int)) for x in candidate_mask
            ):
                score += 2  # 逐段嵌套掩码
    return score


def offsets_from_lengths(lengths: Sequence[int]) -> list[int]:
    """``[3, 0, 3] -> [0, 3, 3, 6]``。"""
    offsets = [0]
    total = 0
    for length in lengths:
        length = int(length)
        if length < 0:
            raise InvalidOffsetsError(f"段长度不能为负: {length}")
        total += length
        offsets.append(total)
    return offsets


def normalize_offsets(offsets: Sequence[Any], values_len: int) -> list[int]:
    """校验并归一化 offsets 为 ``list[int]``。"""
    if not isinstance(offsets, (list, tuple)):
        offsets = list(offsets)
    if len(offsets) < 1:
        raise InvalidOffsetsError("offsets 至少需要包含一个边界（起始 0）")
    result: list[int] = []
    for x in offsets:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise InvalidOffsetsError(f"offsets 必须全部为整数边界，得到 {x!r}")
        if isinstance(x, float) and not x.is_integer():
            raise InvalidOffsetsError(f"offsets 必须是整数边界，得到 {x!r}")
        result.append(int(x))
    if any(result[i] > result[i + 1] for i in range(len(result) - 1)):
        raise InvalidOffsetsError(f"offsets 必须单调不减: {result}")
    if result[0] != 0:
        # 宽容处理带切片起点的 offsets：平移到 0。
        shift = result[0]
        result = [x - shift for x in result]
    if result[-1] != values_len:
        raise InvalidOffsetsError(
            f"offsets 最后一项 {result[-1]} 与 values 长度 {values_len} 不一致"
        )
    return result


def normalize_values(values: Sequence[Any]) -> list[float]:
    if values is None:
        raise InvalidOffsetsError("缺少 values")
    if not isinstance(values, (list, tuple)):
        values = list(values)
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise InvalidOffsetsError(f"values 必须全部是数值: {exc}") from exc


# ---------------------------------------------------------------------------
# mask 归一化
# ---------------------------------------------------------------------------

def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    raise MaskShapeError(f"掩码元素必须是 bool/0/1，得到 {x!r}")


def normalize_masks(
    masks: Any,
    offsets: Sequence[int],
    segment_ids: Optional[Sequence[Any]] = None,
) -> list[bool]:
    """把各种掩码表示归一化为与 values 对齐的 ``list[bool]``。

    约定 ``True`` 表示该位置**有效、允许参与归约**。
    """
    n_segments = len(offsets) - 1
    n_values = offsets[-1]
    if masks is None:
        return [True] * n_values

    # dict: segment_id -> mask
    if isinstance(masks, dict):
        per_segment: list[Any] = []
        for i in range(n_segments):
            key: Any = i
            if segment_ids is not None:
                sid = segment_ids[i]
                if sid in masks:
                    key = sid
                elif str(sid) in masks:
                    key = str(sid)
            if key in masks:
                per_segment.append(masks[key])
            elif i in masks:
                per_segment.append(masks[i])
            else:
                per_segment.append(None)
        masks = per_segment

    if not isinstance(masks, (list, tuple)):
        # 单个标量掩码应用到全部位置
        if isinstance(masks, (bool, int)):
            return [bool(masks)] * n_values  # type: ignore[arg-type]
        masks = list(masks)

    # (1) 真正的嵌套逐段掩码：长度等于段数，每个元素是 list/tuple/None。
    #     裸 bool/int 不算在内——那会与“等长的扁平 bool 掩码”产生歧义，
    #     标量逐段掩码在第 (3) 步单独处理。
    if len(masks) == n_segments and all(
        item is None or isinstance(item, (list, tuple)) for item in masks
    ):
        flat: list[bool] = []
        for i, item in enumerate(masks):
            length = offsets[i + 1] - offsets[i]
            if item is None:
                flat.extend([True] * length)
                continue
            item_list = list(item)
            if len(item_list) != length:
                raise MaskShapeError(
                    f"段 {i} 的掩码长度 {len(item_list)} 与段长度 {length} 不一致"
                )
            flat.extend(_as_bool(x) for x in item_list)
        if len(flat) != n_values:
            raise MaskShapeError("嵌套掩码展开后长度与 offsets 边界不一致")
        return flat

    # (2) 与 values 对齐的扁平掩码（优先于裸标量逐段表示，避免长度巧合时歧义）。
    if len(masks) == n_values:
        return [_as_bool(x) for x in masks]

    # (3) 裸标量的逐段掩码：长度等于段数，每个 bool/int 广播到整段。
    if len(masks) == n_segments and all(isinstance(item, (bool, int)) for item in masks):
        flat = []
        for i, item in enumerate(masks):
            length = offsets[i + 1] - offsets[i]
            flat.extend([bool(item)] * length)
        return flat

    raise MaskShapeError(
        f"掩码长度 {len(masks)} 无法解释：既不是与 values 长度 {n_values} 对齐的扁平掩码，"
        f"也不是 {n_segments} 段的逐段（嵌套）掩码"
    )


def masks_to_nested(flat_mask: Sequence[bool], offsets: Sequence[int]) -> list[list[bool]]:
    """扁平掩码还原为 fixture 风格的逐段嵌套掩码。"""
    return [list(flat_mask[offsets[i] : offsets[i + 1]]) for i in range(len(offsets) - 1)]


# ---------------------------------------------------------------------------
# 统一输入解析
# ---------------------------------------------------------------------------

class CanonicalInput:
    """解析后的标准输入。"""

    __slots__ = ("offsets", "values", "mask", "segment_ids", "original_order", "extra")

    def __init__(
        self,
        offsets: list[int],
        values: list[float],
        mask: list[bool],
        segment_ids: list[Any],
        original_order: Optional[list[Any]],
        extra: dict[str, Any],
    ) -> None:
        self.offsets = offsets
        self.values = values
        self.mask = mask
        self.segment_ids = segment_ids
        self.original_order = original_order
        self.extra = extra


def _default_segment_ids(n_segments: int) -> list[Any]:
    return list(range(n_segments))


def _looks_nested_segments(x: Any) -> bool:
    """判断是否为 ``list[list[number]]`` 形式的逐段输入（允许空段 / None）。"""
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return False
    for item in x:
        if item is None:
            continue
        if isinstance(item, dict) or hasattr(item, "values"):
            continue
        if not isinstance(item, (list, tuple)):
            return False
        for leaf in item:
            if isinstance(leaf, bool):
                continue
            if not isinstance(leaf, (int, float)):
                return False
    return True


def _extract_segment_item(item: Any) -> tuple[list[Any], Any, Any]:
    """从嵌套段元素中取出 (values, mask, segment_id)。"""
    if item is None:
        return [], None, None
    if isinstance(item, dict):
        vals = _pick(item, _VALUE_KEYS)
        msk = _pick(item, _MASK_KEYS)
        sid = _pick(item, _ID_KEYS)
        return list(vals or []), msk, sid
    if hasattr(item, "values") and not isinstance(item, (list, tuple)):
        vals = getattr(item, "values")
        msk = getattr(item, "mask", None)
        sid = getattr(item, "segment_id", getattr(item, "id", None))
        return list(vals or []), msk, sid
    return list(item), None, None


def _coerce_nested_segments(
    nested: Any, masks: Any, segment_ids: Any
) -> tuple[list[float], list[int], Any, Any]:
    flat: list[float] = []
    offsets = [0]
    dict_ids: list[Any] = []
    dict_masks: list[Any] = []
    have_dict_mask = False
    for item in nested:
        vals, item_mask, sid = _extract_segment_item(item)
        flat.extend(float(v) for v in vals)
        offsets.append(len(flat))
        dict_ids.append(sid)
        if item_mask is not None:
            have_dict_mask = True
        dict_masks.append(item_mask if item_mask is not None else [True] * len(vals))
    if segment_ids is None and any(x is not None for x in dict_ids):
        segment_ids = [x if x is not None else i for i, x in enumerate(dict_ids)]
    if masks is None and have_dict_mask:
        masks = dict_masks
    return flat, offsets, masks, segment_ids


def _labels_to_offsets(labels: Sequence[Any]) -> tuple[list[int], list[Any]]:
    """把逐位置 segment 标签（连续段）转成 offsets 与逐段 id。"""
    offsets = [0]
    seg_ids: list[Any] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            offsets.append(i)
            seg_ids.append(labels[start])
            start = i
    return offsets, seg_ids


def coerce_input(*args: Any, **kwargs: Any) -> CanonicalInput:
    """从宽松的位置 / 关键字参数中解析出标准输入。

    支持的典型调用::

        coerce_input(payload_dict)
        coerce_input(values, offsets, masks)
        coerce_input(offsets, values, masks)
        coerce_input(values=..., offsets=..., masks=..., segment_ids=...)
    """
    # 弹出关键字中的已知字段
    offsets = kwargs.pop("offsets", None)
    if offsets is None:
        offsets = _pick(kwargs.pop("offset_spec", None), _OFFSET_KEYS)
    values = kwargs.pop("values", None)
    masks = kwargs.pop("masks", None)
    if masks is None:
        masks = kwargs.pop("mask", None)
    segment_ids = kwargs.pop("segment_ids", None)
    if segment_ids is None:
        segment_ids = kwargs.pop("labels", None)
    original_order = kwargs.pop("original_order", None)
    data = kwargs.pop("data", None)
    if data is None:
        data = kwargs.pop("payload", None)
    nested_segments = kwargs.pop("segments", None)
    if nested_segments is None:
        nested_segments = kwargs.pop("segment_values", None)
    if nested_segments is None:
        nested_segments = kwargs.pop("rows", None)

    # 规范键未直接给出时，从剩余关键字里按别名（indptr/logits/...）兜底。
    if offsets is None:
        for key in _OFFSET_KEYS:
            if key in kwargs:
                offsets = kwargs.pop(key)
                break
    if values is None:
        for key in _VALUE_KEYS:
            if key in kwargs:
                values = kwargs.pop(key)
                break
    if masks is None:
        for key in _MASK_KEYS:
            if key in kwargs:
                masks = kwargs.pop(key)
                break
    if segment_ids is None:
        for key in _ID_KEYS:
            if key in kwargs and key != "id":
                segment_ids = kwargs.pop(key)
                break

    positional = list(args)

    # 第一个位置参数可能是 dict / 路径 / batch 对象
    if positional and (
        isinstance(positional[0], (dict, str, os.PathLike)) or hasattr(positional[0], "read")
    ):
        payload = load_payload(positional.pop(0))
        if isinstance(payload, dict):
            offsets = offsets if offsets is not None else _pick(payload, _OFFSET_KEYS)
            values = values if values is not None else _pick(payload, _VALUE_KEYS)
            masks = masks if masks is not None else _pick(payload, _MASK_KEYS)
            segment_ids = segment_ids if segment_ids is not None else _pick(payload, _ID_KEYS)
            original_order = (
                original_order if original_order is not None else _pick(payload, _ORDER_KEYS)
            )
        else:
            # batch-like 对象
            offsets = offsets if offsets is not None else _pick(payload, _OFFSET_KEYS)
            values = values if values is not None else _pick(payload, _VALUE_KEYS)
            masks = masks if masks is not None else _pick(payload, _MASK_KEYS)
            segment_ids = segment_ids if segment_ids is not None else _pick(payload, _ID_KEYS)

    # data=/payload= 关键字
    if data is not None and (offsets is None or values is None):
        payload = load_payload(data)
        if isinstance(payload, dict):
            offsets = offsets if offsets is not None else _pick(payload, _OFFSET_KEYS)
            values = values if values is not None else _pick(payload, _VALUE_KEYS)
            masks = masks if masks is not None else _pick(payload, _MASK_KEYS)
            segment_ids = segment_ids if segment_ids is not None else _pick(payload, _ID_KEYS)

    # batch-like 单个位置参数（非 dict）
    if positional and offsets is None and values is None and not isinstance(positional[0], (list, tuple)):
        candidate = positional.pop(0)
        offsets = _pick(candidate, _OFFSET_KEYS)
        values = _pick(candidate, _VALUE_KEYS)
        if masks is None:
            masks = _pick(candidate, _MASK_KEYS)
        if segment_ids is None:
            segment_ids = _pick(candidate, _ID_KEYS)

    # 嵌套的逐段输入：segments=[[...], [], [...]]（可同时带嵌套 masks）
    if nested_segments is None and positional and offsets is None and values is None:
        if _looks_nested_segments(positional[0]):
            nested_segments = positional.pop(0)
            if masks is None and positional:
                masks = positional.pop(0)
    if nested_segments is not None and offsets is None and values is None:
        values, offsets, masks, segment_ids = _coerce_nested_segments(
            nested_segments, masks, segment_ids
        )

    # 剩余位置参数：values/offsets/masks 的某种顺序
    if positional:
        if offsets is not None and values is not None:
            # 第一个剩余位置可能是 masks
            if masks is None and len(positional) >= 1:
                masks = positional[0]
        elif len(positional) >= 2:
            first, second = positional[0], positional[1]
            rest_mask = positional[2] if len(positional) >= 3 else None
            if masks is None:
                masks = rest_mask
            # 用打分在 (offsets, values) 与 (values, offsets) 两种顺序间选择。
            first_as_offsets = _offset_score(first, _len_or_zero(second), rest_mask)
            second_as_offsets = _offset_score(second, _len_or_zero(first), rest_mask)
            if first_as_offsets > second_as_offsets:
                offsets = offsets if offsets is not None else first
                values = values if values is not None else second
            elif second_as_offsets > first_as_offsets:
                values = values if values is not None else first
                offsets = offsets if offsets is not None else second
            else:
                # 平分：优先选择末值严格等于另一序列长度的一方；
                # 仍无法区分时采用 offsets-first（与 “offsets/values” 表述一致）。
                if looks_like_offsets(second, _len_or_zero(first)) and not looks_like_offsets(
                    first, _len_or_zero(second)
                ):
                    values, offsets = first, second
                else:
                    offsets, values = first, second
        elif len(positional) == 1:
            only = positional[0]
            # 单个列表参数只能是 values（按一整段处理）；
            # 单独给 offsets 而没有 values 对算子没有意义。
            if values is None and isinstance(only, (list, tuple)):
                values = only
            elif offsets is None and looks_like_offsets(only):
                offsets = only
            elif values is None:
                values = only

    if values is None:
        raise InvalidOffsetsError("缺少 values 输入")
    values = normalize_values(values)

    # 没有 offsets 但给了“逐位置标签”时，用连续标签推导 offsets。
    if (
        offsets is None
        and segment_ids is not None
        and len(segment_ids) == len(values)
        and len(values) > 0
    ):
        offsets, segment_ids = _labels_to_offsets(list(segment_ids))

    if offsets is None:
        # 没有 offsets 时把整个 values 当作一个段。
        offsets = [0, len(values)]
    offsets = normalize_offsets(offsets, len(values))

    n_segments = len(offsets) - 1
    mask = normalize_masks(masks, offsets, segment_ids)

    if segment_ids is None:
        segment_ids = _default_segment_ids(n_segments)
    else:
        segment_ids = list(segment_ids)
        # 给了 offsets + 逐位置标签：从每段起点取该段 id。
        if len(segment_ids) == len(values) and len(segment_ids) != n_segments:
            segment_ids = [segment_ids[offsets[i]] for i in range(n_segments)]
        if len(segment_ids) != n_segments:
            raise InvalidOffsetsError(
                f"segment_ids 数量 {len(segment_ids)} 与段数 {n_segments} 不一致"
            )

    return CanonicalInput(
        offsets=offsets,
        values=values,
        mask=mask,
        segment_ids=segment_ids,
        original_order=list(original_order) if original_order is not None else None,
        extra=kwargs,
    )
