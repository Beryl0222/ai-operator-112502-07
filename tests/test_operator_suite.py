"""覆盖业务契约的自测套件（标准库 unittest）。"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import unittest

_BASE = Path(__file__).resolve().parents[1]
_SRC = _BASE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import operator_ragged_segments as ops  # noqa:: E402
from operator_ragged_segments import (  # noqa: E402
    BlockReducer,
    Gateway,
    OrderMap,
    PartialState,
    Precision,
    RaggedBatch,
    SegmentStatus,
    StreamingSegmentSoftmax,
    segmented_softmax,
)
from operator_ragged_segments.core import segment_max, segment_reduce  # noqa: E402
from operator_ragged_segments.remap import inverse_permutation, remap_aligned  # noqa: E402


def load_fixture():
    return json.loads((_BASE / "data" / "input.json").read_text(encoding="utf-8"))


def ref_softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = math.fsum(es)
    return [e / z for e in es]


class FixtureSoftmaxTests(unittest.TestCase):
    def setUp(self):
        self.payload = load_fixture()
        self.result = segmented_softmax(self.payload)

    def test_flat_length_and_offsets(self):
        self.assertEqual(len(self.result), len(self.payload["values"]))
        self.assertEqual(self.result.offsets, self.payload["offsets"])

    def test_segment_a_uses_only_unmasked(self):
        seg = self.result["a"]
        expected = ref_softmax([1.0, -2.0])  # 中间 0 被屏蔽
        self.assertEqual(seg.status, SegmentStatus.OK)
        self.assertAlmostEqual(seg[0], expected[0], places=12)
        self.assertEqual(seg[1], 0.0)  # 屏蔽位置
        self.assertAlmostEqual(seg[2], expected[1], places=12)
        self.assertAlmostEqual(seg.max_value, 1.0)

    def test_segment_c_partial_mask(self):
        seg = self.result["c"]
        expected = ref_softmax([4.0, 5.0])
        self.assertAlmostEqual(seg[0], expected[0], places=12)
        self.assertAlmostEqual(seg[1], expected[1], places=12)
        self.assertEqual(seg[2], 0.0)
        self.assertEqual(seg.valid_mask, [True, True, False])

    def test_empty_segment_distinct(self):
        seg = self.result["empty"]
        self.assertEqual(seg.status, SegmentStatus.EMPTY)
        self.assertEqual(len(seg), 0)
        self.assertIsNone(seg.max_value)
        self.assertTrue(seg.eligible)

    def test_no_nan_anywhere(self):
        text = json.dumps(self.result.to_dict())
        self.assertNotIn("NaN", text)
        for x in self.result:
            self.assertFalse(math.isnan(x))


class StatusDistinctionTests(unittest.TestCase):
    def test_empty_fully_masked_all_zero(self):
        offsets = [0, 0, 2, 5, 7]
        values = [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
        masks = [[], [False, False], [True, True, True], [True, True]]
        ids = ["empty", "masked", "normal", "zero"]
        r = segmented_softmax(values, offsets, masks, segment_ids=ids)
        self.assertEqual(r["empty"].status, "empty")
        self.assertEqual(r["masked"].status, "fully_masked")
        # 明确的不可用状态，且同义词可用
        self.assertEqual(r["masked"].status, "unavailable")
        self.assertTrue(r["masked"].status.is_unavailable)
        self.assertEqual(r["zero"].status, SegmentStatus.OK)
        # 全零有效值 -> 均匀分布，而不是被当成全屏蔽
        for p in r["zero"]:
            self.assertAlmostEqual(p, 0.5, places=12)
        # 全屏蔽段输出全 0 且有效掩码全 False，没有 NaN
        self.assertTrue(all(p == 0.0 for p in r["masked"]))
        self.assertEqual(r["masked"].valid_mask, [False, False])

    def test_underflow_valid_position_stays_valid(self):
        r = segmented_softmax([0.0, -2000.0, -10**6], [0, 3])
        seg = r.segments[0]
        self.assertEqual(seg.status, SegmentStatus.OK)
        self.assertEqual(seg.valid_mask, [True, True, True])
        self.assertAlmostEqual(seg[0], 1.0, places=12)
        # 下溢项概率为 0，但仍是 valid（与屏蔽区分）
        self.assertEqual(seg[1], 0.0)
        self.assertTrue(seg.valid_mask[1])

    def test_masked_extreme_value_excluded_from_max(self):
        # 1e9 被屏蔽，不能污染 max；屏蔽项也不能“复活”
        r = segmented_softmax([1e9, 0.0, -1.0], [0, 3], [False, True, True])
        seg = r.segments[0]
        self.assertAlmostEqual(seg.max_value, 0.0, places=6)
        self.assertEqual(seg[0], 0.0)
        self.assertFalse(seg.valid_mask[0])
        expected = ref_softmax([0.0, -1.0])
        self.assertAlmostEqual(seg[1], expected[0], places=12)
        self.assertAlmostEqual(seg[2], expected[1], places=12)

    def test_negative_valid_values_participate(self):
        r = segmented_softmax([-5.0, -5.0], [0, 2], [True, True])
        for p in r[0:2]:
            self.assertAlmostEqual(p, 0.5, places=12)

    def test_mask_takes_effect_before_reduction(self):
        # 屏蔽掉唯一的最大值后，剩余元素归一化不受影响
        r = segmented_softmax([100.0, 0.0, 1.0], [0, 3], [False, True, True])
        expected = ref_softmax([0.0, 1.0])
        self.assertAlmostEqual(r[1], expected[0], places=12)
        self.assertAlmostEqual(r[2], expected[1], places=12)


class ReductionTests(unittest.TestCase):
    def test_segment_max_with_mask_and_empty(self):
        r = segment_max([1.0, 9.0, 2.0], [0, 2, 3], [True, False, True])
        # seg0 只有第一个有效 -> 1.0；seg1 -> 2.0
        self.assertAlmostEqual(r[0], 1.0)
        self.assertAlmostEqual(r[1], 2.0)
        self.assertEqual([s.value for s in r.statuses], ["ok", "ok"])

    def test_reduce_unavailable_is_none_not_nan(self):
        r = segment_reduce([1.0, 2.0], [0, 0, 2], op="mean")
        self.assertIsNone(r[0])  # offsets [0,0,2]: seg0 空 -> None
        self.assertAlmostEqual(r[1], 1.5)  # seg1 = [1,2]

    def test_count_op(self):
        r = ops.segment_count([1.0, 2.0, 3.0], [0, 3], [True, False, True])
        self.assertEqual(r[0], 2)


class RemapTests(unittest.TestCase):
    def test_inverse_permutation(self):
        perm = [2, 0, 1]  # batch[0]=orig[2] ...
        inv = inverse_permutation(perm)
        for b, o in enumerate(perm):
            self.assertEqual(inv[o], b)

    def test_remap_softmax_to_original_order(self):
        original = ["a", "b", "c"]
        batch_order = ["c", "a", "b"]
        om = OrderMap(original, batch_order)
        payload = {
            "offsets": [0, 2, 3, 5],
            "values": [1.0, 2.0, 3.0, 4.0, 5.0],
            "segment_ids": batch_order,
        }
        result = segmented_softmax(payload)
        restored = om.to_original(result)
        self.assertEqual(restored.segment_ids, original)
        # 段内容跟着身份走
        self.assertEqual(len(restored["a"]), 1)
        self.assertEqual(len(restored["b"]), 2)
        self.assertEqual(len(restored["c"]), 2)

    def test_remap_aligned_with_placeholder(self):
        aligned = remap_aligned({"a": 1}, ["a", "b"], placeholder=None)
        self.assertEqual(aligned, [1, None])


class GatewayTests(unittest.TestCase):
    def _gw(self):
        gw = RaggedBatch()
        gw.append("a", [1.0, 0.0, -2.0], [True, False, True])
        gw.append("b", [10.0, 20.0])
        return gw

    def test_append_offsets_and_cancel(self):
        gw = self._gw()
        self.assertEqual(gw.offsets, [0, 3, 5])
        gw.cancel("b")
        self.assertEqual(gw.offsets, [0, 3])
        self.assertEqual(gw.segment_ids, ["a"])

    def test_delivered_snapshot_protected_from_new_offsets(self):
        gw = self._gw()
        delivered = gw.deliver("a")
        snapshot = copy.deepcopy(gw.deliveries["a"].to_dict())
        # 追加 + 取消，制造全新 offsets
        gw.append("c", [4.0, 5.0, 6.0], [True, True, False])
        gw.cancel("b")
        gw.append("d", [7.0])
        # 已交付结果完全不变
        self.assertEqual(gw.deliveries["a"].to_dict(), snapshot)
        self.assertTrue(all(not math.isnan(x) for x in gw.deliveries["a"]))
        # 稳定偏移不变
        self.assertEqual(gw.delivered_offsets["a"], (0, 3))
        self.assertEqual(delivered.segment_id, "a")

    def test_duplicate_append_after_delivery_rejected(self):
        gw = self._gw()
        gw.deliver("a")
        with self.assertRaises(ops.SegmentDeliveredError):
            gw.append("a", [9.0, 9.0])

    def test_update_offsets_cannot_overwrite_delivered_range(self):
        gw = self._gw()
        gw.deliver("a")
        with self.assertRaises(ops.SegmentDeliveredError):
            gw.update_offsets([0, 99, 100])
        # 合法的、保持已交付边界的 offsets 可以接受
        accepted = gw.update_offsets([0, 3, 5])
        self.assertEqual(accepted, [0, 3, 5])

    def test_gateway_remap_to_saved_original_order(self):
        gw = Gateway()
        gw.append("a", [1.0, 2.0])
        gw.append("b", [3.0])
        result = gw.softmax()
        restored = gw.remap_to_original(result)
        self.assertEqual(restored.segment_ids, ["a", "b"])

    def test_fixture_ingest_and_deliver_all(self):
        gw = RaggedBatch(load_fixture())
        out = gw.deliver()
        self.assertEqual(set(out.keys()), {"a", "empty", "c"})
        self.assertIn("a", gw.deliveries)


class StreamingTests(unittest.TestCase):
    def test_partial_eligibility_and_recovery(self):
        p = load_fixture()
        flat_mask = [b for seg in p["masks"] for b in seg]
        off, vals, ids = p["offsets"], p["values"], p["segment_ids"]

        s = StreamingSegmentSoftmax(off, segment_ids=ids)
        rep = s.update(vals[:2], flat_mask[:2])
        # 只有部分输入：c 一定还在 pending
        self.assertIn("c", rep["pending_segment_ids"])
        self.assertFalse(rep["all_eligible"])
        self.assertTrue(rep["recoverable"] if "recoverable" in rep else True)

        # 快照 -> JSON -> 新实例恢复，再喂剩余数据
        blob = json.dumps(s.snapshot())
        restored = StreamingSegmentSoftmax.from_snapshot(json.loads(blob))
        restored.update(vals[2:4], flat_mask[2:4])
        restored.update(vals[4:], flat_mask[4:])
        final = restored.finish()
        batch = segmented_softmax(p)
        for a, b in zip(final, batch):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual([x.value for x in final.statuses], ["ok", "empty", "ok"])

    def test_incomplete_result_marks_pending_and_positions(self):
        s = StreamingSegmentSoftmax([0, 4], segment_ids=["long"])
        s.update([1.0, 2.0])
        result = s.result()
        seg = result.segments[0]
        self.assertEqual(seg.status, SegmentStatus.PENDING)
        self.assertFalse(seg.eligible)
        self.assertEqual(len(result.ineligible_positions), 2)
        report = s.last_report
        self.assertEqual(report["not_yet_arrived_positions"], [2, 3])
        with self.assertRaises(ops.IncompleteInputError):
            s.finish()

    def test_online_state_merge_primitive(self):
        a = PartialState("x", 0)
        a.absorb([1.0, 2.0], [True, True])
        b = PartialState("x", 0)
        b.absorb([0.0], [True])
        a.merge(b)
        # 与一次性 [1,2,0] 的 max/sum_exp（平移量）一致
        m = 2.0
        expected_sum = sum(math.exp(v - m) for v in [1.0, 2.0, 0.0])
        self.assertAlmostEqual(a.max_value, 2.0)
        self.assertAlmostEqual(a.sum_exp, expected_sum, places=12)

    def test_block_reducer_crosses_boundary(self):
        br = BlockReducer([0, 3, 6], segment_ids=["a", "b"], block_size=4)
        br.update([1.0, 2.0, 3.0, 4.0])  # 块跨越 a/b 边界且未到齐
        partials = br.block_partials(4)
        seg_ids = {p["segment_id"] for p in partials}
        self.assertEqual(seg_ids, {"a", "b"})
        open_state = br.open_block_state()
        # 已喂 4 个，block_size=4 -> 第一块恰好封满，没有 open block
        self.assertIsNone(open_state)
        br.update([5.0])
        open_state = br.open_block_state()
        self.assertIsNotNone(open_state)
        self.assertTrue(open_state["recoverable"])


class PrecisionTests(unittest.TestCase):
    def test_fast_and_precise_share_identity_and_status(self):
        p = load_fixture()
        fast = segmented_softmax(p, precision="fast")
        precise = segmented_softmax(p, precision="exact")  # 同义词
        self.assertEqual(fast.segment_ids, precise.segment_ids)
        self.assertEqual(
            [s.value for s in fast.statuses], [s.value for s in precise.statuses]
        )
        for a, b in zip(fast, precise):
            self.assertAlmostEqual(a, b, places=9)

    def test_precise_uses_fsum_on_pathological_sum(self):
        n = 100000
        tiny = -math.log(1e16)  # exp(tiny) ~ 1e-16
        values = [0.0] + [tiny] * n
        precise = segmented_softmax(values, [0, n + 1], precision="precise")
        fast = segmented_softmax(values, [0, n + 1], precision="fast")
        # n * 1e-16 = 1e-11；fsum 能保留这些小项，普通顺序求和会被 1 吞掉。
        self.assertGreater(precise.segments[0].sum_exp, 1.0 + 1e-12)
        self.assertLess(precise.segments[0].sum_exp, 1.0 + 1e-10)
        # 普通顺序求和会把这些 1e-16 小项吞掉（结果非常接近 / 等于 1.0）
        self.assertLess(fast.segments[0].sum_exp - 1.0, 1e-14)
        self.assertGreater(precise.segments[0].sum_exp, fast.segments[0].sum_exp)
        # 两条路径的状态解释完全一致
        self.assertEqual(precise.statuses, fast.statuses)

    def test_tier_mapping(self):
        self.assertEqual(ops.precision_for_tier("gold"), Precision.PRECISE)
        self.assertEqual(ops.precision_for_tier("standard"), Precision.FAST)
        policy = ops.PrecisionPolicy.from_tier("premium")
        self.assertTrue(policy.mode.uses_fsum)


class ApiAndAliasTests(unittest.TestCase):
    def test_compute_dict_is_jsonable(self):
        out = ops.compute(load_fixture(), remap=True)
        json.dumps(out)  # 不抛异常
        self.assertIn("probabilities", out)
        self.assertIn("stats", out)

    def test_fuzzy_imports(self):
        # 直接调用类（构造参数带数据）“做我所想”返回结果
        direct = ops.SegmentedSoftmax([0.0, 0.0], [0, 2])
        self.assertAlmostEqual(direct[0], 0.5)
        # 配置后再调用 / forward 风格
        op = ops.SegmentedSoftmax(precision="fast")
        self.assertAlmostEqual(op([0.0, 0.0], [0, 2])[1], 0.5)
        self.assertAlmostEqual(op.forward([0.0, math.log(3.0)], [0, 2])[1], 0.75)
        from operator_ragged_segments.gateway import RaggedGateway  # 兜底别名模块

        self.assertIs(RaggedGateway, RaggedBatch)

    def test_create_operator_factory(self):
        op = ops.create_operator("softmax")
        r = op([0.0, 0.0], [0, 2])
        self.assertAlmostEqual(r[0], 0.5)


class FlexibleInputAndOffsetTableTests(unittest.TestCase):
    def test_nested_segments_input(self):
        r = segmented_softmax(
            [[1.0, 1.0], [], [0.0, 0.0]],
            masks=[[True, True], [], [True, True]],
            segment_ids=["a", "e", "c"],
        )
        self.assertEqual([s.value for s in r.statuses], ["ok", "empty", "ok"])
        self.assertAlmostEqual(r[0], 0.5)
        self.assertAlmostEqual(r[3], 0.5)

    def test_per_position_labels(self):
        r = segmented_softmax([0.0, 0.0, 10.0], segment_ids=["x", "x", "y"])
        self.assertEqual(r.offsets, [0, 2, 3])
        self.assertEqual(r.segment_ids, ["x", "y"])
        self.assertAlmostEqual(r[0], 0.5)
        self.assertAlmostEqual(r[2], 1.0)

    def test_tolist_and_string_key(self):
        r = segmented_softmax([0.0, 0.0], [0, 2])
        self.assertEqual(r.tolist(), [0.5, 0.5])
        self.assertEqual(r["statuses"][0], "ok")

    def test_offset_table_protection(self):
        table = ops.OffsetTable()
        table.append_length(3, "a")
        table.append_length(0, "e")
        table.append_length(3, "c")
        self.assertEqual(table.offsets, [0, 3, 3, 6])
        table.protect("a")
        table.update([0, 3, 3, 9])  # 改动的是末尾边界，a 的 [0,3) 不变
        with self.assertRaises(ops.SegmentDeliveredError):
            table.update([0, 2, 2, 9])  # 移动了 a 的右边界


if __name__ == "__main__":
    unittest.main()
