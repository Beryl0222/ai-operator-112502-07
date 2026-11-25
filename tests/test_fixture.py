import json
from pathlib import Path
import unittest


class FixtureContractTest(unittest.TestCase):
    def test_ragged_fixture_keeps_empty_segment_and_mask_lengths(self):
        base = Path(__file__).parents[1]
        payload = json.loads((base / "data/input.json").read_text(encoding="utf-8"))
        lengths = [right - left for left, right in zip(payload["offsets"], payload["offsets"][1:])]
        self.assertEqual(lengths, [3, 0, 3])
        self.assertEqual(len(payload["segment_ids"]), len(payload["masks"]))
        self.assertEqual(len(payload["values"]), payload["offsets"][-1])

if __name__ == "__main__":
    unittest.main()
