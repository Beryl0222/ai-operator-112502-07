"""``python -m operator_ragged_segments``：在内置 fixture 上演示算子。"""

from __future__ import annotations

import json

from .io import load_input
from .softmax import segmented_softmax


def main() -> None:
    payload = load_input()
    result = segmented_softmax(payload)
    print("segment_ids:", result.segment_ids)
    for seg in result.segments:
        print(
            f"  {seg.segment_id!s:>8}: status={seg.status.value:<12} "
            f"max={seg.max_value} sum_exp={seg.sum_exp} probs={[round(p, 6) for p in seg]}"
        )
    print("flat:", json.dumps([round(p, 9) for p in result]))


if __name__ == "__main__":
    main()
