from __future__ import annotations

import pandas as pd

from biohub.validation import track_length_distribution


def test_real_track_length_traverses_chain() -> None:
    edges = pd.DataFrame(
        {"source_id": [1, 2, 3], "target_id": [2, 3, 4]}
    )
    assert track_length_distribution(edges).tolist() == [4]


def test_track_length_splits_daughters_at_division() -> None:
    edges = pd.DataFrame(
        {"source_id": [1, 2, 2, 3, 4], "target_id": [2, 3, 4, 5, 6]}
    )
    assert sorted(track_length_distribution(edges).tolist()) == [2, 2, 2]
