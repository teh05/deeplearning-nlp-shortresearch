import pandas as pd

from discourse.duplicates import assign_group_ids
from discourse.split import make_splits


def test_no_group_overlap():
    df = pd.DataFrame(
        {
            "text_id": [f"t{i}" for i in range(100)],
            "text_raw": [f"text {i // 2}" for i in range(100)],
        }
    )
    df = assign_group_ids(df)
    parts = make_splits(df, seed=0)
    groups = {k: set(v["group_id"]) for k, v in parts.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
