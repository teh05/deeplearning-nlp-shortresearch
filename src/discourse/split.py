from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from discourse.utils import ensure_dir, save_json, sha256_file


def make_splits(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=df["group_id"]))
    trainval = df.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["group_id"]))
    parts = {
        "train": trainval.iloc[train_idx].copy(),
        "validation": trainval.iloc[val_idx].copy(),
        "test": df.iloc[test_idx].copy(),
    }
    for name, part in parts.items():
        part["split"] = name
    groups = {k: set(v["group_id"]) for k, v in parts.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    return parts


def save_splits(parts: dict[str, pd.DataFrame], out_dir: str | Path) -> dict:
    out_dir = ensure_dir(out_dir)
    paths = {}
    for name, part in parts.items():
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        paths[name] = str(path)
    combined = pd.concat(parts.values(), ignore_index=True)
    ids_path = out_dir / "split_ids.csv"
    combined[["text_id", "group_id", "split"]].to_csv(ids_path, index=False)
    meta = {
        "n_train": len(parts["train"]),
        "n_validation": len(parts["validation"]),
        "n_test": len(parts["test"]),
        "paths": paths,
        "split_ids_csv": str(ids_path),
        "split_ids_sha256": sha256_file(ids_path),
    }
    save_json(meta, Path(out_dir) / "split_meta.json")
    return meta
