from __future__ import annotations

import hashlib

import pandas as pd


def exact_text_group_id(text: str) -> str:
    norm = " ".join(str(text).split()).lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def assign_group_ids(df: pd.DataFrame, text_col: str = "text_raw") -> pd.DataFrame:
    out = df.copy()
    out["group_id"] = out[text_col].map(exact_text_group_id)
    return out


def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("group_id")
        .agg(n_rows=("text_id", "count"), sample_text=("text_raw", "first"))
        .reset_index()
    )
    dups = g[g["n_rows"] > 1].sort_values("n_rows", ascending=False)
    return dups
