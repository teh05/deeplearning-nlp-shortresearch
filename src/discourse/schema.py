from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "text_id": "text_id",
    "text": "text",
    "toxicity": "toxicity",
    "polarization": "polarized",  # actual HF column name
    "annotators": "annotators_id",
}


def load_raw_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_json(path, lines=True)
    return df


def validate_and_map(
    df: pd.DataFrame,
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    columns = columns or REQUIRED_COLUMNS
    missing = [src for src in columns.values() if src not in df.columns]
    if missing:
        raise ValueError(
            f"Schema mismatch. Missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["source_ids"] = df[columns["text_id"]].astype(str)
    out["text_raw"] = df[columns["text"]].astype(str)
    out["votes_tox"] = df[columns["toxicity"]]
    out["votes_pol"] = df[columns["polarization"]]
    out["annotators_id"] = df[columns["annotators"]]

    # keep useful optional metadata if present
    for opt in ("topic", "initial_paragraph"):
        if opt in df.columns:
            out[opt] = df[opt]

    return out


def schema_report(df_raw: pd.DataFrame, df_mapped: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_raw_rows": int(len(df_raw)),
        "n_mapped_rows": int(len(df_mapped)),
        "raw_columns": list(df_raw.columns),
        "mapped_columns": list(df_mapped.columns),
        "empty_text": int((df_mapped["text_raw"].str.strip() == "").sum()),
        "null_text": int(df_mapped["text_raw"].isna().sum()),
    }
