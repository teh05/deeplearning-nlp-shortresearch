from __future__ import annotations

import ast
import json
import math

import pandas as pd


def parse_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError("Expected list")
        return parsed
    raise TypeError(f"Unsupported annotation type: {type(value)}")


def _to_binary(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v in (0, 1, 0.0, 1.0):
            return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s in {"0", "1"}:
            return int(s)
    raise ValueError(f"Votes must contain only 0 or 1, got {v!r}")


def summarize_votes(values):
    raw = parse_list(values)
    votes = [_to_binary(v) for v in raw]
    n = len(votes)
    if n == 0:
        return dict(n=0, q=None, hard=None, mask=False, entropy=None, multi=False)
    q = sum(votes) / n
    entropy = 0.0
    if 0 < q < 1:
        entropy = -q * math.log2(q) - (1 - q) * math.log2(1 - q)
    return dict(
        n=n,
        q=q,
        hard=None if q == 0.5 else int(q > 0.5),
        mask=True,
        entropy=entropy,
        multi=n >= 2,
    )


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for source, suffix in [("votes_tox", "tox"), ("votes_pol", "pol")]:
        stats = out[source].map(summarize_votes)
        stats = pd.DataFrame(stats.tolist(), index=out.index)
        stats.columns = [f"{c}_{suffix}" for c in stats.columns]
        out = pd.concat([out, stats], axis=1)
    return out
