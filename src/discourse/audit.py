from __future__ import annotations

from typing import Any

import pandas as pd

from discourse.targets import parse_list, _to_binary


def audit_annotations(df: pd.DataFrame) -> dict[str, Any]:
    issues = []
    n = len(df)
    empty_text = (df["text_raw"].astype(str).str.strip() == "").sum()

    bad_tox = 0
    bad_pol = 0
    len_mismatch = 0
    for i, row in df.iterrows():
        try:
            tox = [_to_binary(v) for v in parse_list(row["votes_tox"])]
        except Exception as e:
            bad_tox += 1
            issues.append({"text_id": row.get("text_id"), "issue": f"toxicity: {e}"})
            tox = None
        try:
            pol = [_to_binary(v) for v in parse_list(row["votes_pol"])]
        except Exception as e:
            bad_pol += 1
            issues.append({"text_id": row.get("text_id"), "issue": f"polarization: {e}"})
            pol = None
        try:
            anns = parse_list(row.get("annotators_id", []))
        except Exception:
            anns = []
        if tox is not None and pol is not None:
            if not (len(tox) == len(pol) == len(anns) or (len(anns) == 0)):
                # allow missing annotators list length only if empty; else flag
                if len(anns) > 0 and (len(tox) != len(anns) or len(pol) != len(anns)):
                    len_mismatch += 1
                    issues.append(
                        {
                            "text_id": row.get("text_id"),
                            "issue": f"length mismatch tox={len(tox)} pol={len(pol)} ann={len(anns)}",
                        }
                    )

    return {
        "n_rows": int(n),
        "empty_text": int(empty_text),
        "bad_toxicity": int(bad_tox),
        "bad_polarization": int(bad_pol),
        "annotator_length_mismatch": int(len_mismatch),
        "n_issue_rows": len(issues),
        "issues_sample": issues[:50],
    }


def filter_valid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop empty text; keep rows with at least one valid task annotation."""
    out = df.copy()
    out = out[out["text_raw"].astype(str).str.strip() != ""].copy()
    keep = (out["mask_tox"] == True) | (out["mask_pol"] == True)  # noqa: E712
    valid = out[keep].copy()
    dropped = out[~keep].copy()
    return valid, dropped
