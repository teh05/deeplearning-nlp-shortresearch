from __future__ import annotations

from pathlib import Path

import pandas as pd

from discourse.audit import audit_annotations, filter_valid
from discourse.download import prepare_local_dataset
from discourse.duplicates import assign_group_ids, duplicate_report
from discourse.preprocessing import apply_preprocessing
from discourse.schema import load_raw_jsonl, schema_report, validate_and_map
from discourse.split import make_splits, save_splits
from discourse.targets import add_targets
from discourse.utils import ensure_dir, load_yaml, project_root, save_json


def prepare_all(config_path: str | Path = "configs/data.yaml", pipeline: str = "P0") -> dict:
    root = project_root()
    cfg = load_yaml(root / config_path)
    raw_path = prepare_local_dataset(cfg["dataset"]["local_path"])

    df_raw = load_raw_jsonl(raw_path)
    df = validate_and_map(df_raw, cfg["columns"])
    report = schema_report(df_raw, df)
    save_json(report, root / "artifacts" / "audits" / "schema_report.json")

    # stable internal id
    df.insert(0, "text_id", [f"t{i:06d}" for i in range(len(df))])
    df = add_targets(df)
    audit = audit_annotations(df)
    save_json(audit, root / "artifacts" / "audits" / "annotation_audit.json")

    valid, dropped = filter_valid(df)
    dropped.to_csv(root / "artifacts" / "audits" / "dropped_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_raw": len(df_raw),
                "n_mapped": len(df),
                "n_valid": len(valid),
                "n_dropped": len(dropped),
                "n_tox_valid": int(valid["mask_tox"].sum()),
                "n_pol_valid": int(valid["mask_pol"].sum()),
                "n_both_valid": int((valid["mask_tox"] & valid["mask_pol"]).sum()),
                "n_multi_tox": int(valid["multi_tox"].sum()),
                "n_multi_pol": int(valid["multi_pol"].sum()),
            }
        ]
    ).to_csv(root / "artifacts" / "tables" / "sample_counts.csv", index=False)

    valid = assign_group_ids(valid)
    dups = duplicate_report(valid)
    dups.to_csv(root / "artifacts" / "audits" / "exact_duplicates.csv", index=False)

    parts = make_splits(valid, seed=cfg["split"]["seed"])
    # preprocessing after split decision; fit dictionaries on train only later
    for name in parts:
        parts[name] = apply_preprocessing(parts[name], pipeline=pipeline)

    split_meta = save_splits(parts, root / cfg["paths"]["splits"])
    # also save combined processed parquet/csv
    processed_dir = ensure_dir(root / "data" / "processed")
    combined = pd.concat(parts.values(), ignore_index=True)
    combined.to_parquet(processed_dir / "all_splits.parquet", index=False)
    combined.to_csv(processed_dir / "all_splits.csv", index=False)

    # split audit table
    audit_rows = []
    for name, part in parts.items():
        audit_rows.append(
            {
                "split": name,
                "n": len(part),
                "n_groups": part["group_id"].nunique(),
                "tox_pos": int((part["hard_tox"] == 1).sum()),
                "tox_neg": int((part["hard_tox"] == 0).sum()),
                "tox_tie_or_missing": int(part["hard_tox"].isna().sum()),
                "pol_pos": int((part["hard_pol"] == 1).sum()),
                "pol_neg": int((part["hard_pol"] == 0).sum()),
                "pol_tie_or_missing": int(part["hard_pol"].isna().sum()),
            }
        )
    pd.DataFrame(audit_rows).to_csv(root / "artifacts" / "tables" / "split_label_audit.csv", index=False)

    return {
        "raw_path": str(raw_path),
        "n_valid": len(valid),
        "split_meta": split_meta,
        "parts": parts,
        "audit": audit,
        "schema": report,
    }
