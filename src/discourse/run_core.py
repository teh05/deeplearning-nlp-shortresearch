# -*- coding: utf-8 -*-
"""Train E0/E1a/E1b/E1c/E3 across seeds and export per-seed metrics."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from discourse.calibration import expected_calibration_error, brier_score  # noqa: E402
from discourse.evaluate import (  # noqa: E402
    classification_metrics,
    export_metrics_and_figures,
    select_thresholds,
    soft_distribution_metrics,
)
from discourse.predict import load_checkpoint, predict_dataframe  # noqa: E402
from discourse.train import run_training  # noqa: E402
from discourse.utils import ensure_dir, load_yaml, set_seed  # noqa: E402


SEEDS = [42, 43, 44]
EXP_CONFIGS = {
    "E1a_textcnn": "configs/textcnn.yaml",
    "E1b_bilstm": "configs/bilstm.yaml",
    "E1c_indobertweet": "configs/indobertweet.yaml",
    "E3_soft_mtl": "configs/multitask_soft.yaml",
}


def seed_dirs(exp_id: str, seed: int) -> dict[str, Path]:
    base = ROOT / "artifacts"
    return {
        "ckpt": ensure_dir(base / "checkpoints" / exp_id / f"seed_{seed}"),
        "metrics": ensure_dir(base / "metrics" / "seeded"),
        "preds": ensure_dir(base / "predictions" / "seeded"),
        "figures": ensure_dir(base / "figures" / "seeded" / f"{exp_id}_seed{seed}"),
        "tables": ensure_dir(base / "tables" / "seeded"),
    }


def run_tfidf_lr_seed(train_df, val_df, test_df, seed: int) -> pd.DataFrame:
    set_seed(seed)
    rows = []
    for task_name, hard_col in [("tox", "hard_tox"), ("pol", "hard_pol")]:
        tr = train_df[train_df[hard_col].notna()].copy()
        va = val_df[val_df[hard_col].notna()].copy()
        te = test_df[test_df[hard_col].notna()].copy()
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
        Xtr = vec.fit_transform(tr["text_proc"].astype(str))
        Xva = vec.transform(va["text_proc"].astype(str))
        Xte = vec.transform(te["text_proc"].astype(str))
        clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1, random_state=seed)
        clf.fit(Xtr, tr[hard_col].astype(int))
        pva = clf.predict_proba(Xva)[:, 1]
        yva = va[hard_col].astype(int).to_numpy()
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(yva, (pva >= t).astype(int), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pte = clf.predict_proba(Xte)[:, 1]
        yte = te[hard_col].astype(int).to_numpy()
        pred = (pte >= best_t).astype(int)
        ece = expected_calibration_error(yte, pte)
        brier = brier_score(yte, pte)
        rows.append(
            {
                "experiment_id": "E0_tfidf_lr",
                "seed": seed,
                "split": "test",
                "task": task_name,
                "n": int(len(te)),
                "threshold": best_t,
                "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                "pos_f1": float(f1_score(yte, pred, pos_label=1, zero_division=0)),
                "average_precision": float(average_precision_score(yte, pte))
                if len(np.unique(yte)) > 1
                else float("nan"),
                "mse": float("nan"),
                "cross_entropy": float("nan"),
                "ece": ece,
                "brier": brier,
                "train_minutes": None,
                "peak_vram_mb": None,
            }
        )
    return pd.DataFrame(rows)


def metrics_from_pred(exp_id: str, seed: int, pred_df: pd.DataFrame, thresholds: dict, train_minutes, peak_vram) -> pd.DataFrame:
    rows = []
    for task in ("tox", "pol"):
        m = classification_metrics(pred_df, task, thresholds[task])
        s = soft_distribution_metrics(pred_df, task)
        hard_col = f"hard_{task}"
        prob_col = f"prob_{task}"
        sub = pred_df[pred_df[hard_col].notna()]
        if len(sub):
            y_true = sub[hard_col].astype(int).to_numpy()
            y_prob = sub[prob_col].to_numpy()
            ece = expected_calibration_error(y_true, y_prob)
            brier = brier_score(y_true, y_prob)
        else:
            ece, brier = float("nan"), float("nan")
        rows.append(
            {
                "experiment_id": exp_id,
                "seed": seed,
                "split": "test",
                "task": task,
                "n": m["n"],
                "threshold": thresholds[task],
                "macro_f1": m["macro_f1"],
                "pos_f1": m["pos_f1"],
                "average_precision": m["average_precision"],
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "roc_auc": m.get("roc_auc"),
                "mse": s.get("mse"),
                "cross_entropy": s.get("cross_entropy"),
                "n_multi": s.get("n_multi"),
                "ece": ece,
                "brier": brier,
                "train_minutes": train_minutes,
                "peak_vram_mb": peak_vram,
            }
        )
    return pd.DataFrame(rows)


def import_seed42_from_existing(test_df) -> list[pd.DataFrame]:
    """Reuse already-trained seed-42 DL checkpoints if present."""
    frames = []
    for exp_id in EXP_CONFIGS:
        legacy_ckpt = ROOT / "artifacts" / "checkpoints" / exp_id / "best.pt"
        out = seed_dirs(exp_id, 42)
        seeded_ckpt = out["ckpt"] / "best.pt"
        if legacy_ckpt.exists() and not seeded_ckpt.exists():
            shutil.copy2(legacy_ckpt, seeded_ckpt)
            for extra in ("history.csv", "run_meta.json", "vocab.json"):
                src = legacy_ckpt.parent / extra
                if src.exists():
                    shutil.copy2(src, out["ckpt"] / extra)
        if not seeded_ckpt.exists():
            print(f"[skip import] missing checkpoint for {exp_id} seed42")
            continue
        print(f"[import] evaluating existing {exp_id} seed=42")
        model, cfg, thresholds, vocab, tokenizer, device = load_checkpoint(seeded_ckpt, "cuda")
        val_df = test_df  # placeholder overwritten below by caller? need val
        # Will be handled in main with proper val_df; here only if we reload
        frames.append((exp_id, 42, seeded_ckpt))
    return frames


def train_one_dl(exp_id: str, cfg_path: str, train_df, val_df, test_df, seed: int, force: bool = False):
    dirs = seed_dirs(exp_id, seed)
    metrics_path = dirs["metrics"] / f"{exp_id}_seed{seed}_test_metrics.csv"
    ckpt_path = dirs["ckpt"] / "best.pt"
    if metrics_path.exists() and ckpt_path.exists() and not force:
        print(f"[skip] {exp_id} seed={seed} already has metrics")
        return pd.read_csv(metrics_path)

    cfg = load_yaml(ROOT / cfg_path)
    cfg["seed"] = seed
    cfg["experiment_id"] = exp_id
    print(f"\n=== Training {exp_id} seed={seed} ===")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    result = run_training(cfg, train_df, val_df, out_dir=dirs["ckpt"], device_prefer="cuda")
    elapsed = (time.time() - t0) / 60.0
    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    model, cfg_loaded, thresholds, vocab, tokenizer, device = load_checkpoint(result["best_path"], "cuda")
    val_pred = predict_dataframe(model, val_df, cfg_loaded, vocab, tokenizer, device)
    thresholds = select_thresholds(val_pred)
    test_pred = predict_dataframe(model, test_df, cfg_loaded, vocab, tokenizer, device)
    test_pred["experiment_id"] = exp_id
    test_pred["seed"] = seed
    test_pred["split"] = "test"
    pred_path = dirs["preds"] / f"{exp_id}_seed{seed}_test_preds.csv"
    test_pred.to_csv(pred_path, index=False)

    # figures optional (keep light for multi-seed)
    export_metrics_and_figures(
        f"{exp_id}_seed{seed}",
        test_pred,
        thresholds,
        split_name="test",
        metrics_dir=dirs["metrics"],
        figures_dir=dirs["figures"],
    )
    metrics_df = metrics_from_pred(exp_id, seed, test_pred, thresholds, elapsed, peak_vram)
    metrics_df.to_csv(metrics_path, index=False)
    save_meta = {
        "experiment_id": exp_id,
        "seed": seed,
        "train_minutes": elapsed,
        "peak_vram_mb": peak_vram,
        "thresholds": thresholds,
        "best_val_mean_macro_f1": result.get("best_score"),
        "checkpoint": str(result["best_path"]),
    }
    (dirs["ckpt"] / "seed_run_meta.json").write_text(json.dumps(save_meta, indent=2), encoding="utf-8")
    print(f"[done] {exp_id} seed={seed} minutes={elapsed:.2f} peak_vram={peak_vram}")
    return metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--exps",
        nargs="+",
        default=["E0_tfidf_lr", "E1a_textcnn", "E1b_bilstm", "E1c_indobertweet", "E3_soft_mtl"],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-e0", action="store_true")
    args = parser.parse_args()

    all_df = pd.read_parquet(ROOT / "data" / "processed" / "all_splits.parquet")
    train_df = all_df[all_df.split == "train"].copy()
    val_df = all_df[all_df.split == "validation"].copy()
    test_df = all_df[all_df.split == "test"].copy()
    print("splits", {k: len(v) for k, v in {"train": train_df, "val": val_df, "test": test_df}.items()})

    all_rows = []

    # Import/eval seed 42 DL checkpoints first if requested seeds include 42
    if 42 in args.seeds:
        for exp_id, cfg_path in EXP_CONFIGS.items():
            if exp_id not in args.exps:
                continue
            dirs = seed_dirs(exp_id, 42)
            metrics_path = dirs["metrics"] / f"{exp_id}_seed42_test_metrics.csv"
            legacy = ROOT / "artifacts" / "checkpoints" / exp_id / "best.pt"
            seeded = dirs["ckpt"] / "best.pt"
            if legacy.exists() and not seeded.exists():
                shutil.copy2(legacy, seeded)
                for extra in ("history.csv", "run_meta.json", "vocab.json"):
                    src = legacy.parent / extra
                    if src.exists():
                        shutil.copy2(src, dirs["ckpt"] / extra)
            if seeded.exists() and (not metrics_path.exists() or args.force):
                print(f"[eval-only] {exp_id} seed=42 from existing checkpoint")
                model, cfg_loaded, thresholds, vocab, tokenizer, device = load_checkpoint(seeded, "cuda")
                val_pred = predict_dataframe(model, val_df, cfg_loaded, vocab, tokenizer, device)
                thresholds = select_thresholds(val_pred)
                test_pred = predict_dataframe(model, test_df, cfg_loaded, vocab, tokenizer, device)
                test_pred.to_csv(dirs["preds"] / f"{exp_id}_seed42_test_preds.csv", index=False)
                # train minutes from legacy meta if any
                train_minutes, peak_vram = None, None
                meta_path = dirs["ckpt"] / "run_meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                legacy_summary = ROOT / "artifacts" / "metrics" / "training_summary.csv"
                if legacy_summary.exists():
                    summary = pd.read_csv(legacy_summary)
                    hit = summary[summary.experiment_id == exp_id]
                    if len(hit):
                        train_minutes = float(hit.iloc[0]["train_minutes"])
                        peak_vram = float(hit.iloc[0]["peak_vram_mb"])
                mdf = metrics_from_pred(exp_id, 42, test_pred, thresholds, train_minutes, peak_vram)
                mdf.to_csv(metrics_path, index=False)
                all_rows.append(mdf)
            elif metrics_path.exists():
                all_rows.append(pd.read_csv(metrics_path))

    for seed in args.seeds:
        # E0
        if "E0_tfidf_lr" in args.exps and not args.skip_e0:
            e0_path = ensure_dir(ROOT / "artifacts" / "metrics" / "seeded") / f"E0_tfidf_lr_seed{seed}_test_metrics.csv"
            if e0_path.exists() and not args.force:
                print(f"[skip] E0 seed={seed}")
                all_rows.append(pd.read_csv(e0_path))
            else:
                print(f"=== E0 TF-IDF+LR seed={seed} ===")
                e0 = run_tfidf_lr_seed(train_df, val_df, test_df, seed)
                e0.to_csv(e0_path, index=False)
                all_rows.append(e0)

        for exp_id, cfg_path in EXP_CONFIGS.items():
            if exp_id not in args.exps:
                continue
            if seed == 42:
                # already handled via import/eval above
                path = seed_dirs(exp_id, 42)["metrics"] / f"{exp_id}_seed42_test_metrics.csv"
                if path.exists():
                    continue
            mdf = train_one_dl(exp_id, cfg_path, train_df, val_df, test_df, seed, force=args.force)
            all_rows.append(mdf)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(
            subset=["experiment_id", "seed", "task"], keep="last"
        )
        out = ensure_dir(ROOT / "artifacts" / "metrics" / "seeded") / "all_seeds_test_metrics.csv"
        # merge with any existing seeded files
        existing_files = list((ROOT / "artifacts" / "metrics" / "seeded").glob("*_seed*_test_metrics.csv"))
        frames = [pd.read_csv(p) for p in existing_files]
        if frames:
            combined = pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=["experiment_id", "seed", "task"], keep="last"
            )
        combined.to_csv(out, index=False)
        print("Wrote", out, "rows=", len(combined))


if __name__ == "__main__":
    main()
