# -*- coding: utf-8 -*-
"""Collect parameter counts, peak VRAM, train minutes, and inference latency."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from discourse.datasets import DiscourseDataset, collate_batch  # noqa: E402
from discourse.predict import load_checkpoint  # noqa: E402
from discourse.utils import ensure_dir  # noqa: E402

EXPS = [
    "E1a_textcnn",
    "E1b_bilstm",
    "E1c_indobertweet",
    "E2_stl_tox",
    "E2_stl_pol",
    "E3_soft_mtl",
    "E4_consistency",
    "E5_P1_soft_mtl",
    "E5_P2_soft_mtl",
]


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


def latency_ms(model, df, cfg, vocab, tokenizer, device, warmup=5, reps=30) -> float:
    ds = DiscourseDataset(
        df.head(64),
        tokenizer=tokenizer,
        vocab=vocab,
        max_length=cfg["training"]["max_length"],
        target_mode=cfg["training"].get("target_mode", "hard"),
    )
    batch_size = cfg["training"]["micro_batch_size"]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(input_ids, attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    per_batch_ms = (dt / reps) * 1000.0
    per_example_ms = per_batch_ms / input_ids.size(0)
    return {
        "latency_batch_ms": float(per_batch_ms),
        "latency_example_ms": float(per_example_ms),
        "latency_batch_size": int(input_ids.size(0)),
    }


def main():
    seed = 42
    all_df = pd.read_parquet(ROOT / "data" / "processed" / "all_splits.parquet")
    test_df = all_df[all_df.split == "test"].copy()
    rows = []
    for exp_id in EXPS:
        ckpt = ROOT / "artifacts" / "checkpoints" / exp_id / f"seed_{seed}" / "best.pt"
        if not ckpt.exists():
            print("[skip]", exp_id)
            continue
        model, cfg, thresholds, vocab, tokenizer, device = load_checkpoint(ckpt, "cuda")
        meta = {}
        meta_path = ckpt.parent / "seed_run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # fallback training_summary / seeded metrics
        train_minutes = meta.get("train_minutes")
        peak_vram = meta.get("peak_vram_mb")
        metrics_path = ROOT / "artifacts" / "metrics" / "seeded" / f"{exp_id}_seed{seed}_test_metrics.csv"
        if metrics_path.exists():
            m = pd.read_csv(metrics_path)
            if train_minutes is None and "train_minutes" in m.columns:
                train_minutes = float(m["train_minutes"].dropna().iloc[0]) if m["train_minutes"].notna().any() else None
            if peak_vram is None and "peak_vram_mb" in m.columns:
                peak_vram = float(m["peak_vram_mb"].dropna().iloc[0]) if m["peak_vram_mb"].notna().any() else None
        params = count_params(model)
        lat = latency_ms(model, test_df, cfg, vocab, tokenizer, device)
        row = {
            "experiment_id": exp_id,
            "seed": seed,
            "device": str(device),
            "train_minutes": train_minutes,
            "peak_vram_mb": peak_vram,
            **params,
            **lat,
            "max_length": cfg["training"]["max_length"],
            "micro_batch_size": cfg["training"]["micro_batch_size"],
        }
        rows.append(row)
        print(exp_id, params, f"lat_ex={lat['latency_example_ms']:.3f}ms")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = ensure_dir(ROOT / "artifacts" / "metrics" / "efficiency")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "efficiency_summary.csv", index=False)
    print("Wrote", out_dir / "efficiency_summary.csv")


if __name__ == "__main__":
    main()
