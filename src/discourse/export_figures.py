# -*- coding: utf-8 -*-
"""Menyusun ulang seluruh figur hasil eksperimen, termasuk semua seed 42/43/44.

Tahapan:
1) Regenerasi CM / PR / reliability dari prediksi seeded bila perlu
2) Menyalin ke artifacts/figures/laporan (kanonik) dan laporan/by_seed
3) Menjalankan standardisasi nama + grafik banding agregat
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from discourse.calibration import reliability_diagram  # noqa: E402
from discourse.evaluate import plot_confusion, plot_pr_curve  # noqa: E402
from discourse.utils import ensure_dir  # noqa: E402

MODEL = {
    "E0_tfidf_lr": "TFIDF-LR",
    "E1a_textcnn": "TextCNN",
    "E1b_bilstm": "BiLSTM",
    "E1c_indobertweet": "IndoBERTweet-Hard",
    "E2_stl": "IndoBERTweet-STL",
    "E2_stl_tox": "IndoBERTweet-STL-Tox",
    "E2_stl_pol": "IndoBERTweet-STL-Pol",
    "E3_soft_mtl": "IndoBERTweet-Soft",
    "E4_consistency": "IndoBERTweet-Consistency",
    "E5_P1_soft_mtl": "Soft-Preprocess-P1",
    "E5_P2_soft_mtl": "Soft-Preprocess-P2",
}
TASK = {"tox": "toksisitas", "pol": "polarisasi"}
KIND = {"cm": "confusion", "pr": "precision-recall", "reliability": "kalibrasi"}


def threshold_for(exp_id: str, seed: int, task: str) -> float:
    path = ROOT / "artifacts" / "metrics" / "seeded" / f"{exp_id}_seed{seed}_test_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        hit = df[df.task == task]
        if len(hit) and pd.notna(hit.iloc[0].get("threshold")):
            return float(hit.iloc[0]["threshold"])
    return 0.5


def rebuild_seed_figures(exp_id: str, seed: int, force: bool = False) -> list[Path]:
    """Membangun figur evaluasi dari prediksi tersimpan untuk satu seed."""
    pred_path = ROOT / "artifacts" / "predictions" / "seeded" / f"{exp_id}_seed{seed}_test_preds.csv"
    if not pred_path.exists():
        return []
    pred = pd.read_csv(pred_path)
    out_dir = ensure_dir(ROOT / "artifacts" / "figures" / "seeded" / f"{exp_id}_seed{seed}")
    written: list[Path] = []
    for task in ("tox", "pol"):
        hard = f"hard_{task}"
        if hard not in pred.columns or pred[hard].notna().sum() == 0:
            continue
        thr = threshold_for(exp_id, seed, task)
        for kind, plotter in (
            ("cm", lambda p, t, thr_, out, title: plot_confusion(p, t, thr_, out, title)),
            ("pr", lambda p, t, thr_, out, title: plot_pr_curve(p, t, out, title)),
        ):
            out = out_dir / f"{exp_id}_seed{seed}_test_{task}_{kind}.png"
            if out.exists() and not force:
                written.append(out)
                continue
            plotter(pred, task, thr, out, f"{exp_id} seed{seed} {task} {kind}")
            if out.exists():
                written.append(out)

        rel = out_dir / f"{exp_id}_seed{seed}_test_{task}_reliability.png"
        if (not rel.exists()) or force:
            sub = pred[pred[hard].notna()]
            if len(sub) and sub[hard].nunique() > 1:
                reliability_diagram(
                    sub[hard].astype(int).to_numpy(),
                    sub[f"prob_{task}"].to_numpy(),
                    rel,
                    f"Reliability {exp_id} seed{seed} {task}",
                )
        if rel.exists():
            written.append(rel)
    return written


def copy_by_seed(exp_id: str, seed: int, rows: list[dict]) -> None:
    """Menyalin figur seed ke folder laporan/by_seed dengan nama terbaca."""
    src_dir = ROOT / "artifacts" / "figures" / "seeded" / f"{exp_id}_seed{seed}"
    if not src_dir.exists():
        return
    nice = MODEL.get(exp_id, exp_id)
    out_dir = ensure_dir(ROOT / "artifacts" / "figures" / "laporan" / "by_seed")
    for task, task_name in TASK.items():
        for kind, kind_name in KIND.items():
            matches = sorted(src_dir.glob(f"*_{task}_{kind}.png"))
            if not matches:
                continue
            dst_name = f"{nice}_{task_name}_{kind_name}_seed{seed}.png"
            dst = out_dir / dst_name
            shutil.copy2(matches[0], dst)
            rows.append(
                {
                    "nama_laporan": f"by_seed/{dst_name}",
                    "experiment_id": exp_id,
                    "seed": seed,
                    "task": task,
                    "kind": kind_name,
                    "sumber": str(matches[0].relative_to(ROOT)).replace("\\", "/"),
                }
            )


def discover_seeded_experiments() -> list[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    pred_dir = ROOT / "artifacts" / "predictions" / "seeded"
    if pred_dir.exists():
        for path in pred_dir.glob("*_seed*_test_preds.csv"):
            name = path.name.replace("_test_preds.csv", "")
            if "_seed" not in name:
                continue
            exp_id, seed_s = name.rsplit("_seed", 1)
            if seed_s.isdigit():
                pairs.add((exp_id, int(seed_s)))
    fig_dir = ROOT / "artifacts" / "figures" / "seeded"
    if fig_dir.exists():
        for folder in fig_dir.iterdir():
            if not folder.is_dir() or "_seed" not in folder.name:
                continue
            exp_id, seed_s = folder.name.rsplit("_seed", 1)
            if seed_s.isdigit():
                pairs.add((exp_id, int(seed_s)))
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-standardize", action="store_true")
    args = parser.parse_args()

    pairs = [(e, s) for e, s in discover_seeded_experiments() if s in set(args.seeds)]
    print(f"Menyusun figur untuk {len(pairs)} pasangan eksperimen-seed")
    for exp_id, seed in pairs:
        n = rebuild_seed_figures(exp_id, seed, force=args.force_rebuild)
        print(f"  rebuilt/kept {exp_id} seed{seed}: {len(n)} files")

    rows: list[dict] = []
    for exp_id, seed in pairs:
        copy_by_seed(exp_id, seed, rows)

    catalog = pd.DataFrame(rows)
    out_dir = ensure_dir(ROOT / "artifacts" / "figures" / "laporan" / "by_seed")
    catalog_path = out_dir / "daftar_gambar_by_seed.csv"
    catalog.to_csv(catalog_path, index=False)
    print("Wrote", catalog_path, "rows=", len(catalog))

    if not args.skip_standardize:
        # Menjalankan skrip standarisasi nama figur hasil (EDA + seed42 kanonik + banding).
        from discourse.standardize_figures import main as standardize_main

        standardize_main()

    # Gabungan katalog laporan kanonik + by_seed
    canon = ROOT / "artifacts" / "figures" / "laporan" / "daftar_gambar_laporan.csv"
    frames = []
    if canon.exists():
        c = pd.read_csv(canon)
        c["kelompok"] = "laporan_kanonik"
        frames.append(c)
    if catalog_path.exists():
        b = pd.read_csv(catalog_path)
        b["kelompok"] = "by_seed"
        frames.append(b)
    if frames:
        all_df = pd.concat(frames, ignore_index=True, sort=False)
        all_path = ROOT / "artifacts" / "figures" / "laporan" / "daftar_gambar_lengkap.csv"
        all_df.to_csv(all_path, index=False)
        print("Wrote", all_path, "rows=", len(all_df))


if __name__ == "__main__":
    main()
