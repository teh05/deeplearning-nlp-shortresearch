# -*- coding: utf-8 -*-
"""Copy evaluation figures into artifacts/figures/laporan with short readable names."""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "figures"
OUT = ROOT / "artifacts" / "figures" / "laporan"
OUT.mkdir(parents=True, exist_ok=True)

# short model names (no experiment codes)
MODEL = {
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
KIND = {
    "reliability": "kalibrasi",
    "cm": "confusion",
    "pr": "precision-recall",
}

EDA_MAP = {
    "G01_target_availability.png": "01_ketersediaan_label.png",
    "G02_majority_labels.png": "02_label_mayoritas.png",
    "G03_joint_labels.png": "03_label_bersama_tox_pol.png",
    "G04_annotator_counts.png": "04_jumlah_annotator.png",
    "G05_soft_toxicity.png": "05_distribusi_soft_toksisitas.png",
    "G05_soft_polarization.png": "05_distribusi_soft_polarisasi.png",
    "G06_entropy_toxicity.png": "06_entropi_toksisitas.png",
    "G06_entropy_polarization.png": "06_entropi_polarisasi.png",
    "G07_text_length.png": "07_panjang_teks.png",
    "G09_surface_features.png": "08_fitur_permukaan_url_mention.png",
    "G10_split_sizes.png": "09_ukuran_split.png",
}


def copy(src: Path, dst_name: str, rows: list[dict], note: str = "") -> None:
    dst = OUT / dst_name
    shutil.copy2(src, dst)
    rows.append(
        {
            "nama_laporan": dst_name,
            "sumber": str(src.relative_to(ROOT)).replace("\\", "/"),
            "keterangan": note,
        }
    )


def main():
    rows: list[dict] = []

    # EDA
    for old, new in EDA_MAP.items():
        src = SRC / old
        if src.exists():
            copy(src, new, rows, "EDA")

    # Root model figures (legacy seed-42 style exports)
    for path in sorted(SRC.glob("*.png")):
        name = path.name
        if name.startswith("G"):
            continue
        # patterns:
        # E1a_textcnn_tox_reliability.png
        # E1a_textcnn_test_tox_cm.png
        for exp, nice in MODEL.items():
            if not name.startswith(exp):
                continue
            rest = name[len(exp) + 1 :]  # after exp_
            # reliability: {task}_reliability.png
            if rest.endswith("_reliability.png"):
                task = rest.replace("_reliability.png", "")
                if task in TASK:
                    new = f"{nice}_{TASK[task]}_kalibrasi.png"
                    copy(path, new, rows, "kalibrasi")
                break
            # test metrics: test_{task}_{cm|pr}.png
            if rest.startswith("test_"):
                parts = rest.replace(".png", "").split("_")  # test, task, kind
                if len(parts) == 3 and parts[1] in TASK and parts[2] in KIND:
                    new = f"{nice}_{TASK[parts[1]]}_{KIND[parts[2]]}.png"
                    copy(path, new, rows, KIND[parts[2]])
                break

    # Seeded key models seed42 for E2/E4/E5 if root missing
    seeded_wanted = [
        ("E2_stl_tox", "seed_42", "tox"),
        ("E2_stl_pol", "seed_42", "pol"),
        ("E4_consistency", "seed_42", None),
        ("E5_P1_soft_mtl", "seed_42", None),
        ("E5_P2_soft_mtl", "seed_42", None),
    ]
    for exp, seed_folder, only_task in [
        ("E2_stl_tox", "E2_stl_tox_seed42", "tox"),
        ("E2_stl_pol", "E2_stl_pol_seed42", "pol"),
        ("E4_consistency", "E4_consistency_seed42", None),
        ("E5_P1_soft_mtl", "E5_P1_soft_mtl_seed42", None),
        ("E5_P2_soft_mtl", "E5_P2_soft_mtl_seed42", None),
        ("E3_soft_mtl", "E3_soft_mtl_seed42", None),
        ("E1c_indobertweet", "E1c_indobertweet_seed42", None),
    ]:
        nice = MODEL[exp]
        folder = SRC / "seeded" / seed_folder
        if not folder.exists():
            continue
        for task in ("tox", "pol"):
            if only_task and task != only_task:
                continue
            for kind, kind_name in (("cm", "confusion"), ("pr", "precision-recall")):
                # filename like E4_consistency_seed42_test_tox_cm.png
                matches = list(folder.glob(f"*_{task}_{kind}.png"))
                if not matches:
                    continue
                new = f"{nice}_{TASK[task]}_{kind_name}.png"
                # don't overwrite if already from root with same name unless missing
                if not (OUT / new).exists():
                    copy(matches[0], new, rows, f"{kind_name} seed42")

    # Comparison chart from aggregates
    wide = pd.read_csv(ROOT / "artifacts" / "metrics" / "aggregates" / "summary_mean_std_wide.csv")
    order = [
        ("E0_tfidf_lr", "TF-IDF+LR"),
        ("E1a_textcnn", "TextCNN"),
        ("E1b_bilstm", "BiLSTM"),
        ("E1c_indobertweet", "IndoBERTweet-Hard"),
        ("E2_stl", "IndoBERTweet-STL"),
        ("E3_soft_mtl", "IndoBERTweet-Soft"),
        ("E4_consistency", "Consistency"),
        ("E5_P1_soft_mtl", "Soft-P1"),
        ("E5_P2_soft_mtl", "Soft-P2"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task, title in zip(axes, ("tox", "pol"), ("Toksisitas", "Polarisasi")):
        labels, means, stds = [], [], []
        for exp, lab in order:
            hit = wide[(wide.experiment_id == exp) & (wide.task == task)]
            if hit.empty:
                continue
            labels.append(lab)
            means.append(float(hit.iloc[0].macro_f1_mean))
            stds.append(float(hit.iloc[0].macro_f1_std or 0.0))
        ax.bar(range(len(labels)), means, yerr=stds, capsize=3, color="#4C78A8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0.55, 0.75)
        ax.set_ylabel("macro-F1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Perbandingan macro-F1 (mean±std, 3 seed)", fontsize=12)
    fig.tight_layout()
    cmp_path = OUT / "10_perbandingan_macroF1_semua_model.png"
    fig.savefig(cmp_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    rows.append(
        {
            "nama_laporan": cmp_path.name,
            "sumber": "generated_from_aggregates",
            "keterangan": "ringkasan perbandingan model",
        }
    )

    # Robustness chart
    rob = pd.read_csv(ROOT / "artifacts" / "metrics" / "aggregates" / "robustness_perturbed_mean_std.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, deltas, errs = [], [], []
    name_map = {"E3_soft_mtl": "Soft", "E4_consistency": "Consistency"}
    for _, r in rob.iterrows():
        labels.append(f"{name_map.get(r.experiment_id, r.experiment_id)}-{r.task}")
        deltas.append(float(r.delta_vs_clean_mean))
        errs.append(float(r.delta_vs_clean_std or 0.0))
    ax.bar(range(len(labels)), deltas, yerr=errs, capsize=3, color="#F58518")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Δ macro-F1 (perturbed − clean)")
    ax.set_title("Robustness: penurunan F1 setelah perturbasi ringan")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    rob_path = OUT / "11_robustness_delta_F1.png"
    fig.savefig(rob_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    rows.append({"nama_laporan": rob_path.name, "sumber": "generated_from_robustness", "keterangan": "robustness"})

    map_df = pd.DataFrame(rows).drop_duplicates(subset=["nama_laporan"], keep="last")
    map_df.to_csv(OUT / "daftar_gambar_laporan.csv", index=False)
    print(f"Wrote {len(map_df)} figures to {OUT}")
    print(map_df[["nama_laporan", "keterangan"]].to_string(index=False))


if __name__ == "__main__":
    main()
