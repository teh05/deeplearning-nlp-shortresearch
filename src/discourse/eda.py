from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from discourse.utils import ensure_dir


def _save(fig, path: Path):
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(df: pd.DataFrame, fig_dir: str | Path, table_dir: str | Path) -> dict:
    fig_dir = ensure_dir(fig_dir)
    table_dir = ensure_dir(table_dir)
    sns.set_theme(style="whitegrid")
    summary = {}

    # G01 availability
    avail = pd.DataFrame(
        {
            "subset": ["tox_only", "pol_only", "both", "either"],
            "count": [
                int(((df.mask_tox) & (~df.mask_pol)).sum()),
                int(((~df.mask_tox) & (df.mask_pol)).sum()),
                int(((df.mask_tox) & (df.mask_pol)).sum()),
                int(((df.mask_tox) | (df.mask_pol)).sum()),
            ],
        }
    )
    avail.to_csv(table_dir / "G01_target_availability.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=avail, x="subset", y="count", ax=ax, color="#3b82f6")
    ax.set_title("G01 Target availability")
    _save(fig, fig_dir / "G01_target_availability.png")
    summary["G01"] = avail.to_dict(orient="records")

    # G02 majority labels
    rows = []
    for task, hard_col in [("toxicity", "hard_tox"), ("polarization", "hard_pol")]:
        sub = df[df[hard_col].notna()]
        vc = sub[hard_col].value_counts(dropna=False).rename({0: "neg", 1: "pos"})
        for lab, cnt in vc.items():
            rows.append({"task": task, "label": lab, "count": int(cnt)})
    maj = pd.DataFrame(rows)
    maj.to_csv(table_dir / "G02_majority_labels.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=maj, x="task", y="count", hue="label", ax=ax)
    ax.set_title("G02 Majority label distribution (non-tie)")
    _save(fig, fig_dir / "G02_majority_labels.png")

    # G03 joint labels
    both = df[df.hard_tox.notna() & df.hard_pol.notna()].copy()
    ct = pd.crosstab(both["hard_tox"], both["hard_pol"])
    ct.to_csv(table_dir / "G03_joint_labels.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("polarization hard")
    ax.set_ylabel("toxicity hard")
    ax.set_title("G03 Joint majority labels")
    _save(fig, fig_dir / "G03_joint_labels.png")

    # G04 annotator counts
    n_rows = []
    for task, n_col, mask_col in [
        ("toxicity", "n_tox", "mask_tox"),
        ("polarization", "n_pol", "mask_pol"),
    ]:
        vc = df.loc[df[mask_col], n_col].value_counts().sort_index()
        for n_ann, cnt in vc.items():
            n_rows.append({"task": task, "n_annotators": int(n_ann), "count": int(cnt)})
    nann = pd.DataFrame(n_rows)
    nann.to_csv(table_dir / "G04_annotator_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=nann, x="n_annotators", y="count", hue="task", ax=ax)
    ax.set_title("G04 Annotator count distribution")
    _save(fig, fig_dir / "G04_annotator_counts.png")

    # G05 soft label distribution
    soft_rows = []
    for task, q_col, mask_col in [
        ("toxicity", "q_tox", "mask_tox"),
        ("polarization", "q_pol", "mask_pol"),
    ]:
        qs = df.loc[df[mask_col], q_col].dropna()
        soft_rows.append(
            {
                "task": task,
                "mean_q": float(qs.mean()),
                "std_q": float(qs.std()),
                "n": int(len(qs)),
            }
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(qs, bins=11, color="#6366f1", edgecolor="white")
        ax.set_title(f"G05 Soft label distribution — {task}")
        ax.set_xlabel("q (positive vote proportion)")
        _save(fig, fig_dir / f"G05_soft_{task}.png")
    pd.DataFrame(soft_rows).to_csv(table_dir / "G05_soft_summary.csv", index=False)

    # G06 entropy vs n annotators
    ent_rows = []
    for task, ent_col, n_col, mask_col in [
        ("toxicity", "entropy_tox", "n_tox", "mask_tox"),
        ("polarization", "entropy_pol", "n_pol", "mask_pol"),
    ]:
        sub = df.loc[df[mask_col], [ent_col, n_col]].dropna()
        g = sub.groupby(n_col)[ent_col].mean().reset_index()
        g.columns = ["n_annotators", "mean_entropy"]
        g["task"] = task
        ent_rows.append(g)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=sub, x=n_col, y=ent_col, ax=ax, color="#14b8a6")
        ax.set_title(f"G06 Entropy by annotator count — {task}")
        _save(fig, fig_dir / f"G06_entropy_{task}.png")
    pd.concat(ent_rows, ignore_index=True).to_csv(table_dir / "G06_entropy_by_n.csv", index=False)

    # G07 text length (chars / words)
    df = df.copy()
    df["char_len"] = df["text_raw"].astype(str).str.len()
    df["word_len"] = df["text_raw"].astype(str).str.split().str.len()
    len_sum = df[["char_len", "word_len"]].describe().T
    len_sum.to_csv(table_dir / "G07_text_length.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["char_len"].clip(upper=df["char_len"].quantile(0.99)), bins=40, color="#f59e0b")
    axes[0].set_title("G07 Char length")
    axes[1].hist(df["word_len"].clip(upper=df["word_len"].quantile(0.99)), bins=40, color="#ef4444")
    axes[1].set_title("G07 Word length")
    _save(fig, fig_dir / "G07_text_length.png")

    # G09 URL/mention/emoji rough flags
    import re

    text = df["text_raw"].fillna("").astype(str)
    texts = text.tolist()
    emoji_re = re.compile("[\U0001F300-\U0001FAFF]")
    flags = pd.DataFrame(
        {
            "feature": ["has_url", "has_mention", "has_emoji_like"],
            "count": [
                int(text.str.contains(r"https?://|www\.", regex=True, na=False).sum()),
                int(text.str.contains(r"(?<!\w)@\w+", regex=True, na=False).sum()),
                int(sum(bool(emoji_re.search(str(t))) for t in texts)),
            ],
        }
    )
    flags.to_csv(table_dir / "G09_surface_features.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=flags, x="feature", y="count", ax=ax, color="#8b5cf6")
    ax.set_title("G09 URL / mention / emoji-like")
    _save(fig, fig_dir / "G09_surface_features.png")

    # G10 split distributions if present
    if "split" in df.columns:
        split_rows = []
        for split_name, g in df.groupby("split"):
            split_rows.append(
                {
                    "split": split_name,
                    "n": len(g),
                    "tox_pos_rate": float(g.loc[g.hard_tox.notna(), "hard_tox"].mean())
                    if g.hard_tox.notna().any()
                    else None,
                    "pol_pos_rate": float(g.loc[g.hard_pol.notna(), "hard_pol"].mean())
                    if g.hard_pol.notna().any()
                    else None,
                    "mean_char_len": float(g["char_len"].mean()),
                }
            )
        split_df = pd.DataFrame(split_rows)
        split_df.to_csv(table_dir / "G10_split_audit.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=split_df, x="split", y="n", ax=ax, color="#0ea5e9")
        ax.set_title("G10 Split sizes")
        _save(fig, fig_dir / "G10_split_sizes.png")

    summary["n_rows"] = int(len(df))
    return summary
