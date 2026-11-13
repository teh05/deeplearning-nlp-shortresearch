from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from discourse.tokenization import encode_words


class DiscourseDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str = "text_proc",
        target_mode: str = "hard",
        tokenizer=None,
        vocab: dict[str, int] | None = None,
        max_length: int = 128,
        task_mode: str = "multitask",  # multitask | tox | pol
        augment_fn=None,
    ):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.target_mode = target_mode
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self.task_mode = task_mode
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.df)

    def _targets_masks(self, row) -> tuple[np.ndarray, np.ndarray]:
        if self.target_mode == "hard":
            t_tox = row["hard_tox"]
            t_pol = row["hard_pol"]
            # hard mode: only non-tie / available majority
            m_tox = bool(row["mask_tox"]) and pd.notna(t_tox)
            m_pol = bool(row["mask_pol"]) and pd.notna(t_pol)
            y_tox = float(t_tox) if m_tox else 0.0
            y_pol = float(t_pol) if m_pol else 0.0
        else:
            m_tox = bool(row["mask_tox"]) and pd.notna(row["q_tox"])
            m_pol = bool(row["mask_pol"]) and pd.notna(row["q_pol"])
            y_tox = float(row["q_tox"]) if m_tox else 0.0
            y_pol = float(row["q_pol"]) if m_pol else 0.0

        if self.task_mode == "tox":
            m_pol = False
        elif self.task_mode == "pol":
            m_tox = False

        targets = np.array([y_tox, y_pol], dtype=np.float32)
        masks = np.array([float(m_tox), float(m_pol)], dtype=np.float32)
        return targets, masks

    def _encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        ids, attn = encode_words(text, self.vocab, self.max_length)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        input_ids, attention_mask = self._encode(text)

        targets, masks = self._targets_masks(row)
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor(targets),
            "target_masks": torch.tensor(masks),
            "text_id": str(row["text_id"]),
            "n_tox": int(row.get("n_tox", 0) or 0),
            "n_pol": int(row.get("n_pol", 0) or 0),
            "q_tox": row.get("q_tox"),
            "q_pol": row.get("q_pol"),
            "hard_tox": row.get("hard_tox"),
            "hard_pol": row.get("hard_pol"),
        }
        if self.augment_fn is not None:
            aug_text = self.augment_fn(text)
            aug_ids, aug_attn = self._encode(aug_text)
            item["aug_input_ids"] = aug_ids
            item["aug_attention_mask"] = aug_attn
        return item


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    meta_keys = ["text_id", "n_tox", "n_pol", "q_tox", "q_pol", "hard_tox", "hard_pol"]
    out: dict[str, Any] = {}
    tensor_keys = ["input_ids", "attention_mask", "targets", "target_masks"]
    if "aug_input_ids" in batch[0]:
        tensor_keys.extend(["aug_input_ids", "aug_attention_mask"])
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in meta_keys:
        out[k] = [b[k] for b in batch]
    return out
