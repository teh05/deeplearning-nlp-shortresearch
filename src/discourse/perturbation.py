# -*- coding: utf-8 -*-
"""Mild label-preserving text perturbations for consistency / robustness."""
from __future__ import annotations

import random
import re
from typing import Callable

from discourse.preprocessing import MENTION, URL, preprocess

_WS = re.compile(r"\s+")


def perturb_text(text: str, rng: random.Random | None = None, mode: str = "mild") -> str:
    """Apply a mild orthographic perturbation expected to preserve meaning.

    Modes are intentionally conservative (no synonym of toxic cues):
    - whitespace jitter
    - URL/mention placeholder (same as P1)
    - light character duplication on a random alphabetic token
    """
    rng = rng or random.Random(0)
    t = preprocess(str(text), pipeline="P0")
    if mode == "identity":
        return t

    # Always normalize URL/mention like a mild domain shift
    t = URL.sub("URLTOKEN", t)
    t = MENTION.sub("USERTOKEN", t)

    if rng.random() < 0.5:
        # whitespace jitter: double a random space
        parts = t.split(" ")
        if len(parts) > 2:
            i = rng.randrange(1, len(parts))
            parts[i] = " " + parts[i]
            t = " ".join(parts)
            t = _WS.sub(" ", t).strip()

    if rng.random() < 0.4:
        tokens = re.split(r"(\W+)", t)
        alpha_idx = [i for i, tok in enumerate(tokens) if tok.isalpha() and len(tok) >= 4]
        if alpha_idx:
            i = rng.choice(alpha_idx)
            tok = tokens[i]
            j = rng.randrange(len(tok))
            tokens[i] = tok[:j] + tok[j] + tok[j:]
            t = "".join(tokens)

    return t.strip() or str(text)


def make_augment_fn(seed: int = 0) -> Callable[[str], str]:
    rng = random.Random(seed)

    def _fn(text: str) -> str:
        return perturb_text(text, rng=rng, mode="mild")

    return _fn
