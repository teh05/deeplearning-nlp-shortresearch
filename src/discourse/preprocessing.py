from __future__ import annotations

import re
import unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"(?<!\w)@\w+")


def preprocess(text, pipeline="P0", slang=None):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    if pipeline in {"P1", "P2"}:
        text = URL.sub("URLTOKEN", text)
        text = MENTION.sub("USERTOKEN", text)
    if pipeline == "P2":
        if slang is None:
            raise ValueError("P2 requires an audited slang dictionary")
        tokens = re.split(r"(\W+)", text)
        text = "".join(slang.get(t.lower(), t) for t in tokens)
    return text


def apply_preprocessing(df, pipeline="P0", slang=None, text_col="text_raw"):
    out = df.copy()
    out["text_proc"] = out[text_col].map(lambda t: preprocess(t, pipeline=pipeline, slang=slang))
    out["pipeline"] = pipeline
    return out
