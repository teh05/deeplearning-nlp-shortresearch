from __future__ import annotations

import shutil
from pathlib import Path

from discourse.utils import ensure_dir, project_root, save_json, sha256_file

REPO = "Exqrch/IndoDiscourse"
FILENAME = "indotoxic2024_annotated_data_v2_final.jsonl"


def prepare_local_dataset(
    local_src: str | Path | None = None,
    out_dir: str | Path = "data/raw",
) -> Path:
    """Copy local IndoDiscourse file into data/raw and write manifest."""
    root = project_root()
    if local_src is None:
        local_src = root / "IndoDiscourse" / FILENAME
    else:
        local_src = Path(local_src)
        if not local_src.is_absolute():
            local_src = root / local_src

    if not local_src.exists():
        raise FileNotFoundError(f"Dataset not found: {local_src}")

    out = ensure_dir(root / out_dir)
    dest = out / FILENAME
    if local_src.resolve() != dest.resolve():
        shutil.copy2(local_src, dest)

    manifest = {
        "repo": REPO,
        "source": "local_copy",
        "filename": FILENAME,
        "local_src": str(local_src.resolve()),
        "raw_path": str(dest.resolve()),
        "sha256": sha256_file(dest),
        "n_bytes": dest.stat().st_size,
    }
    save_json(manifest, root / "artifacts" / "manifests" / "dataset.json")
    return dest
