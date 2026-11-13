# -*- coding: utf-8 -*-
"""Render Mermaid .mmd files to PNG via npx @mermaid-js/mermaid-cli."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "artifacts" / "figures" / "laporan" / "diagrams"
OUT = ROOT / "artifacts" / "figures" / "laporan"
OUT.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    "01_tahapan_penelitian.mmd": "D01_tahapan_penelitian.png",
    "02_protokol_eksperimen.mmd": "D02_protokol_eksperimen.png",
    "03_pohon_ablasi.mmd": "D03_pohon_ablasi_model.png",
    "04_alur_statistik.mmd": "D04_alur_statistik.png",
}


def main():
    for src_name, dst_name in NAME_MAP.items():
        src = DIAG / src_name
        dst = OUT / dst_name
        if not src.exists():
            raise FileNotFoundError(src)
        cmd = (
            f'npx --yes @mermaid-js/mermaid-cli -i "{src}" -o "{dst}" -b white -s 2'
        )
        print("Rendering", src.name, "->", dst.name)
        subprocess.run(cmd, check=True, cwd=str(ROOT), shell=True)
        print("OK", dst, "bytes=", dst.stat().st_size)


if __name__ == "__main__":
    main()
