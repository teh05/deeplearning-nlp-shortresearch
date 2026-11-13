# IndoDiscourse Short Research

Disagreement-preserving multitask learning untuk toksisitas & polarisasi Bahasa Indonesia (dataset IndoDiscourse).

Repositori ini berisi kode inti agar orang lain bisa **clone → install → Run All notebook** dan melihat data/metrik/visual yang sama: prepare, EDA, preprocessing, training (opsional), evaluasi, dan output artefak.

## Isi

| Path | Keterangan |
|---|---|
| `src/discourse/` | Prepare, EDA, preprocessing, model, train, evaluate, agregasi, ekspor figur |
| `configs/` | YAML eksperimen E0–E5 |
| `notebooks/` | Pipeline Jupyter (`00_IndoDiscourse_Full_Pipeline.ipynb`) |
| `IndoDiscourse/` | Dataset sumber |
| `data/processed/`, `data/splits/` | Data siap latih + split beku |
| `artifacts/` | Metrik, prediksi, figur, tabel, audit (tanpa checkpoint besar) |
| `Spesifikasi_Penelitian_IndoDiscourse.md` | Catatan rancangan penelitian |

## Setup

```powershell
pip install -e .
```

GPU dipakai otomatis jika `torch.cuda.is_available()`.

## Menjalankan

Buka `notebooks/00_IndoDiscourse_Full_Pipeline.ipynb`, lalu **Run All**.

Alur yang dijalankan:

1. **Prepare** — audit, soft target, split  
2. **EDA** — statistik & figur eksplorasi  
3. **Preprocessing / training** — E0–E5 (opsional; default memakai artefak yang sudah ada)  
4. **Agregasi & evaluasi** — mean±std, pairwise, kalibrasi  
5. **Output** — tabel metrik + figur (confusion, PR, reliability, banding)

Flag penting di sel konfigurasi:

- `RUN_CORE_MULTISEED` / `RUN_REMAINING` → `True` untuk train ulang  
- `FORCE` → `True` untuk paksa rebuild  
- Default `False` pada training: cepat setelah clone, tetap menampilkan hasil existing  

Detail flag: `notebooks/README.md`.

## Dataset

`IndoDiscourse/indotoxic2024_annotated_data_v2_final.jsonl`

## Eksperimen

- E0 TF-IDF + LR  
- E1a TextCNN  
- E1b BiLSTM-Attention  
- E1c IndoBERTweet hard MTL  
- E2 STL matched  
- E3 Soft MTL  
- E4 Consistency  
- E5 Soft + preprocessing P1/P2  

Multi-seed: 42, 43, 44.

## Catatan

Checkpoint (`artifacts/checkpoints/`, ~10GB) tidak disertakan. Setelah train ulang, checkpoint akan dibuat lokal di folder tersebut.
