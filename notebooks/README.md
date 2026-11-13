# Notebook Jupyter

## Notebook utama

**`00_IndoDiscourse_Full_Pipeline.ipynb`**

Satu notebook untuk seluruh alur:

1. Prepare  
2. EDA  
3–5. **Inti model & setup** (YAML, `models.py`, dimensi/parameter, hard/soft/loss) lalu train/eval multi-seed opsional (E0, E1a/b/c, E3)  
6. Remaining (E2, E4, E5)  
7. Agregasi compare  
8. Bootstrap CI  
9. Temperature scaling  
10. Efficiency  
11. Ekspor & tampilkan figur  

Step 3–5 mencantumkan **seluruh kode inti** (`src/discourse/*` + YAML) sebagai **komentar penuh** agar terlihat saat scroll/video, tanpa dieksekusi ulang dari cell tersebut. Cell demo (dimensi/loss) dan train opsional tetap aktif via flag.  
Atur flag di sel konfigurasi, lalu **Run All**.

### Mode cepat setelah clone (default disarankan)

Training dimatikan; prepare/EDA/agregasi/figur aktif agar data & visual tetap muncul dari artefak:

```python
RUN_PREPARE = True
RUN_EDA = True
RUN_CORE_MULTISEED = False
RUN_REMAINING = False
RUN_AGGREGATE = True
RUN_EXPORT_FIGURES = True
SHOW_RESULT_FIGS = True
FORCE = False
```

### Mode train ulang penuh

```python
RUN_PREPARE = True
RUN_EDA = True
RUN_CORE_MULTISEED = True
RUN_REMAINING = True
RUN_AGGREGATE = True
RUN_BOOTSTRAP = True
RUN_TEMPERATURE = True
RUN_EFFICIENCY = True
RUN_EXPORT_FIGURES = True
SHOW_RESULT_FIGS = True
SEEDS = [42, 43, 44]
FORCE = True
```

## Notebook modular

| Notebook | Isi |
|---|---|
| `01_prepare_audit_split.ipynb` | Prepare |
| `02_eda.ipynb` | EDA |
| `03_train_eval_gpu.ipynb` | Train seed tunggal |

Untuk alur lengkap, pakai `00_IndoDiscourse_Full_Pipeline.ipynb`.
