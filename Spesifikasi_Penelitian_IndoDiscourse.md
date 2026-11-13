# Spesifikasi Penelitian IndoDiscourse

## Status dokumen

Rancangan penelitian, metodologi, struktur proyek, dan instruksi implementasi. Belum merupakan laporan hasil eksperimen. Jumlah sampel valid, grafik EDA, dan skor model wajib dihasilkan dari eksekusi aktual. Kode berikut merupakan kerangka inti, bukan aplikasi end-to-end yang telah diuji.

## Judul kerja

Disagreement-Preserving Multi-Task Learning untuk Analisis Toksisitas dan Polarisasi Bahasa Indonesia: Studi Pengaruh Normalisasi Teks, Agregasi Anotasi, dan Robustness pada IndoDiscourse.

## 1. Landasan dan ruang lingkup

IndoDiscourse merupakan dataset teks Indonesia untuk klasifikasi toksisitas dan polarisasi. Dokumentasi menyebut 28.477 teks dan sekitar 44,6% teks dengan beberapa anotasi. Jumlah sampel aktual untuk masing-masing eksperimen wajib diperiksa dari versi data yang digunakan. Paper asal telah mempelajari hubungan kedua tugas, sehingga multitask learning saja bukan novelty baru (Susanto et al., 2025).

- Dataset resmi: https://huggingface.co/datasets/Exqrch/IndoDiscourse
- Dokumentasi: https://huggingface.co/datasets/Exqrch/IndoDiscourse/blob/main/README.md
- Paper: https://aclanthology.org/2025.findings-acl.966/
- Input inferensi: teks saja.
- Target training: hard label atau distribusi anotasi.
- Backbone: TextCNN, BiLSTM-Attention, IndoBERTweet.
- Model usulan: IndoBERTweet multitask + soft labels + consistency.
- Preprocessing: minimal cleanup, placeholder, normalisasi slang konservatif.
- Evaluasi: klasifikasi, distribusi anotasi, kalibrasi, robustness, efisiensi.
- Di luar lingkup: pemrofilan individu, prediksi demografi, dan deployment moderasi otomatis.

Informasi demografi tidak digunakan sebagai fitur proyek inti. Periksa lisensi serta batas penggunaan pada revision yang dikunci.

## 2. Rumusan masalah, tujuan, dan hipotesis

### RM1 — Kualitas data dan preprocessing

Bagaimana karakteristik data, struktur anotasi, duplikasi, bahasa informal, dan panjang teks memengaruhi preprocessing serta evaluasi?

Subpertanyaan:
1. Berapa teks valid untuk masing-masing task dan kedua task?
2. Berapa teks dengan satu dan beberapa anotator?
3. Berapa exact duplicate dan kandidat near-duplicate?
4. Apakah normalisasi slang membantu atau menghapus sinyal?
5. Berapa teks yang terpotong pada panjang input tertentu?

Output: audit, EDA, keputusan preprocessing, dan ablation P0–P2.

### RM2 — Perbedaan arsitektur

Bagaimana performa dan efisiensi TextCNN, BiLSTM-Attention, serta IndoBERTweet pada kedua task?

Output: perbandingan performa, biaya komputasi, dan kesalahan antararsitektur.

### RM3 — Single-task versus multitask

Apakah shared representation membantu kedua task atau menyebabkan negative transfer?

Output: model single-task dan multitask dengan backbone serta sampel terkontrol.

### RM4 — Hard label versus soft label

Apakah mempertahankan distribusi anotasi menghasilkan probabilitas yang lebih sesuai dengan penilaian manusia dibanding majority vote?

Analisis dipisahkan menurut jumlah anotator dan disagreement. Output: evaluasi kelas dan distribusi.

### RM5 — Pengembangan metode

Apakah consistency regularization pada perturbasi yang diaudit meningkatkan robustness tanpa merusak performa clean?

Output: ablation dan clean-versus-perturbed evaluation.

### RM6 — Keterandalan prediksi

Apakah model terkalibrasi, terutama pada teks dengan disagreement tinggi?

Output: reliability diagram, ECE, Brier score, dan confident-error analysis.

| Rumusan | Tujuan | Hipotesis yang diuji |
|---|---|---|
| RM1 | Menentukan preprocessing berbasis data | Efek preprocessing bergantung pada transformasi dan model |
| RM2 | Mengukur trade-off arsitektur | Ranking performa dan efisiensi tidak selalu sama |
| RM3 | Mengukur transfer antartask | Multitask dapat membantu satu task dan mengganggu task lain |
| RM4 | Mempertahankan informasi anotator | Soft label memperbaiki kesesuaian distribusi, belum tentu F1 |
| RM5 | Meningkatkan robustness | Consistency mengurangi penurunan performa pada gangguan ringan |
| RM6 | Mengukur probabilitas | Teks ambigu dapat menimbulkan overconfidence |

Hipotesis bukan hasil yang dijanjikan.

## 3. Kerangka metodologi

```text
Kajian literatur dan preregistrasi
→ Unduh data, kunci versi, manifest
→ Audit skema, anotasi, teks, duplikasi
→ Target hard/soft dan metadata disagreement
→ Group split dan bekukan test
→ EDA training dan audit split
→ Preprocessing dan tokenisasi
→ Training baseline arsitektur
→ Single-task versus multitask
→ Hard-label versus soft-label
→ Consistency regularization
→ Validation: tuning dan threshold
→ Final test: clean, robustness, kalibrasi, efisiensi
→ Statistik, error analysis, output evaluasi
```

Aturan wajib:
1. Split sebelum augmentasi.
2. Duplikat dan near-duplicate terverifikasi berada pada satu split.
3. Missing target tidak menjadi negatif.
4. Satu anotator tidak disebut konsensus tinggi.
5. Test tidak digunakan untuk tuning.
6. Evaluasi tiap perbandingan memakai sampel yang sama.
7. Pisahkan pengaruh preprocessing, target, dan tambahan data.
8. ID anotator dan gold label bukan input inferensi.
9. Jangan mengarang hasil atau grafik.

## 4. Struktur proyek

```text
indodiscourse-research/
├── README.md
├── AI_TASKS.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── textcnn.yaml
│   ├── bilstm.yaml
│   ├── indobertweet.yaml
│   ├── multitask_soft.yaml
│   └── multitask_consistency.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── splits/
│   └── annotations/
├── notebooks/
│   ├── 01_schema_audit.ipynb
│   ├── 02_eda.ipynb
│   └── 03_error_analysis.ipynb
├── src/discourse/
│   ├── __init__.py
│   ├── download.py
│   ├── schema.py
│   ├── audit.py
│   ├── targets.py
│   ├── duplicates.py
│   ├── split.py
│   ├── preprocessing.py
│   ├── tokenization.py
│   ├── datasets.py
│   ├── eda.py
│   ├── models.py
│   ├── losses.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── calibration.py
│   ├── robustness.py
│   ├── statistics.py
│   └── utils.py
├── notebooks/
│   └── 00_IndoDiscourse_Full_Pipeline.ipynb
├── tests/
│   ├── test_schema.py
│   ├── test_targets.py
│   ├── test_splits.py
│   ├── test_preprocessing.py
│   └── test_losses.py
└── artifacts/
    ├── manifests/
    ├── audits/
    ├── figures/
    ├── checkpoints/
    ├── predictions/
    └── metrics/
```

Notebook untuk eksplorasi; logika final di `src/`. Struktur ini merupakan target implementasi, bukan daftar file yang telah dibuat.

Dependensi: torch, transformers, huggingface-hub, pandas, numpy, scikit-learn, matplotlib, seaborn, pyarrow, pyyaml, pytest. Instal PyTorch yang sesuai GPU/driver, kemudian kunci versi environment yang terbukti berfungsi.

## 5. Kontrak data

Periksa skema aktual sebelum menetapkan mapping kolom. Dokumentasi menyebut `text`, `toxicity`, `polarization`, dan `annotators_id`; jangan menebak alias atau menyembunyikan ketidaksesuaian.

| Kolom hasil | Isi |
|---|---|
| text_id | ID internal stabil |
| source_ids | ID/baris asal untuk provenance |
| text_raw | Teks asli |
| group_id | Kelompok duplikat/near-duplicate |
| votes_tox, votes_pol | Daftar suara biner valid |
| n_tox, n_pol | Jumlah penilaian |
| q_tox, q_pol | Proporsi positif |
| hard_tox, hard_pol | Mayoritas; kosong saat tie |
| mask_tox, mask_pol | Ketersediaan target |
| entropy_tox, entropy_pol | Entropi distribusi suara |
| multi_tox, multi_pol | Minimal dua penilaian |
| split | Train/validation/test |

Missing q tetap kosong pada tabel analisis. Nilai pengganti 0 hanya pada tensor bersama task mask.

### Kode pembentukan target

```python
# src/discourse/targets.py
import ast
import json
import math
import pandas as pd

def parse_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError("Expected list")
        return parsed
    raise TypeError(f"Unsupported annotation type: {type(value)}")

def summarize_votes(values):
    votes = parse_list(values)
    if any(v not in (0, 1) for v in votes):
        raise ValueError("Votes must contain only 0 or 1")
    n = len(votes)
    if n == 0:
        return dict(n=0, q=None, hard=None, mask=False,
                    entropy=None, multi=False)
    q = sum(votes) / n
    entropy = 0.0
    if 0 < q < 1:
        entropy = -q * math.log2(q) - (1-q) * math.log2(1-q)
    return dict(n=n, q=q, hard=None if q == 0.5 else int(q > 0.5),
                mask=True, entropy=entropy, multi=n >= 2)

def add_targets(df):
    out = df.copy()
    for source, suffix in [("votes_tox", "tox"), ("votes_pol", "pol")]:
        stats = out[source].map(summarize_votes)
        stats = pd.DataFrame(stats.tolist(), index=out.index)
        stats.columns = [f"{c}_{suffix}" for c in stats.columns]
        out = pd.concat([out, stats], axis=1)
    return out
```

Ingestion wajib menangani nilai null sesuai format aktual sebelum memanggil fungsi ini. Jangan menghapus missing votes di tengah daftar tanpa menjaga pasangan anotatornya. Entropi nol pada satu anotator bukan bukti teks tidak ambigu.

## 6. Pengambilan dan audit data

```python
# src/discourse/download.py
from pathlib import Path
import hashlib
import json
from huggingface_hub import HfApi, hf_hub_download

REPO = "Exqrch/IndoDiscourse"
FILENAME = "indotoxic2024_annotated_data_v2_final.jsonl"

def download_dataset(out_dir="data/raw", revision=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved = HfApi().dataset_info(REPO, revision=revision).sha
    path = hf_hub_download(
        repo_id=REPO, repo_type="dataset", filename=FILENAME,
        revision=resolved, local_dir=out)
    manifest = {
        "repo": REPO, "revision": resolved, "filename": FILENAME,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    dest = Path("artifacts/manifests/dataset.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
```

```yaml
# configs/data.yaml
 dataset:
   repo: Exqrch/IndoDiscourse
   revision: null
 columns:
   text: text
   toxicity: toxicity
   polarization: polarization
   annotators: annotators_id
 split:
   seed: 42
   train: 0.70
   validation: 0.15
   test: 0.15
 preprocessing:
   pipeline: P0
   slang_dictionary: null
```

Audit:
- Kolom dan tipe wajib tersedia.
- Teks tidak kosong.
- Anotasi berbentuk daftar biner.
- Struktur daftar anotator dan penilaiannya sesuai.
- Tidak ada anotasi yang diam-diam dibuang.
- Duplikat mempunyai provenance; jangan menghitung ulang suara duplikat dari anotator sama.
- Konflik anotasi dicatat.
- Jumlah sebelum/sesudah filter dilaporkan.

Near-duplicate: buat kandidat dengan character n-grams atau kemiripan teks, verifikasi, lalu kelompokkan. Jangan menghapus otomatis hanya berdasarkan skor kemiripan.

## 7. EDA dan definisi grafik

| ID | Grafik | Tujuan |
|---|---|---|
| G01 | Ketersediaan target | Sampel per task dan kedua task |
| G02 | Distribusi majority label | Imbalance per task, non-tie |
| G03 | Kombinasi dua label | Hubungan empiris task |
| G04 | Jumlah anotator | Dukungan informasi disagreement |
| G05 | Distribusi soft label | Proporsi suara |
| G06 | Entropi menurut jumlah anotator | Pengaruh ukuran panel |
| G07 | Panjang token per tokenizer | Menentukan max length |
| G08 | Truncation pada 128/256 token | Konteks yang terpotong |
| G09 | URL/mention/emoji | Kebutuhan preprocessing |
| G10 | Distribusi antar-split | Audit label dan panjang |
| G11 | PR curves | Precision–recall trade-off |
| G12 | Confusion matrix | Jenis kesalahan |
| G13 | Reliability diagram | Over/underconfidence |
| G14 | Clean versus perturbed | Robustness |
| G15 | Ablation dengan CI | Kontribusi metode |
| G16 | Performa versus latency/VRAM | Trade-off efisiensi |

Audit integritas boleh seluruh korpus. Eksplorasi yang memengaruhi model dilakukan pada training. Distribusi validation/test hanya untuk audit, bukan untuk menyusun aturan berdasarkan kesalahan test.

Narasi tiap grafik:
1. Temuan aktual.
2. Risiko terhadap model.
3. Keputusan metodologis.
4. Eksperimen pembanding.
5. Hasil aktual.
6. Batas interpretasi.

## 8. Preprocessing

| Pipeline | Transformasi |
|---|---|
| P0 | Unicode NFC dan whitespace |
| P1 | P0 + URL/mention placeholder |
| P2 | P1 + kamus slang konservatif |
| P3 | Pipeline terpilih + perturbasi training teraudit |

```python
# src/discourse/preprocessing.py
import re
import unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"(?<!\w)@\w+")

def preprocess(text, pipeline="P0", slang=None):
    text = unicodedata.normalize("NFC", text)
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
```

Jangan default stemming atau stopword removal. Pertahankan negasi. Emoji/tanda baca tidak dibuang tanpa pengujian. Casing mengikuti tokenizer. Kamus tidak disusun dari test. Simpan teks mentah dan hasil transformasi.

## 9. Anotasi tambahan

Dataset sudah beranotasi; tidak perlu melabel ulang seluruh data. Anotasi tambahan untuk audit, bukan mengganti gold label sepihak.

### Audit preprocessing

```text
text_id, pipeline, original_text, transformed_text,
meaning_preserved, toxicity_cue_changed,
polarization_cue_changed, reviewer_id, notes
```

- Makna terjaga: dapat digunakan.
- Tidak pasti: review.
- Berubah: tolak transformasi.

### Audit robustness

Jika memungkinkan dua penilai independen memeriksa kesetaraan makna dan kemungkinan perubahan label, lalu adjudication. Jika hanya satu penilai, laporkan single-reviewer audit; jangan mengklaim inter-annotator agreement.

### Kategori error

Negasi; kutipan; sarkasme; serangan implisit; penyebutan kelompok tanpa serangan; slang/typo; konteks kurang; truncation; ambiguitas anotasi.

AI boleh membantu menyarankan kategori tetapi tidak menjadi ground truth tanpa validasi manusia.

## 10. Split bebas leakage

```python
# src/discourse/split.py
from sklearn.model_selection import GroupShuffleSplit

def make_splits(df, seed=42):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15,
                              random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=df["group_id"]))
    trainval = df.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85,
                              random_state=seed)
    train_idx, val_idx = next(inner.split(
        trainval, groups=trainval["group_id"]))
    parts = {
        "train": trainval.iloc[train_idx].copy(),
        "validation": trainval.iloc[val_idx].copy(),
        "test": df.iloc[test_idx].copy()}
    groups = {k: set(v["group_id"]) for k, v in parts.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    return parts
```

Contoh ini belum stratified. Audit distribusi; bila perlu grouped stratification, tetapkan prosedurnya sebelumnya. Jangan memilih split berdasarkan skor test. Simpan ID dan hash split.

## 11. Model dan rasional baseline

| Model | Fungsi |
|---|---|
| TF-IDF + Logistic Regression | Sanity baseline non-DL |
| TextCNN | Pola lokal |
| BiLSTM-Attention | Konteks sekuens |
| IndoBERTweet | Transfer learning domain media sosial |
| Multitask soft-label | Informasi distribusi anotasi |
| Model + consistency | Pengembangan robustness |

TextCNN/BiLSTM bukan model mutakhir. Variasi keluarga arsitektur tidak otomatis memenuhi seluruh frasa state-of-the-art dalam rubrik. Tambah pembanding modern hanya jika waktu memungkinkan.

```python
# src/discourse/models.py
import torch
from torch import nn
from transformers import AutoModel

class TextCNN(nn.Module):
    def __init__(self, vocab_size, dim=128, channels=128, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, channels, k, padding=k//2) for k in (3, 5, 7)])
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(channels * 3, 2)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids).transpose(1, 2)
        valid = attention_mask.bool().unsqueeze(1)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            h = h.masked_fill(~valid, torch.finfo(h.dtype).min)
            pooled.append(h.max(dim=-1).values)
        return self.head(self.dropout(torch.cat(pooled, dim=-1)))

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, dim=128, hidden=128, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True,
                            bidirectional=True)
        self.attention = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden * 2, 2))

    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(dim=1).cpu()
        x = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        packed_h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_h, batch_first=True, total_length=input_ids.size(1))
        scores = self.attention(h).squeeze(-1)
        scores = scores.masked_fill(~attention_mask.bool(),
                                   torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        z = (h * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(z)

class TransformerMultiTask(nn.Module):
    def __init__(self, checkpoint, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        return self.head(self.dropout(out.last_hidden_state[:, 0]))
```

Output `[batch, 2]`: kolom 0 toxicity, kolom 1 polarization. Untuk single-task, latih model terpisah dan aktifkan loss satu task saja. Vocabulary CNN/BiLSTM hanya dari training. Semua input harus memiliki minimal satu token valid.

## 12. Masked loss dan consistency

Rata-rata loss per task agar task dengan lebih banyak anotasi tidak mendominasi semata-mata karena jumlah sampel.

```python
# src/discourse/losses.py
import torch.nn.functional as F

def masked_task_bce(logits, targets, masks):
    elementwise = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none")
    counts = masks.sum(dim=0)
    per_task = (elementwise * masks).sum(dim=0) / counts.clamp_min(1)
    active = counts > 0
    if not active.any():
        raise ValueError("Batch contains no valid supervision")
    return per_task[active].mean()

def prediction_consistency(logits_a, logits_b):
    return F.mse_loss(logits_a.sigmoid(), logits_b.sigmoid())

def total_loss(logits, targets, masks, augmented_logits=None,
               consistency_weight=0.0):
    loss = masked_task_bce(logits, targets, masks)
    if augmented_logits is not None and consistency_weight > 0:
        loss = loss + consistency_weight * prediction_consistency(
            logits, augmented_logits)
    return loss
```

Hard target: mayoritas non-tie. Soft target: proporsi positif. Perbandingan utama memakai sampel identik. Tambahan tie hanya sebagai eksperimen tambahan. Consistency bukan otomatis valid: perturbasi wajib diaudit agar label-preserving. Jika ingin effective-batch task weighting persis saat accumulation, akumulasikan denominator task di tingkat effective batch, bukan mengasumsikan rata-rata microbatch identik.

## 13. Training dan konfigurasi

```yaml
experiment_id: E3_soft_mtl
seed: 42
model:
  name: transformer
  checkpoint: indolem/indobertweet-base-uncased
training:
  max_length: 128
  micro_batch_size: 4
  gradient_accumulation: 4
  learning_rate: 0.00002
  weight_decay: 0.01
  epochs: 5
  gradient_clip: 1.0
  precision: fp32
  target_mode: soft
  consistency_weight: 0.0
selection:
  metric: validation_mean_macro_f1
```

Konfigurasi awal, bukan benchmark VRAM. Mulai smoke test FP32, kemudian mixed precision jika environment mendukung.

```python
# inti src/discourse/train.py
import torch
from discourse.losses import total_loss

def train_epoch(model, loader, optimizer, device, accumulation=4,
                consistency_weight=0.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    n_batches = len(loader)
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        aug_logits = None
        if consistency_weight > 0:
            aug_logits = model(batch["aug_input_ids"],
                               batch["aug_attention_mask"])
        loss = total_loss(logits, batch["targets"], batch["target_masks"],
                          augmented_logits=aug_logits,
                          consistency_weight=consistency_weight)
        window_start = (step // accumulation) * accumulation
        window_size = min(accumulation, n_batches - window_start)
        (loss / window_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == n_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item()
    return running_loss / max(n_batches, 1)
```

Belum trainer lengkap. Implementasikan: collator/dataloader, validation, best checkpoint, resume, seed, tokenizer/vocabulary saving, logging, mixed precision, dan prediction export. Metadata/string tidak dikirim langsung ke `.to(device)`; collator harus memisahkannya.

## 14. Matriks eksperimen

| ID | Model | Target | Task | Consistency |
|---|---|---|---|---|
| E0 | Logistic Regression | Hard | Terpisah | Tidak |
| E1a | TextCNN | Hard | Terpisah | Tidak |
| E1b | BiLSTM-Attention | Hard | Terpisah | Tidak |
| E1c | IndoBERTweet | Hard | Terpisah | Tidak |
| E2 | IndoBERTweet | Hard | Multitask | Tidak |
| E3 | IndoBERTweet | Soft | Multitask | Tidak |
| E4 | IndoBERTweet | Soft | Multitask | Ya |
| E5 | Konfigurasi terpilih | Soft | Multitask | Sesuai konfigurasi |

E5 untuk preprocessing P0–P2. E3-common memakai sampel identik dengan E2; E3-expanded menambah tie. Perbandingan E3-common versus E3-expanded mengisolasi efek tambahan data.

Untuk klaim single-task vs multitask, samakan paparan sampel relevan, budget tuning, dan protokol pelatihan. Nyatakan jika budget komputasi tidak sama.

## 15. Evaluasi dan statistik

### Kontrak prediksi

```text
experiment_id, seed, text_id, task, n_annotations, q,
hard_label, pred_probability, threshold, pred_label,
split, perturbation_type
```

### Evaluasi kelas

Pada subset non-tie yang sama: macro-F1, precision/recall/F1 positif, average precision atau PR-AUC dengan definisi implementasi, ROC-AUC jika dua kelas tersedia, confusion matrix. Threshold dipilih validation.

### Evaluasi distribusi

Pada multi-annotator: cross-entropy terhadap proporsi anotasi, squared error, analisis menurut jumlah anotator. Bedakan squared error terhadap soft label dari Brier terhadap majority label.

### Kalibrasi

Brier, reliability diagram, ECE dengan bin dinyatakan, confident-error analysis. Temperature scaling hanya fit validation; laporkan sebelum/sesudah.

### Robustness

Laporkan skor clean, skor setiap gangguan, penurunan absolut F1, dan proporsi perturbasi yang lolos audit makna. Jangan menggunakan perturbasi test untuk tuning.

### Statistik dan efisiensi

- Tiga seed konfigurasi final; mean dan standard deviation.
- Paired bootstrap CI selisih metrik pada test.
- Unit resampling teks atau group yang relevan, bukan suara anotator independen.
- Nyatakan metode penggabungan seed dan bootstrap; jangan memperlakukan seed sebagai sampel teks baru.
- Parameter, trainable parameters, peak VRAM, latency dengan batch size disebutkan, waktu training.

## 16. Tests dan quality gates

```python
# tests/test_targets.py
from discourse.targets import summarize_votes

def test_soft_label():
    x = summarize_votes([0, 1, 1])
    assert x["n"] == 3
    assert abs(x["q"] - 2/3) < 1e-8
    assert x["hard"] == 1

def test_tie():
    x = summarize_votes([0, 1])
    assert x["q"] == 0.5
    assert x["hard"] is None

def test_missing_target():
    x = summarize_votes([])
    assert x["mask"] is False
    assert x["q"] is None
```

| Test | Kriteria |
|---|---|
| Schema | Pesan error jelas jika mapping salah |
| Split | Tidak ada group overlap |
| Target | Missing tidak menjadi negatif |
| Preprocessing | Negasi tidak hilang |
| Loss | Target masked tidak memengaruhi loss |
| Padding | Padding tidak menjadi attention target |
| Inference | Tidak memerlukan anotator/gold label |
| Evaluation | ID evaluasi sama antarmodel |
| Reproduksibilitas | Config, seed, split, revision tersimpan |

## 17. Kerangka penyajian hasil

### Bab I — Pendahuluan

1.1 Latar Belakang

1.2 Identifikasi Masalah

1.3 Rumusan Masalah

1.4 Tujuan Penelitian

1.5 Hipotesis

1.6 Manfaat

1.7 Ruang Lingkup

1.8 Kontribusi yang Diusulkan

1.9 Penelitian Terdahulu dan Posisi Penelitian

### Bab II — Pemahaman Data dan Pra-Pemrosesan

2.1 Sumber, Versi, dan Etika Dataset

2.2 Unit Analisis dan Data Dictionary

2.3 Audit Integritas Teks dan Anotasi

2.4 Audit Duplikat dan Leakage

2.5 Karakteristik Label dan Disagreement

2.6 Exploratory Data Analysis

2.7 Hard Label dan Soft Label

2.8 Train/Validation/Test Split

2.9 Pipeline Preprocessing

2.10 Tokenisasi dan Truncation

2.11 Audit Anotasi Tambahan dan Perturbasi

2.12 Ringkasan Keputusan Persiapan Data

### Bab III — Pengembangan Model dan Evaluasi

3.1 Desain Eksperimen

3.2 Rasional Baseline dan Model

3.3 TextCNN

3.4 BiLSTM-Attention

3.5 IndoBERTweet

3.6 Multitask dan Target Distribusi

3.7 Consistency Regularization

3.8 Training dan Tuning

3.9 Matriks Eksperimen dan Ablation

3.10 Metrik dan Protokol

3.11 Hasil Perbandingan Arsitektur

3.12 Hasil Preprocessing dan Agregasi

3.13 Robustness dan Kalibrasi

3.14 Statistik

3.15 Error Analysis

3.16 Pembahasan Hipotesis

3.17 Keterbatasan dan Reproduksibilitas

Untuk proposal, bagian hasil berisi rencana evaluasi, bukan hasil fiktif. Gunakan sitasi dan daftar pustaka APA 7; verifikasi metadata dari penerbit. Referensi utama dalam rentang 2021–2026. Dokumen ini tidak menyertakan ulang matriks 18 publikasi dari diskusi sebelumnya.

| RM | Eksperimen | Metrik | Bagian |
|---|---|---|---|
| RM1 | P0–P2 | F1, truncation, audit makna | II dan III |
| RM2 | E1a–E1c | F1, AP, latency, VRAM | III.11 |
| RM3 | E1c–E2 | Selisih skor per task | III.12 |
| RM4 | E2–E3-common | F1 dan distribusi | III.12 |
| RM5 | E3–E4 | Clean/perturbed | III.13 |
| RM6 | Kalibrasi | ECE, Brier, reliability | III.13 |

## 18. Instruksi AI coding assistant

```text
PROJECT: IndoDiscourse toxicity–polarization research.
GOAL: Reproducible research pipeline, not production moderation.

RULES
1. Never fabricate statistics or results.
2. Preserve raw data and annotation provenance.
3. Inspect actual schema before mapping columns.
4. Never convert missing targets into negative labels.
5. Keep duplicate groups inside one split.
6. Fit vocabulary/dictionaries on training only; thresholds/calibration
   on validation only.
7. Keep test out of model selection.
8. Do not use annotator ID/demographics/gold labels as inference input.
9. Keep hard/soft comparisons sample-matched.
10. Separate single-annotator disagreement analysis.

ORDER
A. Download and manifest.
B. Schema and audit.
C. Targets.
D. Duplicate groups and split.
E. EDA and preprocessing.
F. Dataset/collator and smoke tests.
G. Three model families.
H. Training and checkpointing.
I. Prediction export and evaluation.
J. Multitask and soft-label.
K. Consistency and robustness.
L. Statistics and report.

FOR EVERY TASK
- State assumptions and acceptance criteria.
- Implement a small module and tests.
- Run smoke test.
- Save artifacts and actual output.
- Report unresolved issues.
- Stop if schema/leakage checks fail.
```

## 19. Prioritas, deliverable, dan rubrik

### Tahap 1 — Fondasi

Versi dataset; audit; EDA aktual; split; tiga keluarga model; prediction export; metrik dan error analysis.

### Tahap 2 — Kontribusi

Single/multitask; hard/soft pada sampel sama; consistency ablation; preprocessing ablation.

### Tahap 3 — Penguatan

Tiga seed; bootstrap CI; kalibrasi; robustness; pelaporan kegagalan.

| Rubrik | Bukti |
|---|---|
| LO4 evaluasi (10%) | Metrik benar, split, CI, robustness, kalibrasi, efisiensi |
| LO4 arsitektur (30%) | Implementasi dan analisis arsitektur; jangan klaim model lama sebagai SOTA |
| LO5 riset (30%) | Sintesis literatur, disagreement, domain shift, negative transfer |
| LO6 inovasi (30%) | Hipotesis mekanisme, ablation, peningkatan bermakna, batasan |

Tidak ada jaminan nilai 90+. Temuan negatif tetap ilmiah, tetapi rubrik peningkatan tertinggi memerlukan bukti peningkatan pada sasaran yang ditentukan sebelumnya.

## 20. Definition of done

- [ ] Dataset revision dan checksum tersimpan.
- [ ] Mapping skema diverifikasi.
- [ ] Sampel tidak valid dan alasan pengeluaran dilaporkan.
- [ ] Tidak ada group overlap.
- [ ] EDA aktual, bukan simulasi.
- [ ] Target hard/soft dan missing mask diuji.
- [ ] Semua baseline dapat dilatih dan dievaluasi.
- [ ] Predictions dan config tiap eksperimen tersimpan.
- [ ] Perbandingan sampel terkontrol.
- [ ] Novelty diuji dengan ablation.
- [ ] Robustness diaudit manusia.
- [ ] Seed dan interval ketidakpastian dilaporkan.
- [ ] Kesimpulan menjawab RM berdasarkan hasil aktual.
- [ ] Kode dan laporan membedakan rancangan dari hasil.
