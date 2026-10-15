---
language: id
license: mit
library_name: sklearn
tags:
- text-classification
- tf-idf
- music
- spotify
pipeline_tag: text-classification
---

# Tralala-Trilili — Spotify Indonesia Lyric Virality Classifier

Memprediksi peluang sebuah lirik lagu menjadi *viral* berdasarkan data historis chart
Spotify Indonesia ([kworb.net](https://kworb.net/spotify/country/id_daily_totals.html)).
Dipakai oleh aplikasi Streamlit **Tralala-Trilili**. Model ini dilatih ulang dan divalidasi
otomatis oleh pipeline harian di GitHub Actions.

## Model

- **Arsitektur:** scikit-learn `Pipeline` (TF-IDF -> **{best_model}**), dipilih dari benchmark
  SVM / Logistic Regression / Naive Bayes / Random Forest berdasarkan F1 pada test set.
- **Input:** lirik yang sudah dibersihkan (`src/text_utils.clean_lyrics`: lowercase, tanpa
  tanda baca, tanpa penanda verse/chorus/kredit).
- **Output:** `predict_proba(...)[:, 1]` = peluang Viral. Tingkat potensi di app:
  Rendah < 40% ≤ Sedang < 65% ≤ Tinggi.
- **Kebutuhan hardware:** CPU saja, < 50 MB RAM.

```python
import joblib
from huggingface_hub import hf_hub_download
model = joblib.load(hf_hub_download("{repo}", "virality_model.joblib"))
model.predict_proba(["lirik yang sudah dibersihkan ..."])[:, 1]
```

## Data & label

- {n_songs:,} lagu dengan lirik (train {n_train:,} / golden test {n_test:,}).
- **Viral (1)** = `total_streams / days` ≥ {threshold:,.0f} streams/hari (top {viral_rate:.0%}).
- Golden test set dipilih deterministik lewat hash (artist, title) dan tidak pernah dipakai
  untuk melatih versi model mana pun, sehingga semua versi bisa dibandingkan secara adil.

## Evaluasi (golden test set)

accuracy **{accuracy:.3f}** · precision **{precision:.3f}** · recall **{recall:.3f}** · F1 **{f1:.3f}**

| Model | CV F1 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
{benchmark_rows}

### Validation gate

```json
{validation}
```

## Lineage

- Dilatih: {trained_at}
- Kode (git): `{git_sha}`
- Data (HF dataset revision): `{data_revision}`

## Batasan

Lirik hanyalah salah satu faktor viralitas — artis, promosi, aransemen, dan tren media sosial
sangat berpengaruh. Data hanya mencakup lagu yang pernah masuk chart harian Spotify Indonesia
(bias survivorship). Gunakan hasil sebagai indikasi, bukan kepastian.
