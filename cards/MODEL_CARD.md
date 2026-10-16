---
language: id
license: mit
library_name: sklearn
tags:
- text-classification
- tf-idf
- music
- spotify
- indonesian
pipeline_tag: text-classification
---

# Tralala-Trilili — Indonesian Lyric Popularity Classifier

Memprediksi tingkat popularitas sebuah lirik lagu berbahasa Indonesia (Kurang / Cukup /
Sangat populer) berdasarkan data historis chart harian Spotify Indonesia
([kworb.net](https://kworb.net/spotify/country/id_daily_totals.html)). Dipakai oleh aplikasi
Streamlit **Tralala-Trilili**; dilatih ulang dan divalidasi otomatis oleh pipeline harian.

Proyek ini terinspirasi oleh Lubis & Huda (2025), *Implementation of SVM Algorithm to Predict
Song Popularity based on Sentiment Analysis of Lyrics* (lihat Sitasi).

## Model

- **Arsitektur:** scikit-learn `Pipeline`: TF-IDF (sublinear tf) -> **{best_model}** dengan
  `class_weight="balanced"`. Dipilih dari benchmark 7 metode (termasuk SVM+SMOTE dan
  fine-tuning IndoBERT) sebagai model dengan kualitas setara terbaik yang tetap ringan di CPU.
- **Input:** lirik yang sudah dibersihkan (`src/text_utils.clean_lyrics`: lowercase, tanpa
  tanda baca, tanpa penanda verse/chorus/kredit). Dirancang untuk lirik berbahasa Indonesia.
- **Output:** `predict_proba(...)` -> peluang `[Kurang, Cukup, Sangat populer]`.
- **Kebutuhan hardware:** CPU saja, < 50 MB RAM.

```python
import joblib
from huggingface_hub import hf_hub_download
model = joblib.load(hf_hub_download("{repo}", "virality_model.joblib"))
model.predict_proba(["lirik yang sudah dibersihkan ..."])  # [[p_kurang, p_cukup, p_sangat]]
```

## Data & label

- {n_songs:,} lagu berbahasa Indonesia unik (train {n_train:,} / golden test {n_test:,}),
  lirik dari [LRCLIB](https://lrclib.net).
- **Popularity** = total streams / jumlah hari di chart harian Spotify Indonesia.
- **Label (kuartil):** Kurang < {q1:,.0f} ≤ Cukup < {q3:,.0f} ≤ Sangat populer (streams/hari).
- Golden test set dipilih deterministik lewat hash (artist, title) dan tidak pernah dipakai
  untuk melatih versi model mana pun, sehingga semua versi bisa dibandingkan secara adil.

## Evaluasi (golden test set)

accuracy **{accuracy:.3f}** · balanced accuracy **{balanced_accuracy:.3f}** ·
macro-precision **{precision:.3f}** · macro-recall **{recall:.3f}** · macro-F1 **{f1:.3f}**

| Model | CV macro-F1 | Accuracy | Balanced acc | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|---|---|
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

Sinyal popularitas dari lirik saja tergolong moderat: artis, promosi, aransemen, dan tren media
sosial sangat berpengaruh. Akurasi turun untuk artis yang tidak ada di data latih. Data hanya
mencakup lagu yang pernah masuk chart harian Spotify Indonesia (bias survivorship). Gunakan
hasil sebagai indikasi, bukan kepastian.

## Sitasi

Proyek ini terinspirasi oleh:

> Lubis, Q. L. A., & Huda, A. A. (2025). Implementation of SVM Algorithm to Predict Song
> Popularity based on Sentiment Analysis of Lyrics. *Journal of Applied Informatics and
> Computing (JAIC)*, 9(2), 265–272. http://jurnal.polibatam.ac.id/index.php/JAIC
