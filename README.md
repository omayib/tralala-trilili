# 🎵 Tralala-Trilili — Spotify Lyric Virality Predictor

Pipeline end-to-end untuk memprediksi potensi viral sebuah lirik lagu berdasarkan
data historis chart Spotify Indonesia ([kworb.net](https://kworb.net/spotify/country/id_daily_totals.html)),
dengan antarmuka web **Streamlit**, model registry di **Hugging Face Hub**, dan otomasi
**GitHub Actions** mengikuti pola
[Google ML pipelines](https://developers.google.com/machine-learning/managing-ml-projects/pipelines).

## Arsitektur ML pipeline

```
                  GitHub Actions: ml-pipeline.yml (harian 02:00 WIB / manual)
 ┌──────────────────────────── src/pipeline.py ────────────────────────────┐
 │ DATA            pull HF dataset → scraper_kworb → scraper_lyrics        │
 │                 (inkremental, cache) → features → push HF dataset (v+1) │
 │ TRAINING        train.py: TF-IDF × SVM/LR/NB/RF + GridSearch            │
 │ VALIDATION      validate.py: kandidat vs produksi di golden test set    │
 │                 lolos bila F1 tidak turun > 0.02  ──✗──▶ workflow merah │
 │ RELEASE         push model + model card + tag vYYYYMMDD-HHMM ke HF      │
 └─────────────────────────────────────────────────────────────────────────┘
                                   │ (model repo publik)
 SERVING   Streamlit app ◀─────────┘ memuat model dari HF (cache 1 jam, tanpa redeploy)
           GitHub Actions: ci.yml — test → majukan branch `streamlit-release`
           → Streamlit Community Cloud redeploy otomatis
 MONITORING  app log prediksi (tanpa lirik) → HF dataset `predictions/`
             monitoring.py cek prediction bias harian ──⚠──▶ workflow merah
```

| Konsep Google | Implementasi |
|---|---|
| Data pipeline | `scraper_kworb` → `scraper_lyrics` → `features`; transformasi sama dengan app (`text_utils`) |
| Training pipeline | `train.py` (benchmark 4 model + tuning) |
| Validation pipeline | `validate.py` + golden test set deterministik (hash artist/title, ±20%) |
| Serving pipeline | `app.py` (prediksi online) |
| Model registry | HF model repo: versi = commit + tag, rollback via `HF_MODEL_REVISION` |
| Dataset versioning + data card | HF dataset repo **privat** (lirik berhak cipta) + `cards/DATASET_CARD.md` |
| Model card | `cards/MODEL_CARD.md` (dirender otomatis tiap rilis) |
| Lineage | `metadata.json`: `git_sha`, `data_revision`, `trained_at` |
| Monitoring & alert | log prediksi + `monitoring.py`; kegagalan workflow → email GitHub |

## Struktur

```
tralala-trilili/
├── app.py                     # Serving — UI Streamlit (mobile-first)
├── src/
│   ├── config.py              # repo HF, path, aturan validation gate
│   ├── scraper_kworb.py       # Tahap 1 — scraping & cleaning tabel Kworb
│   ├── scraper_lyrics.py      # Tahap 2 — scraping lirik (LRCLIB) + parsing, inkremental
│   ├── text_utils.py          # pembersih lirik (dipakai training & app)
│   ├── features.py            # Tahap 3 — rasio streams/hari, pivot, label, golden split
│   ├── train.py               # Tahap 4–5 — SVM + tuning, benchmark, serialisasi
│   ├── validate.py            # validation gate
│   ├── registry.py            # artifact store HF Hub (data & model)
│   ├── monitoring.py          # log prediksi & cek prediction bias
│   └── pipeline.py            # orkestrator (lokal = CI)
├── cards/                     # template model card & dataset card
├── tests/                     # unit test + smoke test app
├── .github/workflows/         # ci.yml (test + rilis app), ml-pipeline.yml (harian)
├── data/  models/             # artefak lokal (tidak di-commit; ada di HF Hub)
└── requirements.txt  requirements-dev.txt
```

## Menjalankan lokal

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m src.pipeline                # semua tahap, dry run (tanpa publish)
python -m src.pipeline --skip-scrape  # latih ulang dari data yang ada
python -m src.pipeline --publish      # + unggah data & rilis model (butuh `hf auth login`)
python -m pytest -q

streamlit run app.py                  # http://localhost:8501
```

Tiap tahap juga bisa dijalankan sendiri: `python -m src.scraper_kworb`, `src.scraper_lyrics`
(`--limit 500`), `src.features` (`--quantile 0.8`), `src.train`, `src.validate`.

## Setup otomasi (sekali saja)

1. **Hugging Face** — buat token *write* di https://huggingface.co/settings/tokens.
   Repo `omayib/tralala-trilili` (model, publik) dan `omayib/tralala-trilili-data`
   (dataset, privat) dibuat otomatis pada run pertama.
2. **GitHub** — push repo ini, lalu di *Settings → Secrets and variables → Actions*:
   - secret `HF_TOKEN` = token di atas
   - (opsional) variable `HF_MODEL_REPO`, `HF_DATASET_REPO` untuk nama repo lain
3. Jalankan workflow **ML pipeline** sekali secara manual (*Actions → Run workflow*) untuk
   rilis model pertama.
4. **Streamlit Community Cloud** — https://share.streamlit.io → *Create app* → pilih repo,
   branch **`streamlit-release`**, main file `app.py`, Python 3.12.
   Di *Advanced settings → Secrets* (opsional, untuk log prediksi/monitoring):
   ```toml
   HF_TOKEN = "hf_..."   # token write, sebaiknya fine-grained khusus repo tralala-trilili-data
   ```
   Tanpa secret, app tetap jalan (model publik), hanya logging yang nonaktif.

Setelah itu: push ke `main` → test → app redeploy; setiap hari → model baru dirilis bila
lolos validasi dan app memakainya dalam ≤ 1 jam. **Rollback model:** set secret
`HF_MODEL_REVISION = "v20261016-1900"` (tag lama) di Streamlit.

Catatan: GitHub menonaktifkan *scheduled workflow* setelah 60 hari tanpa aktivitas repo.

## Definisi & catatan metodologi

- **Kolom Kworb**: Artist, Title, Days (hari di chart), T10 (hari di Top 10), Pk (posisi
  puncak), PkStreams (streaming harian tertinggi), Total. Kworb tidak menyediakan streaming
  harian per lagu, sehingga *Streaming Harian* = `peak_streams` dan `avg_daily_streams = Total / Days`.
  *Posisi* = peringkat berdasarkan total streaming.
- **Label Viral (1)**: `total_streams / days` ≥ kuantil ke-75 (top 25%); sisanya Non-Viral (0).
  Label dihitung ulang tiap run, jadi validasi selalu memakai label terkini.
- **Golden test set**: model yang dirilis hanya dilatih pada train split, sehingga test set
  tidak pernah dilihat versi mana pun dan perbandingan antar-versi adil.
- **Lirik** diambil dari [LRCLIB](https://lrclib.net) (API publik, tanpa key); lagu tanpa
  lirik atau < 20 kata (mis. instrumental) dibuang.
- Lirik hanya salah satu faktor viralitas, sehingga akurasi model berbasis lirik saja
  memang terbatas — lihat `models/benchmark.csv`.
