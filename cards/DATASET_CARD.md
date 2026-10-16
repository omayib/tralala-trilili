---
language: id
pretty_name: Tralala-Trilili — Spotify Indonesia chart + lyrics
---

# Tralala-Trilili data (privat)

Dataset internal untuk melatih model Tralala-Trilili. **Repo ini privat** karena berisi lirik
lagu berhak cipta; jangan dipublikasikan.

Diperbarui otomatis oleh pipeline harian (GitHub Actions). Setiap pembaruan = satu commit,
sehingga setiap model di registry bisa ditelusuri ke revisi data yang dipakainya
(`data_revision` di `metadata.json`).

| Berkas | Isi | Sumber |
|---|---|---|
| `kworb_id_daily_totals.csv` | posisi, artist, title, track_id, days, days_top10, peak_position, peak_times, peak_streams, total_streams, avg_daily_streams | kworb.net (scrape harian) |
| `lyrics_cache.jsonl` | `{"key": "artist\|\|\|title", "lyrics": "..."}` — cache lirik mentah (inkremental) | lrclib.net |
| `songs_with_lyrics.csv` | data Kworb + `lyrics_raw`, `lyrics_clean`, `word_count` (≥ 20 kata) | gabungan |
| `dataset_labeled.csv` | lagu berbahasa Indonesia unik + `streams_per_day`, `days_bucket`, `label` (0 kurang / 1 cukup / 2 sangat populer, kuartil) | `src/features.py` |
| `pivot_performance.csv` | median streams/hari & jumlah lagu per durasi chart × kelas | `src/features.py` |

Catatan: Kworb tidak menyediakan streaming harian per lagu; `avg_daily_streams = total / days`.
Label berubah seiring waktu karena total streams terus bertambah dan ambang dihitung ulang.
