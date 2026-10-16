# Benchmark Prediksi Popularitas Lirik Lagu Indonesia

Proyek Tralala-Trilili terinspirasi oleh Lubis & Huda (2025) (lihat [Sitasi](#sitasi)).
Laporan lengkap dengan grafik: [dokumen benchmark](https://claude.ai/code/artifact/732f16c7-359c-4397-bc95-d8c34c663eea).

**Ringkasan:** model teks berbobot kelas (TF-IDF + Logistic Regression, dengan atau tanpa fitur
semantik) dan fine-tuning IndoBERT memberi macro-F1 terbaik (0,44–0,64), sekitar 1,5× metode
SVM + SMOTE ala paper yang dievaluasi tanpa kebocoran data. Lirik membawa sinyal popularitas,
tetapi moderat; selang kepercayaan antar metode terbaik saling tumpang tindih. Lagu yang lebih
populer cenderung lebih sedih, lebih panjang, dan kurang repetitif.

## Hasil (macro-F1 pada test set, 95% CI bootstrap)

| Metode | 3 kelas · per lagu | Biner · per lagu | 3 kelas · per artis | Biner · per artis |
|---|---|---|---|---|
| B0 Baseline kelas mayoritas | 0,223 | 0,427 | 0,214 | 0,445 |
| M0 TF-IDF + SMOTE + SVM One-vs-Rest (metode paper) | 0,294 | 0,470 | 0,370 | 0,504 |
| M1 TF-IDF + LinearSVC (class_weight) | 0,438 | 0,601 | 0,395 | 0,547 |
| **M2 TF-IDF + LogReg (class_weight) — produksi** | 0,448 [0,39–0,51] | 0,605 [0,53–0,68] | 0,407 | 0,554 |
| M3 TF-IDF kata + karakter + LogReg | 0,438 | 0,585 | **0,441** [0,37–0,51] | 0,575 |
| M4 TF-IDF + Complement NB | 0,422 | 0,541 | 0,388 | 0,561 |
| M5 Embedding multilingual MiniLM + LogReg (GPU) | 0,409 | 0,539 | 0,373 | 0,580 |
| M6 Fine-tuning IndoBERT, ensemble 3 seed (GPU) | **0,505** [0,44–0,58] | 0,593 | 0,414 | **0,627** [0,55–0,70] |
| M7 30 fitur semantik (tema, emosi, struktur) + LogReg | 0,375 | 0,534 | 0,380 | 0,546 |
| M8 TF-IDF + fitur semantik + LogReg | 0,439 | **0,643** [0,57–0,71] | 0,405 | 0,561 |

Angka lengkap (accuracy, balanced accuracy, macro-P/R, CI, hyperparameter terpilih, waktu):
`reports/benchmark_{3class,binary}_{song,artist}.csv`.

## Fitur semantik: tema, emosi, dan struktur (M7, M8)

30 fitur dihitung tanpa melihat label popularitas (`experiments/semantic.py`):
12 tema dari ontologi kecil pop Indonesia (kemiripan embedding dengan kalimat prototipe),
5 emosi dari IndoBERT yang di-fine-tune pada IndoNLU EmoT (macro-F1 validasi 0,730), dan
13 fitur struktur (repetisi/hook, panjang, rima, keragaman kosakata, porsi bahasa Inggris, kata ganti).

- **M7 (semantik saja)** jauh di atas baseline dan hampir menyamai TF-IDF pada artis baru
  (0,380 vs 0,407; 0,546 vs 0,554) dengan 30 fitur saja.
- **M8 tidak lebih baik secara signifikan dari M2** (bootstrap berpasangan pada test set yang sama):

| Skenario | M2 | M8 | Selisih [95% CI] |
|---|---|---|---|
| 3 kelas · per lagu | 0,448 | 0,439 | −0,009 [−0,051; +0,036] |
| Biner · per lagu | 0,605 | 0,643 | +0,039 [−0,028; +0,103] |
| 3 kelas · per artis | 0,407 | 0,405 | −0,002 [−0,057; +0,054] |
| Biner · per artis | 0,554 | 0,561 | +0,008 [−0,035; +0,057] |

**Asosiasi dengan popularitas** (Spearman ρ vs log streams/hari, 1.260 lagu; 14 dari 30 fitur
signifikan setelah koreksi Benjamini-Hochberg, semuanya lemah |ρ| ≤ 0,15):

| Lebih banyak pada lagu populer | ρ | Lebih sedikit pada lagu populer | ρ |
|---|---|---|---|
| Jumlah kata | +0,15 | Emosi cinta | −0,14 |
| Kata per baris | +0,15 | Porsi baris berulang | −0,10 |
| Emosi sedih | +0,13 | Pengulangan hook | −0,09 |
| Rasio baris unik | +0,12 | | |
| Tema patah hati / kesepian | +0,12 / +0,11 | | |
| Emosi marah, tema religi | +0,09 | | |

Asosiatif, bukan sebab-akibat. Temuan repetisi berlawanan dengan studi chart berbahasa Inggris;
sebagian bisa efek era (balada "galau" panjang mendominasi tahun dengan volume streaming
tertinggi). Validitas tema berbasis embedding beragam (patah hati dan keluarga tepat sasaran,
pesta/joget banyak salah tangkap). Data: `reports/semantic_associations.csv`,
`reports/semantic_paired_bootstrap.csv`.

## Evaluasi ulang protokol paper (dataset repo resmi, 799 lagu)

| ID | Protokol | Accuracy | Macro-F1 |
|---|---|---|---|
| — | Dilaporkan paper | 0,890 | 0,89 |
| R0 | Notebook repo apa adanya (SMOTE sebelum split) | 0,880 | 0,878 |
| R1 | Metode paper, SMOTE sebelum split | 0,842 | 0,845 |
| R2 | Metode paper, TF-IDF + SMOTE di dalam pipeline (split dulu) | 0,631 | 0,549 |
| R3 | R2 + buang 184 lirik duplikat dan 23 "Lyrics not found" | 0,512 | 0,381 |
| B3 | Baseline kelas mayoritas (test set R3) | 0,488 | 0,219 |

Dataset repo resmi mayoritas berbahasa Inggris, berbeda dari 652 lagu Indonesia yang dideskripsikan
paper. Detail: `reports/reproduction_lubis2025.csv`.

## Dataset dan protokol

- **Lagu:** Spotify Daily Chart Totals Indonesia ([Kworb](https://kworb.net/spotify/country/id_daily_totals.html)); **lirik:** [LRCLIB](https://lrclib.net).
- **Seleksi:** lirik berbahasa Indonesia (`src/text_utils.is_indonesian`), dedup lirik identik → 1.260 lagu.
- **Popularity** = total streams / jumlah hari di chart (pengganti skor `popularity` Spotify).
- **Label:** 3 kelas kuartil (< Q1 = 13.813; ≥ Q3 = 185.327 streams/hari) dan biner (≥ Q3).
- **Split:** 80/20 deterministik via hash, per lagu atau per artis (artis test tidak ada di train).
- **Tanpa kebocoran:** TF-IDF/SMOTE di dalam pipeline; GridSearchCV 5-fold hanya pada train; test dievaluasi sekali.

## Batasan

Tidak ada filter tahun rilis 2017–2024 (butuh Spotify API); sumber lirik LRCLIB, bukan
lirik.kapanlagi.com; semua lagu pernah masuk chart (bias survivorship); test set 228–242 lagu
sehingga CI ±0,07; faktor non-lirik (artis, promosi, aransemen) tidak dimodelkan.

## Reproduksi

```bash
pip install -r requirements-experiments.txt
pip install torch --index-url https://download.pytorch.org/whl/cu130   # M5/M6 di GPU
python -m experiments.reproduce_lubis2025        # butuh references/lirik-lagu-dataset/songs.csv
python -m experiments.benchmark --gpu --split song
python -m experiments.benchmark --gpu --split artist
python -m experiments.benchmark --semantic --split song --only M7 M8   # + artist
python -m experiments.semantic_analysis
```

## Sitasi

Proyek ini terinspirasi oleh:

> Lubis, Q. L. A., & Huda, A. A. (2025). Implementation of SVM Algorithm to Predict Song
> Popularity based on Sentiment Analysis of Lyrics. *Journal of Applied Informatics and
> Computing (JAIC)*, 9(2), 265–272. http://jurnal.polibatam.ac.id/index.php/JAIC
>
> Kode & data: https://github.com/QuiinLatifah/Lyrics-Popularity-Prediction
