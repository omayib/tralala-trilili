"""Konfigurasi bersama pipeline: path lokal, repo Hugging Face, dan aturan validasi.

Nilai repo bisa ditimpa lewat environment variable (GitHub Actions vars / Streamlit secrets).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# Model registry (publik, dibaca app) & dataset store (privat: berisi lirik berhak cipta)
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "omayib/tralala-trilili")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "omayib/tralala-trilili-data")
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main")  # ganti ke tag lama untuk rollback

# Berkas data yang di-versioning di HF dataset repo
DATA_FILES = [
    "kworb_id_daily_totals.csv",
    "lyrics_cache.jsonl",
    "songs_with_lyrics.csv",
    "dataset_labeled.csv",
    "pivot_performance.csv",
]
# Berkas model yang dirilis ke model registry
MODEL_FILES = [
    "virality_model.joblib",
    "tfidf_vectorizer.joblib",
    "classifier.joblib",
    "metadata.json",
    "benchmark.csv",
]

# Golden test set: ±20% lagu dipilih deterministik lewat hash (artist, title) sehingga
# lagu yang sama selalu berada di test set lintas versi -> perbandingan model adil.
TEST_FRACTION = 0.20

# Validation gate (Google ML pipelines: deploy bila model baru "not meaningfully worse")
F1_TOLERANCE = 0.02  # F1 kandidat boleh turun maksimal 0.02 dari model produksi
MIN_SONGS = 1000     # sanity check data pipeline
