"""Tahap 3 — Seleksi lagu Indonesia, feature engineering, tabel pivot, dan label popularitas.

Popularity = total streams / jumlah hari di chart harian Spotify Indonesia (Kworb).
Label 3 kelas berdasarkan kuartil (mengikuti Lubis & Huda, 2025):
  0 kurang populer (< Q1) | 1 cukup populer (Q1..<Q3) | 2 sangat populer (>= Q3)
"""
import argparse
import hashlib

import pandas as pd

from src.config import DATA_DIR, TEST_FRACTION
from src.text_utils import is_indonesian

IN_PATH = DATA_DIR / "songs_with_lyrics.csv"
OUT_PATH = DATA_DIR / "dataset_labeled.csv"
PIVOT_PATH = DATA_DIR / "pivot_performance.csv"

CLASS_NAMES = ["Kurang populer", "Cukup populer", "Sangat populer"]


def select_songs(df: pd.DataFrame) -> pd.DataFrame:
    """Hanya lirik berbahasa Indonesia; lirik identik (versi akustik/remix, dsb.) cukup sekali."""
    df = df[df["lyrics_clean"].fillna("").map(is_indonesian)]
    return df.drop_duplicates(subset="lyrics_clean").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["streams_per_day"] = df["total_streams"] / df["days"]
    df["days_bucket"] = pd.cut(
        df["days"], bins=[0, 7, 30, 90, 365, 10_000],
        labels=["<=1 minggu", "<=1 bulan", "<=3 bulan", "<=1 tahun", ">1 tahun"],
    )
    return df


def label(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    q1, q3 = df["streams_per_day"].quantile([0.25, 0.75])
    df = df.copy()
    df["label"] = 0
    df.loc[df["streams_per_day"] >= q1, "label"] = 1
    df.loc[df["streams_per_day"] >= q3, "label"] = 2
    return df, {"q1": float(q1), "q3": float(q3)}


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot rasio performa: median streams/hari & jumlah lagu per durasi chart × kelas."""
    return pd.pivot_table(
        df, index="days_bucket", columns="label",
        values=["streams_per_day", "total_streams"],
        aggfunc={"streams_per_day": "median", "total_streams": "count"},
        observed=False,
    ).rename(columns={i: n for i, n in enumerate(CLASS_NAMES)} |
             {"total_streams": "jumlah_lagu", "streams_per_day": "median_streams_per_day"})


def is_test(artist: str, title: str) -> bool:
    """Golden test set deterministik: hash (artist, title) -> lagu yang sama selalu di test set."""
    h = hashlib.md5(f"{artist}|||{title}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < TEST_FRACTION


def split_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series([is_test(a, t) for a, t in zip(df["artist"], df["title"])], index=df.index)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    raw = pd.read_csv(IN_PATH)
    df = add_features(select_songs(raw))
    df, thr = label(df)
    build_pivot(df).to_csv(PIVOT_PATH)
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(raw):,} lagu berlirik -> {len(df):,} lagu Indonesia unik")
    print(f"Ambang streams/hari: Q1={thr['q1']:,.0f}  Q3={thr['q3']:,.0f}")
    print(df["label"].value_counts().sort_index().rename(dict(enumerate(CLASS_NAMES))).to_string())


if __name__ == "__main__":
    main()
