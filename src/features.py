"""Tahap 3 — Feature engineering, tabel pivot, dan pelabelan Virality Score."""
import argparse
import hashlib

import pandas as pd

from src.config import DATA_DIR, TEST_FRACTION

IN_PATH = DATA_DIR / "songs_with_lyrics.csv"
OUT_PATH = DATA_DIR / "dataset_labeled.csv"
PIVOT_PATH = DATA_DIR / "pivot_performance.csv"

DEFAULT_QUANTILE = 0.75  # top 25% rasio streams/hari = "Viral"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["streams_per_day"] = df["total_streams"] / df["days"]
    df["days_bucket"] = pd.cut(
        df["days"], bins=[0, 7, 30, 90, 365, 10_000],
        labels=["<=1 minggu", "<=1 bulan", "<=3 bulan", "<=1 tahun", ">1 tahun"],
    )
    return df


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot rasio performa: rata-rata streams/hari per kelompok durasi chart & label."""
    return pd.pivot_table(
        df, index="days_bucket", columns="viral",
        values=["streams_per_day", "total_streams"],
        aggfunc={"streams_per_day": "median", "total_streams": "count"},
        observed=False,
    ).rename(columns={0: "Non-Viral", 1: "Viral", "total_streams": "jumlah_lagu",
                      "streams_per_day": "median_streams_per_day"})


def label(df: pd.DataFrame, quantile: float = DEFAULT_QUANTILE) -> tuple[pd.DataFrame, float]:
    threshold = df["streams_per_day"].quantile(quantile)
    df = df.copy()
    df["viral"] = (df["streams_per_day"] >= threshold).astype(int)
    return df, float(threshold)


def is_test(artist: str, title: str) -> bool:
    """Golden test set deterministik: hash (artist, title) -> lagu yang sama selalu di test set."""
    h = hashlib.md5(f"{artist}|||{title}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < TEST_FRACTION


def split_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series([is_test(a, t) for a, t in zip(df["artist"], df["title"])], index=df.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    args = ap.parse_args()

    df = add_features(pd.read_csv(IN_PATH))
    df, threshold = label(df, args.quantile)
    pivot = build_pivot(df)
    pivot.to_csv(PIVOT_PATH)
    df.to_csv(OUT_PATH, index=False)
    print(f"Ambang Viral: >= {threshold:,.0f} streams/hari (kuantil {args.quantile})")
    print(df["viral"].value_counts().rename({1: "Viral", 0: "Non-Viral"}).to_string())
    print("\nPivot performa:\n", pivot.round(0).to_string())


if __name__ == "__main__":
    main()
