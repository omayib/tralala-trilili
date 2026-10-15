"""Tahap 1 — Data ingestion: tabel Spotify Daily Chart Totals Indonesia dari Kworb."""
import argparse
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

KWORB_URL = "https://kworb.net/spotify/country/id_daily_totals.html"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "kworb_id_daily_totals.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (TralalaTrilili research scraper)"}


def _to_int(text: str) -> int:
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def fetch_html(url: str = KWORB_URL) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def parse_table(html: str) -> pd.DataFrame:
    """Kolom Kworb: Artist and Title | Days | T10 | Pk | (x?) | PkStreams | Total."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="sortable")
    rows = []
    for tr in table.tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue
        links = tds[0].find_all("a")
        if len(links) >= 2:
            artist = links[0].get_text(strip=True)
            title = links[-1].get_text(strip=True)
            track_id = links[-1]["href"].rsplit("/", 1)[-1].replace(".html", "")
        else:  # fallback "Artist - Title" tanpa link
            artist, _, title = tds[0].get_text(strip=True).partition(" - ")
            track_id = ""
        rows.append({
            "artist": artist,
            "title": title,
            "track_id": track_id,
            "days": _to_int(tds[1].get_text()),
            "days_top10": _to_int(tds[2].get_text()),
            "peak_position": _to_int(tds[3].get_text()),
            "peak_times": _to_int(tds[4].get_text()),
            "peak_streams": _to_int(tds[5].get_text()),
            "total_streams": _to_int(tds[6].get_text()),
        })
    df = pd.DataFrame(rows)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["artist"] = df["artist"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["days"] > 0)]
    df = df.drop_duplicates(subset=["artist", "title"]).copy()
    df = df.sort_values("total_streams", ascending=False).reset_index(drop=True)
    df.insert(0, "position", df.index + 1)  # posisi berdasarkan total streaming
    # Streaming harian rata-rata (Kworb hanya menyediakan peak daily streams)
    df["avg_daily_streams"] = (df["total_streams"] / df["days"]).round(0).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    df = parse_table(fetch_html())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Tersimpan {len(df):,} lagu -> {args.out}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
