"""Tahap 2 — Lyric scraping & parsing via LRCLIB (https://lrclib.net, API publik tanpa key).

Hasil di-cache per lagu (data/lyrics_cache.jsonl) sehingga skrip bisa dihentikan
dan dilanjutkan kapan saja tanpa mengulang request.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

from src.text_utils import clean_lyrics

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH = DATA_DIR / "kworb_id_daily_totals.csv"
CACHE_PATH = DATA_DIR / "lyrics_cache.jsonl"
OUT_PATH = DATA_DIR / "songs_with_lyrics.csv"
API = "https://lrclib.net/api"
HEADERS = {"User-Agent": "TralalaTrilili/1.0 (lyric virality research)"}

_lock = Lock()


def _strip_title(title: str) -> str:
    """Buang embel-embel '(feat. X)', '- From \"Film\"', dll. agar pencarian lebih akurat."""
    title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title)
    title = re.split(r"\s+-\s+", title)[0]
    return title.strip()


def _request(path: str, params: dict, retries: int = 6):
    """GET ke LRCLIB dengan backoff; raise bila tetap gagal (agar tidak di-cache sebagai kosong)."""
    for attempt in range(retries):
        try:
            r = requests.get(f"{API}/{path}", params=params, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LRCLIB gagal setelah {retries} percobaan: {params}")


def fetch_lyrics(artist: str, title: str) -> str:
    main_artist = re.split(r",|&| feat\.? | x ", artist, flags=re.IGNORECASE)[0].strip()
    short_title = _strip_title(title) or title
    # 1) pencarian terstruktur (judul + artis); 2) fallback pencarian bebas
    results = _request("search", {"track_name": short_title, "artist_name": main_artist}) or []
    if not any(r.get("plainLyrics") for r in results):
        results = _request("search", {"q": f"{main_artist} {short_title}"}) or []
    for res in results:
        if res.get("plainLyrics") and not res.get("instrumental"):
            return res["plainLyrics"]
    return ""


def load_cache() -> dict:
    cache = {}
    if CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["key"]] = rec["lyrics"]
    return cache


def _key(artist: str, title: str) -> str:
    return f"{artist}|||{title}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Ambil N lagu teratas saja")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    songs = pd.read_csv(IN_PATH)
    if args.limit:
        songs = songs.head(args.limit)
    cache = load_cache()
    todo = [(a, t) for a, t in zip(songs["artist"], songs["title"]) if _key(a, t) not in cache]
    print(f"{len(songs):,} lagu, {len(cache):,} sudah di-cache, {len(todo):,} akan diambil")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, CACHE_PATH.open("a", encoding="utf-8") as f:
        futures = {pool.submit(fetch_lyrics, a, t): (a, t) for a, t in todo}
        for fut in as_completed(futures):
            a, t = futures[fut]
            try:
                lyrics = fut.result()
            except RuntimeError:
                failed += 1
                continue
            with _lock:
                cache[_key(a, t)] = lyrics
                f.write(json.dumps({"key": _key(a, t), "lyrics": lyrics}, ensure_ascii=False) + "\n")
                f.flush()
            done += 1
            if done % 100 == 0:
                print(f"  {done:,}/{len(todo):,}")

    if failed:
        print(f"{failed:,} lagu gagal (rate limit) — jalankan ulang skrip untuk melanjutkan")
    songs["lyrics_raw"] = [cache.get(_key(a, t), "") for a, t in zip(songs["artist"], songs["title"])]
    songs["lyrics_clean"] = songs["lyrics_raw"].map(clean_lyrics)
    songs["word_count"] = songs["lyrics_clean"].str.split().str.len().fillna(0).astype(int)
    found = songs[songs["word_count"] >= 20]
    found.to_csv(OUT_PATH, index=False)
    print(f"Lirik ditemukan untuk {len(found):,}/{len(songs):,} lagu -> {OUT_PATH}")


if __name__ == "__main__":
    main()
