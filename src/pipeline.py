"""Orkestrator ML pipeline (dipakai sama persis secara lokal maupun di GitHub Actions).

  data pipeline     : pull data (HF) -> scrape Kworb -> scrape lirik (inkremental) -> features
                      -> push data (HF dataset, versi baru)
  training pipeline : train + benchmark 4 model
  validation        : kandidat vs produksi pada golden test set
  release           : push model + model card + tag ke HF model registry

Contoh:
  python -m src.pipeline                 # jalankan semua, tanpa publish (dry run)
  python -m src.pipeline --publish       # jalankan semua + rilis ke Hugging Face
  python -m src.pipeline --skip-scrape   # latih ulang dari data yang ada saja
"""
import argparse
import json
import os
import subprocess
import sys

from src.config import MODEL_DIR


def run(module: str, *args: str, env: dict | None = None) -> int:
    print(f"\n=== {module} {' '.join(args)} ===", flush=True)
    return subprocess.call([sys.executable, "-m", module, *args], env={**os.environ, **(env or {})})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publish", action="store_true", help="unggah data & rilis model ke HF Hub")
    ap.add_argument("--skip-scrape", action="store_true", help="lewati scraping Kworb & lirik")
    ap.add_argument("--lyrics-workers", default="3")
    args = ap.parse_args()

    from src import registry

    data_rev = "local"
    if args.publish:
        registry.ensure_repos()
        data_rev = registry.pull_data() or "local"

    # --- Data pipeline ---
    if not args.skip_scrape:
        if run("src.scraper_kworb"):
            return 1
        if run("src.scraper_lyrics", "--workers", args.lyrics_workers):
            return 1
    if run("src.features"):
        return 1
    if args.publish:
        data_rev = registry.push_data("Pipeline data refresh")

    # --- Training + validation pipeline ---
    if run("src.train", env={"DATA_REVISION": data_rev}):
        return 1
    passed = run("src.validate") == 0

    # --- Release ---
    if not passed:
        print("\nModel kandidat DITOLAK — produksi tetap memakai model lama.")
        return 1
    if args.publish:
        registry.push_model(json.loads((MODEL_DIR / "metadata.json").read_text()))
    else:
        print("\nDry run: kandidat lolos validasi; jalankan dengan --publish untuk merilis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
