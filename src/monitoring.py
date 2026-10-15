"""Monitoring — log prediksi dari app & cek prediction bias.

App menulis log ke folder lokal; `CommitScheduler` mengunggahnya berkala ke dataset repo
(`predictions/`). Lirik TIDAK disimpan — hanya peluang, tingkat, jumlah kata, dan versi model.
Logging aktif hanya bila HF_TOKEN tersedia (Streamlit secrets).

`python -m src.monitoring` membandingkan rata-rata peluang prediksi 7 hari terakhir dengan
proporsi label Viral saat training (prediction bias, lihat Google ML pipelines).
"""
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import HF_DATASET_REPO, HF_MODEL_REPO

LOG_DIR = Path(os.getenv("PREDICTION_LOG_DIR", "/tmp/tralala_trilili_predictions"))
BIAS_ALERT = 0.15  # selisih absolut rata-rata prediksi vs viral_rate training


class PredictionLogger:
    def __init__(self):
        self.enabled = bool(os.getenv("HF_TOKEN"))
        self.file = LOG_DIR / f"{uuid.uuid4()}.jsonl"
        if self.enabled:
            from huggingface_hub import CommitScheduler
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.scheduler = CommitScheduler(repo_id=HF_DATASET_REPO, repo_type="dataset",
                                             folder_path=LOG_DIR, path_in_repo="predictions",
                                             every=10, private=True)

    def log(self, proba: float, tier: str, word_count: int, model_version: str) -> None:
        if not self.enabled:
            return
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "proba": round(proba, 4),
               "tier": tier, "word_count": word_count, "model_version": model_version}
        with self.scheduler.lock, self.file.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def check_bias(days: int = 7) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    folder = Path(snapshot_download(HF_DATASET_REPO, repo_type="dataset",
                                    allow_patterns="predictions/*.jsonl"))
    since = datetime.now(timezone.utc) - timedelta(days=days)
    probas = [r["proba"] for f in folder.glob("predictions/*.jsonl")
              for r in map(json.loads, f.read_text().splitlines())
              if datetime.fromisoformat(r["ts"]) >= since]
    meta = json.loads(Path(hf_hub_download(HF_MODEL_REPO, "metadata.json")).read_text())
    if not probas:
        print(f"Belum ada prediksi dalam {days} hari terakhir.")
        return 0
    mean, rate = sum(probas) / len(probas), meta["viral_rate"]
    alert = abs(mean - rate) > BIAS_ALERT
    report = (f"### Monitoring ({days} hari)\n\n{len(probas)} prediksi · rata-rata peluang "
              f"**{mean:.1%}** vs proporsi Viral saat training **{rate:.1%}** "
              f"{'⚠️ prediction bias tinggi' if alert else '✅ normal'}\n")
    print(report)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(report)
    return 1 if alert else 0


if __name__ == "__main__":
    sys.exit(check_bias())
