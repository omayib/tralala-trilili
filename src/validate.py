"""Validation pipeline — gerbang sebelum model baru dirilis.

Kandidat (models/) dan model produksi (HF registry) dievaluasi pada golden test set yang
SAMA dengan label terkini. Kandidat lolos bila macro-F1-nya tidak lebih buruk dari produksi
melebihi toleransi. Hasil ditulis ke models/metadata.json["validation"].

Exit code: 0 = lolos (boleh rilis), 1 = ditolak (alert lewat kegagalan workflow).
"""
import json
import os
import sys
import tempfile

import joblib
import pandas as pd

from src.config import DATA_DIR, F1_TOLERANCE, MIN_SONGS, MODEL_DIR
from src.features import split_mask
from src.registry import download_model
from src.train import evaluate


def decide(cand: dict, prod: dict | None, n_songs: int) -> tuple[bool, str]:
    if n_songs < MIN_SONGS:
        return False, f"dataset terlalu kecil ({n_songs} < {MIN_SONGS}) — cek data pipeline"
    if prod is None:
        return True, "belum ada model produksi — rilis pertama"
    delta = cand["f1"] - prod["f1"]
    if delta < -F1_TOLERANCE:
        return False, f"F1 turun {delta:+.3f} (toleransi -{F1_TOLERANCE})"
    return True, f"F1 {delta:+.3f} vs produksi (toleransi -{F1_TOLERANCE})"


def main() -> int:
    df = pd.read_csv(DATA_DIR / "dataset_labeled.csv").dropna(subset=["lyrics_clean"])
    test = df[split_mask(df)]
    X, y = test["lyrics_clean"], test["label"]

    cand = evaluate(y, joblib.load(MODEL_DIR / "virality_model.joblib").predict(X))
    with tempfile.TemporaryDirectory() as tmp:
        prod_dir = download_model(local_dir=tmp)
        prod = (evaluate(y, joblib.load(prod_dir / "virality_model.joblib").predict(X))
                if prod_dir else None)

    passed, reason = decide(cand, prod, len(df))
    result = {"passed": passed, "reason": reason, "n_test": int(len(test)),
              "candidate": {k: round(v, 4) for k, v in cand.items()},
              "production": {k: round(v, 4) for k, v in prod.items()} if prod else None}

    meta_path = MODEL_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["validation"] = result
    meta_path.write_text(json.dumps(meta, indent=2))

    report = (f"### Validation gate: {'✅ LOLOS' if passed else '❌ DITOLAK'}\n\n{reason}\n\n"
              "| | accuracy | balanced acc | macro-P | macro-R | macro-F1 |\n|---|---|---|---|---|---|\n"
              + "".join(f"| {name} | " + " | ".join(f"{m[k]:.3f}" for k in
                        ("accuracy", "balanced_accuracy", "precision", "recall", "f1")) + " |\n"
                        for name, m in (("kandidat", cand), ("produksi", prod)) if m))
    print(report)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
