"""Tahap 4 & 5 — Training model produksi, evaluasi pada golden test set, dan serialisasi.

Model produksi = M2 dari benchmark (reports/BENCHMARK.md): TF-IDF + Logistic Regression dengan
class_weight="balanced" (pengganti SMOTE). Setara IndoBERT pada benchmark, tetapi ringan dan
dilatih ulang di CPU (GitHub Actions) dalam hitungan detik.

Hyperparameter dipilih lewat 5-fold CV (macro-F1) pada train split saja; golden test set
hanya dipakai sekali untuk melaporkan metrik.
"""
import json
import os
import subprocess
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import DATA_DIR, MODEL_DIR, ROOT
from src.features import CLASS_NAMES, split_mask

DATA_PATH = DATA_DIR / "dataset_labeled.csv"
SEED = 42

PIPELINE = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
])
GRID = {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 1, 10]}


def evaluate(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": p, "recall": r, "f1": f}


def _git_sha() -> str:
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["lyrics_clean"])
    X, y = df["lyrics_clean"], df["label"]
    test = split_mask(df)
    X_tr, X_te, y_tr, y_te = X[~test], X[test], y[~test], y[test]
    print(f"Data: {len(df):,} lagu | train {len(X_tr):,} | test {len(X_te):,} | "
          + " ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, y.value_counts().sort_index())))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(PIPELINE, GRID, scoring="f1_macro", cv=cv, n_jobs=4).fit(X_tr, y_tr)
    model = search.best_estimator_
    metrics = evaluate(y_te, model.predict(X_te))
    baseline = evaluate(y_te, DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te))
    print(f"best {search.best_params_} | CV macro-F1 {search.best_score_:.3f}")
    for name, m in (("model", metrics), ("baseline", baseline)):
        print(f"  {name:<9}" + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    # Model dirilis apa adanya (dilatih pada train split saja) agar golden test set tetap
    # "belum pernah dilihat" oleh semua versi model -> validasi antar-versi tetap jujur.
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "virality_model.joblib")
    joblib.dump(model.named_steps["tfidf"], MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(model.named_steps["clf"], MODEL_DIR / "classifier.joblib")

    bench = pd.DataFrame([
        {"model": "TF-IDF + LogReg (class_weight)", "cv_f1": search.best_score_, **metrics,
         "best_params": json.dumps(search.best_params_, default=str)},
        {"model": "Baseline kelas mayoritas", "cv_f1": float("nan"), **baseline, "best_params": ""},
    ])
    bench.to_csv(MODEL_DIR / "benchmark.csv", index=False)

    spd = df["streams_per_day"]
    now = datetime.now(timezone.utc)
    meta = {
        "best_model": "TF-IDF + Logistic Regression",
        "classes": CLASS_NAMES,
        "trained_on": now.strftime("%Y-%m-%d"),
        "trained_at": now.isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "data_revision": os.getenv("DATA_REVISION", "local"),
        "n_songs": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "class_distribution": {n: int(c) for n, c in zip(CLASS_NAMES, y.value_counts().sort_index())},
        "thresholds_streams_per_day": {"q1": float(spd.quantile(0.25)), "q3": float(spd.quantile(0.75))},
        "best_params": json.loads(json.dumps(search.best_params_, default=str)),
        "cv_f1_macro": float(search.best_score_),
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "baseline_metrics": {k: float(v) for k, v in baseline.items()},
        "benchmark": bench.drop(columns="best_params").round(4).fillna(0).to_dict(orient="records"),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nModel -> {MODEL_DIR / 'virality_model.joblib'}")


if __name__ == "__main__":
    main()
