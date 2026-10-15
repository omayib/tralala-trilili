"""Tahap 4 & 5 — Training SVM (TF-IDF), tuning, benchmarking, dan serialisasi model."""
import json
import os
import subprocess
import warnings
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import DATA_DIR, MODEL_DIR, ROOT
from src.features import DEFAULT_QUANTILE, split_mask

DATA_PATH = DATA_DIR / "dataset_labeled.csv"
# SVC(probability=True) deprecated sejak sklearn 1.9 (dihapus di 1.11; versi dipin 1.9.1).
# Pengganti CalibratedClassifierCV memprediksi lewat proba>=0.5 sehingga efek class_weight
# hilang (F1 SVM jatuh ke 0 pada data imbalanced) — jadi tetap dipakai dengan sengaja.
warnings.filterwarnings("ignore", message="The `probability` parameter", category=FutureWarning)
SEED = 42


def tfidf():
    return TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.8, max_features=20000)


CANDIDATES = {
    "SVM": (
        Pipeline([("tfidf", tfidf()),
                  ("clf", SVC(probability=True, class_weight="balanced", random_state=SEED))]),
        [
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__kernel": ["linear"],
             "clf__C": [0.1, 1, 10]},
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__kernel": ["rbf"],
             "clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.1, 1]},
        ],
    ),
    "Logistic Regression": (
        Pipeline([("tfidf", tfidf()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 1, 10]},
    ),
    "Random Forest": (
        Pipeline([("tfidf", tfidf()),
                  ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                 random_state=SEED, n_jobs=-1))]),
        {"clf__n_estimators": [300], "clf__max_depth": [None, 50],
         "clf__min_samples_leaf": [1, 3]},
    ),
    "Naive Bayes": (
        Pipeline([("tfidf", tfidf()), ("clf", ComplementNB())]),
        {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 0.5, 1.0]},
    ),
}


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


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
    X, y = df["lyrics_clean"], df["viral"]
    test = split_mask(df)
    X_tr, X_te, y_tr, y_te = X[~test], X[test], y[~test], y[test]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Data: {len(df):,} lagu | train {len(X_tr):,} | test {len(X_te):,} | "
          f"viral {y.mean():.1%}")

    results, fitted = [], {}
    for name, (pipe, grid) in CANDIDATES.items():
        print(f"\n> Tuning {name} ...")
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_tr, y_tr)
        metrics = evaluate(y_te, search.predict(X_te))
        results.append({"model": name, "cv_f1": search.best_score_, **metrics,
                        "best_params": json.dumps(search.best_params_, default=str)})
        fitted[name] = search.best_estimator_
        print(f"  best {search.best_params_}")
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    bench = pd.DataFrame(results).sort_values("f1", ascending=False)
    MODEL_DIR.mkdir(exist_ok=True)
    bench.to_csv(MODEL_DIR / "benchmark.csv", index=False)
    print("\n=== Benchmark (test set) ===")
    print(bench.drop(columns="best_params").round(3).to_string(index=False))

    best_name = bench.iloc[0]["model"]
    best = fitted[best_name]
    # Model dirilis apa adanya (dilatih pada train split saja) agar golden test set tetap
    # "belum pernah dilihat" oleh semua versi model -> validasi antar-versi tetap jujur.
    joblib.dump(best, MODEL_DIR / "virality_model.joblib")           # pipeline lengkap
    joblib.dump(best.named_steps["tfidf"], MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(best.named_steps["clf"], MODEL_DIR / "classifier.joblib")

    threshold = (df["total_streams"] / df["days"]).quantile(DEFAULT_QUANTILE)
    meta = {
        "best_model": best_name,
        "trained_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "data_revision": os.getenv("DATA_REVISION", "local"),
        "n_songs": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "viral_rate": float(y.mean()),
        "viral_threshold_streams_per_day": float(threshold),
        "test_metrics": {k: float(v) for k, v in bench.iloc[0][["accuracy", "precision", "recall", "f1"]].items()},
        "benchmark": bench.drop(columns="best_params").round(4).to_dict(orient="records"),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nModel terbaik: {best_name} -> {MODEL_DIR / 'virality_model.joblib'}")


if __name__ == "__main__":
    main()
