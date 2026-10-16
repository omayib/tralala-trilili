"""Benchmark penyempurnaan Lubis & Huda (2025) pada lagu Indonesia (Kworb + LRCLIB).

Dataset (direkonstruksi dengan kriteria paper, tanpa Spotify API):
  data/songs_with_lyrics.csv -> hanya lirik berbahasa Indonesia -> dedup lirik

Popularity = total streams / jumlah hari di chart harian Spotify Indonesia (Kworb), dipakai
sebagai pengganti skor Spotify `popularity` di paper. Label memakai kuartil seperti paper:
  3 kelas : < Q1 kurang populer | Q1..<Q3 cukup populer | >= Q3 sangat populer
  biner   : >= Q3 sangat populer vs lainnya

Protokol (bebas kebocoran):
  - split 80/20 deterministik via hash: per lagu (default) atau per artis (--split artist:
    artis di test tidak pernah muncul di train -> menguji generalisasi, bukan hafalan gaya artis)
  - TF-IDF / SMOTE / scaler di dalam pipeline -> hanya melihat data latih (juga di tiap fold CV)
  - tuning: GridSearchCV 5-fold stratified pada train; evaluasi sekali pada test
  - metrik test: accuracy, balanced accuracy, macro P/R/F1 + bootstrap 95% CI macro-F1

  python -m experiments.benchmark                          # M0-M4 (CPU)
  python -m experiments.benchmark --split artist --gpu     # + M5 embedding & M6 IndoBERT (GPU)
  python -m experiments.benchmark --semantic --only B0 M2 M7 M8   # fitur semantik (experiments/semantic.py)
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.config import DATA_DIR, ROOT
from src.text_utils import is_indonesian

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPORT_DIR = ROOT / "reports"
SEED = 42
N_JOBS = 4  # dibatasi: grid search paralel penuh (12 proses) sempat menghabiskan RAM
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PAPER_GRID = {"C": [0.01, 0.1, 1, 5, 10, 25, 50, 100], "kernel": ["linear", "rbf", "poly"],
              "gamma": ["auto", "scale", 0.001, 0.01, 0.1, 1]}


# ---------------- Dataset ----------------

def build_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "songs_with_lyrics.csv")
    n0 = len(df)
    df = df[df["lyrics_clean"].map(is_indonesian)]
    n_id = len(df)
    df = df.drop_duplicates(subset="lyrics_clean")
    print(f"Dataset: {n0} lagu berlirik -> {n_id} Indonesia -> {len(df)} setelah dedup lirik")
    df["score"] = df["total_streams"] / df["days"]  # popularity = streams per hari di chart
    return df.reset_index(drop=True)


def make_labels(score: pd.Series, task: str) -> tuple[pd.Series, list[str]]:
    q1, q3 = score.quantile([0.25, 0.75])
    if task == "3class":
        return (pd.Series(np.select([score < q1, score < q3], [0, 1], 2), index=score.index),
                ["kurang", "cukup", "sangat"])
    return (score >= q3).astype(int), ["lainnya", "sangat"]


def split_test(df: pd.DataFrame, by: str, frac: float = 0.2) -> np.ndarray:
    keys = df["artist"] if by == "artist" else df["artist"] + "|||" + df["title"]
    return np.array([int(hashlib.md5(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac for k in keys])


# ---------------- Methods ----------------

def methods(gpu: bool, semantic: bool = False) -> dict:
    lr = dict(max_iter=3000, class_weight="balanced")
    m = {
        "B0 Baseline kelas mayoritas": (Pipeline([("tfidf", TfidfVectorizer()),
                                                  ("clf", DummyClassifier(strategy="most_frequent"))]), {}, "accuracy"),
        # Replikasi paper: TF-IDF default + SMOTE + One-vs-Rest SVC, grid & skor (accuracy) seperti paper
        "M0 Paper: TF-IDF+SMOTE+SVM (OvR)": (
            ImbPipeline([("tfidf", TfidfVectorizer()), ("smote", SMOTE(random_state=SEED)),
                         ("clf", OneVsRestClassifier(SVC()))]),
            {f"clf__estimator__{k}": v for k, v in PAPER_GRID.items()}, "accuracy"),
        "M1 TF-IDF + LinearSVC (class_weight)": (
            Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)),
                      ("clf", LinearSVC(class_weight="balanced"))]),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.01, 0.1, 1, 10]}, "f1_macro"),
        "M2 TF-IDF + LogReg (class_weight)": (
            Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)),
                      ("clf", LogisticRegression(**lr))]),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 1, 10]}, "f1_macro"),
        "M3 TF-IDF kata+karakter + LogReg": (
            Pipeline([("tfidf", FeatureUnion([
                          ("word", TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))),
                          ("char", TfidfVectorizer(sublinear_tf=True, min_df=3, analyzer="char_wb",
                                                   ngram_range=(2, 5), max_features=50000))])),
                      ("clf", LogisticRegression(**lr))]),
            {"clf__C": [0.1, 1, 10]}, "f1_macro"),
        "M4 TF-IDF + Complement NB": (
            Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)), ("clf", ComplementNB())]),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 0.5, 1.0]}, "f1_macro"),
    }
    if gpu:
        m["M5 Embedding multibahasa + LogReg"] = (
            Pipeline([("emb", Embedder()), ("clf", LogisticRegression(**lr))]),
            {"clf__C": [0.01, 0.1, 1, 10]}, "f1_macro")
        m["M6 Fine-tuning IndoBERT (3-seed ensemble)"] = None  # bukan sklearn, lihat finetune.py
    if semantic:
        from experiments.semantic import FEATURES
        m["M7 Semantik: tema+emosi+struktur + LogReg"] = (
            Pipeline([("features", ColumnTransformer([("sem", StandardScaler(), FEATURES)])),
                      ("clf", LogisticRegression(**lr))]),
            {"clf__C": [0.01, 0.1, 1, 10]}, "f1_macro")
        m["M8 TF-IDF + semantik + LogReg"] = (
            Pipeline([("features", ColumnTransformer([
                          ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)), "lyrics_clean"),
                          ("sem", StandardScaler(), FEATURES)])),
                      ("clf", LogisticRegression(**lr))]),
            {"features__transformer_weights": [{"tfidf": 1, "sem": w} for w in (0.03, 0.1, 0.3)],
             "clf__C": [0.1, 1, 10]}, "f1_macro")
    return m


class Embedder:
    """Sentence embedding (mean-pool per potongan 100 kata) dengan cache disk. Stateless -> aman
    dipakai di dalam pipeline CV (tidak ada fitting pada data)."""
    _cache: dict = {}

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name

    def get_params(self, deep=True):
        return {"model_name": self.model_name}

    def set_params(self, **p):
        self.__dict__.update(p)
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cache = self._load()
        missing = [t for t in X if self._key(t) not in cache]
        if missing:
            import torch
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name,
                                        device="cuda" if torch.cuda.is_available() else "cpu")
            for t in missing:
                words = t.split()
                chunks = [" ".join(words[i:i + 100]) for i in range(0, max(len(words), 1), 100)]
                cache[self._key(t)] = model.encode(chunks, normalize_embeddings=True).mean(axis=0)
            np.save(self._path(), cache, allow_pickle=True)
        return np.vstack([cache[self._key(t)] for t in X])

    def _key(self, t):
        return hashlib.md5(t.encode()).hexdigest()

    def _path(self) -> Path:
        return DATA_DIR / f"embeddings_{self.model_name.split('/')[-1]}.npy"

    def _load(self) -> dict:
        if self.model_name not in self._cache:
            p = self._path()
            self._cache[self.model_name] = np.load(p, allow_pickle=True).item() if p.exists() else {}
        return self._cache[self.model_name]


# ---------------- Evaluation ----------------

def bootstrap_f1(y_true, y_pred, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = [f1_score(y_true[i], y_pred[i], average="macro", zero_division=0)
             for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def test_metrics(y_te, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
    lo, hi = bootstrap_f1(y_te, y_pred)
    return {"accuracy": accuracy_score(y_te, y_pred), "balanced_acc": balanced_accuracy_score(y_te, y_pred),
            "precision_macro": p, "recall_macro": r, "f1_macro": f, "f1_ci_low": lo, "f1_ci_high": hi}


def evaluate_finetune(name, X_tr, y_tr, X_te, y_te, n_classes: int) -> dict:
    from experiments.finetune import finetune_predict
    t0 = time.time()
    y_pred, info = finetune_predict(X_tr, y_tr, X_te, n_classes)
    val = np.mean([s["val_f1"] for s in info["seeds"]])
    return {"method": name, **test_metrics(y_te, y_pred), "cv_score": val, "cv_metric": "val_f1_macro",
            "best_params": json.dumps(info), "seconds": round(time.time() - t0, 1)}


def evaluate(name, pipe, grid, scoring, X_tr, y_tr, X_te, y_te) -> dict:
    t0 = time.time()
    if "emb" in pipe.named_steps:
        # Hitung embedding sekali di proses utama -> worker CV hanya membaca cache disk
        # (tanpa ini tiap worker memuat model transformer sendiri dan memori habis).
        pipe.named_steps["emb"].transform(pd.concat([X_tr, X_te]))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid or {"clf": [pipe.named_steps["clf"]]}, scoring=scoring, cv=cv,
                          n_jobs=N_JOBS).fit(X_tr, y_tr)
    best = {k: v for k, v in search.best_params_.items() if k != "clf"}
    return {"method": name, **test_metrics(y_te, search.predict(X_te)),
            "cv_score": search.best_score_, "cv_metric": scoring,
            "best_params": json.dumps(best, default=str), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["song", "artist"], default="song")
    ap.add_argument("--tasks", nargs="+", default=["3class", "binary"])
    ap.add_argument("--gpu", action="store_true", help="sertakan M5 embedding & M6 fine-tuning IndoBERT")
    ap.add_argument("--semantic", action="store_true", help="sertakan M7/M8 (fitur semantik, butuh GPU sekali)")
    ap.add_argument("--only", nargs="*", help="jalankan metode tertentu saja, mis. M0 M2")
    args = ap.parse_args()

    df = build_dataset()
    if args.semantic:
        from experiments.semantic import semantic_features
        df = pd.concat([df, semantic_features(df)], axis=1)
    test = split_test(df, args.split)
    print(f"Split per {args.split}: train {(~test).sum()} | test {test.sum()} | "
          f"artis di keduanya: {len(set(df.artist[test]) & set(df.artist[~test]))}")

    REPORT_DIR.mkdir(exist_ok=True)
    for task in args.tasks:
        y, names = make_labels(df["score"], task)
        X = df["lyrics_clean"]
        X_tr, X_te, y_tr, y_te = X[~test], X[test], y[~test], y[test]
        F_tr, F_te = df[~test], df[test]  # M7/M8 butuh kolom fitur semantik, bukan hanya teks
        dist = ", ".join(f"{n}={c}" for n, c in zip(names, np.bincount(y, minlength=len(names))))
        print(f"\n=== {task} | split={args.split} | {dist} ===")
        rows = []
        for name, spec in methods(args.gpu, args.semantic).items():
            if args.only and name.split()[0] not in args.only:
                continue
            if spec is None:
                res = evaluate_finetune(name, X_tr, y_tr, X_te, y_te, len(names))
            elif name.split()[0] in ("M7", "M8"):
                res = evaluate(name, *spec, F_tr, y_tr, F_te, y_te)
            else:
                res = evaluate(name, *spec, X_tr, y_tr, X_te, y_te)
            rows.append({"task": task, "split": args.split,
                         "n_train": len(X_tr), "n_test": len(X_te), **res})
            print(f"  {name:<40} acc={res['accuracy']:.3f} bal={res['balanced_acc']:.3f} "
                  f"F1={res['f1_macro']:.3f} [{res['f1_ci_low']:.3f}-{res['f1_ci_high']:.3f}] "
                  f"({res['seconds']}s)", flush=True)
        out = REPORT_DIR / f"benchmark_{task}_{args.split}.csv"
        if out.exists():  # gabung dengan hasil run sebelumnya: baris metode yang sama diganti
            rows = pd.concat([pd.read_csv(out).loc[lambda d: ~d.method.isin([r["method"] for r in rows])],
                              pd.DataFrame(rows)]).to_dict(orient="records")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
