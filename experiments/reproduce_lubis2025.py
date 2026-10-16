"""Reproduksi Lubis & Huda (2025), "Implementation of SVM Algorithm to Predict Song Popularity
based on Sentiment Analysis of Lyrics", JAIC 9(2):265-272.

Sumber: paper (references/*.pdf) + repo resmi github.com/QuiinLatifah/Lyrics-Popularity-Prediction
(dataset songs.csv = references/lirik-lagu-dataset/songs.csv, identik dengan repo).

Protokol yang dibandingkan:
  R0  repo-exact      : notebook repo apa adanya (stopword ID+EN, lemmatize, token>3,
                        label <=33/<80/>=80, TF-IDF & SMOTE pada SELURUH data lalu split, SVC default)
  R1  paper-leaky     : metode di paper (hanya cleaning + case folding, label kuartil,
                        GridSearch C/kernel/gamma, One-vs-Rest), urutan SMOTE->split seperti paper
  R2  paper-leakfree  : metode R1, tetapi split dulu; TF-IDF + SMOTE di dalam pipeline sehingga
                        hanya melihat data latih (termasuk di tiap fold CV)
  R3  paper-leakfree+dedup : R2 setelah membuang "Lyrics not found" & lirik duplikat
  Bx  baseline mayoritas pada test set yang sama (acuan "tanpa sinyal")

Jalankan:  python -m experiments.reproduce_lubis2025
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "references" / "lirik-lagu-dataset" / "songs.csv"
DATA_URL = "https://raw.githubusercontent.com/QuiinLatifah/Lyrics-Popularity-Prediction/main/songs.csv"
OUT = ROOT / "reports" / "reproduction_lubis2025.csv"
SEED = 42
PAPER_GRID = {"C": [0.01, 0.1, 1, 5, 10, 25, 50, 100], "kernel": ["linear", "rbf", "poly"],
              "gamma": ["auto", "scale", 0.001, 0.01, 0.1, 1]}


# ---------- Preprocessing ----------

def clean_repo(text: str) -> str:
    """clean_lyrics() dari notebook repo (sel 11), disalin apa adanya."""
    text = text.split("\n", 1)[-1]
    text = re.sub(r".*lyrics\s*[:\[]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcontributors\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def repo_text(df: pd.DataFrame) -> pd.Series:
    """Sel 12-17 repo: tokenisasi, stopword ID+EN, lemmatize WordNet, buang token <= 3 huruf."""
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    tok, lem = RegexpTokenizer(r"\w+"), WordNetLemmatizer()
    stop = set(stopwords.words("indonesian")) | set(stopwords.words("english"))
    return df["Lyrics"].map(clean_repo).map(lambda t: " ".join(
        w for w in (lem.lemmatize(x) for x in tok.tokenize(t) if x.lower() not in stop) if len(w) > 3))


def paper_text(df: pd.DataFrame) -> pd.Series:
    """Paper §II.C: cleaning (tanda baca, angka, URL, non a-z) + case folding; stopword DIPERTAHANKAN."""
    def clean(t: str) -> str:
        t = clean_repo(t)  # buang header Genius ("N Contributors ... Lyrics")
        t = re.sub(r"http\S+|#\w+|@\w+", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t.lower())
        return re.sub(r"\s+", " ", t).strip()
    return df["Lyrics"].map(clean)


# ---------- Label ----------

def label_repo(pop: pd.Series) -> pd.Series:
    return pd.Series(np.select([pop <= 33, pop < 80], [0, 1], 2), index=pop.index)


def label_quartile(pop: pd.Series) -> pd.Series:
    """Paper §II.B: < Q1 kurang populer, Q1..<Q3 cukup populer, >= Q3 sangat populer."""
    q1, q3 = pop.quantile([0.25, 0.75])
    return pd.Series(np.select([pop < q1, pop < q3], [0, 1], 2), index=pop.index)


# ---------- Evaluasi ----------

def scores(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p, "recall_macro": r,
            "f1_macro": f}


def run_leaky(X_text, y, svc_grid: dict | None) -> dict:
    """Urutan paper/repo: TF-IDF(fit semua) -> SMOTE(semua) -> split 80/20 -> SVM."""
    X = TfidfVectorizer().fit_transform(X_text)
    X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sm, y_sm, test_size=0.2, random_state=SEED)
    if svc_grid is None:
        model, params = SVC().fit(X_tr, y_tr), "default"
    else:
        grid = {f"estimator__{k}": v for k, v in svc_grid.items()}
        search = GridSearchCV(OneVsRestClassifier(SVC()), grid, cv=5, n_jobs=-1).fit(X_tr, y_tr)
        model, params = search.best_estimator_, json.dumps(search.best_params_)
    cv = cross_val_score(model, X_sm, y_sm, cv=5, n_jobs=-1)
    return {**scores(y_te, model.predict(X_te)), "cv_acc_mean": cv.mean(), "cv_acc_std": cv.std(),
            "n_train": X_tr.shape[0], "n_test": X_te.shape[0], "best_params": params}


def run_leakfree(X_text, y, svc_grid: dict) -> tuple[dict, dict]:
    """Split dulu (stratified); TF-IDF + SMOTE + OvR-SVM dalam satu pipeline."""
    X_tr, X_te, y_tr, y_te = train_test_split(X_text, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = ImbPipeline([("tfidf", TfidfVectorizer()), ("smote", SMOTE(random_state=SEED)),
                        ("clf", OneVsRestClassifier(SVC()))])
    grid = {f"clf__estimator__{k}": v for k, v in svc_grid.items()}
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, cv=cv5, n_jobs=-1).fit(X_tr, y_tr)
    cv = cross_val_score(search.best_estimator_, X_text, y, cv=cv5, n_jobs=-1)
    res = {**scores(y_te, search.predict(X_te)), "cv_acc_mean": cv.mean(), "cv_acc_std": cv.std(),
           "n_train": len(X_tr), "n_test": len(X_te), "best_params": json.dumps(search.best_params_)}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    base = {**scores(y_te, dummy.predict(X_te)), "n_train": len(X_tr), "n_test": len(X_te),
            "best_params": "most_frequent"}
    return res, base


def main():
    if not DATA.exists():  # references/ tidak di-commit: ambil dari repo resmi paper
        import requests
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_bytes(requests.get(DATA_URL, timeout=60).content)
    df = pd.read_csv(DATA)
    print(f"Dataset: {len(df)} lagu | 'Lyrics not found': {(df.Lyrics == 'Lyrics not found').sum()} | "
          f"lirik duplikat: {df.Lyrics.duplicated().sum()}")
    pop = df["Popularity"]
    print(f"Popularity Q1={pop.quantile(.25):.0f} median={pop.median():.0f} Q3={pop.quantile(.75):.0f}")

    rows = []

    def add(rid, desc, res):
        rows.append({"id": rid, "protocol": desc, **res})
        print(f"{rid:<4} {desc:<52} acc={res['accuracy']:.3f}  F1={res['f1_macro']:.3f}"
              + (f"  cv={res['cv_acc_mean']:.3f}±{res['cv_acc_std']:.3f}" if "cv_acc_mean" in res else ""))

    y_repo, y_q = label_repo(pop), label_quartile(pop)
    add("R0", "repo-exact (SMOTE sebelum split, SVC default)", run_leaky(repo_text(df), y_repo, None))
    Xp = paper_text(df)
    add("R1", "paper-leaky (SMOTE sebelum split, GridSearch OvR)", run_leaky(Xp, y_q, PAPER_GRID))
    r2, b2 = run_leakfree(Xp, y_q, PAPER_GRID)
    add("R2", "paper-leakfree (split -> TF-IDF+SMOTE di pipeline)", r2)
    add("B2", "baseline kelas mayoritas (test set R2)", b2)
    clean = df[df.Lyrics.ne("Lyrics not found") & ~df.Lyrics.duplicated()]
    r3, b3 = run_leakfree(paper_text(clean), label_quartile(clean["Popularity"]), PAPER_GRID)
    add("R3", f"paper-leakfree + dedup ({len(clean)} lagu)", r3)
    add("B3", "baseline kelas mayoritas (test set R3)", b3)

    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
