"""Analisis fitur semantik: (1) fitur mana yang berasosiasi dengan popularitas, (2) apakah M8
(TF-IDF + semantik) benar-benar lebih baik dari M2 (TF-IDF saja) — uji bootstrap berpasangan.

(1) Spearman rho tiap fitur vs log(streams/hari) pada seluruh 1.260 lagu (analisis deskriptif,
    bukan seleksi model), p-value dikoreksi Benjamini-Hochberg untuk 30 uji.
(2) Untuk tiap skenario: latih M2 & M8 (hyperparameter terbaik dari CSV benchmark) pada train,
    prediksi test, lalu bootstrap selisih macro-F1 pada lagu test yang SAMA (1.000 resample).

  python -m experiments.semantic_analysis
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import f1_score

from experiments.benchmark import REPORT_DIR, SEED, build_dataset, make_labels, methods, split_test
from experiments.semantic import FEATURES, semantic_features


def bh_adjust(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def associations(df: pd.DataFrame) -> pd.DataFrame:
    y = np.log10(df["score"])
    rows = [(f, *spearmanr(df[f], y)) for f in FEATURES]
    res = pd.DataFrame(rows, columns=["feature", "rho", "p"])
    res["p_bh"] = bh_adjust(res["p"].to_numpy())
    return res.sort_values("rho", key=abs, ascending=False)


def _with_params(pipe, params_json: str):
    params = json.loads(params_json)
    if "features__transformer_weights" in params:  # dari CSV: dict tersimpan sebagai string json
        w = params["features__transformer_weights"]
        params["features__transformer_weights"] = json.loads(w.replace("'", '"')) if isinstance(w, str) else w
    for k, v in list(params.items()):
        if isinstance(v, list):
            params[k] = tuple(v)
    return clone(pipe).set_params(**params)


def paired_bootstrap(df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    m = methods(gpu=False, semantic=True)
    m2, m8 = m["M2 TF-IDF + LogReg (class_weight)"][0], m["M8 TF-IDF + semantik + LogReg"][0]
    rng = np.random.default_rng(SEED)
    rows = []
    for split in ("song", "artist"):
        test = split_test(df, split)
        for task in ("3class", "binary"):
            bench = pd.read_csv(REPORT_DIR / f"benchmark_{task}_{split}.csv").set_index("method")
            y, _ = make_labels(df["score"], task)
            p2 = _with_params(m2, bench.loc["M2 TF-IDF + LogReg (class_weight)", "best_params"])
            p8 = _with_params(m8, bench.loc["M8 TF-IDF + semantik + LogReg", "best_params"])
            yp2 = p2.fit(df.loc[~test, "lyrics_clean"], y[~test]).predict(df.loc[test, "lyrics_clean"])
            yp8 = p8.fit(df[~test], y[~test]).predict(df[test])
            yt = y[test].to_numpy()
            diffs = []
            for _ in range(n):
                i = rng.integers(0, len(yt), len(yt))
                diffs.append(f1_score(yt[i], yp8[i], average="macro") - f1_score(yt[i], yp2[i], average="macro"))
            diffs = np.array(diffs)
            rows.append({"task": task, "split": split,
                         "f1_M2": f1_score(yt, yp2, average="macro"), "f1_M8": f1_score(yt, yp8, average="macro"),
                         "delta": f1_score(yt, yp8, average="macro") - f1_score(yt, yp2, average="macro"),
                         "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
                         "p_M8_not_better": float((diffs <= 0).mean())})
    return pd.DataFrame(rows)


def main():
    df = build_dataset()
    df = pd.concat([df, semantic_features(df)], axis=1)

    assoc = associations(df)
    assoc.to_csv(REPORT_DIR / "semantic_associations.csv", index=False)
    print("Spearman rho vs log10(streams/hari), n =", len(df))
    print(assoc.round(4).to_string(index=False))

    pb = paired_bootstrap(df)
    pb.to_csv(REPORT_DIR / "semantic_paired_bootstrap.csv", index=False)
    print("\nM8 (TF-IDF + semantik) vs M2 (TF-IDF), bootstrap berpasangan pada test set yang sama:")
    print(pb.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
