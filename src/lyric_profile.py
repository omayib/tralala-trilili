"""Profil lirik yang dapat dijelaskan: tema (leksikon/ontologi kecil) + struktur lagu.

Ringan (tanpa model ML) sehingga bisa dijalankan di app Streamlit. Acuan pembanding
(persentil di antara lagu chart, dan arah asosiasi dengan popularitas) dihitung saat training
oleh `reference_stats()` dan dirilis bersama model di metadata.json.
"""
import re

import numpy as np
import pandas as pd

from src.text_utils import clean_lyrics, language_ratios

# Ontologi tema: tema -> kata/frasa penanda (lirik sudah dibersihkan, huruf kecil)
THEMES = {
    "patah_hati": ["patah hati", "hancur", "luka", "terluka", "perih", "air mata", "menangis", "tangis",
                   "kecewa", "remuk", "sakit hati", "menyakiti", "disakiti"],
    "rindu": ["rindu", "merindu", "merindukan", "kangen", "jarak", "menantimu", "menunggumu", "jauh darimu"],
    "merelakan": ["selamat tinggal", "berpisah", "perpisahan", "melepasmu", "melepaskan", "merelakan",
                  "relakan", "ikhlas", "usai", "berakhir", "pergilah"],
    "cinta_bahagia": ["bahagia", "tersenyum", "senyummu", "pelukmu", "memelukmu", "jatuh cinta",
                      "berdua", "selamanya", "indahnya", "sempurna"],
    "pengkhianatan": ["dusta", "bohong", "berbohong", "khianat", "mengkhianati", "selingkuh",
                      "berselingkuh", "tega", "ingkar", "janji palsu", "mendua"],
    "kesepian": ["sepi", "sendiri", "sendirian", "kesepian", "sunyi", "hampa", "lelah", "sendu"],
    "keluarga": ["ibu", "ayah", "bunda", "mama", "papa", "keluarga", "anakku", "adikku", "kakakku",
                 "orang tua", "pulang ke rumah"],
    "religi": ["tuhan", "allah", "doa", "berdoa", "doaku", "ampun", "ampuni", "dosa", "surga", "ilahi",
               "sujud", "iman", "rasul", "nabi"],
    "pesta_joget": ["goyang", "joget", "berjoget", "dansa", "berdansa", "menari", "pesta", "party",
                    "senang senang", "hore"],
    "semangat": ["semangat", "berjuang", "perjuangan", "bangkit", "menyerah", "pantang", "raih",
                 "harapan", "kuat"],
    "persahabatan": ["sahabat", "sahabatku", "teman", "temanku", "kawan", "kawanku"],
}
THEME_LABELS = {
    "patah_hati": "patah hati", "rindu": "rindu", "merelakan": "merelakan/perpisahan",
    "cinta_bahagia": "cinta bahagia", "pengkhianatan": "pengkhianatan", "kesepian": "kesepian",
    "keluarga": "keluarga", "religi": "religi", "pesta_joget": "pesta/joget", "semangat": "semangat",
    "persahabatan": "persahabatan",
}
STRUCTURE = ["n_words", "n_lines", "words_per_line", "type_token_ratio", "unique_line_ratio",
             "repeated_line_share", "top_line_repeats", "rhyme_rate", "english_share",
             "id_function_share", "pron_aku_kamu", "pron_gue_lo", "pron_saya_anda"]
THEME_COLS = [f"lex_{t}" for t in THEMES]
# Fitur yang ditampilkan di app: label, format nilai, arah bila asosiasi positif / negatif
SHOWN = {
    "n_words": ("Panjang lirik", "{:.0f} kata", "lebih panjang", "lebih pendek"),
    "words_per_line": ("Kata per baris", "{:.1f}", "barisnya lebih panjang", "barisnya lebih pendek"),
    "repeated_line_share": ("Baris yang berulang", "{:.0%}", "lebih repetitif", "kurang repetitif"),
    "top_line_repeats": ("Hook diulang", "{:.0f}×", "hook-nya lebih sering diulang",
                         "hook-nya lebih jarang diulang"),
    "type_token_ratio": ("Keragaman kosakata", "{:.2f}", "kosakatanya lebih beragam",
                         "kosakatanya kurang beragam"),
}
_PATTERNS = {t: [re.compile(rf"\b{re.escape(w)}\b") for w in words] for t, words in THEMES.items()}


def _lines(raw: str) -> list[str]:
    lines = [clean_lyrics(line) for line in str(raw).splitlines()]
    return [line for line in lines if line]


def structure_features(raw: str) -> dict:
    lines = _lines(raw)
    words = " ".join(lines).split()
    n_w, n_l = len(words), max(len(lines), 1)
    counts = pd.Series(lines).value_counts() if lines else pd.Series(dtype=int)
    ends = [line.split()[-1] for line in lines]
    rhymes = sum(a[-2:] == b[-2:] for a, b in zip(ends, ends[1:]))
    id_share, en_share = language_ratios(" ".join(words))
    per100 = lambda vocab: sum(w in vocab for w in words) / max(n_w, 1) * 100
    return {
        "n_words": n_w, "n_lines": len(lines), "words_per_line": n_w / n_l,
        "type_token_ratio": len(set(words)) / max(n_w, 1),
        "unique_line_ratio": len(counts) / n_l,
        "repeated_line_share": float(counts[counts > 1].sum() / n_l) if len(counts) else 0.0,
        "top_line_repeats": int(counts.max()) if len(counts) else 0,
        "rhyme_rate": rhymes / max(len(ends) - 1, 1),
        "english_share": en_share, "id_function_share": id_share,
        "pron_aku_kamu": per100({"aku", "kamu", "kau", "engkau", "ku", "mu", "dirimu", "diriku"}),
        "pron_gue_lo": per100({"gue", "gua", "gw", "lo", "lu", "elo", "elu"}),
        "pron_saya_anda": per100({"saya", "anda"}),
    }


def theme_scores(clean_text: str) -> dict:
    """Jumlah penanda tema per 100 kata."""
    n = max(len(clean_text.split()), 1)
    return {f"lex_{t}": sum(len(p.findall(clean_text)) for p in pats) / n * 100
            for t, pats in _PATTERNS.items()}


def profile(raw: str) -> dict:
    return {**structure_features(raw), **theme_scores(clean_lyrics(raw))}


# ---------------- Acuan (dihitung saat training, disimpan di metadata) ----------------

def _bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def reference_stats(raws: pd.Series, popularity: pd.Series) -> dict:
    """Persentil tiap fitur di antara lagu acuan + Spearman rho vs log popularitas (koreksi BH)."""
    from scipy.stats import spearmanr
    feats = pd.DataFrame([profile(r) for r in raws])
    y = np.log10(popularity.to_numpy())
    cols = list(SHOWN) + THEME_COLS
    # Fitur konstan (mis. tema yang tak pernah muncul) -> korelasi tak terdefinisi = tanpa asosiasi
    rho_p = [spearmanr(feats[c], y) if feats[c].nunique() > 1 else (0.0, 1.0) for c in cols]
    p_bh = _bh(np.array([float(p) for _, p in rho_p]))
    return {c: {"quantiles": np.quantile(feats[c], np.linspace(0, 1, 101)).round(4).tolist(),
                "mean": float(feats[c].mean()), "std": float(feats[c].std()),
                "rho": float(r), "p_bh": float(p)}
            for c, (r, _), p in zip(cols, rho_p, p_bh)}


def percentile(value: float, quantiles: list[float]) -> float:
    """Persentase lagu acuan dengan nilai fitur di bawah `value` (0-100)."""
    return float(np.interp(value, quantiles, np.linspace(0, 100, len(quantiles))))


def explain(raw: str, ref: dict, alpha: float = 0.05) -> dict:
    """Tema menonjol + baris struktur dengan petunjuk berbasis data untuk ditampilkan di app."""
    prof = profile(raw)
    themes = []
    for t in THEMES:
        r = ref.get(f"lex_{t}")
        if r and r["std"] > 0 and prof[f"lex_{t}"] > 0:
            z = (prof[f"lex_{t}"] - r["mean"]) / r["std"]
            if z >= 1:
                themes.append((THEME_LABELS[t], z, r["rho"] if r["p_bh"] < alpha else None))
    themes.sort(key=lambda x: -x[1])
    rows = []
    for c, (label, fmt, up, down) in SHOWN.items():
        r = ref.get(c)
        if not r:
            continue
        pct = percentile(prof[c], r["quantiles"])
        position = (f"lebih tinggi dari {pct:.0f}% lagu chart" if pct >= 50
                    else f"lebih rendah dari {100 - pct:.0f}% lagu chart")
        hint = f"lagu populer cenderung {up if r['rho'] > 0 else down}" if r["p_bh"] < alpha else None
        rows.append({"label": label, "value": fmt.format(prof[c]), "percentile": pct,
                     "position": position, "hint": hint})
    return {"themes": themes[:3], "structure": rows}
