"""Tahap 6 — Antarmuka web mobile-first (Streamlit) untuk prediksi viralitas lirik."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# Streamlit secrets -> env, agar src.config/monitoring membacanya dengan cara yang sama
for _k in ("HF_TOKEN", "HF_MODEL_REPO", "HF_DATASET_REPO", "HF_MODEL_REVISION"):
    try:
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = str(st.secrets[_k])
    except FileNotFoundError:  # tidak ada secrets.toml (lokal)
        break

from src.config import HF_MODEL_REPO, HF_MODEL_REVISION, MODEL_DIR  # noqa: E402
from src.monitoring import PredictionLogger  # noqa: E402
from src.text_utils import clean_lyrics  # noqa: E402

TIERS = [(0.65, "Tinggi", "🔥", "#1DB954"), (0.40, "Sedang", "✨", "#F5A623"), (0.0, "Rendah", "🌱", "#E05252")]

st.set_page_config(page_title="Tralala-Trilili", page_icon="🎵", layout="centered")
st.markdown(
    """
    <style>
      .block-container {max-width: 640px; padding: 1.5rem 1rem 3rem;}
      textarea {font-size: 16px !important;}  /* cegah auto-zoom di iOS */
      .stButton > button {width: 100%; border-radius: 999px; font-weight: 700; padding: .7rem;}
      .result {border-radius: 16px; padding: 1.25rem; text-align: center; margin-top: 1rem;
               background: #1E1E1E; border: 2px solid var(--c);}
      .result .pct {font-size: 3rem; font-weight: 800; color: var(--c); line-height: 1.1;}
      .result .tier {font-size: 1.1rem; font-weight: 600;}
      .chip {display: inline-block; padding: .2rem .6rem; margin: .15rem; border-radius: 999px;
             background: #2A2A2A; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(ttl=3600, show_spinner="Memuat model…")
def load_model():
    """Ambil model produksi dari HF registry (dicek ulang tiap jam -> rilis baru otomatis
    terpakai tanpa redeploy). Fallback ke models/ lokal bila Hub tidak terjangkau."""
    source = f"{HF_MODEL_REPO}@{HF_MODEL_REVISION}"
    try:
        from src.registry import download_model
        model_dir = download_model() or MODEL_DIR
    except Exception:  # jaringan / Hub bermasalah
        model_dir = MODEL_DIR
    if model_dir == MODEL_DIR:
        source = "lokal (models/)"
    model = joblib.load(model_dir / "virality_model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text())
    return model, meta, source


@st.cache_resource
def get_logger():
    return PredictionLogger()


def tier_of(p: float):
    return next(t for t in TIERS if p >= t[0])


def top_terms(model, text: str, k: int = 8):
    """Kata dalam lirik yang paling mendorong ke arah 'Viral' (hanya untuk model linear)."""
    clf, vec = model.named_steps["clf"], model.named_steps["tfidf"]
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return []
    coef = coef.toarray().ravel() if hasattr(coef, "toarray") else np.asarray(coef).ravel()
    x = vec.transform([text])
    contrib = x.multiply(coef).tocsr()
    idx = contrib.indices[np.argsort(contrib.data)[::-1]]
    names = vec.get_feature_names_out()
    return [names[i] for i in idx[:k] if coef[i] > 0]


model, meta, model_source = load_model()

st.title("🎵 Tralala-Trilili")
st.caption("Prediksi peluang viral lirik lagu berdasarkan data historis Spotify Indonesia.")

lyrics = st.text_area("Lirik lagu kamu", height=260,
                      placeholder="Tempel atau tulis lirik lagu di sini…")

if st.button("Prediksi Viralitas", type="primary"):
    cleaned = clean_lyrics(lyrics)
    if len(cleaned.split()) < 20:
        st.warning("Lirik terlalu pendek — masukkan minimal ±20 kata agar prediksi bermakna.")
    else:
        p = float(model.predict_proba([cleaned])[0, 1])
        _, label, icon, color = tier_of(p)
        get_logger().log(p, label, len(cleaned.split()), meta.get("trained_at", meta["trained_on"]))
        st.markdown(
            f'<div class="result" style="--c:{color}"><div class="pct">{p:.0%}</div>'
            f'<div class="tier">{icon} Potensi {label}</div></div>',
            unsafe_allow_html=True,
        )
        terms = top_terms(model, cleaned)
        if terms:
            st.markdown("**Kata pendorong viral:** " +
                        "".join(f'<span class="chip">{t}</span>' for t in terms),
                        unsafe_allow_html=True)

with st.expander("Tentang model"):
    m = meta["test_metrics"]
    st.markdown(
        f"- Model: **{meta['best_model']}** (TF-IDF), dilatih {meta['trained_on']} "
        f"pada **{meta['n_songs']:,}** lagu dari chart Spotify Indonesia (kworb.net).\n"
        f"- *Viral* = rasio total streams / hari di chart ≥ "
        f"**{meta['viral_threshold_streams_per_day']:,.0f}** (top {meta['viral_rate']:.0%}).\n"
        f"- Test set: accuracy {m['accuracy']:.2f} · precision {m['precision']:.2f} · "
        f"recall {m['recall']:.2f} · F1 {m['f1']:.2f}.\n"
        f"- Tingkat potensi: Rendah < 40% ≤ Sedang < 65% ≤ Tinggi.\n"
        f"- Sumber model: `{model_source}`.\n\n"
        "Lirik hanyalah salah satu faktor viralitas — artis, promosi, dan aransemen "
        "musik juga sangat berpengaruh. Gunakan hasil ini sebagai indikasi, bukan kepastian."
    )
    st.dataframe(meta["benchmark"], hide_index=True, width="stretch")
