"""Tahap 6 — Antarmuka web mobile-first (Streamlit) untuk prediksi popularitas lirik."""
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
from src.lyric_profile import explain  # noqa: E402
from src.monitoring import PredictionLogger  # noqa: E402
from src.text_utils import clean_lyrics, is_indonesian  # noqa: E402

# (nama kelas model untuk log, label tampilan, ajakan, ikon, warna): 0 kurang, 1 cukup, 2 sangat populer
CLASSES = [("Kurang populer", "Perlu revisi", "Perkuat hook-nya, lalu coba lagi", "🌱", "#E05252"),
           ("Cukup populer", "Hampir viral", "Poles lagi biar makin viral", "✨", "#F5A623"),
           ("Sangat populer", "Berpotensi viral", "Ayo nyanyikan!", "🔥", "#1DB954")]
PAPER = ("Lubis, Q. L. A., & Huda, A. A. (2025). Implementation of SVM Algorithm to Predict Song "
         "Popularity based on Sentiment Analysis of Lyrics. *JAIC*, 9(2), 265–272.")

st.set_page_config(page_title="Tralala-Trilili", page_icon="🎵", layout="centered")
st.markdown(
    """
    <style>
      .block-container {max-width: 640px; padding: 1.5rem 1rem 3rem;}
      textarea {font-size: 16px !important;}  /* cegah auto-zoom di iOS */
      .stButton > button {width: 100%; border-radius: 999px; font-weight: 700; padding: .7rem;}
      .result {border-radius: 16px; padding: 1.25rem; text-align: center; margin-top: 1rem;
               background: #1E1E1E; border: 2px solid var(--c);}
      .result .pct {font-size: 2.2rem; font-weight: 800; color: var(--c); line-height: 1.1;}
      .result .tier {font-size: 1.1rem; font-weight: 600;}
      .bar {display: flex; align-items: center; gap: .5rem; margin: .35rem 0; font-size: .9rem;}
      .bar .name {width: 8.5rem; flex: none;}
      .bar .track {flex: 1; height: .6rem; border-radius: 999px; background: #2A2A2A; overflow: hidden;}
      .bar .fill {height: 100%; border-radius: 999px;}
      .bar .val {width: 2.8rem; text-align: right; flex: none;}
      .chip {display: inline-block; padding: .2rem .6rem; margin: .15rem; border-radius: 999px;
             background: #2A2A2A; font-size: .85rem;}
      .prof {display: grid; grid-template-columns: 1fr auto; gap: .1rem .75rem; padding: .55rem 0;
             border-bottom: 1px solid #2A2A2A; font-size: .9rem;}
      .prof .v {font-weight: 700; text-align: right;}
      .prof .d {grid-column: 1 / -1; color: #A7A7A7; font-size: .8rem;}
      .note {color: #A7A7A7; font-size: .8rem; margin-top: .4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def release_sha() -> str | None:
    """Sha rilis terbaru di HF registry, dicek tiap 5 menit (murah: hanya metadata repo)."""
    from src.registry import latest_revision
    return latest_revision()


@st.cache_resource(max_entries=2, show_spinner="Memuat model…")
def load_model(sha: str | None):
    """Model + metadata untuk satu rilis. Di-cache per sha, jadi rilis baru terpakai dalam
    ±5 menit tanpa redeploy. Fallback ke models/ lokal bila Hub tidak terjangkau."""
    source = f"{HF_MODEL_REPO}@{(sha or HF_MODEL_REVISION)[:7]}"
    try:
        from src.registry import download_model
        model_dir = download_model(revision=sha or HF_MODEL_REVISION) or MODEL_DIR
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


def top_terms(model, text: str, k: int = 8):
    """Kata dalam lirik yang paling mendorong ke kelas 'Sangat populer' (hanya model linear)."""
    clf, vec = model.named_steps["clf"], model.named_steps["tfidf"]
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return []
    coef = coef.toarray() if hasattr(coef, "toarray") else np.asarray(coef)
    coef = coef[-1]  # baris kelas terakhir = Sangat populer (biner: satu-satunya baris)
    x = vec.transform([text])
    contrib = x.multiply(coef).tocsr()
    idx = contrib.indices[np.argsort(contrib.data)[::-1]]
    names = vec.get_feature_names_out()
    return [names[i] for i in idx[:k] if coef[i] > 0]


def render_profile(ex: dict) -> None:
    """Tema & struktur lirik dibanding lagu chart, plus arah asosiasi yang didukung data."""
    st.markdown("#### Profil lirik")
    if ex["themes"]:
        arrow = lambda rho: "" if rho is None else (" ↑" if rho > 0 else " ↓")
        st.markdown("**Tema menonjol:** " + "".join(
            f'<span class="chip">{label}{arrow(rho)}</span>' for label, _, rho in ex["themes"]),
            unsafe_allow_html=True)
    else:
        st.markdown("**Tema menonjol:** tidak ada yang menonjol dibanding lagu chart.")
    st.markdown("".join(
        f'<div class="prof"><span>{r["label"]}</span><span class="v">{r["value"]}</span>'
        f'<span class="d">{r["position"]}'
        + (f' · {r["hint"]}' if r["hint"] else "") + "</span></div>"
        for r in ex["structure"]), unsafe_allow_html=True)
    st.markdown('<div class="note">↑/↓ = tema lebih sering/jarang pada lagu populer. Petunjuk hanya '
                "ditampilkan bila asosiasinya signifikan pada data chart (koreksi Benjamini-Hochberg); "
                "asosiasi lemah dan bukan sebab-akibat.</div>", unsafe_allow_html=True)


model, meta, model_source = load_model(release_sha())

st.title("🎵 Tralala-Trilili")
st.caption("Prediksi tingkat popularitas lirik lagu berbahasa Indonesia berdasarkan data "
           "historis chart Spotify Indonesia.")

lyrics = st.text_area("Lirik lagu kamu", height=260,
                      placeholder="Tempel atau tulis lirik lagu di sini…")

if st.button("Prediksi Popularitas", type="primary", width="stretch"):
    cleaned = clean_lyrics(lyrics)
    if len(cleaned.split()) < 20:
        st.warning("Lirik terlalu pendek — masukkan minimal ±20 kata agar prediksi bermakna.")
    else:
        if not is_indonesian(cleaned):
            st.info("Lirik ini tampaknya bukan berbahasa Indonesia — model dilatih khusus pada "
                    "lirik Indonesia, jadi hasilnya kurang dapat diandalkan.")
        proba = model.predict_proba([cleaned])[0]
        k = int(np.argmax(proba))
        name, label, cta, icon, color = CLASSES[k]
        get_logger().log(float(proba[-1]), name, len(cleaned.split()),
                         meta.get("trained_at", meta["trained_on"]))
        st.markdown(
            f'<div class="result" style="--c:{color}"><div class="pct">{icon} {label}</div>'
            f'<div class="tier">{cta}</div><div class="note">keyakinan model {proba[k]:.0%}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("".join(
            f'<div class="bar"><span class="name">{n}</span><span class="track">'
            f'<span class="fill" style="display:block;width:{p:.0%};background:{c}"></span></span>'
            f'<span class="val">{p:.0%}</span></div>'
            for (_, n, _, _, c), p in zip(CLASSES, proba)), unsafe_allow_html=True)
        terms = top_terms(model, cleaned)
        if terms:
            st.markdown(f"**Kata pendorong '{CLASSES[-1][1]}':** " +
                        "".join(f'<span class="chip">{t}</span>' for t in terms),
                        unsafe_allow_html=True)
        if ref := meta.get("lyric_profile"):  # rilis model lama belum memuatnya
            render_profile(explain(lyrics, ref))

with st.expander("Tentang model"):
    m, thr = meta["test_metrics"], meta["thresholds_streams_per_day"]
    st.markdown(
        f"- Model: **{meta['best_model']}**, dilatih {meta['trained_on']} pada "
        f"**{meta['n_songs']:,}** lagu berbahasa Indonesia dari chart Spotify Indonesia (kworb.net).\n"
        f"- *Popularity* = total streams / jumlah hari di chart. Kelas (kuartil): {CLASSES[0][1]} < "
        f"**{thr['q1']:,.0f}** ≤ {CLASSES[1][1]} < **{thr['q3']:,.0f}** ≤ {CLASSES[2][1]} "
        "(streams/hari).\n"
        f"- Test set: accuracy {m['accuracy']:.2f} · balanced accuracy {m['balanced_accuracy']:.2f} · "
        f"macro-F1 {m['f1']:.2f}.\n"
        f"- Sumber model: `{model_source}`.\n\n"
        "Lirik hanyalah salah satu faktor popularitas — artis, promosi, dan aransemen musik juga "
        "sangat berpengaruh. Gunakan hasil ini sebagai indikasi, bukan kepastian.\n\n"
        f"Terinspirasi oleh {PAPER}"
    )
    st.dataframe(meta["benchmark"], hide_index=True, width="stretch")
