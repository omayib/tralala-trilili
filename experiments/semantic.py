"""Fitur semantik lirik: tema (ontologi kecil pop Indonesia), emosi, dan struktur lagu.

Tidak ada bagian yang dilatih pada label popularitas kita (prototipe tema ditulis tangan,
model emosi dilatih pada dataset eksternal IndoNLU EmoT), sehingga fitur aman dihitung sekali
untuk semua lagu tanpa kebocoran ke test set. Hasil di-cache per lirik.

  tema     : cosine similarity embedding lirik (paraphrase-multilingual-MiniLM, sama dengan M5)
             terhadap prototipe tiap tema (rata-rata embedding kalimat deskriptif)
  emosi    : IndoBERT di-fine-tune pada IndoNLU EmoT (anger/fear/happy/love/sadness),
             diterapkan per potongan lirik lalu dirata-rata
  struktur : repetisi/hook, keragaman kosakata, rima, panjang, porsi bahasa Inggris, register kata ganti
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.text_utils import clean_lyrics, language_ratios

CACHE = DATA_DIR / "semantic_features.csv"
EMOT_DIR = DATA_DIR / "emot_model"
EMOT_URL = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/{}.csv"
EMOTIONS = ["anger", "fear", "happy", "love", "sadness"]

# Ontologi tema: nama -> kalimat prototipe (bahasa lirik sehari-hari, bukan definisi kamus)
THEMES = {
    "cinta_bahagia": ["aku jatuh cinta padamu", "bahagia saat bersamamu", "kau membuatku tersenyum setiap hari"],
    "patah_hati": ["hatiku hancur karena kau pergi", "luka ini terlalu dalam", "air mata jatuh saat kau meninggalkanku"],
    "rindu": ["aku rindu padamu", "merindukan kehadiranmu", "jarak memisahkan kita berdua"],
    "merelakan": ["aku merelakanmu pergi", "ikhlas melepasmu bahagia dengannya", "selamat tinggal kasih"],
    "cinta_terpendam": ["cinta dalam diam", "tak berani mengungkapkan perasaanku", "kau tak pernah tahu aku mencintaimu"],
    "pengkhianatan": ["kau mengkhianati cintaku", "teganya kau berselingkuh", "semua janjimu hanya dusta"],
    "keluarga": ["ibu yang selalu mendoakanku", "ayah bekerja untuk keluarga", "pulang ke rumah orang tua"],
    "religi": ["ya Tuhan ampuni dosaku", "aku berdoa kepada Allah", "bersujud memohon ampunan"],
    "pesta_joget": ["ayo goyang semua", "berjoget sampai pagi", "pesta malam ini bersenang-senang"],
    "semangat": ["jangan menyerah teruslah berjuang", "bangkit dan raih mimpimu", "aku kuat menghadapi semua"],
    "persahabatan": ["sahabat selamanya", "teman yang selalu ada", "kita bersama melewati hari"],
    "kesepian": ["aku sendiri dalam sepi", "lelah menjalani hidup ini", "tak ada yang peduli padaku"],
}
STRUCTURE = ["n_words", "n_lines", "words_per_line", "type_token_ratio", "unique_line_ratio",
             "repeated_line_share", "top_line_repeats", "rhyme_rate", "english_share",
             "id_function_share", "pron_aku_kamu", "pron_gue_lo", "pron_saya_anda"]
THEME_COLS = [f"theme_{t}" for t in THEMES]
EMOTION_COLS = [f"emo_{e}" for e in EMOTIONS]
FEATURES = THEME_COLS + EMOTION_COLS + STRUCTURE


def _key(raw: str) -> str:
    return hashlib.md5(raw.encode()).hexdigest()


def _lines(raw: str) -> list[str]:
    lines = [clean_lyrics(l) for l in str(raw).splitlines()]
    return [l for l in lines if l]


# ---------------- Struktur ----------------

def structure_features(raw: str) -> dict:
    lines = _lines(raw)
    words = " ".join(lines).split()
    n_w, n_l = len(words), max(len(lines), 1)
    counts = pd.Series(lines).value_counts() if lines else pd.Series(dtype=int)
    ends = [l.split()[-1] for l in lines if l.split()]
    rhymes = sum(a[-2:] == b[-2:] for a, b in zip(ends, ends[1:]))
    id_share, en_share = language_ratios(" ".join(words))
    per = lambda vocab: sum(w in vocab for w in words) / max(n_w, 1) * 100  # per 100 kata
    return {
        "n_words": n_w, "n_lines": len(lines), "words_per_line": n_w / n_l,
        "type_token_ratio": len(set(words)) / max(n_w, 1),
        "unique_line_ratio": len(counts) / n_l,
        "repeated_line_share": counts[counts > 1].sum() / n_l if len(counts) else 0.0,  # porsi baris yang berulang
        "top_line_repeats": int(counts.max()) if len(counts) else 0,                   # kekuatan hook
        "rhyme_rate": rhymes / max(len(ends) - 1, 1),
        "english_share": en_share,
        "id_function_share": id_share,
        "pron_aku_kamu": per({"aku", "kamu", "kau", "engkau", "ku", "mu", "dirimu", "diriku"}),
        "pron_gue_lo": per({"gue", "gua", "gw", "lo", "lu", "elo", "elu"}),
        "pron_saya_anda": per({"saya", "anda"}),
    }


# ---------------- Tema ----------------

def theme_features(clean_texts: list[str]) -> np.ndarray:
    from experiments.benchmark import Embedder
    emb = Embedder()
    lyr = emb.transform(clean_texts)                                  # (n, d), sudah ternormalisasi
    model = _sentence_model(emb.model_name)
    protos = np.vstack([model.encode(s, normalize_embeddings=True).mean(axis=0) for s in THEMES.values()])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    lyr = lyr / np.linalg.norm(lyr, axis=1, keepdims=True)
    return lyr @ protos.T


def _sentence_model(name):
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Emosi ----------------

def train_emotion_model() -> float:
    """Fine-tune IndoBERT pada IndoNLU EmoT (train), return macro-F1 pada split valid."""
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from experiments.finetune import MODEL_NAME
    tr, va = (pd.read_csv(EMOT_URL.format(s)) for s in ("train_preprocess", "valid_preprocess"))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    lab = {e: i for i, e in enumerate(EMOTIONS)}

    def ds(df):
        enc = tok(list(df.tweet.map(clean_lyrics)), truncation=True, max_length=128, padding="max_length",
                  return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(df.label.map(lab).values))

    torch.manual_seed(42)
    dev = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(EMOTIONS), id2label=dict(enumerate(EMOTIONS)), label2id=lab).to(dev)
    loader = DataLoader(ds(tr), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    epochs = 3
    sched = get_linear_schedule_with_warmup(opt, len(loader) // 10, epochs * len(loader))
    for _ in range(epochs):
        model.train()
        for ids, mask, y in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=y.to(dev)).loss
            loss.backward()
            opt.step(), sched.step(), opt.zero_grad()
    model.eval()
    va_ds = ds(va)
    preds = []
    with torch.no_grad():
        for ids, mask, _ in DataLoader(va_ds, batch_size=64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.argmax(-1).cpu())
    f1 = f1_score(va_ds.tensors[2].numpy(), torch.cat(preds).numpy(), average="macro")
    model.save_pretrained(EMOT_DIR)
    tok.save_pretrained(EMOT_DIR)
    return float(f1)


def emotion_features(raws: list[str]) -> np.ndarray:
    """Rata-rata probabilitas emosi per potongan ±4 baris lirik (emosi lirik berubah antar bait)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMOT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(EMOT_DIR).to("cuda").eval()
    out = []
    for raw in raws:
        lines = _lines(raw) or [""]
        chunks = [" ".join(lines[i:i + 4]) for i in range(0, len(lines), 4)]
        enc = tok(chunks, truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(model(**enc).logits.float(), -1).mean(0)
        out.append(p.cpu().numpy())
    return np.vstack(out)


# ---------------- Semua fitur ----------------

def semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur untuk tiap baris df (kolom lyrics_raw & lyrics_clean), dengan cache per lirik."""
    cache = pd.read_csv(CACHE, index_col="key") if CACHE.exists() else pd.DataFrame(columns=FEATURES)
    keys = df["lyrics_raw"].map(_key)
    todo = df[~keys.isin(cache.index)].drop_duplicates(subset="lyrics_raw")
    if len(todo):
        if not (EMOT_DIR / "config.json").exists():
            print(f"Melatih model emosi (IndoNLU EmoT)... valid macro-F1 = {train_emotion_model():.3f}")
        new = pd.DataFrame([structure_features(r) for r in todo["lyrics_raw"]])
        new[THEME_COLS] = theme_features(list(todo["lyrics_clean"]))
        new[EMOTION_COLS] = emotion_features(list(todo["lyrics_raw"]))
        new.index = todo["lyrics_raw"].map(_key).values
        new.index.name = "key"
        cache = pd.concat([cache, new[FEATURES]]).astype(float)
        cache.index.name = "key"
        cache.to_csv(CACHE)
    return cache.loc[keys.values, FEATURES].astype(float).reset_index(drop=True).set_index(df.index)
