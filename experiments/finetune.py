"""M6 — fine-tuning IndoBERT (GPU) untuk klasifikasi popularitas dari lirik.

- Model: indobenchmark/indobert-base-p1 (IndoNLU), kepala klasifikasi baru.
- Input: lirik bersih, dipotong 512 token (head) — mayoritas lirik muat utuh.
- Imbalance: cross-entropy berbobot kelas (pengganti SMOTE, yang tidak berlaku untuk teks mentah).
- Seleksi epoch: 15% train (stratified) sebagai validasi, ambil epoch dengan macro-F1 terbaik.
- Varians: dilatih dengan 3 seed, prediksi test = rata-rata probabilitas ketiganya (ensemble).
  Test set tidak pernah dipakai untuk memilih apa pun.
"""
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 512
BATCH = 8
ACCUM = 2          # batch efektif 16
EPOCHS = 6
LR = 2e-5
SEEDS = (42, 43, 44)


def _encode(tok, texts, labels=None):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    tensors = [enc["input_ids"], enc["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor(np.asarray(labels), dtype=torch.long))
    return TensorDataset(*tensors)


@torch.no_grad()
def _predict_proba(model, ds, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in DataLoader(ds, batch_size=BATCH * 2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device)).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.vstack(out)


def _train_one(seed, tok, X_tr, y_tr, X_val, y_val, n_classes, device):
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_classes).to(device)
    train_ds, val_ds = _encode(tok, X_tr, y_tr), _encode(tok, X_val)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    counts = np.bincount(y_tr, minlength=n_classes)
    weights = torch.tensor(len(y_tr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(loader) // ACCUM
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best_f1, best_state, best_epoch = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for i, (ids, mask, y) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), y.to(device)) / ACCUM
            loss.backward()
            if (i + 1) % ACCUM == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        f1 = f1_score(y_val, _predict_proba(model, val_ds, device).argmax(1), average="macro")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_f1, best_epoch


def finetune_predict(X_tr, y_tr, X_te, n_classes: int) -> tuple[np.ndarray, dict]:
    """Latih 3 seed pada train, return (prediksi kelas test, info seleksi)."""
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("M6 membutuhkan GPU CUDA")
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    X_tr, y_tr = np.asarray(X_tr), np.asarray(y_tr)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)
    test_ds = _encode(tok, X_te)

    probas, info = [], []
    for seed in SEEDS:
        model, val_f1, epoch = _train_one(seed, tok, X_fit, y_fit, X_val, y_val, n_classes, device)
        probas.append(_predict_proba(model, test_ds, device))
        info.append({"seed": seed, "val_f1": round(val_f1, 4), "epoch": epoch})
        print(f"    seed {seed}: val macro-F1 {val_f1:.3f} (epoch {epoch})", flush=True)
        del model
        torch.cuda.empty_cache()
    return np.mean(probas, axis=0).argmax(1), {"model": MODEL_NAME, "seeds": info}
