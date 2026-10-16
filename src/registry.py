"""Artifact store di Hugging Face Hub.

- Dataset repo (privat)  : versioning data mentah, cache lirik, dataset berlabel.
- Model repo (registry)  : model + metadata + model card; tiap rilis diberi tag `vYYYYMMDD-HHMM`
                           sehingga bisa ditinjau dan di-rollback (set HF_MODEL_REVISION).
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

from src.config import (DATA_DIR, DATA_FILES, HF_DATASET_REPO, HF_MODEL_REPO, HF_MODEL_REVISION,
                        MODEL_DIR, MODEL_FILES)

_MISSING = (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError)


def ensure_repos(api: HfApi | None = None) -> None:
    api = api or HfApi()
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True)


# ---------- Data ----------

def pull_data() -> str | None:
    """Unduh snapshot data terakhir ke data/. Return commit sha, atau None bila repo kosong."""
    try:
        path = snapshot_download(HF_DATASET_REPO, repo_type="dataset", allow_patterns=DATA_FILES,
                                 local_dir=DATA_DIR)
    except _MISSING:
        print(f"Dataset {HF_DATASET_REPO} belum ada — mulai dari data lokal.")
        return None
    sha = HfApi().dataset_info(HF_DATASET_REPO).sha
    print(f"Data ditarik dari {HF_DATASET_REPO}@{sha[:7]} -> {path}")
    return sha


def push_data(message: str) -> str:
    """Unggah data/ sebagai commit baru di dataset repo. Return commit sha."""
    api = HfApi()
    card = Path(__file__).resolve().parent.parent / "cards" / "DATASET_CARD.md"
    shutil.copy(card, DATA_DIR / "README.md")
    info = api.upload_folder(repo_id=HF_DATASET_REPO, repo_type="dataset", folder_path=DATA_DIR,
                             allow_patterns=DATA_FILES + ["README.md"], commit_message=message)
    print(f"Data diunggah -> {info.commit_url}")
    return info.oid


# ---------- Model ----------

def download_model(revision: str = HF_MODEL_REVISION, local_dir: Path | None = None) -> Path | None:
    """Unduh model produksi. Return folder berisi MODEL_FILES, atau None bila belum ada rilis."""
    try:
        return Path(snapshot_download(HF_MODEL_REPO, revision=revision, allow_patterns=MODEL_FILES,
                                      local_dir=local_dir))
    except _MISSING:
        return None


def production_metadata(revision: str = HF_MODEL_REVISION) -> dict | None:
    try:
        return json.loads(Path(hf_hub_download(HF_MODEL_REPO, "metadata.json",
                                               revision=revision)).read_text())
    except _MISSING:
        return None


def _model_card(meta: dict) -> str:
    template = (Path(__file__).resolve().parent.parent / "cards" / "MODEL_CARD.md").read_text()
    rows = "\n".join(
        f"| {b['model']} | {b['cv_f1']:.3f} | {b['accuracy']:.3f} | {b['balanced_accuracy']:.3f} | "
        f"{b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} |" for b in meta["benchmark"])
    m = meta["test_metrics"]
    return template.format(
        repo=HF_MODEL_REPO, best_model=meta["best_model"], trained_at=meta["trained_at"], git_sha=meta["git_sha"],
        data_revision=meta["data_revision"], n_songs=meta["n_songs"], n_train=meta["n_train"],
        n_test=meta["n_test"], q1=meta["thresholds_streams_per_day"]["q1"],
        q3=meta["thresholds_streams_per_day"]["q3"], accuracy=m["accuracy"],
        balanced_accuracy=m["balanced_accuracy"], precision=m["precision"], recall=m["recall"],
        f1=m["f1"], benchmark_rows=rows,
        validation=json.dumps(meta.get("validation", {}), indent=2),
    )


def push_model(meta: dict) -> str:
    """Rilis models/ ke registry + model card, lalu beri tag versi. Return nama tag."""
    api = HfApi()
    (MODEL_DIR / "README.md").write_text(_model_card(meta))
    tag = "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    m = meta["test_metrics"]
    info = api.upload_folder(
        repo_id=HF_MODEL_REPO, folder_path=MODEL_DIR, allow_patterns=MODEL_FILES + ["README.md"],
        commit_message=f"Release {tag}: {meta['best_model']} F1={m['f1']:.3f} "
                       f"({meta['n_songs']} lagu)")
    api.create_tag(HF_MODEL_REPO, tag=tag, revision=info.oid, exist_ok=True)
    print(f"Model dirilis -> {info.commit_url} (tag {tag})")
    return tag
