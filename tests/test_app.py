"""Smoke test UI: app termuat & memprediksi memakai model kecil lokal (tanpa jaringan)."""
import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest


@pytest.fixture
def tiny_model_dir(tmp_path, monkeypatch):
    texts = ["aku cinta kamu selamanya", "hujan turun di kota tua"] * 10
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    model.fit(texts, [1, 0] * 10)
    joblib.dump(model, tmp_path / "virality_model.joblib")
    m = {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    (tmp_path / "metadata.json").write_text(json.dumps({
        "best_model": "Logistic Regression", "trained_on": "2026-01-01", "n_songs": 20,
        "viral_rate": 0.5, "viral_threshold_streams_per_day": 1000.0, "test_metrics": m,
        "benchmark": [{"model": "Logistic Regression", **m}]}))
    monkeypatch.setattr("src.config.MODEL_DIR", tmp_path)
    monkeypatch.setattr("src.registry.download_model", lambda *a, **k: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return tmp_path


def test_app_predicts(tiny_model_dir):
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    at.text_area[0].input("aku cinta kamu selamanya " * 6).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Potensi" in m.value for m in at.markdown)
    assert any("lokal" in m.value for m in at.markdown)
