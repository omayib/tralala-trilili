import pandas as pd

from src.features import add_features, is_test, label, split_mask
from src.text_utils import clean_lyrics
from src.validate import decide


def test_clean_lyrics_removes_sections_and_punctuation():
    raw = "[Verse 1]\nAku CINTA kamu!\nChorus:\nLirik oleh: Budi\n[00:12.30] Sungguh, sungguh"
    assert clean_lyrics(raw) == "aku cinta kamu sungguh sungguh"


def test_label_top_quantile_is_viral():
    df = add_features(pd.DataFrame({"total_streams": [100, 200, 300, 400], "days": [1, 1, 1, 1]}))
    df, thr = label(df, 0.75)
    assert thr == 325
    assert df["viral"].tolist() == [0, 0, 0, 1]


def test_golden_split_is_deterministic_and_about_20_percent():
    df = pd.DataFrame({"artist": [f"a{i}" for i in range(2000)], "title": ["t"] * 2000})
    m1, m2 = split_mask(df), split_mask(df.sample(frac=1, random_state=0)).sort_index()
    assert (m1 == m2).all()
    assert 0.17 < m1.mean() < 0.23
    assert is_test("a1", "t") == is_test("a1", "t")


def test_validation_gate():
    assert decide({"f1": 0.5}, None, 5000)[0]                 # rilis pertama
    assert decide({"f1": 0.49}, {"f1": 0.50}, 5000)[0]        # dalam toleransi
    assert not decide({"f1": 0.40}, {"f1": 0.50}, 5000)[0]    # memburuk
    assert not decide({"f1": 0.90}, None, 10)[0]              # data terlalu sedikit
