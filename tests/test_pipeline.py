import pandas as pd

from src.features import add_features, is_test, label, split_mask
from src.text_utils import clean_lyrics
from src.validate import decide


def test_clean_lyrics_removes_sections_and_punctuation():
    raw = "[Verse 1]\nAku CINTA kamu!\nChorus:\nLirik oleh: Budi\n[00:12.30] Sungguh, sungguh"
    assert clean_lyrics(raw) == "aku cinta kamu sungguh sungguh"


def test_label_three_classes_by_quartile():
    df = add_features(pd.DataFrame({"total_streams": [100, 200, 300, 400, 500], "days": [1] * 5}))
    df, thr = label(df)
    assert thr == {"q1": 200, "q3": 400}
    assert df["label"].tolist() == [0, 1, 1, 2, 2]


def test_select_songs_keeps_unique_indonesian_lyrics():
    from src.features import select_songs
    ind = "aku cinta kamu dan kau tak tahu " * 5
    df = pd.DataFrame({"lyrics_clean": [ind, ind, "i love you baby and you know it " * 5, None]})
    assert select_songs(df)["lyrics_clean"].tolist() == [ind]


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


def test_is_indonesian_handles_english_hooks_and_rejects_english():
    from src.text_utils import is_indonesian
    ind = clean_lyrics("Kau bertanya dengan rasa ragu, seberapa besar cintaku padamu. "
                       "Oh oh my lady, cukup kau di sampingku, tak perlu kau ragu lagi")
    eng = clean_lyrics("I know you want me, baby, and I will love you till the end of time")
    assert is_indonesian(ind)
    assert not is_indonesian(eng)
    assert not is_indonesian("")
