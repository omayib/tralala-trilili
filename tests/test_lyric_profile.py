import pandas as pd

from src.lyric_profile import explain, percentile, reference_stats, structure_features, theme_scores

SONG = """Hatiku hancur karena kau pergi
Air mata jatuh malam ini
Hatiku hancur karena kau pergi
Aku sendiri dalam sepi"""


def test_structure_counts_repetition_and_hook():
    f = structure_features(SONG)
    assert f["n_lines"] == 4 and f["top_line_repeats"] == 2
    assert f["repeated_line_share"] == 0.5


def test_theme_lexicon_matches_whole_words_and_phrases():
    t = theme_scores("hatiku hancur air mata ibu ibukota")
    assert t["lex_patah_hati"] > 0          # 'hancur' + 'air mata'
    assert t["lex_keluarga"] > 0            # 'ibu' tetapi bukan 'ibukota'
    assert t["lex_pesta_joget"] == 0
    assert theme_scores("ibukota")["lex_keluarga"] == 0


def test_percentile_interpolates():
    assert percentile(5, [0, 10]) == 50.0


def test_explain_only_hints_significant_associations():
    other = "aku cinta kamu selamanya\nbahagia berdua"
    raws = pd.Series([SONG] + [other] * 4)
    ref = reference_stats(pd.concat([raws] * 8), pd.Series([1000.0, 10.0, 12.0, 9.0, 11.0] * 8))
    ex = explain(SONG, ref)
    assert ex["themes"] and ex["themes"][0][0] == "patah hati"
    assert {r["label"] for r in ex["structure"]} >= {"Panjang lirik", "Baris yang berulang"}
    assert all(r["hint"] is None or "lagu populer" in r["hint"] for r in ex["structure"])
