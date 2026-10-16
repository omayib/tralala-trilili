"""Pembersihan lirik — dipakai bersama oleh scraper, training, dan app agar konsisten."""
import re
import unicodedata

# Penanda struktur lagu / metadata yang bukan bagian lirik
_SECTION_RE = re.compile(
    r"^\s*[\[\(]?\s*(verse|chorus|reff?|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|"
    r"hook|interlude|instrumental|refrain|coda|bait|ulang|repeat|x\d+|\d+x)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_CREDIT_RE = re.compile(
    r"^\s*(lyrics?|lirik|written|composed|composer|music|musik|cipt(aan)?|produced|"
    r"producer|arr(anged|ansemen)?|vocal|vokal)\s*(by|oleh)?\s*[:\-].*$",
    re.IGNORECASE | re.MULTILINE,
)
_BRACKET_RE = re.compile(r"\[[^\]]*\]")          # [Verse 1], [Chorus: Nama]
_TIMESTAMP_RE = re.compile(r"\[\d{1,2}:\d{2}(?:\.\d+)?\]")  # format LRC
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_lyrics(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _TIMESTAMP_RE.sub(" ", text)
    text = _SECTION_RE.sub(" ", text)
    text = _CREDIT_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)
    text = text.lower().replace("'", "").replace("’", "")
    text = _NON_ALPHA_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


# Kata fungsi paling umum (bukan daftar stopword lengkap) — cukup untuk membedakan bahasa lirik.
_ID_WORDS = frozenset("""
aku kau kamu engkau dia kita kami mereka ku mu nya yang dan di ke dari ini itu tak tidak bukan
akan sudah telah masih belum pernah selalu hanya juga lagi saja pun jangan bila jika kalau saat
ketika karena tapi namun atau dengan untuk pada dalam bagai seperti apa siapa mengapa kenapa
bagaimana dimana kini nanti mungkin ingin mau bisa harus ada semua satu hati cinta rindu sayang
""".split())
_EN_WORDS = frozenset("""
i you he she we they me my your our their it is am are was were be been the a an and or but of
to in on at for with from this that what when where why how not no do does did don can will
just so all if love like know want get got go now one never
""".split())


def language_ratios(clean_text: str) -> tuple[float, float]:
    """Proporsi token yang merupakan kata fungsi Indonesia / Inggris (input: hasil clean_lyrics)."""
    words = clean_text.split()
    n = len(words) or 1
    return sum(w in _ID_WORDS for w in words) / n, sum(w in _EN_WORDS for w in words) / n


def is_indonesian(clean_text: str, min_ratio: float = 0.08) -> bool:
    """Lirik berbahasa Indonesia: kata fungsi Indonesia cukup banyak dan bahasa Inggris tidak
    dominan (reff/interjeksi Inggris tetap boleh). Lagu berbahasa daerah umumnya < min_ratio."""
    id_r, en_r = language_ratios(clean_text)
    return id_r >= min_ratio and en_r < 2 * id_r
