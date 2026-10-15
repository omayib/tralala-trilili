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
