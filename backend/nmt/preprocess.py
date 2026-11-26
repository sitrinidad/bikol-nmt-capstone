"""Input normalisation identical to Phase 1 cleaning, so the model sees text like its training data."""
from __future__ import annotations

import html
import re
import unicodedata

from ..languages import BIKOL

TAG_RE = re.compile(r"<[^>]+>")
ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
SQUOTE = {ord(c): "'" for c in "\u2019\u2018\u02bc\u0060\u00b4"}
DQUOTE = {ord(c): '"' for c in "\u201c\u201d\u201e"}
DASHES = {ord(c): "-" for c in "\u2013\u2014\u2010\u2011\u2212"}
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Same reviewed spelling-variant map as notebook section 2.5.
VARIANT_MAP: dict[str, dict[str, str]] = {
    "bcl_South": {
        "marhay": "maray", "marahay": "maray",
        "paano": "pano", "papano": "pano",
        "biyahe": "byahe",
        "al'daw": "aldaw",
        "novembr": "november",
        "dretso": "deretso",
        "pad-padon": "padpadon",
        "mag-tukaw": "magtukaw",
    },
    "bcl_North": {
        "niyo": "ninyo",
        "madiyag": "madyag",
    },
}
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*", re.UNICODE)


def generic_clean(s: str) -> str:
    s = html.unescape(str(s))
    s = TAG_RE.sub(" ", s)
    s = unicodedata.normalize("NFC", s)
    s = s.translate(ZW).translate(SQUOTE).translate(DQUOTE).translate(DASHES)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    s = re.sub(r"([.!?]){4,}", "...", s)
    return s


def strip_diacritics(s: str) -> str:
    """Remove combining marks but keep ñ (as in Phase 1)."""
    out, prev = [], ""
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.combining(ch) and not (ch == "\u0303" and prev.lower() == "n"):
            continue
        out.append(ch)
        prev = ch
    return unicodedata.normalize("NFC", "".join(out))


def apply_variants(s: str, mapping: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        word = m.group(0)
        target = mapping.get(word.lower())
        if target is None:
            return word
        if word.isupper():
            return target.upper()
        if word[0].isupper():
            return target[0].upper() + target[1:]
        return target

    return _WORD_RE.sub(sub, s)


def normalize_source(text: str, lang: str) -> str:
    s = generic_clean(text)
    if lang in BIKOL:
        s = strip_diacritics(s)
        s = apply_variants(s, VARIANT_MAP.get(lang, {}))
    return s


def split_segments(text: str) -> list[list[str]]:
    """Split into lines, then sentences. The model was trained on single sentences (max 64 tokens)."""
    lines = []
    for raw_line in str(text).splitlines():
        if raw_line.strip():
            parts = [p for p in SENTENCE_SPLIT_RE.split(raw_line.strip()) if p.strip()]
            lines.append(parts)
    return lines
