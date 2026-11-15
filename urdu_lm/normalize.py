"""Urdu text normalisation.

The same Urdu word can be typed with several different Unicode code points
(an Arabic keyboard gives U+064A for "ye", an Urdu keyboard gives U+06CC), and
web text is full of invisible formatting characters. Left alone, these turn one
word into several vocabulary entries, which hurts every tokenizer differently
and would bias a tokenizer comparison. This module maps text to one canonical
form before anything else sees it.

Every rule is listed in RULES below so the README and the code cannot drift.
"""
from __future__ import annotations

import re
import unicodedata

RULES = [
    "Unicode NFKC, so Arabic presentation forms (U+FB50-U+FDFF, U+FE70-U+FEFF) become plain letters",
    "Arabic look-alike letters mapped to their Urdu code points (yeh, kaf, heh, teh marbuta)",
    "Arabic-Indic and Extended Arabic-Indic digits mapped to ASCII digits",
    "Optional vowel marks (zabar, zer, pesh, tanween, shadda, sukun, superscript alef) removed",
    "Quranic annotation signs and honorific marks removed",
    "Tatweel (kashida) removed",
    "Invisible formatting removed: zero-width space/joiner, direction marks, BOM, soft hyphen",
    "Zero-width non-joiner kept only between two Arabic-script letters, and never doubled",
    "URLs and email addresses removed",
    "Control characters removed, whitespace runs collapsed, at most one blank line in a row",
    "Unicode NFC at the end, so hamza and madda combine with their base letter",
]


def _rng(first: int, last: int) -> range:
    return range(first, last + 1)


ZWNJ = "‌"

_TABLE: dict[int, str | None] = {}

# --- characters that are deleted ---------------------------------------------
_DELETE = [
    *_rng(0x0610, 0x061A),  # honorific signs (sallallahu alayhi wasallam etc.)
    *_rng(0x064B, 0x0652),  # tanween, zabar, pesh, zer, shadda, sukun
    *_rng(0x0656, 0x065F),  # rarer vowel marks
    0x0670,                 # superscript alef (khari zabar)
    *_rng(0x06D6, 0x06ED),  # Quranic annotation signs
    0x0640,                 # tatweel
    0x00AD,                 # soft hyphen
    0x061C,                 # Arabic letter mark
    0x200B,                 # zero-width space
    0x200D,                 # zero-width joiner
    0x200E, 0x200F,         # left-to-right / right-to-left marks
    *_rng(0x202A, 0x202E),  # direction embeddings and overrides
    *_rng(0x2060, 0x2064),  # word joiner and invisible operators
    *_rng(0x2066, 0x2069),  # direction isolates
    0xFEFF,                 # byte-order mark
    0xFFFC, 0xFFFD,         # object replacement / replacement character
]
# U+0653 (madda), U+0654 (hamza above) and U+0655 (hamza below) are kept on
# purpose: NFC needs them to build alef-madda, heh-hamza, yeh-barree-hamza.
for _cp in _DELETE:
    _TABLE[_cp] = None

# control characters: drop all except newline, turn tab into a space
for _cp in (*_rng(0x00, 0x1F), *_rng(0x7F, 0x9F)):
    _TABLE[_cp] = None
_TABLE[0x0A] = "\n"
_TABLE[0x0D] = "\n"   # lone carriage return ("\r\n" is collapsed before this)
_TABLE[0x09] = " "
_TABLE[0x2028] = "\n"  # line separator
_TABLE[0x2029] = "\n"  # paragraph separator

# --- Arabic look-alikes -> Urdu code points -----------------------------------
_TABLE[0x064A] = "ی"  # Arabic yeh            -> Farsi/Urdu yeh
_TABLE[0x0649] = "ی"  # alef maksura          -> Farsi/Urdu yeh
_TABLE[0x0643] = "ک"  # Arabic kaf            -> keheh
_TABLE[0x0647] = "ہ"  # Arabic heh            -> heh goal
_TABLE[0x0629] = "ۃ"  # teh marbuta           -> teh marbuta goal
_TABLE[0x06C0] = "ۂ"  # heh with yeh above    -> heh goal with hamza above

# --- digits and number punctuation --------------------------------------------
for _i in range(10):
    _TABLE[0x0660 + _i] = str(_i)  # Arabic-Indic digits
    _TABLE[0x06F0 + _i] = str(_i)  # Extended Arabic-Indic digits
_TABLE[0x066A] = "%"
_TABLE[0x066B] = "."
_TABLE[0x066C] = ","

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_ARABIC = "؀-ۿݐ-ݿ"
_ZWNJ_RUN_RE = re.compile(ZWNJ + "{2,}")
_ZWNJ_STRAY_RE = re.compile(f"(?<![{_ARABIC}]){ZWNJ}|{ZWNJ}(?![{_ARABIC}])")
_SPACE_RUN_RE = re.compile(r" {2,}")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" *\n *")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Return the canonical form of an Urdu document. Idempotent."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = text.translate(_TABLE)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _ZWNJ_RUN_RE.sub(ZWNJ, text)
    # Removing a stray ZWNJ can expose another one next to it, so loop until stable.
    while True:
        cleaned = _ZWNJ_STRAY_RE.sub("", text)
        if cleaned == text:
            break
        text = cleaned
    text = _SPACE_RUN_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = unicodedata.normalize("NFC", text)
    return text.strip()
