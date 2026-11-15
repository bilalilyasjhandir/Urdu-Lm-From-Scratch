"""Document-level quality filters, applied after normalisation.

FineWeb2 has already been language-identified and near-deduplicated, so these
filters are deliberately few. They catch what a web-scale language identifier
still lets through: very short pages, pages that are mostly Latin text or
numbers, and pages in another Arabic-script language (Persian, Punjabi in
Shahmukhi, Pashto, Sindhi, Uyghur) that were labelled Urdu.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Very common Urdu function words, in normalised spelling. Together they make
# up roughly a third of the words in ordinary Urdu prose, and most of them do
# not exist in the neighbouring Arabic-script languages.
URDU_FUNCTION_WORDS = frozenset({
    "کے",              # ke
    "کی",              # ki
    "کا",              # ka
    "میں",        # mein
    "ہے",              # hai
    "ہیں",        # hain
    "اور",        # aur
    "سے",              # se
    "کو",              # ko
    "نے",              # ne
    "پر",              # par
    "بھی",        # bhi
    "تھا",        # tha
    "تھی",        # thi
    "تھے",        # thay
    "نہیں",  # nahin
    "یہ",              # yeh
    "وہ",              # woh
    "اس",              # is / us
    "ان",              # in / un
    "ہو",              # ho
    "ایک",        # aik
})

_PUNCT = "۔،؟؛.,!?:;\"'()[]{}<>‘’“”-"
_ARABIC_RUN_RE = re.compile(r"[؀-ۿݐ-ݿ]+")
_LATIN_RUN_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class FilterConfig:
    min_chars: int = 200
    # share of non-space characters that are Arabic script
    min_arabic_share: float = 0.60
    # Share of the first `function_word_window` words that are Urdu function words.
    # Measured on 12,000 FineWeb2 documents: median 0.28, 1st percentile 0.17, and
    # the few documents under 0.12 were still real Urdu (event notices full of names,
    # recipes, poetry). 0.05 therefore only removes text that is not Urdu prose at all.
    min_function_word_share: float = 0.05
    function_word_window: int = 3000


def arabic_share(text: str) -> float:
    non_space = len(text) - text.count(" ") - text.count("\n")
    if non_space == 0:
        return 0.0
    arabic = len(text) - len(_ARABIC_RUN_RE.sub("", text))
    return arabic / non_space


def latin_share(text: str) -> float:
    non_space = len(text) - text.count(" ") - text.count("\n")
    if non_space == 0:
        return 0.0
    latin = len(text) - len(_LATIN_RUN_RE.sub("", text))
    return latin / non_space


def function_word_share(text: str, window: int = 3000) -> float:
    words = text.split(None, window)[:window]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.strip(_PUNCT) in URDU_FUNCTION_WORDS)
    return hits / len(words)


def reject_reason(text: str, cfg: FilterConfig = FilterConfig()) -> str | None:
    """Return None if the document is kept, otherwise the name of the failed filter."""
    if len(text) < cfg.min_chars:
        return "too_short"
    if arabic_share(text) < cfg.min_arabic_share:
        return "low_arabic_share"
    if function_word_share(text, cfg.function_word_window) < cfg.min_function_word_share:
        return "low_function_word_share"
    return None
