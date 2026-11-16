"""One interface over every tokenizer in the study.

All arms share the same three special tokens and see exactly the same text:

    id 0  <unk>   never produced in practice (see ALPHABET below)
    id 1  <eod>   placed before every document
    id 2  <nl>    a line break inside a document

ALPHABET. The corpus has thousands of distinct characters, most of them stray
symbols from other scripts. A fixed alphabet (the characters that cover 99.99%
of the training text) is chosen once, and every character outside it is replaced
by U+FFFD before any tokenizer sees the text. Every arm therefore models the
same character sequence, which is what makes bits per character comparable.

MORPHOLOGY ARM. Words are first split into morphs by a segmentation table
learned with Morfessor. Inside a word, each non-initial morph is written as a
separate "word" that starts with the joiner mark U+06DE, so the subword
tokenizer can never build a token that crosses a morph boundary. Decoding
removes the space and the mark again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

UNK_ID, EOD_ID, NL_ID = 0, 1, 2
SPECIALS = ["<unk>", "<eod>", "<nl>"]
REPLACEMENT = "�"
# An Arabic-script symbol that normalisation removes from all text, so it cannot
# occur naturally. It must be Arabic script: SentencePiece does not build tokens
# that mix scripts, and the mark has to merge with the letters that follow it.
JOINER = "۞"

# Arabic-script letters (not digits, not punctuation). A "word" for the
# morphological segmenter is a maximal run of these.
WORD_RE = re.compile(
    "[ء-غف-يٓ-ٕٮ-ۓەۮۯۺ-ۿݐ-ݿ]+"
)


def out_of_alphabet_re(alphabet: str) -> re.Pattern:
    return re.compile("[^" + re.escape(alphabet) + "\n]")


class Tokenizer:
    def __init__(self, path: str | Path):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.name: str = meta["name"]
        self.kind: str = meta["kind"]            # "spm" or "char"
        self.alphabet: str = meta["alphabet"]
        self.vocab_size: int = meta["vocab_size"]
        self.morph: bool = bool(meta.get("morph"))
        self._oov_re = out_of_alphabet_re(self.alphabet)
        self._seg: dict[str, str] = {}
        if self.morph:
            self._seg = json.loads((path / "segmentation.json").read_text(encoding="utf-8"))
        if self.kind == "spm":
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=str(path / "spm.model"))
            assert self.sp.get_piece_size() == self.vocab_size
            assert self.sp.piece_to_id("<eod>") == EOD_ID and self.sp.piece_to_id("<nl>") == NL_ID
        elif self.kind == "char":
            self.chars: list[str] = meta["chars"]  # id - 3 -> character
            table = np.full(0x110000, len(SPECIALS) + self.chars.index(REPLACEMENT), dtype=np.uint16)
            for i, ch in enumerate(self.chars):
                table[ord(ch)] = len(SPECIALS) + i
            table[ord("\n")] = NL_ID
            self._table = table
        else:
            raise ValueError(f"unknown tokenizer kind {self.kind!r}")

    # -- text preparation -----------------------------------------------------
    def clean(self, text: str) -> str:
        """Replace every character outside the alphabet. Identical for all arms."""
        return self._oov_re.sub(REPLACEMENT, text)

    def segment(self, text: str) -> str:
        seg = self._seg
        return WORD_RE.sub(lambda m: seg.get(m.group(), m.group()), text)

    # -- encoding ---------------------------------------------------------------
    def encode_docs(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """Encode documents into one token stream.

        Returns (tokens, characters). `characters` counts every character of the
        cleaned text plus one per document for the <eod> mark. It is the same
        number for every tokenizer, and is the denominator of bits per character.
        """
        cleaned = [self.clean(t) for t in texts]
        n_chars = sum(len(t) for t in cleaned) + len(cleaned)
        if self.kind == "char":
            parts = []
            for t in cleaned:
                parts.append(np.array([EOD_ID], dtype=np.uint16))
                codepoints = np.frombuffer(t.encode("utf-32-le"), dtype=np.uint32)
                parts.append(self._table[codepoints])
            tokens = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)
            return tokens, n_chars

        lines: list[str] = []
        line_counts: list[int] = []
        for t in cleaned:
            if self.morph:
                t = self.segment(t)
            doc_lines = t.split("\n")
            line_counts.append(len(doc_lines))
            lines.extend(doc_lines)
        encoded = self.sp.encode(lines)
        out: list[int] = []
        i = 0
        for count in line_counts:
            out.append(EOD_ID)
            for j in range(count):
                if j:
                    out.append(NL_ID)
                out.extend(encoded[i + j])
            i += count
        return np.asarray(out, dtype=np.uint16), n_chars

    def encode(self, text: str) -> np.ndarray:
        return self.encode_docs([text])[0]

    # -- decoding ---------------------------------------------------------------
    def decode(self, ids) -> str:
        """Inverse of encode for one or more documents (documents are joined by a blank line)."""
        ids = [int(i) for i in ids]
        docs: list[list[list[int]]] = []
        for i in ids:
            if i == EOD_ID:
                docs.append([[]])
                continue
            if not docs:
                docs.append([[]])
            if i == NL_ID:
                docs[-1].append([])
            else:
                docs[-1][-1].append(i)
        texts = []
        for doc in docs:
            if self.kind == "char":
                rendered = ["".join(self.chars[i - len(SPECIALS)] if i >= len(SPECIALS) else "" for i in line)
                            for line in doc]
            else:
                rendered = [self.sp.decode(line) for line in doc]
                if self.morph:
                    rendered = [line.replace(" " + JOINER, "") for line in rendered]
            texts.append("\n".join(rendered))
        return "\n\n".join(texts)

    def pieces(self, text: str) -> list[str]:
        """Human-readable tokens of one line of text, for inspection."""
        text = self.clean(text)
        if self.kind == "char":
            return list(text)
        if self.morph:
            text = self.segment(text)
        return self.sp.encode(text, out_type=str)
