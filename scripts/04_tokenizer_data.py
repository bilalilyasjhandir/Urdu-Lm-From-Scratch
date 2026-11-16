"""Prepare what the tokenizers are trained from. Safe to re-run.

    python scripts/04_tokenizer_data.py --data-dir D:/urdu-lm-data

One pass over the training split produces, under <data-dir>/tok:
    alphabet.json     the characters that cover 99.99% of the training text
    sample.txt        about 4% of the training documents, one line per line of text,
                      with every character outside the alphabet replaced
    word_counts.tsv   every Arabic-script word seen at least twice, with its count
                      (input for the morphological segmenter)
and reports/alphabet.json in the repository.

Only the training split is read, so nothing from validation or test sites can
influence a tokenizer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from urdu_lm.tokenizer import JOINER, REPLACEMENT, WORD_RE, out_of_alphabet_re  # noqa: E402

MAX_LINE_CHARS = 2000
SPM_SPACE = "\u2581"


def split_long(line: str) -> list[str]:
    """Break a very long line at spaces so no training line exceeds MAX_LINE_CHARS."""
    if len(line) <= MAX_LINE_CHARS:
        return [line]
    pieces, current, size = [], [], 0
    for word in line.split(" "):
        if size + len(word) + 1 > MAX_LINE_CHARS and current:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def scan_shard(task: tuple[str, str, int]) -> tuple[Counter, Counter, int]:
    path, part_path, per_mille = task
    table = pq.read_table(path, columns=["text", "hash"])
    chars: Counter = Counter()
    words: Counter = Counter()
    sampled_docs = 0
    with open(part_path, "w", encoding="utf-8", newline="\n") as out:
        for text, h in zip(table.column("text").to_pylist(), table.column("hash").to_pylist()):
            chars.update(text)
            words.update(WORD_RE.findall(text))
            if h % 1000 < per_mille:
                sampled_docs += 1
                for line in text.split("\n"):
                    if line:
                        for piece in split_long(line):
                            out.write(piece + "\n")
    return chars, words, sampled_docs


def choose_alphabet(chars: Counter, coverage: float) -> tuple[str, float, int]:
    chars = Counter(chars)
    chars.pop("\n", None)
    for reserved in (REPLACEMENT, JOINER, SPM_SPACE):
        chars.pop(reserved, None)
    total = sum(chars.values())
    ranked = sorted(chars.items(), key=lambda kv: (-kv[1], ord(kv[0])))
    picked, covered = [], 0
    for ch, count in ranked:
        if covered >= coverage * total:
            break
        picked.append(ch)
        covered += count
    if " " not in picked:
        picked.append(" ")
    return "".join(sorted(picked)), covered / total, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=REPO / "reports")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--coverage", type=float, default=0.9999)
    parser.add_argument("--sample-per-mille", type=int, default=40, help="share of training documents in sample.txt")
    parser.add_argument("--min-word-count", type=int, default=2)
    args = parser.parse_args()

    shards = sorted((args.data_dir / "clean" / "train").glob("*.parquet"))
    if not shards:
        sys.exit("No cleaned training shards found. Run scripts/02_clean.py first.")
    tok_dir = args.data_dir / "tok"
    parts_dir = tok_dir / "_sample_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chars: Counter = Counter()
    words: Counter = Counter()
    sampled_docs = 0
    tasks = [(str(p), str(parts_dir / f"{p.stem}.txt"), args.sample_per_mille) for p in shards]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (c, w, n) in enumerate(pool.map(scan_shard, tasks), start=1):
            chars.update(c)
            words.update(w)
            sampled_docs += n
            print(f"  scanned {i}/{len(tasks)} shards, {len(words):,} distinct words so far", flush=True)

    alphabet, covered, total_chars = choose_alphabet(chars, args.coverage)
    (tok_dir / "alphabet.json").write_text(json.dumps({"alphabet": alphabet}, ensure_ascii=False), encoding="utf-8")

    oov = out_of_alphabet_re(alphabet)
    sample_lines = sample_chars = 0
    with open(tok_dir / "sample.txt", "w", encoding="utf-8", newline="\n") as out:
        for _, part_path, _ in tasks:
            with open(part_path, encoding="utf-8") as part:
                text = oov.sub(REPLACEMENT, part.read())
            out.write(text)
            sample_lines += text.count("\n")
            sample_chars += len(text)
    shutil.rmtree(parts_dir, ignore_errors=True)

    kept_words = [(w, c) for w, c in words.items() if c >= args.min_word_count]
    kept_words.sort(key=lambda kv: (-kv[1], kv[0]))
    with open(tok_dir / "word_counts.tsv", "w", encoding="utf-8", newline="\n") as out:
        for w, c in kept_words:
            out.write(f"{w}\t{c}\n")

    report = {
        "coverage_target": args.coverage,
        "coverage_reached": round(covered, 7),
        "alphabet_size": len(alphabet),
        "distinct_characters_in_training_text": len(chars),
        "characters_counted": total_chars,
        "sample": {"per_mille_of_documents": args.sample_per_mille, "documents": sampled_docs,
                   "lines": sample_lines, "characters": sample_chars},
        "words": {"tokens": sum(words.values()), "distinct": len(words),
                  f"distinct_with_count_at_least_{args.min_word_count}": len(kept_words)},
        "alphabet": [{"codepoint": f"U+{ord(ch):04X}", "name": unicodedata.name(ch, "?"),
                      "share": round(chars[ch] / total_chars, 8)} for ch in sorted(alphabet, key=lambda c: -chars[c])],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "alphabet.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "alphabet"}
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {tok_dir / 'alphabet.json'}, sample.txt, word_counts.tsv and reports/alphabet.json")


if __name__ == "__main__":
    main()
