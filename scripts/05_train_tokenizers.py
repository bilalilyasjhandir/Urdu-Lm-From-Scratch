"""Train every tokenizer in the study. Safe to re-run: finished arms are skipped.

    python scripts/05_train_tokenizers.py --data-dir D:/urdu-lm-data
    python scripts/05_train_tokenizers.py --data-dir D:/urdu-lm-data --arms bpe-32k char

Arms
    bpe-8k, bpe-16k, bpe-32k, bpe-64k   SentencePiece BPE
    unigram-32k                          SentencePiece Unigram
    morph-bpe-32k                        Morfessor morphs first, then BPE inside each morph
                                         (see MORPHOLOGY below)
    char                                 one token per character

Each arm is written to <data-dir>/tokenizers/<arm>/. Afterwards every trained arm
is measured on the validation sample, and these files are written to reports/:
    tokenizers.json          tokens per word and per character, round-trip check
    tokenizer_examples.md    the same sentences split by every arm, to read by eye
    morph_examples.md        how the morphology arm splits 150 words of mixed frequency

MORPHOLOGY. Morfessor Baseline is unsupervised: it learns morphs from a word list
alone. Two settings were chosen by reading its output on training words:
  * It is trained on word types (every word counts once). With frequency-weighted
    counts it left every one of the most frequent words whole.
  * A split is used only if its first morph has at least 3 letters. Without this
    rule it also cuts short, very frequent words into meaningless pieces (for
    example the postposition "se" into two single letters). Real Urdu suffixes
    survive the rule because they attach to longer stems.
Both settings are judgment calls on an unsupervised method. Neither was tuned on
validation or test data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from urdu_lm.tokenizer import JOINER, REPLACEMENT, SPECIALS, UNK_ID, WORD_RE, Tokenizer  # noqa: E402

ARMS = {
    "bpe-32k": {"kind": "spm", "model_type": "bpe", "vocab_size": 32000},
    "unigram-32k": {"kind": "spm", "model_type": "unigram", "vocab_size": 32000},
    "morph-bpe-32k": {"kind": "spm", "model_type": "bpe", "vocab_size": 32000, "morph": True},
    "char": {"kind": "char"},
    "bpe-8k": {"kind": "spm", "model_type": "bpe", "vocab_size": 8000},
    "bpe-16k": {"kind": "spm", "model_type": "bpe", "vocab_size": 16000},
    "bpe-64k": {"kind": "spm", "model_type": "bpe", "vocab_size": 64000},
}


# ---------------------------------------------------------------------------- SentencePiece
def train_spm(sample: Path, out_dir: Path, model_type: str, vocab_size: int, alphabet: str,
              morph: bool, threads: int) -> None:
    import sentencepiece as spm

    required = alphabet.replace(" ", "") + REPLACEMENT + (JOINER if morph else "")
    spm.SentencePieceTrainer.train(
        input=str(sample),
        model_prefix=str(out_dir / "spm"),
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=1.0,
        required_chars=required,
        normalization_rule_name="identity",   # the corpus is already normalised
        unk_id=0, unk_piece="<unk>",
        bos_id=1, bos_piece="<eod>",
        eos_id=-1, pad_id=-1,
        control_symbols=["<nl>"],
        input_sentence_size=0,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
        num_threads=threads,
        minloglevel=1,
    )


# ---------------------------------------------------------------------------- Morfessor
_MODEL = None


def _load_model(path: str) -> None:
    global _MODEL
    import morfessor
    _MODEL = morfessor.MorfessorIO().read_binary_model_file(path)


def _segment_chunk(task: tuple[list[str], int]) -> tuple[dict[str, str], int]:
    words, min_stem = task
    out, proposed = {}, 0
    for word in words:
        morphs, _ = _MODEL.viterbi_segment(word)
        if len(morphs) > 1:
            proposed += 1
            if len(morphs[0]) >= min_stem:
                out[word] = (" " + JOINER).join(morphs)
    return out, proposed


def read_word_counts(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, count = line.rstrip("\n").split("\t")
            rows.append((word, int(count)))
    return rows


def build_segmentation(tok_dir: Path, out_dir: Path, train_types: int, min_count: int, min_stem: int,
                       workers: int) -> dict:
    import morfessor

    words = read_word_counts(tok_dir / "word_counts.tsv")
    model_path = out_dir / "morfessor.bin"
    stats: dict = {"training_types": min(train_types, len(words))}
    if not model_path.exists():
        started = time.time()
        model = morfessor.BaselineModel()
        # Type-based training: every word counts once (see MORPHOLOGY in the module docstring).
        model.load_data([(c, w) for w, c in words[:train_types]], count_modifier=lambda c: 1)
        model.train_batch()
        morfessor.MorfessorIO().write_binary_model_file(str(model_path), model)
        stats["morfessor_training_seconds"] = round(time.time() - started)

    to_segment = [w for w, c in words if c >= min_count]
    counts = dict(words)
    chunks = [(to_segment[i:i + 20000], min_stem) for i in range(0, len(to_segment), 20000)]
    table: dict[str, str] = {}
    proposed = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_load_model, initargs=(str(model_path),)) as pool:
        for i, (part, n) in enumerate(pool.map(_segment_chunk, chunks), start=1):
            table.update(part)
            proposed += n
            if i % 20 == 0 or i == len(chunks):
                print(f"    segmented {i}/{len(chunks)} word chunks", flush=True)
    (out_dir / "segmentation.json").write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")

    tokens = sum(counts[w] for w in to_segment)
    morph_tokens = sum(counts[w] * (table[w].count(JOINER) + 1 if w in table else 1) for w in to_segment)
    stats.update({
        "count_modifier": "types (every word counts once)",
        "min_word_count_for_segmentation": min_count,
        "min_stem_letters": min_stem,
        "word_types_considered": len(to_segment),
        "word_types_morfessor_would_split": proposed,
        "word_types_split": len(table),
        "share_of_types_split": round(len(table) / max(1, len(to_segment)), 4),
        "share_of_running_words_split": round(sum(counts[w] for w in table) / max(1, tokens), 4),
        "morphs_per_running_word": round(morph_tokens / max(1, tokens), 4),
    })
    return stats


def segment_sample(sample: Path, out: Path, table: dict[str, str]) -> None:
    with open(sample, encoding="utf-8") as src, open(out, "w", encoding="utf-8", newline="\n") as dst:
        while True:
            block = src.readlines(64 * 1024 * 1024)
            if not block:
                break
            dst.write(WORD_RE.sub(lambda m: table.get(m.group(), m.group()), "".join(block)))


# ---------------------------------------------------------------------------- arms
def build_arm(name: str, data_dir: Path, args) -> None:
    spec = ARMS[name]
    tok_dir = data_dir / "tok"
    out_dir = data_dir / "tokenizers" / name
    if (out_dir / "meta.json").exists():
        print(f"[{name}] already trained, skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    alphabet = json.loads((tok_dir / "alphabet.json").read_text(encoding="utf-8"))["alphabet"]
    meta: dict = {"name": name, "kind": spec["kind"], "alphabet": alphabet}
    started = time.time()
    print(f"[{name}] training ...", flush=True)

    if spec["kind"] == "char":
        chars = sorted(alphabet) + [REPLACEMENT]
        meta.update({"chars": chars, "vocab_size": len(SPECIALS) + len(chars)})
    else:
        sample = tok_dir / "sample.txt"
        vocab_size = args.vocab_override or spec["vocab_size"]
        if spec.get("morph"):
            meta["morph"] = True
            meta["morfessor"] = build_segmentation(tok_dir, out_dir, args.morfessor_types,
                                                   args.morfessor_min_count, args.morfessor_min_stem, args.workers)
            table = json.loads((out_dir / "segmentation.json").read_text(encoding="utf-8"))
            sample = tok_dir / "sample_morph.txt"
            segment_sample(tok_dir / "sample.txt", sample, table)
        train_spm(sample, out_dir, spec["model_type"], vocab_size, alphabet, bool(spec.get("morph")), args.threads)
        meta.update({"model_type": spec["model_type"], "vocab_size": vocab_size})

    meta["training_seconds"] = round(time.time() - started)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] done in {meta['training_seconds']}s", flush=True)


# ---------------------------------------------------------------------------- report
def report(data_dir: Path, report_dir: Path, examples: int) -> None:
    val = pq.read_table(data_dir / "clean" / "heldout" / "val.parquet", columns=["text"]).column("text").to_pylist()
    trained = [n for n in ARMS if (data_dir / "tokenizers" / n / "meta.json").exists()]
    if not trained:
        return
    words = sum(len(t.split()) for t in val)
    example_lines: list[str] = []
    for text in val:
        for line in text.split("\n"):
            if 60 <= len(line) <= 160:
                example_lines.append(line)
                break
        if len(example_lines) == examples:
            break

    results = {}
    shown: dict[str, list[list[str]]] = {}
    for name in trained:
        tok = Tokenizer(data_dir / "tokenizers" / name)
        tokens, chars = tok.encode_docs(val)
        checked = val[:300]
        exact = sum(1 for t in checked if tok.decode(tok.encode(t)) == tok.clean(t))
        meta = json.loads((data_dir / "tokenizers" / name / "meta.json").read_text(encoding="utf-8"))
        results[name] = {
            "vocab_size": tok.vocab_size,
            "val_tokens": int(len(tokens)),
            "val_characters": int(chars),
            "tokens_per_word": round(len(tokens) / words, 4),
            "characters_per_token": round(chars / len(tokens), 4),
            "unk_tokens": int((tokens == UNK_ID).sum()),
            "round_trip_docs_checked": len(checked),
            "round_trip_docs_exact": exact,
            "training_seconds": meta.get("training_seconds"),
            **({"morfessor": meta["morfessor"]} if "morfessor" in meta else {}),
        }
        shown[name] = [tok.pieces(line) for line in example_lines]

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "tokenizers.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = ["# Tokenizer examples", "", "Generated by `scripts/05_train_tokenizers.py`. Tokens are separated by ` | `.",
          "In the SentencePiece arms `\u2581` marks the start of a word. In the morphology arm "
          f"`{JOINER}` marks a morph that continues the previous one.", ""]
    for i, line in enumerate(example_lines):
        md += [f"## Sentence {i + 1}", "", line, ""]
        for name in trained:
            pieces = shown[name][i]
            if name == "char":
                md.append(f"- **{name}** ({len(pieces)} tokens): one token per character")
            else:
                md.append(f"- **{name}** ({len(pieces)} tokens): " + " | ".join(pieces))
        md.append("")
    (report_dir / "tokenizer_examples.md").write_text("\n".join(md), encoding="utf-8")

    morph_report(data_dir, report_dir)

    print(f"\n{'arm':16}{'vocab':>8}{'tokens/word':>13}{'chars/token':>13}{'unk':>6}{'round trip':>12}")
    for name, r in results.items():
        print(f"{name:16}{r['vocab_size']:>8}{r['tokens_per_word']:>13}{r['characters_per_token']:>13}"
              f"{r['unk_tokens']:>6}{r['round_trip_docs_exact']:>8}/{r['round_trip_docs_checked']}")
    print(f"\nSaved {report_dir / 'tokenizers.json'} and tokenizer_examples.md")


def morph_report(data_dir: Path, report_dir: Path) -> None:
    """Write a frequency-stratified sample of segmentations for reading by eye."""
    seg_path = data_dir / "tokenizers" / "morph-bpe-32k" / "segmentation.json"
    if not seg_path.exists():
        return
    import random
    table = json.loads(seg_path.read_text(encoding="utf-8"))
    words = read_word_counts(data_dir / "tok" / "word_counts.tsv")
    rng = random.Random(0)
    middle, tail = words[1000:10000], words[10000:200000]
    bands = [("The 50 most frequent words", words[:50]),
             ("50 words from ranks 1,000 to 10,000", rng.sample(middle, min(50, len(middle)))),
             ("50 words from ranks 10,000 to 200,000", rng.sample(tail, min(50, len(tail))))]
    md = ["# Morphological segmentation examples", "",
          "Generated by `scripts/05_train_tokenizers.py`. Words with no `+` are kept whole.", ""]
    for title, band in bands:
        md += [f"## {title}", "", "| Word | Count | Segmentation |", "|---|---:|---|"]
        for word, count in band:
            shown = table.get(word, word).replace(" " + JOINER, " + ")
            md.append(f"| {word} | {count:,} | {shown} |")
        md.append("")
    (report_dir / "morph_examples.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=REPO / "reports")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--morfessor-types", type=int, default=200_000,
                        help="most frequent word types used to train the segmenter")
    parser.add_argument("--morfessor-min-count", type=int, default=5,
                        help="words rarer than this are left unsegmented")
    parser.add_argument("--morfessor-min-stem", type=int, default=3,
                        help="a split is used only if its first morph has at least this many letters")
    parser.add_argument("--vocab-override", type=int, default=0, help="for small test runs only")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    if not (args.data_dir / "tok" / "alphabet.json").exists():
        sys.exit("Run scripts/04_tokenizer_data.py first.")
    for name in args.arms:
        build_arm(name, args.data_dir, args)
    report(args.data_dir, args.report_dir, args.examples)


if __name__ == "__main__":
    main()
