"""Turn the cleaned corpus into token files, one set per tokenizer. Safe to re-run.

    python scripts/06_encode.py --data-dir D:/urdu-lm-data
    python scripts/06_encode.py --data-dir D:/urdu-lm-data --arms bpe-32k unigram-32k

Output, under <data-dir>/tokens/<arm>/:
    train.bin, val.bin, test.bin    flat arrays of 16-bit token ids
    meta.json                       token and character counts per split
Every document starts with the <eod> token. reports/encoding.json collects the
counts of all encoded arms.

Disk use for the full corpus is about 7 GB per subword arm and 24 GB for the
character arm.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from urdu_lm.tokenizer import Tokenizer  # noqa: E402

DOCS_PER_CALL = 2000
_TOK: Tokenizer | None = None


def _init(tokenizer_dir: str) -> None:
    global _TOK
    _TOK = Tokenizer(tokenizer_dir)


def encode_texts(tok: Tokenizer, texts: list[str]) -> tuple[np.ndarray, int]:
    parts, chars = [], 0
    for i in range(0, len(texts), DOCS_PER_CALL):
        tokens, n = tok.encode_docs(texts[i:i + DOCS_PER_CALL])
        parts.append(tokens)
        chars += n
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)), chars


def encode_shard(task: tuple[str, str]) -> tuple[str, int, int, int]:
    shard, out_path = task
    done_path = out_path + ".json"
    if os.path.exists(out_path) and os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            info = json.load(f)
        return out_path, info["tokens"], info["chars"], info["docs"]
    texts = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
    tokens, chars = encode_texts(_TOK, texts)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        tokens.tofile(f)
    os.replace(tmp, out_path)
    with open(done_path, "w", encoding="utf-8") as f:
        json.dump({"tokens": int(len(tokens)), "chars": int(chars), "docs": len(texts)}, f)
    return out_path, int(len(tokens)), int(chars), len(texts)


def encode_arm(name: str, data_dir: Path, workers: int) -> dict:
    tok_dir = data_dir / "tokenizers" / name
    out_dir = data_dir / "tokens" / name
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        print(f"[{name}] already encoded, skipping")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    parts_dir = out_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    shards = sorted((data_dir / "clean" / "train").glob("*.parquet"))
    tasks = [(str(s), str(parts_dir / f"{s.stem}.bin")) for s in shards]
    tok = Tokenizer(tok_dir)
    # The morphology arm loads a large segmentation table in every worker.
    n_workers = min(workers, 8) if tok.morph else workers
    splits: dict[str, dict] = {"train": {"tokens": 0, "chars": 0, "docs": 0}}
    results = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init, initargs=(str(tok_dir),)) as pool:
        for i, res in enumerate(pool.map(encode_shard, tasks), start=1):
            results.append(res)
            print(f"[{name}] encoded {i}/{len(tasks)} training shards, {time.time() - started:.0f}s", flush=True)

    with open(out_dir / "train.bin.tmp", "wb") as out:
        for path, tokens, chars, docs in results:
            with open(path, "rb") as part:
                shutil.copyfileobj(part, out, 64 * 1024 * 1024)
            splits["train"]["tokens"] += tokens
            splits["train"]["chars"] += chars
            splits["train"]["docs"] += docs
    os.replace(out_dir / "train.bin.tmp", out_dir / "train.bin")

    for split in ("val", "test"):
        texts = pq.read_table(data_dir / "clean" / "heldout" / f"{split}.parquet", columns=["text"]) \
            .column("text").to_pylist()
        tokens, chars = encode_texts(tok, texts)
        tokens.tofile(out_dir / f"{split}.bin")
        splits[split] = {"tokens": int(len(tokens)), "chars": int(chars), "docs": len(texts)}

    for s in splits.values():
        s["tokens_per_char"] = round(s["tokens"] / max(1, s["chars"]), 6)
    meta = {"name": name, "vocab_size": tok.vocab_size, "dtype": "uint16", "splits": splits,
            "seconds": round(time.time() - started)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"[{name}] {splits['train']['tokens']:,} training tokens, "
          f"{1 / splits['train']['tokens_per_char']:.3f} characters per token, {meta['seconds']}s", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=REPO / "reports")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    trained = sorted(p.parent.name for p in (args.data_dir / "tokenizers").glob("*/meta.json"))
    arms = args.arms or trained
    missing = [a for a in arms if a not in trained]
    if missing or not arms:
        sys.exit(f"No trained tokenizer for: {missing or 'any arm'}. Run scripts/05_train_tokenizers.py first.")

    for name in arms:
        encode_arm(name, args.data_dir, args.workers)

    report = {}
    for meta_path in sorted((args.data_dir / "tokens").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        report[meta["name"]] = meta
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "encoding.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'arm':16}{'vocab':>8}{'train tokens':>17}{'chars/token':>13}")
    for name, meta in report.items():
        tr = meta["splits"]["train"]
        print(f"{name:16}{meta['vocab_size']:>8}{tr['tokens']:>17,}{1 / tr['tokens_per_char']:>13.3f}")
    print(f"\nSaved {args.report_dir / 'encoding.json'}")


if __name__ == "__main__":
    main()
