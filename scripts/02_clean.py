"""Normalise, filter, deduplicate and split the raw corpus. Safe to re-run.

    python scripts/02_clean.py --data-dir D:/urdu-lm-data

Step 1 (parallel): every block of raw rows is normalised and filtered, and written
    as an intermediate part file. Finished parts are skipped on a re-run.
Step 2 (single process): exact duplicates are dropped, and documents are routed to
    train / validation / test.

The split is by website, not by document. Pages from one site share templates,
writers and topics, so a random document split would let the model see near-copies
of its test pages during training. Here about 0.5% of sites go to validation and
0.5% to test, and no page from those sites is ever trained on.

Output, under <data-dir>/clean:
    train/train-00000.parquet ...   columns: text, url, domain, hash
    heldout/val_full.parquet, test_full.parquet     every held-out document
    heldout/val.parquet, test.parquet               a fixed sample, used for evaluation
and reports/cleaning_report.json in the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from urdu_lm.filters import FilterConfig, reject_reason  # noqa: E402
from urdu_lm.normalize import RULES, normalize  # noqa: E402

SPLIT_KEY = b"urdu-lm-split-v1"
VAL_PER_MILLE = 5
TEST_PER_MILLE = 5
SCHEMA = pa.schema([
    ("text", pa.large_string()),
    ("url", pa.string()),
    ("domain", pa.string()),
    ("hash", pa.uint64()),
    ("split", pa.string()),
])


def text_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def domain_of(url: str | None, fallback: int) -> str:
    host = urlsplit(url).netloc.lower() if url else ""
    host = host.split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host or f"unknown-{fallback:016x}"


def split_of(domain: str) -> str:
    digest = hashlib.blake2b(domain.encode("utf-8"), digest_size=8, key=SPLIT_KEY).digest()
    bucket = int.from_bytes(digest, "little") % 1000
    if bucket < VAL_PER_MILLE:
        return "val"
    if bucket < VAL_PER_MILLE + TEST_PER_MILLE:
        return "test"
    return "train"


def clean_task(task: tuple[int, str, list[int], str, dict]) -> dict:
    index, path, row_groups, parts_dir, cfg_dict = task
    out = Path(parts_dir) / f"part-{index:05d}.parquet"
    report_path = out.with_suffix(".json")
    if out.exists() and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    cfg = FilterConfig(**cfg_dict)
    table = pq.ParquetFile(path).read_row_groups(row_groups, columns=["text", "url"])
    counts: Counter = Counter()
    rows: dict[str, list] = {name: [] for name in SCHEMA.names}
    for text, url in zip(table.column("text").to_pylist(), table.column("url").to_pylist()):
        counts["docs_in"] += 1
        counts["chars_in"] += len(text or "")
        clean = normalize(text or "")
        reason = reject_reason(clean, cfg)
        if reason:
            counts[f"rejected_{reason}"] += 1
            continue
        h = text_hash(clean)
        domain = domain_of(url, h)
        rows["text"].append(clean)
        rows["url"].append(url)
        rows["domain"].append(domain)
        rows["hash"].append(h)
        rows["split"].append(split_of(domain))
        counts["docs_kept"] += 1
        counts["chars_kept"] += len(clean)

    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=SCHEMA), tmp, compression="zstd")
    os.replace(tmp, out)
    report = dict(counts)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report


def list_tasks(raw_dir: Path, groups_per_task: int) -> list[tuple[str, list[int]]]:
    tasks = []
    for path in sorted(raw_dir.rglob("*.parquet")):
        if ".cache" in path.parts:
            continue
        n = pq.ParquetFile(path).metadata.num_row_groups
        for start in range(0, n, groups_per_task):
            tasks.append((str(path), list(range(start, min(n, start + groups_per_task)))))
    return tasks


class ShardWriter:
    """Collects tables and writes a shard each time `shard_chars` characters are reached."""

    def __init__(self, out_dir: Path, prefix: str, shard_chars: int):
        self.out_dir, self.prefix, self.shard_chars = out_dir, prefix, shard_chars
        self.buffer: list[pa.Table] = []
        self.chars = 0
        self.count = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, table: pa.Table, chars: int) -> None:
        if table.num_rows:
            self.buffer.append(table)
            self.chars += chars
            if self.chars >= self.shard_chars:
                self.flush()

    def flush(self) -> None:
        if self.buffer:
            path = self.out_dir / f"{self.prefix}-{self.count:05d}.parquet"
            pq.write_table(pa.concat_tables(self.buffer), path, compression="zstd")
            self.buffer, self.chars = [], 0
            self.count += 1


def total_chars(table: pa.Table) -> int:
    return int(pa.compute.sum(pa.compute.utf8_length(table.column("text"))).as_py() or 0)


def merge(parts_dir: Path, clean_dir: Path, eval_docs: int, eval_docs_per_site: int, shard_chars: int) -> dict:
    for name in ("train", "heldout"):
        shutil.rmtree(clean_dir / name, ignore_errors=True)
    writer = ShardWriter(clean_dir / "train", "train", shard_chars)
    held: dict[str, list[pa.Table]] = {"val": [], "test": []}
    seen: set[int] = set()
    stats: Counter = Counter()

    for part in sorted(parts_dir.glob("part-*.parquet")):
        table = pq.read_table(part)
        keep = []
        for h in table.column("hash").to_pylist():
            if h in seen:
                keep.append(False)
            else:
                seen.add(h)
                keep.append(True)
        stats["exact_duplicates_removed"] += keep.count(False)
        table = table.filter(pa.array(keep))
        split_col = table.column("split")
        table = table.drop_columns(["split"])
        for split in ("train", "val", "test"):
            sub = table.filter(pa.compute.equal(split_col, split))
            chars = total_chars(sub)
            stats[f"{split}_docs"] += sub.num_rows
            stats[f"{split}_chars"] += chars
            if split == "train":
                writer.add(sub, chars)
            elif sub.num_rows:
                held[split].append(sub)
    writer.flush()
    stats["train_shards"] = writer.count

    heldout_dir = clean_dir / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    for split, tables in held.items():
        full = pa.concat_tables(tables) if tables else pa.table({n: [] for n in SCHEMA.names[:-1]},
                                                                 schema=pa.schema(list(SCHEMA)[:-1]))
        pq.write_table(full, heldout_dir / f"{split}_full.parquet", compression="zstd")
        # A fixed, order-independent sample: the documents with the smallest hashes,
        # with a cap per site so one large held-out site cannot dominate the evaluation.
        full = full.sort_by("hash")
        taken: Counter = Counter()
        picked = []
        for i, domain in enumerate(full.column("domain").to_pylist()):
            if taken[domain] < eval_docs_per_site:
                taken[domain] += 1
                picked.append(i)
                if len(picked) == eval_docs:
                    break
        sample = full.take(pa.array(picked, type=pa.int64()))
        pq.write_table(sample, heldout_dir / f"{split}.parquet", compression="zstd")
        stats[f"{split}_eval_docs"] = sample.num_rows
        stats[f"{split}_eval_chars"] = total_chars(sample)
        stats[f"{split}_eval_sites"] = len(taken)
        stats[f"{split}_sites"] = len(set(full.column("domain").to_pylist()))
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=REPO / "reports")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--groups-per-task", type=int, default=50, help="row groups (1,000 rows each) per part")
    parser.add_argument("--eval-docs", type=int, default=5000, help="documents in the val and test samples")
    parser.add_argument("--eval-docs-per-site", type=int, default=50)
    parser.add_argument("--shard-chars", type=int, default=250_000_000)
    parser.add_argument("--keep-parts", action="store_true", help="keep the intermediate part files")
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw" / "fineweb2"
    clean_dir = args.data_dir / "clean"
    parts_dir = clean_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    cfg = FilterConfig()

    tasks = list_tasks(raw_dir, args.groups_per_task)
    if not tasks:
        sys.exit(f"No parquet files under {raw_dir}. Run scripts/01_download.py first.")
    print(f"{len(tasks)} parts, {args.workers} workers")

    started = time.time()
    totals: Counter = Counter()
    jobs = [(i, path, groups, str(parts_dir), asdict(cfg)) for i, (path, groups) in enumerate(tasks)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(clean_task, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            totals.update(future.result())
            if done % 5 == 0 or done == len(futures):
                print(f"  cleaned {done}/{len(futures)} parts, {totals['docs_kept']:,} of "
                      f"{totals['docs_in']:,} documents kept, {time.time() - started:.0f}s", flush=True)

    print("Removing duplicates and writing the splits ...", flush=True)
    split_stats = merge(parts_dir, clean_dir, args.eval_docs, args.eval_docs_per_site, args.shard_chars)

    sources_path = args.data_dir / "raw" / "SOURCES.json"
    report = {
        "sources": json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {},
        "raw_files": sorted({Path(path).name for path, _ in tasks}),
        "normalisation_rules": RULES,
        "filters": asdict(cfg),
        "split": {"by": "website (URL host)", "val_per_mille_of_sites": VAL_PER_MILLE,
                  "test_per_mille_of_sites": TEST_PER_MILLE, "key": SPLIT_KEY.decode()},
        "cleaning": dict(totals),
        "splits": split_stats,
        "seconds": round(time.time() - started),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / "cleaning_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)

    print(json.dumps({"cleaning": report["cleaning"], "splits": split_stats}, indent=2))
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
