"""Download the raw data. Safe to re-run: finished files are skipped.

    python scripts/01_download.py --data-dir D:/urdu-lm-data            # everything (about 7 GB)
    python scripts/01_download.py --data-dir D:/urdu-lm-data --sample   # 71 MB, for a dry run

Sources
  FineWeb2, Urdu subset (HuggingFaceFW/fineweb-2, config urd_Arab, ODC-By):
      web text from Common Crawl, already language-identified and near-deduplicated.
  MultiBLiMP, Urdu part (jumelet/multiblimp, CC BY 4.0):
      grammatical / ungrammatical sentence pairs, used only for evaluation.

The exact dataset revisions are written to <data-dir>/raw/SOURCES.json the first
time, and reused afterwards, so a resumed download cannot mix two versions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SOURCES = {
    "fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "full": ["data/urd_Arab/train/*.parquet", "data/urd_Arab/test/*.parquet"],
        "sample": ["data/urd_Arab/test/*.parquet"],
    },
    "multiblimp": {
        "repo_id": "jumelet/multiblimp",
        "full": ["urd/*"],
        "sample": ["urd/*"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample", action="store_true", help="only the small FineWeb2 test file")
    args = parser.parse_args()

    raw = args.data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pins_path = raw / "SOURCES.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8")) if pins_path.exists() else {}

    api = HfApi()
    for name, src in SOURCES.items():
        if name not in pins:
            pins[name] = {"repo_id": src["repo_id"], "revision": api.dataset_info(src["repo_id"]).sha}
            pins_path.write_text(json.dumps(pins, indent=2), encoding="utf-8")
        patterns = src["sample"] if args.sample else src["full"]
        print(f"[{name}] {src['repo_id']} @ {pins[name]['revision'][:12]}  {patterns}")
        snapshot_download(
            repo_id=src["repo_id"],
            repo_type="dataset",
            revision=pins[name]["revision"],
            allow_patterns=patterns,
            local_dir=raw / name,
        )

    files = sorted(p for p in raw.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name != "SOURCES.json")
    total = sum(p.stat().st_size for p in files)
    print(f"\n{len(files)} files, {total / 1e9:.2f} GB under {raw}")
    for p in files:
        print(f"  {p.stat().st_size / 1e6:9.1f} MB  {p.relative_to(raw)}")


if __name__ == "__main__":
    main()
