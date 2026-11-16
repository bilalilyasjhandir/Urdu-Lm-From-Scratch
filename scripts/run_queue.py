"""Run several training runs one after another on one GPU.

    python scripts/run_queue.py --data-dir D:/urdu-lm-data --gpu 0 --runs bpe-32k:1 morph-bpe-32k:1 char:1
    python scripts/run_queue.py --data-dir D:/urdu-lm-data --gpu 1 --runs unigram-32k:1 bpe-32k:2

Each item is arm:seed. Start one queue per GPU, each in its own terminal window.
Finished runs are skipped and interrupted runs resume, so after a restart or a
crash the same command simply continues. If one run fails, the queue moves on
and lists the failure at the end.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="items of the form arm:seed")
    parser.add_argument("--micro-batch", type=int, default=0)
    args = parser.parse_args()

    failed = []
    for item in args.runs:
        arm, seed = item.rsplit(":", 1)
        cmd = [sys.executable, str(REPO / "scripts" / "train.py"), "--data-dir", str(args.data_dir),
               "--arm", arm, "--seed", seed, "--gpu", str(args.gpu)]
        if args.micro_batch:
            cmd += ["--micro-batch", str(args.micro_batch)]
        print(f"\n===== {time.strftime('%Y-%m-%d %H:%M')}  GPU {args.gpu}  {arm} seed {seed} =====", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(item)
            print(f"!!!!! {item} failed, continuing with the next run", flush=True)

    print(f"\nQueue finished at {time.strftime('%Y-%m-%d %H:%M')}.")
    if failed:
        print("Failed runs:", " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
