"""Measure real training speed and memory on one GPU, before any long run.

    python scripts/07_calibrate.py --data-dir D:/urdu-lm-data --gpu 0
    python scripts/07_calibrate.py --data-dir D:/urdu-lm-data --gpu 1

For each arm and each micro-batch size it runs real training steps on real data
and records tokens per second and peak GPU memory. Nothing is saved from the
model. Results are merged into reports/calibration.json, which decides the
micro-batch size and how long the training budget takes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from urdu_lm.training import (TokenData, build_model, build_optimizer, lr_at, sample_step,  # noqa: E402
                              setup_backend, train_step)


def measure(cfg: dict, data: TokenData, micro_batch: int, device: torch.device, warmup: int, steps: int) -> dict:
    block = cfg["model"]["block_size"]
    torch.manual_seed(0)
    model = build_model(cfg, data.vocab_size, device)
    optimizer = build_optimizer(model, cfg, device)
    train = data.split("train")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for i in range(warmup + steps):
            if i == warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
            # One micro-batch per step here: speed per token is what is being measured.
            x, y = sample_step(train, 0, i, micro_batch, block)
            loss = train_step(model, optimizer, x, y, micro_batch, device, lr_at(i, cfg), cfg["grad_clip"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        result = {
            "tokens_per_sec": round(steps * micro_batch * block / seconds),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None,
            "loss_after": round(loss, 3),
        }
    except torch.OutOfMemoryError:
        result = {"out_of_memory": True}
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=REPO / "reports")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=["bpe-32k", "bpe-64k", "char"])
    parser.add_argument("--micro-batches", nargs="+", type=int, default=[8, 16, 24, 32, 48])
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "base.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cpu", action="store_true", help="for tiny test runs only")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    setup_backend()
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    key = f"gpu{args.gpu}" if device.type == "cuda" else "cpu"
    tokens_per_step = cfg["sequences_per_step"] * cfg["model"]["block_size"]

    results: dict = {"device": gpu_name, "torch": torch.__version__, "arms": {}}
    for arm in args.arms:
        if not (args.data_dir / "tokens" / arm / "meta.json").exists():
            print(f"[{arm}] not encoded yet, skipping")
            continue
        data = TokenData(args.data_dir, arm)
        results["arms"][arm] = {}
        for mb in args.micro_batches:
            r = measure(cfg, data, mb, device, args.warmup, args.steps)
            results["arms"][arm][str(mb)] = r
            if r.get("out_of_memory"):
                print(f"[{arm}] micro-batch {mb:>3}: out of memory", flush=True)
                break
            hours = cfg["max_steps"] * tokens_per_step / r["tokens_per_sec"] / 3600
            r["hours_for_full_run"] = round(hours, 1)
            print(f"[{arm}] micro-batch {mb:>3}: {r['tokens_per_sec'] / 1000:6.1f}k tokens/s, "
                  f"{r['peak_memory_gb']} GB peak, full run {hours:.1f} h", flush=True)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / "calibration.json"
    merged = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    merged[key] = results
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
