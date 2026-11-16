"""Train one model: one tokenizer arm, one seed, one GPU.

    python scripts/train.py --data-dir D:/urdu-lm-data --arm bpe-32k --seed 1 --gpu 0

The run lives in <data-dir>/runs/<arm>-s<seed>/:
    config.json   everything needed to repeat the run
    log.jsonl     training loss, validation loss, speed
    ckpt.pt       latest checkpoint (model, optimiser, step)
    final.pt      model weights at the end
    done.json     written last, with the summary

Stopping and starting again with the same command resumes from the last
checkpoint, and trains on exactly the data it would have seen. A finished run
is skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from urdu_lm.training import (TokenData, build_model, build_optimizer, lr_at, quick_val,  # noqa: E402
                              sample_step, setup_backend, train_step)


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "base.json")
    parser.add_argument("--micro-batch", type=int, default=0, help="override; does not change what is trained on")
    parser.add_argument("--max-steps", type=int, default=0, help="override, for test runs only")
    parser.add_argument("--run-suffix", default="", help="extra text in the run folder name, for test runs")
    parser.add_argument("--cpu", action="store_true", help="for tiny test runs only")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.micro_batch:
        cfg["micro_batch"] = args.micro_batch
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
        cfg["warmup_steps"] = min(cfg["warmup_steps"], max(1, args.max_steps // 10))

    run_dir = args.data_dir / "runs" / f"{args.arm}-s{args.seed}{args.run_suffix}"
    if (run_dir / "done.json").exists():
        print(f"{run_dir.name} is already finished, skipping.")
        return
    if not (args.data_dir / "tokens" / args.arm / "meta.json").exists():
        sys.exit(f"No token files for arm {args.arm!r}. Run scripts/06_encode.py first.")

    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    setup_backend()
    data = TokenData(args.data_dir, args.arm)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = data.split("train")
    val = data.split("val")
    block = cfg["model"]["block_size"]

    torch.manual_seed(args.seed)
    model = build_model(cfg, data.vocab_size, device)
    optimizer = build_optimizer(model, cfg, device)
    params = model.parameter_counts()

    step, elapsed, best_val = 0, 0.0, math.inf
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, elapsed, best_val = ckpt["step"], ckpt["elapsed"], ckpt["best_val"]
        print(f"Resumed {run_dir.name} at step {step:,}")
    else:
        run_config = {
            "arm": args.arm, "seed": args.seed, "config": cfg, "vocab_size": data.vocab_size,
            "parameters": params, "train_tokens": int(len(train)),
            "token_positions_in_budget": cfg["max_steps"] * cfg["sequences_per_step"] * block,
            "torch": torch.__version__, "platform": platform.platform(),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        }
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    tokens_per_step = cfg["sequences_per_step"] * block
    val_tokens_per_char = data.tokens_per_char("val")
    print(f"{run_dir.name}: {params['total'] / 1e6:.1f}M parameters ({params['body'] / 1e6:.1f}M body), "
          f"{cfg['max_steps']:,} steps of {tokens_per_step:,} tokens on {device}", flush=True)

    log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
    model.train()
    window_loss, window_steps, window_start = 0.0, 0, time.time()
    session_start, session_first_step = time.time(), step

    def evaluate(at_step: int) -> float:
        nonlocal best_val
        val_loss = quick_val(model, val, block, cfg["micro_batch"], cfg["eval_sequences"], device)
        val_bpc = val_loss / math.log(2) * val_tokens_per_char
        best_val = min(best_val, val_loss)
        log.write(json.dumps({"step": at_step, "val_loss": round(val_loss, 5), "val_bpc": round(val_bpc, 5)}) + "\n")
        log.flush()
        print(f"  step {at_step:>7,}  validation loss {val_loss:.4f}  bits per character {val_bpc:.4f}", flush=True)
        return val_bpc

    def checkpoint() -> None:
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                     "elapsed": elapsed + time.time() - session_start, "best_val": best_val}, ckpt_path)

    while step < cfg["max_steps"]:
        x, y = sample_step(train, args.seed, step, cfg["sequences_per_step"], block)
        loss = train_step(model, optimizer, x, y, cfg["micro_batch"], device, lr_at(step, cfg), cfg["grad_clip"])
        step += 1
        if not math.isfinite(loss):
            sys.exit(f"Loss became {loss} at step {step}. Stopping. The last checkpoint is intact.")
        window_loss += loss
        window_steps += 1

        if step % cfg["log_every"] == 0:
            now = time.time()
            speed = window_steps * tokens_per_step / (now - window_start)
            session_speed = (step - session_first_step) / (now - session_start)
            remaining = (cfg["max_steps"] - step) / session_speed
            record = {"step": step, "train_loss": round(window_loss / window_steps, 5),
                      "lr": round(lr_at(step - 1, cfg), 8), "tokens_per_sec": round(speed),
                      "elapsed_s": round(elapsed + now - session_start)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"step {step:>7,}/{cfg['max_steps']:,}  loss {record['train_loss']:.4f}  "
                  f"{speed / 1000:.1f}k tokens/s  remaining {fmt_time(remaining)}", flush=True)
            window_loss, window_steps, window_start = 0.0, 0, time.time()

        if step % cfg["eval_every"] == 0 and step < cfg["max_steps"]:
            evaluate(step)
            window_start = time.time()
        if step % cfg["ckpt_every"] == 0 and step < cfg["max_steps"]:
            checkpoint()
            window_start = time.time()

    final_bpc = evaluate(step)
    total_elapsed = elapsed + time.time() - session_start
    save_atomic({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "arm": args.arm,
                 "seed": args.seed, "step": step}, run_dir / "final.pt")
    done = {"arm": args.arm, "seed": args.seed, "steps": step, "final_val_bpc_quick": round(final_bpc, 5),
            "hours": round(total_elapsed / 3600, 2), "parameters": params}
    (run_dir / "done.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
    if ckpt_path.exists():
        ckpt_path.unlink()
    log.close()
    print(f"Finished {run_dir.name} in {fmt_time(total_elapsed)}.")


if __name__ == "__main__":
    main()
