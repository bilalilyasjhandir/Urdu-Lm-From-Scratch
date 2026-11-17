"""Score finished runs. Safe to re-run: runs that already have eval.json are skipped.

    python scripts/evaluate.py --data-dir D:/urdu-lm-data --gpu 0

For every finished run under <data-dir>/runs it measures

  Bits per character on the validation and test sets. Every token is scored
      exactly once with a sliding window, so each prediction has at least half a
      context window behind it. Total bits are divided by the number of
      characters, which is identical for all tokenizers.

  MultiBLiMP (Urdu). For each pair of sentences that differ in one agreement
      feature, the model is right if it gives the grammatical sentence the higher
      total probability.

Results go to <run>/eval.json, and all runs are combined in reports/results.json
and reports/results.md (mean and standard deviation over seeds, per arm).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from urdu_lm.model import GPT, ModelConfig  # noqa: E402
from urdu_lm.normalize import normalize  # noqa: E402
from urdu_lm.tokenizer import Tokenizer  # noqa: E402
from urdu_lm.training import TokenData, autocast, setup_backend  # noqa: E402


@torch.no_grad()
def token_nll(model: GPT, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Negative log-likelihood of every target token, shape (batch, time), in nats."""
    with autocast(device):
        logits = model(x)
    return F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").double()


@torch.no_grad()
def bits_per_character(model: GPT, tokens: np.ndarray, chars: int, block: int, batch: int,
                       device: torch.device) -> dict:
    n = len(tokens)
    stride = block // 2
    starts = list(range(0, max(1, n - 1 - block + stride), stride))
    total, scored = 0.0, 0
    for i in range(0, len(starts), batch):
        group = starts[i:i + batch]
        full = [s for s in group if s + block + 1 <= n]
        partial = [s for s in group if s + block + 1 > n]
        for subset in ([full] if full else []) + [[s] for s in partial]:
            length = min(block, n - 1 - subset[0])
            x = torch.from_numpy(np.stack([tokens[s:s + length] for s in subset]).astype(np.int64)).to(device)
            y = torch.from_numpy(np.stack([tokens[s + 1:s + 1 + length] for s in subset]).astype(np.int64)).to(device)
            nll = token_nll(model, x, y, device)
            for row, s in enumerate(subset):
                first = 0 if s == 0 else block - stride
                total += nll[row, first:].sum().item()
                scored += length - first
    assert scored == n - 1, (scored, n)
    # The first token (<eod> of the first document) is given, not predicted.
    return {"bits_per_character": total / math.log(2) / (chars - 1),
            "loss_per_token": total / scored, "tokens": n, "characters": chars}


@torch.no_grad()
def multiblimp(model: GPT, tok: Tokenizer, path: Path, block: int, batch: int, device: torch.device) -> dict:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    # The source treebank joins multiword tokens with an underscore, which never
    # occurs in running Urdu text. Both sentences of a pair get the space back.
    with_underscore = sum("_" in r["sen"] for r in rows)
    for r in rows:
        r["sen"] = normalize(r["sen"].replace("_", " "))
        r["wrong_sen"] = normalize(r["wrong_sen"].replace("_", " "))

    def score(sentences: list[str]) -> list[float]:
        encoded = [tok.encode(s).astype(np.int64) for s in sentences]
        out = []
        for i in range(0, len(encoded), batch):
            chunk = encoded[i:i + batch]
            width = max(len(e) for e in chunk)
            x = np.zeros((len(chunk), width - 1), dtype=np.int64)
            y = np.zeros((len(chunk), width - 1), dtype=np.int64)
            mask = np.zeros((len(chunk), width - 1), dtype=bool)
            for r, e in enumerate(chunk):
                x[r, :len(e) - 1], y[r, :len(e) - 1], mask[r, :len(e) - 1] = e[:-1], e[1:], True
            nll = token_nll(model, torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), device)
            out.extend((nll * torch.from_numpy(mask).to(device)).sum(dim=1).tolist())
        return out

    usable = [r for r in rows if max(len(tok.encode(r["sen"])), len(tok.encode(r["wrong_sen"]))) <= block]
    good = score([r["sen"] for r in usable])
    bad = score([r["wrong_sen"] for r in usable])
    by_phenomenon: dict[str, list[bool]] = defaultdict(list)
    for r, g, b in zip(usable, good, bad):
        by_phenomenon[r["phenomenon"]].append(g < b)   # lower negative log-likelihood wins
    correct = [c for group in by_phenomenon.values() for c in group]
    return {"accuracy": sum(correct) / max(1, len(correct)), "pairs": len(correct),
            "pairs_skipped_too_long": len(rows) - len(usable),
            "pairs_with_underscores_replaced_by_spaces": with_underscore,
            "by_phenomenon": {k: {"accuracy": sum(v) / len(v), "pairs": len(v)} for k, v in sorted(by_phenomenon.items())}}


def evaluate_run(run_dir: Path, data_dir: Path, device: torch.device, batch: int) -> dict:
    ckpt = torch.load(run_dir / "final.pt", map_location=device, weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    arm = ckpt["arm"]
    block = model.cfg.block_size
    data = TokenData(data_dir, arm)
    tok = Tokenizer(data_dir / "tokenizers" / arm)
    run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    done = json.loads((run_dir / "done.json").read_text(encoding="utf-8"))

    result = {"run": run_dir.name, "arm": arm, "seed": ckpt["seed"], "steps": ckpt["step"],
              "vocab_size": data.vocab_size, "parameters": done["parameters"], "hours": done["hours"],
              "characters_per_token_train": round(1 / data.tokens_per_char("train"), 4),
              "characters_read_in_training": round(run_cfg["token_positions_in_budget"] / data.tokens_per_char("train"))}
    for split in ("val", "test"):
        result[split] = bits_per_character(model, data.split(split), data.meta["splits"][split]["chars"],
                                           block, batch, device)
    blimp_path = data_dir / "raw" / "multiblimp" / "urd" / "data.tsv"
    if blimp_path.exists():
        result["multiblimp"] = multiblimp(model, tok, blimp_path, block, batch, device)
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def write_summary(results: list[dict], report_dir: Path) -> str:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)

    def cell(values: list[float], digits: int) -> str:
        m, s = mean_std(values)
        return f"{m:.{digits}f}" if math.isnan(s) else f"{m:.{digits}f} ± {s:.{digits}f}"

    lines = ["# Results", "", "Generated by `scripts/evaluate.py`. Do not edit by hand.",
             "Mean ± standard deviation over seeds. Lower bits per character is better.", "",
             "| Arm | Seeds | Test bits per character | Validation bits per character | MultiBLiMP accuracy | "
             "Characters per token | Text read in training (characters) | Parameters |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    order = sorted(by_arm, key=lambda a: statistics.mean(r["test"]["bits_per_character"] for r in by_arm[a]))
    for arm in order:
        runs = by_arm[arm]
        blimp = [r["multiblimp"]["accuracy"] * 100 for r in runs if "multiblimp" in r]
        lines.append(
            f"| {arm} | {len(runs)} | {cell([r['test']['bits_per_character'] for r in runs], 4)} | "
            f"{cell([r['val']['bits_per_character'] for r in runs], 4)} | "
            f"{cell(blimp, 1) + '%' if blimp else 'n/a'} | {runs[0]['characters_per_token_train']} | "
            f"{runs[0]['characters_read_in_training'] / 1e9:.2f} billion | {runs[0]['parameters']['total'] / 1e6:.1f}M |")
    text = "\n".join(lines) + "\n"
    (report_dir / "results.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=REPO / "reports")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="score again even if eval.json exists")
    parser.add_argument("--cpu", action="store_true", help="for tiny test runs only")
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")   # the Windows console cannot print every character
    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    setup_backend()
    results = []
    for run_dir in sorted((args.data_dir / "runs").glob("*")):
        # Folders with an underscore suffix (for example _smoke) are test runs.
        if not (run_dir / "done.json").exists() or "_" in run_dir.name:
            continue
        eval_path = run_dir / "eval.json"
        if eval_path.exists() and not args.force:
            results.append(json.loads(eval_path.read_text(encoding="utf-8")))
            continue
        print(f"Scoring {run_dir.name} ...", flush=True)
        r = evaluate_run(run_dir, args.data_dir, device, args.batch)
        print(f"  test {r['test']['bits_per_character']:.4f} bits per character"
              + (f", MultiBLiMP {r['multiblimp']['accuracy']:.1%}" if "multiblimp" in r else ""), flush=True)
        results.append(r)

    if not results:
        sys.exit("No finished runs found.")
    print("\n" + write_summary(results, args.report_dir))


if __name__ == "__main__":
    main()
