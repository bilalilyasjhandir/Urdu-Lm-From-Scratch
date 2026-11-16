"""Pieces shared by the training, calibration and evaluation scripts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .model import GPT, ModelConfig


class TokenData:
    """Memory-mapped token files of one tokenizer arm."""

    def __init__(self, data_dir: str | Path, arm: str):
        self.dir = Path(data_dir) / "tokens" / arm
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.vocab_size: int = self.meta["vocab_size"]
        self._maps: dict[str, np.memmap] = {}

    def split(self, name: str) -> np.memmap:
        if name not in self._maps:
            self._maps[name] = np.memmap(self.dir / f"{name}.bin", dtype=np.uint16, mode="r")
        return self._maps[name]

    def tokens_per_char(self, name: str) -> float:
        s = self.meta["splits"][name]
        return s["tokens"] / s["chars"]


def sample_step(data: np.memmap, seed: int, step: int, sequences: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """The training sequences of one optimiser step.

    They depend only on (seed, step), so a run that is stopped and resumed sees
    exactly the data it would have seen, and the micro-batch size (how the
    sequences are fed through the GPU) cannot change what is trained on.
    """
    rng = np.random.default_rng([seed, step])
    starts = rng.integers(0, len(data) - block_size - 1, size=sequences)
    x = np.stack([data[s:s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + 1 + block_size] for s in starts]).astype(np.int64)
    return x, y


def lr_at(step: int, cfg: dict) -> float:
    """Linear warm-up, then cosine decay to lr * min_lr_ratio at max_steps."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"]))
    floor = cfg["lr"] * cfg["min_lr_ratio"]
    return floor + 0.5 * (cfg["lr"] - floor) * (1.0 + math.cos(math.pi * progress))


def build_model(cfg: dict, vocab_size: int, device: torch.device) -> GPT:
    model = GPT(ModelConfig(vocab_size=vocab_size, **cfg["model"]))
    return model.to(device)


def build_optimizer(model: GPT, cfg: dict, device: torch.device) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.optimizer_groups(cfg["weight_decay"]), lr=cfg["lr"],
                             betas=tuple(cfg["betas"]), fused=(device.type == "cuda"))


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def setup_backend() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train_step(model: GPT, optimizer: torch.optim.Optimizer, x: np.ndarray, y: np.ndarray,
               micro_batch: int, device: torch.device, lr: float, grad_clip: float) -> float:
    """One optimiser step over all sequences in x, fed through the GPU micro_batch at a time."""
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    n = x.shape[0]
    for i in range(0, n, micro_batch):
        xb = torch.from_numpy(x[i:i + micro_batch]).to(device, non_blocking=True)
        yb = torch.from_numpy(y[i:i + micro_batch]).to(device, non_blocking=True)
        with autocast(device):
            loss = model.loss(xb, yb)
        weight = xb.shape[0] / n
        (loss * weight).backward()
        total += loss.item() * weight
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return total


@torch.no_grad()
def quick_val(model: GPT, data: np.memmap, block_size: int, micro_batch: int, max_sequences: int,
              device: torch.device) -> float:
    """Mean loss per token over the first non-overlapping windows of the validation stream.

    A fast progress signal during training. The reported results come from
    scripts/evaluate.py, which scores every test token with a sliding window.
    """
    model.eval()
    n_seq = min(max_sequences, (len(data) - 1) // block_size)
    total, count = 0.0, 0
    for i in range(0, n_seq, micro_batch):
        idx = range(i, min(i + micro_batch, n_seq))
        x = np.stack([data[j * block_size:(j + 1) * block_size] for j in idx]).astype(np.int64)
        y = np.stack([data[j * block_size + 1:(j + 1) * block_size + 1] for j in idx]).astype(np.int64)
        xb, yb = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with autocast(device):
            loss = model.loss(xb, yb, reduction="sum")
        total += loss.item()
        count += yb.numel()
    model.train()
    return total / max(1, count)
