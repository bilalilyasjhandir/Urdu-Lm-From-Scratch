"""Print what the machine has, so every later result can be traced to an environment.

    python scripts/00_check_env.py

Writes reports/environment.json. Runs a short matrix-multiply test on every GPU.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def package_versions() -> dict[str, str]:
    out = {}
    for name in ["torch", "numpy", "pyarrow", "huggingface_hub", "sentencepiece", "tqdm"]:
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = "NOT INSTALLED"
    return out


def gpu_report() -> list[dict]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        dev = torch.device(f"cuda:{i}")
        x = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
        for _ in range(3):  # warm-up: the first calls on a card include library start-up
            y = x @ x
        torch.cuda.synchronize(dev)
        start = time.perf_counter()
        for _ in range(20):
            y = x @ x
        torch.cuda.synchronize(dev)
        seconds = time.perf_counter() - start
        del x, y
        torch.cuda.empty_cache()
        gpus.append({
            "index": i,
            "name": props.name,
            "memory_total_gb": round(total / 2**30, 2),
            "memory_free_gb": round(free / 2**30, 2),
            "bf16_autocast": bool(torch.cuda.is_bf16_supported()),
            "fp16_matmul_8192_x20_seconds": round(seconds, 3),
        })
    return gpus


def main() -> None:
    try:
        import torch
        torch_info = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                      "cuda_available": torch.cuda.is_available()}
    except Exception as exc:  # noqa: BLE001
        torch_info = {"error": str(exc)}

    disk = shutil.disk_usage(REPO)
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch_info,
        "packages": package_versions(),
        "gpus": gpu_report(),
        "repo_drive_free_gb": round(disk.free / 2**30, 1),
    }
    try:
        import psutil  # optional
        report["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass

    print(json.dumps(report, indent=2))
    out = REPO / "reports" / "environment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")
    if not report["gpus"]:
        print("\nWARNING: no CUDA GPU visible. Training cannot run in this environment.")


if __name__ == "__main__":
    main()
