from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any


def expected_all_reduce_sum(world_size: int) -> float:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return float(world_size * (world_size + 1) // 2)


def run_nccl_smoke(timeout_seconds: int = 120) -> dict[str, Any] | None:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed; install a driver-compatible CUDA build "
            "inside an isolated project environment before running this smoke test"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    if not dist.is_nccl_available():
        raise RuntimeError("the installed PyTorch build does not provide NCCL")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    started = time.perf_counter()
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=timeout_seconds),
    )
    try:
        value = torch.tensor(
            [float(rank + 1)],
            device=f"cuda:{local_rank}",
            dtype=torch.float32,
        )
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        observed = float(value.item())
        expected = expected_all_reduce_sum(world_size)
        if observed != expected:
            raise RuntimeError(
                f"all_reduce mismatch: observed={observed}, expected={expected}"
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = {
            "status": "succeeded",
            "backend": "nccl",
            "world_size": world_size,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "device_name": torch.cuda.get_device_name(local_rank),
            "observed_sum": observed,
            "expected_sum": expected,
            "elapsed_ms": round(elapsed_ms, 3),
        }
        return payload if rank == 0 else None
    finally:
        dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal deterministic NCCL all-reduce smoke test."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    payload = run_nccl_smoke(timeout_seconds=args.timeout_seconds)
    if payload is not None:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
