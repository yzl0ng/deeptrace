from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.agentic.runtime_smoke_plan import (
    run_runtime_smoke_plan,
    write_runtime_smoke_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute the reserved Phase 0 CUDA extension and vLLM "
            "single-GPU smokes. Planning is the default and launches no GPU."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/system/gpu-runtime-smoke-v1"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--reservation-confirmed",
        action="store_true",
        help="Assert that the selected shared GPU is reserved for this run.",
    )
    parser.add_argument(
        "--reservation-reference",
        default="",
        help="Auditable ticket, message, or coordination record.",
    )
    args = parser.parse_args()

    payload = run_runtime_smoke_plan(
        python_executable=args.python,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        timeout_seconds=args.timeout_seconds,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        execute=args.execute,
        reservation_confirmed=args.reservation_confirmed,
        reservation_reference=args.reservation_reference,
    )
    write_runtime_smoke_plan(payload, args.output_dir)
    print(
        f"wrote {len(payload['checks'])} checks with "
        f"status={payload['status']} to {args.output_dir}"
    )
    return 0 if payload["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
