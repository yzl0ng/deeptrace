from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.agentic.nccl_matrix import (
    NCCL_MATRIX,
    run_nccl_matrix,
    write_nccl_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute the Phase 0 1/4/8-GPU NCCL smoke matrix. "
            "Planning is the default and launches no GPU work."
        )
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=[check.name for check in NCCL_MATRIX],
        default=[check.name for check in NCCL_MATRIX],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--max-used-memory-mib",
        type=int,
        default=64,
        help=(
            "A GPU is eligible only when it has no reported compute process "
            "and used memory does not exceed this threshold."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/system/nccl-smoke-v4"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    payload = run_nccl_matrix(
        selected_names=args.checks,
        python_executable=args.python,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        execute=args.execute,
        max_used_memory_mib=args.max_used_memory_mib,
    )
    write_nccl_matrix(payload, args.output_dir)
    print(
        f"wrote {len(payload['planned_checks'])} checks with "
        f"status={payload['status']} to {args.output_dir}"
    )
    return 0 if payload["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
