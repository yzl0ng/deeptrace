from __future__ import annotations

import argparse

from app.agentic.nccl_matrix import discover_idle_devices


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a comma-separated list of dynamically selected idle GPUs."
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--max-used-memory-mib", type=int, default=64)
    parser.add_argument(
        "--allow-compute-processes",
        action="store_true",
        help=(
            "Allow GPUs with existing compute processes when their total "
            "reported memory remains within the configured limit."
        ),
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    idle_devices = discover_idle_devices(
        max_used_memory_mib=args.max_used_memory_mib,
        require_no_compute_processes=not args.allow_compute_processes,
    )
    if len(idle_devices) < args.count:
        parser.error(
            f"requires {args.count} idle GPUs but found {len(idle_devices)}: "
            f"{idle_devices}"
        )

    print(",".join(map(str, idle_devices[: args.count])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
