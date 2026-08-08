from __future__ import annotations

import argparse
from pathlib import Path

from app.agentic.system_audit import AUDIT_VERSION, collect_audit, write_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect a redacted, reproducible SearchLab server audit."
    )
    parser.add_argument(
        "--ssh-host",
        help="SSH config alias. The resolved hostname is never stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--audit-version", default=AUDIT_VERSION)
    parser.add_argument("--timeout-seconds", type=float, default=20)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("data/system") / args.audit_version

    audit = collect_audit(
        ssh_host=args.ssh_host,
        timeout_seconds=args.timeout_seconds,
        audit_version=args.audit_version,
    )
    write_audit(audit, output_dir)
    failed = sum(
        result["status"] != "succeeded" for result in audit["results"]
    )
    print(
        f"wrote {len(audit['results'])} checks to {output_dir}; "
        f"{failed} unavailable or failed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
