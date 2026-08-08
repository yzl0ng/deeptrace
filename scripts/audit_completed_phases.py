from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.phase_audit import (
    audit_completed_phases,
    write_phase_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the saved acceptance evidence for Phase 0 through 4."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/system/phase0-4-gap-audit-v1"),
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    audit = audit_completed_phases(PROJECT_ROOT)
    write_phase_audit(audit, output_dir)
    for phase in audit.phases:
        passed = sum(phase.checks.values())
        print(
            f"{phase.phase}: {phase.status} "
            f"({passed}/{len(phase.checks)} checks)"
        )
    print(f"overall: {audit.overall_status}")
    return 0 if audit.overall_status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
