from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agentic.training_readiness import (
    READINESS_VERSION,
    OperatorInputs,
    TrainingStackRequirements,
    evaluate_training_readiness,
    write_readiness_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved Phase 0 training readiness gates."
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/system/server-audit-v2/environment.json"),
    )
    parser.add_argument(
        "--nccl-validation",
        type=Path,
        default=Path("data/system/server-audit-v1/nccl-validation.json"),
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path("config/training-stack.lock.json"),
    )
    parser.add_argument(
        "--operator-inputs",
        type=Path,
        default=Path("config/training-operator-inputs.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/system/training-readiness-v1"),
    )
    parser.add_argument("--readiness-version", default=READINESS_VERSION)
    args = parser.parse_args()

    audit = _read_json(args.audit)
    nccl_validation = _read_json(args.nccl_validation)
    requirements = TrainingStackRequirements.model_validate(
        _read_json(args.requirements)
    )
    operator_inputs = OperatorInputs.model_validate(
        _read_json(args.operator_inputs)
    )
    result = evaluate_training_readiness(
        audit=audit,
        nccl_validation=nccl_validation,
        operator_inputs=operator_inputs,
        requirements=requirements,
        readiness_version=args.readiness_version,
    )
    write_readiness_artifacts(
        result,
        requirements=requirements,
        operator_inputs=operator_inputs,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "passed": sum(
                    gate.status.value == "passed" for gate in result.gates
                ),
                "pending": sum(
                    gate.status.value == "pending" for gate in result.gates
                ),
                "blocked": sum(
                    gate.status.value == "blocked" for gate in result.gates
                ),
            },
            indent=2,
        )
    )
    return 0 if result.status == "ready" else 2


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
