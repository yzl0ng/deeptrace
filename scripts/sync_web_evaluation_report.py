from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "data" / "reports" / "retrieval_quality_baseline.json"
)
TARGET = PROJECT_ROOT / "web" / "app" / "evaluation-report.json"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            "Run scripts/run_retrieval_evaluation.py before syncing the report."
        )
    shutil.copyfile(SOURCE, TARGET)
    print(f"Synced {SOURCE.relative_to(PROJECT_ROOT)} to {TARGET.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
