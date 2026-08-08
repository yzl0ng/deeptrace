from __future__ import annotations

import argparse
from pathlib import Path

from app.agentic.cuda_toolkit_smoke import (
    run_cuda_toolkit_smoke,
    write_cuda_toolkit_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a minimal PTX kernel with an isolated CUDA toolkit."
    )
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_89")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/system/cuda-toolkit-bootstrap-v1"),
    )
    args = parser.parse_args()
    evidence = run_cuda_toolkit_smoke(
        toolkit_root=args.toolkit_root,
        architecture=args.architecture,
    )
    write_cuda_toolkit_artifacts(evidence, args.output_dir)
    print(
        f"nvcc {evidence['nvcc_version']} compiled "
        f"{evidence['ptx_bytes']} PTX bytes for {evidence['compile_target']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
