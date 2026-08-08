from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def run_cuda_extension_smoke() -> dict[str, Any]:
    try:
        import flash_attn
        import torch
        import vllm
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise RuntimeError(
            "the pinned torch, vLLM, and FlashAttention packages must be "
            "installed before running the GPU runtime smoke"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")

    started = time.perf_counter()
    torch.cuda.set_device(0)
    left = torch.arange(256, device="cuda", dtype=torch.float32).reshape(
        16, 16
    )
    product = left @ left.T
    expected_product = left.cpu() @ left.cpu().T
    if not torch.allclose(
        product.cpu(),
        expected_product,
        rtol=1e-4,
        atol=1e-3,
    ):
        raise RuntimeError("CUDA matmul does not match the CPU reference")
    observed = float(product.sum().item())

    query = torch.randn(
        1,
        32,
        4,
        64,
        device="cuda",
        dtype=torch.float16,
    )
    attention = flash_attn_func(
        query,
        query,
        query,
        dropout_p=0.0,
        causal=True,
    )
    if attention.shape != query.shape or not torch.isfinite(attention).all():
        raise RuntimeError("FlashAttention returned an invalid tensor")
    torch.cuda.synchronize()

    return {
        "status": "succeeded",
        "mode": "cuda_extensions",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": vllm.__version__,
        "flash_attn_version": flash_attn.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "cuda_matmul_sum": observed,
        "flash_attention_shape": list(attention.shape),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def run_vllm_inference_smoke(
    *,
    model_path: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "the pinned torch and vLLM packages must be installed before "
            "running the inference smoke"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    if not Path(model_path).is_dir():
        raise RuntimeError(f"model path is not a directory: {model_path}")

    started = time.perf_counter()
    engine = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=False,
        enforce_eager=True,
    )
    outputs = engine.generate(
        ["Reply with exactly: READY"],
        SamplingParams(temperature=0.0, max_tokens=8),
        use_tqdm=False,
    )
    generated = outputs[0].outputs[0].text.strip()
    if not generated:
        raise RuntimeError("vLLM returned an empty completion")
    torch.cuda.synchronize()

    return {
        "status": "succeeded",
        "mode": "vllm_inference",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": vllm.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "generated_text": generated,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute one reserved single-GPU Phase 0 runtime check."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["cuda_extensions", "vllm_inference"],
    )
    parser.add_argument("--model-path", default="")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "cuda_extensions":
        payload = run_cuda_extension_smoke()
    else:
        payload = run_vllm_inference_smoke(
            model_path=args.model_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
