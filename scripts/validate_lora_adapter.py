from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.agentic.nccl_matrix import discover_idle_devices


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a base model plus PEFT LoRA adapter on one idle GPU."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-used-memory-mib", type=int, default=64)
    parser.add_argument("--allow-compute-processes", action="store_true")
    args = parser.parse_args()

    idle_devices = discover_idle_devices(
        max_used_memory_mib=args.max_used_memory_mib,
        require_no_compute_processes=not args.allow_compute_processes,
    )
    if not idle_devices:
        parser.error("no idle GPU passed the dynamic preflight")
    physical_device = idle_devices[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        local_files_only=True,
        is_trainable=False,
    )
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Reply with one short sentence about reliable search.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if not output_text.strip():
        raise RuntimeError("adapter-loaded model produced empty output")

    payload = {
        "validation_version": "lora-adapter-load-v1",
        "status": "succeeded",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "physical_device": physical_device,
        "logical_device": 0,
        "base_model": args.base_model,
        "adapter": args.adapter.as_posix(),
        "active_adapters": list(model.active_adapters),
        "adapter_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        ),
        "generated_tokens": int(new_tokens.numel()),
        "generated_text_sha256": hashlib.sha256(
            output_text.encode("utf-8")
        ).hexdigest(),
        "truth_boundary": (
            "This proves that PEFT loaded the exported adapter with the pinned "
            "base model and completed bounded generation. It does not prove "
            "model-quality improvement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"validated adapter on physical GPU {physical_device}: "
        f"{payload['generated_tokens']} generated tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
