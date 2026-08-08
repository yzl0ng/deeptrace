from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from app.agentic.nccl_matrix import discover_idle_devices
from app.agentic.trajectory import (
    AGENT_SFT_SYSTEM_PROMPT,
    parse_sft_response,
)
ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = ARTICLES.sub(" ", value)
    value = NON_WORD.sub(" ", value)
    return " ".join(value.split())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Base and LoRA SFT tool-format validity."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
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
    model.eval()
    cases = _read_jsonl(args.test_file)
    predictions: list[dict[str, Any]] = []

    def evaluate_mode(mode: str, active_model: Any) -> None:
        for case in cases:
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": AGENT_SFT_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": str(case["question"])},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            started = time.perf_counter()
            with torch.inference_mode():
                generated = active_model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed = parse_sft_response(text)
            hit_token_limit = int(new_tokens.numel()) >= args.max_new_tokens
            format_failure = (
                None
                if parsed is not None
                else (
                    "max_new_tokens_reached"
                    if hit_token_limit
                    else "invalid_trajectory_schema"
                )
            )
            expected_answer = str(case.get("expected_answer", ""))
            answer_exact = (
                parsed is not None
                and bool(expected_answer)
                and _normalize_answer(parsed.final_answer)
                == _normalize_answer(expected_answer)
            )
            predictions.append(
                {
                    "case_id": case["case_id"],
                    "mode": mode,
                    "generated_tokens": int(new_tokens.numel()),
                    "latency_seconds": elapsed,
                    "format_valid": parsed is not None,
                    "hit_token_limit": hit_token_limit,
                    "format_failure": format_failure,
                    "expected_answer": expected_answer,
                    "answer_exact": answer_exact,
                    "actions": (
                        [step.action for step in parsed.steps]
                        if parsed is not None
                        else []
                    ),
                    "text": text,
                }
            )
            print(
                f"{mode} {case['case_id']}: "
                f"valid={parsed is not None} tokens={new_tokens.numel()}",
                flush=True,
            )

    evaluate_mode("base", model)
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        local_files_only=True,
        is_trainable=False,
    )
    model.eval()
    evaluate_mode("sft", model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in predictions
        ),
        encoding="utf-8",
    )
    metrics = {}
    for mode in ("base", "sft"):
        rows = [row for row in predictions if row["mode"] == mode]
        valid = sum(bool(row["format_valid"]) for row in rows)
        scored = [
            row for row in rows if bool(row["expected_answer"])
        ]
        exact = sum(bool(row["answer_exact"]) for row in scored)
        token_limited = sum(bool(row["hit_token_limit"]) for row in rows)
        metrics[mode] = {
            "cases": len(rows),
            "format_valid": valid,
            "format_valid_rate": valid / len(rows),
            "answer_scored_cases": len(scored),
            "answer_exact": exact,
            "answer_exact_rate": (
                exact / len(scored) if scored else None
            ),
            "token_limit_failures": token_limited,
            "token_limit_failure_rate": token_limited / len(rows),
            "mean_latency_seconds": sum(
                float(row["latency_seconds"]) for row in rows
            )
            / len(rows),
            "mean_generated_tokens": sum(
                int(row["generated_tokens"]) for row in rows
            )
            / len(rows),
        }
    base_metrics = metrics["base"]
    sft_metrics = metrics["sft"]
    quality_checks = {
        "format_valid_at_least_98_percent": (
            sft_metrics["format_valid_rate"] >= 0.98
        ),
        "answer_exact_improves_by_5_points": (
            sft_metrics["answer_exact_rate"]
            >= base_metrics["answer_exact_rate"] + 0.05
        ),
        "no_token_limit_failures": (
            sft_metrics["token_limit_failures"] == 0
        ),
    }
    summary = {
        "evaluation_version": "base-sft-tool-format-v2",
        "status": "succeeded",
        "physical_device": physical_device,
        "test_cases": len(cases),
        "max_new_tokens": args.max_new_tokens,
        "metrics": metrics,
        "quality_gate": {
            "passed": all(quality_checks.values()),
            "checks": quality_checks,
            "decision": (
                "eligible_for_tool_loop_evaluation"
                if all(quality_checks.values())
                else "hold_before_rl"
            ),
        },
        "truth_boundary": (
            "This fixed comparison measures JSON trajectory schema validity "
            "and normalized exact match against dev labels. It does not "
            "execute tools or establish evidence grounding."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "evaluation_version": summary["evaluation_version"],
        "status": summary["status"],
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (predictions_path, summary_path)
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
