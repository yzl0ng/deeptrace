from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from app.agentic.nccl_matrix import discover_idle_devices
from app.agentic.tool_loop import (
    ACTION_SYSTEM_PROMPT,
    EvidenceSufficiencyAssessment,
    QuestionComplexityAssessment,
    ToolLoopAction,
    ToolLoopEvalCase,
    ToolLoopStep,
    adaptive_evidence_advice,
    run_tool_loop,
    summarize_tool_loop_results,
)

def _strip_code_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped


class TransformersToolLoopPolicy:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int,
        max_steps: int,
        schema_retries: int,
        min_read_evidence: int,
        require_evaluate_before_answer: bool,
        adaptive_soft_evidence_gate: bool,
        complexity_aware_soft_evidence_gate: bool,
        evidence_sufficiency_soft_gate: bool,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_steps = max_steps
        self.schema_retries = schema_retries
        self.min_read_evidence = min_read_evidence
        self.require_evaluate_before_answer = (
            require_evaluate_before_answer
        )
        self.adaptive_soft_evidence_gate = adaptive_soft_evidence_gate
        self.complexity_aware_soft_evidence_gate = (
            complexity_aware_soft_evidence_gate
        )
        self.evidence_sufficiency_soft_gate = (
            evidence_sufficiency_soft_gate
        )
        self._complexity_cache: dict[str, QuestionComplexityAssessment] = {}
        self._sufficiency_cache: dict[
            tuple[str, tuple[str, ...]],
            EvidenceSufficiencyAssessment,
        ] = {}

    def _generate_json(
        self,
        *,
        system_content: str,
        user_content: str,
        max_new_tokens: int,
        schema: type[Any],
    ) -> Any:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return schema.model_validate_json(_strip_code_fence(text))

    def _classify_question(
        self,
        question: str,
    ) -> QuestionComplexityAssessment:
        cached = self._complexity_cache.get(question)
        if cached is not None:
            return cached
        try:
            assessment = self._generate_json(
                system_content=(
                    "Classify research-question evidence complexity. Return "
                    "exactly one JSON object and no markdown with schema "
                    '{"complexity":"simple_fact|multi_hop|comparison|'
                    'ambiguous","rationale_summary":string}. simple_fact '
                    "asks one attribute of a subject explicitly named in the "
                    "question, for example the birth date of a named person. "
                    "multi_hop applies whenever the answer subject must first "
                    "be identified through a relation such as director of a "
                    "film, performer of a song, leader of a group, parent of "
                    "a person, winner of an event, or organization behind an "
                    "award, and then another attribute must be found. "
                    "comparison requires facts about two or more subjects. "
                    "Classify by the reasoning chain in the question, not by "
                    "whether a search result might contain the final answer "
                    "directly. Use ambiguous only when the wording does not "
                    "permit a reliable classification."
                ),
                user_content=f"Question: {question}",
                max_new_tokens=96,
                schema=QuestionComplexityAssessment,
            )
        except Exception as error:
            assessment = QuestionComplexityAssessment(
                complexity="ambiguous",
                rationale_summary=(
                    "Classifier output was invalid; use conservative "
                    f"budget-aware guidance ({type(error).__name__})."
                ),
            )
        self._complexity_cache[question] = assessment
        return assessment

    def complexity_assessment(
        self,
        question: str,
    ) -> QuestionComplexityAssessment | None:
        return self._complexity_cache.get(question)

    def _assess_evidence_sufficiency(
        self,
        *,
        question: str,
        history: Sequence[ToolLoopStep],
    ) -> EvidenceSufficiencyAssessment | None:
        read_items = [
            {
                "evidence_id": str(item.observation["evidence_id"]),
                "title": str(item.observation.get("title", "")),
                "content": str(item.observation.get("content", "")),
            }
            for item in history
            if item.status == "succeeded"
            and item.action.action == "read_page"
            and item.observation.get("evidence_id")
        ]
        unique_items = {
            item["evidence_id"]: item for item in read_items
        }
        if not unique_items:
            return None
        cache_key = (question, tuple(sorted(unique_items)))
        cached = self._sufficiency_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            assessment = self._generate_json(
                system_content=(
                    "Judge whether the currently read evidence is sufficient "
                    "to answer a research question with complete provenance. "
                    "Return exactly one JSON object and no markdown with "
                    'schema {"status":"sufficient|missing_link|conflicting|'
                    'uncertain","covered_information":[string],'
                    '"missing_information":[string],'
                    '"rationale_summary":string}. Evidence is sufficient only '
                    "when it explicitly supports every relationship needed "
                    "by the question and the final answer attribute. For a "
                    "nested question such as an attribute of the director of "
                    "a film, evidence must support both who directed the film "
                    "and that person's requested attribute. A passage that "
                    "contains the final attribute but does not prove the "
                    "intermediate relationship is missing_link. Comparisons "
                    "must support every compared subject. Do not use outside "
                    "knowledge, search snippets, or the expected answer."
                ),
                user_content=(
                    f"Question: {question}\n"
                    "Successfully read evidence: "
                    f"{json.dumps(list(unique_items.values()), ensure_ascii=False)}"
                ),
                max_new_tokens=256,
                schema=EvidenceSufficiencyAssessment,
            )
        except Exception as error:
            assessment = EvidenceSufficiencyAssessment(
                status="uncertain",
                missing_information=[
                    "Evidence checker did not return a valid assessment."
                ],
                rationale_summary=(
                    "Use conservative budget-aware evidence collection "
                    f"({type(error).__name__})."
                ),
            )
        self._sufficiency_cache[cache_key] = assessment
        return assessment

    def sufficiency_assessments(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "read_evidence_ids": list(read_ids),
                **assessment.model_dump(mode="json"),
            }
            for (cached_question, read_ids), assessment
            in self._sufficiency_cache.items()
            if cached_question == question
        ]

    def next_action(
        self,
        *,
        question: str,
        history: Sequence[ToolLoopStep],
    ) -> ToolLoopAction:
        import torch

        transcript = [
            {
                "step": item.step,
                "action": item.action.model_dump(mode="json"),
                "status": item.status,
                "observation": item.observation,
                "error_code": item.error_code,
            }
            for item in history
        ]
        discovered_ids = sorted(
            {
                str(result["evidence_id"])
                for item in history
                for result in item.observation.get("results", [])
                if isinstance(result, dict) and result.get("evidence_id")
            }
        )
        read_ids = sorted(
            {
                str(item.observation["evidence_id"])
                for item in history
                if item.status == "succeeded"
                and item.action.action == "read_page"
                and item.observation.get("evidence_id")
            }
        )
        evaluated_ids = sorted(
            {
                str(evidence_id)
                for item in history
                if item.status == "succeeded"
                and item.action.action == "evaluate_evidence"
                for evidence_id in item.observation.get(
                    "evaluated_evidence_ids", []
                )
            }
        )
        remaining_steps = self.max_steps - len(history)
        complexity_assessment = (
            self._classify_question(question)
            if self.complexity_aware_soft_evidence_gate
            else None
        )
        sufficiency_assessment = (
            self._assess_evidence_sufficiency(
                question=question,
                history=history,
            )
            if self.evidence_sufficiency_soft_gate
            else None
        )
        adaptive_advice = (
            adaptive_evidence_advice(
                read_evidence_count=len(read_ids),
                evaluated_evidence_count=len(evaluated_ids),
                remaining_steps=remaining_steps,
                complexity=(
                    complexity_assessment.complexity
                    if complexity_assessment is not None
                    else None
                ),
            )
            if (
                self.adaptive_soft_evidence_gate
                or self.complexity_aware_soft_evidence_gate
            )
            else None
        )
        user_content = (
            f"Question: {question}\n"
            f"Remaining actions including answer: {remaining_steps}\n"
            f"Evidence IDs returned by search: {discovered_ids}\n"
            f"Evidence IDs successfully read and allowed in answer: {read_ids}\n"
            f"Evidence IDs explicitly evaluated: {evaluated_ids}\n"
            f"Minimum distinct evidence required before answer: "
            f"{self.min_read_evidence}\n"
            f"evaluate_evidence required before answer: "
            f"{self.require_evaluate_before_answer}\n"
            "Executed history (environment observations are authoritative): "
            f"{json.dumps(transcript, ensure_ascii=False)}"
        )
        if adaptive_advice is not None:
            user_content += (
                "\nAdaptive soft evidence guidance (advisory, not a hard "
                "protocol requirement):\n"
                f"- Soft evidence target: "
                f"{adaptive_advice.soft_target_evidence}\n"
                f"- Evidence evaluation recommended now: "
                f"{adaptive_advice.evaluate_recommended}\n"
                f"- Prioritize answering now: "
                f"{adaptive_advice.prioritize_answer}\n"
                f"- Guidance: {adaptive_advice.message}\n"
                "Evidence relevance and answer correctness take priority over "
                "hitting the soft count. Never invent or cite an unread ID."
            )
        if complexity_assessment is not None:
            user_content += (
                "\nQuestion complexity assessment:\n"
                f"- Class: {complexity_assessment.complexity}\n"
                f"- Reason: {complexity_assessment.rationale_summary}"
            )
        if sufficiency_assessment is not None:
            if remaining_steps <= 2:
                sufficiency_next_action = (
                    "Prioritize a grounded answer using successfully read "
                    "evidence; do not invent IDs."
                )
            elif sufficiency_assessment.status == "sufficient":
                sufficiency_next_action = (
                    "The preferred next action is answer."
                )
            elif sufficiency_assessment.status == "conflicting":
                sufficiency_next_action = (
                    "The preferred next action is evaluate_evidence using "
                    "the relevant successfully read IDs."
                )
            else:
                sufficiency_next_action = (
                    "The preferred next action is read_page for a relevant "
                    "discovered ID not already in the successfully read list. "
                    "Do not repeat a read."
                )
            user_content += (
                "\nEvidence sufficiency checker (advisory, not a hard "
                "protocol requirement):\n"
                f"- Status: {sufficiency_assessment.status}\n"
                "- Covered: "
                f"{sufficiency_assessment.covered_information}\n"
                "- Missing: "
                f"{sufficiency_assessment.missing_information}\n"
                f"- Reason: {sufficiency_assessment.rationale_summary}\n"
                f"- Controller advice: {sufficiency_next_action}"
            )
        if remaining_steps <= 1:
            user_content += (
                "\nThis is the final allowed action. Return answer using only "
                "successfully read evidence IDs."
            )
        invalid_text = ""
        invalid_error = ""
        for attempt in range(self.schema_retries + 1):
            repair = ""
            if attempt:
                repair = (
                    "\nYour previous output was invalid. Return a corrected "
                    "single JSON object only.\n"
                    f"Validation error: {invalid_error}\n"
                    f"Invalid output: {invalid_text[:1000]}"
                )
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content + repair},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            invalid_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            try:
                return ToolLoopAction.model_validate_json(
                    _strip_code_fence(invalid_text)
                )
            except Exception as error:
                invalid_error = str(error)
        raise ValueError(invalid_error)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Base and LoRA through a real fixed-evidence tool loop."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-action-tokens", type=int, default=256)
    parser.add_argument("--schema-retries", type=int, default=1)
    parser.add_argument("--min-read-evidence", type=int, default=1)
    parser.add_argument(
        "--require-evaluate-before-answer",
        action="store_true",
    )
    parser.add_argument(
        "--adaptive-soft-evidence-gate",
        action="store_true",
        help=(
            "Advise a second relevant read and selective evidence evaluation "
            "while keeping the protocol hard minimum at one read."
        ),
    )
    parser.add_argument(
        "--complexity-aware-soft-evidence-gate",
        action="store_true",
        help=(
            "Classify each question as simple, multi-hop, comparison, or "
            "ambiguous and provide budget-aware evidence guidance."
        ),
    )
    parser.add_argument(
        "--evidence-sufficiency-soft-gate",
        action="store_true",
        help=(
            "Assess whether successfully read evidence covers every required "
            "reasoning link before advising answer or another read."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("base", "sft"),
        default=("base", "sft"),
    )
    parser.add_argument("--max-used-memory-mib", type=int, default=2048)
    parser.add_argument("--allow-compute-processes", action="store_true")
    args = parser.parse_args()
    soft_gate_count = sum(
        (
            args.adaptive_soft_evidence_gate,
            args.complexity_aware_soft_evidence_gate,
            args.evidence_sufficiency_soft_gate,
        )
    )
    if soft_gate_count > 1:
        parser.error("choose only one adaptive soft evidence gate")
    if soft_gate_count and (
        args.min_read_evidence != 1
        or args.require_evaluate_before_answer
    ):
        parser.error(
            "adaptive soft gate requires --min-read-evidence=1 and cannot "
            "be combined with --require-evaluate-before-answer"
        )

    idle_devices = discover_idle_devices(
        max_used_memory_mib=args.max_used_memory_mib,
        require_no_compute_processes=not args.allow_compute_processes,
    )
    if not idle_devices:
        parser.error("no GPU passed the dynamic preflight")
    physical_device = idle_devices[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = [
        ToolLoopEvalCase.model_validate(json.loads(line))
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        parser.error("tool-loop cases are empty")

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
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    def evaluate_mode(mode: str, active_model: Any) -> None:
        policy = TransformersToolLoopPolicy(
            model=active_model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_action_tokens,
            max_steps=args.max_steps,
            schema_retries=args.schema_retries,
            min_read_evidence=args.min_read_evidence,
            require_evaluate_before_answer=(
                args.require_evaluate_before_answer
            ),
            adaptive_soft_evidence_gate=(
                args.adaptive_soft_evidence_gate
            ),
            complexity_aware_soft_evidence_gate=(
                args.complexity_aware_soft_evidence_gate
            ),
            evidence_sufficiency_soft_gate=(
                args.evidence_sufficiency_soft_gate
            ),
        )
        results = []
        for case in cases:
            result = run_tool_loop(
                case,
                policy,
                max_steps=args.max_steps,
                min_read_evidence=args.min_read_evidence,
                require_evaluate_before_answer=(
                    args.require_evaluate_before_answer
                ),
            )
            results.append(result)
            rows.append(
                {
                    "mode": mode,
                    "question_complexity": (
                        policy.complexity_assessment(
                            case.question
                        ).model_dump(mode="json")
                        if policy.complexity_assessment(case.question)
                        is not None
                        else None
                    ),
                    "evidence_sufficiency_assessments": (
                        policy.sufficiency_assessments(case.question)
                    ),
                    **result.model_dump(mode="json"),
                }
            )
            print(
                f"{mode} {case.case_id}: completed={result.completed} "
                f"exact={result.answer_exact} steps={len(result.steps)} "
                f"stop={result.stop_reason}",
                flush=True,
            )
        summaries[mode] = summarize_tool_loop_results(results)
        complexity_counts: dict[str, int] = {}
        for case in cases:
            assessment = policy.complexity_assessment(case.question)
            if assessment is None:
                continue
            complexity_counts[assessment.complexity] = (
                complexity_counts.get(assessment.complexity, 0) + 1
            )
        if complexity_counts:
            summaries[mode]["question_complexity_counts"] = (
                complexity_counts
            )
        final_sufficiency_counts: dict[str, int] = {}
        for case in cases:
            assessments = policy.sufficiency_assessments(case.question)
            if not assessments:
                continue
            status = str(assessments[-1]["status"])
            final_sufficiency_counts[status] = (
                final_sufficiency_counts.get(status, 0) + 1
            )
        if final_sufficiency_counts:
            summaries[mode]["final_evidence_sufficiency_counts"] = (
                final_sufficiency_counts
            )

    if "base" in args.modes:
        evaluate_mode("base", model)
    if "sft" in args.modes:
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            local_files_only=True,
            is_trainable=False,
        )
        model.eval()
        evaluate_mode("sft", model)

    comparison_gate = None
    if {"base", "sft"}.issubset(summaries):
        base_exact = summaries["base"]["metrics"]["answer_exact_rate"]
        sft_exact = summaries["sft"]["metrics"]["answer_exact_rate"]
        comparison_checks = {
            "sft_tool_loop_gate_passed": (
                summaries["sft"]["quality_gate"]["passed"]
            ),
            "sft_answer_exact_improves_by_5_points": (
                sft_exact >= base_exact + 0.05
            ),
        }
        comparison_gate = {
            "passed": all(comparison_checks.values()),
            "checks": comparison_checks,
            "decision": (
                "eligible_for_rl"
                if all(comparison_checks.values())
                else "hold_before_rl"
            ),
        }
    summary = {
        "evaluation_version": "base-sft-real-tool-loop-v1",
        "status": "succeeded",
        "physical_device": physical_device,
        "cases": len(cases),
        "max_steps": args.max_steps,
        "max_action_tokens": args.max_action_tokens,
        "schema_retries": args.schema_retries,
        "min_read_evidence": args.min_read_evidence,
        "require_evaluate_before_answer": (
            args.require_evaluate_before_answer
        ),
        "adaptive_soft_evidence_gate": (
            args.adaptive_soft_evidence_gate
        ),
        "complexity_aware_soft_evidence_gate": (
            args.complexity_aware_soft_evidence_gate
        ),
        "evidence_sufficiency_soft_gate": (
            args.evidence_sufficiency_soft_gate
        ),
        "evaluated_modes": list(args.modes),
        "modes": summaries,
        "comparison_gate": comparison_gate,
        "truth_boundary": (
            "Actions were executed against frozen dev evidence with strict "
            "ID visibility rules. This is tuning-only dev, not final test."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "evaluation_version": summary["evaluation_version"],
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (predictions, summary_path)
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
