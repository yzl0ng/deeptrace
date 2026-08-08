from __future__ import annotations

import hashlib
import json
import platform
import threading
import time
from pathlib import Path
from typing import Any

from app.agentic.models import ModelOutput, ModelUsage, ResearchBudget
from app.agentic.repository import AgenticRunRepository
from app.agentic.supervisor import SupervisorResearchService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "experiments" / "supervisor-recovery-v1"
)
DATABASE_PATH = OUTPUT_DIR / "runs.db"
QUERY = "Why should BM25 and dense retrieval be combined with RRF?"


class ScriptedSupervisorModel:
    model_name = "scripted-supervisor-v1"

    def invoke(self, stage: str, payload: dict[str, Any]) -> ModelOutput:
        if stage == "scope":
            data = {
                "needs_clarification": False,
                "clarification_question": None,
                "normalized_query": payload["query"],
            }
        elif stage == "brief":
            data = {
                "research_brief": (
                    "Compare lexical and semantic retrieval, then explain "
                    "why rank fusion is appropriate."
                )
            }
        elif stage == "plan":
            data = {
                "subtasks": [
                    "Explain the BM25 scoring space.",
                    "Explain dense retrieval and RRF.",
                ]
            }
        elif stage == "evidence_grade":
            observations = payload["observations"]
            needs_rewrite = (
                "BM25" in str(payload["question"])
                and len(observations) == 1
            )
            data = {
                "sufficient": not needs_rewrite,
                "reason": (
                    "The initial result lacks BM25 normalization details."
                    if needs_rewrite
                    else "The evidence directly covers the subtask."
                ),
                "missing_questions": (
                    ["Which components define the BM25 score?"]
                    if needs_rewrite
                    else []
                ),
            }
        elif stage == "query_rewrite":
            data = {
                "rewritten_query": (
                    "BM25 term frequency IDF length normalization"
                ),
                "reason": "Target the missing score components.",
            }
        elif stage == "contradictions":
            identifiers = [
                identifier
                for item in payload["evidence"]
                for identifier in item["evidence_ids"]
            ]
            data = {
                "contradictions": [
                    {
                        "claim": (
                            "BM25 and cosine similarity share one score scale."
                        ),
                        "evidence_ids": identifiers[:2],
                        "severity": "material",
                        "explanation": (
                            "The evidence describes lexical and vector scores "
                            "with different semantics."
                        ),
                    }
                ]
            }
        elif stage == "memory_fold":
            data = {
                "summary": (
                    "BM25 ranks lexical matches; dense retrieval ranks "
                    "semantic similarity; RRF combines their rank positions "
                    "without assuming comparable raw scores."
                )
            }
        elif stage == "report":
            data = {
                "final_report": (
                    "BM25 and dense retrieval provide complementary signals. "
                    "RRF is suitable because it combines ranks instead of "
                    "adding incomparable raw scores."
                )
            }
        else:
            raise ValueError(stage)
        return ModelOutput(
            data=data,
            model=self.model_name,
            usage=ModelUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class SnapshotEvidenceTool:
    name = "snapshot_evidence"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"])
        with self._lock:
            self.queries.append(query)
            ordinal = len(self.queries)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
        return {
            "query": query,
            "evidence": [
                {
                    "evidence_id": f"snapshot-evidence-{ordinal}",
                    "source_id": f"snapshot-source-{ordinal}",
                    "content": _content_for(query),
                }
            ],
        }


class SimulatedProcessExit(BaseException):
    pass


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{DATABASE_PATH}{suffix}").unlink(missing_ok=True)

    repository = AgenticRunRepository(DATABASE_PATH)
    model = ScriptedSupervisorModel()
    tool = SnapshotEvidenceTool()
    interrupted_run_id: list[str] = []

    def crash_after_research(run, stage: str) -> None:
        if stage == "research":
            interrupted_run_id.append(run.run_id)
            raise SimulatedProcessExit()

    first_service = _service(
        model=model,
        tool=tool,
        repository=repository,
        checkpoint_hook=crash_after_research,
    )
    recovery_event: dict[str, Any]
    try:
        first_service.run(QUERY)
        raise AssertionError("the fixed interruption did not occur")
    except SimulatedProcessExit:
        checkpoint = repository.get(interrupted_run_id[0])
        if checkpoint is None:
            raise RuntimeError("interrupted checkpoint was not persisted")
        recovery_event = {
            "event": "simulated_process_exit",
            "run_id": checkpoint.run_id,
            "checkpoint_version": checkpoint.checkpoint.version,
            "checkpoint_stage": checkpoint.checkpoint.stage,
            "completed_stages": checkpoint.checkpoint.completed_stages,
            "tool_calls_before_resume": len(checkpoint.tool_calls),
        }
        _write_json(
            "interrupted-checkpoint.json",
            checkpoint.model_dump(mode="json"),
        )

    second_service = _service(
        model=model,
        tool=tool,
        repository=repository,
    )
    final = second_service.resume(interrupted_run_id[0])
    if final is None or final.status.value != "completed":
        raise RuntimeError("checkpoint resume did not complete")

    initial_queries = {
        "Explain the BM25 scoring space.",
        "Explain dense retrieval and RRF.",
    }
    repeated_initial_queries = sum(
        max(tool.queries.count(query) - 1, 0)
        for query in initial_queries
    )
    metrics = {
        "status": final.status.value,
        "subtasks": len(final.subtasks),
        "max_parallel_observed": tool.max_active,
        "configured_parallel_limit": (
            final.budget.max_parallel_research_units
        ),
        "tool_calls_before_resume": (
            recovery_event["tool_calls_before_resume"]
        ),
        "tool_calls_total": len(final.tool_calls),
        "repeated_completed_initial_queries": repeated_initial_queries,
        "query_rewrites": len(final.query_rewrites),
        "evidence_assessments": len(final.evidence_assessments),
        "insufficient_assessments": sum(
            not item.sufficient for item in final.evidence_assessments
        ),
        "contradictions": len(final.contradictions),
        "memory_evidence_ids": (
            len(final.memory.evidence_ids) if final.memory else 0
        ),
        "memory_tool_call_ids": (
            len(final.memory.tool_call_ids) if final.memory else 0
        ),
        "checkpoint_version": final.checkpoint.version,
        "agent_steps": final.usage.agent_steps,
        "search_calls": final.usage.search_calls,
        "total_tokens": final.usage.total_tokens,
    }
    config = {
        "query": QUERY,
        "workflow": "supervisor",
        "max_query_rewrites": 1,
        "max_parallel_research_units": 2,
        "max_search_calls": 6,
        "interruption_stage": "research",
        "provider": "deterministic_snapshot",
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "network_used": False,
        "model": model.model_name,
    }
    trajectories = [
        {
            "call_id": call.call_id,
            "tool_name": call.tool_name,
            "query": call.arguments["query"],
            "status": call.status,
            "evidence_ids": [
                item["evidence_id"]
                for item in call.observation.get("evidence", [])
            ],
        }
        for call in final.tool_calls
    ]

    _write_json("config.json", config)
    _write_json("environment.json", environment)
    _write_json("metrics.json", metrics)
    _write_json("final-run.json", final.model_dump(mode="json"))
    _write_jsonl("recovery-events.jsonl", [recovery_event])
    _write_jsonl("trajectories.jsonl", trajectories)
    _write_jsonl("failures.jsonl", [])
    _write_report(metrics)
    manifest = {
        "experiment": "supervisor-recovery-v1",
        "status": "completed",
        "truth_boundary": (
            "Deterministic scripted model and evidence tool; validates "
            "workflow semantics, not research answer quality."
        ),
        "artifacts": {},
    }
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.name.startswith("runs.db") or path.name == "manifest.json":
            continue
        manifest["artifacts"][path.name] = _sha256(path)
    _write_json("manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _service(
    *,
    model: ScriptedSupervisorModel,
    tool: SnapshotEvidenceTool,
    repository: AgenticRunRepository,
    checkpoint_hook=None,
) -> SupervisorResearchService:
    return SupervisorResearchService(
        model=model,
        tool=tool,
        repository=repository,
        default_budget=ResearchBudget(
            max_agent_steps=20,
            max_search_calls=6,
            max_page_reads=0,
            max_total_tokens=1000,
            max_parallel_research_units=2,
        ),
        max_rewrite_attempts=1,
        checkpoint_hook=checkpoint_hook,
    )


def _content_for(query: str) -> str:
    if "term frequency" in query:
        return (
            "BM25 uses term frequency, inverse document frequency, and "
            "document-length normalization."
        )
    if "BM25" in query:
        return "BM25 is a lexical ranking method."
    return (
        "Dense retrieval ranks vector similarity. RRF combines rank "
        "positions and does not add raw scores."
    )


def _write_json(name: str, payload: Any) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
    (OUTPUT_DIR / name).write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in rows
        ),
        encoding="utf-8",
    )


def _write_report(metrics: dict[str, Any]) -> None:
    report = f"""# Supervisor Recovery Baseline v1

Status: {metrics["status"]}

This deterministic experiment validates Phase 3 workflow semantics. It uses a
scripted model and snapshot evidence tool; it does not measure answer quality.

## Results

- Subtasks: {metrics["subtasks"]}
- Maximum parallel calls: {metrics["max_parallel_observed"]}
- Configured parallel limit: {metrics["configured_parallel_limit"]}
- Tool calls before interruption: {metrics["tool_calls_before_resume"]}
- Tool calls after resume: {metrics["tool_calls_total"]}
- Repeated completed initial queries:
  {metrics["repeated_completed_initial_queries"]}
- Query rewrites: {metrics["query_rewrites"]}
- Insufficient assessments: {metrics["insufficient_assessments"]}
- Contradictions: {metrics["contradictions"]}
- Memory evidence/tool references:
  {metrics["memory_evidence_ids"]}/{metrics["memory_tool_call_ids"]}
- Final checkpoint version: {metrics["checkpoint_version"]}

The process is deliberately interrupted after the parallel research checkpoint.
Resume starts at evidence grading, does not repeat the two completed initial
queries, rewrites the one insufficient BM25 query, records one contradiction,
folds memory while retaining every evidence/tool identifier, and completes the
report.
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
