from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agentic.models import ModelOutput, ResearchBudget
from app.agentic.repository import AgenticRunRepository
from app.agentic.runtime import (
    DeepResearchService,
    DeepSeekResearchModel,
    LocalSearchTool,
)
from app.core.bm25 import BM25Index
from app.core.llm import DeepSeekClient, DeepSeekSettings
from app.corpus import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "experiments" / (
    "deep-research-baseline-v1"
)
FIXED_QUERY = (
    "BM25、Dense Retrieval 和 RRF 各自解决什么问题？"
    "为什么不能直接相加 BM25 分数与余弦相似度？"
)


class ScriptedResearchModel:
    model_name = "scripted-phase1-smoke"

    def invoke(self, stage: str, payload: dict[str, Any]) -> ModelOutput:
        outputs = {
            "scope": {
                "needs_clarification": False,
                "clarification_question": None,
                "normalized_query": FIXED_QUERY,
            },
            "brief": {
                "research_brief": (
                    "解释三种检索组件的职责，并说明异构分数的量纲问题。"
                )
            },
            "plan": {
                "subtasks": [
                    "BM25 解决什么问题",
                    "Dense Retrieval 解决什么问题",
                    "RRF 如何融合排名以及为何不直接相加异构分数",
                ]
            },
            "report": {
                "final_report": (
                    "脚本化 smoke 已完成真实本地检索；"
                    "此文本不构成真实 LLM baseline。"
                )
            },
        }
        return ModelOutput(data=outputs[stage], model=self.model_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and archive the fixed Phase-1 research baseline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scripted-model",
        action="store_true",
        help="Use a deterministic model for a no-network smoke test.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    search_index = BM25Index()
    search_index.build(
        load_jsonl(PROJECT_ROOT / "data" / "sample_documents.jsonl")
    )
    if args.scripted_model:
        model: Any = ScriptedResearchModel()
    else:
        model = DeepSeekResearchModel(
            DeepSeekClient(DeepSeekSettings.from_environment())
        )
    budget = ResearchBudget(
        max_wall_time_seconds=180,
        max_agent_steps=10,
        max_search_calls=6,
        max_total_tokens=20_000,
        max_parallel_research_units=3,
    )
    repository = AgenticRunRepository(output_dir / "runs.db")
    service = DeepResearchService(
        model=model,
        tool=LocalSearchTool(
            search_index,
            top_k=5,
            max_chars_per_hit=2000,
        ),
        repository=repository,
        default_budget=budget,
    )

    started = time.perf_counter()
    run = service.run(FIXED_QUERY)
    elapsed_ms = (time.perf_counter() - started) * 1000
    generated_at = datetime.now(UTC).isoformat()
    config = (
        "experiment_id: deep-research-baseline-v1\n"
        f"generated_at: \"{generated_at}\"\n"
        f"model: {run.model_name}\n"
        "tool: local_search\n"
        "corpus: demo-30\n"
        f"scripted_model: {str(args.scripted_model).lower()}\n"
        "budget:\n"
        f"  max_wall_time_seconds: {budget.max_wall_time_seconds}\n"
        f"  max_agent_steps: {budget.max_agent_steps}\n"
        f"  max_search_calls: {budget.max_search_calls}\n"
        f"  max_total_tokens: {budget.max_total_tokens}\n"
        "  max_parallel_research_units: "
        f"{budget.max_parallel_research_units}\n"
    )
    metrics = {
        "status": run.status.value,
        "elapsed_ms": round(elapsed_ms, 3),
        "agent_steps": run.usage.agent_steps,
        "search_calls": run.usage.search_calls,
        "prompt_tokens": run.usage.prompt_tokens,
        "completion_tokens": run.usage.completion_tokens,
        "total_tokens": run.usage.total_tokens,
        "subtasks": len(run.subtasks),
        "successful_tool_calls": sum(
            call.status == "succeeded" for call in run.tool_calls
        ),
        "failed_tool_calls": sum(
            call.status != "succeeded" for call in run.tool_calls
        ),
    }
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "deepseek_key_recorded": False,
        "network_model_call": not args.scripted_model,
    }
    report = _render_report(run.model_dump(mode="json"), metrics, args.scripted_model)
    files = {
        "config.yaml": config,
        "environment.json": _json(environment),
        "metrics.json": _json(metrics),
        "predictions.jsonl": _jsonl(
            {
                "run_id": run.run_id,
                "query": run.user_query,
                "status": run.status.value,
                "answer": run.final_report,
                "model": run.model_name,
            }
        ),
        "trajectories.jsonl": run.model_dump_json() + "\n",
        "failures.jsonl": "".join(
            _jsonl(error.model_dump(mode="json")) for error in run.errors
        ),
        "report.md": report,
    }
    for name, content in files.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    _write_manifest(output_dir, files, generated_at)
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "model": run.model_name,
                "output_dir": str(output_dir),
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    )
    return 0 if run.status.value == "completed" else 1


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _jsonl(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) + "\n"


def _write_manifest(
    output_dir: Path,
    files: dict[str, str],
    generated_at: str,
) -> None:
    artifacts = []
    for name in sorted(files):
        data = (output_dir / name).read_bytes()
        artifacts.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "experiment_id": "deep-research-baseline-v1",
        "generated_at": generated_at,
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        _json(manifest),
        encoding="utf-8",
    )


def _render_report(
    run: dict[str, Any],
    metrics: dict[str, Any],
    scripted_model: bool,
) -> str:
    model_kind = "scripted smoke model" if scripted_model else "real DeepSeek API"
    limitations = (
        "This is a deterministic plumbing smoke, not a model-quality result."
        if scripted_model
        else (
            "The corpus is the controlled demo-30 collection and citation "
            "support is not yet independently verified."
        )
    )
    return (
        "# Deep Research baseline v1\n\n"
        f"- Status: `{run['status']}`\n"
        f"- Model path: {model_kind}\n"
        "- Search tool: real local BM25 over demo-30\n"
        f"- Agent steps: {metrics['agent_steps']}\n"
        f"- Search calls: {metrics['search_calls']}\n"
        f"- Total tokens: {metrics['total_tokens']}\n"
        f"- Elapsed: {metrics['elapsed_ms']:.3f} ms\n\n"
        "## Final report\n\n"
        f"{run['final_report'] or 'No report was generated.'}\n\n"
        "## Truthfulness boundary\n\n"
        f"{limitations}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
