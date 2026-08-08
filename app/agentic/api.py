from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.agentic.config import DeepTraceV2Config
from app.agentic.evidence import EvidenceRetriever, EvidenceStore
from app.agentic.models import (
    CreateResearchRunRequest,
    ResearchBudget,
    ResearchRun,
)
from app.agentic.repository import AgenticRunRepository
from app.agentic.runtime import (
    DeepResearchService,
    DeepSeekResearchModel,
    LocalSearchTool,
)
from app.agentic.supervisor import SupervisorResearchService
from app.agentic.web import (
    BraveSearchProvider,
    SafePageReader,
    WebEvidenceTool,
)
from app.core.bm25 import BM25Index
from app.core.llm import (
    DeepSeekClient,
    DeepSeekSettings,
    RAGNotConfiguredError,
)
from app.corpus import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sample_documents.jsonl"

router = APIRouter(prefix="/api/v2/research", tags=["deeptrace-v2"])


ResearchService = DeepResearchService | SupervisorResearchService


@router.get("/status")
def get_research_status() -> dict[str, object]:
    """Return non-secret runtime capabilities for the web control surface."""
    config = DeepTraceV2Config.from_env()
    model_configured = bool(os.getenv("DEEPSEEK_API_KEY", "").strip())
    web_search_configured = (
        config.search_provider != "brave"
        or bool(os.getenv("BRAVE_SEARCH_API_KEY", "").strip())
    )
    return {
        "api_version": "v2",
        "enabled": config.enabled,
        "ready": (
            config.enabled
            and model_configured
            and web_search_configured
        ),
        "workflow": config.workflow,
        "search_provider": config.search_provider,
        "model_configured": model_configured,
        "web_search_configured": web_search_configured,
        "supports_cancel_resume": config.workflow == "supervisor",
        "limits": {
            "max_wall_time_seconds": config.max_wall_time_seconds,
            "max_agent_steps": config.max_agent_steps,
            "max_search_calls": config.max_search_calls,
            "max_page_reads": config.max_page_reads,
            "max_total_tokens": config.max_total_tokens,
            "max_parallel_research_units": (
                config.max_parallel_research_units
            ),
        },
    }


def get_deep_research_service() -> ResearchService:
    config = DeepTraceV2Config.from_env()
    if not config.enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "deeptrace_v2_disabled",
                "message": "Set DEEPTRACE_ENABLED=true to enable the v2 API.",
            },
        )
    try:
        settings = DeepSeekSettings.from_environment()
    except RAGNotConfiguredError as error:
        raise HTTPException(
            status_code=503,
            detail={"code": error.code, "message": str(error)},
        ) from error

    database_path = Path(
        os.getenv("DEEPTRACE_DATABASE_PATH", "data/deeptrace-v2.db")
    )
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    repository = AgenticRunRepository(database_path)
    if config.search_provider == "brave":
        try:
            provider = BraveSearchProvider.from_environment()
        except ValueError as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "web_search_not_configured",
                    "message": str(error),
                },
            ) from error
        evidence_database_path = Path(
            os.getenv(
                "DEEPTRACE_EVIDENCE_DATABASE_PATH",
                "data/deeptrace-evidence-v2.db",
            )
        )
        if not evidence_database_path.is_absolute():
            evidence_database_path = PROJECT_ROOT / evidence_database_path
        evidence_store = EvidenceStore(evidence_database_path)
        tool = WebEvidenceTool(
            provider=provider,
            reader=SafePageReader(),
            store=evidence_store,
            retriever=EvidenceRetriever(evidence_store),
        )
    else:
        search_index = BM25Index()
        search_index.build(load_jsonl(DEFAULT_CORPUS_PATH))
        tool = LocalSearchTool(search_index)
    model = DeepSeekResearchModel(DeepSeekClient(settings))
    budget = ResearchBudget(
        max_wall_time_seconds=config.max_wall_time_seconds,
        max_agent_steps=config.max_agent_steps,
        max_search_calls=config.max_search_calls,
        max_page_reads=config.max_page_reads,
        max_total_tokens=config.max_total_tokens,
        max_parallel_research_units=config.max_parallel_research_units,
    )
    if config.workflow == "supervisor":
        return SupervisorResearchService(
            model=model,
            tool=tool,
            repository=repository,
            default_budget=budget,
            max_rewrite_attempts=config.max_query_rewrites,
            max_context_chars=config.max_context_tokens * 4,
        )
    return DeepResearchService(
        model=model,
        tool=tool,
        repository=repository,
        default_budget=budget,
    )


@router.post("/runs", response_model=ResearchRun)
def create_research_run(
    request: CreateResearchRunRequest,
    service: ResearchService = Depends(get_deep_research_service),
) -> ResearchRun:
    return service.run(request.query, budget=request.budget)


@router.get("/runs/{run_id}", response_model=ResearchRun)
def get_research_run(
    run_id: str,
    service: ResearchService = Depends(get_deep_research_service),
) -> ResearchRun:
    run = service.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "agentic_run_not_found",
                "message": "The research run was not found.",
            },
        )
    return run


@router.post("/runs/{run_id}/cancel", response_model=ResearchRun)
def cancel_research_run(
    run_id: str,
    service: ResearchService = Depends(get_deep_research_service),
) -> ResearchRun:
    if not isinstance(service, SupervisorResearchService):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "supervisor_workflow_required",
                "message": "Cancel requires DEEPTRACE_WORKFLOW=supervisor.",
            },
        )
    run = service.cancel(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "agentic_run_not_found"},
        )
    return run


@router.post("/runs/{run_id}/resume", response_model=ResearchRun)
def resume_research_run(
    run_id: str,
    service: ResearchService = Depends(get_deep_research_service),
) -> ResearchRun:
    if not isinstance(service, SupervisorResearchService):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "supervisor_workflow_required",
                "message": "Resume requires DEEPTRACE_WORKFLOW=supervisor.",
            },
        )
    run = service.resume(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "agentic_run_not_found"},
        )
    return run
