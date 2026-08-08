from __future__ import annotations

import json
import os
from pathlib import Path
import time

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from app.agentic.api import router as agentic_router
from app.core.dense import (
    DenseRuntime,
    DenseSettings,
    DenseUnavailableError,
)
from app.core.hybrid import HybridRetriever
from app.core.index_manager import IndexManager, RebuildInProgressError
from app.core.context_builder import ContextBuilder
from app.core.llm import (
    DeepSeekClient,
    DeepSeekSettings,
    LLMAuthenticationError,
    LLMBadResponseError,
    LLMError,
    LLMRateLimitedError,
    LLMTimeoutError,
    RAGNotConfiguredError,
)
from app.core.rag import RAGService
from app.core.reranker import (
    RerankerRuntime,
    RerankerSettings,
    RerankerUnavailableError,
)
from app.corpus import load_jsonl
from app.ingestion.models import (
    ChunkListResponse,
    ChunkSummary,
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentSummary,
    IndexStatusResponse,
    IngestionAccepted,
    IngestionJobResponse,
    RebuildResponse,
    ReindexRequest,
)
from app.ingestion.parsers import DocumentParseError, parse_document
from app.ingestion.security import (
    UploadSettings,
    UploadValidationError,
)
from app.ingestion.service import IngestionService
from app.models import (
    DenseIndexStats,
    DenseSearchResponse,
    HybridSearchResponse,
    LLMPromptInfo,
    PlainLLMAnswerRequest,
    PlainLLMAnswerResponse,
    PlainLLMLatency,
    RAGAnswerRequest,
    RAGAnswerResponse,
    RAGModelInfo,
    RAGUsage,
    SearchResponse,
)
from app.storage.database import Database
from app.storage.repositories import (
    DocumentRepository,
    ImmutableCorpusError,
    StoredDocument,
    StoredJob,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sample_documents.jsonl"
DEFAULT_EVALUATION_REPORT_PATH = (
    PROJECT_ROOT / "data" / "reports" / "retrieval_quality_baseline.json"
)
database_path = Path(
    os.getenv("SEARCHLAB_DATABASE_PATH", "data/searchlab.db")
)
if not database_path.is_absolute():
    database_path = PROJECT_ROOT / database_path
upload_settings = UploadSettings.from_environment(PROJECT_ROOT)

documents = load_jsonl(DEFAULT_CORPUS_PATH)
database = Database(database_path)
document_repository = DocumentRepository(database)
document_repository.seed_demo(documents)
index_manager = IndexManager(
    document_repository,
    DenseSettings.from_environment(),
)
reranker_runtime = RerankerRuntime(RerankerSettings.from_environment())
ingestion_service = IngestionService(
    document_repository,
    index_manager,
    upload_settings,
)

# Backward-compatible module aliases used by existing tests and scripts. API
# handlers resolve the current snapshot dynamically after every rebuild.
search_index = index_manager.current().bm25
dense_runtime = index_manager.current().dense
hybrid_retriever = index_manager.current().hybrid


def load_default_corpus() -> None:
    global search_index, dense_runtime, hybrid_retriever
    document_repository.seed_demo(documents)
    index_manager.rebuild(eager_dense=False)
    search_index = index_manager.current().bm25
    dense_runtime = index_manager.current().dense
    hybrid_retriever = index_manager.current().hybrid

app = FastAPI(
    title="SearchLab API",
    version="1.0.0",
    description="Explainable lexical, semantic and RAG experiments.",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "SEARCHLAB_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:3001,"
        "http://localhost:3002,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(agentic_router)


def get_index_manager() -> IndexManager:
    return index_manager


def get_document_repository() -> DocumentRepository:
    return document_repository


def get_ingestion_service() -> IngestionService:
    return ingestion_service


@app.get("/health")
def health(
    manager: IndexManager = Depends(get_index_manager),
) -> dict[str, str | int]:
    snapshot = manager.current()
    return {
        "status": "ok",
        "stage": "bm25-baseline",
        "engine": "python-bm25",
        "documents": int(snapshot.bm25.stats()["documents"]),
    }


@app.get("/api/v1/index/stats")
def index_stats(
    manager: IndexManager = Depends(get_index_manager),
) -> dict[str, float | int]:
    return manager.current().bm25.stats()


@app.get("/api/v1/evaluation/latest")
def latest_evaluation() -> dict[str, object]:
    if not DEFAULT_EVALUATION_REPORT_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "evaluation_report_not_ready",
                "message": "Run scripts/run_retrieval_evaluation.py first.",
            },
        )
    return json.loads(
        DEFAULT_EVALUATION_REPORT_PATH.read_text(encoding="utf-8")
    )


@app.get("/api/v1/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, max_length=500),
    top_k: int = Query(default=5, ge=1, le=50),
    manager: IndexManager = Depends(get_index_manager),
) -> SearchResponse:
    return manager.current().bm25.search(q, top_k=top_k)


def get_dense_runtime(
    manager: IndexManager = Depends(get_index_manager),
) -> DenseRuntime:
    return manager.current().dense


def get_hybrid_retriever(
    manager: IndexManager = Depends(get_index_manager),
) -> HybridRetriever:
    return manager.current().hybrid


def get_reranker_runtime() -> RerankerRuntime:
    return reranker_runtime


def get_llm_client() -> DeepSeekClient:
    try:
        settings = DeepSeekSettings.from_environment()
    except RAGNotConfiguredError as error:
        raise HTTPException(
            status_code=503,
            detail={"code": error.code, "message": str(error)},
        ) from error
    return DeepSeekClient(settings)


def get_rag_service(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    llm_client: DeepSeekClient = Depends(get_llm_client),
) -> RAGService:
    max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
    if max_context_chars < 1:
        raise ValueError("RAG_MAX_CONTEXT_CHARS must be at least 1")
    return RAGService(
        retriever,
        llm_client,
        ContextBuilder(),
        max_context_chars=max_context_chars,
    )


def get_reranked_rag_service(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: RerankerRuntime = Depends(get_reranker_runtime),
    llm_client: DeepSeekClient = Depends(get_llm_client),
) -> RAGService:
    max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
    if max_context_chars < 1:
        raise ValueError("RAG_MAX_CONTEXT_CHARS must be at least 1")
    return RAGService(
        retriever,
        llm_client,
        ContextBuilder(),
        max_context_chars=max_context_chars,
        reranker=reranker,
    )


@app.get("/api/v1/index/dense/stats", response_model=DenseIndexStats)
def dense_index_stats(
    runtime: DenseRuntime = Depends(get_dense_runtime),
) -> DenseIndexStats:
    return runtime.stats()


@app.get("/api/v1/search/dense", response_model=DenseSearchResponse)
def dense_search(
    q: str = Query(min_length=1, max_length=500),
    top_k: int = Query(default=5, ge=1, le=50),
    runtime: DenseRuntime = Depends(get_dense_runtime),
) -> DenseSearchResponse:
    try:
        return runtime.search(q, top_k)
    except DenseUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "dense_unavailable",
                "message": str(error),
                "status": runtime.stats().status,
            },
        ) from error


@app.get("/api/v1/search/hybrid", response_model=HybridSearchResponse)
def hybrid_search(
    q: str = Query(min_length=1, max_length=500),
    top_k: int = Query(default=5, ge=1, le=50),
    candidate_k: int = Query(default=20, ge=1, le=100),
    rank_constant: int = Query(default=60, ge=1, le=1000),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> HybridSearchResponse:
    if candidate_k < top_k:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_candidate_k",
                "message": "candidate_k must be greater than or equal to top_k",
            },
        )
    try:
        return retriever.search(
            q,
            top_k,
            candidate_k=candidate_k,
            rank_constant=rank_constant,
        )
    except DenseUnavailableError as error:
        message = str(error)
        lowered = message.lower()
        if "cuda" in lowered and "not available" in lowered:
            code = "cuda_unavailable"
        elif "load" in lowered or "model" in lowered:
            code = "model_load_failed"
        else:
            code = "dense_model_not_ready"
        raise HTTPException(
            status_code=503,
            detail={
                "code": code,
                "message": message,
            },
        ) from error


@app.post("/api/v1/rag/answer", response_model=RAGAnswerResponse)
def rag_answer(
    request: RAGAnswerRequest,
    service: RAGService = Depends(get_rag_service),
) -> RAGAnswerResponse:
    if request.candidate_k < request.retrieval_top_k:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_candidate_k",
                "message": (
                    "candidate_k must be greater than or equal to "
                    "retrieval_top_k"
                ),
            },
        )
    try:
        return service.answer(
            request.query,
            retrieval_top_k=request.retrieval_top_k,
            candidate_k=request.candidate_k,
            rank_constant=request.rank_constant,
        )
    except DenseUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "dense_model_not_ready",
                "message": str(error),
            },
        ) from error
    except LLMError as error:
        raise _llm_http_exception(error) from error


@app.post("/api/v1/rag/reranked-answer", response_model=RAGAnswerResponse)
def reranked_rag_answer(
    request: RAGAnswerRequest,
    service: RAGService = Depends(get_reranked_rag_service),
) -> RAGAnswerResponse:
    if request.candidate_k < request.retrieval_top_k:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_candidate_k",
                "message": (
                    "candidate_k must be greater than or equal to "
                    "retrieval_top_k"
                ),
            },
        )
    try:
        return service.answer(
            request.query,
            retrieval_top_k=request.retrieval_top_k,
            candidate_k=request.candidate_k,
            rank_constant=request.rank_constant,
        )
    except DenseUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "dense_model_not_ready",
                "message": str(error),
            },
        ) from error
    except RerankerUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "reranker_not_ready",
                "message": str(error),
            },
        ) from error
    except LLMError as error:
        raise _llm_http_exception(error) from error


@app.post("/api/v1/llm/answer", response_model=PlainLLMAnswerResponse)
def plain_llm_answer(
    request: PlainLLMAnswerRequest,
    llm_client: DeepSeekClient = Depends(get_llm_client),
) -> PlainLLMAnswerResponse:
    started_at = time.perf_counter()
    generation_started_at = time.perf_counter()
    try:
        generated = llm_client.generate_plain(query=request.query)
    except LLMError as error:
        raise _llm_http_exception(error) from error
    generation_ms = (time.perf_counter() - generation_started_at) * 1000
    return PlainLLMAnswerResponse(
        query=request.query,
        answer=generated.text,
        model=RAGModelInfo(
            provider=llm_client.provider,
            name=generated.model,
        ),
        retrieval_used=False,
        citations=[],
        prompt=LLMPromptInfo(
            system=generated.system_prompt,
            user=generated.user_prompt,
        ),
        latency=PlainLLMLatency(
            generation_ms=generation_ms,
            total_ms=(time.perf_counter() - started_at) * 1000,
        ),
        usage=RAGUsage(
            prompt_tokens=generated.usage.prompt_tokens,
            completion_tokens=generated.usage.completion_tokens,
            total_tokens=generated.usage.total_tokens,
        ),
    )


@app.post(
    "/api/v1/documents",
    response_model=IngestionAccepted,
    status_code=202,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    title: str | None = Form(default=None),
    source: str | None = Form(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionAccepted:
    content = await file.read(service.settings.max_upload_bytes + 1)
    try:
        prepared = service.prepare_upload(
            filename=file.filename,
            content_type=file.content_type,
            content=content,
            chunk_size=(
                chunk_size
                if chunk_size is not None
                else service.settings.default_chunk_size
            ),
            chunk_overlap=(
                chunk_overlap
                if chunk_overlap is not None
                else service.settings.default_chunk_overlap
            ),
            title=title,
            source=source,
        )
    except (UploadValidationError, DocumentParseError) as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": error.code,
                "message": str(error),
            },
        ) from error
    if not prepared.duplicate:
        background_tasks.add_task(service.process, prepared)
    return IngestionAccepted(
        document_id=prepared.document_id,
        job_id=prepared.job_id,
        status="completed" if prepared.duplicate else "pending",
        duplicate=prepared.duplicate,
    )


@app.get("/api/v1/documents", response_model=DocumentListResponse)
def list_documents(
    corpus_namespace: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentListResponse:
    if corpus_namespace not in {None, "demo", "uploaded", "evaluation"}:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_corpus_namespace",
                "message": "Unknown corpus namespace.",
            },
        )
    rows, total = repository.list_documents(
        corpus_namespace=corpus_namespace,
        status=status,
        limit=limit,
        offset=offset,
    )
    return DocumentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        documents=[_document_summary(row) for row in rows],
    )


@app.get("/api/v1/documents/{document_id}", response_model=DocumentSummary)
def get_document(
    document_id: str,
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentSummary:
    document = repository.get_document(document_id)
    if document is None:
        raise _not_found("document_not_found", "Document was not found.")
    return _document_summary(document)


@app.get(
    "/api/v1/documents/{document_id}/chunks",
    response_model=ChunkListResponse,
)
def get_document_chunks(
    document_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    repository: DocumentRepository = Depends(get_document_repository),
) -> ChunkListResponse:
    if repository.get_document(document_id) is None:
        raise _not_found("document_not_found", "Document was not found.")
    chunks, total = repository.list_chunks(
        document_id,
        limit=limit,
        offset=offset,
    )
    return ChunkListResponse(
        total=total,
        limit=limit,
        offset=offset,
        chunks=[
            ChunkSummary(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                corpus_namespace=chunk.corpus_namespace,
                chunk_index=chunk.chunk_index,
                title=chunk.title,
                section=chunk.section,
                page_number=chunk.page_number,
                text=chunk.text,
                content_hash=chunk.content_hash,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ],
    )


@app.get(
    "/api/v1/ingestions/{job_id}",
    response_model=IngestionJobResponse,
)
def get_ingestion_job(
    job_id: str,
    repository: DocumentRepository = Depends(get_document_repository),
) -> IngestionJobResponse:
    job = repository.get_job(job_id)
    if job is None:
        raise _not_found("ingestion_not_found", "Ingestion job was not found.")
    return _job_response(job)


@app.get("/api/v1/index/status", response_model=IndexStatusResponse)
def dynamic_index_status(
    manager: IndexManager = Depends(get_index_manager),
) -> IndexStatusResponse:
    return IndexStatusResponse.model_validate(manager.status())


@app.post("/api/v1/index/rebuild", response_model=RebuildResponse)
def rebuild_dynamic_index(
    manager: IndexManager = Depends(get_index_manager),
) -> RebuildResponse:
    try:
        version = manager.rebuild(eager_dense=True)
    except RebuildInProgressError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "rebuild_in_progress",
                "message": str(error),
            },
        ) from error
    except DenseUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "dense_model_not_ready",
                "message": str(error),
            },
        ) from error
    return RebuildResponse(status="completed", index_version=version)


@app.delete(
    "/api/v1/documents/{document_id}",
    response_model=DeleteDocumentResponse,
)
def delete_document(
    document_id: str,
    repository: DocumentRepository = Depends(get_document_repository),
    manager: IndexManager = Depends(get_index_manager),
) -> DeleteDocumentResponse:
    document = repository.get_document(document_id)
    if document is None:
        raise _not_found("document_not_found", "Document was not found.")
    if document.corpus_namespace != "uploaded":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "read_only_corpus",
                "message": "Demo and evaluation documents are read-only.",
            },
        )
    repository.update_document_status(document_id, "deleting")
    try:
        version = manager.rebuild(eager_dense=True)
    except Exception as error:
        repository.update_document_status(document_id, "completed")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "index_rebuild_failed",
                "message": "The active index was preserved.",
            },
        ) from error
    deleted = repository.delete_document(document_id)
    if deleted.stored_filename:
        stored_path = upload_settings.upload_dir / deleted.stored_filename
        stored_path.unlink(missing_ok=True)
    return DeleteDocumentResponse(
        document_id=document_id,
        status="deleted",
        index_version=version,
    )


@app.post(
    "/api/v1/documents/{document_id}/reindex",
    response_model=IngestionAccepted,
    status_code=202,
)
def reindex_document(
    document_id: str,
    request: ReindexRequest,
    background_tasks: BackgroundTasks,
    repository: DocumentRepository = Depends(get_document_repository),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionAccepted:
    document = repository.get_document(document_id)
    if document is None:
        raise _not_found("document_not_found", "Document was not found.")
    if document.corpus_namespace != "uploaded":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "read_only_corpus",
                "message": "Demo and evaluation documents are read-only.",
            },
        )
    if not document.stored_filename:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "stored_file_missing",
                "message": "The original uploaded file is unavailable.",
            },
        )
    path = upload_settings.upload_dir / document.stored_filename
    try:
        content = path.read_bytes()
        prepared = service.prepare_upload(
            filename=document.original_filename,
            content_type=document.mime_type,
            content=content,
            chunk_size=request.chunk_size or document.chunk_size,
            chunk_overlap=(
                request.chunk_overlap
                if request.chunk_overlap is not None
                else document.chunk_overlap
            ),
            source=document.source,
        )
    except OSError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "stored_file_missing",
                "message": "The original uploaded file is unavailable.",
            },
        ) from error
    except (UploadValidationError, DocumentParseError) as error:
        raise HTTPException(
            status_code=422,
            detail={"code": error.code, "message": str(error)},
        ) from error
    if not prepared.duplicate:
        background_tasks.add_task(service.process, prepared)
    return IngestionAccepted(
        document_id=document_id,
        job_id=prepared.job_id,
        status="completed" if prepared.duplicate else "pending",
        duplicate=prepared.duplicate,
    )


def _llm_http_exception(error: LLMError) -> HTTPException:
    status_codes = {
        RAGNotConfiguredError: 503,
        LLMTimeoutError: 504,
        LLMRateLimitedError: 429,
        LLMAuthenticationError: 502,
        LLMBadResponseError: 502,
    }
    return HTTPException(
        status_code=status_codes.get(type(error), 502),
        detail={
            "code": error.code,
            "message": str(error),
        },
    )


def _document_summary(document: StoredDocument) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.document_id,
        corpus_namespace=document.corpus_namespace,
        original_filename=document.original_filename,
        file_type=document.file_type,
        mime_type=document.mime_type,
        source=document.source,
        size_bytes=document.size_bytes,
        status=document.status,
        chunk_size=document.chunk_size,
        chunk_overlap=document.chunk_overlap,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
        error_code=document.error_code,
        error_message=document.error_message,
        index_version=document.index_version,
    )


def _job_response(job: StoredJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        status=job.status,
        current_stage=job.current_stage,
        progress=job.progress,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def _not_found(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": code, "message": message},
    )
