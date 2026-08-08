from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    title: str
    content: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TermContribution(BaseModel):
    term: str
    term_frequency: float
    document_frequency: int
    inverse_document_frequency: float
    score: float


class SearchHit(BaseModel):
    rank: int
    score: float
    document: Document
    matched_terms: list[str]
    term_contributions: list[TermContribution]


class SearchResponse(BaseModel):
    query: str
    query_tokens: list[str]
    total_hits: int
    elapsed_ms: float
    index_version: int
    hits: list[SearchHit]


class DenseSearchHit(BaseModel):
    rank: int
    score: float
    document: Document


class DenseSearchResponse(BaseModel):
    query: str
    model_name: str
    device: str
    vector_dimension: int
    total_hits: int
    elapsed_ms: float
    index_version: int
    hits: list[DenseSearchHit]


class DenseIndexStats(BaseModel):
    status: str
    model_name: str
    device: str
    vector_dimension: int
    documents: int
    index_version: int
    build_time_ms: float
    error: str | None = None


class RetrievalLatency(BaseModel):
    bm25_ms: float
    dense_ms: float
    fusion_ms: float


class DenseModelInfo(BaseModel):
    name: str
    device: str
    dimension: int


class HybridSearchHit(BaseModel):
    rank: int
    rrf_score: float
    document: Document
    sources: list[str]
    source_ranks: dict[str, int]
    source_scores: dict[str, float]
    rrf_contributions: dict[str, float]
    rrf_rank: int | None = None
    reranker_score: float | None = None
    rank_delta: int | None = None


class HybridSearchResponse(BaseModel):
    query: str
    rank_constant: int
    candidate_k: int
    elapsed_ms: float
    retrieval_latency: RetrievalLatency
    dense_model: DenseModelInfo
    hits: list[HybridSearchHit]


class RAGAnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    rank_constant: int = Field(default=60, ge=1, le=1000)


class RerankerModelInfo(BaseModel):
    name: str
    device: str


class RerankTrace(BaseModel):
    document_id: str
    document_title: str
    rrf_rank: int
    rerank_rank: int
    rank_delta: int
    reranker_score: float


class RerankingInfo(BaseModel):
    model: RerankerModelInfo
    candidate_count: int
    output_count: int
    elapsed_ms: float
    traces: list[RerankTrace]


class RerankResult(BaseModel):
    query: str
    model: RerankerModelInfo
    candidate_count: int
    elapsed_ms: float
    hits: list[HybridSearchHit]
    traces: list[RerankTrace]


class PlainLLMAnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class RAGModelInfo(BaseModel):
    provider: str
    name: str


class RAGRetrievalInfo(BaseModel):
    method: str = "rrf"
    top_k: int
    candidate_k: int
    rank_constant: int
    hits: list[HybridSearchHit]
    reranking: RerankingInfo | None = None


class RAGContextInfo(BaseModel):
    documents: int
    characters: int
    truncated: bool


class RAGCitation(BaseModel):
    citation_id: str
    document: Document
    rrf_rank: int
    rerank_rank: int | None = None


class RAGLatency(BaseModel):
    retrieval_ms: float
    reranker_ms: float | None = None
    context_build_ms: float
    generation_ms: float
    total_ms: float


class RAGUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float | None = None
    currency: str | None = None


class PlainLLMLatency(BaseModel):
    generation_ms: float
    total_ms: float


class LLMPromptInfo(BaseModel):
    system: str
    user: str


class PlainLLMAnswerResponse(BaseModel):
    query: str
    answer: str
    model: RAGModelInfo
    retrieval_used: bool = False
    citations: list[RAGCitation] = Field(default_factory=list)
    prompt: LLMPromptInfo
    latency: PlainLLMLatency
    usage: RAGUsage


class RAGAnswerResponse(BaseModel):
    query: str
    answer: str
    model: RAGModelInfo
    retrieval: RAGRetrievalInfo
    context: RAGContextInfo
    citations: list[RAGCitation]
    invalid_citation_ids: list[str]
    abstained: bool
    abstention_reason: str | None
    prompt: LLMPromptInfo | None
    latency: RAGLatency
    usage: RAGUsage
