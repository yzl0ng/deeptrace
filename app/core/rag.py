from __future__ import annotations

import re
import time
from typing import Protocol

from app.core.context_builder import ContextBuilder
from app.core.llm import LLMClient
from app.core.reranker import Reranker
from app.models import (
    HybridSearchResponse,
    LLMPromptInfo,
    RAGAnswerResponse,
    RAGCitation,
    RAGContextInfo,
    RAGLatency,
    RAGModelInfo,
    RAGRetrievalInfo,
    RAGUsage,
    RerankingInfo,
)


CITATION_PATTERN = re.compile(
    r"\[((?:doc|chunk)-[A-Za-z0-9._:-]+)\]",
    re.IGNORECASE,
)
ABSTENTION_MARKER = "INSUFFICIENT_EVIDENCE:"


class HybridSearchProtocol(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        candidate_k: int | None = None,
        rank_constant: int | None = None,
    ) -> HybridSearchResponse: ...


class RAGService:
    def __init__(
        self,
        hybrid_retriever: HybridSearchProtocol,
        llm_client: LLMClient,
        context_builder: ContextBuilder,
        *,
        max_context_chars: int = 12000,
        reranker: Reranker | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.llm_client = llm_client
        self.context_builder = context_builder
        self.max_context_chars = max_context_chars
        self.reranker = reranker

    def answer(
        self,
        query: str,
        retrieval_top_k: int = 5,
        candidate_k: int = 20,
        rank_constant: int = 60,
    ) -> RAGAnswerResponse:
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        if self.reranker is None:
            retrieval = self.hybrid_retriever.search(
                query,
                retrieval_top_k,
                candidate_k=candidate_k,
                rank_constant=rank_constant,
            )
        else:
            retrieval = self.hybrid_retriever.search(
                query,
                candidate_k,
                candidate_k=candidate_k,
                rank_constant=rank_constant,
            )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        reranking: RerankingInfo | None = None
        rrf_ranks = {
            hit.document.id: hit.rrf_rank or hit.rank
            for hit in retrieval.hits
        }
        if self.reranker is not None:
            reranked = self.reranker.rerank(
                query,
                retrieval.hits,
                retrieval_top_k,
            )
            reranking = RerankingInfo(
                model=reranked.model,
                candidate_count=reranked.candidate_count,
                output_count=len(reranked.hits),
                elapsed_ms=reranked.elapsed_ms,
                traces=reranked.traces,
            )
            retrieval = retrieval.model_copy(
                update={"hits": reranked.hits}
            )

        context_started = time.perf_counter()
        built = self.context_builder.build(
            query, retrieval.hits, self.max_context_chars
        )
        context_build_ms = (time.perf_counter() - context_started) * 1000

        if not built.hits:
            return self._response(
                query=query,
                answer="检索上下文为空，无法基于知识库回答。",
                retrieval=retrieval,
                retrieval_top_k=retrieval_top_k,
                built_documents=0,
                context_characters=0,
                context_truncated=built.truncated,
                citations=[],
                invalid_citation_ids=[],
                abstained=True,
                abstention_reason="no_retrieval_context",
                retrieval_ms=retrieval_ms,
                context_build_ms=context_build_ms,
                generation_ms=0.0,
                total_started=total_started,
                usage=RAGUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                ),
                model_name=self.llm_client.model_name,
                prompt=None,
                reranking=reranking,
            )

        generation_started = time.perf_counter()
        generated = self.llm_client.generate(query=query, context=built.text)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        answer = generated.text
        abstained = answer.startswith(ABSTENTION_MARKER)
        if abstained:
            answer = answer[len(ABSTENTION_MARKER) :].lstrip()
            if not answer:
                answer = "当前检索证据不足，无法可靠回答。"

        valid_by_id = {hit.document.id: hit for hit in built.hits}
        seen: set[str] = set()
        valid_ids: list[str] = []
        invalid_ids: list[str] = []
        for match in CITATION_PATTERN.finditer(answer):
            citation_id = match.group(1)
            canonical_id = next(
                (
                    document_id
                    for document_id in valid_by_id
                    if document_id.lower() == citation_id.lower()
                ),
                citation_id,
            )
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            if canonical_id in valid_by_id:
                valid_ids.append(canonical_id)
            else:
                invalid_ids.append(citation_id)

        citations = [
            RAGCitation(
                citation_id=citation_id,
                document=valid_by_id[citation_id].document,
                rrf_rank=rrf_ranks.get(
                    citation_id,
                    valid_by_id[citation_id].rank,
                ),
                rerank_rank=(
                    valid_by_id[citation_id].rank
                    if reranking is not None
                    else None
                ),
            )
            for citation_id in valid_ids
        ]
        return self._response(
            query=query,
            answer=answer,
            retrieval=retrieval,
            retrieval_top_k=retrieval_top_k,
            built_documents=len(built.hits),
            context_characters=built.characters,
            context_truncated=built.truncated,
            citations=citations,
            invalid_citation_ids=invalid_ids,
            abstained=abstained,
            abstention_reason=(
                "insufficient_evidence" if abstained else None
            ),
            retrieval_ms=retrieval_ms,
            context_build_ms=context_build_ms,
            generation_ms=generation_ms,
            total_started=total_started,
            usage=RAGUsage(
                prompt_tokens=generated.usage.prompt_tokens,
                completion_tokens=generated.usage.completion_tokens,
                total_tokens=generated.usage.total_tokens,
            ),
            model_name=generated.model,
            prompt=LLMPromptInfo(
                system=generated.system_prompt,
                user=generated.user_prompt,
            ),
            reranking=reranking,
        )

    def _response(
        self,
        *,
        query: str,
        answer: str,
        retrieval: HybridSearchResponse,
        retrieval_top_k: int,
        built_documents: int,
        context_characters: int,
        context_truncated: bool,
        citations: list[RAGCitation],
        invalid_citation_ids: list[str],
        abstained: bool,
        abstention_reason: str | None,
        retrieval_ms: float,
        context_build_ms: float,
        generation_ms: float,
        total_started: float,
        usage: RAGUsage,
        model_name: str,
        prompt: LLMPromptInfo | None,
        reranking: RerankingInfo | None,
    ) -> RAGAnswerResponse:
        return RAGAnswerResponse(
            query=query,
            answer=answer,
            model=RAGModelInfo(
                provider=self.llm_client.provider,
                name=model_name,
            ),
            retrieval=RAGRetrievalInfo(
                method=(
                    "rrf_reranker"
                    if reranking is not None
                    else "rrf"
                ),
                top_k=retrieval_top_k,
                candidate_k=retrieval.candidate_k,
                rank_constant=retrieval.rank_constant,
                hits=retrieval.hits,
                reranking=reranking,
            ),
            context=RAGContextInfo(
                documents=built_documents,
                characters=context_characters,
                truncated=context_truncated,
            ),
            citations=citations,
            invalid_citation_ids=invalid_citation_ids,
            abstained=abstained,
            abstention_reason=abstention_reason,
            prompt=prompt,
            latency=RAGLatency(
                retrieval_ms=retrieval_ms,
                reranker_ms=(
                    reranking.elapsed_ms
                    if reranking is not None
                    else None
                ),
                context_build_ms=context_build_ms,
                generation_ms=generation_ms,
                total_ms=(time.perf_counter() - total_started) * 1000,
            ),
            usage=usage,
        )
