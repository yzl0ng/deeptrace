from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import time
from typing import Iterable

from app.core.tokenizer import tokenize
from app.models import Document, SearchHit, SearchResponse, TermContribution


@dataclass(slots=True)
class _IndexedDocument:
    document: Document
    term_frequencies: Counter[str]
    length: float


@dataclass(slots=True)
class BM25Index:
    """Small, readable BM25 implementation used as the lexical baseline."""

    k1: float = 1.5
    b: float = 0.75
    title_weight: float = 2.0
    _documents: list[_IndexedDocument] = field(default_factory=list)
    _document_frequency: Counter[str] = field(default_factory=Counter)
    _average_document_length: float = 0.0
    _version: int = 0

    def build(self, documents: Iterable[Document]) -> None:
        indexed_documents: list[_IndexedDocument] = []
        document_frequency: Counter[str] = Counter()

        for document in documents:
            title_terms = Counter(tokenize(document.title))
            content_terms = Counter(tokenize(document.content))
            weighted_terms = content_terms.copy()
            for term, frequency in title_terms.items():
                weighted_terms[term] += frequency * self.title_weight

            document_length = float(sum(weighted_terms.values()))
            indexed_documents.append(
                _IndexedDocument(
                    document=document,
                    term_frequencies=weighted_terms,
                    length=document_length,
                )
            )
            document_frequency.update(weighted_terms.keys())

        self._documents = indexed_documents
        self._document_frequency = document_frequency
        total_length = sum(item.length for item in indexed_documents)
        self._average_document_length = (
            total_length / len(indexed_documents) if indexed_documents else 0.0
        )
        self._version += 1

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        started_at = time.perf_counter()
        query_terms = tokenize(query)
        query_frequencies = Counter(query_terms)
        scored_documents: list[tuple[float, _IndexedDocument, list[TermContribution]]] = []

        for indexed_document in self._documents:
            score = 0.0
            contributions: list[TermContribution] = []
            for term, query_frequency in query_frequencies.items():
                term_frequency = indexed_document.term_frequencies.get(term, 0.0)
                if term_frequency <= 0:
                    continue

                inverse_document_frequency = self._idf(term)
                length_normalization = self.k1 * (
                    1
                    - self.b
                    + self.b
                    * indexed_document.length
                    / max(self._average_document_length, 1.0)
                )
                term_score = (
                    inverse_document_frequency
                    * (term_frequency * (self.k1 + 1))
                    / (term_frequency + length_normalization)
                    * query_frequency
                )
                score += term_score
                contributions.append(
                    TermContribution(
                        term=term,
                        term_frequency=term_frequency,
                        document_frequency=self._document_frequency[term],
                        inverse_document_frequency=inverse_document_frequency,
                        score=term_score,
                    )
                )

            if score > 0:
                contributions.sort(key=lambda item: item.score, reverse=True)
                scored_documents.append((score, indexed_document, contributions))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        hits = [
            SearchHit(
                rank=rank,
                score=score,
                document=indexed_document.document,
                matched_terms=[item.term for item in contributions],
                term_contributions=contributions,
            )
            for rank, (score, indexed_document, contributions) in enumerate(
                scored_documents[:top_k],
                start=1,
            )
        ]
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        return SearchResponse(
            query=query,
            query_tokens=query_terms,
            total_hits=len(scored_documents),
            elapsed_ms=elapsed_ms,
            index_version=self._version,
            hits=hits,
        )

    def stats(self) -> dict[str, float | int]:
        return {
            "documents": len(self._documents),
            "unique_terms": len(self._document_frequency),
            "average_document_length": round(self._average_document_length, 3),
            "index_version": self._version,
            "k1": self.k1,
            "b": self.b,
            "title_weight": self.title_weight,
        }

    def _idf(self, term: str) -> float:
        document_count = len(self._documents)
        document_frequency = self._document_frequency[term]
        return math.log(
            1
            + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
