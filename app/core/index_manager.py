from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time
from typing import Callable, Sequence

import numpy as np

from app.core.bm25 import BM25Index
from app.core.dense import (
    DenseIndex,
    DenseIndexStats,
    DenseSearchResponse,
    DenseSettings,
    DenseStatus,
    DenseUnavailableError,
    Embedder,
    SentenceTransformerEmbedder,
    select_device,
)
from app.core.hybrid import HybridRetriever
from app.models import Document
from app.storage.repositories import DocumentRepository, StoredChunk


class RebuildInProgressError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    version: int
    documents: tuple[Document, ...]
    bm25: BM25Index
    dense: CachedDenseRetriever
    hybrid: HybridRetriever
    created_at_ms: float


class CachedDenseRetriever:
    """Lazy exact retriever backed by content-addressed SQLite embeddings."""

    def __init__(
        self,
        documents: Sequence[Document],
        chunks: Sequence[StoredChunk],
        repository: DocumentRepository,
        settings: DenseSettings,
        embedder_provider: Callable[[], Embedder],
    ) -> None:
        self._documents = list(documents)
        self._chunks = list(chunks)
        self._repository = repository
        self.settings = settings
        self._embedder_provider = embedder_provider
        self._index: DenseIndex | None = None
        self._status = DenseStatus.NOT_INITIALIZED
        self._error: str | None = None
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = Lock()

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    def prepare(self) -> None:
        self._ensure_index()

    def search(self, query: str, top_k: int) -> DenseSearchResponse:
        if not self._documents:
            return DenseSearchResponse(
                query=query,
                model_name=self.settings.model_name,
                device=_safe_device(self.settings.device),
                vector_dimension=0,
                total_hits=0,
                elapsed_ms=0.0,
                index_version=1,
                hits=[],
            )
        return self._ensure_index().search(query, top_k)

    def stats(self) -> DenseIndexStats:
        if self._index is not None:
            return self._index.stats()
        return DenseIndexStats(
            status=self._status,
            model_name=self.settings.model_name,
            device=_safe_device(self.settings.device),
            vector_dimension=0,
            documents=len(self._documents),
            index_version=0,
            build_time_ms=0.0,
            error=self._error,
        )

    def _ensure_index(self) -> DenseIndex:
        if self._index is not None:
            return self._index
        if self._status == DenseStatus.ERROR:
            raise DenseUnavailableError(
                self._error or "Dense index initialization failed."
            )
        with self._lock:
            if self._index is not None:
                return self._index
            self._status = DenseStatus.LOADING
            self._error = None
            try:
                embedder = self._embedder_provider()
                cached = self._repository.get_embeddings(
                    self._chunks,
                    model_name=embedder.model_name,
                    normalized=True,
                )
                missing = [
                    chunk
                    for chunk in self._chunks
                    if chunk.chunk_id not in cached
                ]
                self._cache_hits = len(self._chunks) - len(missing)
                self._cache_misses = len(missing)
                if missing:
                    texts = [
                        f"{chunk.title}\n{chunk.text}"
                        for chunk in missing
                    ]
                    vectors = np.asarray(
                        embedder.encode(texts),
                        dtype=np.float32,
                    )
                    if (
                        vectors.ndim != 2
                        or vectors.shape[0] != len(missing)
                    ):
                        raise ValueError(
                            "Embedder returned an invalid batch shape."
                        )
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    if np.any(norms == 0):
                        raise ValueError(
                            "Embedding vectors must not be zero vectors."
                        )
                    vectors = np.asarray(vectors / norms, dtype=np.float32)
                    self._repository.upsert_embeddings(
                        (
                            chunk.chunk_id,
                            embedder.model_name,
                            True,
                            chunk.content_hash,
                            vector,
                        )
                        for chunk, vector in zip(
                            missing,
                            vectors,
                            strict=True,
                        )
                    )
                    cached.update(
                        {
                            chunk.chunk_id: vector
                            for chunk, vector in zip(
                                missing,
                                vectors,
                                strict=True,
                            )
                        }
                    )
                matrix = (
                    np.stack(
                        [cached[chunk.chunk_id] for chunk in self._chunks]
                    ).astype(np.float32)
                    if self._chunks
                    else np.empty((0, 0), dtype=np.float32)
                )
                index = DenseIndex(embedder)
                index.build_from_vectors(self._documents, matrix)
            except Exception as error:
                self._status = DenseStatus.ERROR
                self._error = str(error)
                if isinstance(error, DenseUnavailableError):
                    raise
                raise DenseUnavailableError(str(error)) from error
            self._index = index
            self._status = DenseStatus.READY
            return index


class IndexManager:
    def __init__(
        self,
        repository: DocumentRepository,
        settings: DenseSettings,
        *,
        embedder_factory: Callable[[DenseSettings], Embedder] = (
            SentenceTransformerEmbedder
        ),
        reranker_available: bool = True,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self._embedder_factory = embedder_factory
        self._embedder: Embedder | None = None
        self._embedder_lock = Lock()
        self._rebuild_lock = Lock()
        self._snapshot_lock = Lock()
        self._snapshot: IndexSnapshot | None = None
        self._rebuild_in_progress = False
        self._last_build_time_ms = 0.0
        self._last_error: str | None = None
        self._reranker_available = reranker_available
        self.rebuild(eager_dense=False)

    def current(self) -> IndexSnapshot:
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("The search index has not been initialized.")
        return snapshot

    def rebuild(self, *, eager_dense: bool = False) -> int:
        if not self._rebuild_lock.acquire(blocking=False):
            raise RebuildInProgressError(
                "Another index rebuild is already in progress."
            )
        self._rebuild_in_progress = True
        started_at = time.perf_counter()
        try:
            indexed = self.repository.list_index_chunks()
            chunks = [item[0] for item in indexed]
            documents = [
                chunk.to_search_document(filename)
                for chunk, filename in indexed
            ]
            bm25 = BM25Index()
            bm25.build(documents)
            dense = CachedDenseRetriever(
                documents,
                chunks,
                self.repository,
                self.settings,
                self._get_embedder,
            )
            if eager_dense and documents:
                dense.prepare()
            hybrid = HybridRetriever(bm25, dense)
            version = self.repository.current_index_version() + 1
            snapshot = IndexSnapshot(
                version=version,
                documents=tuple(documents),
                bm25=bm25,
                dense=dense,
                hybrid=hybrid,
                created_at_ms=time.time() * 1000,
            )
            with self._snapshot_lock:
                self.repository.set_index_version(version)
                self._snapshot = snapshot
            self._last_error = None
            return version
        except Exception as error:
            self._last_error = str(error)
            raise
        finally:
            self._last_build_time_ms = (
                time.perf_counter() - started_at
            ) * 1000
            self._rebuild_in_progress = False
            self._rebuild_lock.release()

    def status(self) -> dict[str, object]:
        snapshot = self.current()
        counts = self.repository.counts()
        dense_stats = snapshot.dense.stats()
        return {
            "current_version": snapshot.version,
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "bm25_ready": True,
            "dense_ready": dense_stats.status == DenseStatus.READY,
            "reranker_ready": self._reranker_available,
            "rebuild_in_progress": self._rebuild_in_progress,
            "last_build_time_ms": self._last_build_time_ms,
            "last_error": self._last_error,
            "embedding_model": dense_stats.model_name,
            "embedding_device": dense_stats.device,
            "embedding_dimension": dense_stats.vector_dimension,
        }

    def _get_embedder(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = self._embedder_factory(self.settings)
            return self._embedder


def _safe_device(requested: str) -> str:
    try:
        return select_device(requested)
    except (DenseUnavailableError, ValueError):
        return requested
