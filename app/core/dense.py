from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from threading import Lock
import time
from typing import Callable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from app.models import (
    DenseIndexStats,
    DenseSearchHit,
    DenseSearchResponse,
    Document,
)


FloatMatrix = NDArray[np.float32]


class DenseUnavailableError(RuntimeError):
    """Raised when the configured embedding model cannot serve requests."""


class Embedder(Protocol):
    """Minimal embedding contract used by the index and deterministic tests."""

    model_name: str
    device: str

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Return one unnormalized vector per input text."""


class DenseStatus(StrEnum):
    NOT_INITIALIZED = "not_initialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DenseSettings:
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"
    batch_size: int = 8

    @classmethod
    def from_environment(cls) -> DenseSettings:
        batch_size = int(os.getenv("DENSE_BATCH_SIZE", "8"))
        if batch_size < 1:
            raise ValueError("DENSE_BATCH_SIZE must be at least 1")
        return cls(
            model_name=os.getenv("DENSE_MODEL_NAME", "BAAI/bge-m3"),
            device=os.getenv("DENSE_DEVICE", "auto").lower(),
            batch_size=batch_size,
        )


def select_device(
    requested: str,
    *,
    cuda_available: bool | None = None,
) -> str:
    """Resolve auto/cpu/cuda without making unit tests depend on a GPU."""
    normalized = requested.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("DENSE_DEVICE must be one of: auto, cpu, cuda")

    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False

    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    if normalized == "cuda" and not cuda_available:
        raise DenseUnavailableError(
            "DENSE_DEVICE=cuda was requested, but CUDA is not available."
        )
    return normalized


class SentenceTransformerEmbedder:
    """Production embedder. The model object is constructed exactly once."""

    def __init__(self, settings: DenseSettings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise DenseUnavailableError(
                "Dense dependencies are not installed. Install the project with "
                "its dense dependencies before using this endpoint."
            ) from error

        self.model_name = settings.model_name
        self.device = select_device(settings.device)
        self.batch_size = settings.batch_size
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        except Exception as error:
            raise DenseUnavailableError(
                f"Unable to load embedding model {self.model_name!r} "
                f"on {self.device}: {error}"
            ) from error

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class DenseIndex:
    """In-memory exact vector index with cached normalized document vectors."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._documents: list[Document] = []
        self._document_vectors: FloatMatrix = np.empty((0, 0), dtype=np.float32)
        self._vector_dimension = 0
        self._index_version = 0
        self._build_time_ms = 0.0

    def build(self, documents: Sequence[Document]) -> None:
        texts = [
            f"{document.title}\n{document.content}"
            for document in documents
        ]
        vectors = (
            self.embedder.encode(texts)
            if texts
            else np.empty((0, 0), dtype=np.float32)
        )
        self.build_from_vectors(documents, vectors)

    def build_from_vectors(
        self,
        documents: Sequence[Document],
        vectors: FloatMatrix,
    ) -> None:
        """Build from cached vectors without re-embedding unchanged documents."""
        started_at = time.perf_counter()
        self._documents = list(documents)
        if not self._documents:
            self._document_vectors = np.empty((0, 0), dtype=np.float32)
            self._vector_dimension = 0
        else:
            matrix = _as_matrix(vectors, len(self._documents))
            self._document_vectors = _l2_normalize(matrix)
            self._vector_dimension = int(matrix.shape[1])
        self._index_version += 1
        self._build_time_ms = (time.perf_counter() - started_at) * 1000

    def search(self, query: str, top_k: int = 10) -> DenseSearchResponse:
        started_at = time.perf_counter()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not self._documents:
            return DenseSearchResponse(
                query=query,
                model_name=self.embedder.model_name,
                device=self.embedder.device,
                vector_dimension=self._vector_dimension,
                total_hits=0,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                index_version=self._index_version,
                hits=[],
            )

        query_matrix = _as_matrix(self.embedder.encode([query]), 1)
        if query_matrix.shape[1] != self._vector_dimension:
            raise DenseUnavailableError(
                "Query vector dimension does not match the document index."
            )
        query_vector = _l2_normalize(query_matrix)[0]
        similarities = self._document_vectors @ query_vector
        result_count = min(top_k, len(self._documents))
        ranked_indices = np.argsort(-similarities, kind="stable")[:result_count]
        hits = [
            DenseSearchHit(
                rank=rank,
                score=float(similarities[document_index]),
                document=self._documents[int(document_index)],
            )
            for rank, document_index in enumerate(ranked_indices, start=1)
        ]
        return DenseSearchResponse(
            query=query,
            model_name=self.embedder.model_name,
            device=self.embedder.device,
            vector_dimension=self._vector_dimension,
            total_hits=len(self._documents),
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            index_version=self._index_version,
            hits=hits,
        )

    def stats(self) -> DenseIndexStats:
        return DenseIndexStats(
            status=DenseStatus.READY,
            model_name=self.embedder.model_name,
            device=self.embedder.device,
            vector_dimension=self._vector_dimension,
            documents=len(self._documents),
            index_version=self._index_version,
            build_time_ms=self._build_time_ms,
            error=None,
        )


EmbedderFactory = Callable[[DenseSettings], Embedder]


class DenseRuntime:
    """Thread-safe lazy lifecycle for the single production model and index."""

    def __init__(
        self,
        documents: Sequence[Document],
        settings: DenseSettings,
        embedder_factory: EmbedderFactory = SentenceTransformerEmbedder,
    ) -> None:
        self._documents = list(documents)
        self.settings = settings
        self._embedder_factory = embedder_factory
        self._index: DenseIndex | None = None
        self._status = DenseStatus.NOT_INITIALIZED
        self._error: str | None = None
        self._lock = Lock()

    def search(self, query: str, top_k: int) -> DenseSearchResponse:
        return self._ensure_index().search(query, top_k)

    def stats(self) -> DenseIndexStats:
        if self._index is not None:
            return self._index.stats()
        try:
            device = select_device(self.settings.device)
        except (DenseUnavailableError, ValueError):
            device = self.settings.device
        return DenseIndexStats(
            status=self._status,
            model_name=self.settings.model_name,
            device=device,
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
                self._error or "Dense model initialization failed."
            )

        with self._lock:
            if self._index is not None:
                return self._index
            if self._status == DenseStatus.ERROR:
                raise DenseUnavailableError(
                    self._error or "Dense model initialization failed."
                )
            self._status = DenseStatus.LOADING
            self._error = None
            try:
                index = DenseIndex(self._embedder_factory(self.settings))
                index.build(self._documents)
            except Exception as error:
                self._status = DenseStatus.ERROR
                self._error = str(error)
                if isinstance(error, DenseUnavailableError):
                    raise
                raise DenseUnavailableError(str(error)) from error
            self._index = index
            self._status = DenseStatus.READY
            return index


def _as_matrix(vectors: FloatMatrix, expected_rows: int) -> FloatMatrix:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(
            "Embedder must return a two-dimensional matrix with one row per text."
        )
    if matrix.shape[1] == 0:
        raise ValueError("Embedding vectors must have at least one dimension.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must contain only finite values.")
    return matrix


def _l2_normalize(vectors: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero vectors.")
    return np.asarray(vectors / norms, dtype=np.float32)
