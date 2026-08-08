from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from app.core.dense import (
    DenseSettings,
    Embedder,
    SentenceTransformerEmbedder,
)
from app.models import DenseSearchHit, DenseSearchResponse, Document


FloatMatrix = NDArray[np.float32]
EmbedderFactory = Callable[[DenseSettings], Embedder]
CACHE_FORMAT_VERSION = 1


def embedding_cache_key(
    *,
    model_name: str,
    corpus_hash: str,
    normalized: bool = True,
) -> str:
    payload = json.dumps(
        {
            "format_version": CACHE_FORMAT_VERSION,
            "model_name": model_name,
            "corpus_hash": corpus_hash,
            "normalized": normalized,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class CachedExactDenseIndex:
    """Exact NumPy retrieval with a corpus/model-addressed vector cache."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        corpus_hash: str,
        settings: DenseSettings,
        cache_root: Path,
        embedder_factory: EmbedderFactory = SentenceTransformerEmbedder,
    ) -> None:
        self.documents = list(documents)
        self.corpus_hash = corpus_hash
        self.settings = settings
        self.cache_key = embedding_cache_key(
            model_name=settings.model_name,
            corpus_hash=corpus_hash,
        )
        self.cache_dir = cache_root / self.cache_key
        self.embedder = embedder_factory(settings)
        self.vectors: FloatMatrix
        self.vector_dimension = 0
        self.cache_hit = False
        self.build_time_ms = 0.0
        self._query_vectors: dict[str, NDArray[np.float32]] = {}
        self._load_or_build()

    def _load_or_build(self) -> None:
        started_at = time.perf_counter()
        metadata_path = self.cache_dir / "metadata.json"
        vectors_path = self.cache_dir / "vectors.npy"
        ids_path = self.cache_dir / "document_ids.json"
        expected_ids = [document.id for document in self.documents]

        if metadata_path.is_file() and vectors_path.is_file() and ids_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            vectors = np.asarray(
                np.load(vectors_path, allow_pickle=False),
                dtype=np.float32,
            )
            if (
                metadata.get("cache_key") == self.cache_key
                and cached_ids == expected_ids
                and vectors.ndim == 2
                and vectors.shape[0] == len(expected_ids)
                and bool(metadata.get("normalized")) is True
            ):
                self.vectors = vectors
                self.vector_dimension = (
                    int(vectors.shape[1]) if vectors.ndim == 2 else 0
                )
                self.cache_hit = True
                self.build_time_ms = (
                    time.perf_counter() - started_at
                ) * 1000
                return

        texts = [
            f"{document.title}\n{document.content}"
            for document in self.documents
        ]
        if texts:
            encoded = np.asarray(
                self.embedder.encode(texts),
                dtype=np.float32,
            )
            if encoded.ndim != 2 or encoded.shape[0] != len(texts):
                raise ValueError(
                    "embedder returned an invalid document vector matrix"
                )
            self.vectors = _l2_normalize(encoded)
            self.vector_dimension = int(encoded.shape[1])
        else:
            self.vectors = np.empty((0, 0), dtype=np.float32)
            self.vector_dimension = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, self.vectors, allow_pickle=False)
        ids_path.write_text(
            json.dumps(expected_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "format_version": CACHE_FORMAT_VERSION,
                    "cache_key": self.cache_key,
                    "model_name": self.embedder.model_name,
                    "device": self.embedder.device,
                    "dimension": self.vector_dimension,
                    "normalized": True,
                    "corpus_hash": self.corpus_hash,
                    "documents": len(self.documents),
                    "created_at": datetime.now(UTC).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.build_time_ms = (time.perf_counter() - started_at) * 1000

    def search(self, query: str, top_k: int) -> DenseSearchResponse:
        started_at = time.perf_counter()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.documents:
            return DenseSearchResponse(
                query=query,
                model_name=self.embedder.model_name,
                device=self.embedder.device,
                vector_dimension=self.vector_dimension,
                total_hits=0,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                index_version=1,
                hits=[],
            )
        query_vector = self._query_vectors.get(query)
        if query_vector is None:
            query_matrix = np.asarray(
                self.embedder.encode([query]),
                dtype=np.float32,
            )
            if query_matrix.shape != (1, self.vector_dimension):
                raise ValueError("query vector dimension does not match cache")
            query_vector = _l2_normalize(query_matrix)[0]
            self._query_vectors[query] = query_vector
        similarities = self.vectors @ query_vector
        result_count = min(top_k, len(self.documents))
        ranked_indices = np.argsort(
            -similarities,
            kind="stable",
        )[:result_count]
        return DenseSearchResponse(
            query=query,
            model_name=self.embedder.model_name,
            device=self.embedder.device,
            vector_dimension=self.vector_dimension,
            total_hits=len(self.documents),
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            index_version=1,
            hits=[
                DenseSearchHit(
                    rank=rank,
                    score=float(similarities[index]),
                    document=self.documents[int(index)],
                )
                for rank, index in enumerate(ranked_indices, start=1)
            ],
        )

    def cache_metadata(self) -> dict[str, object]:
        return {
            "cache_key": self.cache_key,
            "cache_dir": str(self.cache_dir),
            "cache_hit": self.cache_hit,
            "model_name": self.embedder.model_name,
            "device": self.embedder.device,
            "dimension": self.vector_dimension,
            "normalized": True,
            "corpus_hash": self.corpus_hash,
            "documents": len(self.documents),
            "build_time_ms": self.build_time_ms,
        }

    def clear_query_cache(self) -> None:
        """Start a fresh latency run without rebuilding document vectors."""
        self._query_vectors.clear()


def _l2_normalize(vectors: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must have non-zero norm")
    return np.asarray(vectors / norms, dtype=np.float32)
