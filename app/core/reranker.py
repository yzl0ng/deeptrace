from __future__ import annotations

from dataclasses import dataclass
import os
from threading import Lock
import time
from typing import Callable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from app.models import (
    HybridSearchHit,
    RerankerModelInfo,
    RerankResult,
    RerankTrace,
)


FloatVector = NDArray[np.float32]


class RerankerUnavailableError(RuntimeError):
    """Raised when the configured cross-encoder cannot serve requests."""


class PairScorer(Protocol):
    model_name: str
    device: str

    def score(self, query: str, documents: Sequence[str]) -> FloatVector:
        """Return one raw relevance score for every query-document pair."""


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        hits: Sequence[HybridSearchHit],
        top_k: int,
    ) -> RerankResult: ...


@dataclass(frozen=True, slots=True)
class RerankerSettings:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    batch_size: int = 4
    max_length: int = 512

    @classmethod
    def from_environment(cls) -> RerankerSettings:
        batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "4"))
        max_length = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
        if batch_size < 1:
            raise ValueError("RERANKER_BATCH_SIZE must be at least 1")
        if max_length < 32:
            raise ValueError("RERANKER_MAX_LENGTH must be at least 32")
        return cls(
            model_name=os.getenv(
                "RERANKER_MODEL_NAME",
                "BAAI/bge-reranker-v2-m3",
            ),
            device=os.getenv("RERANKER_DEVICE", "auto").lower(),
            batch_size=batch_size,
            max_length=max_length,
        )


def select_reranker_device(
    requested: str,
    *,
    cuda_available: bool | None = None,
) -> str:
    normalized = requested.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "RERANKER_DEVICE must be one of: auto, cpu, cuda"
        )
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False
    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    if normalized == "cuda" and not cuda_available:
        raise RerankerUnavailableError(
            "RERANKER_DEVICE=cuda was requested, but CUDA is not available."
        )
    return normalized


class CrossEncoderPairScorer:
    """Sentence Transformers adapter for a production cross-encoder."""

    def __init__(self, settings: RerankerSettings) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RerankerUnavailableError(
                "Reranker dependencies are not installed."
            ) from error

        self.model_name = settings.model_name
        try:
            self.device = select_reranker_device(settings.device)
        except Exception as error:
            raise RerankerUnavailableError(str(error)) from error
        self.batch_size = settings.batch_size
        try:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=settings.max_length,
                # Sentence Transformers otherwise applies Sigmoid for a
                # single-label model. Keep logits so the API does not suggest
                # that the score is a calibrated relevance probability.
                activation_fn=torch.nn.Identity(),
            )
        except Exception as error:
            raise RerankerUnavailableError(
                f"Unable to load reranker model {self.model_name!r} "
                f"on {self.device}: {error}"
            ) from error

    def score(self, query: str, documents: Sequence[str]) -> FloatVector:
        if not documents:
            return np.empty((0,), dtype=np.float32)
        pairs = [(query, document) for document in documents]
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as error:
            raise RerankerUnavailableError(
                f"Reranker inference failed: {error}"
            ) from error
        return np.asarray(scores, dtype=np.float32).reshape(-1)


class CrossEncoderReranker:
    """Rerank a bounded RRF candidate set while preserving the full trace."""

    def __init__(self, scorer: PairScorer) -> None:
        self.scorer = scorer

    def rerank(
        self,
        query: str,
        hits: Sequence[HybridSearchHit],
        top_k: int,
    ) -> RerankResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidates = list(hits)
        started_at = time.perf_counter()
        if not candidates:
            return RerankResult(
                query=query,
                model=RerankerModelInfo(
                    name=self.scorer.model_name,
                    device=self.scorer.device,
                ),
                candidate_count=0,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                hits=[],
                traces=[],
            )

        documents = [
            f"{hit.document.title}\n{hit.document.content}"
            for hit in candidates
        ]
        scores = np.asarray(
            self.scorer.score(query, documents),
            dtype=np.float32,
        ).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError(
                "PairScorer must return exactly one score per candidate."
            )
        if not np.isfinite(scores).all():
            raise ValueError("Reranker scores must contain only finite values.")

        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (
                -float(item[1]),
                item[0].rank,
                item[0].document.id,
            ),
        )
        reranked_hits: list[HybridSearchHit] = []
        traces: list[RerankTrace] = []
        for rerank_rank, (hit, score) in enumerate(scored, start=1):
            traces.append(
                RerankTrace(
                    document_id=hit.document.id,
                    document_title=hit.document.title,
                    rrf_rank=hit.rank,
                    rerank_rank=rerank_rank,
                    rank_delta=hit.rank - rerank_rank,
                    reranker_score=float(score),
                )
            )
            if rerank_rank <= top_k:
                reranked_hits.append(
                    hit.model_copy(
                        update={
                            "rank": rerank_rank,
                            "rrf_rank": hit.rank,
                            "reranker_score": float(score),
                            "rank_delta": hit.rank - rerank_rank,
                        }
                    )
                )

        return RerankResult(
            query=query,
            model=RerankerModelInfo(
                name=self.scorer.model_name,
                device=self.scorer.device,
            ),
            candidate_count=len(candidates),
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            hits=reranked_hits,
            traces=traces,
        )


PairScorerFactory = Callable[[RerankerSettings], PairScorer]


class RerankerRuntime:
    """Thread-safe lazy lifecycle for the production cross-encoder."""

    def __init__(
        self,
        settings: RerankerSettings,
        scorer_factory: PairScorerFactory = CrossEncoderPairScorer,
    ) -> None:
        self.settings = settings
        self._scorer_factory = scorer_factory
        self._reranker: CrossEncoderReranker | None = None
        self._error: str | None = None
        self._lock = Lock()

    def rerank(
        self,
        query: str,
        hits: Sequence[HybridSearchHit],
        top_k: int,
    ) -> RerankResult:
        return self._ensure_reranker().rerank(query, hits, top_k)

    def _ensure_reranker(self) -> CrossEncoderReranker:
        if self._reranker is not None:
            return self._reranker
        if self._error is not None:
            raise RerankerUnavailableError(self._error)
        with self._lock:
            if self._reranker is not None:
                return self._reranker
            if self._error is not None:
                raise RerankerUnavailableError(self._error)
            try:
                reranker = CrossEncoderReranker(
                    self._scorer_factory(self.settings)
                )
            except Exception as error:
                self._error = str(error)
                if isinstance(error, RerankerUnavailableError):
                    raise
                raise RerankerUnavailableError(str(error)) from error
            self._reranker = reranker
            return reranker
