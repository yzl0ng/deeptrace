from dataclasses import dataclass

from app.evaluation.evaluator import (
    EvaluationQuery,
    evaluate_retriever,
)
from app.models import Document


@dataclass
class FakeHit:
    document: Document
    score: float


@dataclass
class FakeResponse:
    hits: list[FakeHit]


class RecordingFakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int) -> FakeResponse:
        self.queries.append(query)
        documents = [
            Document(id="d1", title="one", content="one"),
            Document(id="d2", title="two", content="two"),
        ]
        return FakeResponse(
            hits=[
                FakeHit(document=document, score=1.0 / rank)
                for rank, document in enumerate(
                    documents[:top_k],
                    start=1,
                )
            ]
        )


def test_evaluator_uses_same_queries_and_aggregates_metrics() -> None:
    queries = [
        EvaluationQuery("q1", "first", "exact-term", {"d1": 3}, True),
        EvaluationQuery("q2", "second", "semantic", {"d2": 2}, True),
    ]
    retriever = RecordingFakeRetriever()

    summary, per_query = evaluate_retriever(
        "fake",
        retriever,
        queries,
        top_k=2,
    )

    assert retriever.queries == ["first", "second"]
    assert summary["queries"] == 2
    assert summary["recall@1"] == 0.5
    assert summary["recall@3"] == 1.0
    assert summary["mrr"] == 0.75
    assert summary["latency"]["queries"] == 2
    assert len(per_query) == 2
    assert per_query[0]["ranked_chunk_ids"] == ["d1", "d2"]
