import pytest

from app.core.bm25 import BM25Index
from app.models import Document


def build_test_index() -> BM25Index:
    index = BM25Index()
    index.build(
        [
            Document(
                id="bm25",
                title="BM25 关键词检索",
                content="词频、逆文档频率与长度归一化。",
            ),
            Document(
                id="dense",
                title="Dense Retrieval",
                content="向量语义检索能够处理不同措辞。",
            ),
            Document(
                id="rrf",
                title="RRF",
                content="融合关键词检索和向量检索的结果排名。",
            ),
        ]
    )
    return index


def test_exact_title_term_ranks_first() -> None:
    response = build_test_index().search("BM25")
    assert response.hits[0].document.id == "bm25"


def test_chinese_keyword_retrieval() -> None:
    response = build_test_index().search("关键词检索")
    assert response.hits[0].document.id == "bm25"


def test_score_breakdown_is_exposed() -> None:
    response = build_test_index().search("向量语义")
    first_hit = response.hits[0]
    assert first_hit.term_contributions
    assert sum(item.score for item in first_hit.term_contributions) == pytest.approx(
        first_hit.score
    )


def test_unknown_query_returns_no_hits() -> None:
    response = build_test_index().search("quantum-teleportation")
    assert response.hits == []
    assert response.total_hits == 0
