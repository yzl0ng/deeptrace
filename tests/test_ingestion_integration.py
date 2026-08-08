from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

from app.core.dense import DenseSettings
from app.core.index_manager import IndexManager
from app.core.reranker import RerankerRuntime, RerankerSettings
from app.ingestion.security import UploadSettings
from app.ingestion.service import IngestionService
from app.storage.database import Database
from app.storage.repositories import DocumentRepository


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INGESTION_INTEGRATION") != "1",
        reason="Set RUN_INGESTION_INTEGRATION=1 for real BGE ingestion.",
    ),
]


def test_real_bge_dynamic_ingestion_cache_rerank_and_delete(
    tmp_path: Path,
) -> None:
    repository = DocumentRepository(Database(tmp_path / "ingestion.db"))
    manager = IndexManager(
        repository,
        DenseSettings(
            model_name="BAAI/bge-m3",
            device=os.getenv("DENSE_DEVICE", "auto"),
            batch_size=8,
        ),
    )
    service = IngestionService(
        repository,
        manager,
        UploadSettings(upload_dir=tmp_path / "uploads"),
    )
    assert manager.current().bm25.search("ZephyrGraph", 5).hits == []

    content = (
        "SearchLab Upload Test\n\n"
        "ZephyrGraph 是本次上传测试中的唯一术语。\n"
        "它用于验证动态 BM25、Dense 和 RRF 索引更新。"
    ).encode("utf-8")
    started = time.perf_counter()
    prepared = service.prepare_upload(
        filename="zephyr.txt",
        content_type="text/plain",
        content=content,
        chunk_size=300,
        chunk_overlap=40,
        title="SearchLab Upload Test",
    )
    service.process(prepared)
    ingestion_ms = (time.perf_counter() - started) * 1000

    document = repository.get_document(prepared.document_id)
    assert document is not None
    assert document.status == "completed"
    assert document.chunk_count == 1
    snapshot = manager.current()
    bm25 = snapshot.bm25.search("ZephyrGraph", 5)
    dense = snapshot.dense.search(
        "用于验证动态混合检索更新的唯一术语",
        5,
    )
    hybrid = snapshot.hybrid.search(
        "ZephyrGraph 动态索引",
        1,
        candidate_k=1,
    )
    assert bm25.hits[0].document.metadata["document_id"] == prepared.document_id
    assert dense.hits[0].document.metadata["document_id"] == prepared.document_id
    assert hybrid.hits[0].document.metadata["document_id"] == prepared.document_id
    assert dense.vector_dimension == 1024
    assert dense.device == "cuda"

    reranker = RerankerRuntime(RerankerSettings.from_environment())
    reranked = reranker.rerank(
        "ZephyrGraph 动态索引",
        hybrid.hits,
        top_k=1,
    )
    assert (
        reranked.hits[0].document.metadata["document_id"]
        == prepared.document_id
    )
    assert reranked.model.device == "cuda"

    manager.rebuild(eager_dense=True)
    assert manager.current().dense.cache_hits == 1
    assert manager.current().dense.cache_misses == 0

    repository.update_document_status(prepared.document_id, "deleting")
    manager.rebuild(eager_dense=True)
    repository.delete_document(prepared.document_id)
    assert manager.current().bm25.search("ZephyrGraph", 5).hits == []
    assert manager.current().dense.search("ZephyrGraph", 5).hits == []

    print(
        {
            "ingestion_ms": round(ingestion_ms, 2),
            "device": dense.device,
            "dimension": dense.vector_dimension,
            "bm25_rank": bm25.hits[0].rank,
            "dense_rank": dense.hits[0].rank,
            "rrf_rank": hybrid.hits[0].rank,
            "reranker_rank": reranked.hits[0].rank,
        }
    )
