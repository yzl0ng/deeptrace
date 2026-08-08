from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.dense import DenseSettings
from app.core.index_manager import IndexManager
from app.ingestion.security import UploadSettings
from app.ingestion.service import IngestionService
from app.main import (
    app,
    get_document_repository,
    get_index_manager,
    get_ingestion_service,
)
from app.storage.database import Database
from app.storage.repositories import DocumentRepository


class ApiIngestionEmbedder:
    model_name = "fake-ingestion"
    device = "cpu"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0]
                if "ZephyrGraph" in text
                else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_upload_search_duplicate_preview_and_delete(tmp_path: Path) -> None:
    repository = DocumentRepository(Database(tmp_path / "api.db"))
    manager = IndexManager(
        repository,
        DenseSettings(model_name="fake-ingestion"),
        embedder_factory=lambda _: ApiIngestionEmbedder(),
    )
    service = IngestionService(
        repository,
        manager,
        UploadSettings(upload_dir=tmp_path / "uploads"),
    )
    app.dependency_overrides[get_document_repository] = lambda: repository
    app.dependency_overrides[get_index_manager] = lambda: manager
    app.dependency_overrides[get_ingestion_service] = lambda: service
    client = TestClient(app)
    try:
        assert client.get(
            "/api/v1/search",
            params={"q": "ZephyrGraph"},
        ).json()["hits"] == []

        response = client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "../Zephyr.txt",
                    (
                        "SearchLab Upload Test\n\n"
                        "ZephyrGraph 是本次上传测试中的唯一术语。"
                    ).encode(),
                    "text/plain",
                )
            },
            data={"chunk_size": "300", "chunk_overlap": "40"},
        )
        assert response.status_code == 202
        accepted = response.json()
        assert accepted["status"] == "pending"

        job = client.get(
            f"/api/v1/ingestions/{accepted['job_id']}"
        ).json()
        assert job["status"] == "completed"
        document = client.get(
            f"/api/v1/documents/{accepted['document_id']}"
        ).json()
        assert document["original_filename"] == "Zephyr.txt"
        assert document["chunk_count"] == 1

        chunks = client.get(
            f"/api/v1/documents/{accepted['document_id']}/chunks"
        ).json()
        assert chunks["chunks"][0]["chunk_id"].startswith("chunk-")
        assert "ZephyrGraph" in chunks["chunks"][0]["text"]

        lexical = client.get(
            "/api/v1/search",
            params={"q": "ZephyrGraph", "top_k": 3},
        ).json()
        dense = client.get(
            "/api/v1/search/dense",
            params={"q": "ZephyrGraph", "top_k": 3},
        ).json()
        hybrid = client.get(
            "/api/v1/search/hybrid",
            params={
                "q": "ZephyrGraph",
                "top_k": 1,
                "candidate_k": 1,
            },
        ).json()
        assert lexical["hits"][0]["document"]["id"].startswith("chunk-")
        assert dense["hits"][0]["document"]["id"].startswith("chunk-")
        assert hybrid["hits"][0]["document"]["id"].startswith("chunk-")

        duplicate = client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "renamed.txt",
                    (
                        "SearchLab Upload Test\n\n"
                        "ZephyrGraph 是本次上传测试中的唯一术语。"
                    ).encode(),
                    "text/plain",
                )
            },
            data={"chunk_size": "300", "chunk_overlap": "40"},
        ).json()
        assert duplicate["duplicate"] is True
        assert duplicate["document_id"] == accepted["document_id"]

        deleted = client.delete(
            f"/api/v1/documents/{accepted['document_id']}"
        )
        assert deleted.status_code == 200
        assert client.get(
            "/api/v1/search",
            params={"q": "ZephyrGraph"},
        ).json()["hits"] == []
    finally:
        app.dependency_overrides.clear()


def test_upload_validation_errors_are_explicit(tmp_path: Path) -> None:
    repository = DocumentRepository(Database(tmp_path / "invalid.db"))
    manager = IndexManager(
        repository,
        DenseSettings(model_name="fake-ingestion"),
        embedder_factory=lambda _: ApiIngestionEmbedder(),
    )
    service = IngestionService(
        repository,
        manager,
        UploadSettings(upload_dir=tmp_path / "uploads", max_upload_mb=1),
    )
    app.dependency_overrides[get_document_repository] = lambda: repository
    app.dependency_overrides[get_index_manager] = lambda: manager
    app.dependency_overrides[get_ingestion_service] = lambda: service
    client = TestClient(app)
    try:
        unsupported = client.post(
            "/api/v1/documents",
            files={"file": ("bad.exe", b"hello", "application/octet-stream")},
        )
        empty = client.post(
            "/api/v1/documents",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        scanned = client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "blank.pdf",
                    _blank_pdf(),
                    "application/pdf",
                )
            },
        )
        assert unsupported.json()["detail"]["code"] == "unsupported_file_type"
        assert empty.json()["detail"]["code"] == "empty_file"
        assert scanned.json()["detail"]["code"] == "scanned_pdf_or_no_text"
    finally:
        app.dependency_overrides.clear()


def _blank_pdf() -> bytes:
    from io import BytesIO
    from pypdf import PdfWriter

    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(output)
    return output.getvalue()
