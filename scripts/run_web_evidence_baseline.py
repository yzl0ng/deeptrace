from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx

from app.agentic.evidence import EvidenceRetriever, EvidenceStore
from app.agentic.web import (
    SafePageReader,
    StaticWebSearchProvider,
    WebEvidenceTool,
    WebSearchResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "web-evidence-v1"
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "experiments" / "web-evidence-baseline-v1"
)
DATABASE_PATH = OUTPUT_DIR / "evidence.db"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{DATABASE_PATH}{suffix}").unlink(missing_ok=True)
    snapshot = json.loads(
        (FIXTURE_DIR / "search-results.json").read_text(encoding="utf-8")
    )
    query = str(snapshot["query"])
    provider = StaticWebSearchProvider(
        {
            query: [
                WebSearchResult.model_validate(item)
                for item in snapshot["results"]
            ]
        }
    )
    request_log: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(str(request.url))
        if request.url.path == "/bm25":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=(FIXTURE_DIR / "bm25.html").read_bytes(),
            )
        if request.url.path == "/dense":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=(FIXTURE_DIR / "dense.html").read_bytes(),
            )
        return httpx.Response(
            503,
            headers={"content-type": "text/plain"},
            text="snapshot upstream unavailable",
        )

    reader = SafePageReader(
        resolver=lambda host, port: [
            (2, 1, 6, "", ("93.184.216.34", port))
        ],
        transport=httpx.MockTransport(handler),
    )
    store = EvidenceStore(DATABASE_PATH)
    tool = WebEvidenceTool(
        provider=provider,
        reader=reader,
        store=store,
        retriever=EvidenceRetriever(store),
        search_result_count=3,
        evidence_top_k=5,
        cache_ttl=timedelta(hours=24),
    )

    started = time.perf_counter()
    first = tool.execute({"query": query, "max_page_reads": 3})
    first_elapsed_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    second = tool.execute({"query": query, "max_page_reads": 3})
    second_elapsed_ms = (time.perf_counter() - started) * 1000

    counts = store.counts()
    metrics = {
        "search_results": len(first["search_results"]),
        "first_page_reads": first["_usage"]["page_reads"],
        "first_cache_hits": first["_usage"]["cache_hits"],
        "second_page_reads": second["_usage"]["page_reads"],
        "second_cache_hits": second["_usage"]["cache_hits"],
        "failed_pages": len(first["failures"]),
        "evidence_returned": len(first["evidence"]),
        "sources": counts["sources"],
        "documents": counts["documents"],
        "passages": counts["passages"],
        "evidence_records": counts["evidence"],
        "first_elapsed_ms": round(first_elapsed_ms, 3),
        "second_elapsed_ms": round(second_elapsed_ms, 3),
        "http_requests_total": len(request_log),
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "provider": "fixed_snapshot",
        "network_used": False,
    }
    config = {
        "query": query,
        "fixture": "data/fixtures/web-evidence-v1",
        "search_result_count": 3,
        "evidence_top_k": 5,
        "cache_ttl_hours": 24,
        "max_page_reads": 3,
        "retrieval": "bm25",
    }
    failures = first["failures"]
    manifest = {
        "experiment": "web-evidence-baseline-v1",
        "status": "completed",
        "truth_boundary": (
            "Offline deterministic HTTP snapshot; not a live-Web quality run."
        ),
        "artifacts": {},
    }

    _write_json("config.json", config)
    _write_json("environment.json", environment)
    _write_json("metrics.json", metrics)
    _write_json("first-run.json", first)
    _write_json("second-run.json", second)
    _write_jsonl("failures.jsonl", failures)
    _write_report(metrics)
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.name.startswith("evidence.db") or path.name == "manifest.json":
            continue
        manifest["artifacts"][path.name] = _sha256(path)
    _write_json("manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _write_json(name: str, payload: Any) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    (OUTPUT_DIR / name).write_text(content, encoding="utf-8")


def _write_report(metrics: dict[str, Any]) -> None:
    report = f"""# Web Evidence Baseline v1

Status: completed

This is a deterministic offline HTTP snapshot test. It validates the Phase 2
data flow and safety boundaries; it does not measure live-Web search quality.

## Result

- Search results: {metrics["search_results"]}
- First run: {metrics["first_page_reads"]} page reads,
  {metrics["first_cache_hits"]} cache hits
- Second run: {metrics["second_page_reads"]} page reads,
  {metrics["second_cache_hits"]} cache hits
- Failed pages: {metrics["failed_pages"]}
- Stored sources/documents/passages:
  {metrics["sources"]}/{metrics["documents"]}/{metrics["passages"]}
- Returned evidence: {metrics["evidence_returned"]}

The unavailable result remains a structured page-read failure and is never
stored as document text. The second run reads only that failed URL; the two
successful pages are served from cache. Evidence contains Source, Document and
Passage identifiers plus canonical URL and BM25 trace.
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
