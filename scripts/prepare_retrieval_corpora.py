from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.request
import zipfile

import duckdb
from huggingface_hub import hf_hub_download

from app.evaluation.dataset_sampling import (
    Qrel,
    filter_qrels,
    select_document_ids,
    select_query_ids,
)


T2_REPO = "mteb/T2Retrieval"
T2_REVISION = "921dd3af6e78d1ae7ee0368aa8d7eaee02c8f08e"
SCIFACT_REPO = "BeIR/scifact"
SCIFACT_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
SCIFACT_QRELS_REPO = "BeIR/scifact-qrels"
SCIFACT_QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
BEIR_BASE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets"
)
BEIR_SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
BEIR_TREC_COVID_MD5 = "ce62140cb23feb9becf6270d0d1fe6d1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download pinned public retrieval datasets and create deterministic "
            "SearchLab quality/scale profiles."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments"),
    )
    parser.add_argument("--query-count", type=int, default=50)
    parser.add_argument("--t2-quality-documents", type=int, default=1000)
    parser.add_argument("--t2-scale-documents", type=int, default=20000)
    parser.add_argument("--trec-covid-documents", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--include-t2",
        action="store_true",
        help="Also prepare T2Retrieval from Hugging Face when its CDN is reachable.",
    )
    parser.add_argument("--skip-scifact", action="store_true")
    parser.add_argument("--skip-trec-covid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.include_t2:
        prepare_t2(
            output_root=args.output_dir / "t2_retrieval_zh",
            query_count=args.query_count,
            quality_documents=args.t2_quality_documents,
            scale_documents=args.t2_scale_documents,
            seed=args.seed,
        )
    if not args.skip_scifact:
        prepare_beir_scifact(
            output_root=args.output_dir / "scifact_en",
            query_count=args.query_count,
            seed=args.seed,
        )
    if not args.skip_trec_covid:
        prepare_beir_trec_covid(
            output_root=args.output_dir / "trec_covid_en",
            document_count=args.trec_covid_documents,
            seed=args.seed,
        )


def prepare_t2(
    *,
    output_root: Path,
    query_count: int,
    quality_documents: int,
    scale_documents: int,
    seed: int,
) -> None:
    if scale_documents < quality_documents:
        raise ValueError(
            "T2 scale corpus must be at least as large as the quality corpus"
        )
    corpus_path = _download(
        T2_REPO,
        "corpus/dev-00000-of-00001.parquet",
        T2_REVISION,
    )
    queries_path = _download(
        T2_REPO,
        "queries/dev-00000-of-00001.parquet",
        T2_REVISION,
    )
    qrels_path = _download(
        T2_REPO,
        "data/dev-00000-of-00001.parquet",
        T2_REVISION,
    )

    connection = duckdb.connect()
    all_document_ids = [
        row[0]
        for row in connection.execute(
            "SELECT _id FROM read_parquet(?) ORDER BY _id",
            [str(corpus_path)],
        ).fetchall()
    ]
    query_rows = connection.execute(
        """
        SELECT _id, text
        FROM read_parquet(?)
        WHERE text IS NOT NULL AND length(trim(text)) > 0
        """,
        [str(queries_path)],
    ).fetchall()
    query_map = {str(query_id): str(text) for query_id, text in query_rows}
    qrels: list[Qrel] = [
        (str(query_id), str(document_id), int(score))
        for query_id, document_id, score in connection.execute(
            """
            SELECT "query-id", "corpus-id", score
            FROM read_parquet(?)
            WHERE score > 0
            """,
            [str(qrels_path)],
        ).fetchall()
    ]
    available_document_ids = set(all_document_ids)
    qrels = [
        item
        for item in qrels
        if item[0] in query_map and item[1] in available_document_ids
    ]
    selected_queries = select_query_ids(
        qrels,
        count=query_count,
        seed=seed,
    )
    selected_qrels = [
        item for item in qrels if item[0] in set(selected_queries)
    ]
    judged_documents = {document_id for _, document_id, _ in selected_qrels}

    quality_ids = select_document_ids(
        all_document_ids,
        required_ids=judged_documents,
        count=quality_documents,
        seed=seed + 1,
    )
    scale_ids = select_document_ids(
        all_document_ids,
        required_ids=quality_ids,
        count=scale_documents,
        seed=seed + 2,
    )
    queries = [
        {"id": query_id, "text": query_map[query_id]}
        for query_id in selected_queries
    ]

    quality_documents_rows = _fetch_documents(
        connection,
        corpus_path,
        quality_ids,
        dataset="t2_retrieval",
        language="zh",
    )
    scale_documents_rows = _fetch_documents(
        connection,
        corpus_path,
        scale_ids,
        dataset="t2_retrieval",
        language="zh",
    )
    connection.close()

    common_source = {
        "dataset": T2_REPO,
        "revision": T2_REVISION,
        "license": "apache-2.0",
        "source_corpus_documents": len(all_document_ids),
        "source_queries": len(query_map),
        "source_positive_qrels": len(qrels),
        "seed": seed,
    }
    _write_profile(
        output_root / "quality",
        documents=quality_documents_rows,
        queries=queries,
        qrels=filter_qrels(
            selected_qrels,
            query_ids=selected_queries,
            document_ids=quality_ids,
        ),
        manifest={
            **common_source,
            "profile": "quality",
            "purpose": "retrieval quality evaluation with judged queries",
        },
    )
    _write_profile(
        output_root / "scale_20k",
        documents=scale_documents_rows,
        queries=queries,
        qrels=filter_qrels(
            selected_qrels,
            query_ids=selected_queries,
            document_ids=scale_ids,
        ),
        manifest={
            **common_source,
            "profile": "scale",
            "purpose": "Exact versus ANN/HNSW latency and recall experiments",
        },
    )


def prepare_beir_scifact(
    *,
    output_root: Path,
    query_count: int,
    seed: int,
) -> None:
    raw_dir = _download_and_extract_beir(
        "scifact",
        output_root.parent,
        expected_md5=BEIR_SCIFACT_MD5,
    )
    corpus_path = _find_one(raw_dir, "corpus.jsonl")
    queries_path = _find_one(raw_dir, "queries.jsonl")
    qrels_path = _find_one(raw_dir, "test.tsv")
    documents = _read_beir_documents(corpus_path, dataset="scifact")
    query_map = _read_beir_queries(queries_path)
    qrels = _read_tsv_qrels(qrels_path)
    available_documents = {document["id"] for document in documents}
    qrels = [
        item
        for item in qrels
        if item[0] in query_map
        and item[1] in available_documents
        and item[2] > 0
    ]
    selected_queries = select_query_ids(
        qrels,
        count=min(query_count, len({item[0] for item in qrels})),
        seed=seed,
    )
    selected_qrels = filter_qrels(
        qrels,
        query_ids=selected_queries,
        document_ids=available_documents,
    )
    queries = [
        {"id": query_id, "text": query_map[query_id]}
        for query_id in selected_queries
    ]
    _write_profile(
        output_root / "quality",
        documents=documents,
        queries=queries,
        qrels=selected_qrels,
        manifest={
            "dataset": "BEIR/scifact",
            "download_url": f"{BEIR_BASE_URL}/scifact.zip",
            "archive_md5": BEIR_SCIFACT_MD5,
            "license": "see original SciFact dataset and BEIR disclaimer",
            "profile": "quality",
            "purpose": "English scientific retrieval evaluation",
            "source_corpus_documents": len(documents),
            "source_queries": len(query_map),
            "source_positive_qrels": len(qrels),
            "seed": seed,
        },
    )


def prepare_beir_trec_covid(
    *,
    output_root: Path,
    document_count: int,
    seed: int,
) -> None:
    raw_dir = _download_and_extract_beir(
        "trec-covid",
        output_root.parent,
        expected_md5=BEIR_TREC_COVID_MD5,
    )
    corpus_path = _find_one(raw_dir, "corpus.jsonl")
    queries_path = _find_one(raw_dir, "queries.jsonl")
    qrels_path = _find_one(raw_dir, "test.tsv")
    query_map = _read_beir_queries(queries_path)
    qrels = _read_tsv_qrels(qrels_path)
    all_document_ids = _read_beir_document_ids(corpus_path)
    available_documents = set(all_document_ids)
    qrels = [
        item
        for item in qrels
        if item[0] in query_map
        and item[1] in available_documents
        and item[2] > 0
    ]
    selected_queries = sorted({query_id for query_id, _, _ in qrels})
    judged_documents = {document_id for _, document_id, _ in qrels}
    selected_documents = select_document_ids(
        all_document_ids,
        required_ids=judged_documents,
        count=document_count,
        seed=seed + 10,
    )
    selected_document_set = set(selected_documents)
    documents = _read_beir_documents(
        corpus_path,
        dataset="trec-covid",
        selected_ids=selected_document_set,
    )
    queries = [
        {"id": query_id, "text": query_map[query_id]}
        for query_id in selected_queries
    ]
    selected_qrels = filter_qrels(
        qrels,
        query_ids=selected_queries,
        document_ids=selected_documents,
    )
    if len(selected_qrels) != len(qrels):
        raise ValueError(
            "TREC-COVID scale profile lost judged documents; "
            "increase --trec-covid-documents"
        )
    profile_name = (
        f"scale_{document_count // 1000}k"
        if document_count % 1000 == 0
        else f"scale_{document_count}"
    )
    _write_profile(
        output_root / profile_name,
        documents=documents,
        queries=queries,
        qrels=selected_qrels,
        manifest={
            "dataset": "BEIR/trec-covid",
            "download_url": f"{BEIR_BASE_URL}/trec-covid.zip",
            "archive_md5": BEIR_TREC_COVID_MD5,
            "license": (
                "see TREC-COVID, CORD-19 and individual document licenses"
            ),
            "profile": "scale",
            "purpose": "Exact versus ANN/HNSW latency and recall experiments",
            "source_corpus_documents": len(all_document_ids),
            "source_queries": len(query_map),
            "source_positive_qrels": len(qrels),
            "seed": seed,
            "qrels_complete": True,
        },
    )


def _download(repo_id: str, filename: str, revision: str) -> Path:
    print(f"Downloading {repo_id}/{filename} @ {revision[:12]}...")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
        )
    )


def _download_and_extract_beir(
    dataset: str,
    output_root: Path,
    *,
    expected_md5: str,
) -> Path:
    downloads_dir = output_root / "_downloads"
    raw_dir = output_root / "_raw" / dataset
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = downloads_dir / f"{dataset}.zip"
    if not archive_path.exists() or _md5(archive_path) != expected_md5:
        part_path = archive_path.with_suffix(".zip.part")
        url = f"{BEIR_BASE_URL}/{dataset}.zip"
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            next_report = 10 * 1024 * 1024
            with part_path.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(
                            f"  {downloaded / 1024 / 1024:.1f} MiB"
                            f" / {total / 1024 / 1024:.1f} MiB"
                        )
                        next_report += 10 * 1024 * 1024
        part_path.replace(archive_path)
    actual_md5 = _md5(archive_path)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"{dataset} archive MD5 mismatch: {actual_md5} != {expected_md5}"
        )
    marker = raw_dir / ".extracted-md5"
    if not marker.exists() or marker.read_text(encoding="ascii") != expected_md5:
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            resolved_root = raw_dir.resolve()
            for member in archive.infolist():
                destination = (raw_dir / member.filename).resolve()
                if (
                    destination != resolved_root
                    and resolved_root not in destination.parents
                ):
                    raise ValueError(
                        f"unsafe ZIP member in {dataset}: {member.filename}"
                    )
            archive.extractall(raw_dir)
        marker.write_text(expected_md5, encoding="ascii")
    return raw_dir


def _find_one(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {filename} under {root}, found {len(matches)}"
        )
    return matches[0]


def _read_beir_document_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(str(row["_id"]))
    return ids


def _read_beir_documents(
    path: Path,
    *,
    dataset: str,
    selected_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            document_id = str(row["_id"])
            if selected_ids is not None and document_id not in selected_ids:
                continue
            documents.append(
                _document_record(
                    document_id=document_id,
                    title=str(row.get("title") or ""),
                    content=str(row.get("text") or ""),
                    dataset=dataset,
                    language="en",
                )
            )
    return sorted(documents, key=lambda item: item["id"])


def _read_beir_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if text:
                queries[str(row["_id"])] = text
    return queries


def _fetch_documents(
    connection: duckdb.DuckDBPyConnection,
    corpus_path: Path,
    document_ids: list[str],
    *,
    dataset: str,
    language: str,
) -> list[dict[str, Any]]:
    connection.execute("DROP TABLE IF EXISTS selected_document_ids")
    connection.execute(
        "CREATE TEMP TABLE selected_document_ids (document_id VARCHAR PRIMARY KEY)"
    )
    connection.executemany(
        "INSERT INTO selected_document_ids VALUES (?)",
        [(document_id,) for document_id in document_ids],
    )
    rows = connection.execute(
        """
        SELECT corpus._id, coalesce(corpus.title, ''), corpus.text
        FROM read_parquet(?) AS corpus
        INNER JOIN selected_document_ids AS selected
            ON corpus._id = selected.document_id
        ORDER BY corpus._id
        """,
        [str(corpus_path)],
    ).fetchall()
    return [
        _document_record(
            document_id=str(document_id),
            title=str(title),
            content=str(text),
            dataset=dataset,
            language=language,
        )
        for document_id, title, text in rows
    ]


def _document_record(
    *,
    document_id: str,
    title: str,
    content: str,
    dataset: str,
    language: str,
) -> dict[str, Any]:
    return {
        "id": document_id,
        "title": title,
        "content": content,
        "source": dataset,
        "metadata": {
            "dataset": dataset,
            "language": language,
        },
    }


def _read_tsv_qrels(path: Path) -> list[Qrel]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            (
                row.get("query-id") or row.get("query_id") or "",
                row.get("corpus-id") or row.get("document_id") or "",
                int(row.get("score") or row.get("relevance") or 0),
            )
            for row in reader
            if row
        ]


def _write_profile(
    profile_dir: Path,
    *,
    documents: list[dict[str, Any]],
    queries: list[dict[str, str]],
    qrels: list[Qrel],
    manifest: dict[str, Any],
) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = profile_dir / "corpus.jsonl"
    queries_path = profile_dir / "queries.jsonl"
    qrels_path = profile_dir / "qrels.tsv"
    _write_jsonl(corpus_path, documents)
    _write_jsonl(queries_path, queries)
    with qrels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", "document_id", "relevance"])
        writer.writerows(qrels)

    completed_manifest = {
        **manifest,
        "documents": len(documents),
        "queries": len(queries),
        "qrels": len(qrels),
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (corpus_path, queries_path, qrels_path)
        },
    }
    manifest_path = profile_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(completed_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {profile_dir}: {len(documents)} documents, "
        f"{len(queries)} queries, {len(qrels)} qrels"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
