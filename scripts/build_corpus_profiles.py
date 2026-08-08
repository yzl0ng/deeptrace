from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.corpus import (
    CorpusChunk,
    content_hash,
    normalize_text,
    stable_chunk_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "experiments"
CORPUS_ROOT = PROJECT_ROOT / "data" / "corpora"

PROFILES = [
    {
        "dataset": "scifact",
        "input": EXPERIMENT_ROOT / "scifact_en" / "quality" / "corpus.jsonl",
        "input_manifest": (
            EXPERIMENT_ROOT / "scifact_en" / "quality" / "manifest.json"
        ),
        "output": CORPUS_ROOT / "quality-v1.jsonl",
        "id_map": CORPUS_ROOT / "id_maps" / "scifact.json",
        "split": "quality",
        "source": (
            "https://public.ukp.informatik.tu-darmstadt.de/"
            "thakur/BEIR/datasets/scifact.zip"
        ),
        "source_type": "benchmark",
        "language": "en",
        "topic": "scientific-claim-verification",
        "license": "See SciFact source dataset and BEIR disclaimer",
    },
    {
        "dataset": "trec-covid",
        "input": (
            EXPERIMENT_ROOT
            / "trec_covid_en"
            / "scale_50k"
            / "corpus.jsonl"
        ),
        "input_manifest": (
            EXPERIMENT_ROOT
            / "trec_covid_en"
            / "scale_50k"
            / "manifest.json"
        ),
        "output": CORPUS_ROOT / "scale-v1.jsonl",
        "id_map": CORPUS_ROOT / "id_maps" / "trec-covid.json",
        "split": "scale",
        "source": (
            "https://public.ukp.informatik.tu-darmstadt.de/"
            "thakur/BEIR/datasets/trec-covid.zip"
        ),
        "source_type": "benchmark",
        "language": "en",
        "topic": "biomedical-retrieval",
        "license": "See TREC-COVID, CORD-19 and document-level licenses",
    },
]


def main() -> None:
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    (CORPUS_ROOT / "id_maps").mkdir(parents=True, exist_ok=True)
    built_profiles: dict[str, dict[str, Any]] = {}
    for profile in PROFILES:
        built_profiles[profile["split"]] = build_profile(profile)

    combined = hashlib.sha256()
    for split in sorted(built_profiles):
        combined.update(
            built_profiles[split]["content_hash"].encode("ascii")
        )
    manifest = {
        "version": "corpora-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_model": None,
        "chunking": {
            "strategy": "source-record-as-chunk",
            "chunk_size": 0,
            "chunk_overlap": 0,
        },
        "quality_corpus": built_profiles["quality"],
        "scale_corpus": built_profiles["scale"],
        "sources": sorted(
            {profile["source"] for profile in PROFILES}
        ),
        "licenses": sorted(
            {profile["license"] for profile in PROFILES}
        ),
        "content_hash": combined.hexdigest(),
    }
    (CORPUS_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Built corpus profiles: "
        f"quality={built_profiles['quality']['chunks']}, "
        f"scale={built_profiles['scale']['chunks']}"
    )


def build_profile(profile: dict[str, Any]) -> dict[str, Any]:
    input_manifest = json.loads(
        profile["input_manifest"].read_text(encoding="utf-8")
    )
    canonical_by_hash: dict[str, CorpusChunk] = {}
    original_to_chunk: dict[str, str] = {}
    source_records = 0
    empty_records = 0

    with profile["input"].open("r", encoding="utf-8") as handle:
        for line in handle:
            source_records += 1
            row = json.loads(line)
            original_id = str(row["id"])
            title = normalize_text(str(row.get("title") or ""))
            body = normalize_text(str(row.get("content") or ""))
            text = body or title
            if not text:
                empty_records += 1
                continue
            hash_value = content_hash(text)
            existing = canonical_by_hash.get(hash_value)
            if existing is not None:
                original_to_chunk[original_id] = existing.chunk_id
                existing.metadata.setdefault(
                    "duplicate_original_ids", []
                ).append(original_id)
                continue
            chunk = CorpusChunk(
                document_id=f"{profile['dataset']}:{original_id}",
                chunk_id=stable_chunk_id(profile["dataset"], original_id),
                title=title,
                text=text,
                source=profile["source"],
                source_type=profile["source_type"],
                language=profile["language"],
                topic=profile["topic"],
                section="",
                page_number=None,
                license=profile["license"],
                corpus_split=profile["split"],
                content_hash=hash_value,
                metadata={
                    "dataset": profile["dataset"],
                    "original_id": original_id,
                },
            )
            canonical_by_hash[hash_value] = chunk
            original_to_chunk[original_id] = chunk.chunk_id

    chunks = sorted(
        canonical_by_hash.values(),
        key=lambda item: item.chunk_id,
    )
    profile["output"].parent.mkdir(parents=True, exist_ok=True)
    with profile["output"].open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    profile["id_map"].write_text(
        json.dumps(original_to_chunk, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    output_hash = _sha256(profile["output"])
    return {
        "version": f"{profile['split']}-v1",
        "path": str(profile["output"].relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "source_manifest": str(
            profile["input_manifest"].relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "source_manifest_hash": _sha256(profile["input_manifest"]),
        "documents": len(chunks),
        "chunks": len(chunks),
        "source_records": source_records,
        "empty_records_removed": empty_records,
        "exact_duplicates_removed": (
            source_records - empty_records - len(chunks)
        ),
        "content_hash": output_hash,
        "source": profile["source"],
        "license": profile["license"],
        "source_archive_md5": input_manifest["archive_md5"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
