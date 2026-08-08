from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.corpus import CorpusChunk
from app.evaluation.corpus_audit import audit_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT / "data" / "corpora"
REPORT_ROOT = PROJECT_ROOT / "data" / "reports"


def main() -> None:
    manifest = json.loads(
        (CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    reports: dict[str, Any] = {}
    for split, manifest_key in (
        ("quality", "quality_corpus"),
        ("scale", "scale_corpus"),
    ):
        profile = manifest[manifest_key]
        chunks = load_chunks(PROJECT_ROOT / profile["path"])
        reports[split] = audit_chunks(
            chunks,
            source_records=profile["source_records"],
            exact_duplicates_removed=profile["exact_duplicates_removed"],
        )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_version": manifest["version"],
        "corpus_content_hash": manifest["content_hash"],
        "thresholds": {
            "too_short_chars": 200,
            "too_long_chars": 4000,
        },
        "profiles": reports,
        "sources": manifest["sources"],
        "licenses": manifest["licenses"],
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "corpus_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_ROOT / "corpus_audit.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        "Corpus audit generated: "
        f"quality={reports['quality']['chunks']}, "
        f"scale={reports['scale']['chunks']}"
    )


def load_chunks(path: Path) -> list[CorpusChunk]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            CorpusChunk.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SearchLab Corpus Audit",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Corpus version: `{report['corpus_version']}`",
        f"- Content hash: `{report['corpus_content_hash']}`",
        "",
        "The near-duplicate count is an approximate screening signal, not a "
        "ground-truth duplicate label.",
        "",
    ]
    for split in ("quality", "scale"):
        item = report["profiles"][split]
        length = item["character_length"]
        lines.extend(
            [
                f"## {split.title()} Corpus",
                "",
                "| Check | Result |",
                "|---|---:|",
                f"| Source records | {item['source_records']} |",
                f"| Documents / chunks | {item['documents']} |",
                f"| Empty text | {item['empty_text']} |",
                f"| Exact duplicates removed | "
                f"{item['exact_duplicates_removed']} |",
                f"| Exact duplicates after dedup | "
                f"{item['duplicate_text_after_dedup']} |",
                f"| Approx. near-duplicate pairs | "
                f"{item['near_duplicate_pairs_approx']} |",
                f"| Average characters | {length['average']} |",
                f"| P50 characters | {length['p50']} |",
                f"| P95 characters | {length['p95']} |",
                f"| Too short (<{item['too_short_threshold']}) | "
                f"{item['too_short_chunks']} |",
                f"| Too long (>{item['too_long_threshold']}) | "
                f"{item['too_long_chunks']} |",
                f"| Missing title | {item['missing_title']} |",
                f"| Missing source | {item['missing_source']} |",
                f"| Invalid URL | {item['invalid_url']} |",
                f"| Encoding anomaly | {item['encoding_anomaly']} |",
                f"| HTML/script residue | "
                f"{item['html_or_script_residue']} |",
                "",
                f"- Languages: `{item['language_distribution']}`",
                f"- Topics: `{item['topic_distribution']}`",
                f"- Sources: `{item['source_distribution']}`",
                f"- Licenses: `{item['license_distribution']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
