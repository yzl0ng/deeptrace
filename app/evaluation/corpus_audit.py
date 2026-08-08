from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from app.evaluation.corpus import CorpusChunk, normalize_text


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
HTML_PATTERN = re.compile(
    r"<!doctype|<html\b|<script\b|</?[a-z][^>]{0,200}>",
    re.IGNORECASE,
)
ENCODING_ANOMALY_PATTERN = re.compile(r"\ufffd|\x00|Ã.|Â.|â€™")


def audit_chunks(
    chunks: list[CorpusChunk],
    *,
    source_records: int,
    exact_duplicates_removed: int,
    too_short_chars: int = 200,
    too_long_chars: int = 4000,
) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    content_hashes = Counter(chunk.content_hash for chunk in chunks)
    near_pairs, near_samples = find_near_duplicates(chunks)
    return {
        "source_records": source_records,
        "documents": len({chunk.document_id for chunk in chunks}),
        "chunks": len(chunks),
        "unique_chunk_ids": len({chunk.chunk_id for chunk in chunks}),
        "empty_text": sum(not chunk.text.strip() for chunk in chunks),
        "duplicate_text_after_dedup": sum(
            count - 1 for count in content_hashes.values() if count > 1
        ),
        "exact_duplicates_removed": exact_duplicates_removed,
        "near_duplicate_method": (
            "64-bit token SimHash, four 16-bit bands, Hamming distance <= 3"
        ),
        "near_duplicate_pairs_approx": near_pairs,
        "near_duplicate_samples": near_samples,
        "character_length": {
            "average": round(sum(lengths) / max(len(lengths), 1), 2),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "minimum": min(lengths, default=0),
            "maximum": max(lengths, default=0),
        },
        "too_short_threshold": too_short_chars,
        "too_short_chunks": sum(
            length < too_short_chars for length in lengths
        ),
        "too_long_threshold": too_long_chars,
        "too_long_chunks": sum(
            length > too_long_chars for length in lengths
        ),
        "language_distribution": dict(
            sorted(Counter(chunk.language for chunk in chunks).items())
        ),
        "source_distribution": dict(
            sorted(Counter(chunk.source for chunk in chunks).items())
        ),
        "source_type_distribution": dict(
            sorted(Counter(chunk.source_type for chunk in chunks).items())
        ),
        "topic_distribution": dict(
            sorted(Counter(chunk.topic for chunk in chunks).items())
        ),
        "license_distribution": dict(
            sorted(Counter(chunk.license for chunk in chunks).items())
        ),
        "missing_title": sum(not chunk.title.strip() for chunk in chunks),
        "missing_source": sum(not chunk.source.strip() for chunk in chunks),
        "invalid_url": sum(not valid_http_url(chunk.source) for chunk in chunks),
        "encoding_anomaly": sum(
            bool(ENCODING_ANOMALY_PATTERN.search(chunk.text))
            for chunk in chunks
        ),
        "html_or_script_residue": sum(
            bool(HTML_PATTERN.search(chunk.text)) for chunk in chunks
        ),
    }


def percentile(values: list[int] | list[float], quantile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(
        ordered[lower] * (1 - fraction) + ordered[upper] * fraction,
        2,
    )


def valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def simhash64(text: str) -> int:
    tokens = TOKEN_PATTERN.findall(normalize_text(text).lower())
    features: Iterable[str]
    if len(tokens) > 1:
        features = (
            f"{tokens[index]}\0{tokens[index + 1]}"
            for index in range(len(tokens) - 1)
        )
    else:
        features = tokens
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(
            hashlib.blake2b(
                feature.encode("utf-8"),
                digest_size=8,
            ).digest(),
            "big",
        )
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def find_near_duplicates(
    chunks: list[CorpusChunk],
    *,
    max_hamming_distance: int = 3,
    sample_limit: int = 20,
) -> tuple[int, list[dict[str, Any]]]:
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()
    count = 0
    samples: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        fingerprint = simhash64(chunk.text)
        candidates: set[tuple[int, int]] = set()
        for band in range(4):
            key = (band, (fingerprint >> (band * 16)) & 0xFFFF)
            candidates.update(buckets[key])
        for other_index, other_fingerprint in candidates:
            pair = (other_index, index)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            distance = (fingerprint ^ other_fingerprint).bit_count()
            if distance <= max_hamming_distance:
                count += 1
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "left_chunk_id": chunks[other_index].chunk_id,
                            "right_chunk_id": chunk.chunk_id,
                            "hamming_distance": distance,
                        }
                    )
        for band in range(4):
            key = (band, (fingerprint >> (band * 16)) & 0xFFFF)
            buckets[key].append((index, fingerprint))
    return count, samples
