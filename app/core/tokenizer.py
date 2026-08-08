from __future__ import annotations

import re
import unicodedata


_TOKEN_PATTERN = re.compile(
    r"[a-z0-9][a-z0-9_.+\-/#]*|[\u4e00-\u9fff]+",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize width and case while keeping technical punctuation."""
    return unicodedata.normalize("NFKC", text).lower().strip()


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and technical English without external packages.

    English terms are kept as words. Chinese spans are converted to overlapping
    character bigrams so that the baseline remains deterministic and easy to
    inspect. A single Chinese character is retained as-is.
    """
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalize_text(text)):
        value = match.group(0)
        if "\u4e00" <= value[0] <= "\u9fff":
            if len(value) == 1:
                tokens.append(f"zh:{value}")
            else:
                tokens.extend(
                    f"zh:{value[index:index + 2]}"
                    for index in range(len(value) - 1)
                )
        else:
            tokens.append(value)
    return tokens
