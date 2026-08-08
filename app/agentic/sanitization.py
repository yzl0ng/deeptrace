from __future__ import annotations

import re

SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([a-z0-9_]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"password|secret)[a-z0-9_]*)"
    r"\b(\s*[:=]\s*)(\S+)"
)
HOME_PATH = re.compile(r"/home/[^/\s=:]+")
DATA_USER_PATH = re.compile(r"/data/[^/\s=:]+(?=/)")
SSH_TARGET = re.compile(r"(?i)\b(?:ssh|scp)\s+[^@\s]+@[^:\s]+")
QUOTA_USER = re.compile(
    r"(?im)^Disk quotas for user \S+(?:\s+\(uid\s+\d+\))?:?"
)


def sanitize_text(value: str) -> str:
    sanitized = SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", value)
    sanitized = HOME_PATH.sub("$HOME", sanitized)
    sanitized = DATA_USER_PATH.sub("/data/$USER", sanitized)
    sanitized = QUOTA_USER.sub(
        "Disk quotas for current user",
        sanitized,
    )
    return SSH_TARGET.sub("ssh <redacted-target>", sanitized)
