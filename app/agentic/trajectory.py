from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

ALLOWED_ACTIONS = {"search", "read_page", "evaluate_evidence", "answer"}
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|password|secret|bearer)\s*[:=]\s*\S+"
)
AGENT_SFT_SYSTEM_PROMPT = (
    "You are the DeepTrace search-trajectory policy. Return exactly one "
    "JSON object and no markdown or surrounding prose. The required schema "
    'is {"steps":[{"rationale_summary":string,"action":'
    '"search|read_page|evaluate_evidence|answer","arguments":object,'
    '"observation":string,"evidence_ids":[string]}],"final_answer":string}. '
    "Use 1-8 concise steps. Use exactly one answer action and make it the "
    "last step. Give only brief decision summaries, never private "
    "chain-of-thought. Do not invent evidence IDs or secrets."
)


class TrajectorySeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_id: str
    query: str
    language: str
    variant: Literal["research", "verification"]
    expected_answer: str = ""
    evidence: list[dict[str, str]] = Field(default_factory=list)


class TrajectoryStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale_summary: str = Field(min_length=3, max_length=500)
    action: Literal["search", "read_page", "evaluate_evidence", "answer"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: str = Field(max_length=4000)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_tool_observation(self) -> TrajectoryStep:
        if self.action != "answer" and not self.observation.strip():
            raise ValueError("non-answer steps require an observation")
        return self


class TeacherTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_id: str
    query: str
    variant: Literal["research", "verification"]
    steps: list[TrajectoryStep] = Field(min_length=1, max_length=8)
    final_answer: str = Field(min_length=1, max_length=6000)


class AgentSFTResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[TrajectoryStep] = Field(min_length=1, max_length=8)
    final_answer: str = Field(min_length=1, max_length=6000)


def build_pilot_seeds(root: Path, *, target: int = 100) -> list[TrajectorySeed]:
    records: list[TrajectorySeed] = []
    records.extend(_agentic_seeds(root))
    records.extend(_retrieval_seeds(root))
    records.extend(_failure_seeds(root))
    if not records:
        raise RuntimeError("no Phase 4 or retrieval seed records found")
    base = list(records)
    used_ids = {item.seed_id for item in records}
    index = 0
    while len(records) < target:
        source = base[index % len(base)]
        variant = (
            "verification"
            if source.variant == "research"
            else "research"
        )
        candidate_id = f"{source.seed_id}-{variant}"
        duplicate_index = 2
        while candidate_id in used_ids:
            candidate_id = (
                f"{source.seed_id}-{variant}-{duplicate_index}"
            )
            duplicate_index += 1
        records.append(
            source.model_copy(
                update={
                    "seed_id": candidate_id,
                    "variant": variant,
                }
            )
        )
        used_ids.add(candidate_id)
        index += 1
    return records[:target]


def teacher_messages(seed: TrajectorySeed) -> list[dict[str, str]]:
    schema = {
        "seed_id": seed.seed_id,
        "query": seed.query,
        "variant": seed.variant,
        "steps": [
            {
                "rationale_summary": "brief decision summary, not hidden CoT",
                "action": "search|read_page|evaluate_evidence|answer",
                "arguments": {},
                "observation": "result available to the agent",
                "evidence_ids": [],
            }
        ],
        "final_answer": "grounded answer",
    }
    context = {
        "expected_answer": seed.expected_answer,
        "evidence": seed.evidence,
    }
    return [
        {
            "role": "system",
            "content": (
                "You create concise tool-use training trajectories for a "
                "search agent. Return exactly one JSON object and no markdown. "
                "Use only search, read_page, evaluate_evidence, and answer. "
                "Give short decision summaries, never private chain-of-thought. "
                "Do not invent evidence IDs or secrets."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Required schema: {json.dumps(schema, ensure_ascii=False)}\n"
                f"Task context: {json.dumps(context, ensure_ascii=False)}"
            ),
        },
    ]


def parse_and_filter_trajectory(
    text: str,
    seed: TrajectorySeed,
) -> tuple[TeacherTrajectory | None, list[str]]:
    reasons: list[str] = []
    try:
        payload = json.loads(_strip_code_fence(text))
        trajectory = TeacherTrajectory.model_validate(payload)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    except ValidationError as exc:
        details = sorted(
            {
                (
                    ".".join(map(str, error["loc"])) or "root"
                )
                + ":"
                + str(error["type"])
                for error in exc.errors()
            }
        )
        return None, [f"invalid_schema:{detail}" for detail in details]
    if trajectory.seed_id != seed.seed_id:
        reasons.append("seed_id_mismatch")
    if trajectory.query.strip() != seed.query.strip():
        reasons.append("query_mismatch")
    if trajectory.variant != seed.variant:
        reasons.append("variant_mismatch")
    if any(step.action not in ALLOWED_ACTIONS for step in trajectory.steps):
        reasons.append("invalid_action")
    answer_count = sum(
        step.action == "answer" for step in trajectory.steps
    )
    if answer_count == 0:
        reasons.append("missing_answer_action")
    elif answer_count > 1:
        reasons.append("multiple_answer_actions")
    if trajectory.steps[-1].action != "answer":
        reasons.append("answer_action_not_final")
    serialized = trajectory.model_dump_json()
    if SECRET_PATTERN.search(serialized):
        reasons.append("possible_secret")
    known_ids = {
        str(item.get("evidence_id", ""))
        for item in seed.evidence
        if item.get("evidence_id")
    }
    used_ids = {
        evidence_id
        for step in trajectory.steps
        for evidence_id in step.evidence_ids
    }
    if known_ids and not used_ids.issubset(known_ids):
        reasons.append("invented_evidence_id")
    return (None if reasons else trajectory), reasons


def to_sft_record(trajectory: TeacherTrajectory) -> dict[str, Any]:
    assistant = {
        "steps": [
            step.model_dump(mode="json") for step in trajectory.steps
        ],
        "final_answer": trajectory.final_answer,
    }
    return {
        "id": trajectory.seed_id,
        "messages": [
            {
                "role": "system",
                "content": AGENT_SFT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": trajectory.query},
            {
                "role": "assistant",
                "content": json.dumps(assistant, ensure_ascii=False),
            },
        ],
        "trajectory_sha256": hashlib.sha256(
            trajectory.model_dump_json().encode()
        ).hexdigest(),
    }


def parse_sft_response(text: str) -> AgentSFTResponse | None:
    try:
        return AgentSFTResponse.model_validate_json(_strip_code_fence(text))
    except (ValidationError, ValueError):
        return None


def _agentic_seeds(root: Path) -> list[TrajectorySeed]:
    paths = [
        root / "data/evaluation/agentic-search-v1/test.jsonl",
        root / "data/evaluation/agentic-search-v1/chinese-draft.jsonl",
    ]
    seeds: list[TrajectorySeed] = []
    for path in paths:
        for row in _read_jsonl(path):
            seeds.append(
                TrajectorySeed(
                    seed_id=str(row["case_id"]),
                    query=str(row["question"]),
                    language=str(row.get("language", "en")),
                    variant="research",
                    expected_answer=str(row.get("expected_answer", "")),
                    evidence=list(row.get("evidence", [])),
                )
            )
    return seeds


def _retrieval_seeds(root: Path) -> list[TrajectorySeed]:
    return [
        TrajectorySeed(
            seed_id=str(row["query_id"]),
            query=str(row["query"]),
            language=str(row.get("language", "en")),
            variant="verification",
        )
        for row in _read_jsonl(
            root / "data/evaluation/retrieval_queries.jsonl"
        )
    ]


def _failure_seeds(root: Path) -> list[TrajectorySeed]:
    return [
        TrajectorySeed(
            seed_id=str(row["case_id"]),
            query=str(row["query"]),
            language="en",
            variant="research",
        )
        for row in _read_jsonl(
            root / "data/experiments/failure-analysis-v1/failure_cases.jsonl"
        )
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strip_code_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped
