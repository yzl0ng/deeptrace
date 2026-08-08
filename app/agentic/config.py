from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

ENV_PREFIX = "DEEPTRACE_"


class DeepTraceV2Config(BaseModel):
    """Phase-0 configuration contract for the isolated v2 namespace.

    This model deliberately contains no API keys or provider credentials. Later
    phases can compose secret-bearing runtime settings without serializing them
    into experiment manifests or server-audit artifacts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: str = "v2"
    enabled: bool = False
    storage_namespace: str = Field(
        default="agentic_v2",
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    experiment_namespace: str = Field(
        default="deeptrace-v2",
        min_length=1,
        pattern=r"^[a-z][a-z0-9-]*$",
    )
    search_provider: str = Field(
        default="local",
        pattern=r"^(local|brave)$",
    )
    workflow: str = Field(
        default="baseline",
        pattern=r"^(baseline|supervisor)$",
    )
    max_query_rewrites: int = Field(default=1, ge=0, le=3)
    max_wall_time_seconds: int = Field(default=900, ge=1)
    max_agent_steps: int = Field(default=40, ge=1)
    max_search_calls: int = Field(default=20, ge=0)
    max_page_reads: int = Field(default=20, ge=0)
    max_total_tokens: int = Field(default=120_000, ge=1)
    max_parallel_research_units: int = Field(default=4, ge=1)
    max_context_tokens: int = Field(default=32_000, ge=1)
    max_api_cost: float = Field(default=5.0, ge=0)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> DeepTraceV2Config:
        source = os.environ if environ is None else environ
        field_parsers = {
            "enabled": _parse_bool,
            "storage_namespace": str,
            "experiment_namespace": str,
            "search_provider": str,
            "workflow": str,
            "max_query_rewrites": int,
            "max_wall_time_seconds": int,
            "max_agent_steps": int,
            "max_search_calls": int,
            "max_page_reads": int,
            "max_total_tokens": int,
            "max_parallel_research_units": int,
            "max_context_tokens": int,
            "max_api_cost": float,
        }
        values: dict[str, object] = {}
        for field_name, parser in field_parsers.items():
            env_name = f"{ENV_PREFIX}{field_name.upper()}"
            raw_value = source.get(env_name)
            if raw_value is not None:
                values[field_name] = parser(raw_value)
        return cls.model_validate(values)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "boolean environment values must be one of "
        "1/0, true/false, yes/no, or on/off"
    )
