from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agentic.config import DeepTraceV2Config


def test_v2_config_is_disabled_and_isolated_by_default() -> None:
    config = DeepTraceV2Config.from_env({})

    assert config.config_version == "v2"
    assert config.enabled is False
    assert config.storage_namespace == "agentic_v2"
    assert config.experiment_namespace == "deeptrace-v2"
    assert config.search_provider == "local"
    assert config.workflow == "baseline"
    assert config.max_query_rewrites == 1


def test_v2_config_reads_only_deeptrace_prefixed_values() -> None:
    config = DeepTraceV2Config.from_env(
        {
            "DEEPTRACE_ENABLED": "yes",
            "DEEPTRACE_MAX_AGENT_STEPS": "12",
            "DEEPTRACE_MAX_API_COST": "1.25",
            "DEEPTRACE_SEARCH_PROVIDER": "brave",
            "DEEPTRACE_WORKFLOW": "supervisor",
            "DEEPTRACE_MAX_QUERY_REWRITES": "2",
            "DEEPSEEK_API_KEY": "must-not-be-read",
        }
    )

    assert config.enabled is True
    assert config.max_agent_steps == 12
    assert config.max_api_cost == 1.25
    assert config.search_provider == "brave"
    assert config.workflow == "supervisor"
    assert config.max_query_rewrites == 2
    assert "api_key" not in config.model_dump()


@pytest.mark.parametrize("value", ["sometimes", "", "2"])
def test_v2_config_rejects_ambiguous_boolean_values(value: str) -> None:
    with pytest.raises(ValueError, match="boolean environment values"):
        DeepTraceV2Config.from_env({"DEEPTRACE_ENABLED": value})


def test_v2_config_rejects_invalid_budget() -> None:
    with pytest.raises(ValidationError):
        DeepTraceV2Config(max_agent_steps=0)


def test_v2_config_rejects_unknown_search_provider() -> None:
    with pytest.raises(ValidationError):
        DeepTraceV2Config(search_provider="unreviewed")


def test_v2_config_rejects_unknown_workflow() -> None:
    with pytest.raises(ValidationError):
        DeepTraceV2Config(workflow="unreviewed")
