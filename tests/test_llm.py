import httpx
import pytest

from app.core.llm import (
    DeepSeekClient,
    DeepSeekSettings,
    LLMAuthenticationError,
    LLMBadResponseError,
    LLMRateLimitedError,
    LLMTimeoutError,
    PLAIN_SYSTEM_PROMPT,
    SHARED_ANSWER_REQUIREMENTS,
    SYSTEM_PROMPT,
)


def client_for(handler) -> DeepSeekClient:
    return DeepSeekClient(
        DeepSeekSettings(api_key="test-only-key", model="deepseek-chat"),
        transport=httpx.MockTransport(handler),
    )


def test_deepseek_usage_comes_from_real_response_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-only-key"
        return httpx.Response(
            200,
            json={
                "model": "deepseek-chat",
                "choices": [
                    {"message": {"content": "答案 [doc-024]"}}
                ],
                "usage": {
                    "prompt_tokens": 101,
                    "completion_tokens": 12,
                    "total_tokens": 113,
                },
            },
        )

    result = client_for(handler).generate(
        query="问题", context="[DOC doc-024]\nContent:\n证据"
    )
    assert result.usage.prompt_tokens == 101
    assert result.usage.completion_tokens == 12
    assert result.usage.total_tokens == 113


def test_plain_deepseek_control_has_no_retrieved_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["temperature"] == 0
        assert "RETRIEVED CONTEXT" not in payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "model": "deepseek-chat",
                "choices": [{"message": {"content": "普通回答"}}],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 3,
                    "total_tokens": 12,
                },
            },
        )

    result = client_for(handler).generate_plain(query="问题")
    assert result.text == "普通回答"
    assert result.usage.total_tokens == 12
    assert result.system_prompt == PLAIN_SYSTEM_PROMPT
    assert result.user_prompt == "问题"


def test_system_prompt_marks_documents_as_untrusted_data() -> None:
    assert SHARED_ANSWER_REQUIREMENTS in SYSTEM_PROMPT
    assert SHARED_ANSWER_REQUIREMENTS in PLAIN_SYSTEM_PROMPT
    assert "untrusted data" in SYSTEM_PROMPT
    assert "Never create or cite an ID" in SYSTEM_PROMPT
    assert "INSUFFICIENT_EVIDENCE:" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, LLMAuthenticationError),
        (429, LLMRateLimitedError),
        (500, LLMBadResponseError),
    ],
)
def test_deepseek_maps_http_failures(status, error_type) -> None:
    client = client_for(
        lambda _: httpx.Response(status, json={"error": "redacted"})
    )
    with pytest.raises(error_type):
        client.generate(query="q", context="context")


def test_deepseek_rejects_invalid_response() -> None:
    client = client_for(lambda _: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMBadResponseError):
        client.generate(query="q", context="context")


def test_deepseek_maps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(LLMTimeoutError):
        client_for(handler).generate(query="q", context="context")
