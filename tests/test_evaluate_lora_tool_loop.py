from scripts.evaluate_lora_tool_loop import _strip_code_fence


def test_tool_action_code_fence_is_removed() -> None:
    assert _strip_code_fence("```json\n{\"action\":\"search\"}\n```") == (
        '{"action":"search"}'
    )
