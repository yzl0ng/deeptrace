from scripts.assess_tool_loop_evaluation import assess


def test_assessment_separates_answer_and_protocol_gates() -> None:
    result = assess(
        {
            "modes": {
                "sft": {
                    "metrics": {
                        "answer_exact_rate": 0.60,
                        "completion_rate": 0.80,
                        "unknown_evidence_id_attempts": 2,
                        "invalid_action_attempts": 0,
                        "mean_supporting_evidence_recall": 0.70,
                    }
                }
            }
        },
        answer_target=0.50,
    )

    assert result["answer_target_passed"] is True
    assert result["production_quality_gate_passed"] is False
    assert result["decision"] == (
        "answer_target_passed_protocol_gate_failed"
    )
