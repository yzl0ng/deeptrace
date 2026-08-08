from scripts.export_lora_adapter import (
    normalize_lora_key,
    target_module_from_lora_key,
)


def test_normalize_lora_key_matches_peft_adapter_format() -> None:
    name = (
        "base_model.model.model.layers.0.self_attn.q_proj."
        "lora_A.default.weight"
    )
    assert normalize_lora_key(name) == (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    )


def test_target_module_from_lora_key() -> None:
    name = (
        "base_model.model.model.layers.0.mlp.down_proj."
        "lora_B.default.weight"
    )
    assert target_module_from_lora_key(name) == "down_proj"
