from __future__ import annotations

import argparse
import json
from pathlib import Path


def _percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate veRL multi-turn SFT Parquet files."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True
    )
    config = OmegaConf.create(
        {
            "max_length": args.max_length,
            "truncation": "error",
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "ignore_input_ids_mismatch": True,
        }
    )
    split_results = {}
    for split, path in (
        ("train", args.train),
        ("validation", args.validation),
    ):
        dataset = MultiTurnSFTDataset(path, tokenizer, config)
        sequence_tokens: list[int] = []
        loss_tokens: list[int] = []
        for item in dataset:
            sequence_tokens.append(int(item["attention_mask"].sum().item()))
            loss_tokens.append(int(item["loss_mask"].sum().item()))
        if not loss_tokens or min(loss_tokens) < 1:
            raise ValueError(f"{split} contains a record with zero loss tokens")
        split_results[split] = {
            "records": len(dataset),
            "sequence_tokens": {
                "minimum": min(sequence_tokens),
                "median": _percentile(sequence_tokens, 0.5),
                "p95": _percentile(sequence_tokens, 0.95),
                "maximum": max(sequence_tokens),
            },
            "assistant_loss_tokens": {
                "minimum": min(loss_tokens),
                "median": _percentile(loss_tokens, 0.5),
                "p95": _percentile(loss_tokens, 0.95),
                "maximum": max(loss_tokens),
                "zero": sum(value == 0 for value in loss_tokens),
            },
        }

    payload = {
        "validation_version": "sft-dataset-validation-v1",
        "status": "succeeded",
        "model": args.model,
        "max_length": args.max_length,
        "ignore_input_ids_mismatch": True,
        "splits": split_results,
        "truth_boundary": (
            "This proves tokenizer and veRL dataset compatibility plus nonzero "
            "assistant loss masks. It does not prove semantic quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
