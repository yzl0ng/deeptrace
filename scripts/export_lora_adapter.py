from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def normalize_lora_key(name: str) -> str:
    return name.replace(".default.weight", ".weight")


def target_module_from_lora_key(name: str) -> str:
    parts = normalize_lora_key(name).split(".")
    if len(parts) < 3:
        raise ValueError(f"unexpected LoRA parameter name: {name}")
    return parts[-3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_adapter(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    base_model: str,
    lora_alpha: int,
) -> dict[str, Any]:
    from peft import LoraConfig
    from safetensors.torch import save_file
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(checkpoint_dir),
        target_dir=str(output_dir),
        hf_model_config_path=str(checkpoint_dir / "huggingface"),
    )
    merger = FSDPModelMerger(config)
    world_size = merger._get_world_size()
    rank_zero_state = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(
        rank_zero_state, world_size
    )
    del rank_zero_state
    total_shards, mesh_shape = merger._calculate_shard_configuration(
        mesh, mesh_dim_names
    )
    merged_state = merger._load_and_merge_state_dicts(
        world_size, total_shards, mesh_shape, mesh_dim_names
    )

    adapter_state = {
        normalize_lora_key(name): tensor.detach().cpu().contiguous()
        for name, tensor in merged_state.items()
        if "lora_" in name
    }
    del merged_state
    if not adapter_state:
        raise ValueError(f"no LoRA parameters found in {checkpoint_dir}")

    target_modules = sorted(
        {target_module_from_lora_key(name) for name in adapter_state}
    )
    ranks = {
        min(tensor.shape)
        for name, tensor in adapter_state.items()
        if name.endswith("lora_A.weight") or name.endswith("lora_B.weight")
    }
    if len(ranks) != 1:
        raise ValueError(f"inconsistent LoRA ranks: {sorted(ranks)}")
    lora_rank = ranks.pop()

    adapter_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    adapter_config.base_model_name_or_path = base_model
    adapter_config.save_pretrained(output_dir)

    weights_path = output_dir / "adapter_model.safetensors"
    tensor_count = len(adapter_state)
    save_file(adapter_state, weights_path, metadata={"format": "pt"})
    del adapter_state

    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            artifacts.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "export_version": "lora-adapter-export-v1",
        "status": "succeeded",
        "source_checkpoint": str(checkpoint_dir),
        "base_model": base_model,
        "world_size": world_size,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "tensor_count": tensor_count,
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge veRL FSDP checkpoint shards in CPU memory and export only "
            "the PEFT LoRA adapter."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    args = parser.parse_args()

    if args.lora_alpha < 1:
        parser.error("--lora-alpha must be at least 1")
    manifest = export_adapter(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        lora_alpha=args.lora_alpha,
    )
    print(
        f"exported {manifest['tensor_count']} LoRA tensors "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
