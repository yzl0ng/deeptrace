from pathlib import Path

from scripts.build_sft_experiment_snapshot import (
    _checkpoint_status,
    _training_metrics,
    _training_status,
)


def test_training_log_and_status_are_parsed(tmp_path: Path) -> None:
    status = tmp_path / "status.tsv"
    status.write_text(
        "2026-07-29T17:00:00+08:00\tlaunching\t4,5\n",
        encoding="utf-8",
    )
    log = tmp_path / "train.log"
    log.write_text(
        "SELECTED_GPUS=4,5\n"
        "step:1 - train/loss:1.5 - train/time(s):3.0\n"
        "step:2 - train/loss:1.2 - train/time(s):3.0\n"
        "step:2 - val/loss:1.1\n",
        encoding="utf-8",
    )

    assert _training_status(status)["state"] == "launching"
    metrics = _training_metrics(log)
    assert metrics["selected_gpus"] == "4,5"
    assert metrics["latest_step"] == 2
    assert metrics["latest_train_loss"] == 1.2
    assert metrics["latest_validation_loss"] == 1.1


def test_checkpoint_requires_all_resume_components(tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_step_210"
    checkpoint.mkdir()
    for rank in (0, 1):
        (checkpoint / f"model_world_size_2_rank_{rank}.pt").write_bytes(b"m")
        (checkpoint / f"optim_world_size_2_rank_{rank}.pt").write_bytes(b"o")
        (checkpoint / f"extra_state_world_size_2_rank_{rank}.pt").write_bytes(
            b"e"
        )
    (checkpoint / "data.pt").write_bytes(b"d")
    (checkpoint / "huggingface").mkdir()

    result = _checkpoint_status(tmp_path)

    assert result["state"] == "complete"
    assert result["latest_step"] == 210
