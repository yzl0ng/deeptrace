# Training and evaluation

## What is trained

The Student is Qwen3-8B with a LoRA adapter. SFT teaches action selection and protocol completion; it does not train BM25, the page reader, the evidence store, budgets, checkpoints or citation allowlists.

| Component | Learned? | Responsibility |
|---|---:|---|
| Qwen3-8B LoRA policy | yes | choose the next typed action and stopping point |
| DeepSeek teacher | no | synthesize candidate trajectories |
| search/read tools | no | return observations |
| action parser | no | validate JSON schema |
| evidence gates | no | enforce runtime invariants |
| checkpoint/budget | no | reliability and resource control |

## Data pipeline

1. Fetch pinned multi-hop question seeds.
2. Ask the teacher for complete action trajectories.
3. Reject schema errors, unknown Evidence IDs, unsupported answers and incomplete loops.
4. Freeze manifests, source revisions and content hashes.
5. Split accepted records into train/validation before training.
6. Convert conversations to veRL multi-turn Parquet.

The recorded project run generated 500 candidates, accepted 479, rejected 21, and split accepted records into 431 train and 48 validation examples. Public manifests and stage reports are under `data/training/`; raw generated trajectories are omitted because they can contain provider output and are reproducible from the scripts.

## GPU directory contract

`run_sft_epoch_when_gpus_free.sh` expects the following logical layout under `DEEPTRACE_TASK_ROOT`:

```text
$DEEPTRACE_TASK_ROOT/
├── project/                 # this repository, or override PROJECT_ROOT
├── runtime/py310-cu128/     # or override DEEPTRACE_PYTHON
├── sources/verl-f9c855f7/  # or override DEEPTRACE_VERL_ROOT
├── models/Qwen--Qwen3-8B/<revision>/
├── datasets/<dataset-version>/
├── checkpoints/<run-name>/
├── adapters/<run-name>/
└── locks/
```

Important environment variables:

| Variable | Meaning |
|---|---|
| `DEEPTRACE_TASK_ROOT` | models, datasets and output root |
| `DEEPTRACE_PROJECT_ROOT` | repository checkout |
| `DEEPTRACE_PYTHON` | training Python executable |
| `DEEPTRACE_VERL_ROOT` | pinned veRL source |
| `TASK_RUN_NAME` | checkpoint/adapter run name |
| `TASK_DATASET_VERSION` | dataset directory name |
| `TASK_TOTAL_STEPS` | optimizer-step ceiling |
| `TASK_TOTAL_EPOCHS` | epoch ceiling |
| `TASK_MAX_USED_MEMORY_MIB` | GPU idle threshold |
| `TASK_CHECKPOINT_FREQUENCY` | save/evaluation interval |

The scheduler selects any two eligible GPUs. It waits when disk or GPU constraints are not met, writes status snapshots, and resumes an incomplete `global_step_*` checkpoint.

## Evaluation protocol

The frozen final set has 90 questions: 30 each from HotpotQA, 2WikiMultiHopQA and MuSiQue. Each case includes a controlled candidate evidence pool. Base and SFT receive the same question, evidence, action schema, maximum steps and generation budget.

Metrics:

- normalized answer EM and token F1;
- successful answer action within the step budget;
- Gold evidence recall from read/cited IDs;
- invalid action attempts;
- unknown Evidence ID attempts;
- insufficient-evidence attempts;
- recovered protocol errors;
- final protocol failure.

Do not tune prompts, gates, decoding or checkpoints after reading the final-test result. Use a separate development split for those changes.

## Reproduction commands

All Python scripts expose `--help`. A typical sequence is:

```bash
python scripts/prepare_trajectory_seed_pool.py --help
python scripts/run_trajectory_pilot.py --help
python scripts/audit_trajectory_quality.py --help
python scripts/prepare_action_sft_dataset.py --help
python scripts/validate_sft_dataset.py --help
python scripts/evaluate_lora_tool_loop.py --help
python scripts/assess_tool_loop_evaluation.py --help
```

Every published experiment should archive:

- git commit and dirty-state marker;
- base model ID and revision;
- adapter hash;
- dataset manifest and SHA-256;
- prompts/action schema;
- generation settings and step budget;
- hardware/runtime versions;
- predictions, failures, metrics and exit status.

Large predictions, checkpoints and adapters should be attached to a release or model registry, not committed to Git.
