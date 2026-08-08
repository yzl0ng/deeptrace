# Phase 5 two-GPU SFT smoke v1

- Status: `succeeded`
- Dynamically selected GPUs: 4 and 5
- Model: pinned Qwen3-8B revision
- Trainer: veRL FSDP2, BF16, LoRA rank 16 / alpha 32
- Data: 89 train / 10 validation trajectories
- Completed optimizer steps: 1
- Train loss: 1.0762
- Validation loss: 1.3273
- Training-step time: 5.84 seconds

Both ranks loaded the five model shards, initialized NCCL 2.27.3, completed a
forward/backward/optimizer step, ran validation, and wrote rank-specific model
and extra-state files. The selected GPUs returned to 11 MiB and zero
utilization with no residual compute process.

The Qwen3 chat template does not produce identical token IDs when veRL formats
each turn separately versus formatting the whole conversation. The run used
veRL's explicit `ignore_input_ids_mismatch=true` compatibility option; every
sample retained non-zero assistant loss tokens.

The default checkpoint is about 16.47 GB because FSDP2 wrote two full model
shards. Before a longer run, validate adapter-only export and resume behavior.
This smoke proves execution, not convergence or an SFT quality gain.
