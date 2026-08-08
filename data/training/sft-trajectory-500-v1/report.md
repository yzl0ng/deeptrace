# Phase 5 controlled 500-trajectory SFT v1

- Status: `succeeded`
- Data: 431 train / 48 validation
- GPUs: dynamically selected physical devices 4 and 5
- Model: pinned Qwen3-8B
- Training: FSDP2, BF16, LoRA rank 16 / alpha 32
- Optimizer steps: 10
- Total train-step time: 32.09 seconds
- Final validation loss: 1.4181

Ten train losses were recorded, ranging from 1.1020 to 1.5487 across different
mini-batches. This short run is a pipeline and checkpoint validation, not a
convergence curve. The final checkpoint contains model, optimizer, RNG,
scheduler, and dataloader state for both ranks.

The step-10 checkpoint exported to a PEFT adapter with 504 tensors and
43,646,976 parameters. Its 87,361,592-byte safetensors loaded with the pinned
base model and completed bounded generation. A separate resume run loaded
model and optimizer shards plus RNG, scheduler, and dataloader state, then
completed step 11 and validation. GPU 4 and 5 returned to 11 MiB afterward.

Human semantic review and a fixed Base-versus-SFT task evaluation remain
required before claiming a quality gain.
