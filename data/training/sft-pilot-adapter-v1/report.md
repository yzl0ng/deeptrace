# Phase 5 LoRA adapter and resume validation v1

- Status: `succeeded`
- Exported adapter tensors: 504
- Adapter parameters: 43,646,976
- Adapter weights: 87,361,592 bytes (about 84 MiB)
- Full FSDP source checkpoint: about 16.47 GB
- PEFT load and bounded generation: succeeded on dynamically selected GPU 4
- Resume from step 1 to step 2: succeeded on dynamically selected GPUs 4 and 5

The exporter merged the two veRL FSDP2 shards in CPU memory, selected only
LoRA parameters, normalized PEFT keys, and wrote `adapter_config.json` plus
`adapter_model.safetensors`. It did not write another copy of the base model.
PEFT loaded the pinned Qwen3-8B base and the exported `default` adapter, then
completed an eight-token generation. GPU 4 returned to 11 MiB afterward.

The resume smoke loaded each rank's model shard plus RNG and learning-rate
scheduler state, restored the StatefulDataLoader, consumed the next batch, and
completed optimizer step 2. Step-2 train loss was 1.5602, validation loss was
1.3273, and the training step took 5.49 seconds. The source smoke deliberately
did not save optimizer state, so optimizer moments were reinitialized. This
proves model/data-state recovery, not exact optimizer-continuous training.
