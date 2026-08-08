#!/usr/bin/env bash
set -euo pipefail

task_root="${DEEPTRACE_TASK_ROOT:-/data/$USER/deeptrace-r1}"
python_bin="${STUDENT_PYTHON:-${task_root}/runtime/py310-cu128/bin/python}"
base_model="${STUDENT_BASE_MODEL:-${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218}"
adapter="${STUDENT_ADAPTER:-${task_root}/adapters/sft-action-trajectory-500-epoch2-v1}"
port="${STUDENT_PORT:-8000}"

if [[ -z "${STUDENT_API_TOKEN:-}" ]]; then
  echo "STUDENT_API_TOKEN is required" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" || ! -f "${adapter}/adapter_model.safetensors" ]]; then
  echo "Pinned Student runtime or adapter is unavailable" >&2
  exit 3
fi

gpu_index="${STUDENT_GPU_INDEX:-}"
if [[ -z "${gpu_index}" ]]; then
  gpu_index="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2 + 0 < 1024 {gsub(/ /, "", $1); print $1; exit}')"
fi
if [[ -z "${gpu_index}" ]]; then
  echo "No GPU with less than 1 GiB allocated memory is currently available" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="${gpu_index}"
exec "${python_bin}" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "${port}" \
  --model "${base_model}" \
  --served-model-name qwen3-base \
  --enable-lora \
  --lora-modules "deeptrace-student=${adapter}" \
  --api-key "${STUDENT_API_TOKEN}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --disable-log-stats
