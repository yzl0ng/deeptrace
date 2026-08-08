#!/usr/bin/env bash
set -uo pipefail

task_root="${DEEPTRACE_TASK_ROOT:-/data/${USER}/deeptrace-r1}"
task_project="${DEEPTRACE_PROJECT_ROOT:-${task_root}/project}"
task_python="${DEEPTRACE_PYTHON:-${task_root}/runtime/py310-cu128/bin/python}"
task_v3_run="${task_root}/checkpoints/sft-trajectory-500-epoch2-v3"
task_v3_adapter="${task_root}/adapters/sft-trajectory-500-epoch2-v3"
task_v3_dataset="${task_root}/datasets/sft-trajectory-500-v3"
task_format_output="${task_v3_dataset}/dev-eval-results-1024"
task_tool_output="${task_v3_dataset}/tool-loop-dev-results-smoke10"
task_action_dataset="${task_root}/datasets/sft-action-trajectory-500-v1"
task_status="${task_root}/checkpoints/post-v3-quality-pipeline/status.tsv"

mkdir -p "$(dirname "${task_status}")" "${task_root}/locks"
exec 9>"${task_root}/locks/post-v3-quality-pipeline.lock"
if ! flock -n 9; then
    printf '%s\talready_running\n' "$(date --iso-8601=seconds)"
    exit 3
fi

cd "${task_project}" || exit 4
export PYTHONPATH="${task_project}:${task_root}/sources/verl-f9c855f7"

while true; do
    v3_state="$(
        awk -F '\t' 'NF >= 2 {print $2}' "${task_v3_run}/status.tsv" \
            2>/dev/null |
            tail -n 1
    )"
    if [[ "${v3_state}" == "completed" ]]; then
        break
    fi
    if [[ "${v3_state}" == failed_* ]]; then
        printf '%s\tblocked_v3_%s\n' \
            "$(date --iso-8601=seconds)" "${v3_state}" >"${task_status}"
        exit 5
    fi
    printf '%s\twaiting_for_v3\t%s\n' \
        "$(date --iso-8601=seconds)" \
        "${v3_state:-unknown}" >"${task_status}"
    sleep 30
done

while ! "${task_python}" scripts/select_idle_gpus.py \
    --count 1 \
    --max-used-memory-mib 6144 \
    --allow-compute-processes \
    >/dev/null 2>&1; do
    printf '%s\twaiting_for_format_eval_gpu\n' \
        "$(date --iso-8601=seconds)" >"${task_status}"
    sleep 30
done

if [[ ! -f "${task_format_output}/summary.json" ]]; then
    printf '%s\tevaluating_v3_format_1024\n' \
        "$(date --iso-8601=seconds)" >"${task_status}"
    "${task_python}" scripts/evaluate_lora_tool_format.py \
        --base-model "${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218" \
        --adapter "${task_v3_adapter}" \
        --test-file "${task_v3_dataset}/dev-eval.jsonl" \
        --output-dir "${task_format_output}" \
        --max-new-tokens 1024 \
        --max-used-memory-mib 6144 \
        --allow-compute-processes \
        >"${task_format_output}.log" 2>&1
    format_exit=$?
    if [[ ${format_exit} -ne 0 ]]; then
        printf '%s\tfailed_format_eval_%s\n' \
            "$(date --iso-8601=seconds)" "${format_exit}" >"${task_status}"
        exit "${format_exit}"
    fi
fi

while ! "${task_python}" scripts/select_idle_gpus.py \
    --count 1 \
    --max-used-memory-mib 6144 \
    --allow-compute-processes \
    >/dev/null 2>&1; do
    printf '%s\twaiting_for_tool_loop_gpu\n' \
        "$(date --iso-8601=seconds)" >"${task_status}"
    sleep 30
done

printf '%s\tevaluating_v3_tool_loop_smoke10\n' \
    "$(date --iso-8601=seconds)" >"${task_status}"
"${task_python}" scripts/evaluate_lora_tool_loop.py \
    --base-model "${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218" \
    --adapter "${task_v3_adapter}" \
    --cases "${task_v3_dataset}/tool-loop-dev.jsonl" \
    --output-dir "${task_tool_output}" \
    --max-cases 10 \
    --max-steps 8 \
    --max-action-tokens 256 \
    --max-used-memory-mib 6144 \
    --allow-compute-processes \
    >"${task_tool_output}.log" 2>&1
tool_exit=$?
if [[ ${tool_exit} -ne 0 ]]; then
    printf '%s\tfailed_tool_loop_eval_%s\n' \
        "$(date --iso-8601=seconds)" "${tool_exit}" >"${task_status}"
    exit "${tool_exit}"
fi

if "${task_python}" -c \
    'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); raise SystemExit(0 if a["quality_gate"]["passed"] and b["comparison_gate"]["passed"] else 1)' \
    "${task_format_output}/summary.json" \
    "${task_tool_output}/summary.json"; then
    printf '%s\tcompleted_v3_passed_all_gates\n' \
        "$(date --iso-8601=seconds)" >"${task_status}"
    exit 0
fi

printf '%s\tlaunching_action_sft\n' \
    "$(date --iso-8601=seconds)" >"${task_status}"
export TASK_RUN_NAME="sft-action-trajectory-500-epoch2-v1"
export TASK_DATASET_VERSION="sft-action-trajectory-500-v1"
export TASK_TOTAL_STEPS=420
export TASK_TOTAL_EPOCHS=2
export TASK_MAX_LENGTH=1400
export TASK_MAX_USED_MEMORY_MIB=6144
export TASK_EVAL_MODE="tool_loop"
export TASK_TOOL_LOOP_CASES="${task_v3_dataset}/tool-loop-dev.jsonl"
export TASK_TOOL_LOOP_MAX_CASES=47
export TASK_CHECKPOINT_FREQUENCY=210
export TASK_MIN_FREE_DISK_GIB=40
exec bash scripts/run_sft_epoch_when_gpus_free.sh
