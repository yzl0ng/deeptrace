#!/usr/bin/env bash
set -uo pipefail

task_root="${DEEPTRACE_TASK_ROOT:-/data/${USER}/deeptrace-r1}"
task_project="${DEEPTRACE_PROJECT_ROOT:-${task_root}/project}"
task_python="${DEEPTRACE_PYTHON:-${task_root}/runtime/py310-cu128/bin/python}"
task_adapter="${task_root}/adapters/sft-trajectory-500-epoch1-v2"
task_cases="${task_root}/datasets/sft-trajectory-500-v2/dev-eval.jsonl"
task_output="${task_root}/datasets/sft-trajectory-500-v2/dev-eval-results-v2"
task_status="${task_output}/status.tsv"
task_log="${task_output}/evaluation.log"
task_snapshot="${task_root}/datasets/sft-trajectory-500-v2/experiment-snapshot.json"

mkdir -p "${task_output}" "${task_root}/locks"
exec 9>"${task_root}/locks/sft-dev-evaluation-v2.lock"
if ! flock -n 9; then
    printf '%s\talready_running\n' "$(date --iso-8601=seconds)"
    exit 3
fi

cd "${task_project}" || exit 4
export PYTHONPATH="${task_project}"
printf '%s\twaiting_for_gpu\n' "$(date --iso-8601=seconds)" >"${task_status}"

while ! "${task_python}" scripts/select_idle_gpus.py \
    --count 1 \
    --max-used-memory-mib 2048 \
    --allow-compute-processes \
    >/dev/null 2>&1; do
    sleep 30
done

printf '%s\tevaluating\n' "$(date --iso-8601=seconds)" >"${task_status}"
"${task_python}" scripts/evaluate_lora_tool_format.py \
    --base-model "${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218" \
    --adapter "${task_adapter}" \
    --test-file "${task_cases}" \
    --output-dir "${task_output}" \
    --max-new-tokens 1024 \
    --max-used-memory-mib 2048 \
    --allow-compute-processes \
    >"${task_log}" 2>&1
eval_exit=$?
if [[ ${eval_exit} -ne 0 ]]; then
    printf '%s\tfailed_exit_%s\n' \
        "$(date --iso-8601=seconds)" "${eval_exit}" >"${task_status}"
    exit "${eval_exit}"
fi

"${task_python}" scripts/build_sft_experiment_snapshot.py \
    --trajectory-audit "${task_root}/datasets/trajectory-500-v2/quality-audit-v1/report.json" \
    --dataset-manifest "${task_root}/datasets/sft-trajectory-500-v2/manifest.json" \
    --dataset-validation "${task_root}/datasets/sft-trajectory-500-v2/validation-report.json" \
    --status "${task_root}/checkpoints/sft-trajectory-500-epoch1-v2/status.tsv" \
    --train-log "${task_root}/checkpoints/sft-trajectory-500-epoch1-v2/train.log" \
    --checkpoint-root "${task_root}/checkpoints/sft-trajectory-500-epoch1-v2" \
    --adapter-dir "${task_adapter}" \
    --dev-evaluation "${task_output}/summary.json" \
    --output "${task_snapshot}" \
    >>"${task_log}" 2>&1

printf '%s\tcompleted\t%s\n' \
    "$(date --iso-8601=seconds)" "${task_output}/summary.json" >"${task_status}"
