#!/usr/bin/env bash
set -uo pipefail

task_root="${DEEPTRACE_TASK_ROOT:-/data/${USER}/deeptrace-r1}"
task_project="${DEEPTRACE_PROJECT_ROOT:-${task_root}/project}"
task_python="${DEEPTRACE_PYTHON:-${task_root}/runtime/py310-cu128/bin/python}"
task_verl="${DEEPTRACE_VERL_ROOT:-${task_root}/sources/verl-f9c855f7}"
task_run_name="${TASK_RUN_NAME:-sft-trajectory-500-epoch1-v2}"
task_dataset_version="${TASK_DATASET_VERSION:-sft-trajectory-500-v2}"
task_total_steps="${TASK_TOTAL_STEPS:-210}"
task_total_epochs="${TASK_TOTAL_EPOCHS:-1}"
task_max_length="${TASK_MAX_LENGTH:-1024}"
task_max_used_memory_mib="${TASK_MAX_USED_MEMORY_MIB:-2048}"
task_eval_max_new_tokens="${TASK_EVAL_MAX_NEW_TOKENS:-1024}"
task_eval_mode="${TASK_EVAL_MODE:-full_trajectory}"
task_tool_loop_max_cases="${TASK_TOOL_LOOP_MAX_CASES:-47}"
task_checkpoint_frequency="${TASK_CHECKPOINT_FREQUENCY:-210}"
task_min_free_disk_gib="${TASK_MIN_FREE_DISK_GIB:-40}"
task_dataset="${task_root}/datasets/${task_dataset_version}"
task_tool_loop_cases="${TASK_TOOL_LOOP_CASES:-${task_dataset}/tool-loop-dev.jsonl}"
task_output="${task_root}/checkpoints/${task_run_name}"
task_adapter="${task_root}/adapters/${task_run_name}"
task_dev_cases="${task_dataset}/dev-eval.jsonl"
task_dev_output="${task_dataset}/dev-eval-results"
task_snapshot="${task_dataset}/experiment-snapshot.json"
task_lock_dir="${task_root}/locks"
task_status="${task_output}/status.tsv"

mkdir -p "${task_output}" "${task_lock_dir}"
exec 9>"${task_lock_dir}/${task_run_name}.lock"
if ! flock -n 9; then
    printf '%s\talready_running\n' "$(date --iso-8601=seconds)"
    exit 3
fi

cd "${task_project}" || exit 4
export PYTHONPATH="${task_project}:${task_verl}"

write_snapshot() {
    "${task_python}" scripts/build_sft_experiment_snapshot.py \
        --trajectory-audit "${task_root}/datasets/trajectory-500-v2/quality-audit-v1/report.json" \
        --dataset-manifest "${task_dataset}/manifest.json" \
        --dataset-validation "${task_dataset}/validation-report.json" \
        --status "${task_status}" \
        --train-log "${task_output}/train.log" \
        --checkpoint-root "${task_output}" \
        --adapter-dir "${task_adapter}" \
        --dev-evaluation "${task_dev_output}/summary.json" \
        --output "${task_snapshot}"
}

trap 'write_snapshot >/dev/null 2>&1 || true' EXIT

printf '%s\twaiting_for_two_idle_gpus\n' \
    "$(date --iso-8601=seconds)" >"${task_status}"
write_snapshot

while true; do
    free_disk_gib="$(
        df -BG --output=avail "${task_root}" |
            tail -n 1 |
            tr -dc '0-9'
    )"
    if [[ -z "${free_disk_gib}" ]] ||
        [[ "${free_disk_gib}" -lt "${task_min_free_disk_gib}" ]]; then
        printf '%s\twaiting_for_disk\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" \
            "${free_disk_gib:-unknown}" \
            "${task_min_free_disk_gib}" >"${task_status}"
        sleep 30
        continue
    fi
    selected_gpus="$(
        "${task_python}" scripts/select_idle_gpus.py \
            --count 2 \
            --max-used-memory-mib "${task_max_used_memory_mib}" \
            --allow-compute-processes \
            2>/dev/null
    )"
    selector_exit=$?
    if [[ ${selector_exit} -eq 0 ]]; then
        resume_args=(trainer.resume_mode=disable)
        latest_before_run="$(
            find "${task_output}" \
                -maxdepth 1 \
                -type d \
                -name 'global_step_*' \
                -printf '%p\n' |
                sort -V |
                tail -n 1
        )"
        if [[ -n "${latest_before_run}" ]]; then
            resume_step="${latest_before_run##*_}"
            if [[ "${resume_step}" -lt "${task_total_steps}" ]]; then
                resume_args=(
                    trainer.resume_mode=resume_path
                    trainer.resume_from_path="${latest_before_run}"
                )
                printf '%s\tresuming\t%s\t%s\n' \
                    "$(date --iso-8601=seconds)" \
                    "${selected_gpus}" \
                    "${latest_before_run}" >"${task_status}"
            else
                printf '%s\tpostprocessing_existing_step_%s\t%s\n' \
                    "$(date --iso-8601=seconds)" \
                    "${resume_step}" \
                    "${selected_gpus}" >"${task_status}"
            fi
        else
            resume_step=0
            printf '%s\tlaunching\t%s\n' \
                "$(date --iso-8601=seconds)" \
                "${selected_gpus}" >"${task_status}"
        fi
        export CUDA_VISIBLE_DEVICES="${selected_gpus}"

        if [[ "${resume_step}" -lt "${task_total_steps}" ]]; then
            printf 'SELECTED_GPUS=%s\n' "${selected_gpus}" \
                >"${task_output}/train.log"
            printf '%s\ttraining\t%s\t%s\t%s\n' \
                "$(date --iso-8601=seconds)" \
                "${selected_gpus}" \
                "${resume_step}" \
                "${task_total_steps}" >"${task_status}"
            "${task_python}" -m torch.distributed.run \
            --standalone \
            --nproc-per-node=2 \
            -m verl.trainer.fsdp_sft_trainer \
            data.train_files="${task_dataset}/train.parquet" \
            data.val_files="${task_dataset}/validation.parquet" \
            data.train_batch_size=2 \
            data.micro_batch_size_per_gpu=1 \
            data.max_length="${task_max_length}" \
            data.multiturn.enable=true \
            +data.ignore_input_ids_mismatch=true \
            model.partial_pretrain="${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218" \
            model.fsdp_config.model_dtype=bf16 \
            model.enable_gradient_checkpointing=true \
            model.lora_rank=16 \
            model.lora_alpha=32 \
            model.strategy=fsdp2 \
            trainer.default_local_dir="${task_output}" \
            trainer.total_epochs="${task_total_epochs}" \
            trainer.total_training_steps="${task_total_steps}" \
            trainer.n_gpus_per_node=2 \
            trainer.save_freq="${task_checkpoint_frequency}" \
            trainer.test_freq="${task_checkpoint_frequency}" \
            trainer.max_ckpt_to_keep=2 \
            "${resume_args[@]}" \
            'trainer.checkpoint.save_contents=[model,optimizer,extra]' \
            'trainer.logger=[console]' \
            2>&1 | tee -a "${task_output}/train.log"
            train_exit=${PIPESTATUS[0]}
            if [[ ${train_exit} -ne 0 ]]; then
                printf '%s\tfailed_exit_%s\t%s\n' \
                    "$(date --iso-8601=seconds)" \
                    "${train_exit}" \
                    "${selected_gpus}" >"${task_status}"
                exit "${train_exit}"
            fi
        fi

        latest_checkpoint="$(
            find "${task_output}" \
                -maxdepth 1 \
                -type d \
                -name 'global_step_*' \
                -printf '%p\n' |
                sort -V |
                tail -n 1
        )"
        if [[ -z "${latest_checkpoint}" ]]; then
            printf '%s\tfailed_missing_final_checkpoint\n' \
                "$(date --iso-8601=seconds)" >"${task_status}"
            exit 5
        fi

        printf '%s\texporting_adapter\t%s\n' \
            "$(date --iso-8601=seconds)" \
            "${latest_checkpoint}" >"${task_status}"
        "${task_python}" scripts/export_lora_adapter.py \
            --checkpoint-dir "${latest_checkpoint}" \
            --output-dir "${task_adapter}" \
            --base-model "${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218" \
            --lora-alpha 32 \
            >>"${task_output}/postprocess.log" 2>&1
        export_exit=$?
        if [[ ${export_exit} -ne 0 ]]; then
            printf '%s\tfailed_adapter_export_%s\n' \
                "$(date --iso-8601=seconds)" \
                "${export_exit}" >"${task_status}"
            exit "${export_exit}"
        fi

        while ! "${task_python}" scripts/select_idle_gpus.py \
            --count 1 \
            --max-used-memory-mib "${task_max_used_memory_mib}" \
            --allow-compute-processes \
            >/dev/null 2>&1; do
            printf '%s\twaiting_for_postprocess_gpu\n' \
                "$(date --iso-8601=seconds)" >"${task_status}"
            sleep 30
        done

        printf '%s\tvalidating_and_evaluating_adapter\n' \
            "$(date --iso-8601=seconds)" >"${task_status}"
        "${task_python}" scripts/validate_lora_adapter.py \
            --base-model "${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218" \
            --adapter "${task_adapter}" \
            --output "${task_adapter}/load-validation.json" \
            --max-used-memory-mib "${task_max_used_memory_mib}" \
            --allow-compute-processes \
            >>"${task_output}/postprocess.log" 2>&1
        validate_exit=$?
        if [[ ${validate_exit} -ne 0 ]]; then
            printf '%s\tfailed_adapter_validation_%s\n' \
                "$(date --iso-8601=seconds)" \
                "${validate_exit}" >"${task_status}"
            exit "${validate_exit}"
        fi

        if [[ "${task_eval_mode}" == "tool_loop" ]]; then
            "${task_python}" scripts/evaluate_lora_tool_loop.py \
                --base-model "${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218" \
                --adapter "${task_adapter}" \
                --cases "${task_tool_loop_cases}" \
                --output-dir "${task_dev_output}" \
                --max-cases "${task_tool_loop_max_cases}" \
                --max-used-memory-mib "${task_max_used_memory_mib}" \
                --allow-compute-processes \
                >>"${task_output}/postprocess.log" 2>&1
        else
            "${task_python}" scripts/evaluate_lora_tool_format.py \
                --base-model "${task_root}/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218" \
                --adapter "${task_adapter}" \
                --test-file "${task_dev_cases}" \
                --output-dir "${task_dev_output}" \
                --max-new-tokens "${task_eval_max_new_tokens}" \
                --max-used-memory-mib "${task_max_used_memory_mib}" \
                --allow-compute-processes \
                >>"${task_output}/postprocess.log" 2>&1
        fi
        eval_exit=$?
        if [[ ${eval_exit} -ne 0 ]]; then
            printf '%s\tfailed_dev_evaluation_%s\n' \
                "$(date --iso-8601=seconds)" \
                "${eval_exit}" >"${task_status}"
            exit "${eval_exit}"
        fi

        printf '%s\tcompleted\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" \
            "${selected_gpus}" \
            "${latest_checkpoint}" >"${task_status}"
        exit 0
    fi
    sleep 30
done
