#!/bin/bash
# FaceScrub slot-attention hyperparameter search for the root project.
# This script keeps the same train -> attack pipeline as run_exp1.sh,
# but runs a broader sweep with longer schedules and different hyperparameters.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
cd "$repo_root"

timestamp="$(date +"%Y%m%d_%H%M%S")"
run_dir_name="run2log_${timestamp}"
run_dir="${repo_root}/${run_dir_name}"
mkdir -p "$run_dir"
experiments_dir="${run_dir}/experiments"
mkdir -p "$experiments_dir"

log_file="${run_dir}/run2_console.txt"
manifest_file="${run_dir}/run2_manifest.tsv"
summary_csv="${run_dir}/run2_summary.csv"

export PYTHONUNBUFFERED=1
exec > >(tee -a "$log_file") 2>&1

echo "Run folder: $run_dir"
echo "Console:    $log_file"
echo "Manifest:   $manifest_file"
echo "Summary:    $summary_csv"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Neither python nor python3 was found in PATH"
        exit 1
    fi
fi
echo "Python:     ${PYTHON_BIN}"

slug() {
    echo "$1" | tr '.' 'p'
}

GPU_id="${GPU_ID:-0}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-999}"

arch="vgg11_bn_sgm"
dataset="facescrub"
scheme="V2_epoch"
cutlayer="4"
num_client="1"
random_seed="233"

AT_regularization="SCA_new"
AT_regularization_strength="0.35"
ssim_threshold="0.45"
gan_loss_type="SSIM"
train_gan_AE_type="res_normN4C64"
test_gan_AE_type="res_normN8C64"

regularization="Gaussian_kl"
log_entropy="1"
local_lr="-1"
pretrain="False"
bottleneck_option="noRELU_C8S1"
folder_name="saves/facescrub/slotatt_run2_search"

attack_scheme="MIA"
attack_epochs="50"
average_time="1"

# Curated 20-run sweep:
# batch_size | lr | epochs | noise | lambd | var_threshold | slots | heads | iterations | attention_scale | warmup
EXPERIMENTS=(
    "224|0.035|280|0.025|6|0.08|5|2|3|0.25|6"
    "224|0.035|280|0.025|10|0.08|6|2|3|0.30|6"
    "224|0.032|300|0.030|14|0.10|6|4|4|0.35|8"
    "224|0.032|300|0.030|20|0.10|8|4|4|0.40|8"
    "256|0.030|300|0.030|12|0.12|8|2|4|0.40|10"
    "256|0.030|300|0.035|18|0.12|8|4|4|0.45|10"
    "256|0.028|300|0.035|24|0.12|10|2|4|0.50|10"
    "256|0.028|300|0.040|30|0.12|10|4|4|0.55|10"
    "256|0.026|320|0.040|10|0.15|8|2|5|0.45|12"
    "256|0.026|320|0.040|18|0.15|10|2|5|0.55|12"
    "256|0.024|320|0.045|26|0.15|10|4|5|0.60|12"
    "256|0.024|320|0.045|34|0.15|12|4|5|0.65|12"
    "288|0.022|320|0.050|12|0.18|10|2|5|0.60|14"
    "288|0.022|320|0.050|20|0.18|12|2|5|0.70|14"
    "288|0.021|320|0.050|28|0.18|12|4|5|0.75|14"
    "288|0.021|320|0.055|36|0.18|14|4|5|0.80|14"
    "256|0.020|340|0.055|15|0.20|12|2|6|0.70|16"
    "256|0.020|340|0.055|23|0.20|12|4|6|0.80|16"
    "224|0.018|340|0.060|31|0.22|14|2|6|0.90|18"
    "224|0.018|340|0.060|40|0.22|14|4|6|1.00|18"
)

echo -e "exp_id\tbatch_size\tlearning_rate\tnum_epochs\tnoise\tlambd\tvar_threshold\tnum_slots\tnum_heads\tnum_iterations\tattention_scale\twarmup\tstatus" > "$manifest_file"
echo "exp_id,status,save_dir,tested_model,best_val_acc,train_mse,train_ssim,train_psnr,infer_mse,infer_ssim,infer_psnr,batch_size,learning_rate,num_epochs,noise,lambd,var_threshold,num_slots,num_heads,num_iterations,attention_scale,warmup" > "$summary_csv"

append_summary_row() {
    local exp_id="$1"
    local status="$2"
    local save_dir="$3"
    local batch_size="$4"
    local learning_rate="$5"
    local num_epochs="$6"
    local regularization_strength="$7"
    local lambd="$8"
    local var_threshold="$9"
    local attention_num_slots="${10}"
    local attention_num_heads="${11}"
    local attention_num_iterations="${12}"
    local attention_loss_scale="${13}"
    local attention_warmup_epochs="${14}"

    local train_log="${save_dir}/MIA.log"
    local attack_log="${save_dir}/MIA_attack_0_0.log"

    "$PYTHON_BIN" - "$summary_csv" "$exp_id" "$status" "$save_dir" "$train_log" "$attack_log" "$batch_size" "$learning_rate" "$num_epochs" "$regularization_strength" "$lambd" "$var_threshold" "$attention_num_slots" "$attention_num_heads" "$attention_num_iterations" "$attention_loss_scale" "$attention_warmup_epochs" <<'PY'
import csv
import os
import re
import sys

(
    summary_csv,
    exp_id,
    status,
    save_dir,
    train_log,
    attack_log,
    batch_size,
    learning_rate,
    num_epochs,
    regularization_strength,
    lambd,
    var_threshold,
    attention_num_slots,
    attention_num_heads,
    attention_num_iterations,
    attention_loss_scale,
    attention_warmup_epochs,
) = sys.argv[1:]

best_val_acc = ""
tested_model = ""
train_mse = ""
train_ssim = ""
train_psnr = ""
infer_mse = ""
infer_ssim = ""
infer_psnr = ""

if os.path.isfile(train_log):
    with open(train_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"Best Average Validation Accuracy is ([0-9eE+.\-]+)", line)
            if m:
                best_val_acc = m.group(1)

if os.path.isfile(attack_log):
    with open(attack_log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        m = re.search(r"The tested model is:\s*(.+)", line)
        if m:
            tested_model = m.group(1).strip()

        if "MIA performance Score training time is" in line and i + 1 < len(lines):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lines[i + 1])
            if len(nums) >= 3:
                train_mse, train_ssim, train_psnr = nums[:3]

        m = re.search(
            r"MIA performance Score inference time is .*:\s*([0-9eE+.\-]+),\s*([0-9eE+.\-]+),\s*([0-9eE+.\-]+)",
            line,
        )
        if m:
            infer_mse, infer_ssim, infer_psnr = m.groups()

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            exp_id,
            status,
            save_dir,
            tested_model,
            best_val_acc,
            train_mse,
            train_ssim,
            train_psnr,
            infer_mse,
            infer_ssim,
            infer_psnr,
            batch_size,
            learning_rate,
            num_epochs,
            regularization_strength,
            lambd,
            var_threshold,
            attention_num_slots,
            attention_num_heads,
            attention_num_iterations,
            attention_loss_scale,
            attention_warmup_epochs,
        ]
    )
PY
}

total_experiments="${#EXPERIMENTS[@]}"
selected_runs=0
train_success=0
attack_success=0
failures=0

echo "Total configured experiments: $total_experiments"
echo "Running range: ${START_INDEX} to ${END_INDEX}"
echo

for idx in "${!EXPERIMENTS[@]}"; do
    exp_id=$((idx + 1))

    if (( exp_id < START_INDEX || exp_id > END_INDEX )); then
        continue
    fi

    selected_runs=$((selected_runs + 1))

    IFS='|' read -r batch_size learning_rate num_epochs regularization_strength lambd var_threshold attention_num_slots attention_num_heads attention_num_iterations attention_loss_scale attention_warmup_epochs <<< "${EXPERIMENTS[$idx]}"

    filename="run2_exp$(printf '%02d' "$exp_id")_bs$(slug "$batch_size")_lr$(slug "$learning_rate")_ep$(slug "$num_epochs")_noise$(slug "$regularization_strength")_lam$(slug "$lambd")_vt$(slug "$var_threshold")_s$(slug "$attention_num_slots")_h$(slug "$attention_num_heads")_it$(slug "$attention_num_iterations")_as$(slug "$attention_loss_scale")_w$(slug "$attention_warmup_epochs")"
    relative_folder="${run_dir_name}/saves"
    save_dir="${run_dir}/saves/${filename}"
    exp_dir="${experiments_dir}/exp$(printf '%02d' "$exp_id")"
    exp_log="${exp_dir}/experiment.log"
    params_file="${exp_dir}/params.txt"
    mkdir -p "$exp_dir"

    cat > "$params_file" <<EOF
exp_id=${exp_id}
filename=${filename}
save_dir=${save_dir}
batch_size=${batch_size}
learning_rate=${learning_rate}
num_epochs=${num_epochs}
regularization_strength=${regularization_strength}
lambd=${lambd}
var_threshold=${var_threshold}
attention_num_slots=${attention_num_slots}
attention_num_heads=${attention_num_heads}
attention_num_iterations=${attention_num_iterations}
attention_loss_scale=${attention_loss_scale}
attention_warmup_epochs=${attention_warmup_epochs}
arch=${arch}
dataset=${dataset}
scheme=${scheme}
cutlayer=${cutlayer}
AT_regularization=${AT_regularization}
AT_regularization_strength=${AT_regularization_strength}
bottleneck_option=${bottleneck_option}
gan_loss_type=${gan_loss_type}
train_gan_AE_type=${train_gan_AE_type}
test_gan_AE_type=${test_gan_AE_type}
attack_epochs=${attack_epochs}
EOF

    echo "======================================================================"
    echo "[$(date +"%F %T")] Experiment ${exp_id}/${total_experiments}"
    echo "filename=${filename}"
    echo "batch_size=${batch_size}, lr=${learning_rate}, epochs=${num_epochs}, noise=${regularization_strength}, lambd=${lambd}, var_threshold=${var_threshold}"
    echo "slots=${attention_num_slots}, heads=${attention_num_heads}, iterations=${attention_num_iterations}, attention_scale=${attention_loss_scale}, warmup=${attention_warmup_epochs}"
    echo "experiment_log=${exp_log}"
    echo

    train_status="train_failed"
    attack_status="skipped"

    {
        echo "======================================================================"
        echo "Experiment ${exp_id}/${total_experiments}"
        echo "Start time: $(date +"%F %T")"
        echo "filename=${filename}"
        echo "save_dir=${save_dir}"
        echo "batch_size=${batch_size}, learning_rate=${learning_rate}, num_epochs=${num_epochs}"
        echo "noise=${regularization_strength}, lambd=${lambd}, var_threshold=${var_threshold}"
        echo "slots=${attention_num_slots}, heads=${attention_num_heads}, iterations=${attention_num_iterations}, attention_scale=${attention_loss_scale}, warmup=${attention_warmup_epochs}"
        echo
        echo "[TRAIN COMMAND]"
        echo "CUDA_VISIBLE_DEVICES=${GPU_id} ${PYTHON_BIN} main_MIA.py --arch=${arch} --cutlayer=${cutlayer} --batch_size=${batch_size} --filename=${filename} --num_client=${num_client} --num_epochs=${num_epochs} --dataset=${dataset} --scheme=${scheme} --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength} --random_seed=${random_seed} --learning_rate=${learning_rate} --lambd=${lambd} --gan_AE_type=${train_gan_AE_type} --gan_loss_type=${gan_loss_type} --local_lr=${local_lr} --bottleneck_option=${bottleneck_option} --folder=${relative_folder} --ssim_threshold=${ssim_threshold} --var_threshold=${var_threshold} --attention_num_slots=${attention_num_slots} --attention_num_heads=${attention_num_heads} --attention_num_iterations=${attention_num_iterations} --attention_loss_scale=${attention_loss_scale} --attention_warmup_epochs=${attention_warmup_epochs}"
        echo
    } > "$exp_log"

    if (
        env CUDA_VISIBLE_DEVICES="${GPU_id}" "$PYTHON_BIN" main_MIA.py \
        --arch="${arch}" \
        --cutlayer="${cutlayer}" \
        --batch_size="${batch_size}" \
        --filename="${filename}" \
        --num_client="${num_client}" \
        --num_epochs="${num_epochs}" \
        --dataset="${dataset}" \
        --scheme="${scheme}" \
        --regularization="${regularization}" \
        --regularization_strength="${regularization_strength}" \
        --log_entropy="${log_entropy}" \
        --AT_regularization="${AT_regularization}" \
        --AT_regularization_strength="${AT_regularization_strength}" \
        --random_seed="${random_seed}" \
        --learning_rate="${learning_rate}" \
        --lambd="${lambd}" \
        --gan_AE_type="${train_gan_AE_type}" \
        --gan_loss_type="${gan_loss_type}" \
        --local_lr="${local_lr}" \
        --bottleneck_option="${bottleneck_option}" \
        --folder="${relative_folder}" \
        --ssim_threshold="${ssim_threshold}" \
        --var_threshold="${var_threshold}" \
        --attention_num_slots="${attention_num_slots}" \
        --attention_num_heads="${attention_num_heads}" \
        --attention_num_iterations="${attention_num_iterations}" \
        --attention_loss_scale="${attention_loss_scale}" \
        --attention_warmup_epochs="${attention_warmup_epochs}"
    ) 2>&1 | tee -a "$exp_log"
    then
        train_status="train_ok"
        train_success=$((train_success + 1))
    else
        failures=$((failures + 1))
        echo "Training failed for experiment ${exp_id}; attack stage skipped."
        {
            echo
            echo "Training failed at $(date +"%F %T")"
            echo "Final status: ${train_status}"
        } >> "$exp_log"
        echo -e "${exp_id}\t${batch_size}\t${learning_rate}\t${num_epochs}\t${regularization_strength}\t${lambd}\t${var_threshold}\t${attention_num_slots}\t${attention_num_heads}\t${attention_num_iterations}\t${attention_loss_scale}\t${attention_warmup_epochs}\t${train_status}" >> "$manifest_file"
        append_summary_row "$exp_id" "$train_status" "$save_dir" "$batch_size" "$learning_rate" "$num_epochs" "$regularization_strength" "$lambd" "$var_threshold" "$attention_num_slots" "$attention_num_heads" "$attention_num_iterations" "$attention_loss_scale" "$attention_warmup_epochs"
        echo
        continue
    fi

    {
        echo
        echo "[ATTACK COMMAND]"
        echo "CUDA_VISIBLE_DEVICES=${GPU_id} ${PYTHON_BIN} main_test_MIA.py --arch=${arch} --cutlayer=${cutlayer} --batch_size=${batch_size} --filename=${filename} --num_client=${num_client} --num_epochs=${num_epochs} --dataset=${dataset} --scheme=${scheme} --regularization=${regularization} --regularization_strength=${regularization_strength} --log_entropy=${log_entropy} --AT_regularization=${AT_regularization} --AT_regularization_strength=${AT_regularization_strength} --random_seed=${random_seed} --gan_AE_type=${test_gan_AE_type} --gan_loss_type=${gan_loss_type} --attack_epochs=${attack_epochs} --bottleneck_option=${bottleneck_option} --folder=${relative_folder} --var_threshold=${var_threshold} --attention_num_slots=${attention_num_slots} --attention_num_heads=${attention_num_heads} --attention_num_iterations=${attention_num_iterations} --attention_loss_scale=${attention_loss_scale} --attention_warmup_epochs=${attention_warmup_epochs} --average_time=${average_time} --test_best"
        echo
    } >> "$exp_log"

    if (
        env CUDA_VISIBLE_DEVICES="${GPU_id}" "$PYTHON_BIN" main_test_MIA.py \
        --arch="${arch}" \
        --cutlayer="${cutlayer}" \
        --batch_size="${batch_size}" \
        --filename="${filename}" \
        --num_client="${num_client}" \
        --num_epochs="${num_epochs}" \
        --dataset="${dataset}" \
        --scheme="${scheme}" \
        --regularization="${regularization}" \
        --regularization_strength="${regularization_strength}" \
        --log_entropy="${log_entropy}" \
        --AT_regularization="${AT_regularization}" \
        --AT_regularization_strength="${AT_regularization_strength}" \
        --random_seed="${random_seed}" \
        --gan_AE_type="${test_gan_AE_type}" \
        --gan_loss_type="${gan_loss_type}" \
        --attack_epochs="${attack_epochs}" \
        --bottleneck_option="${bottleneck_option}" \
        --folder="${relative_folder}" \
        --var_threshold="${var_threshold}" \
        --attention_num_slots="${attention_num_slots}" \
        --attention_num_heads="${attention_num_heads}" \
        --attention_num_iterations="${attention_num_iterations}" \
        --attention_loss_scale="${attention_loss_scale}" \
        --attention_warmup_epochs="${attention_warmup_epochs}" \
        --average_time="${average_time}" \
        --test_best
    ) 2>&1 | tee -a "$exp_log"
    then
        attack_status="ok"
        attack_success=$((attack_success + 1))
        echo "Experiment ${exp_id} completed successfully."
    else
        attack_status="attack_failed"
        failures=$((failures + 1))
        echo "Attack failed for experiment ${exp_id}."
    fi

    {
        echo
        echo "End time: $(date +"%F %T")"
        echo "Final status: ${attack_status}"
    } >> "$exp_log"

    echo -e "${exp_id}\t${batch_size}\t${learning_rate}\t${num_epochs}\t${regularization_strength}\t${lambd}\t${var_threshold}\t${attention_num_slots}\t${attention_num_heads}\t${attention_num_iterations}\t${attention_loss_scale}\t${attention_warmup_epochs}\t${attack_status}" >> "$manifest_file"
    append_summary_row "$exp_id" "$attack_status" "$save_dir" "$batch_size" "$learning_rate" "$num_epochs" "$regularization_strength" "$lambd" "$var_threshold" "$attention_num_slots" "$attention_num_heads" "$attention_num_iterations" "$attention_loss_scale" "$attention_warmup_epochs"
    echo
done

echo "======================================================================"
echo "run2.sh finished at $(date +"%F %T")"
echo "Selected experiments: ${selected_runs}"
echo "Training successes:   ${train_success}"
echo "Attack successes:     ${attack_success}"
echo "Failures:             ${failures}"
echo "Run folder:            ${run_dir}"
echo "Master log:            ${log_file}"
echo "Manifest:              ${manifest_file}"
echo "Summary CSV:           ${summary_csv}"
