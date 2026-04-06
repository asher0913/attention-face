#!/bin/bash
# =============================================================================
# run_exp.sh — TinyImageNet Noise_ARL+CEM(SlotAttn) with ResNet-110
#
# "大力出奇迹" — ResNet-110 (~1.7M params) vs paper's ResNet-20 (~0.27M params)
# 6x more parameters, much deeper network for better classification
#
# Defense: AT_REG=SCA_new, matching FaceScrub run_exp.sh best configs (exp13 & exp17)
#
# exp01 = FaceScrub exp13 config (best privacy): λ=24, σ=0.06, sdim=64
# exp02 = FaceScrub exp17 config (strong):       λ=40, σ=0.06, sdim=128
#
# Usage: bash run_exp.sh [GPU_ID]   (default GPU=0)
# =============================================================================

set -uo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! python -c "import sys; assert sys.version_info[0] >= 3" 2>/dev/null; then
    echo "ERROR: 'python' is not Python 3."
    echo "       Activate your conda environment first, then re-run."
    exit 1
fi
PY=python

RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${SCRIPT_DIR}/runlog_exp_${RUN_TS}"
mkdir -p "${RUN_DIR}"
MASTER_LOG="${RUN_DIR}/master.log"

lm()   { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"; }
lexp() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" \
             | tee -a "${EXP_LOG}" \
             | tee -a "${MASTER_LOG}"; }

cat > "${RUN_DIR}/README.txt" << RDEOF
run_exp.sh — TinyImageNet ResNet-110 + SCA_new CEM
Run timestamp : ${RUN_TS}
GPU           : ${GPU_ID}

Architecture: ResNet-110 (~1.7M params, 6x larger than paper's ResNet-20)
Defense: SCA_new (SlotAttention CEM with ARL regularization)
Based on FaceScrub run_exp.sh exp13 & exp17 (best configs)

Reference (paper ResNet-20 + nopeek):
  Noise_Nopeek+CEM(GMM): Acc=53.39%, Dec Infer MSE=0.0098
  run_no.sh (ResNet-20): Acc~54%, attack all failed (fixed now)
RDEOF

lm "======================================================================"
lm " run_exp.sh  |  TinyImageNet ResNet-110 + SCA_new  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm "======================================================================"

# ── fixed parameters ─────────────────────────────────────────────────────────
ARCH=resnet110              # 大力出奇迹: ResNet-110 (~1.7M params)
BATCH_SIZE=128
NUM_CLIENT=1
RANDOM_SEED=125
CUTLAYER=4                  # conv5x5 + 3 stage_1 blocks (same split point)
BOTTLENECK=noRELU_C8S1
DATASET=tinyimagenet
SCHEME=V2_epoch
REGULARIZATION=Gaussian_kl
LOG_ENTROPY=1
AT_REG=SCA_new
AT_REG_STR=0.3
TRAIN_AE=res_normN4C64
TEST_AE=res_normN8C64
GAN_LOSS=SSIM
SSIM_THR=0.5
LOCAL_LR=-1
ATTACK_EPOCHS=50
LR=0.05

# ── experiment table ──────────────────────────────────────────────────────────
# Columns: EID  LAMBD  NOISE  VT  LS  SLOTS  ITERS  BANK  SDIM  WARMUP  EPOCHS
#
# exp01 = FaceScrub exp13 config (best privacy)
# exp02 = FaceScrub exp17 config (strong defense)
EXPERIMENTS=(
  "exp01  24  0.06  0.20  1.0   8  3   64   64  3  120"
  "exp02  40  0.06  0.25  0.8   8  4   64  128  3  120"
)

TOTAL=${#EXPERIMENTS[@]}
PASS=0
FAIL=0

for ENTRY in "${EXPERIMENTS[@]}"; do

    read -r EID LAMBD NOISE VT LS SLOTS ITERS BANK SDIM WARMUP EPOCHS <<< "${ENTRY}"

    EXP_LOG="${RUN_DIR}/${EID}.log"
    CFG="${RUN_DIR}/${EID}_config.txt"

    EFF=$(awk "BEGIN{ printf \"%.1f\", ${LAMBD} * ${LS} }")
    THR=$(awk "BEGIN{ printf \"%.2e\", ${VT} * ${NOISE}^2 }")

    FOLDER="saves/tinyimagenet/${AT_REG}_cemfixed_r110_lg${LOG_ENTROPY}_vt${VT}"
    FNAME="l${LAMBD}_n${NOISE}_ep${EPOCHS}_vt${VT}_ls${LS}_sl${SLOTS}_it${ITERS}_bk${BANK}_sd${SDIM}_wu${WARMUP}"
    HEADS=4

    cat > "${CFG}" << CFGEOF
exp_id=${EID}
arch=${ARCH}
lambd=${LAMBD}
noise=${NOISE}
var_thr=${VT}
loss_scale=${LS}
slots=${SLOTS}
slot_dim=${SDIM}
iters=${ITERS}
bank=${BANK}
warmup=${WARMUP}
epochs=${EPOCHS}
effective_strength=${EFF}
threshold=${THR}
train_status=PENDING
attack_status=PENDING
CFGEOF

    {
        printf '\n'
        printf '%.0s=' {1..72}; printf '\n'
        printf ' %s  (ResNet-110)\n' "${EID}"
        printf '   lambd=%-4s  noise=%-5s  var_thr=%-5s  loss_scale=%s\n' \
               "${LAMBD}" "${NOISE}" "${VT}" "${LS}"
        printf '   effective_strength=%-5s  threshold=%s\n' "${EFF}" "${THR}"
        printf '   slots=%-3s  slot_dim=%s  iters=%s  bank=%s  warmup=%s  epochs=%s\n' \
               "${SLOTS}" "${SDIM}" "${ITERS}" "${BANK}" "${WARMUP}" "${EPOCHS}"
        printf '   log: %s\n' "${EXP_LOG}"
        printf '%.0s=' {1..72}; printf '\n'
    } | tee -a "${EXP_LOG}" | tee -a "${MASTER_LOG}"

    # ── Phase 1: Training ────────────────────────────────────────────────────
    lexp "[${EID}] >>>>>> TRAINING START  $(date)"

    ${PY} main_MIA.py \
        --arch="${ARCH}" \
        --cutlayer="${CUTLAYER}" \
        --batch_size="${BATCH_SIZE}" \
        --filename="${FNAME}" \
        --num_client="${NUM_CLIENT}" \
        --num_epochs="${EPOCHS}" \
        --dataset="${DATASET}" \
        --scheme="${SCHEME}" \
        --regularization="${REGULARIZATION}" \
        --regularization_strength="${NOISE}" \
        --log_entropy="${LOG_ENTROPY}" \
        --AT_regularization="${AT_REG}" \
        --AT_regularization_strength="${AT_REG_STR}" \
        --random_seed="${RANDOM_SEED}" \
        --learning_rate="${LR}" \
        --lambd="${LAMBD}" \
        --gan_AE_type="${TRAIN_AE}" \
        --gan_loss_type="${GAN_LOSS}" \
        --local_lr="${LOCAL_LR}" \
        --bottleneck_option="${BOTTLENECK}" \
        --folder="${FOLDER}" \
        --ssim_threshold="${SSIM_THR}" \
        --var_threshold="${VT}" \
        --attention_num_slots="${SLOTS}" \
        --attention_num_heads="${HEADS}" \
        --attention_num_iterations="${ITERS}" \
        --attention_loss_scale="${LS}" \
        --attention_warmup_epochs="${WARMUP}" \
        --attention_bank_size="${BANK}" \
        --attention_slot_dim="${SDIM}" \
        2>&1 | tee -a "${EXP_LOG}" | tee -a "${MASTER_LOG}"
    TRAIN_RC=${PIPESTATUS[0]}

    if [ "${TRAIN_RC}" -ne 0 ]; then
        lexp "[${EID}] !!! TRAINING FAILED (exit=${TRAIN_RC})  $(date)"
        sed -i "s/^train_status=PENDING/train_status=FAILED/"   "${CFG}"
        sed -i "s/^attack_status=PENDING/attack_status=SKIPPED/" "${CFG}"
        FAIL=$((FAIL + 1))
        continue
    fi

    lexp "[${EID}] <<<<<< TRAINING OK  $(date)"
    sed -i "s/^train_status=PENDING/train_status=OK/" "${CFG}"

    # ── Phase 2: MIA Attack ──────────────────────────────────────────────────
    lexp "[${EID}] >>>>>> ATTACK START  $(date)"

    ${PY} main_test_MIA.py \
        --arch="${ARCH}" \
        --cutlayer="${CUTLAYER}" \
        --batch_size="${BATCH_SIZE}" \
        --filename="${FNAME}" \
        --num_client="${NUM_CLIENT}" \
        --num_epochs="${EPOCHS}" \
        --dataset="${DATASET}" \
        --scheme="${SCHEME}" \
        --regularization="${REGULARIZATION}" \
        --regularization_strength="${NOISE}" \
        --log_entropy="${LOG_ENTROPY}" \
        --AT_regularization="${AT_REG}" \
        --AT_regularization_strength="${AT_REG_STR}" \
        --random_seed="${RANDOM_SEED}" \
        --gan_loss_type="${GAN_LOSS}" \
        --attack_epochs="${ATTACK_EPOCHS}" \
        --bottleneck_option="${BOTTLENECK}" \
        --folder="${FOLDER}" \
        --var_threshold="${VT}" \
        --attention_num_slots="${SLOTS}" \
        --attention_num_heads="${HEADS}" \
        --attention_num_iterations="${ITERS}" \
        --attention_loss_scale="${LS}" \
        --attention_warmup_epochs="${WARMUP}" \
        --attention_bank_size="${BANK}" \
        --attention_slot_dim="${SDIM}" \
        --average_time=1 \
        --gan_AE_type="${TEST_AE}" \
        --test_best \
        2>&1 | tee -a "${EXP_LOG}" | tee -a "${MASTER_LOG}"
    ATTACK_RC=${PIPESTATUS[0]}

    if [ "${ATTACK_RC}" -ne 0 ]; then
        lexp "[${EID}] !!! ATTACK FAILED (exit=${ATTACK_RC})  $(date)"
        sed -i "s/^attack_status=PENDING/attack_status=FAILED/" "${CFG}"
        FAIL=$((FAIL + 1))
        continue
    fi

    lexp "[${EID}] <<<<<< ATTACK OK  $(date)"
    sed -i "s/^attack_status=PENDING/attack_status=OK/" "${CFG}"
    PASS=$((PASS + 1))

done

lm ""
lm "======================================================================"
lm " All experiments done.  Passed=${PASS}/${TOTAL}  Failed=${FAIL}"
lm " Generating summary.csv ..."
lm "======================================================================"

${PY} - "${RUN_DIR}" << 'PYEOF'
import csv, os, re, sys

run_dir = sys.argv[1]

def load_config(path):
    cfg = {}
    try:
        with open(path, errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if '=' in line:
                    k, v = line.split('=', 1)
                    cfg[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return cfg

def parse_log(path):
    result = {
        'best_acc'  : 'N/A',
        'train_mse' : 'N/A', 'train_ssim': 'N/A', 'train_psnr': 'N/A',
        'infer_mse' : 'N/A', 'infer_ssim': 'N/A', 'infer_psnr': 'N/A',
    }
    try:
        with open(path, errors='replace') as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return result

    last_acc = None
    for i, line in enumerate(lines):
        m = re.search(r'Best Average Validation Accuracy is\s+([0-9.]+)', line)
        if m:
            last_acc = m.group(1)

        if 'MIA performance Score training time is' in line:
            for j in range(i + 1, min(i + 6, len(lines))):
                nxt = lines[j].strip()
                if nxt and re.match(r'^[0-9]', nxt):
                    parts = [p.strip() for p in nxt.split(',')]
                    if len(parts) >= 3:
                        result['train_mse']  = parts[0]
                        result['train_ssim'] = parts[1]
                        result['train_psnr'] = parts[2]
                    break

        m = re.search(
            r'MIA performance Score inference time is \(MSE, SSIM, PSNR\):'
            r'\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)',
            line
        )
        if m:
            result['infer_mse']  = m.group(1)
            result['infer_ssim'] = m.group(2)
            result['infer_psnr'] = m.group(3)

    if last_acc is not None:
        result['best_acc'] = last_acc
    return result

exp_ids = sorted(
    f[: -len('_config.txt')]
    for f in os.listdir(run_dir)
    if f.endswith('_config.txt')
)

FIELDS = [
    'exp_id', 'arch',
    'lambd', 'noise', 'var_thr', 'loss_scale',
    'slots', 'slot_dim', 'iters', 'bank', 'warmup', 'epochs',
    'effective_strength', 'threshold',
    'train_status', 'attack_status',
    'best_acc',
    'train_mse', 'train_ssim', 'train_psnr',
    'infer_mse', 'infer_ssim', 'infer_psnr',
]

rows = []
for eid in exp_ids:
    cfg  = load_config(os.path.join(run_dir, f'{eid}_config.txt'))
    mets = parse_log(os.path.join(run_dir, f'{eid}.log'))
    row  = {f: 'N/A' for f in FIELDS}
    row.update(cfg)
    row.update(mets)
    row['exp_id'] = eid
    rows.append(row)

csv_path = os.path.join(run_dir, 'summary.csv')
with open(csv_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

def _ssim_key(r):
    try:
        return float(r.get('infer_ssim', '99'))
    except ValueError:
        return 99.0

W = 110
print()
print('=' * W)
print(f"  SUMMARY  ({len(rows)} experiments)   ->   {csv_path}")
print('=' * W)
print(
    f"{'ID':<7} {'arch':>10} {'eff':>5} {'noise':>5} {'vt':>5} "
    f"{'sl':>4} {'sd':>4} {'it':>3} {'bk':>4} {'ep':>4} "
    f"| {'acc':>7} "
    f"| {'in_mse':>8} {'in_ssim':>8} "
    f"| status"
)
print('-' * W)
for r in sorted(rows, key=_ssim_key):
    st = f"{r.get('train_status','?')}/{r.get('attack_status','?')}"
    print(
        f"{r.get('exp_id','?'):<7}"
        f" {r.get('arch','?'):>10}"
        f" {r.get('effective_strength','?'):>5}"
        f" {r.get('noise','?'):>5}"
        f" {r.get('var_thr','?'):>5}"
        f" {r.get('slots','?'):>4}"
        f" {r.get('slot_dim','?'):>4}"
        f" {r.get('iters','?'):>3}"
        f" {r.get('bank','?'):>4}"
        f" {r.get('epochs','?'):>4}"
        f" | {r.get('best_acc','N/A'):>7}"
        f" | {r.get('infer_mse','N/A'):>8} {r.get('infer_ssim','N/A'):>8}"
        f" | {st}"
    )
print('-' * W)
print("  Paper (ResNet-20 + Nopeek+CEM): Acc=53.39%, MSE=0.0098")
print()

PYEOF
CSV_RC=${PIPESTATUS[0]}

if [ "${CSV_RC}" -eq 0 ]; then
    lm "summary.csv written  ->  ${RUN_DIR}/summary.csv"
else
    lm "WARNING: CSV generator exited with code ${CSV_RC}"
fi

lm ""
lm "======================================================================"
lm " DONE"
lm "   ${RUN_DIR}/"
lm "   Experiments: ${PASS}/${TOTAL} succeeded,  ${FAIL} failed"
lm "======================================================================"
