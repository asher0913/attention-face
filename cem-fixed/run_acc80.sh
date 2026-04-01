#!/bin/bash
# =============================================================================
# run_acc80.sh — FaceScrub CEM-fixed: accuracy-optimized sweep (8 experiments)
#
# Target:  Acc ≈ 80%  (paper Noise_ARL+CEM = 80.33%)
#          MSE ≈ 0.023 (~10% above paper's 0.0211)
#
# Strategy vs run_exp.sh (which produced Acc 77-79%, MSE ~0.025):
#   - Lower noise σ:      0.025 - 0.04   (was 0.05 - 0.10)
#   - Lower lambd:         8 - 16        (was 16 - 48)
#   - Lower AT_REG_STR:   0.10 - 0.20   (was 0.30)
#   - Keep exp13 slot params: slot_dim=64, slots=8, iters=3, bank=64
#
# Usage     : bash run_acc80.sh [GPU_ID]          (default GPU=0)
# Creates   : runlog_acc80_YYYYMMDD_HHMMSS/
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
RUN_DIR="${SCRIPT_DIR}/runlog_acc80_${RUN_TS}"
mkdir -p "${RUN_DIR}"
MASTER_LOG="${RUN_DIR}/master.log"

lm()   { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"; }
lexp() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" \
             | tee -a "${EXP_LOG}" \
             | tee -a "${MASTER_LOG}"; }

cat > "${RUN_DIR}/README.txt" << RDEOF
run_acc80.sh — FaceScrub CEM-fixed accuracy-optimized sweep
Run timestamp : ${RUN_TS}
GPU           : ${GPU_ID}

Goal: Acc ≈ 80%, MSE ≈ 0.023 (10% above paper's 0.0211)
Strategy: weaker defense than run_exp.sh (lower σ, λ, AT_REG_STR)

Reference points:
  Paper Noise_ARL+CEM : Acc=80.33%, MSE=0.0211
  run_exp.sh exp13    : Acc=78.15%, MSE=0.0258 (σ=0.06, λ=24, AT_REG_STR=0.3)
  No defense          : Acc=89.43%, MSE=0.00067
RDEOF

lm "======================================================================"
lm " run_acc80.sh  |  FaceScrub accuracy-optimized  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm "======================================================================"

# ── fixed parameters ─────────────────────────────────────────────────────────
ARCH=vgg11_bn_sgm
BATCH_SIZE=256
NUM_CLIENT=1
RANDOM_SEED=125
CUTLAYER=4
BOTTLENECK=noRELU_C8S1
DATASET=facescrub
SCHEME=V2_epoch
REGULARIZATION=Gaussian_kl
LOG_ENTROPY=1
AT_REG=SCA_new
TRAIN_AE=res_normN4C64
TEST_AE=res_normN8C64
GAN_LOSS=SSIM
SSIM_THR=0.5
LOCAL_LR=-1
ATTACK_EPOCHS=50
LR=0.05

# ── experiment table ──────────────────────────────────────────────────────────
# Columns: EID  LAMBD  NOISE  VT  LS  SLOTS  ITERS  BANK  SDIM  WARMUP  EPOCHS  AT_STR
#
# exp13 baseline: λ=24, σ=0.06, vt=0.20, AT_STR=0.3 → Acc=78.15%, MSE=0.0258
# Paper baseline: σ=0.025, λ=~16 → Acc=80.33%, MSE=0.0211
#
# Group 1 (01-03): σ sweep with moderate λ — find the noise sweet spot
# Group 2 (04-06): λ sweep with low σ — find CEM strength sweet spot
# Group 3 (07-08): AT_REG_STR sweep — find regularization sweet spot
EXPERIMENTS=(
  "exp01   12  0.025  0.15  1.0   8  3   64   64  3  300  0.15"
  "exp02   12  0.030  0.15  1.0   8  3   64   64  3  300  0.15"
  "exp03   12  0.040  0.18  1.0   8  3   64   64  3  300  0.15"
  "exp04    8  0.030  0.15  1.0   8  3   64   64  3  300  0.15"
  "exp05   16  0.030  0.15  1.0   8  3   64   64  3  300  0.15"
  "exp06   16  0.025  0.20  1.0   8  3   64   64  3  300  0.15"
  "exp07   12  0.030  0.15  1.0   8  3   64   64  3  300  0.10"
  "exp08   12  0.030  0.15  1.0   8  3   64   64  3  300  0.20"
)

TOTAL=${#EXPERIMENTS[@]}
PASS=0
FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
#  Main experiment loop
# ══════════════════════════════════════════════════════════════════════════════
for ENTRY in "${EXPERIMENTS[@]}"; do

    read -r EID LAMBD NOISE VT LS SLOTS ITERS BANK SDIM WARMUP EPOCHS AT_STR <<< "${ENTRY}"

    EXP_LOG="${RUN_DIR}/${EID}.log"
    CFG="${RUN_DIR}/${EID}_config.txt"

    EFF=$(awk "BEGIN{ printf \"%.1f\", ${LAMBD} * ${LS} }")
    THR=$(awk "BEGIN{ printf \"%.2e\", ${VT} * ${NOISE}^2 }")

    FOLDER="saves/facescrub/${AT_REG}_cemfixed_acc80_vt${VT}"
    FNAME="l${LAMBD}_n${NOISE}_ep${EPOCHS}_vt${VT}_ls${LS}_sl${SLOTS}_it${ITERS}_bk${BANK}_sd${SDIM}_wu${WARMUP}_at${AT_STR}"
    HEADS=4

    cat > "${CFG}" << CFGEOF
exp_id=${EID}
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
at_reg_str=${AT_STR}
effective_strength=${EFF}
threshold=${THR}
train_status=PENDING
attack_status=PENDING
CFGEOF

    {
        printf '\n'
        printf '%.0s=' {1..72}; printf '\n'
        printf ' %s\n' "${EID}"
        printf '   lambd=%-4s  noise=%-5s  var_thr=%-5s  loss_scale=%s  at_reg_str=%s\n' \
               "${LAMBD}" "${NOISE}" "${VT}" "${LS}" "${AT_STR}"
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
        --AT_regularization_strength="${AT_STR}" \
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
        --AT_regularization_strength="${AT_STR}" \
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

# ══════════════════════════════════════════════════════════════════════════════
#  Summary CSV generator
# ══════════════════════════════════════════════════════════════════════════════
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
    'exp_id',
    'lambd', 'noise', 'var_thr', 'loss_scale', 'at_reg_str',
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

W = 115
print()
print('=' * W)
print(f"  SUMMARY  ({len(rows)} experiments)   ->   {csv_path}")
print('=' * W)
print(
    f"{'ID':<7} {'eff':>5} {'noise':>5} {'vt':>5} {'at_s':>5} "
    f"{'sl':>4} {'sd':>4} {'it':>3} {'bk':>4} {'wu':>3} {'ep':>4} "
    f"| {'acc':>7} "
    f"| {'tr_ssim':>8} {'tr_psnr':>7} "
    f"| {'in_ssim':>8} {'in_psnr':>7} "
    f"| status"
)
print('-' * W)
for r in sorted(rows, key=_ssim_key):
    st = f"{r.get('train_status','?')}/{r.get('attack_status','?')}"
    print(
        f"{r.get('exp_id','?'):<7}"
        f" {r.get('effective_strength','?'):>5}"
        f" {r.get('noise','?'):>5}"
        f" {r.get('var_thr','?'):>5}"
        f" {r.get('at_reg_str','?'):>5}"
        f" {r.get('slots','?'):>4}"
        f" {r.get('slot_dim','?'):>4}"
        f" {r.get('iters','?'):>3}"
        f" {r.get('bank','?'):>4}"
        f" {r.get('warmup','?'):>3}"
        f" {r.get('epochs','?'):>4}"
        f" | {r.get('best_acc','N/A'):>7}"
        f" | {r.get('train_ssim','N/A'):>8} {r.get('train_psnr','N/A'):>7}"
        f" | {r.get('infer_ssim','N/A'):>8} {r.get('infer_psnr','N/A'):>7}"
        f" | {st}"
    )
print('-' * W)
print("  Target       : Acc ~80%, MSE ~0.023  (paper: 80.33% / 0.0211)")
print("  exp13 ref    : Acc 78.15%, MSE 0.0258 (sigma=0.06, lambda=24, AT_STR=0.3)")
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
