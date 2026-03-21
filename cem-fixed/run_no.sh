#!/bin/bash
# =============================================================================
# run_no.sh — FaceScrub Noise_Nopeek+CEM(SlotAttn) sweep  (16 experiments)
#   Directly comparable to paper Table 3 "Noise_Nopeek+CEM" on FaceScrub
#   Fixed: nopeek defense, σ=0.025, λ=16, 240 epochs (all matching paper)
#   Sweep: SlotAttn hyperparams only (slots, iters, bank, slot_dim)
#
# Usage     : bash run_no.sh [GPU_ID]          (default GPU=0)
# Creates   : runlog_YYYYMMDD_HHMMSS/  in the same directory as this script
#   master.log         all output, chronological
#   expXX.log          full output per experiment (training + attack)
#   expXX_config.txt   hyperparams snapshot (used by CSV generator)
#   summary.csv        one row per experiment, all metrics
#   README.txt         file-structure description
# =============================================================================

set -uo pipefail        # nounset + pipefail; no -e so individual failures continue

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"   # set once; all python calls inherit it
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"  # ensure all relative paths resolve correctly

# ── verify Python 3 (conda env must already be active) ───────────────────────
if ! python -c "import sys; assert sys.version_info[0] >= 3" 2>/dev/null; then
    echo "ERROR: 'python' is not Python 3."
    echo "       Activate your conda environment first, then re-run."
    exit 1
fi
PY=python

# ── create timestamped run directory ─────────────────────────────────────────
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${SCRIPT_DIR}/runlog_${RUN_TS}"
mkdir -p "${RUN_DIR}"
MASTER_LOG="${RUN_DIR}/master.log"

# ── logging helpers ───────────────────────────────────────────────────────────
lm()   { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"; }
lexp() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" \
             | tee -a "${EXP_LOG}" \
             | tee -a "${MASTER_LOG}"; }

# ── README ────────────────────────────────────────────────────────────────────
cat > "${RUN_DIR}/README.txt" << RDEOF
run_no.sh  —  FaceScrub Noise_Nopeek+CEM(SlotAttn) sweep
Run timestamp : ${RUN_TS}
GPU           : ${GPU_ID}

Purpose: Compare SlotAttn CEM vs GMM CEM on FaceScrub
  Base defense: Noise_Nopeek (distance correlation loss + Gaussian noise)
  Fixed params: σ=0.025, λ=16, 240 epochs (matching paper exactly)
  Variable: SlotAttn hyperparams (slots, iters, bank, slot_dim)
  Compare against: Paper Table 3 Noise_Nopeek+CEM on FaceScrub
    Acc=81.96, Dec Infer MSE=0.0075, GAN Infer MSE=0.0090

Files in this directory
  README.txt         This file
  master.log         Complete chronological output of all experiments
  expXX.log          Full output for experiment XX  (training phase + attack phase)
  expXX_config.txt   Hyperparameters for experiment XX  (parsed by CSV generator)
  summary.csv        All metrics in one table, sorted by infer_ssim ascending

summary.csv columns
  exp_id               experiment identifier (exp01 … exp16)
  lambd                CEM gradient weight (double-backward scale)
  noise                Gaussian noise sigma added to smashed data
  var_thr              variance threshold multiplier  (threshold = var_thr × noise²)
  loss_scale           pre-backward scale on rob_loss
  slots                number of SlotAttention prototypes
  slot_dim             projection dimension (2048 → slot_dim via proj_down)
  iters                SlotAttention GRU refinement iterations
  bank                 memory bank size per class
  warmup               epochs before CEM activates
  epochs               total training epochs
  effective_strength   lambd × loss_scale  (full post-period CEM gradient)
  threshold            var_thr × noise²    (actual variance target value)
  train_status         OK / FAILED / PENDING
  attack_status        OK / FAILED / SKIPPED / PENDING
  best_acc             best validation accuracy during training
  train_mse/ssim/psnr  MIA metrics when attacker trains on training set
  infer_mse/ssim/psnr  MIA metrics when attacker trains on val set  ← primary metric

Paper Table 3 Noise_Nopeek+CEM: Acc=81.96, Dec Infer MSE=0.0075, GAN Infer MSE=0.0090
RDEOF

lm "======================================================================"
lm " run_no.sh  |  FaceScrub Noise_Nopeek+CEM(SlotAttn) sweep  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm "======================================================================"

# ── fixed parameters (same for every experiment) ─────────────────────────────
# ── All fixed to match paper Section 5.1 for FaceScrub ─────────────────────
ARCH=vgg11_bn_sgm
BATCH_SIZE=256              # FaceScrub default batch size
NUM_CLIENT=1
RANDOM_SEED=125
CUTLAYER=4                  # paper "first two conv layers" = cutlayer=4 for FaceScrub
BOTTLENECK=noRELU_C8S1      # paper: 8-channel bottleneck for all methods
DATASET=facescrub
SCHEME=V2_epoch
REGULARIZATION=Gaussian_kl  # Gaussian noise corruption
LOG_ENTROPY=1
AT_REG=nopeek               # Noise_Nopeek: distance correlation loss
AT_REG_STR=1                # nopeek strength (matching CEM-main default)
TRAIN_AE=res_normN4C64
TEST_AE=res_normN8C64
GAN_LOSS=SSIM
SSIM_THR=0.5
LOCAL_LR=-1
ATTACK_EPOCHS=50
LR=0.05                    # paper: SGD lr=0.05

# ── experiment table ──────────────────────────────────────────────────────────
# Columns (space-separated, read by 'read -r'):
#   EID  LAMBD  NOISE  VT  LS  SLOTS  ITERS  BANK  SDIM  WARMUP  EPOCHS
#
# FIXED (matching paper): LAMBD=16, NOISE=0.025, EPOCHS=240
# SWEEP: SlotAttn hyperparams only (slots, iters, bank, slot_dim, loss_scale, var_thr)
#
# cutlayer=4 + vgg11_bn_sgm(feature_size=16) + noRELU_C8S1:
#   input resized to 64x64 → Conv64→MaxPool(32)→Conv128→MaxPool(16)→BN_C8S1
#   smashed data: [B, 8, 16, 16] → flatten [B, 2048]
#   proj_down projects 2048 → SDIM for Slot Attention
#
# Group A (01-03): slot_dim sweep (64, 128, 256)
# Group B (04-06): slots + iters sweep (4/12 slots, 5 iters)
# Group C (07-08): bank size sweep (32, 128)
# Group D (09-11): loss_scale + var_threshold + warmup tuning
# Group E (12-14): combined configs
# Group F (15-16): diverse combinations
EXPERIMENTS=(
  "exp01  16  0.025  0.15  1.0   8  3   64   64  3  240"
  "exp02  16  0.025  0.15  1.0   8  3   64  128  3  240"
  "exp03  16  0.025  0.15  1.0   8  3   64  256  3  240"
  "exp04  16  0.025  0.15  1.0   4  3   64  128  3  240"
  "exp05  16  0.025  0.15  1.0  12  3   64  128  3  240"
  "exp06  16  0.025  0.15  1.0   8  5   64  128  3  240"
  "exp07  16  0.025  0.15  1.0   8  3   32  128  3  240"
  "exp08  16  0.025  0.15  1.0   8  3  128  128  3  240"
  "exp09  16  0.025  0.10  0.5   8  3   64  128  3  240"
  "exp10  16  0.025  0.15  1.0   8  3   64  128  5  240"
  "exp11  16  0.025  0.20  2.0   8  3   64  128  3  240"
  "exp12  16  0.025  0.15  1.5   8  5   64  128  3  240"
  "exp13  16  0.025  0.15  1.0  12  5  128  128  5  240"
  "exp14  16  0.025  0.20  1.5   8  5  128  256  3  240"
  "exp15  16  0.025  0.15  1.0   4  5   64   64  3  240"
  "exp16  16  0.025  0.20  1.0  12  3  128  256  5  240"
)

TOTAL=${#EXPERIMENTS[@]}
PASS=0
FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
#  Main experiment loop
# ══════════════════════════════════════════════════════════════════════════════
for ENTRY in "${EXPERIMENTS[@]}"; do

    read -r EID LAMBD NOISE VT LS SLOTS ITERS BANK SDIM WARMUP EPOCHS <<< "${ENTRY}"

    EXP_LOG="${RUN_DIR}/${EID}.log"
    CFG="${RUN_DIR}/${EID}_config.txt"

    EFF=$(awk "BEGIN{ printf \"%.1f\", ${LAMBD} * ${LS} }")
    THR=$(awk "BEGIN{ printf \"%.2e\", ${VT} * ${NOISE}^2 }")

    # Path used by main_MIA.py to save/load checkpoints
    FOLDER="saves/facescrub/${AT_REG}_cemfixed_lg${LOG_ENTROPY}_vt${VT}"
    FNAME="l${LAMBD}_n${NOISE}_ep${EPOCHS}_vt${VT}_ls${LS}_sl${SLOTS}_it${ITERS}_bk${BANK}_sd${SDIM}_wu${WARMUP}"
    HEADS=4  # kept for argparse compatibility (CrossAttention removed, value unused)

    # ── write hyperparameter snapshot (CSV generator reads this later) ────────
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
effective_strength=${EFF}
threshold=${THR}
train_status=PENDING
attack_status=PENDING
CFGEOF

    # ── experiment header → stdout + expXX.log + master.log ──────────────────
    {
        printf '\n'
        printf '%.0s=' {1..72}; printf '\n'
        printf ' %s\n' "${EID}"
        printf '   lambd=%-4s  noise=%-5s  var_thr=%-5s  loss_scale=%s\n' \
               "${LAMBD}" "${NOISE}" "${VT}" "${LS}"
        printf '   effective_strength=%-5s  threshold=%s\n' "${EFF}" "${THR}"
        printf '   slots=%-3s  slot_dim=%s  iters=%s  bank=%s  warmup=%s  epochs=%s\n' \
               "${SLOTS}" "${SDIM}" "${ITERS}" "${BANK}" "${WARMUP}" "${EPOCHS}"
        printf '   log: %s\n' "${EXP_LOG}"
        printf '%.0s=' {1..72}; printf '\n'
    } | tee -a "${EXP_LOG}" | tee -a "${MASTER_LOG}"

    # ──────────────────────────────────────────────────────────────────────────
    #  Phase 1 : Training
    # ──────────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────
    #  Phase 2 : MIA Attack
    # ──────────────────────────────────────────────────────────────────────────
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

done  # ── end main loop ────────────────────────────────────────────────────────

lm ""
lm "======================================================================"
lm " All experiments done.  Passed=${PASS}/${TOTAL}  Failed=${FAIL}"
lm " Generating summary.csv ..."
lm "======================================================================"

# ══════════════════════════════════════════════════════════════════════════════
#  Embedded Python: parse per-experiment logs + configs → summary.csv
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

# ── gather all experiments ────────────────────────────────────────────────────

exp_ids = sorted(
    f[: -len('_config.txt')]
    for f in os.listdir(run_dir)
    if f.endswith('_config.txt')
)

FIELDS = [
    'exp_id',
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

# ── write CSV ─────────────────────────────────────────────────────────────────

csv_path = os.path.join(run_dir, 'summary.csv')
with open(csv_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

# ── console table sorted by infer_ssim ascending (lower = better privacy) ─────

def _ssim_key(r):
    try:
        return float(r.get('infer_ssim', '99'))
    except ValueError:
        return 99.0

W = 110
print()
print('=' * W)
print(f"  SUMMARY  ({len(rows)} experiments)   →   {csv_path}")
print('=' * W)
print(
    f"{'ID':<7} {'eff':>5} {'noise':>5} {'vt':>5} "
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
print("  Paper Table 3 Noise_Nopeek+CEM: Acc=81.96, Dec Infer MSE=0.0075, GAN Infer MSE=0.0090")
print()

PYEOF
CSV_RC=${PIPESTATUS[0]}

if [ "${CSV_RC}" -eq 0 ]; then
    lm "summary.csv written  →  ${RUN_DIR}/summary.csv"
else
    lm "WARNING: CSV generator exited with code ${CSV_RC} — inspect logs manually"
fi

lm ""
lm "======================================================================"
lm " DONE"
lm "   ${RUN_DIR}/"
lm "     master.log        full chronological log"
lm "     expXX.log         per-experiment log (training + attack)"
lm "     expXX_config.txt  hyperparameter snapshot"
lm "     summary.csv       all metrics, one row per experiment"
lm "     README.txt        file-structure description"
lm " Experiments: ${PASS}/${TOTAL} succeeded,  ${FAIL} failed"
lm "======================================================================"
