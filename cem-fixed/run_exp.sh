#!/bin/bash
# =============================================================================
# run_exp.sh — FaceScrub CEM-fixed hyperparameter sweep  (20 experiments)
#
# Usage     : bash run_exp.sh [GPU_ID]          (default GPU=0)
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
run_exp.sh  —  FaceScrub CEM-fixed hyperparameter sweep
Run timestamp : ${RUN_TS}
GPU           : ${GPU_ID}

Architecture fixes vs original SCA:
  1. Loss direction: relu(log(var) - log(thr))  (matches CEM-main)
  2. Geometric variance: weighted squared distance, not neural-net predicted
  3. Cross-sample clustering via FeatureMemoryBank (not token-level)
  4. Sample-level features: z_private → [B, 512] (not [B, 64, 8])
  5. proj_down 512→slot_dim for efficient Slot Attention

Files in this directory
  README.txt         This file
  master.log         Complete chronological output of all experiments
  expXX.log          Full output for experiment XX  (training phase + attack phase)
  expXX_config.txt   Hyperparameters for experiment XX  (parsed by CSV generator)
  summary.csv        All metrics in one table, sorted by infer_ssim ascending

summary.csv columns
  exp_id               experiment identifier (exp01 … exp20)
  lambd                CEM gradient weight (double-backward scale)
  noise                Gaussian noise sigma added to smashed data
  var_thr              variance threshold multiplier  (threshold = var_thr × noise²)
  loss_scale           pre-backward scale on rob_loss
  slots                number of SlotAttention prototypes
  heads                number of CrossAttention heads
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

Baseline (CEM-main SOTA): infer_ssim ≈ 0.532   (lower = better privacy)
RDEOF

lm "======================================================================"
lm " run_exp.sh  |  FaceScrub CEM-fixed sweep  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm "======================================================================"

# ── fixed parameters (same for every experiment) ─────────────────────────────
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
AT_REG_STR=0.3
TRAIN_AE=res_normN4C64
TEST_AE=res_normN8C64
GAN_LOSS=SSIM
SSIM_THR=0.5
LOCAL_LR=-1
ATTACK_EPOCHS=50
LR=0.05

# ── experiment table ──────────────────────────────────────────────────────────
# Columns (space-separated, read by 'read -r'):
#   EID  LAMBD  NOISE  VT  LS  SLOTS  ITERS  BANK  WARMUP  EPOCHS
#
# CrossAttention removed (unused output); HEADS column dropped.
# Slot Attention operates in full 512-dim with detached centroids.
#
# Group A (01-03): CEM strength baseline — lambd sweep
# Group B (04-06): noise + threshold push
# Group C (07-09): bank size sweep
# Group D (10-12): slot architecture (slots / iters)
# Group E (13-15): warmup timing
# Group F (16-18): combined strong configs
# Group G (19-20): aggressive SOTA push
EXPERIMENTS=(
  "exp01  16  0.05  0.15  1.0   8  3   64  3  300"
  "exp02  24  0.05  0.15  1.0   8  3   64  3  300"
  "exp03  32  0.05  0.15  1.0   8  3   64  3  300"
  "exp04  24  0.06  0.20  1.0   8  3   64  3  300"
  "exp05  32  0.08  0.20  0.8   8  3   64  3  300"
  "exp06  24  0.10  0.25  1.0   8  3   64  3  300"
  "exp07  24  0.05  0.15  1.0   8  3   32  3  300"
  "exp08  24  0.05  0.15  1.0   8  3   64  3  300"
  "exp09  24  0.05  0.15  1.0   8  3  128  3  300"
  "exp10  24  0.05  0.15  1.0   4  3   64  3  300"
  "exp11  24  0.05  0.15  1.0  12  5   64  3  300"
  "exp12  32  0.05  0.15  1.0   8  5   64  3  300"
  "exp13  24  0.06  0.20  1.0   8  3   64  1  300"
  "exp14  24  0.06  0.20  1.0   8  3   64  3  300"
  "exp15  24  0.06  0.20  1.0   8  3   64  5  300"
  "exp16  32  0.08  0.20  1.0  12  5  128  3  300"
  "exp17  40  0.06  0.25  0.8   8  4   64  3  300"
  "exp18  28  0.07  0.20  1.0  10  4   96  3  300"
  "exp19  48  0.08  0.20  1.0   8  5  128  3  360"
  "exp20  40  0.06  0.25  1.2  12  4  128  5  360"
)

TOTAL=${#EXPERIMENTS[@]}
PASS=0
FAIL=0

# ══════════════════════════════════════════════════════════════════════════════
#  Main experiment loop
# ══════════════════════════════════════════════════════════════════════════════
for ENTRY in "${EXPERIMENTS[@]}"; do

    read -r EID LAMBD NOISE VT LS SLOTS ITERS BANK WARMUP EPOCHS <<< "${ENTRY}"

    EXP_LOG="${RUN_DIR}/${EID}.log"
    CFG="${RUN_DIR}/${EID}_config.txt"

    EFF=$(awk "BEGIN{ printf \"%.1f\", ${LAMBD} * ${LS} }")
    THR=$(awk "BEGIN{ printf \"%.2e\", ${VT} * ${NOISE}^2 }")

    # Path used by main_MIA.py to save/load checkpoints
    FOLDER="saves/facescrub/${AT_REG}_cemfixed_lg${LOG_ENTROPY}_vt${VT}"
    FNAME="l${LAMBD}_n${NOISE}_ep${EPOCHS}_vt${VT}_ls${LS}_sl${SLOTS}_it${ITERS}_bk${BANK}_wu${WARMUP}"
    HEADS=4  # kept for argparse compatibility (CrossAttention removed, value unused)

    # ── write hyperparameter snapshot (CSV generator reads this later) ────────
    cat > "${CFG}" << CFGEOF
exp_id=${EID}
lambd=${LAMBD}
noise=${NOISE}
var_thr=${VT}
loss_scale=${LS}
slots=${SLOTS}
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
        printf '   slots=%-3s  iters=%s  bank=%s  warmup=%s  epochs=%s\n' \
               "${SLOTS}" "${ITERS}" "${BANK}" "${WARMUP}" "${EPOCHS}"
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
    'slots', 'iters', 'bank', 'warmup', 'epochs',
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
    f"{'sl':>4} {'it':>3} {'bk':>4} {'wu':>3} {'ep':>4} "
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
print("  CEM-main SOTA baseline : infer_ssim ≈ 0.532   (lower = better privacy)")
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
