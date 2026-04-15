#!/bin/bash
# =============================================================================
# run_final.sh — CIFAR-10 (final thesis configuration, single experiment)
#
# Purpose: Reproduce the exact FaceScrub main defended configuration on CIFAR-10
#          (Table "Main defended configuration" in the final report).
#
# Identical to the FaceScrub run in every hyperparameter except --dataset:
#   arch       = vgg11_bn_sgm
#   cutlayer   = 4
#   bottleneck = noRELU_C16S1          (16-channel bottleneck)
#   batch size = 256
#   epochs     = 300
#   LR         = 0.05   (SGD, momentum/weight-decay set in main_MIA.py)
#   AT_reg     = SCA_new   (my Attention-CEM)
#   AT_reg_str = 0.15      (SCA strength)
#   lambd      = 10        (CEM weight λ)
#   noise σ    = 0.025
#   var_thr    = 0.15
#   loss_scale = 1.0
#   slots = 8,  iters = 3,  slot_dim = 64,  bank = 64,  warmup = 3
#   attack_epochs = 50
#
# Usage: bash run_final.sh [GPU_ID]   (default GPU=0)
# =============================================================================

set -uo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! python -c "import sys; assert sys.version_info[0] >= 3" 2>/dev/null; then
    echo "ERROR: 'python' is not Python 3. Activate conda env first."
    exit 1
fi
PY=python

RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${SCRIPT_DIR}/runlog_final_${RUN_TS}"
mkdir -p "${RUN_DIR}"
MASTER_LOG="${RUN_DIR}/master.log"

lm()   { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"; }
lexp() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" \
             | tee -a "${EXP_LOG}" \
             | tee -a "${MASTER_LOG}"; }

# ── fixed parameters (IDENTICAL to FaceScrub final config) ───────────────────
ARCH=vgg11_bn_sgm
BATCH_SIZE=256
NUM_CLIENT=1
RANDOM_SEED=125
CUTLAYER=4
BOTTLENECK=noRELU_C16S1
DATASET=cifar100
SCHEME=V2_epoch
REGULARIZATION=Gaussian_kl
LOG_ENTROPY=1
AT_REG=SCA_new
AT_REG_STR=0.15
TRAIN_AE=res_normN4C64
TEST_AE=res_normN8C64
GAN_LOSS=SSIM
SSIM_THR=0.5
LOCAL_LR=-1
ATTACK_EPOCHS=50
LR=0.05
EPOCHS=300

LAMBD=10
NOISE=0.025
VT=0.15
LS=1.0
SLOTS=8
ITERS=3
BANK=64
SDIM=64
WARMUP=3
HEADS=4

EID="final"
EXP_LOG="${RUN_DIR}/${EID}.log"
CFG="${RUN_DIR}/${EID}_config.txt"

EFF=$(awk "BEGIN{ printf \"%.1f\", ${LAMBD} * ${LS} }")
THR=$(awk "BEGIN{ printf \"%.2e\", ${VT} * ${NOISE}^2 }")

FOLDER="saves/${DATASET}/${AT_REG}_final_lg${LOG_ENTROPY}_vt${VT}"
FNAME="l${LAMBD}_n${NOISE}_ep${EPOCHS}_vt${VT}_ls${LS}_sl${SLOTS}_it${ITERS}_bk${BANK}_sd${SDIM}_wu${WARMUP}_at${AT_REG_STR}_bn${BOTTLENECK}"

cat > "${CFG}" << CFGEOF
exp_id=${EID}
dataset=${DATASET}
arch=${ARCH}
cutlayer=${CUTLAYER}
bottleneck=${BOTTLENECK}
at_reg=${AT_REG}
at_reg_str=${AT_REG_STR}
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
batch_size=${BATCH_SIZE}
learning_rate=${LR}
attack_epochs=${ATTACK_EPOCHS}
effective_strength=${EFF}
threshold=${THR}
train_status=PENDING
attack_status=PENDING
CFGEOF

lm "======================================================================"
lm " run_final.sh  |  ${DATASET} (final thesis config)  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm "======================================================================"

{
    printf '\n'
    printf '%.0s=' {1..72}; printf '\n'
    printf ' %s  dataset=%s  arch=%s  cutlayer=%s  bottleneck=%s\n' \
           "${EID}" "${DATASET}" "${ARCH}" "${CUTLAYER}" "${BOTTLENECK}"
    printf '   lambd=%-4s  noise=%-5s  var_thr=%-5s  loss_scale=%s  at_str=%s\n' \
           "${LAMBD}" "${NOISE}" "${VT}" "${LS}" "${AT_REG_STR}"
    printf '   effective_strength=%-5s  threshold=%s\n' "${EFF}" "${THR}"
    printf '   slots=%-3s  slot_dim=%s  iters=%s  bank=%s  warmup=%s  epochs=%s\n' \
           "${SLOTS}" "${SDIM}" "${ITERS}" "${BANK}" "${WARMUP}" "${EPOCHS}"
    printf '   log: %s\n' "${EXP_LOG}"
    printf '%.0s=' {1..72}; printf '\n'
} | tee -a "${EXP_LOG}" | tee -a "${MASTER_LOG}"

# ── Phase 1: Training ────────────────────────────────────────────────────────
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
    sed -i "s/^train_status=PENDING/train_status=FAILED/"    "${CFG}"
    sed -i "s/^attack_status=PENDING/attack_status=SKIPPED/" "${CFG}"
    lm "Aborting — training failed."
    exit 1
fi

lexp "[${EID}] <<<<<< TRAINING OK  $(date)"
sed -i "s/^train_status=PENDING/train_status=OK/" "${CFG}"

# ── Phase 2: MIA Attack ──────────────────────────────────────────────────────
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
    lm "Attack phase failed — check ${EXP_LOG}"
    exit 1
fi

lexp "[${EID}] <<<<<< ATTACK OK  $(date)"
sed -i "s/^attack_status=PENDING/attack_status=OK/" "${CFG}"

# ── Summary (grep the final metrics out of the log for convenience) ──────────
lm ""
lm "======================================================================"
lm " FINAL RESULTS  (${DATASET})"
lm "======================================================================"
BEST_ACC=$(grep -oE 'Best Average Validation Accuracy is [0-9.]+' "${EXP_LOG}" | tail -1 | awk '{print $NF}')
INFER_LINE=$(grep -oE 'MIA performance Score inference time is \(MSE, SSIM, PSNR\):[^$]*' "${EXP_LOG}" | tail -1)
lm " Accuracy      : ${BEST_ACC:-N/A}"
lm " MIA (inference): ${INFER_LINE:-N/A}"
lm "======================================================================"
lm " All files in: ${RUN_DIR}/"
