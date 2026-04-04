#!/bin/bash
# =============================================================================
# run_acc80.sh — FaceScrub CEM-fixed: single accuracy-optimized experiment
#
# Target:  Acc ≈ 80%  (paper Noise_ARL+CEM = 80.33%)
#          MSE ≈ 0.023 (~10% above paper's 0.0211)
#
# Key change: bottleneck C8→C16 (8→16 channels)
#   Previous runs showed σ/λ/AT_STR reductions alone couldn't break 78.5%
#   The 8-channel bottleneck is the true accuracy ceiling.
#   C16 doubles information capacity while still providing privacy compression
#   (original 128ch → 16ch = 87.5% compression, vs 128→8 = 93.75%)
#
# Based on acc80 exp01 (Acc=78.43%, MSE=0.0247, σ=0.03, λ=12, AT_STR=0.15, C8)
# Changes:
#   bottleneck: noRELU_C8S1 → noRELU_C16S1  (key change)
#   σ:  0.03 → 0.035  (slightly increased to compensate C16 privacy loss)
#   λ:  12 → 14       (slightly increased for same reason)
#   AT_STR: 0.15 (unchanged)
#   Slot params unchanged: slot_dim=64, slots=8, iters=3, bank=64
#
# NOTE: vgg.py LCALayer was fixed to use bottleneck_channel_size dynamically
#       (was hardcoded to 8). feature_dim for SCA_new is auto-detected.
#       Attack decoder input_nc/input_dim are auto-detected from tensor shape.
#
# Usage: bash run_acc80.sh [GPU_ID]   (default GPU=0)
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

lm "======================================================================"
lm " run_acc80.sh  |  FaceScrub accuracy-optimized (C16)  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm "======================================================================"

# ── parameters ───────────────────────────────────────────────────────────────
ARCH=vgg11_bn_sgm
BATCH_SIZE=256
NUM_CLIENT=1
RANDOM_SEED=125
CUTLAYER=4
BOTTLENECK=noRELU_C16S1       # KEY CHANGE: 8→16 channels
DATASET=facescrub
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

# ── accuracy-optimized hyperparams ──────────────────────────────────────────
LAMBD=14                      # slightly above previous 12, compensate C16
NOISE=0.035                   # slightly above previous 0.03, compensate C16
VT=0.15
LS=1.0
SLOTS=8
ITERS=3
BANK=64
SDIM=64
WARMUP=3
HEADS=4

EID="exp01"
EXP_LOG="${RUN_DIR}/${EID}.log"

EFF=$(awk "BEGIN{ printf \"%.1f\", ${LAMBD} * ${LS} }")
THR=$(awk "BEGIN{ printf \"%.2e\", ${VT} * ${NOISE}^2 }")

FOLDER="saves/facescrub/${AT_REG}_cemfixed_acc80c16_vt${VT}"
FNAME="l${LAMBD}_n${NOISE}_ep${EPOCHS}_vt${VT}_ls${LS}_sl${SLOTS}_it${ITERS}_bk${BANK}_sd${SDIM}_wu${WARMUP}_at${AT_REG_STR}_bn${BOTTLENECK}"

{
    printf '\n'
    printf '%.0s=' {1..72}; printf '\n'
    printf ' Accuracy-optimized experiment (C16 bottleneck)\n'
    printf '   lambd=%-4s  noise=%-5s  var_thr=%-5s  loss_scale=%s  at_reg_str=%s\n' \
           "${LAMBD}" "${NOISE}" "${VT}" "${LS}" "${AT_REG_STR}"
    printf '   bottleneck=%s\n' "${BOTTLENECK}"
    printf '   effective_strength=%-5s  threshold=%s\n' "${EFF}" "${THR}"
    printf '   slots=%-3s  slot_dim=%s  iters=%s  bank=%s  warmup=%s  epochs=%s\n' \
           "${SLOTS}" "${SDIM}" "${ITERS}" "${BANK}" "${WARMUP}" "${EPOCHS}"
    printf '   Target: Acc ~80%%, MSE ~0.023\n'
    printf '   Ref (C8):  acc80 exp01 Acc=78.43%%, MSE=0.0247\n'
    printf '   Ref (C8):  exp13       Acc=78.15%%, MSE=0.0258\n'
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
    lm "Aborting — training failed."
    exit 1
fi

lexp "[${EID}] <<<<<< TRAINING OK  $(date)"

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
    exit 1
fi

lexp "[${EID}] <<<<<< ATTACK OK  $(date)"

lm ""
lm "======================================================================"
lm " DONE — single experiment complete"
lm "   Log: ${EXP_LOG}"
lm "   Target: Acc ~80%, MSE ~0.023"
lm "   Key change: bottleneck C8→C16 (noRELU_C16S1)"
lm "======================================================================"
