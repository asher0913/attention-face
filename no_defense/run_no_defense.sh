#!/bin/bash
# =============================================================================
# run_no_defense.sh — FaceScrub No Defense baseline (1 experiment)
#   Reproduces paper Table 3 "No_defense" row on FaceScrub
#   NO noise, NO nopeek, NO CEM, NO bottleneck — pure split learning
#   Paper target: Acc=86.69%, Dec Infer MSE=0.0011
#
# Usage     : bash run_no_defense.sh [GPU_ID]          (default GPU=0)
# Creates   : runlog_nodefense_YYYYMMDD_HHMMSS/
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
RUN_DIR="${SCRIPT_DIR}/runlog_nodefense_${RUN_TS}"
mkdir -p "${RUN_DIR}"
MASTER_LOG="${RUN_DIR}/master.log"

lm() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"; }

# ── Configuration matching paper "No_defense" row ────────────────────────────
ARCH=vgg11_bn_sgm
BATCH_SIZE=256
NUM_CLIENT=1
RANDOM_SEED=125
CUTLAYER=4                  # paper: first two conv layers for FaceScrub
BOTTLENECK=None             # NO bottleneck — pure split learning
DATASET=facescrub
SCHEME=V2_epoch
EPOCHS=240                  # paper: 240 epochs
LR=0.05                     # paper: SGD lr=0.05

# ── All defense mechanisms DISABLED ──────────────────────────────────────────
REGULARIZATION=None         # NO Gaussian noise
NOISE=0                     # noise strength = 0
AT_REG=None                 # NO nopeek / NO SCA_new / NO CEM
AT_REG_STR=0                # AT regularization strength = 0
LAMBD=0                     # CEM weight = 0 (no CEM)
LOG_ENTROPY=0               # no entropy-based defense
VT=0.1                      # var_threshold (irrelevant when lambd=0)

# ── Attack settings ──────────────────────────────────────────────────────────
TRAIN_AE=res_normN4C64
TEST_AE=res_normN8C64
GAN_LOSS=SSIM
SSIM_THR=0.5
LOCAL_LR=-1
ATTACK_EPOCHS=50

# ── Paths ────────────────────────────────────────────────────────────────────
FOLDER="saves/facescrub/no_defense"
FNAME="no_defense_ep${EPOCHS}_lr${LR}"

# ── SlotAttention params (unused when AT_REG=None, but needed by argparse) ───
SLOTS=8
HEADS=4
ITERS=3
LS=0
WARMUP=999                  # warmup > epochs means CEM never activates
BANK=64
SDIM=128

EXP_LOG="${RUN_DIR}/exp.log"

lm "======================================================================"
lm " run_no_defense.sh  |  FaceScrub No Defense Baseline  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm " Paper target: Acc=86.69%, Dec Infer MSE=0.0011"
lm "======================================================================"

# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1 : Training
# ══════════════════════════════════════════════════════════════════════════════
lm ">>>>>> TRAINING START  $(date)"

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
    lm "!!! TRAINING FAILED (exit=${TRAIN_RC})  $(date)"
    exit 1
fi
lm "<<<<<< TRAINING OK  $(date)"

# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 : MIA Attack
# ══════════════════════════════════════════════════════════════════════════════
lm ">>>>>> ATTACK START  $(date)"

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
    lm "!!! ATTACK FAILED (exit=${ATTACK_RC})  $(date)"
    exit 1
fi
lm "<<<<<< ATTACK OK  $(date)"

lm ""
lm "======================================================================"
lm " DONE — No Defense Baseline"
lm "   Log: ${EXP_LOG}"
lm "   Paper target: Acc=86.69%, Dec Infer MSE=0.0011"
lm "======================================================================"
