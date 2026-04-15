#!/bin/bash
# =============================================================================
# run_exp.sh — FaceScrub WRN-28-10 + SCA_new defense  (Round 2)
#
# Round 1 result: cutlayer=4 → Acc=91%, MSE=0.011 (defense failed)
# Root cause: residual connections in local model let info bypass bottleneck
#
# Fix: cut BEFORE residual blocks so local model = plain conv (like VGG)
#   cutlayer=1 → local = 7x7 conv only (0 residual blocks)
#   cutlayer=2 → local = 7x7 conv + 1 BasicBlock (1 residual block)
# Cloud keeps all remaining ResNet blocks → strong classification
#
# Smashed data matched to VGG: 8×16×16 = 2048 dim in all experiments
#   cutlayer=1: 16ch 32×32 → C8S2 bottleneck → 8ch 16×16
#   cutlayer=2: 160ch 16×16 → C8S1 bottleneck → 8ch 16×16
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
    echo "ERROR: 'python' is not Python 3. Activate conda env first."
    exit 1
fi
PY=python

# ── timestamped run directory ────────────────────────────────────────────────
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${SCRIPT_DIR}/runlog_${RUN_TS}"
mkdir -p "${RUN_DIR}"
MASTER_LOG="${RUN_DIR}/master.log"

lm()   { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"; }
lexp() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" \
             | tee -a "${EXP_LOG}" \
             | tee -a "${MASTER_LOG}"; }

# ── fixed parameters ─────────────────────────────────────────────────────────
ARCH=wideresnet28_10
BATCH_SIZE=128
NUM_CLIENT=1
RANDOM_SEED=125
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

# ── experiment table ─────────────────────────────────────────────────────────
# Columns: EID  CUTLAYER  BOTTLENECK  AT_STR  LAMBD  NOISE  VT  LS  SLOTS  ITERS  BANK  SDIM  WARMUP  EPOCHS
#
# Round 2 results (completed):
#   exp01: cut=1 C8S2 l24 n.06 vt.20 sd64  → Acc=87.31%, MSE=0.0035, SSIM=0.8443
#   exp02: cut=1 C8S2 l32 n.08 vt.25 sd128 → Acc=86.14%, MSE=0.0047, SSIM=0.8219
#   exp03: cut=2 C8S1 l24 n.06 vt.20 sd64  → Acc=88.99%, MSE=0.0071, SSIM=0.7743 ★ winner
#
# VGG11 best:                              → Acc=79.49%, MSE=0.0274, SSIM=0.532
#
# Round 3 goal: push defense aggressively on exp03 recipe (cut=2 + C8S1)
#   target: Acc drop to ~80-82%, MSE > 0.0274, SSIM < 0.53 (beat VGG on privacy)
#
# exp04 = exp03 winning recipe + MUCH stronger defense:
#   lambd  24 → 64  (2.67x)   dominant CEM weight
#   noise  0.06 → 0.12 (2x)   stronger Gaussian perturbation (threshold grows 4x)
#   var_thr 0.20 → 0.30 (1.5x) higher variance floor
#   at_str 0.3 → 0.5         stronger slot-attention regularization
#   sdim   64 → 128          richer clustering capacity
#   effective_strength = 64 * 1.0 = 64      (was 24)
#   threshold = 0.30 * 0.12^2 = 4.32e-03    (was 7.20e-04, 6x)
EXPERIMENTS=(
  "exp04  2  noRELU_C8S1  0.5  64  0.12  0.30  1.0   8  3   64  128  3  300"
)

TOTAL=${#EXPERIMENTS[@]}
PASS=0
FAIL=0

lm "======================================================================"
lm " run_exp.sh  |  FaceScrub WRN-28-10 Round 2 (cutlayer fix)  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm " Experiments: ${TOTAL}"
lm "======================================================================"

# ══════════════════════════════════════════════════════════════════════════════
#  Main experiment loop
# ══════════════════════════════════════════════════════════════════════════════
for ENTRY in "${EXPERIMENTS[@]}"; do

    read -r EID CUTLAYER BOTTLENECK AT_REG_STR LAMBD NOISE VT LS SLOTS ITERS BANK SDIM WARMUP EPOCHS <<< "${ENTRY}"

    EXP_LOG="${RUN_DIR}/${EID}.log"
    CFG="${RUN_DIR}/${EID}_config.txt"

    EFF=$(awk "BEGIN{ printf \"%.1f\", ${LAMBD} * ${LS} }")
    THR=$(awk "BEGIN{ printf \"%.2e\", ${VT} * ${NOISE}^2 }")

    FOLDER="saves/facescrub/${AT_REG}_wrn2810_cut${CUTLAYER}_lg${LOG_ENTROPY}_vt${VT}"
    FNAME="l${LAMBD}_n${NOISE}_ep${EPOCHS}_vt${VT}_ls${LS}_sl${SLOTS}_it${ITERS}_bk${BANK}_sd${SDIM}_wu${WARMUP}"
    HEADS=4

    cat > "${CFG}" << CFGEOF
exp_id=${EID}
arch=${ARCH}
cutlayer=${CUTLAYER}
bottleneck=${BOTTLENECK}
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
effective_strength=${EFF}
threshold=${THR}
train_status=PENDING
attack_status=PENDING
CFGEOF

    {
        printf '\n'
        printf '%.0s=' {1..72}; printf '\n'
        printf ' %s  [%s]  cutlayer=%s  bottleneck=%s\n' "${EID}" "${ARCH}" "${CUTLAYER}" "${BOTTLENECK}"
        printf '   lambd=%-4s  noise=%-5s  var_thr=%-5s  loss_scale=%s  at_str=%s\n' \
               "${LAMBD}" "${NOISE}" "${VT}" "${LS}" "${AT_REG_STR}"
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
        --bottleneck_option="${BOTTLENECK}" \
        --folder="${FOLDER}" \
        --var_threshold="${VT}" \
        --test_best \
        --attack_epochs="${ATTACK_EPOCHS}" \
        --gan_AE_type="${TEST_AE}" \
        --gan_loss_type="${GAN_LOSS}" \
        --attack_loss_type=MSE \
        --attention_num_slots="${SLOTS}" \
        --attention_num_heads="${HEADS}" \
        --attention_num_iterations="${ITERS}" \
        --attention_loss_scale="${LS}" \
        --attention_warmup_epochs="${WARMUP}" \
        --attention_bank_size="${BANK}" \
        --attention_slot_dim="${SDIM}" \
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

# ── Summary CSV ──────────────────────────────────────────────────────────────
lm "======================================================================"
lm " DONE  |  ${PASS} passed  |  ${FAIL} failed  |  ${TOTAL} total"
lm "======================================================================"

CSV="${RUN_DIR}/summary.csv"
echo "exp_id,arch,cutlayer,bottleneck,at_reg_str,lambd,noise,var_thr,loss_scale,slots,slot_dim,iters,bank,warmup,epochs,effective_strength,threshold,train_status,attack_status,best_acc,train_mse,train_ssim,train_psnr,infer_mse,infer_ssim,infer_psnr" > "${CSV}"

for CFG in "${RUN_DIR}"/exp*_config.txt; do
    [ -f "${CFG}" ] || continue
    EID=$(grep '^exp_id=' "${CFG}" | cut -d= -f2)
    EARCH=$(grep '^arch=' "${CFG}" | cut -d= -f2)
    ECUT=$(grep '^cutlayer=' "${CFG}" | cut -d= -f2)
    EBTL=$(grep '^bottleneck=' "${CFG}" | cut -d= -f2)
    EATSTR=$(grep '^at_reg_str=' "${CFG}" | cut -d= -f2)
    ELAMBD=$(grep '^lambd=' "${CFG}" | cut -d= -f2)
    ENOISE=$(grep '^noise=' "${CFG}" | cut -d= -f2)
    EVT=$(grep '^var_thr=' "${CFG}" | cut -d= -f2)
    ELS=$(grep '^loss_scale=' "${CFG}" | cut -d= -f2)
    ESLOTS=$(grep '^slots=' "${CFG}" | cut -d= -f2)
    ESDIM=$(grep '^slot_dim=' "${CFG}" | cut -d= -f2)
    EITERS=$(grep '^iters=' "${CFG}" | cut -d= -f2)
    EBANK=$(grep '^bank=' "${CFG}" | cut -d= -f2)
    EWARMUP=$(grep '^warmup=' "${CFG}" | cut -d= -f2)
    EEPOCHS=$(grep '^epochs=' "${CFG}" | cut -d= -f2)
    EEFF=$(grep '^effective_strength=' "${CFG}" | cut -d= -f2)
    ETHR=$(grep '^threshold=' "${CFG}" | cut -d= -f2)
    ETRAIN=$(grep '^train_status=' "${CFG}" | cut -d= -f2)
    EATTACK=$(grep '^attack_status=' "${CFG}" | cut -d= -f2)

    LOG="${RUN_DIR}/${EID}.log"
    BEST_ACC=$(grep -o 'Best Average Validation Accuracy is [0-9.]*' "${LOG}" 2>/dev/null | tail -1 | awk '{print $NF}')
    TRAIN_MSE=$(grep 'MSE Loss on ALL Image.*train' "${LOG}" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
    TRAIN_SSIM=$(grep 'SSIM Loss on ALL Image.*train' "${LOG}" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
    TRAIN_PSNR=$(grep 'PSNR Loss on ALL Image.*train' "${LOG}" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
    INFER_MSE=$(grep 'MSE Loss on ALL Image.*Real Attack' "${LOG}" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
    INFER_SSIM=$(grep 'SSIM Loss on ALL Image.*Real Attack' "${LOG}" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
    INFER_PSNR=$(grep 'PSNR Loss on ALL Image.*Real Attack' "${LOG}" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')

    echo "${EID},${EARCH:-},${ECUT:-},${EBTL:-},${EATSTR:-},${ELAMBD:-},${ENOISE:-},${EVT:-},${ELS:-},${ESLOTS:-},${ESDIM:-},${EITERS:-},${EBANK:-},${EWARMUP:-},${EEPOCHS:-},${EEFF:-},${ETHR:-},${ETRAIN:-},${EATTACK:-},${BEST_ACC:-},${TRAIN_MSE:-},${TRAIN_SSIM:-},${TRAIN_PSNR:-},${INFER_MSE:-},${INFER_SSIM:-},${INFER_PSNR:-}" >> "${CSV}"
done

lm "Summary CSV: ${CSV}"
