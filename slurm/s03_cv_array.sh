#!/bin/sh
#
#SBATCH --job-name=burn_cv
#
#SBATCH --partition=Nvidia
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=1-6%3
#SBATCH --output=logs/%x_%j.out

set -e
# SLURM runs a spool copy of this script, so $0 is useless here; the submit
# directory is where config.sh lives. Always submit from slurm/.
. "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.sh"
activate_env

# ONE FOLD PER ARRAY TASK: memory is reclaimed between folds, a crash costs one
# fold, folds run in parallel. %3 caps concurrency at three GPUs. --array must
# match FOLDS in config.sh. Resubmitting skips folds already written.
#
#   sbatch s03_cv_array.sh classification efficientnetb0
#   sbatch s03_cv_array.sh detection scratch

TASK=${1:?usage: sbatch s03_cv_array.sh <task> <model> [repeat] [lc_patients lc_seed]}
MODEL=${2:?usage: sbatch s03_cv_array.sh <task> <model> [repeat] [lc_patients lc_seed]}
REPEAT=${3:-0}
LCK=${4:-}
LCS=${5:-}
FOLD=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}

# where this run lives
if [ -n "${LCK}" ]; then
    OUT=${RESULTS}/${TASK}/learning_curve/${MODEL}/n${LCK}_s${LCS}
    LC="--train-patients ${LCK} --subsample-seed ${LCS}"
    SAVE=""
elif [ "${REPEAT}" = "0" ]; then
    OUT=${RESULTS}/${TASK}/${MODEL}
    LC=""
    case "${MODEL}" in *_randominit_widelr|*_randominit_bracket) SAVE="" ;; *_randominit*) SAVE="--save-models" ;; *) SAVE="" ;; esac
else
    OUT=${RESULTS}/${TASK}/repeats/r${REPEAT}/${MODEL}
    LC=""
    SAVE="--save-models"
fi
COMMON="--raw-dir ${RAW} --manifest ${MANIFEST} --task ${TASK} --out-dir ${OUT} \
        --folds ${FOLDS} --only-fold ${FOLD} --repeat ${REPEAT} \
        --n-boot ${BOOT} --seed 42 ${SAVE} ${LC}"

case "${MODEL}" in
scratch)
    # the original custom CNN (legacy pipeline), nested search per fold
    python "${SCRIPTS}/03_run_grouped_cv.py" ${COMMON} --arch scratch \
        --nested-search --search-calls 30 --search-folds 3 --search-epochs 40 \
        --rotations 24 --epochs 100 --patience 15 \
        --loader auto --memory-budget-gb "${MEM_BUDGET_GB}" ${SCRATCH_EXTRA} ;;
*_randominit_widelr)
    # POST HOC sensitivity analysis (not in the written plan): the registered
    # grid's top value (3e-3) was chosen in 26 of 48 folds and cross-entropy
    # was still falling there in 21, so the grid is extended to 1e-2 and 3e-2.
    # Everything else is identical to the primary random-init arm.
    python "${SCRIPTS}/03_run_grouped_cv.py" ${COMMON} --arch randominit \
        --backbone "${MODEL%_randominit_widelr}" --inner-folds 3 \
        --ri-epochs "${RI_EPOCHS}" --warmup-epochs 5 --lr-grid ${RI_WIDE_LR_GRID} \
        --weight-decay 1e-4 --passes-per-epoch 4 --dropout 0.3 --batch-size 16 \
        --label-smoothing 0.05 --tta 8 ${RI_EXTRA} ;;
*_randominit_bracket)
    # POST HOC bracketing step: the extended grid's top value (3e-2) was
    # still chosen in 10 of 12 folds, so two higher rates are tried. Only the
    # NEW rates are trained; the choice is made over the union of both grids
    # by the same rule, and the earlier run's predictions are kept unchanged
    # wherever its choice still has the lowest inner cross-entropy.
    python "${SCRIPTS}/03_run_grouped_cv.py" ${COMMON} --arch randominit \
        --backbone "${MODEL%_randominit_bracket}" --inner-folds 3 \
        --extend-from "${RESULTS}/${TASK}/${MODEL%_bracket}_widelr" \
        --ri-epochs "${RI_EPOCHS}" --warmup-epochs 5 --lr-grid ${RI_BRACKET_LR_GRID} \
        --weight-decay 1e-4 --passes-per-epoch 4 --dropout 0.3 --batch-size 16 \
        --label-smoothing 0.05 --tta 8 ${RI_EXTRA} ;;
*_randominit_steps)
    # secondary random-init arm matched on optimiser steps (45 epochs)
    python "${SCRIPTS}/03_run_grouped_cv.py" ${COMMON} --arch randominit \
        --backbone "${MODEL%_randominit_steps}" --inner-folds 3 \
        --ri-epochs "${RI_STEPS_EPOCHS}" --warmup-epochs 5 --lr-grid ${RI_LR_GRID} \
        --weight-decay 1e-4 --passes-per-epoch 4 --dropout 0.3 --batch-size 16 \
        --label-smoothing 0.05 --tta 8 ${RI_EXTRA} ;;
*_randominit)
    # PRE-SPECIFIED random-initialisation arm: same architecture and
    # protocol; all layers trained, BatchNorm in training mode, own LR grid,
    # longer schedule
    python "${SCRIPTS}/03_run_grouped_cv.py" ${COMMON} --arch randominit \
        --backbone "${MODEL%_randominit}" --inner-folds 3 \
        --ri-epochs "${RI_EPOCHS}" --warmup-epochs 5 --lr-grid ${RI_LR_GRID} \
        --weight-decay 1e-4 --passes-per-epoch 4 --dropout 0.3 --batch-size 16 \
        --label-smoothing 0.05 --tta 8 ${RI_EXTRA} ;;
*)
    # PRE-SPECIFIED transfer recipe. Do not tune these on the 18 patients.
    python "${SCRIPTS}/03_run_grouped_cv.py" ${COMMON} --arch transfer \
        --backbone "${MODEL}" --weights "${WEIGHTS}" --inner-folds 3 \
        --probe-epochs 15 --finetune-epochs 30 --passes-per-epoch 4 \
        --probe-lr 1e-3 --finetune-lr 1e-5 \
        --dropout 0.3 --batch-size 16 --label-smoothing 0.05 \
        --tta 8 ${TL_EXTRA} ;;
esac
