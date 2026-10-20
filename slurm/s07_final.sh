#!/bin/sh
#
#SBATCH --job-name=burn_final
#
#SBATCH --partition=Nvidia
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -e
# SLURM runs a spool copy of this script, so $0 is useless here; the submit
# directory is where config.sh lives. Always submit from slurm/.
. "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.sh"
activate_env

# Archival model for the Zenodo deposit: the primary backbone trained on all
# patients with the same recipe, using the median CV temperature. Never the
# source of a reported number.
#
#   sbatch s07_final.sh classification efficientnetb0

TASK=${1:?usage: sbatch s07_final.sh <task> <backbone>}
MODEL=${2:?usage: sbatch s07_final.sh <task> <backbone>}
CV=${RESULTS}/${TASK}/${MODEL}

python "${SCRIPTS}/04_train_final.py" --raw-dir "${RAW}" \
    --manifest "${MANIFEST}" --task "${TASK}" --arch transfer \
    --backbone "${MODEL}" --weights "${WEIGHTS}" --cv-dir "${CV}" \
    --out-dir "${CV}/final" \
    --probe-epochs 15 --finetune-epochs 30 --passes-per-epoch 4 \
    --probe-lr 1e-3 --finetune-lr 1e-5 --dropout 0.3 --batch-size 16 \
    --label-smoothing 0.05 --seed 42 ${TL_EXTRA_FINAL}
