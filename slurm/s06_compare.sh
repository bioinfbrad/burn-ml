#!/bin/sh
#
#SBATCH --job-name=burn_compare
#
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -e
# SLURM runs a spool copy of this script, so $0 is useless here; the submit
# directory is where config.sh lives. Always submit from slurm/.
. "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.sh"
activate_env

# Paired comparison of every completed model on identical patient folds.
#
#   sbatch s06_compare.sh classification

TASK=${1:?usage: sbatch s06_compare.sh <task>}

python "${SCRIPTS}/06_compare_backbones.py" \
    --results-dir "${RESULTS}/${TASK}" --manifest "${MANIFEST}" \
    --task "${TASK}" --primary "${PRIMARY}" --n-boot "${BOOT}" --seed 42
