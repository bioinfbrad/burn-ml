#!/bin/sh
#
#SBATCH --job-name=burn_aggregate
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

# Pools the per-fold files. Refuses to write a result while any crop is
# unpredicted, or if folds were run with different settings.
#
#   sbatch s04_aggregate.sh classification efficientnetb0

TASK=${1:?usage: sbatch s04_aggregate.sh <task> <model> [run-subdir]}
MODEL=${2:?usage: sbatch s04_aggregate.sh <task> <model> [run-subdir]}
DIR=${RESULTS}/${3:-${TASK}/${MODEL}}

python "${SCRIPTS}/03b_aggregate.py" --raw-dir "${RAW}" \
    --manifest "${MANIFEST}" --task "${TASK}" \
    --out-dir "${DIR}" --n-boot "${BOOT}" --seed 42
