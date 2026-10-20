#!/bin/sh
#
#SBATCH --job-name=burn_repeats
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

#   sbatch s09_repeats.sh classification
TASK=${1:?usage: sbatch s09_repeats.sh <task>}
python "${SCRIPTS}/09_repeats.py" --task-dir "${RESULTS}/${TASK}" \
    --primary "${PRIMARY}" --comparator "${PRIMARY}_randominit" --n-boot "${BOOT}"
