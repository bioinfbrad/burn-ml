#!/bin/sh
#
#SBATCH --job-name=burn_learning
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

#   sbatch s11_learning.sh classification
TASK=${1:?usage: sbatch s11_learning.sh <task>}
python "${SCRIPTS}/11_learning_curves.py" --task-dir "${RESULTS}/${TASK}" --model "${PRIMARY}"
