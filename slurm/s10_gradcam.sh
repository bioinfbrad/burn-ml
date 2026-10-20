#!/bin/sh
#
#SBATCH --job-name=burn_gradcam
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

# Grad-CAM with the randomisation sanity check, on archived fold models.
#   sbatch s10_gradcam.sh classification <run-subdir>
TASK=${1:?usage: sbatch s10_gradcam.sh <task> <run-subdir>}
python "${SCRIPTS}/10_gradcam.py" --run-dir "${RESULTS}/${2:?run-subdir}" \
    --raw-dir "${RAW}" --task "${TASK}" --seed 42
