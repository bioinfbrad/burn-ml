#!/bin/sh
#
#SBATCH --job-name=burn_env
#
#SBATCH --partition=Nvidia
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -e
# SLURM runs a spool copy of this script, so $0 is useless here; the submit
# directory is where config.sh lives. Always submit from slurm/.
. "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.sh"
activate_env

# Proves the GPU executes kernels (not a silent CPU fallback) and that every
# pre-specified backbone's ImageNet weights load OFFLINE. Fails the chain
# otherwise, before any GPU hours are spent.
if [ "${WEIGHTS}" = "imagenet" ]; then
    CHECK="--check-weights ${BACKBONES}"
else
    CHECK=""
fi
python "${SCRIPTS}/00_check_env.py" --out-dir "${RESULTS}" --strict ${CHECK}
nvidia-smi || true
