#!/bin/sh
#
#SBATCH --job-name=burn_selftest
#
#SBATCH --partition=Nvidia
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -e
# SLURM runs a spool copy of this script, so $0 is useless here; the submit
# directory is where config.sh lives. Always submit from slurm/.
. "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.sh"
activate_env

# Runs on a GPU node: TensorFlow needs AVX, which is guaranteed on the nodes
# that do the real training but not necessarily elsewhere.
#
# 60 correctness checks on the real data, including one end-to-end scratch
# fold and one end-to-end transfer fold. Exits non-zero on any failure, which
# holds back every job below it.
python "${SCRIPTS}/99_selftest.py" --raw-dir "${RAW}" --manifest "${MANIFEST}" \
    --task classification --folds "${FOLDS}" --full --end-to-end
python "${SCRIPTS}/99_selftest.py" --raw-dir "${RAW}" --manifest "${MANIFEST}" \
    --task detection --folds "${FOLDS}"
