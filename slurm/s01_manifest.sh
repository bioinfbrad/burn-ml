#!/bin/sh
#
#SBATCH --job-name=burn_manifest
#
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -e
# SLURM runs a spool copy of this script, so $0 is useless here; the submit
# directory is where config.sh lives. Always submit from slurm/.
. "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.sh"
activate_env

if [ -n "${PATIENT_MAP}" ]; then
    python "${SCRIPTS}/01_prepare_manifest.py" --raw-dir "${RAW}" \
        --out "${MANIFEST}" --patient-map "${PATIENT_MAP}"
else
    python "${SCRIPTS}/01_prepare_manifest.py" --raw-dir "${RAW}" \
        --out "${MANIFEST}"
fi
