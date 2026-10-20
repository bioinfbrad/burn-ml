#!/bin/sh
#
#SBATCH --job-name=burn_report
#
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -e
# SLURM runs a spool copy of this script, so $0 is useless here; the submit
# directory is where config.sh lives. Always submit from slurm/.
. "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.sh"
activate_env

#   sbatch s05_report.sh classification efficientnetb0

TASK=${1:?usage: sbatch s05_report.sh <task> <model> [run-subdir]}
MODEL=${2:?usage: sbatch s05_report.sh <task> <model> [run-subdir]}
DIR=${RESULTS}/${3:-${TASK}/${MODEL}}

python "${SCRIPTS}/05_make_report.py" --results-dir "${DIR}" --manifest "${MANIFEST}"
# patient-weighted metrics, dominant-patient exclusion, calibration CIs, forest plot
python "${SCRIPTS}/07_sensitivity.py" --results-dir "${DIR}" --n-boot "${BOOT}" --seed 42
# selective prediction, aggregation, clustered AUC
python "${SCRIPTS}/08_posthoc.py" --results-dir "${DIR}" --manifest "${MANIFEST}" \
    --n-boot "${BOOT}" --seed 42
