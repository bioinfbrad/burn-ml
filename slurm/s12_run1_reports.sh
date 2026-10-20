#!/bin/sh
#
#SBATCH --job-name=burn_run1_reports
#
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -e
# SLURM runs a spool copy of this script, so $0 is useless here; the submit
# directory is where config.sh lives. Always submit from slurm/.
. "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.sh"
activate_env

# Refresh the run-1 (primary of record) comparison with every random-init arm
# that exists in results/, including the post hoc wide-grid arm. Run 1's own
# predictions are never modified; only the random-init arms are copied in.
#   sbatch s12_run1_reports.sh
[ -d "${RUN1_DIR}" ] || { echo "ERROR: ${RUN1_DIR} not found (unzip results_run1.zip -d run1)"; exit 1; }
for T in classification detection; do
    for SRC in "${RESULTS}/${T}/${PRIMARY}"_randominit*; do
        [ -f "${SRC}/metrics.json" ] || continue
        M=$(basename "${SRC}")
        # copy with Python: it is on every node (the pipeline runs on it),
        # whereas tar and rsync are missing from some compute-node images
        python - "${SRC}" "${RUN1_DIR}/${T}/${M}" <<'PYCOPY'
import shutil, sys
shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("models"))
PYCOPY
        python "${SCRIPTS}/05_make_report.py" --results-dir "${RUN1_DIR}/${T}/${M}" --manifest "${RUN1_DIR}/manifest.csv"
        python "${SCRIPTS}/07_sensitivity.py" --results-dir "${RUN1_DIR}/${T}/${M}" --n-boot "${BOOT}"
        python "${SCRIPTS}/08_posthoc.py" --results-dir "${RUN1_DIR}/${T}/${M}" --manifest "${RUN1_DIR}/manifest.csv" --n-boot "${BOOT}"
    done
    [ -e "${RUN1_DIR}/${T}/repeats" ] || ln -s "${RESULTS}/${T}/repeats" "${RUN1_DIR}/${T}/repeats"
    python "${SCRIPTS}/06_compare_backbones.py" --results-dir "${RUN1_DIR}/${T}" \
        --manifest "${RUN1_DIR}/manifest.csv" --task "${T}" --primary "${PRIMARY}" --n-boot "${BOOT}"
    python "${SCRIPTS}/09_repeats.py" --task-dir "${RUN1_DIR}/${T}" --primary "${PRIMARY}" \
        --comparator "${PRIMARY}_randominit" --n-boot "${BOOT}"
done
