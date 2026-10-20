#!/usr/bin/env bash
# =============================================================================
# run_local.sh: run the study on ONE machine, without SLURM.
#
# It runs the same job scripts as the cluster (slurm/s0*.sh), one after the
# other, so local and cluster results come from identical commands. A GPU is
# used if TensorFlow can see one; otherwise everything runs on the CPU.
#
#   bash run_local.sh --smoke                 # minutes: check that everything works
#   bash run_local.sh                         # primary model (EfficientNetB0), both tasks
#   bash run_local.sh --all                   # every network in the paper (days on a CPU)
#   bash run_local.sh --tasks detection       # one task only
#   bash run_local.sh --raw-dir /path/to/raw_Images
#
# Activate your Python environment first (see README: pip install -r requirements.txt).
# Progress is logged to logs_local/; results go to results/ (results_smoke/ with --smoke).
# =============================================================================
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TASKS="classification detection"; MODELS=""; ALL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke)   export SMOKE=1 ;;
        --all)     ALL=1 ;;
        --tasks)   TASKS="$2"; shift ;;
        --models)  MODELS="$2"; shift ;;
        --raw-dir) export BURNML_RAW="$(cd "$2" && pwd)"; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)"; exit 2 ;;
    esac
    shift
done
export BURNML_PROJECT="${ROOT}" BURNML_ENV_MODE=current SLURM_SUBMIT_DIR="${ROOT}/slurm"
cd "${ROOT}/slurm"
. ./config.sh
check_package
[ -d "${RAW}" ] || { echo "ERROR: no image folder at ${RAW} (see data/README.md, or use --raw-dir)"; exit 1; }
if [ -z "${MODELS}" ]; then
    MODELS="${PRIMARY}"
    [ "${ALL}" = "1" ] && MODELS="${BACKBONES} scratch"
fi
LOGS="${ROOT}/logs_local"; mkdir -p "${LOGS}"
GPUS=$(python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null || echo 0)
echo "burn-ml local run | tasks: ${TASKS} | models: ${MODELS} | GPUs: ${GPUS} | smoke: ${SMOKE}"
echo "results: ${RESULTS}   logs: ${LOGS}"
[ "${GPUS}" = "0" ] && [ "${SMOKE}" != "1" ] && echo "NOTE: no GPU found; the full analysis will take a long time on a CPU."

step() {   # step <log-name> <command...>
    local name="$1"; shift
    printf "%-48s" "  ${name}"
    if "$@" > "${LOGS}/${name}.log" 2>&1; then echo "done"
    else echo "FAILED (see logs_local/${name}.log)"; tail -20 "${LOGS}/${name}.log"; exit 1; fi
}

STRICT=""; [ "${GPUS}" != "0" ] && STRICT="--strict"
CHECK=""; [ "${WEIGHTS}" = "imagenet" ] && CHECK="--check-weights ${MODELS/scratch/}"
step 00_environment python "${SCRIPTS}/00_check_env.py" --out-dir "${RESULTS}" ${STRICT} ${CHECK}
step 01_manifest   bash s01_manifest.sh
step 02_selftest   bash s02_selftest.sh
for TASK in ${TASKS}; do
    n=0
    for MODEL in ${MODELS}; do
        n=$((n + 1))
        for FOLD in $(seq 1 "${FOLDS}"); do
            SLURM_ARRAY_TASK_ID=${FOLD} step "03_cv_${TASK}_${MODEL}_fold${FOLD}" bash s03_cv_array.sh "${TASK}" "${MODEL}"
        done
        step "04_aggregate_${TASK}_${MODEL}" bash s04_aggregate.sh "${TASK}" "${MODEL}"
        step "05_report_${TASK}_${MODEL}"    bash s05_report.sh "${TASK}" "${MODEL}"
        if [ "${MODEL}" = "${PRIMARY}" ]; then
            step "07_final_${TASK}_${MODEL}" bash s07_final.sh "${TASK}" "${MODEL}"
        fi
    done
    [ "${n}" -gt 1 ] && step "06_compare_${TASK}" bash s06_compare.sh "${TASK}"
done
echo
echo "Finished. Results are in ${RESULTS}."
echo "To package the trained models for prediction:  python scripts/export_models.py --results ${RESULTS}"
