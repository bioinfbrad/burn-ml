#!/bin/sh
# submit_sensitivity.sh — POST HOC learning-rate sensitivity analysis.
#
# Not in the written analysis plan. Added because the random-init arm chose
# the top of its grid (3e-3) in 26 of 48 folds, with cross-entropy still
# falling there in 21, so the arm may have been under-tuned. Reruns the
# random-init arm at the original split with the grid extended to
# {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}; everything else identical.
#
#   cd slurm
#   SMOKE=1 sh submit_sensitivity.sh   # minutes; tiny settings, results_smoke/
#   sh submit_sensitivity.sh           # ~20 GPU-hours, then refreshes run 1
#   sh submit_sensitivity.sh bracket   # ~8 GPU-hours: adds 1e-1 and 3e-1 on top
#                                      # of the extended grid, then refreshes run 1
#
# Fail-fast; GPU jobs carry GPU_RESERVATION, CPU jobs never do.

set -e
cd "$(dirname "$0")"
. ./config.sh
check_package || exit 1
mkdir -p logs
STEP=${1:-widelr}
case "${STEP}" in
    widelr)  ARM=${PRIMARY}_randominit_widelr;  GRID=${RI_WIDE_LR_GRID} ;;
    bracket) ARM=${PRIMARY}_randominit_bracket; GRID="${RI_WIDE_LR_GRID} + ${RI_BRACKET_LR_GRID}"
             for T in classification detection; do
                 [ -f "${RESULTS}/${T}/${PRIMARY}_randominit_widelr/folds/fold_6.json" ] || {
                     echo "ERROR: the bracketing step extends ${RESULTS}/${T}/${PRIMARY}_randominit_widelr,"
                     echo "       which is missing or incomplete. Run 'sh submit_sensitivity.sh' first."; exit 1; }
             done ;;
    *) echo "usage: sh submit_sensitivity.sh [widelr|bracket]"; exit 1 ;;
esac

[ -f "${MANIFEST}" ] || { echo "ERROR: ${MANIFEST} not found."; exit 1; }
RES=""
if [ -n "${GPU_RESERVATION}" ]; then
    RES="--reservation=${GPU_RESERVATION}"
    if command -v scontrol >/dev/null 2>&1 && \
       ! scontrol show reservation "${GPU_RESERVATION}" >/dev/null 2>&1; then
        echo "ERROR: reservation '${GPU_RESERVATION}' not found (expired?). Set"
        echo "       GPU_RESERVATION= (empty) in slurm/config.sh, or fix the name."
        exit 1
    fi
fi
SUBMITTED=""; N=0
abort() { echo; echo "STOPPED: '$1' was refused by sbatch (error above)."
          [ -n "${SUBMITTED}" ] && { echo "Cancelling:${SUBMITTED}"; scancel ${SUBMITTED} 2>/dev/null || true; }
          exit 1; }
sub() { label=$1; shift
        id=$(sbatch --parsable "$@" | cut -d';' -f1) || true
        [ -z "${id}" ] && abort "${label}"
        SUBMITTED="${SUBMITTED} ${id}"; N=$((N + 1)); LAST=${id}; }

[ "${SMOKE}" = "1" ] && echo "SMOKE TEST -> ${RESULTS}"
echo "POST HOC sensitivity: ${ARM}, learning-rate grid {${GRID}}"
sub "s00 environment" ${RES} s00_env.sh; ENV=${LAST}
AGGS=""
for TASK in classification detection; do
    sub "cv ${TASK}" ${RES} --dependency=afterok:${ENV} --job-name=cv_${TASK}_${ARM} \
        s03_cv_array.sh ${TASK} ${ARM} 0;                                    CV=${LAST}
    sub "aggregate ${TASK}" --dependency=afterok:${CV} --job-name=agg_${TASK}_${ARM} \
        s04_aggregate.sh ${TASK} ${ARM};                                     AGG=${LAST}
    sub "report ${TASK}" --dependency=afterok:${AGG} --job-name=rep_${TASK}_${ARM} \
        s05_report.sh ${TASK} ${ARM}
    AGGS="${AGGS}:${AGG}"
    echo "  ${TASK}: cv ${CV}  aggregate ${AGG}  report ${LAST}"
done
if [ "${SMOKE}" != "1" ]; then
    sub "run-1 refresh" --dependency=afterok${AGGS} s12_run1_reports.sh
    echo "  run-1 comparison refresh ${LAST}  -> ${RUN1_DIR}/<task>/comparison/"
else
    echo "  (smoke: run-1 refresh skipped so run1/ is never touched by smoke output)"
fi
echo; echo "Submitted ${N} jobs."
