#!/bin/sh
# submit_all.sh — submit the whole study with dependencies.
#
#   cd slurm
#   SMOKE=1 sh submit_all.sh     # FIRST: whole chain, tiny settings, minutes
#   sh submit_all.sh             # then the real run, both tasks
#   sh submit_all.sh detection   # or a single task
#
# GPU jobs get GPU_RESERVATION from config.sh; CPU jobs never do.
# Stops at the first refused submission, so a problem never cascades into
# dozens of jobs with broken dependencies.
#
# Progress: squeue -u $USER        Logs: slurm/logs/
# Resuming: run it again; folds already written are skipped.

set -e
cd "$(dirname "$0")"
. ./config.sh
check_package || exit 1
mkdir -p logs

TASKS=${*:-"classification detection"}
MODELS="${BACKBONES}"
[ "${RUN_SCRATCH_BASELINE}" = "yes" ] && MODELS="${MODELS} scratch"

RES=""
if [ -n "${GPU_RESERVATION}" ]; then
    RES="--reservation=${GPU_RESERVATION}"
    if command -v scontrol >/dev/null 2>&1 && \
       ! scontrol show reservation "${GPU_RESERVATION}" >/dev/null 2>&1; then
        echo "ERROR: reservation '${GPU_RESERVATION}' not found (expired?)."
        echo "       Set GPU_RESERVATION= (empty) in slurm/config.sh, or fix the name."
        exit 1
    fi
fi

SUBMITTED=""
abort() {
    echo
    echo "STOPPED: '$1' was refused by sbatch (error shown above)."
    if [ -n "${SUBMITTED}" ]; then
        echo "Cancelling the ${2} job(s) already submitted in this run:${SUBMITTED}"
        scancel ${SUBMITTED} 2>/dev/null || true
    fi
    exit 1
}
N=0
sub() {         # sub <label> <sbatch args...>   -> prints the job id
    label=$1; shift
    id=$(sbatch --parsable "$@" | cut -d';' -f1) || true
    if [ -z "${id}" ]; then abort "${label}" "${N}"; fi
    SUBMITTED="${SUBMITTED} ${id}"; N=$((N + 1))
    LAST=${id}
}
gpu() { sub "$@"; }                       # GPU job: reservation added below
# (kept as separate names so the rule is visible at each call site)

[ "${SMOKE}" = "1" ] && echo "SMOKE TEST -> ${RESULTS}"
[ -n "${RES}" ] && echo "GPU jobs use reservation ${GPU_RESERVATION}"
echo

gpu "s00 environment" ${RES} s00_env.sh;                               ENV_ID=${LAST}
sub "s01 manifest" --dependency=afterok:${ENV_ID} s01_manifest.sh;     MAN_ID=${LAST}
gpu "s02 self-test" ${RES} --dependency=afterok:${MAN_ID} s02_selftest.sh; TEST_ID=${LAST}
echo "s00 environment + weights   ${ENV_ID}   (GPU)"
echo "s01 manifest                ${MAN_ID}   (CPU)"
echo "s02 self-test               ${TEST_ID}   (GPU, gates everything below)"

for TASK in ${TASKS}; do
    echo
    echo "=== ${TASK} ==="
    REPORTS=""
    for MODEL in ${MODELS}; do
        gpu "cv ${TASK} ${MODEL}" ${RES} --dependency=afterok:${TEST_ID} \
            --job-name=cv_${TASK}_${MODEL} s03_cv_array.sh ${TASK} ${MODEL}; CV=${LAST}
        sub "aggregate ${TASK} ${MODEL}" --dependency=afterok:${CV} \
            --job-name=agg_${TASK}_${MODEL} s04_aggregate.sh ${TASK} ${MODEL}; AGG=${LAST}
        sub "report ${TASK} ${MODEL}" --dependency=afterok:${AGG} \
            --job-name=rep_${TASK}_${MODEL} s05_report.sh ${TASK} ${MODEL}; REP=${LAST}
        REPORTS="${REPORTS}:${REP}"
        LINE=$(printf "  %-15s cv %-8s aggregate %-8s report %-8s" \
            "${MODEL}" "${CV}" "${AGG}" "${REP}")
        if [ "${MODEL}" = "${PRIMARY}" ]; then
            gpu "archival ${TASK} ${MODEL}" ${RES} --dependency=afterok:${AGG} \
                --job-name=final_${TASK}_${MODEL} s07_final.sh ${TASK} ${MODEL}; FIN=${LAST}
            LINE="${LINE} archival ${FIN}   <- pre-specified primary"
        fi
        echo "${LINE}"
    done
    sub "comparison ${TASK}" --dependency=afterok${REPORTS} \
        --job-name=cmp_${TASK} s06_compare.sh ${TASK}; CMP=${LAST}
    echo "  comparison      ${CMP}"
done

echo
echo "Submitted ${N} jobs. Deliverables:"
echo "  ${RESULTS}/<task>/<model>/report/numbers_for_text.txt"
echo "  ${RESULTS}/<task>/comparison/backbone_comparison.md"
echo "  ${RESULTS}/<task>/${PRIMARY}/final/   (archival model)"
