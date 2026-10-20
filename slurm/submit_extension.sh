#!/bin/sh
# submit_extension.sh — the pre-specified additions (ANALYSIS_PLAN.md), reusing the existing
# official results untouched.
#
#   cd slurm
#   SMOKE=1 sh submit_extension.sh    # FIRST: tiny settings, into results_smoke/
#   sh submit_extension.sh            # then the real run
#
# Submitted in priority order:
#   1. random-init EfficientNetB0 (the confirmatory comparison) + step-matched arm
#   2. paired comparison with confirmatory contrast and decomposition
#   3. repeated CV (pretrained x4, random-init x2) and their summary
#   4. learning curves (6/9/12 training patients x 3 subsamples)
#   5. Grad-CAM with sanity check; 3-member archival ensemble
#   6. sensitivity + post hoc reports for the EXISTING models
# GPU jobs carry GPU_RESERVATION; CPU jobs never do. Stops at the first refusal.

set -e
cd "$(dirname "$0")"
. ./config.sh
check_package || exit 1
mkdir -p logs

TASKS=${*:-"classification detection"}
P=${PRIMARY}
RI=${PRIMARY}_randominit
RIS=${PRIMARY}_randominit_steps

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: ${MANIFEST} not found. The extension reuses the manifest of the"
    echo "       official run; run submit_all.sh first (or SMOKE=1 for the smoke)."
    exit 1
fi

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
abort() {
    echo; echo "STOPPED: '$1' was refused by sbatch (error shown above)."
    [ -n "${SUBMITTED}" ] && { echo "Cancelling the ${N} job(s) already submitted:${SUBMITTED}"; scancel ${SUBMITTED} 2>/dev/null || true; }
    exit 1
}
sub() { label=$1; shift
        id=$(sbatch --parsable "$@" | cut -d';' -f1) || true
        [ -z "${id}" ] && abort "${label}"
        SUBMITTED="${SUBMITTED} ${id}"; N=$((N + 1)); LAST=${id}; }

[ "${SMOKE}" = "1" ] && echo "SMOKE TEST -> ${RESULTS}"
[ -n "${RES}" ] && echo "GPU jobs use reservation ${GPU_RESERVATION}"
echo

sub "s00 environment" ${RES} s00_env.sh;                                   ENV=${LAST}
sub "s02 self-test" ${RES} --dependency=afterok:${ENV} s02_selftest.sh;    T=${LAST}
echo "s00 environment ${ENV} | s02 self-test ${T} (gates everything)"

run() {   # run <label> <task> <model> <repeat> <subdir> [lck lcs] -> CV, AGG
    sub "cv $1" ${RES} --dependency=afterok:${T} --job-name=cv_${2}_${3}_r${4}${6:+_n$6s$7} \
        s03_cv_array.sh ${2} ${3} ${4} ${6:-} ${7:-};                      CV=${LAST}
    sub "aggregate $1" --dependency=afterok:${CV} --job-name=agg_${2}_${3}_r${4}${6:+_n$6s$7} \
        s04_aggregate.sh ${2} ${3} ${5};                                   AGG=${LAST}
}

for TASK in ${TASKS}; do
    echo; echo "=== ${TASK} ==="
    # 1. confirmatory random-init arm and step-matched secondary arm
    run "random-init" ${TASK} ${RI} 0 ${TASK}/${RI};   RI_CV=${CV}; RI_AGG=${AGG}
    sub "report ri" --dependency=afterok:${RI_AGG} --job-name=rep_${TASK}_${RI} \
        s05_report.sh ${TASK} ${RI};                    RI_REP=${LAST}
    run "random-init steps" ${TASK} ${RIS} 0 ${TASK}/${RIS}; RIS_AGG=${AGG}
    sub "report ris" --dependency=afterok:${RIS_AGG} --job-name=rep_${TASK}_${RIS} \
        s05_report.sh ${TASK} ${RIS};                   RIS_REP=${LAST}
    echo "  random-init (confirmatory)   cv ${RI_CV}  report ${RI_REP}"
    echo "  random-init, step-matched    report ${RIS_REP}"
    # 2. comparison with confirmatory contrast + decomposition
    sub "compare" --dependency=afterok:${RI_REP}:${RIS_REP} --job-name=cmp_${TASK} \
        s06_compare.sh ${TASK};                         echo "  comparison                   ${LAST}"
    # 3. repeated CV
    WAIT=":${RI_AGG}"; PRE_R=""
    for R in ${REPEATS_PRETRAINED}; do
        run "pretrained r${R}" ${TASK} ${P} ${R} ${TASK}/repeats/r${R}/${P}
        WAIT="${WAIT}:${AGG}"; [ "${R}" = "${GRADCAM_REPEAT}" ] && PRE_R=${CV}
    done
    for R in ${REPEATS_RANDOMINIT}; do
        run "random-init r${R}" ${TASK} ${RI} ${R} ${TASK}/repeats/r${R}/${RI}
        WAIT="${WAIT}:${AGG}"
    done
    sub "repeats summary" --dependency=afterok${WAIT} --job-name=reps_${TASK} \
        s09_repeats.sh ${TASK};                         echo "  repeats summary              ${LAST}"
    # 4. learning curves
    LW=""
    for K in ${LC_PATIENTS}; do for S in ${LC_SEEDS}; do
        run "learning n${K} s${S}" ${TASK} ${P} 0 ${TASK}/learning_curve/${P}/n${K}_s${S} ${K} ${S}
        LW="${LW}:${AGG}"
    done; done
    sub "learning summary" --dependency=afterok${LW} --job-name=lc_${TASK} \
        s11_learning.sh ${TASK};                        echo "  learning-curve summary       ${LAST}"
    # 5. Grad-CAM and archival ensemble (GPU)
    [ -n "${PRE_R}" ] && { sub "gradcam pretrained" ${RES} --dependency=afterok:${PRE_R} \
        --job-name=cam_${TASK}_${P} s10_gradcam.sh ${TASK} ${TASK}/repeats/r${GRADCAM_REPEAT}/${P}
        echo "  Grad-CAM pretrained          ${LAST}"; }
    sub "gradcam random-init" ${RES} --dependency=afterok:${RI_CV} --job-name=cam_${TASK}_${RI} \
        s10_gradcam.sh ${TASK} ${TASK}/${RI};           echo "  Grad-CAM random-init         ${LAST}"
    sub "archival" ${RES} --dependency=afterok:${T} --job-name=final_${TASK}_${P} \
        s07_final.sh ${TASK} ${P};                      echo "  archival 3-member ensemble   ${LAST}"
    # 6. sensitivity + post hoc for the models that already exist
    for M in ${BACKBONES} scratch; do
        if [ -f "${RESULTS}/${TASK}/${M}/metrics.json" ]; then
            sub "report ${M}" --dependency=afterok:${T} --job-name=rep_${TASK}_${M} \
                s05_report.sh ${TASK} ${M}
        fi
    done
    echo "  reports refreshed for existing models"
done
echo; echo "Submitted ${N} jobs."
