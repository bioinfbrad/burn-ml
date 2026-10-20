#!/bin/sh
# config.sh — the ONLY file you need to edit. Sourced by every job.
#
# Jobs must be submitted from this directory (submit_all.sh does that), because
# SLURM runs a copy of each script from its spool directory and the jobs find
# this file through $SLURM_SUBMIT_DIR.

# ----------------------------------------------------------------- paths --- #
# PROJECT is found automatically from where this file lives, so the project
# can sit anywhere. Works when sourced interactively (". slurm/config.sh"),
# from submit_all.sh, inside a batch job, and under srun.
if [ -n "${BURNML_PROJECT:-}" ]; then
    PROJECT=${BURNML_PROJECT}
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/config.sh" ]; then
    PROJECT=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)
elif [ -n "${BASH_SOURCE:-}" ] && [ -f "${BASH_SOURCE}" ]; then
    PROJECT=$(cd "$(dirname "${BASH_SOURCE}")/.." && pwd)
elif [ -f "./slurm/config.sh" ]; then
    PROJECT=$(pwd)
elif [ -f "./config.sh" ] && [ -d "../scripts" ]; then
    PROJECT=$(cd .. && pwd)
else
    echo "ERROR: cannot locate the project. Source this file from the project"
    echo "       directory (. slurm/config.sh) or from slurm/ (. ./config.sh)."
    return 1 2>/dev/null || exit 1
fi
RAW=${PROJECT}/raw_Images
SCRIPTS=${PROJECT}/scripts

# Real patient identifiers, if you have them (CSV header: source_id,patient_id).
# Leave empty to cluster accession numbers automatically.
PATIENT_MAP=

# ----------------------------------------------------------- environment --- #
# conda  : the environment built by env/setup_env.sh (TF 2.15 or 2.14)
# module : the cluster's TensorFlow 2.6 module (EfficientNetB0, DenseNet121,
#          ResNet50V2 and MobileNetV2 are all available there too)
ENV_MODE=conda
CONDA_ENV=burnml
TF_MODULE=tensorflow-gpu
# The environment's absolute path is recorded by env/setup_env.sh in
# env/env_prefix.txt; jobs put it first on PATH directly.

# ImageNet weights live in ~/.keras/models: Keras 2.15 always reads them
# from there (KERAS_HOME does not move them), and home is mounted on the
# compute nodes. Fetch them with scripts/00b_fetch_weights.py.

# ------------------------------------------------------------ scheduling --- #
# Your GPU reservation. Applied by submit_all.sh to the GPU jobs ONLY
# (s00, s02, s03, s07): a reservation covers the GPU node, so a compute-
# partition job that carries it is refused ("Requested node configuration
# is not available"). Set it to empty when the reservation ends.
GPU_RESERVATION=${BURNML_RESERVATION-}

# ---------------------------------------------------------------- design --- #
FOLDS=6                          # must match --array in s03_cv_array.sh
MEM_BUDGET_GB=60                 # 20 CPUs carry ~90 GB on this cluster

# The PRE-SPECIFIED comparison set, fixed before running anything. PRIMARY is
# designated in advance as the paper's main result.
BACKBONES=${BURNML_BACKBONES:-"efficientnetb0 densenet121 resnet50v2 mobilenetv2"}
PRIMARY=${BURNML_PRIMARY:-efficientnetb0}
RUN_SCRATCH_BASELINE=${BURNML_SCRATCH:-yes}   # original CNN, as the baseline

# imagenet for every real result. 'none' exists only for testing the
# pipeline on a machine without the weights; it is recorded in metrics.json.
WEIGHTS=${BURNML_WEIGHTS:-imagenet}

# ------------------------------------------------------------ extension --- #
# Pre-specified additions (submit_extension.sh; plan in ANALYSIS_PLAN.md). The existing official
# results are reused untouched; only new runs are submitted.
RI_LR_GRID="3e-4 1e-3 3e-3"     # random-init learning rates, chosen per fold
RI_EPOCHS=150                   # ~3.3x the pretrained arm's optimiser steps
RI_STEPS_EPOCHS=45              # secondary arm matched on optimiser steps
REPEATS_PRETRAINED="1 2 3 4"    # extra CV repeats (repeat 0 = original split)
REPEATS_RANDOMINIT="1 2"
LC_PATIENTS="6 9 12"            # learning curve: training patients per fold
LC_SEEDS="1 2 3"                # subsamples per size
GRADCAM_REPEAT=1                # pretrained models archived from this repeat
# Post hoc sensitivity (submit_sensitivity.sh): extended random-init grid
RI_WIDE_LR_GRID="3e-4 1e-3 3e-3 1e-2 3e-2"
RI_BRACKET_LR_GRID="1e-1 3e-1"  # bracketing step, trained on top of the above
# Run 1, the primary of record, for the regenerated reports
RUN1_DIR=${PROJECT}/run1/results

# ----------------------------------------------------------- smoke test --- #
# SMOKE=1 runs the ENTIRE chain with tiny settings into results_smoke/, in
# minutes rather than days. Do this once before the real run:
#     SMOKE=1 sh submit_all.sh
SMOKE=${SMOKE:-0}
if [ "${SMOKE}" = "1" ]; then
    RESULTS=${PROJECT}/results_smoke
    TL_EXTRA="--probe-epochs 1 --finetune-epochs 1 --passes-per-epoch 1 --inner-folds 2 --tta 2"
    TL_EXTRA_FINAL="--probe-epochs 1 --finetune-epochs 1 --passes-per-epoch 1"
    SCRATCH_EXTRA="--search-calls 3 --search-folds 2 --search-epochs 1 --epochs 2 --patience 1 --rotations 2"
    RI_EXTRA="--ri-epochs 1 --warmup-epochs 1 --passes-per-epoch 1 --inner-folds 2 --tta 2"
    RI_LR_GRID="1e-3 3e-3"
    RI_EPOCHS=1; RI_STEPS_EPOCHS=1
    REPEATS_PRETRAINED="1"; REPEATS_RANDOMINIT="1"
    LC_PATIENTS="6"; LC_SEEDS="1"
    RI_WIDE_LR_GRID="1e-3 1e-2"
    RI_BRACKET_LR_GRID="1e-1"
    BOOT=200
else
    RESULTS=${PROJECT}/results
    TL_EXTRA=""
    TL_EXTRA_FINAL=""
    RI_EXTRA=""
    SCRATCH_EXTRA=""
    BOOT=${BURNML_BOOT:-2000}
fi
MANIFEST=${RESULTS}/manifest.csv

export PROJECT RAW SCRIPTS RESULTS MANIFEST PATIENT_MAP FOLDS MEM_BUDGET_GB
export BACKBONES PRIMARY RUN_SCRATCH_BASELINE SMOKE BOOT GPU_RESERVATION
export RI_LR_GRID RI_EPOCHS RI_STEPS_EPOCHS REPEATS_PRETRAINED REPEATS_RANDOMINIT
export LC_PATIENTS LC_SEEDS GRADCAM_REPEAT RI_EXTRA RI_WIDE_LR_GRID RI_BRACKET_LR_GRID RUN1_DIR
export TL_EXTRA TL_EXTRA_FINAL SCRATCH_EXTRA ENV_MODE CONDA_ENV TF_MODULE WEIGHTS

# ------------------------------------------------------- package check --- #
# scripts/ and slurm/ must come from the same pipeline package. A partial
# unzip once left new job scripts calling old Python, which failed with
# "unrecognized arguments". Every submission and every job checks this first.
check_package() {
    _a=$(cat "${PROJECT}/scripts/PACKAGE_VERSION" 2>/dev/null)
    _b=$(cat "${PROJECT}/slurm/PACKAGE_VERSION" 2>/dev/null)
    if [ -z "${_a}" ] || [ "${_a}" != "${_b}" ]; then
        echo "ERROR: scripts/ and slurm/ come from different pipeline packages."
        echo "       scripts/: ${_a:-no PACKAGE_VERSION (older package)}"
        echo "       slurm/:   ${_b:-no PACKAGE_VERSION (older package)}"
        echo "       Unzip the latest package from the project directory:"
        echo "           cd ${PROJECT} && git pull   (or unzip the latest release)"
        return 1
    fi
}

# ------------------------------------------------------------ activation --- #
activate_env() {
    check_package || return 1
    if command -v module >/dev/null 2>&1; then
        module purge >/dev/null 2>&1 || true
    fi
    if [ "${ENV_MODE}" = "module" ]; then
        module load "${TF_MODULE}"
    else
        # Use the environment's absolute path, recorded by env/setup_env.sh.
        # No `conda activate`: in batch jobs it can silently do nothing,
        # leaving the system Python in charge.
        ENV_PREFIX=$(cat "${PROJECT}/env/env_prefix.txt" 2>/dev/null)
        if [ -z "${ENV_PREFIX}" ] || [ ! -x "${ENV_PREFIX}/bin/python" ]; then
            echo "ERROR: environment not registered (env/env_prefix.txt)."
            echo "       On the login node run:  sh env/setup_env.sh --record ${CONDA_ENV}"
            return 1
        fi
        export PATH="${ENV_PREFIX}/bin:${PATH}"
        export CONDA_PREFIX="${ENV_PREFIX}" CONDA_DEFAULT_ENV="${CONDA_ENV}"
        export PYTHONNOUSERSITE=1          # never import packages from ~/.local
        hash -r 2>/dev/null || true
        if [ "$(command -v python)" != "${ENV_PREFIX}/bin/python" ]; then
            echo "ERROR: python resolves to $(command -v python), not ${ENV_PREFIX}/bin/python"
            return 1
        fi
        # Point the loader at the CUDA libraries that pip installed with
        # tensorflow[and-cuda]. Harmless when TensorFlow finds them itself;
        # fixes the "GPU not found" failure when it does not.
        SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
        NVLIBS=$(ls -d "${SITE}"/nvidia/*/lib 2>/dev/null | tr '\n' ':')
        export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH}"
        if [ -d "${SITE}/nvidia/cuda_nvcc" ]; then
            export PATH="${SITE}/nvidia/cuda_nvcc/bin:${PATH}"
            export XLA_FLAGS="--xla_gpu_cuda_data_dir=${SITE}/nvidia/cuda_nvcc"
        fi
    fi
    mkdir -p "${RESULTS}"
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
    export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
    export TF_CPP_MIN_LOG_LEVEL=2
    echo "environment: ${ENV_MODE} | python $(python -c 'import sys; print(sys.version.split()[0])') | results ${RESULTS}$( [ "${SMOKE}" = "1" ] && echo ' | SMOKE TEST')"
}
