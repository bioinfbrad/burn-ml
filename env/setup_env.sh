#!/bin/sh
# setup_env.sh — build (or repair) the Conda environment for burn-ml.
#
# Run on a LOGIN node, from the project directory, in a clean shell:
#
#     module purge && module load Miniconda3
#     sh env/setup_env.sh burnml 535.104.12      # install or repair
#     sh env/setup_env.sh --record burnml        # just register an existing env
#
# Robust by design: it never relies on `conda activate` inside this script,
# which can silently do nothing depending on the parent shell's state. It
# finds the environment's absolute path and calls ITS python directly, so pip
# can only ever install into this environment.
#
# The driver decides the CUDA stack (P100 = sm_60; never use a CUDA 13 build):
#   driver >= 525.60.13 -> TensorFlow 2.15.1, CUDA 12.2, cuDNN 8.9
#   driver >= 450.80.02 -> TensorFlow 2.14.1, CUDA 11.8, cuDNN 8.7
#   older               -> use the cluster TF 2.6 module (ENV_MODE=module)
# Python 3.10 because tensorflow[and-cuda]==2.14.1 has no 3.11 build.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)

MODE=install
if [ "${1:-}" = "--record" ]; then MODE=record; shift; fi
ENV_NAME=${1:-burnml}
DRIVER_ARG=${2:-}

die() { echo "ERROR: $*"; exit 1; }

command -v conda >/dev/null 2>&1 || die "conda is not on PATH. Run:
       module purge && module load Miniconda3"
CONDA_BASE=$(conda info --base)
echo "conda   : $(command -v conda)  ($(conda --version 2>&1))"
echo "base    : ${CONDA_BASE}"

find_prefix() {
    for d in "${HOME}/.conda/envs/${ENV_NAME}" "${CONDA_BASE}/envs/${ENV_NAME}"; do
        if [ -x "${d}/bin/python" ]; then echo "${d}"; return 0; fi
    done
    conda env list 2>/dev/null | awk -v n="${ENV_NAME}" '$1==n {print $NF; exit}'
}

PREFIX=$(find_prefix || true)

if [ "${MODE}" = "install" ]; then
    DRIVER=${DRIVER_ARG:-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)}
    [ -n "${DRIVER}" ] || die "no driver version. Pass it: sh env/setup_env.sh ${ENV_NAME} 535.104.12"
    MAJOR=$(echo "${DRIVER}" | cut -d. -f1)
    if [ "${MAJOR}" -ge 525 ]; then
        LOCK=requirements-tf215-cuda12.txt; STACK="TensorFlow 2.15.1, CUDA 12.2, cuDNN 8.9"
    elif [ "${MAJOR}" -ge 450 ]; then
        LOCK=requirements-tf214-cuda11.txt; STACK="TensorFlow 2.14.1, CUDA 11.8, cuDNN 8.7"
    else
        die "driver ${DRIVER} is too old for CUDA 11.8; set ENV_MODE=module in slurm/config.sh"
    fi
    echo "driver  : ${DRIVER}  ->  ${STACK}"

    if [ -z "${PREFIX}" ]; then
        echo "creating environment '${ENV_NAME}' (Python 3.10)"
        if ! conda create -y -n "${ENV_NAME}" python=3.10; then
            echo "retrying with conda-forge only"
            conda create -y -n "${ENV_NAME}" -c conda-forge --override-channels python=3.10
        fi
        PREFIX=$(find_prefix || true)
    else
        echo "reusing existing environment at ${PREFIX}"
    fi
fi

[ -n "${PREFIX}" ] && [ -x "${PREFIX}/bin/python" ] || die "cannot locate environment '${ENV_NAME}'"
PY="${PREFIX}/bin/python"
PYV=$("${PY}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "python  : ${PY}  (Python ${PYV})"
[ "${PYV}" = "3.10" ] || die "the environment has Python ${PYV}, expected 3.10. Remove it and rerun:
       rm -rf ${PREFIX}"

# never install into, or import from, the user site (~/.local)
export PYTHONNOUSERSITE=1 PIP_USER=0

if [ "${MODE}" = "install" ]; then
    "${PY}" -m pip install --upgrade pip
    "${PY}" -m pip install --no-cache-dir -r "${HERE}/${LOCK}"
    echo "${LOCK}" > "${HERE}/installed_lock.txt"
fi

# Verify WITHOUT importing TensorFlow: official wheels need AVX, which the
# login node lacks ("Illegal instruction"). Every package below is AVX-free.
"${PY}" - <<'PYCHK'
import importlib.metadata as m, sys
import numpy, cv2, skopt, sklearn
def v(*names):
    for n in names:
        try:
            return m.version(n)
        except m.PackageNotFoundError:
            pass
    sys.exit("ERROR: TensorFlow is not installed in this environment")
print("verified: TensorFlow", v("tensorflow", "tensorflow-cpu"), "| Keras", v("keras"),
      "| NumPy", numpy.__version__, "| OpenCV", cv2.__version__,
      "| scikit-optimize", skopt.__version__)
PYCHK

echo "${PREFIX}" > "${HERE}/env_prefix.txt"
echo "${CONDA_BASE}" > "${HERE}/conda_base.txt"
echo "${ENV_NAME}" > "${HERE}/conda_env.txt"
echo "recorded: ${PREFIX}  (env/env_prefix.txt; jobs use this path directly)"

if ! grep -qw avx /proc/cpuinfo 2>/dev/null; then
    echo
    echo "NOTE: this node has no AVX, so TensorFlow cannot be imported HERE."
    echo "      Expected on the login node; the GPU nodes are unaffected."
fi
echo
echo "Next (login node, no TensorFlow needed):"
echo "    ${PY} scripts/00b_fetch_weights.py"
