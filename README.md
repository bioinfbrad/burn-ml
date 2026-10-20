# burn-ml: patient-level evaluation of convolutional neural networks for burn images

Code for the study *Machine learning as a tool to differentiate between images of burnt and normal skin* (manuscript under review at *Burns & Trauma*). It trains and evaluates convolutional neural networks (CNNs) for two tasks on clinical photographs of burns:

- **detection**: burnt skin (partial- or full-thickness) versus normal skin;
- **classification**: partial-thickness burn, full-thickness burn or normal skin.

The central design rule is that **no patient contributes to both training and testing**. Performance is estimated by patient-grouped cross-validation, and every confidence interval comes from a patient-level bootstrap.

> **No images or patient data are in this repository.** The dataset consists of clinical photographs and is held in a restricted-access repository; see [`data/README.md`](data/README.md).

## Why this version exists

An earlier analysis of the same images (repository version 1.0.0) divided image crops between training and test sets without tracing them to their source photograph or patient. Every test crop shared its patient with crops in the training set, so performance was overestimated: full-thickness burns appeared to be detected in 94% of test crops. This version rebuilds the pipeline around patient-grouped evaluation, with automated checks that no patient appears on both sides of a split. The lesson for anyone collecting burn images: record an anonymised patient identifier for every photograph and carry it with every crop derived from it.

## Main results (run of record)

| | Accuracy (95% CI) | Macro AUC (95% CI) |
|---|---|---|
| Detection, pretrained EfficientNetB0 | 0.944 (0.886–0.985) | 0.979 (0.956–0.998) |
| Classification, pretrained EfficientNetB0 | 0.856 (0.738–0.938) | 0.954 (0.902–0.984) |

Full-thickness burns were the weak point: sensitivity was 0.640 at crop level but **0.431 with each patient counted equally**, because one patient contributed 40% of the full-thickness crops. Every number in the paper, with the file it was read from, is in [`results/RESULTS_LEDGER.md`](results/RESULTS_LEDGER.md).

## What is in the repository

```
ANALYSIS_PLAN.md       analysis plan, written before the secondary analyses were run
data/README.md         how to obtain the data, and the folder layout the pipeline expects
docs/HPC_GUIDE.md      step-by-step guide for running on a SLURM GPU cluster
env/                   environment setup script and pinned package versions
scripts/               the pipeline (see below)
slurm/                 SLURM job scripts and submission scripts
results/               anonymised results ledger and the paper's figures
```

| Script | Purpose |
|---|---|
| `00_check_env.py`, `00b_fetch_weights.py` | check the GPU environment; download ImageNet weights |
| `01_prepare_manifest.py` | list crops and assign each to a patient group |
| `02_search_hyperparams.py` | nested Bayesian search for the custom CNN baseline |
| `03_run_grouped_cv.py`, `03b_aggregate.py` | patient-grouped cross-validation (pretrained, randomly initialised and custom models) |
| `04_train_final.py` | archival ensemble trained on all patients |
| `05_make_report.py` … `11_learning_curves.py` | reports, network comparison, patient-level and post hoc analyses, repeated cross-validation, Grad-CAM, learning curves |
| `12_results_ledger.py` | every reported number, read from the result files |
| `13_paper_figures.py` | the manuscript's figures (anonymised) |
| `99_selftest.py` | automated correctness tests, run before every analysis |

## Running it

Requirements: Linux, an NVIDIA GPU with at least 16 GB of memory, conda, Python 3.10 and TensorFlow 2.15. The exact package versions are pinned in `env/requirements-tf215-cuda12.txt`.

```sh
git clone https://github.com/bioinfbrad/burn-ml && cd burn-ml
sh env/setup_env.sh                      # creates the conda environment
python scripts/99_selftest.py            # correctness tests
```

The complete study (all models, both tasks, six-fold patient-grouped cross-validation with three-member ensembles) takes several GPU-days and is run as SLURM job arrays:

```sh
export SBATCH_ACCOUNT=<your-account>
cd slurm && sh submit_all.sh             # primary analysis
sh submit_extension.sh                   # pre-specified secondary analyses
```

[`docs/HPC_GUIDE.md`](docs/HPC_GUIDE.md) describes every step, its outputs and how to check it. When the runs have finished, the paper's numbers and figures are regenerated with:

```sh
python scripts/12_results_ledger.py --run1 run1/results --results results --out RESULTS_LEDGER.md
python scripts/13_paper_figures.py  --run1 run1/results --results results --out figures
```

## Protecting patient data

The pipeline's raw outputs (manifests, predictions, reports and their figures) contain file names and patient group names derived from hospital accession numbers. **Never commit or publish them.** The `.gitignore` excludes data, results, images and model weights; publish only the outputs of `12_results_ledger.py` and `13_paper_figures.py`, which number patients 1..N instead.

## Citation

If you use this code, please cite the paper (reference to follow on publication) and the software archive (see [`CITATION.cff`](CITATION.cff)).

## Licence

GNU Affero General Public License v3.0; see [`LICENSE`](LICENSE).

## Contact

Krzysztof Poterlowicz, University of Bradford: kpoterl1@bradford.ac.uk
