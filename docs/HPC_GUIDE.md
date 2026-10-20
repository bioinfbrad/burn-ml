# Running the pipeline on a SLURM cluster

This guide was written for the GPU cluster on which the published analysis was run (NVIDIA Tesla P100, SLURM). Site-specific settings are supplied through environment variables, so nothing in the scripts needs editing:

```sh
export SBATCH_ACCOUNT=<your-account>          # read by sbatch for every job
export BURNML_RESERVATION=<your-reservation>  # optional; applied to GPU jobs only
```

Partition names are set in `slurm/config.sh` and the `#SBATCH -p` lines; change them if your cluster uses different names.


Pipeline for the *Burns & Trauma* submission. Two tasks (3-class burn-depth
classification and binary burn detection), a pre-specified comparison of four
ImageNet-pretrained backbones plus the original CNN as a baseline, all
evaluated on identical patient-grouped folds.

## Quick start on a SLURM cluster

Tested for your node: Tesla P100-PCIE-16GB, driver 535.104.12 (CUDA 12.2).
This selects **TensorFlow 2.15.1, Keras 2.15, CUDA 12.2, cuDNN 8.9, Python 3.10**.

The project can live anywhere; `slurm/config.sh` finds it automatically.

```sh
cd /path/to/burn-ml   # wherever you cloned the repository
#    raw_Images/{Full_thickness_burn,Partial_thickness_burn,Normal_skin}/ go here

# 1. environment, ONCE, on the login node, in a CLEAN shell
module purge && module load Miniconda3
sh env/setup_env.sh burnml 535.104.12
#    safe to rerun: an existing environment is reused and completed.
#    already fully installed? just register it:  sh env/setup_env.sh --record burnml

# 2. ImageNet weights, ONCE, on the login node. Standard library only, so any
#    python works, and the login node never has to import TensorFlow (no AVX).
python scripts/00b_fetch_weights.py

# 3. confirm TensorFlow and the GPU work on a GPU node (a few minutes)
srun -p Nvidia --account=<your-account> --gres=gpu:1 --cpus-per-task=4 \
     --time=00:20:00 sh -c '. slurm/config.sh && activate_env && \
     python scripts/00_check_env.py --out-dir /tmp/burnml_check \
     --check-weights efficientnetb0 densenet121 resnet50v2 mobilenetv2'

# 4. smoke test: the ENTIRE chain with tiny settings, minutes not days
cd slurm
SMOKE=1 sh submit_all.sh

# 5. the real run
sh submit_all.sh
```

**Never import TensorFlow on the login node.** Official TensorFlow wheels
need the CPU's AVX instructions; the login node lacks them and crashes with
"Illegal instruction". Every step that runs there (setup, weight download)
avoids TensorFlow, and every job that uses it runs on a GPU node.

**No `conda activate` anywhere.** Inside scripts and batch jobs it can
silently do nothing and leave the system Python in charge (this happened on
the first install attempt). Setup records the environment's absolute path in
`env/env_prefix.txt`; every job puts that first on `PATH` and verifies that
`python` really is the environment's before running anything.

**GPU reservation.** `GPU_RESERVATION` in `slurm/config.sh` (default
`<your-reservation>`) is applied by `submit_all.sh` to the GPU jobs only. Never
add `--reservation` to a job script: the reservation covers the GPU node, so
a compute-partition job carrying it is refused with "Requested node
configuration is not available". When the reservation ends, set
`GPU_RESERVATION=` (empty).

**Always submit from `slurm/`.** SLURM runs a spool copy of each job script,
so jobs locate `config.sh` through `$SLURM_SUBMIT_DIR`; `submit_all.sh`
handles this for you.

**Weights** live in `~/.keras/models`, the only place Keras 2.15 reads them
from; your home directory is mounted on the compute nodes.

## Extension: the pre-specified additions

After the official run, a second wave adds what reviewers of small-sample
medical AI now expect, specified in `ANALYSIS_PLAN.md` before they were run:

```sh
cd slurm
SMOKE=1 sh submit_extension.sh    # minutes; proves every new job on your node
sh submit_extension.sh            # ~2 days of GPU time on one P100
```

Your existing official results are reused untouched. New output:

| Path | Contents |
|---|---|
| `results/<task>/efficientnetb0_randominit/` | the confirmatory random-init arm |
| `results/<task>/efficientnetb0_randominit_steps/` | the step-matched secondary arm |
| `results/<task>/comparison/backbone_comparison.md` | confirmatory contrast (BCa CI), decomposition scratch → random-init → pretrained, Holm-adjusted backbone contrasts |
| `results/<task>/repeats/r<k>/` and `repeats_summary.md` | repeated CV: mean, SD, range; paired Δ per repeat |
| `results/<task>/learning_curve_efficientnetb0.{md,png}` | performance vs training patients |
| `results/<task>/<run>/report/sensitivity.md`, `posthoc.md` | patient-weighted metrics, forest plot, selective prediction, aggregation, clustered AUC |
| `results/<task>/<run>/report/fig_gradcam_panel.png`, `gradcam.json` | saliency with the randomisation sanity check |
| `results/<task>/efficientnetb0/final/` | 3-member archival ensemble |

Measured on your P100 from the official run: one pretrained fold takes about
3 minutes; the random-init arm (150 epochs x 3 learning rates) about an hour
per fold. Total extension ≈ 46 GPU-hours.

## Post hoc learning-rate sensitivity

The random-init arm chose the top of its grid in 26 of 48 folds, so it is
rerun with the grid extended to 3e-2 (ANALYSIS_PLAN.md, section 8):

```sh
cd slurm
SMOKE=1 sh submit_sensitivity.sh   # minutes
sh submit_sensitivity.sh           # ~20 GPU-hours, then refreshes run 1
sh submit_sensitivity.sh bracket   # ~8 GPU-hours: adds 1e-1 and 3e-1 on top
```

The bracketing step trains only the new rates (`--extend-from`), chooses over
the union of all seven by the same rule, and keeps the earlier predictions
unchanged wherever the earlier choice still wins.

When it finishes, `run1/results/<task>/comparison/backbone_comparison.md`
holds the confirmatory result and the sensitivity result side by side.

## Run 1 is the primary of record

The complete pipeline was run twice with identical folds and seeds. Run 1
(`run1/results/`) is the primary of record; run 2 (`results/`) is reported as
a reproducibility replicate. `s12_run1_reports.sh` copies every random-init
arm into run 1 and regenerates its comparison, without modifying run 1's own
predictions.

## Why the pipeline was rebuilt

The original run split **crops**, not patients. The 284 crops come from 38
accession IDs clustering into **18 patient episodes**, so every held-out crop
had siblings from the same photograph in training: 41/41 classification and
39/39 detection test crops. That explains AUC ≈ 1.00. The old `k = 4` loop did
not fix it: it sliced the augmented, shuffled array contiguously, and ran only
during the hyperparameter search.

## Design

| Component | Choice | Reason |
|---|---|---|
| Split | 6-fold, whole patients, class-balanced | no patient on both sides; every fold contains every class |
| Primary model | EfficientNetB0, ImageNet | pre-specified; best accuracy per parameter at small n |
| Comparison | DenseNet121, ResNet50V2, MobileNetV2, scratch CNN | first patient-level backbone comparison for burns; published ones split by image |
| Training | linear probe 15 epochs (lr 1e-3), then top ~20 layers 30 epochs (lr 1e-5, cosine decay), 4 augmented passes per epoch | LP-FT preserves pretrained features; ~1,800 optimiser steps per member |
| BatchNorm | always in inference mode | the silent failure of Keras fine-tuning; verified by the self-test |
| Augmentation | online dihedral + random resized crop, gentle colour jitter (hue ±0.02) | colour is the diagnostic signal for burn depth |
| Ensemble | the 3 inner-fold models averaged per outer fold | removes the single-model lottery; citable (nnU-Net) |
| Inference | 8-transform TTA, temperature scaling on inner out-of-fold predictions | calibration without touching held-out patients |
| Not used | class weights, early stopping, hyperparameter search | distort calibration / noise-driven with 2–4 validation patients / overfits 18 patients |

Everything above is **pre-specified**. Do not tune it on these patients.

## Output, per task

```
results/<task>/<model>/metrics.json            all metrics, CIs, design record
results/<task>/<model>/oof_predictions.csv     per-image predictions
results/<task>/<model>/report/
    numbers_for_text.txt                       Methods + Results numbers
    table1_dataset.*  table2_performance.*  table3_ablation.*
    fig_confusion_matrix  fig_roc  fig_calibration  fig_training_curves
    fig_fold_accuracy                          (all 300 dpi)
results/<task>/comparison/backbone_comparison.{md,csv,json}
results/<task>/efficientnetb0/final/           archival model for Zenodo
```

`table3_ablation` is the supplementary ladder: single model → ensemble →
+TTA → +temperature scaling, with accuracy, AUC, Brier and ECE at each step.

## Scripts

| Python | Purpose |
|---|---|
| `burnml.py`, `backbones.py` | shared library; the pre-specified backbone set |
| `00_check_env.py` | proves kernels run on the GPU; checks weights load offline |
| `00b_fetch_weights.py` | downloads ImageNet weights on the login node, no TensorFlow |
| `01_prepare_manifest.py` | assigns each crop to a patient group |
| `02_search_hyperparams.py` | nested search, scratch baseline only |
| `03_run_grouped_cv.py` | the experiment, one fold per process |
| `03b_aggregate.py` | pools folds; refuses incomplete or mismatched sets |
| `04_train_final.py` | archival model on all patients |
| `05_make_report.py`, `06_compare_backbones.py` | tables, figures, paired comparison |
| `07_sensitivity.py` | patient-weighted metrics, dominant-patient exclusion, calibration CIs, forest plot |
| `08_posthoc.py` | selective prediction, photograph/patient aggregation, clustered AUC |
| `09_repeats.py`, `11_learning_curves.py` | repeated-CV and learning-curve summaries |
| `10_gradcam.py` | Grad-CAM with model-randomisation sanity check |
| `12_results_ledger.py` | every final number, read from the result files: `python scripts/12_results_ledger.py --run1 run1/results --results results --out RESULTS_LEDGER.md` |
| `99_selftest.py` | 60 correctness checks on the real data |

| SLURM (run in this order by `submit_all.sh`) | Partition |
|---|---|
| `s00_env` GPU + offline weights check (halts the chain on failure) | Nvidia |
| `s01_manifest` | compute |
| `s02_selftest` (gates everything below; needs AVX, so a GPU node) | Nvidia |
| `s03_cv_array` one fold per array task, `--array=1-6%3` | Nvidia |
| `s04_aggregate`, `s05_report`, `s06_compare` | compute |
| `s07_final` archival primary model | Nvidia |

Resuming after any failure: rerun `sh submit_all.sh`; finished folds are
skipped. Fold files are written atomically, so a killed job leaves nothing
half-written.

## Validation record

Before delivery the pipeline was run end to end in an exact replica of your
environment (Python 3.10, TensorFlow 2.15.1, Keras 2.15, NumPy 1.26.4, OpenCV
4.11, scikit-learn 1.7.2, scikit-optimize 0.10.2; CPU build, identical code
paths), with every SLURM script executed from a spool copy with SLURM's
environment variables, exactly as the scheduler runs them:

* `submit_all.sh` builds a correct 37-job graph in real and smoke modes;
* every job type ran to completion on the real 284 images: environment,
  manifest, self-test (60/60 classification, 30/30 detection), all 6 transfer
  folds, all 6 scratch folds with nested search, aggregation, reports,
  comparison and the archival model;
* the dependency lock files resolve from binary wheels for both CUDA paths.

Bugs found and fixed during validation, any of which would have broken the
HPC run: job scripts locating `config.sh` via `$0` (fails under SLURM's spool
copy); `tensorflow[and-cuda]==2.14.1` unsatisfiable on Python 3.11; `module
purge` unloading Conda; OpenCV pulling NumPy 2 when installed separately;
`tf.keras.__version__` missing in TF 2.15; Keras 2 rejecting a `class_weight`
dict without every class; too few optimiser steps per epoch; the archival
script lacking the transfer model; a calibration figure lost in an edit.
After first contact with Bradford HPC: a hard-coded project path; the
weight download and install check importing TensorFlow on a login node
without AVX; Keras 2.15 ignoring KERAS_HOME for weights; and `conda
activate` silently failing inside scripts, so pip ran under the system
Python 3.9; and a GPU reservation added to a compute-partition job, which
the scheduler refuses; and the self-test's own process reserving the whole
GPU so that its child processes ran out of memory (it now stays on CPU). `submit_all.sh` now also stops at the first refused
submission and cancels what it had queued, instead of cascading.

Not testable outside your cluster: real GPU execution and the ImageNet
download. `s00_env` checks both before any GPU hours are spent, and the smoke
run proves the whole chain on your hardware.

## Before you deposit anything

Filenames contain hospital identifiers (e.g. `<hospital prefix>_<accession number> a.jpg`).
Rename every file before any deposit, including the restricted Zenodo record,
and keep the mapping offline.
