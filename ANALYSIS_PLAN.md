# Pre-specified secondary analysis plan

**Status:** written before any of the analyses below were run, and run
unchanged. **Not lodged in a public registry**, so the manuscript must
describe these analyses as *pre-specified in a written analysis plan*, not as
pre-registered, and should include this document as supplementary material.

Study: *Machine learning to differentiate burnt from normal skin in clinical
photographs* — single NHS burns unit, 284 cropped images from 18 patient
episodes, patient-grouped internal validation.

## Context and honesty statement

The primary analysis (ImageNet-pretrained EfficientNetB0, 6-fold
patient-grouped cross-validation, recipe pre-specified before training) was
completed before this plan was written, as were descriptive comparisons with
DenseNet121, ResNet50V2, MobileNetV2 and the original custom CNN. Those
results are known to the authors. This plan specifies additional analyses
before any of them were run. Analyses not listed here are exploratory.

**Replicate run (disclosed).** Before this plan was written, the complete primary
pipeline was run a second time with identical folds, seeds and recipe, as a
pipeline check. The first run is the primary of record; the second is
reported as a reproducibility replicate. Pretrained models were reproducible
(EfficientNetB0: 99.3% identical crop labels, accuracy 0.856 in both runs);
the legacy CNN was not (63% identical labels; accuracy 0.570 vs 0.739),
because its nested hyperparameter search selected different architectures.
For this reason the legacy CNN is treated as descriptive only, and the
random-initialisation arm below is the sole confirmatory comparator.

## 1. Confirmatory comparison (the only one)

**Question.** Under an identical training and evaluation protocol, does
ImageNet initialisation change performance relative to random
initialisation of the same architecture?

- Arms: pretrained EfficientNetB0 (existing primary) vs EfficientNetB0 from
  random initialisation, on the **same** original patient folds.
- Effect measures: Δ macro one-vs-rest AUC (primary) and Δ balanced accuracy
  (secondary), 3-class task. Binary task reported as secondary.
- Inference: paired patient-level cluster bootstrap, 2,000 resamples, BCa
  95% CI (percentile CI also reported).
- No direction is pre-committed; the result is reported whatever it shows.

## 2. Random-initialisation recipe

Identical to the pretrained arm except where random initialisation makes the
pretrained setting meaningless:

| Setting | Pretrained arm | Random-init arm |
|---|---|---|
| Weights | ImageNet | random (Keras defaults) |
| Input standardisation | ImageNet mean/std | ImageNet mean/std (set explicitly) |
| Trainable layers | linear probe, then last ~20 | all, from step 0 |
| BatchNorm | inference mode | training mode |
| Optimiser | Adam, clipnorm 1.0 | AdamW, weight decay 1e-4, clipnorm 1.0 |
| Learning rate | 1e-3 probe / 1e-5 fine-tune | chosen per outer fold from {3e-4, 1e-3, 3e-3} by lowest inner out-of-fold cross-entropy |
| Schedule | 15 + 30 epochs, cosine | 5-epoch warm-up, cosine to 0, 150 epochs (~3.3x steps) |
| Everything else | — | identical: 4 augmented passes/epoch, batch 16, head dropout 0.3, label smoothing 0.05, augmentation, 3-member inner-fold ensemble, 8-transform TTA, temperature scaling on inner out-of-fold predictions, no early stopping, no class weights |

Secondary arm (pre-specified): same, matched on optimiser steps (45 epochs).

## 3. Repeated cross-validation

Repeat 0 is the original split and remains the primary estimate. Repeats
1–4 (pretrained) and 1–2 (random-init) use new class-balanced,
patient-grouped fold assignments (seed 1000 + repeat) and new training seeds.
Reported: mean, SD and range of each metric across repeats; the paired
pretrained − random-init difference within each repeat.

## 4. Pre-specified post hoc analyses (no retraining)

- Patient-weighted per-class sensitivity; per-patient table; forest plot;
  exclusion of the patient contributing most crops of each class.
- Calibration slope and intercept with patient-bootstrap CIs.
- Selective prediction: confidence = maximum calibrated probability;
  coverage 100/90/80/70%; accuracy, macro F1, per-class sensitivity
  (crop-level and patient-weighted); AURC.
- Aggregation: photograph × reference class (mean calibrated probability)
  and patient level ("any crop of the class detected").
- Clustered AUC (Obuchowski 1997) as an analytical check.
- Descriptive backbone contrasts reported with Holm-adjusted bootstrap p.

## 5. Learning curves (exploratory, descriptive)

Pretrained EfficientNetB0 trained on 6, 9 and 12 of each outer fold's
training patients, 3 random subsamples each (every class required).
No extrapolation. The fraction of subsamples containing the patient who
contributes 30 of the 75 full-thickness crops is reported.

## 6. Saliency (exploratory)

Grad-CAM on archived held-out models, averaged over the fold's ensemble.
Panel fixed by seed 42: 2 correct and 2 incorrect crops per class plus every
full-thickness error. Model-randomisation sanity check (cascade from the
head down), reported as Spearman correlation and SSIM. Presented as regions
influencing the output, not as evidence of clinical reasoning.

## 7. Not done, and why

Leave-one-patient-out CV (superseded by repeated grouped CV), colour
constancy retraining, decision-curve claims of clinical utility (crop
prevalence is not clinical prevalence), and web-scraped external datasets.
An external test on the Seville BIP_US set will be added only if its licence
permits, as an exploratory analysis.


## 8. Post hoc sensitivity analysis (added after the results above were known)

**Not part of the plan.** Reported separately and labelled post hoc.

Reason: the random-initialisation arm chose the top of its learning-rate grid
(3e-3) in 26 of 48 fold-level selections, and its inner cross-entropy was
still falling at that value in 21, so the arm may have been under-tuned, which
would overstate the benefit of pretraining.

Analysis: the random-initialisation arm is rerun at the original split, both
tasks, with the grid extended to {3e-4, 1e-3, 3e-3, 1e-2, 3e-2} and everything
else identical. It is compared with the pretrained primary of record (run 1)
using the same paired patient bootstrap. Whatever the result, the
pre-specified confirmatory result in section 1 remains the primary finding;
this analysis tests only its robustness to the learning-rate grid.

**Outcome and bracketing step (also post hoc).** The extended grid improved
the random-initialisation arm (detection Δ macro AUC +0.041, 95% CI +0.007 to
+0.092), but its new top value (3e-2) was again chosen in 10 of 12 folds. To
bracket the optimum, two higher rates (1e-1, 3e-1) are trained on top of the
extended grid. Only the new rates are trained; the choice is made over the
union of all seven rates by the same rule (lowest inner out-of-fold
cross-entropy), and the earlier predictions are kept unchanged wherever the
earlier choice still wins. This is the final step: if the top rate still wins,
the paper states that the random-initialisation optimum was not reached.

