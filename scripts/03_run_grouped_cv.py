#!/usr/bin/env python3
"""
03_run_grouped_cv.py — the main experiment.

Patient-grouped cross-validation. Every patient is held out exactly once, and
the reported metrics come from pooled out-of-fold predictions with 95%
confidence intervals from a bootstrap that resamples patients.

Two architectures
-----------------
  --arch transfer --backbone efficientnetb0   ImageNet-pretrained backbone,
        linear probe then top-block fine-tuning (LP-FT), a 3-member
        cross-validation ensemble per outer fold, 8-transform test-time
        augmentation, and temperature scaling fitted on inner out-of-fold
        predictions.

  --arch scratch                              the original custom CNN, kept as
        the baseline so the paper can quantify what transfer learning buys.

One fold per process
--------------------
Each fold writes folds/fold_<k>.json and nothing depends on the others until
aggregation, so a fold can be its own SLURM array task: memory is reclaimed
between folds, a crash costs one fold, and folds run in parallel.

    --only-fold K   train fold K, write its file, exit
    (no flag)       run every fold here, skipping any already written

Usage:
    python 03_run_grouped_cv.py --raw-dir raw_Images \
        --manifest results/manifest.csv --task classification \
        --arch transfer --backbone efficientnetb0 \
        --out-dir results/classification/efficientnetb0 --only-fold 1
"""
import argparse
import csv
import importlib.util
import json
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import burnml
import backbones as B

_spec = importlib.util.spec_from_file_location(
    "search_mod", os.path.join(HERE, "02_search_hyperparams.py"))
search_mod = importlib.util.module_from_spec(_spec)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", choices=list(burnml.TASKS), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arch", choices=["transfer", "randominit", "scratch"],
                    default="transfer")
    ap.add_argument("--backbone", default=B.PRIMARY)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--inner-folds", type=int, default=3,
                    help="ensemble members per outer fold")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true")
    # transfer recipe (pre-specified; do not tune on these 18 patients)
    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=30)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--no-photometric", action="store_true")
    ap.add_argument("--passes-per-epoch", type=int, default=4,
                    help="augmented passes over the training crops per epoch; "
                         "with ~160 crops and batch 16 one pass is only ~10 "
                         "optimiser steps")
    ap.add_argument("--no-cosine", action="store_true",
                    help="constant fine-tuning learning rate")
    ap.add_argument("--weights", choices=["imagenet", "none"],
                    default="imagenet",
                    help="'none' is for smoke tests only; never report a "
                         "result produced with random initialisation")
    ap.add_argument("--tta", type=int, default=8,
                    help="dihedral transforms at inference; 1 disables TTA")
    ap.add_argument("--no-temperature", action="store_true")
    # scratch-model options
    ap.add_argument("--params", default=None)
    ap.add_argument("--nested-search", action="store_true")
    ap.add_argument("--rotations", type=int, default=24)
    ap.add_argument("--rotation-mode", choices=["bound", "inscribed"],
                    default="bound")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--inner-val-frac", type=float, default=0.2)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--loader", choices=["auto", "memory", "stream"],
                    default="auto")
    ap.add_argument("--memory-budget-gb", type=float, default=60.0)
    ap.add_argument("--search-calls", type=int, default=30)
    ap.add_argument("--search-folds", type=int, default=3)
    ap.add_argument("--search-epochs", type=int, default=40)
    ap.add_argument("--search-max-batch", type=int, default=128,
                    help="batch-size ceiling for the scratch search; 128 fits "
                         "a 16 GB P100")
    # random-initialisation arm (same architecture, no pretrained weights)
    ap.add_argument("--extend-from", default=None,
                    help="random-init: an earlier run of the same arm on the "
                         "same folds. Only the NEW learning rates are trained; "
                         "the choice is made over the union of both grids by "
                         "the same rule, and the earlier choice is kept "
                         "unchanged unless a new rate has lower inner CE")
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3],
                    help="random-init learning rates; one is chosen per outer "
                         "fold by inner out-of-fold cross-entropy")
    ap.add_argument("--ri-epochs", type=int, default=150)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # repeated CV, model archiving, learning curves
    ap.add_argument("--repeat", type=int, default=0,
                    help="0 = the original, pre-specified split; k>0 = a new "
                         "class-balanced patient-grouped split seeded by k")
    ap.add_argument("--save-models", action="store_true",
                    help="archive every ensemble member (for saliency and "
                         "external scoring)")
    ap.add_argument("--train-patients", type=int, default=None,
                    help="learning curve: train on this many of each outer "
                         "fold's training patients")
    ap.add_argument("--subsample-seed", type=int, default=0)
    args = ap.parse_args()
    if args.arch == "scratch" and not args.params and not args.nested_search:
        sys.exit("--arch scratch needs --params FILE or --nested-search.")
    if args.arch in ("transfer", "randominit") and not B.available(args.backbone):
        sys.exit(f"Backbone '{args.backbone}' is not in this TensorFlow "
                 f"({', '.join(B.list_available())}).")
    return args


def settings_record(args, size):
    base = {
        "task": args.task, "architecture": args.arch, "image_size": size,
        "seed": args.seed, "outer_folds": args.folds,
    }
    if args.arch == "transfer":
        base.update({
            "backbone": args.backbone,
            "backbone_keras_name": B.spec(args.backbone)["keras_name"],
            "weights": args.weights,
            "strategy": "linear probe then top-block fine-tuning (LP-FT), "
                        "BatchNorm kept in inference mode throughout",
            "probe_epochs": args.probe_epochs,
            "finetune_epochs": args.finetune_epochs,
            "passes_per_epoch": args.passes_per_epoch,
            "finetune_schedule": ("cosine decay to zero" if not args.no_cosine
                                  else "constant"),
            "optimiser": "Adam, gradient clipping at norm 1.0",
            "probe_lr": args.probe_lr, "finetune_lr": args.finetune_lr,
            "unfreeze_layers": B.spec(args.backbone)["unfreeze_layers"],
            "dropout": args.dropout, "batch_size": args.batch_size,
            "label_smoothing": args.label_smoothing,
            "augmentation": ("online: dihedral, random resized crop"
                             + ("; gentle colour jitter (brightness 0.10, "
                                "contrast/saturation 0.9-1.1, hue 0.02)"
                                if not args.no_photometric else "")),
            "ensemble_members": args.inner_folds,
            "ensemble_type": "cross-validation ensemble over inner "
                             "patient-grouped folds",
            "tta_transforms": args.tta,
            "temperature_scaling": not args.no_temperature,
            "class_weighted": False,
            "early_stopping": "none: fixed epoch budget",
        })
    else:
        base.update({
            "rotations_per_training_image": args.rotations,
            "rotation_mode": args.rotation_mode,
            "class_weighted": not args.no_class_weight,
            "epochs_max": args.epochs,
            "early_stopping_patience": args.patience,
            "inner_validation_fraction": args.inner_val_frac,
            "hyperparameter_selection": (
                f"nested: {args.search_calls} Bayesian calls x "
                f"{args.search_folds} patient-grouped inner folds"
                if args.nested_search else
                f"fixed set from {os.path.basename(args.params or '')}"),
        })
    if args.arch == "randominit":
        base.update({
            "backbone": args.backbone,
            "backbone_keras_name": B.spec(args.backbone)["keras_name"],
            "weights": "random initialisation",
            "strategy": "all layers trained from random initialisation; "
                        "BatchNorm in training mode; input standardised with "
                        "ImageNet mean/std exactly as in the pretrained arm",
            "optimiser": "AdamW, weight decay %g, gradient clipping at norm 1.0"
                         % args.weight_decay,
            "lr_grid": prior_grid(args) + sorted(set(args.lr_grid) - set(prior_grid(args))),
            "lr_selection": "per outer fold, lowest inner out-of-fold "
                            "cross-entropy",
            "extended_from": (os.path.abspath(args.extend_from)
                              if args.extend_from else None),
            "epochs": args.ri_epochs, "warmup_epochs": args.warmup_epochs,
            "schedule": "linear warm-up then cosine decay to zero",
            "passes_per_epoch": args.passes_per_epoch,
            "dropout": args.dropout, "batch_size": args.batch_size,
            "label_smoothing": args.label_smoothing,
            "augmentation": ("online: dihedral, random resized crop"
                             + ("; gentle colour jitter (brightness 0.10, "
                                "contrast/saturation 0.9-1.1, hue 0.02)"
                                if not args.no_photometric else "")),
            "ensemble_members": args.inner_folds,
            "ensemble_type": "cross-validation ensemble over inner "
                             "patient-grouped folds",
            "tta_transforms": args.tta,
            "temperature_scaling": not args.no_temperature,
            "class_weighted": False,
            "early_stopping": "none: fixed epoch budget",
        })
    base.update({"repeat": args.repeat,
                 "fold_assignment": ("original pre-specified split"
                                     if args.repeat == 0 else
                                     f"seeded class-balanced split, seed {1000 + args.repeat}"),
                 "train_patients": args.train_patients,
                 "subsample_seed": (args.subsample_seed
                                    if args.train_patients else None),
                 "models_archived": bool(args.save_models)})
    base["environment"] = burnml.env_record()
    return base


def prior_grid(args):
    """Learning-rate grid of the run being extended (empty if none)."""
    if not args.extend_from:
        return []
    p = os.path.join(args.extend_from, "folds", "fold_1.json")
    return [float(x) for x in json.load(open(p))["settings"]["lr_grid"]]


def restrict_training(tr, groups, y, n_classes, k, seed):
    """Learning curve: keep k of the outer fold's training patients, drawn at
    random but requiring every class to remain represented."""
    pats = sorted(set(groups[tr].tolist()))
    if k is None or k >= len(pats):
        return tr, pats
    rng = np.random.RandomState(seed)
    for _ in range(500):
        pick = sorted(rng.choice(pats, size=k, replace=False).tolist())
        sub = tr[np.isin(groups[tr], pick)]
        if len(set(y[sub].tolist())) == n_classes:
            return sub, pick
    raise SystemExit(f"cannot draw {k} training patients covering every class")


def run_transfer_fold(f, args, images, y, groups, files, fold_idx, labels,
                      size, fold_dir):
    """One outer fold: a cross-validation ensemble over inner patient folds."""
    te = np.where(fold_idx == f)[0]
    tr = np.where(fold_idx != f)[0]
    tr, train_pats = restrict_training(tr, groups, y, len(labels),
                                       args.train_patients,
                                       args.subsample_seed * 100 + f)
    held = sorted(set(groups[te]))
    print(f"\n=== Fold {f+1}/{args.folds} | backbone {args.backbone} | "
          f"held-out patients {', '.join(held)} ({len(te)} crops) ===",
          flush=True)

    inner_of = burnml.grouped_folds(groups[tr], args.inner_folds, y[tr],
                                    len(labels))
    inner_idx = np.array([inner_of[g] for g in groups[tr]])

    member_test_probs, member_test_notta, histories, inner_oof = [], [], {}, []
    monitor_patients, steps = [], []
    pre = B.preprocess_fn(args.backbone)
    for m in range(args.inner_folds):
        va = tr[inner_idx == m]
        it = tr[inner_idx != m]
        if len(va) == 0 or len(np.unique(y[it])) < 2:
            print(f"  member {m+1}: skipped (degenerate inner split)")
            continue
        missing = sorted(set(range(len(labels))) - set(np.unique(y[it])))
        if missing:
            print(f"  member {m+1}: WARNING training data lacks "
                  f"{[labels[i] for i in missing]}")
        mon = sorted(set(groups[va]))
        monitor_patients.append(mon)
        print(f"  member {m+1}/{args.inner_folds}: train {len(it)} crops, "
              f"monitor {len(va)} crops from patients {', '.join(mon)}",
              flush=True)
        burnml.free_memory()
        model, hist, n_tr, n_steps = burnml.fit_transfer(
            args.backbone,
            [images[i] for i in it], y[it],
            [images[i] for i in va], y[va],
            len(labels), size, args.seed + 100 * f + m,
            probe_epochs=args.probe_epochs,
            finetune_epochs=args.finetune_epochs,
            probe_lr=args.probe_lr, finetune_lr=args.finetune_lr,
            dropout=args.dropout, batch_size=args.batch_size,
            label_smoothing=args.label_smoothing,
            photometric=not args.no_photometric, verbose=2,
            weights=None if args.weights == "none" else "imagenet",
            passes_per_epoch=args.passes_per_epoch,
            cosine=not args.no_cosine)
        histories[m + 1] = hist
        steps.append(int(n_steps))

        # inner out-of-fold predictions: only this member is clean on these
        # patients, so only this member contributes to the calibration set
        p_va = burnml.predict_tta(model, [images[i] for i in va], size, pre,
                                  n_tta=args.tta)
        inner_oof.append((va, p_va))

        te_imgs = [images[i] for i in te]
        p_te = burnml.predict_tta(model, te_imgs, size, pre, n_tta=args.tta)
        p_te1 = (p_te if args.tta <= 1 else
                 burnml.predict_tta(model, te_imgs, size, pre, n_tta=1))
        member_test_probs.append(p_te)
        member_test_notta.append(p_te1)
        acc_m = float((p_te.argmax(1) == y[te]).mean())
        print(f"    member {m+1} accuracy on held-out patients: {acc_m:.3f}")
        if args.save_models:
            burnml.save_member(model, os.path.join(
                args.out_dir, "models", f"fold_{f+1}", f"member_{m+1}"))
        del model
        burnml.free_memory()

    if not member_test_probs:
        raise SystemExit(f"Fold {f+1}: no ensemble member trained.")
    raw = np.mean(member_test_probs, axis=0)
    probs = raw

    temperature = 1.0
    if not args.no_temperature and inner_oof:
        cal_idx = np.concatenate([v for v, _ in inner_oof])
        cal_p = np.concatenate([p for _, p in inner_oof])
        temperature = burnml.fit_temperature(cal_p, y[cal_idx])
        print(f"  temperature fitted on {len(cal_idx)} inner out-of-fold "
              f"crops: T = {temperature:.3f}"
              + ("  (>1 means the raw model was over-confident)"
                 if temperature > 1 else ""))
        probs = burnml.apply_temperature(raw, temperature)

    acc = float((probs.argmax(1) == y[te]).mean())
    print(f"  ENSEMBLE accuracy on held-out patients: {acc:.3f} "
          f"({int((probs.argmax(1) == y[te]).sum())}/{len(te)})", flush=True)

    return {
        "fold": f + 1, "held_out_patients": ";".join(held),
        "n_patients": len(held), "n_crops": int(len(te)),
        "accuracy": acc, "loader": "tf.data online augmentation",
        "augmented_training_images": None,
        "epochs_run": args.probe_epochs + args.finetune_epochs,
        "ensemble_members": len(member_test_probs),
        "member_accuracies": [float((p.argmax(1) == y[te]).mean())
                              for p in member_test_probs],
        "member_monitor_patients": monitor_patients,
        "optimiser_steps_per_member": steps,
        "training_patients": train_pats,
        "temperature": temperature,
        "files": [files[i] for i in te],
        "true_labels": [labels[int(v)] for v in y[te]],
        # primary: ensemble + TTA + temperature scaling
        "probs": probs.tolist(),
        # ablation ladder for the supplement
        "probs_uncalibrated": raw.tolist(),
        "probs_no_tta": np.mean(member_test_notta, axis=0).tolist(),
        "probs_single_model": member_test_probs[0].tolist(),
        "hyperparameters": {"backbone": args.backbone,
                            "probe_lr": args.probe_lr,
                            "finetune_lr": args.finetune_lr,
                            "dropout": args.dropout,
                            "batch_size": args.batch_size},
        "history": histories,
    }


def run_randominit_fold(f, args, images, y, groups, files, fold_idx, labels,
                        size, fold_dir):
    """One outer fold of the random-initialisation arm.

    For each learning rate in the grid, the three inner-fold members are
    trained; the rate with the lowest inner out-of-fold cross-entropy is
    chosen, its members form the ensemble, and the temperature is fitted on
    the same inner out-of-fold predictions. The held-out patients influence
    nothing but the reported predictions.
    """
    import shutil
    te = np.where(fold_idx == f)[0]
    tr = np.where(fold_idx != f)[0]
    tr, train_pats = restrict_training(tr, groups, y, len(labels),
                                       args.train_patients,
                                       args.subsample_seed * 100 + f)
    held = sorted(set(groups[te]))
    print(f"\n=== Fold {f+1}/{args.folds} | RANDOM-INIT {args.backbone} | "
          f"held-out patients {', '.join(held)} ({len(te)} crops) ===",
          flush=True)
    inner_of = burnml.grouped_folds(groups[tr], args.inner_folds, y[tr],
                                    len(labels))
    inner_idx = np.array([inner_of[g] for g in groups[tr]])
    pre = B.preprocess_fn(args.backbone)
    te_imgs = [images[i] for i in te]
    stage = os.path.join(args.out_dir, "models", f"fold_{f+1}", "_lr_candidates")

    results = {}
    prior, prior_ce = None, {}
    if args.extend_from:
        pf = os.path.join(args.extend_from, "folds", f"fold_{f+1}.json")
        prior = json.load(open(pf))
        if prior.get("held_out_patients") != ";".join(held):
            raise SystemExit(f"--extend-from fold {f+1} held out different patients")
        if prior["settings"].get("backbone") != args.backbone:
            raise SystemExit("--extend-from is a different backbone")
        prior_ce = {float(k): float(v) for k, v in prior["lr_selection_inner_ce"].items()}
        print(f"  extending an earlier run: its inner CE "
              + ", ".join("%g: %.4f" % kv for kv in sorted(prior_ce.items())), flush=True)
    for lr in args.lr_grid:
        if lr in prior_ce:
            continue                       # already trained in the earlier run
        r = {"test": [], "test_notta": [], "oof": [], "hist": {}, "steps": [],
             "monitor": [], "member_acc": []}
        for m in range(args.inner_folds):
            va = tr[inner_idx == m]
            it = tr[inner_idx != m]
            if len(va) == 0 or len(np.unique(y[it])) < 2:
                continue
            r["monitor"].append(sorted(set(groups[va].tolist())))
            print(f"  lr {lr:g} | member {m+1}/{args.inner_folds}: train "
                  f"{len(it)} crops, monitor {len(va)}", flush=True)
            burnml.free_memory()
            model, hist, n_steps = burnml.fit_randominit(
                args.backbone, [images[i] for i in it], y[it],
                [images[i] for i in va], y[va], len(labels), size,
                args.seed + 100 * f + m, lr=lr, epochs=args.ri_epochs,
                warmup_epochs=args.warmup_epochs,
                weight_decay=args.weight_decay, dropout=args.dropout,
                batch_size=args.batch_size,
                label_smoothing=args.label_smoothing,
                photometric=not args.no_photometric,
                passes_per_epoch=args.passes_per_epoch, verbose=2)
            r["hist"][m + 1] = hist
            r["steps"].append(n_steps)
            r["oof"].append((va, burnml.predict_tta(
                model, [images[i] for i in va], size, pre, n_tta=args.tta)))
            p_te = burnml.predict_tta(model, te_imgs, size, pre, n_tta=args.tta)
            r["test"].append(p_te)
            r["test_notta"].append(p_te if args.tta <= 1 else
                                   burnml.predict_tta(model, te_imgs, size,
                                                      pre, n_tta=1))
            r["member_acc"].append(float((p_te.argmax(1) == y[te]).mean()))
            if args.save_models:
                burnml.save_member(model, os.path.join(
                    stage, f"lr_{lr:g}", f"member_{m+1}"))
            del model
            burnml.free_memory()
        if r["oof"]:
            ci = np.concatenate([v for v, _ in r["oof"]])
            cp = np.concatenate([q for _, q in r["oof"]])
            r["inner_ce"] = burnml.cross_entropy(cp, y[ci])
            print(f"  lr {lr:g}: inner out-of-fold cross-entropy "
                  f"{r['inner_ce']:.4f}", flush=True)
            results[lr] = r

    all_ce = dict(prior_ce)
    all_ce.update({k: v["inner_ce"] for k, v in results.items()})
    if prior is not None and all_ce and min(all_ce, key=all_ce.get) in prior_ce:
        best = min(all_ce, key=all_ce.get)
        ce_txt = ", ".join("%g: %.4f" % kv for kv in sorted(all_ce.items()))
        print(f"  earlier choice {best:g} still has the lowest inner CE "
              f"({ce_txt}); its predictions are kept unchanged")
        rec = dict(prior)
        rec["lr_selection_inner_ce"] = {f"{k:g}": v for k, v in sorted(all_ce.items())}
        rec["hyperparameters"] = dict(prior.get("hyperparameters", {}),
                                      lr_grid=sorted(all_ce))
        rec["extension"] = {"new_rates": sorted(results), "chosen_from": "earlier run"}
        return rec
    if not results:
        raise SystemExit(f"Fold {f+1}: no random-init member trained.")
    best = min(all_ce, key=all_ce.get)           # a NEW rate (prior lost above)
    R = results[best]
    ce_txt = ", ".join("%g: %.4f" % kv for kv in sorted(all_ce.items()))
    print(f"  chosen learning rate {best:g} (inner CE {ce_txt})")

    if args.save_models:
        final = os.path.join(args.out_dir, "models", f"fold_{f+1}")
        src = os.path.join(stage, f"lr_{best:g}")
        for name in sorted(os.listdir(src)):
            dst = os.path.join(final, name)
            if os.path.exists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
            shutil.move(os.path.join(src, name), dst)
        shutil.rmtree(stage, ignore_errors=True)

    raw = np.mean(R["test"], axis=0)
    temperature = 1.0
    if not args.no_temperature:
        ci = np.concatenate([v for v, _ in R["oof"]])
        cp = np.concatenate([q for _, q in R["oof"]])
        temperature = burnml.fit_temperature(cp, y[ci])
        print(f"  temperature T = {temperature:.3f}")
    probs = burnml.apply_temperature(raw, temperature)
    acc = float((probs.argmax(1) == y[te]).mean())
    print(f"  ENSEMBLE accuracy on held-out patients: {acc:.3f}", flush=True)
    return {
        "fold": f + 1, "held_out_patients": ";".join(held),
        "n_patients": len(held), "n_crops": int(len(te)), "accuracy": acc,
        "loader": "tf.data online augmentation",
        "augmented_training_images": None, "epochs_run": args.ri_epochs,
        "ensemble_members": len(R["test"]),
        "member_accuracies": R["member_acc"],
        "member_monitor_patients": R["monitor"],
        "optimiser_steps_per_member": R["steps"],
        "training_patients": train_pats,
        "chosen_lr": best,
        "extension": ({"new_rates": sorted(results), "chosen_from": "new rate"}
                      if prior is not None else None),
        "lr_selection_inner_ce": {f"{k:g}": v for k, v in sorted(all_ce.items())},
        "temperature": temperature,
        "files": [files[i] for i in te],
        "true_labels": [labels[int(v)] for v in y[te]],
        "probs": probs.tolist(),
        "probs_uncalibrated": raw.tolist(),
        "probs_no_tta": np.mean(R["test_notta"], axis=0).tolist(),
        "probs_single_model": R["test"][0].tolist(),
        "hyperparameters": {"backbone": args.backbone, "lr": best,
                            "lr_grid": sorted(all_ce),
                            "epochs": args.ri_epochs,
                            "weight_decay": args.weight_decay,
                            "dropout": args.dropout,
                            "batch_size": args.batch_size},
        "history": R["hist"],
    }


def run_scratch_fold(f, args, images, y, groups, files, fold_idx, sizes,
                     labels, size, fixed_params):
    te = np.where(fold_idx == f)[0]
    tr = np.where(fold_idx != f)[0]
    held = sorted(set(groups[te]))
    print(f"\n=== Fold {f+1}/{args.folds} | scratch CNN | held-out patients "
          f"{', '.join(held)} ({len(te)} crops) ===", flush=True)

    if args.nested_search:
        print("  nested hyperparameter search on this fold's training "
              "patients only", flush=True)
        _spec.loader.exec_module(search_mod)
        params, _ = search_mod.run_search(
            [images[i] for i in tr], y[tr], groups[tr], labels, size,
            n_calls=args.search_calls, inner_folds=args.search_folds,
            epochs=args.search_epochs, patience=8, rotations=args.rotations,
            rotation_mode=args.rotation_mode, seed=args.seed + f,
            verbose=True, loader=args.loader,
            budget_gb=args.memory_budget_gb, max_batch=args.search_max_batch)
    else:
        params = fixed_params

    rng = random.Random(args.seed + f)
    val_groups = burnml.holdout_groups(groups[tr], sizes,
                                       args.inner_val_frac, rng)
    inner_val = tr[np.isin(groups[tr], list(val_groups))]
    inner_tr = tr[~np.isin(groups[tr], list(val_groups))]
    burnml.free_memory()
    model, hist, loader_used, n_train = burnml.fit_fold(
        params,
        [images[i] for i in inner_tr], y[inner_tr],
        [images[i] for i in inner_val], y[inner_val],
        len(labels), size, args.rotations, args.rotation_mode,
        args.epochs, args.patience, args.seed + f,
        use_class_weight=not args.no_class_weight,
        loader=args.loader, budget_gb=args.memory_budget_gb, verbose=2)
    probs = burnml.predict_uint8(model, [images[i] for i in te], size)
    acc = float((probs.argmax(1) == y[te]).mean())
    print(f"  fold accuracy on held-out patients: {acc:.3f}", flush=True)
    out = {
        "fold": f + 1, "held_out_patients": ";".join(held),
        "n_patients": len(held), "n_crops": int(len(te)), "accuracy": acc,
        "loader": loader_used, "augmented_training_images": int(n_train),
        "inner_validation_patients": sorted(val_groups),
        "epochs_run": len(hist.history["loss"]), "temperature": 1.0,
        "files": [files[i] for i in te],
        "true_labels": [labels[int(v)] for v in y[te]],
        "probs": probs.tolist(),
        "hyperparameters": burnml.jsonable(dict(params)),
        "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
    }
    del model
    burnml.free_memory()
    return out


def main():
    args = parse_args()
    size = args.img_size or (B.spec(args.backbone)["size"]
                             if args.arch in ("transfer", "randominit")
                             else burnml.DEFAULT_IMG_SIZE[args.task])
    labels, _ = burnml.TASKS[args.task]
    fold_dir = os.path.join(args.out_dir, "folds")
    os.makedirs(fold_dir, exist_ok=True)
    burnml.configure_gpu()
    burnml.set_seeds(args.seed)

    fixed_params = None
    if args.params:
        with open(args.params, "rb") as fh:
            fixed_params = pickle.load(fh)

    images, y, groups, files = burnml.load_dataset(args.manifest, args.raw_dir,
                                                   args.task)
    sizes = Counter(groups)
    print(f"Task '{args.task}' | {args.arch}"
          + (f" | {B.describe(args.backbone)}" if args.arch in ("transfer", "randominit") else ""))
    print(f"  {len(images)} crops from {len(sizes)} patients at {size}x{size}px")
    print("  class counts:",
          {labels[i]: int((y == i).sum()) for i in range(len(labels))})

    n_splits = min(args.folds, len(sizes))
    args.folds = n_splits
    fold_of = burnml.grouped_folds_seeded(
        groups, n_splits, y, len(labels),
        None if args.repeat == 0 else 1000 + args.repeat)
    fold_idx = np.array([fold_of[g] for g in groups])
    for g in sizes:
        assert len(set(fold_idx[groups == g])) == 1, f"patient {g} split"

    todo = ([args.only_fold - 1] if args.only_fold is not None
            else list(range(n_splits)))
    if args.only_fold is not None and not 1 <= args.only_fold <= n_splits:
        sys.exit(f"--only-fold must be between 1 and {n_splits}")

    for f in todo:
        out_path = os.path.join(fold_dir, f"fold_{f + 1}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"Fold {f + 1}: already done, skipping")
            continue
        if args.arch == "randominit":
            rec = run_randominit_fold(f, args, images, y, groups, files,
                                      fold_idx, labels, size, fold_dir)
        elif args.arch == "transfer":
            rec = run_transfer_fold(f, args, images, y, groups, files,
                                    fold_idx, labels, size, fold_dir)
        else:
            rec = run_scratch_fold(f, args, images, y, groups, files, fold_idx,
                                   sizes, labels, size, fixed_params)
        rec["settings"] = settings_record(args, size)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(burnml.jsonable(rec), fh, indent=2)
        os.replace(tmp, out_path)          # atomic: a killed job leaves no half file
        print(f"  wrote {out_path}")

    if args.only_fold is not None:
        print("\nFold complete. Aggregate with 03b_aggregate.py once every "
              "fold has been written.")
        return

    metrics, oof, y_pred, agg_fold, histories = burnml.aggregate_folds(
        fold_dir, files, y, groups, labels, args.n_boot, args.seed)
    design = dict(metrics.pop("_settings", {}))
    design.update({"labels": labels, "patients": len(sizes),
                   "crops": len(images), "outer_folds": n_splits,
                   "bootstrap_draws": args.n_boot})
    metrics["design"] = burnml.jsonable(design)
    metrics["calibration"] = burnml.jsonable(
        burnml.calibration_metrics(oof, y, labels))

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(burnml.jsonable(metrics), fh, indent=2)
    with open(os.path.join(args.out_dir, "training_history.json"), "w") as fh:
        json.dump(histories, fh, indent=2)
    with open(os.path.join(args.out_dir, "oof_predictions.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "patient_group", "fold", "true_label",
                    "predicted_label"] + [f"p_{c}" for c in labels])
        for i in range(len(files)):
            w.writerow([files[i], groups[i], agg_fold[i], labels[y[i]],
                        labels[y_pred[i]]] + [f"{p:.6f}" for p in oof[i]])

    burnml.print_results(metrics, labels,
                         f"{args.task} / {args.arch}"
                         + (f" / {args.backbone}" if args.arch in ("transfer", "randominit")
                            else ""))
    burnml.print_ablation(metrics)
    cal = metrics["calibration"]
    print(f"\n  calibration: Brier {cal['brier']:.3f} | ECE {cal['ece']:.3f} | "
          f"{cal['high_confidence_wrong']}/{cal['high_confidence_n']} of "
          f">0.9-confidence predictions wrong")
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
