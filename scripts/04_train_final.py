#!/usr/bin/env python3
"""
04_train_final.py — train the archival model on all patients, for deposit.

This model is what goes to Zenodo. It is NOT the source of any reported
number: every reported metric comes from the out-of-fold predictions of
03_run_grouped_cv.py.

  --arch transfer  the pre-specified recipe, trained on all 284 crops. The
                   recipe uses a fixed epoch budget, so no hold-out is needed.
                   The temperature applied at inference is the median of the
                   per-fold temperatures from cross-validation, read from that
                   run's metrics.json, and is saved alongside the model.
  --arch scratch   the original CNN baseline with best_params.pickle, as before.

Usage:
    python 04_train_final.py --raw-dir raw_Images \
        --manifest results/manifest.csv --task classification \
        --arch transfer --backbone efficientnetb0 \
        --cv-dir results/classification/efficientnetb0 \
        --out-dir results/classification/efficientnetb0/final
"""
import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml
import backbones as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", choices=list(burnml.TASKS), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arch", choices=["transfer", "scratch"], default="transfer")
    ap.add_argument("--backbone", default=B.PRIMARY)
    ap.add_argument("--cv-dir", default=None,
                    help="the CV results for this model, to read the fitted "
                         "temperatures from")
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--members", type=int, default=3,
                    help="independently seeded models trained on all patients; "
                         "averaged at inference (the archived ensemble)")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # transfer recipe: must match the cross-validation run
    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=30)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--passes-per-epoch", type=int, default=4)
    ap.add_argument("--no-photometric", action="store_true")
    ap.add_argument("--no-cosine", action="store_true")
    # scratch options
    ap.add_argument("--params", default=None)
    ap.add_argument("--rotations", type=int, default=24)
    ap.add_argument("--rotation-mode", choices=["bound", "inscribed"],
                    default="bound")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--loader", choices=["auto", "memory", "stream"],
                    default="auto")
    ap.add_argument("--memory-budget-gb", type=float, default=60.0)
    args = ap.parse_args()

    burnml.configure_gpu()
    burnml.set_seeds(args.seed)
    labels, _ = burnml.TASKS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    images, y, groups, _ = burnml.load_dataset(args.manifest, args.raw_dir,
                                               args.task)
    record = {"task": args.task, "architecture": args.arch, "labels": labels,
              "seed": args.seed, "patients": len(set(groups)),
              "crops": len(images), "environment": burnml.env_record(),
              "note": ("Archival artefact only. All reported performance comes "
                       "from the out-of-fold predictions produced by "
                       "03_run_grouped_cv.py.")}

    if args.arch == "transfer":
        if not B.available(args.backbone):
            sys.exit(f"Backbone '{args.backbone}' not in this TensorFlow.")
        size = args.img_size or B.spec(args.backbone)["size"]
        temperature, temps = 1.0, []
        if args.cv_dir and os.path.exists(os.path.join(args.cv_dir,
                                                       "metrics.json")):
            temps = json.load(open(os.path.join(
                args.cv_dir, "metrics.json"))).get("temperatures", [])
            if temps:
                temperature = float(np.median(temps))
        print(f"Final {args.backbone} on all {len(images)} crops from "
              f"{len(set(groups))} patients at {size}px; inference "
              f"temperature {temperature:.3f} (median of {len(temps)} CV folds)")
        # the monitoring set only feeds the logged curve; nothing is selected
        # on it, so any patients will do
        rng = random.Random(args.seed)
        mon = burnml.holdout_groups(groups, Counter(groups), 0.15, rng)
        mi = [i for i in range(len(images)) if groups[i] in mon]
        paths, hists, steps = [], [], []
        for k in range(args.members):
            burnml.free_memory()
            model, hist, n_tr, n_steps = burnml.fit_transfer(
                args.backbone, images, y, [images[i] for i in mi], y[mi],
                len(labels), size, args.seed + k,
                probe_epochs=args.probe_epochs, finetune_epochs=args.finetune_epochs,
                probe_lr=args.probe_lr, finetune_lr=args.finetune_lr,
                dropout=args.dropout, batch_size=args.batch_size,
                label_smoothing=args.label_smoothing,
                photometric=not args.no_photometric, verbose=2,
                weights=None if args.weights == "none" else "imagenet",
                passes_per_epoch=args.passes_per_epoch, cosine=not args.no_cosine)
            paths.append(burnml.save_model(model, os.path.join(
                args.out_dir, f"final_{args.task}_{args.backbone}_m{k + 1}")))
            hists.append(hist); steps.append(int(n_steps))
            del model
        path = paths[0]
        hist = {f"member_{k + 1}": h for k, h in enumerate(hists)}
        n_steps = steps[0]
        record.update({
            "backbone": args.backbone,
            "backbone_keras_name": B.spec(args.backbone)["keras_name"],
            "weights": args.weights, "image_size": size,
            "training_crops": len(images), "optimiser_steps": int(n_steps),
            "unfrozen_backbone_layers": int(n_tr),
            "inference_temperature": temperature,
            "cv_temperatures": temps,
            "preprocessing": f"tf.keras.applications."
                             f"{B.spec(args.backbone)['module']}.preprocess_input "
                             "on RGB input in [0, 255]",
            "inference": "mean of the members' softmax outputs with 8-transform "
                         "dihedral TTA, then softmax(log(p) / T)",
            "members": args.members, "model_paths": paths,
            "recipe": {k: getattr(args, k) for k in (
                "probe_epochs", "finetune_epochs", "probe_lr", "finetune_lr",
                "dropout", "batch_size", "label_smoothing", "passes_per_epoch")},
            "history": hist,
        })
    else:
        if not args.params:
            sys.exit("--arch scratch needs --params best_params.pickle")
        size = args.img_size or burnml.DEFAULT_IMG_SIZE[args.task]
        with open(args.params, "rb") as fh:
            params = pickle.load(fh)
        rng = random.Random(args.seed)
        vg = burnml.holdout_groups(groups, Counter(groups), args.holdout_frac,
                                   rng)
        va = np.where(np.isin(groups, list(vg)))[0]
        tr = np.where(~np.isin(groups, list(vg)))[0]
        model, hist, loader, n_train = burnml.fit_fold(
            params, [images[i] for i in tr], y[tr], [images[i] for i in va],
            y[va], len(labels), size, args.rotations, args.rotation_mode,
            args.epochs, args.patience, args.seed,
            use_class_weight=not args.no_class_weight, loader=args.loader,
            budget_gb=args.memory_budget_gb, verbose=2)
        path = burnml.save_model(model, os.path.join(
            args.out_dir, f"final_{args.task}_scratch"))
        record.update({"image_size": size, "training_crops": int(len(tr)),
                       "early_stopping_patients": sorted(vg),
                       "augmented_training_images": int(n_train),
                       "hyperparameters": burnml.normalise_params(params)})

    record["model_path"] = path
    with open(os.path.join(args.out_dir, "final_model_record.json"), "w") as fh:
        json.dump(burnml.jsonable(record), fh, indent=2)
    print(f"\nSaved {path}\nWrote final_model_record.json")


if __name__ == "__main__":
    main()
