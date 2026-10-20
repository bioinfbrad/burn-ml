#!/usr/bin/env python3
"""
03b_aggregate.py — pool per-fold predictions into the final result set.

Needed when the folds were run as separate SLURM array tasks. Reads every
folds/fold_<k>.json, checks that each crop was predicted exactly once, then
writes metrics.json, oof_predictions.csv and training_history.json.

It refuses to produce a result while any crop is still unpredicted, so a
half-finished array cannot silently yield an optimistic subset.

Usage:
    python 03b_aggregate.py --raw-dir raw_Images \
        --manifest results/manifest.csv --task classification \
        --out-dir results/classification
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", choices=list(burnml.TASKS), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels, _ = burnml.TASKS[args.task]
    fold_dir = os.path.join(args.out_dir, "folds")
    images, y, groups, files = burnml.load_dataset(args.manifest, args.raw_dir,
                                                   args.task)
    del images                                    # only labels are needed here

    metrics, oof, y_pred, fold_idx, histories = burnml.aggregate_folds(
        fold_dir, files, y, groups, labels, args.n_boot, args.seed)

    design_path = os.path.join(args.out_dir, "metrics.json")
    design = dict(metrics.pop("_settings", {}))       # recorded by each fold
    if os.path.exists(design_path):
        prior = json.load(open(design_path)).get("design", {})
        for k, v in prior.items():
            design.setdefault(k, v)
    design.update({"task": args.task, "labels": labels,
                   "patients": len(set(groups)), "crops": len(files),
                   "outer_folds": len(metrics["per_fold"]),
                   "aggregated_by": "03b_aggregate.py",
                   "bootstrap_draws": args.n_boot, "seed": args.seed})
    metrics["design"] = burnml.jsonable(design)
    metrics["calibration"] = burnml.jsonable(
        burnml.calibration_metrics(oof, y, labels))

    with open(design_path, "w") as fh:
        json.dump(burnml.jsonable(metrics), fh, indent=2)
    with open(os.path.join(args.out_dir, "training_history.json"), "w") as fh:
        json.dump(histories, fh, indent=2)
    with open(os.path.join(args.out_dir, "oof_predictions.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "patient_group", "fold", "true_label",
                    "predicted_label"] + [f"p_{c}" for c in labels])
        for i in range(len(files)):
            w.writerow([files[i], groups[i], fold_idx[i], labels[y[i]],
                        labels[y_pred[i]]] + [f"{p:.6f}" for p in oof[i]])

    burnml.print_results(metrics, labels, args.task)
    burnml.print_ablation(metrics)
    cal = metrics["calibration"]
    print(f"\n  calibration: Brier {cal['brier']:.3f} | ECE {cal['ece']:.3f} | "
          f"{cal['high_confidence_wrong']}/{cal['high_confidence_n']} of "
          f">0.9-confidence predictions wrong")
    for c in labels:
        pc = cal["per_class"][c]
        print(f"    {c}: calibration intercept "
              f"{pc['calibration_intercept']:+.2f}, slope "
              f"{pc['calibration_slope']:.2f}")
    print(f"\nWrote metrics.json, oof_predictions.csv, training_history.json "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
