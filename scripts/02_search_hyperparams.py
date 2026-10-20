#!/usr/bin/env python3
"""
02_search_hyperparams.py — Bayesian hyperparameter search scored by
patient-grouped inner cross-validation.

Differences from the original Model.py / model.py:
  * folds are formed from whole PATIENTS, not from contiguous slices of an
    already-shuffled augmented array;
  * augmentation happens inside each inner fold's training set only, and the
    validation images are never augmented;
  * the score is macro-F1 at the restored best epoch, not the maximum
    validation accuracy reached at any epoch during training;
  * the learning rate is sampled log-uniformly rather than uniformly, so small
    learning rates are actually explored.

Usage:
    python 02_search_hyperparams.py --raw-dir raw_Images \
        --manifest results/manifest.csv --task classification \
        --out results/classification/best_params.pickle

Exposes run_search() so 03_run_grouped_cv.py can perform a nested search
inside each outer fold.
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml


def make_space(img_size, max_batch=128):
    from skopt.space import Real, Integer, Categorical
    max_conv = burnml.max_conv_layers(img_size)
    space = [
        Integer(2, max_conv, name="num_conv_layers"),
        Integer(1, 3, name="num_dense_layers"),
    ]
    space += [Integer(32, 256, name=f"conv{i+1}_filters") for i in range(max_conv)]
    space += [Categorical([True, False], name=f"conv{i+1}_batchnorm") for i in range(max_conv)]
    space += [Categorical(["same", "valid"], name=f"conv{i+1}_padding") for i in range(max_conv)]
    space += [Categorical([True, False], name=f"conv{i+1}_l2") for i in range(max_conv)]
    space += [Integer(32, 256, name=f"dense{i+1}_units") for i in range(3)]
    space += [Categorical([True, False], name=f"dense{i+1}_batchnorm") for i in range(3)]
    space += [Real(0.0, 0.5, name=f"dense{i+1}_dropout") for i in range(3)]
    space += [
        Real(1e-5, 1e-2, prior="log-uniform", name="learning_rate"),
        Real(0.0, 0.5, name="global_dropout"),
        # default range 16-128; a lower ceiling shrinks the floor too, because
        # skopt needs low < high
        Integer(min(16, max(2, int(max_batch) // 2)), max(3, int(max_batch)),
                name="batch_size"),
    ]
    return space


def score_params(params, images, y, groups, labels, cfg):
    """Mean macro-F1 across patient-grouped inner folds.

    Memory is released between every fold, so a 30-call search does not
    accumulate graphs or arrays.
    """
    fold_of = burnml.grouped_folds(groups, cfg["inner_folds"], y,
                                   len(labels))
    fold_idx = np.array([fold_of[g] for g in groups])
    scores = []

    for f in range(cfg["inner_folds"]):
        va = np.where(fold_idx == f)[0]
        tr = np.where(fold_idx != f)[0]
        if len(va) == 0 or len(np.unique(y[tr])) < len(labels):
            continue
        burnml.free_memory()
        model, _hist, _loader, _n = burnml.fit_fold(
            params,
            [images[i] for i in tr], y[tr],
            [images[i] for i in va], y[va],
            len(labels), cfg["img_size"], cfg["rotations"],
            cfg["rotation_mode"], cfg["epochs"], cfg["patience"],
            cfg["seed"] + f, loader=cfg["loader"],
            budget_gb=cfg["budget_gb"], verbose=0)
        probs = burnml.predict_uint8(model, [images[i] for i in va],
                                     cfg["img_size"])
        pred = probs.argmax(axis=1)
        scores.append(burnml.compute_metrics(y[va], pred, probs,
                                             labels)["macro_f1"])
        del model
        burnml.free_memory()
    return float(np.mean(scores)) if scores else 0.0


def run_search(images, y, groups, labels, img_size, n_calls=40, inner_folds=3,
               epochs=40, patience=8, rotations=24, rotation_mode="bound",
               seed=42, verbose=True, loader="auto", budget_gb=60.0,
               max_batch=128):
    """Return (best_params dict, search log list)."""
    from skopt import gp_minimize
    from skopt.utils import use_named_args

    space = make_space(img_size, max_batch)
    cfg = dict(inner_folds=inner_folds, epochs=epochs, patience=patience,
               rotations=rotations, rotation_mode=rotation_mode,
               img_size=img_size, seed=seed, loader=loader,
               budget_gb=budget_gb)
    log = []

    @use_named_args(space)
    def objective(**params):
        if verbose:
            p = burnml.normalise_params(params)
            n = p["num_conv_layers"]
            print(f"  call {len(log)+1:>3} starting: conv={n} "
                  f"filters={[p[f'conv{i}_filters'] for i in range(1, n+1)]} "
                  f"padding={[p[f'conv{i}_padding'][0] for i in range(1, n+1)]} "
                  f"bn={[int(p[f'conv{i}_batchnorm']) for i in range(1, n+1)]} "
                  f"dense={[p[f'dense{i}_units'] for i in range(1, p['num_dense_layers']+1)]} "
                  f"bs={p['batch_size']} lr={p['learning_rate']:.1e} "
                  f"flatten={burnml.flatten_features(p, img_size)}",
                  flush=True)
        s = score_params(params, images, y, groups, labels, cfg)
        log.append({"params": burnml.jsonable(burnml.normalise_params(params)),
                    "macro_f1": s})
        if verbose:
            print(f"  call {len(log):>3}  macro-F1 {s:.4f}  "
                  f"conv={params['num_conv_layers']} "
                  f"dense={params['num_dense_layers']} "
                  f"lr={params['learning_rate']:.2e} "
                  f"bs={params['batch_size']}", flush=True)
        return -s

    # skopt defaults to 10 random probes and errors if n_calls is smaller,
    # after the data has already been loaded. Scale the probes to the budget
    # instead so a short exploratory run is possible.
    n_initial = max(3, min(10, n_calls // 3 if n_calls < 30 else 10))
    if n_calls < n_initial:
        raise SystemExit(f"--n-calls must be at least {n_initial}")
    result = gp_minimize(objective, space, n_calls=n_calls,
                         n_initial_points=n_initial,
                         random_state=seed, verbose=False)
    best = burnml.normalise_params({d.name: v for d, v in zip(space, result.x)})
    return best, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", choices=list(burnml.TASKS), required=True)
    ap.add_argument("--out", required=True, help="best_params.pickle path")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--n-calls", type=int, default=40)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--rotations", type=int, default=24)
    ap.add_argument("--rotation-mode", choices=["bound", "inscribed"],
                    default="bound")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loader", choices=["auto", "memory", "stream"],
                    default="auto",
                    help="auto keeps rotations in RAM as uint8 while they fit "
                         "inside --memory-budget-gb, otherwise streams them")
    ap.add_argument("--memory-budget-gb", type=float, default=60.0)
    ap.add_argument("--max-batch", type=int, default=128,
                    help="upper bound of the batch-size search range; lower it "
                         "on GPUs with less than 16 GB")
    args = ap.parse_args()

    size = args.img_size or burnml.DEFAULT_IMG_SIZE[args.task]
    burnml.set_seeds(args.seed)
    burnml.configure_gpu()
    labels, _ = burnml.TASKS[args.task]
    images, y, groups, _ = burnml.load_dataset(args.manifest, args.raw_dir,
                                               args.task)
    est = burnml.estimate_gb(len(images), args.rotations, size)
    print(f"Search for '{args.task}': {len(images)} crops, "
          f"{len(set(groups))} patients, {size}x{size}px, "
          f"{args.n_calls} calls x {args.inner_folds} patient-grouped folds")
    print(f"  augmented training set ~{est:.2f} GB as uint8 | loader "
          f"{args.loader} | budget {args.memory_budget_gb:.1f} GB")

    best, log = run_search(images, y, groups, labels, size,
                           n_calls=args.n_calls, inner_folds=args.inner_folds,
                           epochs=args.epochs, patience=args.patience,
                           rotations=args.rotations,
                           rotation_mode=args.rotation_mode, seed=args.seed,
                           loader=args.loader,
                           budget_gb=args.memory_budget_gb,
                           max_batch=args.max_batch)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(best, fh)
    logpath = os.path.splitext(args.out)[0] + "_search_log.json"
    with open(logpath, "w") as fh:
        json.dump({"task": args.task, "img_size": size, "seed": args.seed,
                   "n_calls": args.n_calls, "inner_folds": args.inner_folds,
                   "best": burnml.jsonable(best),
                   "best_macro_f1": max(e["macro_f1"] for e in log),
                   "calls": log}, fh, indent=2)
    print("\nBest parameters:")
    for k in sorted(best):
        print(f"  {k:<22} {best[k]}")
    print(f"\nWrote {args.out}\nWrote {logpath}")


if __name__ == "__main__":
    main()
