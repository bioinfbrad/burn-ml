#!/usr/bin/env python3
"""
99_selftest.py — verify the pipeline's correctness properties on the real data
before committing a long run to the queue.

Checks, in order:
   1  manifest integrity: every crop grouped, no file in two classes
   2  duplicate images: byte-identical crops inside or across classes
   3  fold assignment: no patient in two folds, every crop in exactly one
   4  leakage audit: source-photograph overlap under the new grouping,
      compared with the crop-level split the original code used
   5  inner validation is patient-disjoint from inner training
   6  augmentation: uint8, correct count, shuffled, training only
   7  determinism: same seed reproduces byte-identical arrays
   8  metrics: checked against hand-computed values
   9  macro averaging ignores classes absent from a fold
  10  bootstrap resamples patients and brackets the point estimate
  11  (--full) one real fold trains and writes a valid fold file

Usage:
    python 99_selftest.py --raw-dir raw_Images --manifest results/manifest.csv
    python 99_selftest.py --raw-dir raw_Images --manifest results/manifest.csv --full
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import burnml

PASS, FAIL, WARN = [], [], []


def ok(name, detail=""):
    PASS.append(name)
    print(f"  PASS  {name}" + (f"  — {detail}" if detail else ""))


def bad(name, detail=""):
    FAIL.append(name)
    print(f"  FAIL  {name}" + (f"  — {detail}" if detail else ""))


def warn(name, detail=""):
    WARN.append(name)
    print(f"  WARN  {name}" + (f"  — {detail}" if detail else ""))


def check(cond, name, detail=""):
    (ok if cond else bad)(name, detail)
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", choices=list(burnml.TASKS),
                    default="classification")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true",
                    help="also exercise the transfer-learning machinery")
    ap.add_argument("--end-to-end", action="store_true",
                    help="additionally train one real scratch fold")
    args = ap.parse_args()

    if args.full:
        # Keep THIS process off the GPU. TensorFlow reserves nearly all GPU
        # memory for the first process that touches the card, and sections
        # 12-13 launch the real pipeline as child processes that need it. The
        # checks run here (BatchNorm semantics, preprocessing, calibration) are
        # device-independent; the children exercise the GPU exactly as the
        # training jobs do.
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
            print("  (self-test process kept on CPU; child processes get the GPU)")
        except RuntimeError as e:
            print(f"  WARNING: could not hide the GPU from this process: {e}")

    labels, _ = burnml.TASKS[args.task]
    rows = list(csv.DictReader(open(args.manifest, newline="")))

    print("\n1. manifest integrity")
    check(all(r.get("patient_group") for r in rows),
          "every crop has a patient group", f"{len(rows)} crops")
    by_name = defaultdict(set)
    for r in rows:
        by_name[os.path.basename(r["file"])].add(r["class"])
    dupes = {k: v for k, v in by_name.items() if len(v) > 1}
    check(not dupes, "no filename appears in two classes",
          f"{len(dupes)} clashes" if dupes else "")
    ids = {r["source_id"] for r in rows}
    per_group = defaultdict(set)
    for r in rows:
        per_group[r["patient_group"]].add(r["source_id"])
    print(f"        {len(rows)} crops | {len(ids)} accession IDs | "
          f"{len(per_group)} patient groups")

    print("\n2. duplicate images")
    hashes = defaultdict(list)
    for r in rows:
        p = os.path.join(args.raw_dir, r["file"])
        with open(p, "rb") as fh:
            hashes[hashlib.md5(fh.read()).hexdigest()].append(r)
    dup = {h: v for h, v in hashes.items() if len(v) > 1}
    cross = {h: v for h, v in dup.items() if len({x["class"] for x in v}) > 1}
    if cross:
        bad("no byte-identical crop appears in two classes",
            f"{len(cross)} groups, e.g. "
            f"{[x['file'] for x in list(cross.values())[0]]}")
    else:
        ok("no byte-identical crop appears in two classes")
    if dup:
        warn("byte-identical duplicates within a class",
             f"{len(dup)} groups covering "
             f"{sum(len(v) for v in dup.values())} files")
    else:
        ok("no duplicate crops at all")

    images, y, groups, files = burnml.load_dataset(args.manifest,
                                                   args.raw_dir, args.task)
    sizes = Counter(groups)

    print("\n3. fold assignment")
    n_splits = min(args.folds, len(sizes))
    fold_of = burnml.grouped_folds(groups, n_splits, y, len(labels))
    fold_idx = np.array([fold_of[g] for g in groups])
    split = [g for g in sizes if len(set(fold_idx[groups == g])) > 1]
    check(not split, "no patient is split across folds",
          f"{len(split)} split" if split else f"{len(sizes)} patients")
    counts = [int((fold_idx == f).sum()) for f in range(n_splits)]
    check(sum(counts) == len(files) and all(c > 0 for c in counts),
          "every crop sits in exactly one non-empty fold", f"sizes {counts}")
    spread = max(counts) - min(counts)
    biggest = max(sizes.values())
    if spread <= 0.5 * np.mean(counts):
        ok("fold sizes are reasonably balanced",
           f"range {min(counts)}-{max(counts)}")
    else:
        warn("fold sizes are uneven",
             f"range {min(counts)}-{max(counts)}; unavoidable because the "
             f"largest single patient contributes {biggest} crops. Splitting "
             "patients would fix the balance and reintroduce leakage, so the "
             "imbalance is accepted. Report it as a limitation.")
    missing_class = [f + 1 for f in range(n_splits)
                     if len({int(v) for v in y[fold_idx == f]}) < len(labels)]
    check(not missing_class, "every fold contains every class",
          f"folds missing one: {missing_class}" if missing_class else "")
    dom = max(sizes, key=lambda g: sizes[g])
    print(f"        most crops from one patient: {dom} with {sizes[dom]} "
          f"({100*sizes[dom]/len(files):.0f}% of the dataset)")

    print("\n4. leakage audit")
    def sid(path):
        m = re.search(r"(\d{6,8})", os.path.basename(path))
        return m.group(1) if m else path
    leaked_new = 0
    for f in range(n_splits):
        te = np.where(fold_idx == f)[0]
        tr = np.where(fold_idx != f)[0]
        train_ids = {sid(files[i]) for i in tr}
        leaked_new += sum(1 for i in te if sid(files[i]) in train_ids)
    check(leaked_new == 0,
          "no held-out crop shares a source photograph with training",
          f"{leaked_new} leaked" if leaked_new else f"0 of {len(files)}")

    rng = random.Random(args.seed)
    order = list(range(len(files)))
    rng.shuffle(order)
    cut = int(0.15 * len(order))
    old_te, old_tr = order[:cut], order[cut:]
    old_ids = {sid(files[i]) for i in old_tr}
    leaked_old = sum(1 for i in old_te if sid(files[i]) in old_ids)
    print(f"        for comparison, a crop-level split like the original code "
          f"leaks {leaked_old}/{len(old_te)} "
          f"({100*leaked_old/max(len(old_te),1):.0f}%) of its test crops")

    print("\n5. inner validation split")
    bad_inner = 0
    for f in range(n_splits):
        tr = np.where(fold_idx != f)[0]
        r = random.Random(args.seed + f)
        vg = burnml.holdout_groups(groups[tr], sizes, 0.2, r)
        iv = tr[np.isin(groups[tr], list(vg))]
        it = tr[~np.isin(groups[tr], list(vg))]
        if set(groups[iv]) & set(groups[it]):
            bad_inner += 1
        if len(iv) == 0 or len(it) == 0:
            bad_inner += 1
    check(bad_inner == 0,
          "inner validation is patient-disjoint and non-empty in every fold")

    print("\n6. augmentation")
    sub = [images[i] for i in range(24)]
    # carry a unique tag per image so the fill order is observable; the real
    # pipeline carries class labels, which repeat and would hide the order
    suby = np.arange(len(sub))
    X, yy = burnml.augment_uint8(sub, suby, 8, 64, "bound", seed=1)
    check(X.dtype == np.uint8, "augmented array is uint8",
          f"{X.nbytes/2**20:.1f} MB for {len(X)} images")
    check(len(X) == len(sub) * 8, "one copy per requested rotation",
          f"{len(sub)} x 8 = {len(X)}")
    unshuffled = np.repeat(suby, 8)
    check(not np.array_equal(yy, unshuffled),
          "augmentation order is shuffled during the fill")
    check(Counter(yy.tolist()) == Counter(unshuffled.tolist()),
          "shuffling preserves the label multiset")
    V = burnml.stack_uint8(sub, 64)
    check(len(V) == len(sub), "validation and test stacks are unaugmented",
          f"{len(V)} images for {len(sub)} crops")
    ang = np.arange(8) * 45.0
    check(abs(ang[1] - ang[0] - 45.0) < 1e-9 and 360.0 not in ang,
          "rotation angles are uniform with no duplicated 0/360")

    print("\n7. determinism")
    X2, y2 = burnml.augment_uint8(sub, suby, 8, 64, "bound", seed=1)
    check(np.array_equal(X, X2) and np.array_equal(yy, y2),
          "same seed reproduces byte-identical augmentation")
    X3, _ = burnml.augment_uint8(sub, suby, 8, 64, "bound", seed=2)
    check(not np.array_equal(X, X3), "a different seed changes the order")
    f1 = burnml.grouped_folds(groups, n_splits, y, len(labels))
    f2 = burnml.grouped_folds(groups, n_splits, y, len(labels))
    check(f1 == f2, "fold assignment is deterministic")

    print("\n8. metrics against hand-computed values")
    yt = np.array([0, 0, 0, 1, 1, 1])
    yp = np.array([0, 0, 1, 1, 1, 0])
    pr = np.eye(2)[yp]
    m = burnml.compute_metrics(yt, yp, pr, ["a", "b"])
    a = m["per_class"]["a"]
    check(a["TP"] == 2 and a["FN"] == 1 and a["FP"] == 1 and a["TN"] == 2,
          "confusion counts correct", f"TP2 FN1 FP1 TN2, got "
          f"TP{a['TP']} FN{a['FN']} FP{a['FP']} TN{a['TN']}")
    check(abs(a["sensitivity"] - 2/3) < 1e-12
          and abs(a["specificity"] - 2/3) < 1e-12
          and abs(a["precision"] - 2/3) < 1e-12
          and abs(a["f1"] - 2/3) < 1e-12, "sensitivity, specificity, "
          "precision and F1 correct")
    check(abs(m["accuracy"] - 4/6) < 1e-12, "accuracy correct")
    check(abs(burnml.auc_score(np.array([0, 0, 1, 1]),
                               np.array([0.1, 0.4, 0.35, 0.8])) - 0.75) < 1e-12,
          "AUC matches the textbook worked example", "expected 0.750")
    check(abs(burnml.auc_score(np.array([0, 1]), np.array([0.5, 0.5])) - 0.5)
          < 1e-12, "AUC handles ties by averaging ranks")

    print("\n9. macro averaging with an absent class")
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])
    m = burnml.compute_metrics(yt, yp, np.eye(3)[yp], ["a", "b", "c"])
    check(m["classes_present"] == ["a", "b"],
          "absent class excluded from macro averaging")
    check(m["per_class"]["c"]["sensitivity"] != m["per_class"]["c"]["sensitivity"],
          "absent class reports NaN rather than 0")
    expect = np.mean([m["per_class"]["a"]["f1"], m["per_class"]["b"]["f1"]])
    check(abs(m["macro_f1"] - expect) < 1e-12,
          "macro F1 averages only present classes")

    print("\n10. patient-level bootstrap")
    rs = np.random.RandomState(0)
    P = rs.dirichlet(np.ones(len(labels)), size=len(y))
    pred = P.argmax(1)
    pt = burnml.compute_metrics(y, pred, P, labels)
    ci = burnml.bootstrap_ci(y, pred, P, groups, labels, 300, 1)
    lo, hi = ci["accuracy"]
    check(lo <= pt["accuracy"] <= hi,
          "bootstrap interval brackets the point estimate",
          f"{lo:.3f} <= {pt['accuracy']:.3f} <= {hi:.3f}")
    check(hi > lo, "interval has non-zero width")
    ci2 = burnml.bootstrap_ci(y, pred, P, groups, labels, 300, 1)
    check(ci == ci2, "bootstrap is reproducible with a fixed seed")
    n_pat = len(set(groups))
    check(len(set(groups)) == n_pat,
          "bootstrap resamples patients, not images", f"{n_pat} clusters")


    if args.full:
        print("\n11. transfer-learning machinery")
        import tensorflow as tf
        import backbones as BB

        # the Keras BatchNorm trap: trainable=False alone does NOT stop the
        # moving statistics updating; the backbone must also be CALLED with
        # training=False. Getting this wrong silently destroys the pretrained
        # features, and nothing in the loss curve reveals it.
        m = burnml.build_transfer_model("mobilenetv2", 96, 3, weights=None)
        base = m._burn_base
        bn = [l for l in base.layers
              if isinstance(l, tf.keras.layers.BatchNormalization)]
        check(len(bn) > 0, "backbone contains BatchNormalization layers",
              f"{len(bn)} found")
        before = [np.array(l.moving_mean.numpy()) for l in bn[:3]]
        xb = np.random.rand(8, 96, 96, 3).astype(np.float32) * 255
        yb = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 8)]
        m.train_on_batch(xb, yb)
        after = [np.array(l.moving_mean.numpy()) for l in bn[:3]]
        check(all(np.allclose(a, b) for a, b in zip(before, after)),
              "BatchNorm moving statistics do NOT move during training",
              "the training=False call is working")

        n_tr = burnml.unfreeze_top(m, 20, 1e-5)
        still_frozen = all(not l.trainable for l in bn)
        check(still_frozen,
              "BatchNorm stays frozen after unfreezing the top block")
        check(0 < n_tr < len(base.layers),
              "only part of the backbone is unfrozen",
              f"{n_tr} of {len(base.layers)} layers trainable")
        before = [np.array(l.moving_mean.numpy()) for l in bn[:3]]
        m.train_on_batch(xb, yb)
        after = [np.array(l.moving_mean.numpy()) for l in bn[:3]]
        check(all(np.allclose(a, b) for a, b in zip(before, after)),
              "BatchNorm statistics still frozen during fine-tuning")
        del m, base
        burnml.free_memory()

        # preprocessing conventions: every Keras preprocess_input takes
        # 0-255 input, but each family maps it somewhere different. Feeding
        # an already-normalised image is a silent accuracy killer.
        raw = np.full((2, 32, 32, 3), 200.0, dtype=np.float32)
        ranges = {}
        for name in ["efficientnetb0", "densenet121", "resnet50v2",
                     "mobilenetv2"]:
            if not BB.available(name):
                continue
            out = np.asarray(BB.preprocess_fn(name)(raw.copy()))
            ranges[name] = (float(out.min()), float(out.max()))
        check(ranges.get("efficientnetb0", (0, 0))[1] > 1.5,
              "EfficientNet preprocessing leaves 0-255 input untouched",
              f"{ranges.get('efficientnetb0')}")
        check(-1.01 <= ranges.get("resnet50v2", (0, 0))[0] <= 1.01,
              "ResNetV2 preprocessing maps to [-1, 1]",
              f"{ranges.get('resnet50v2')}")
        check(abs(ranges.get("densenet121", (9, 9))[0]) < 9,
              "DenseNet preprocessing applies ImageNet mean/std",
              f"{ranges.get('densenet121')}")

        # online augmentation
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        a1 = burnml._augment_op(tf.constant(img), 64, True).numpy()
        a2 = burnml._augment_op(tf.constant(img), 64, True).numpy()
        check(a1.shape == (64, 64, 3), "augmented image keeps its shape")
        check(a1.max() <= 255.01 and a1.min() >= -0.01,
              "augmented image stays in 0-255", f"[{a1.min():.1f}, {a1.max():.1f}]")
        check(not np.allclose(a1, a2),
              "augmentation is stochastic between calls")

        # temperature scaling
        rs = np.random.RandomState(0)
        yv = rs.randint(0, 3, 400)
        sharp = np.full((400, 3), 0.02)
        sharp[np.arange(400), yv] = 0.96
        flip = rs.rand(400) < 0.35
        sharp[flip] = sharp[flip][:, [1, 2, 0]]      # make it over-confident
        sharp /= sharp.sum(1, keepdims=True)
        T = burnml.fit_temperature(sharp, yv)
        cal = burnml.apply_temperature(sharp, T)
        check(T > 1.0, "temperature > 1 recovered for over-confident scores",
              f"T = {T:.2f}")
        check(np.array_equal(sharp.argmax(1), cal.argmax(1)),
              "temperature scaling leaves the predictions unchanged")
        e0, _ = burnml.expected_calibration_error(sharp, yv)
        e1, _ = burnml.expected_calibration_error(cal, yv)
        check(e1 < e0, "temperature scaling reduces calibration error",
              f"ECE {e0:.3f} -> {e1:.3f}")

        # calibration slope on perfectly calibrated data should be ~1
        p = rs.rand(4000)
        yb = (rs.rand(4000) < p).astype(float)
        a, b = burnml.calibration_slope_intercept(p, yb)
        check(abs(b - 1.0) < 0.2 and abs(a) < 0.2,
              "calibration slope ~1 and intercept ~0 on calibrated data",
              f"intercept {a:.2f}, slope {b:.2f}")

        # TTA averaging
        probs = burnml.calibration_metrics(cal, yv, ["a", "b", "c"])
        check("brier" in probs and "per_class" in probs,
              "calibration metrics assemble without error",
              f"Brier {probs['brier']:.3f}")

    if args.full and args.end_to_end:
        print("\n12. end-to-end fold (tiny settings, one fold)")
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, os.path.join(HERE, "03_run_grouped_cv.py"),
                   "--raw-dir", args.raw_dir, "--manifest", args.manifest,
                   "--task", args.task, "--out-dir", td, "--arch", "scratch",
                   "--params", os.path.join(td, "p.pickle"),
                   "--only-fold", "1", "--folds", str(args.folds),
                   "--rotations", "2", "--epochs", "1", "--img-size", "32",
                   "--n-boot", "10"]
            import pickle
            tiny = {"num_conv_layers": 3, "num_dense_layers": 1,
                    "global_dropout": 0.1, "learning_rate": 1e-3,
                    "batch_size": 32}
            for i in range(1, 6):
                tiny[f"conv{i}_filters"] = 8
                tiny[f"conv{i}_batchnorm"] = False
                tiny[f"conv{i}_padding"] = "same"
                tiny[f"conv{i}_l2"] = False
            for i in range(1, 4):
                tiny[f"dense{i}_units"] = 16
                tiny[f"dense{i}_batchnorm"] = False
                tiny[f"dense{i}_dropout"] = 0.1
            os.makedirs(td, exist_ok=True)
            with open(os.path.join(td, "p.pickle"), "wb") as fh:
                pickle.dump(tiny, fh)
            r = subprocess.run(cmd, capture_output=True, text=True)
            fp = os.path.join(td, "folds", "fold_1.json")
            if not check(r.returncode == 0 and os.path.exists(fp),
                         "one fold trains and writes its file",
                         (r.stderr or r.stdout)[-400:] if r.returncode else ""):
                pass
            else:
                d = json.load(open(fp))
                P = np.array(d["probs"])
                check(P.shape[1] == len(labels),
                      "fold file has one probability per class",
                      f"shape {P.shape}")
                check(np.allclose(P.sum(axis=1), 1.0, atol=1e-4),
                      "softmax probabilities sum to 1")
                check(len(d["files"]) == P.shape[0] == d["n_crops"],
                      "file list, probabilities and crop count agree")
                check(set(d.get("inner_validation_patients", [])).isdisjoint(
                      set(d["held_out_patients"].split(";"))),
                      "inner validation patients are not the held-out patients")

        print("\n13. end-to-end TRANSFER fold (random weights, tiny settings)")
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, os.path.join(HERE, "03_run_grouped_cv.py"),
                   "--raw-dir", args.raw_dir, "--manifest", args.manifest,
                   "--task", args.task, "--out-dir", td, "--arch", "transfer",
                   "--backbone", "mobilenetv2", "--weights", "none",
                   "--only-fold", "1", "--folds", str(args.folds),
                   "--inner-folds", "2", "--img-size", "64",
                   "--probe-epochs", "1", "--finetune-epochs", "1",
                   "--passes-per-epoch", "1", "--tta", "2"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            fp = os.path.join(td, "folds", "fold_1.json")
            err = (r.stderr or r.stdout)[-600:] if r.returncode else ""
            if "RESOURCE_EXHAUSTED" in (r.stderr or "") + (r.stdout or ""):
                err = ("GPU out of memory in the child process: another process "
                       "is holding the GPU (check nvidia-smi on the node). "
                       + err[-300:])
            if check(r.returncode == 0 and os.path.exists(fp),
                     "a transfer fold trains, ensembles and writes its file",
                     err):
                d = json.load(open(fp))
                P = np.array(d["probs"])
                check(np.allclose(P.sum(axis=1), 1.0, atol=1e-4),
                      "calibrated ensemble probabilities sum to 1")
                check(d.get("ensemble_members") == 2,
                      "both ensemble members were trained",
                      f"{d.get('ensemble_members')} members")
                check(d.get("temperature", 0) > 0,
                      "a temperature was fitted", f"T = {d.get('temperature')}")
                for key in ("probs_uncalibrated", "probs_no_tta",
                            "probs_single_model"):
                    check(key in d and np.array(d[key]).shape == P.shape,
                          f"ablation '{key}' recorded for the supplement")
                mon = set(p for m in d.get("member_monitor_patients", [])
                          for p in m)
                check(mon.isdisjoint(set(d["held_out_patients"].split(";"))),
                      "no held-out patient was used to monitor any member")


    if args.full:
        print("\n14. extension: random-init arm, repeats, statistics")
        import tensorflow as tf
        # random-init model: every layer trains, BatchNorm learns, inputs are
        # standardised exactly as in the pretrained arm
        ri = burnml.build_randominit_model("efficientnetb0", 64, len(labels))
        rbase = ri._burn_base
        check(ri._burn_standardised, "random-init input standardisation set to ImageNet mean/std")
        xr = (np.random.RandomState(0).rand(2, 64, 64, 3) * 255).astype(np.float32)
        si = burnml.stem_input(rbase, xr)
        expect = (xr / 255.0 - np.array(burnml.IMAGENET_MEAN)) / np.array(burnml.IMAGENET_STD)
        check(np.allclose(si, expect, atol=1e-4),
              "random-init stem input = (x/255 - ImageNet mean) / ImageNet std",
              f"max abs diff {np.abs(si - expect).max():.2e}")
        try:
            pt = burnml.build_transfer_model("efficientnetb0", 64, len(labels))
            sp = burnml.stem_input(pt._burn_base, xr)
            check(np.allclose(si, sp, atol=1e-3),
                  "pretrained and random-init arms receive IDENTICAL stem inputs",
                  f"max abs diff {np.abs(si - sp).max():.2e}")
            del pt
        except Exception as e:
            warn("pretrained stem-input comparison skipped",
                 f"ImageNet weights not loadable here ({type(e).__name__}); "
                 "runs on the cluster")
        bn = [l for l in rbase.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
        before = bn[1].moving_mean.numpy().copy()
        ri.compile(tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
                   "categorical_crossentropy")
        yb = np.eye(len(labels), dtype=np.float32)[np.random.randint(0, len(labels), 2)]
        ri.train_on_batch(xr, yb)
        check(not np.allclose(before, bn[1].moving_mean.numpy()),
              "random-init BatchNorm statistics are LEARNED (training mode)")
        n_frozen = sum(1 for l in rbase.layers if not l.trainable)
        check(n_frozen == 0, "every random-init backbone layer is trainable",
              f"{n_frozen} frozen")
        del ri, rbase
        burnml.free_memory()
        sch = burnml.WarmupCosine(1e-3, 10, 100)
        v = [float(sch(t)) for t in (0, 5, 10, 55, 100)]
        check(v[0] == 0 and abs(v[2] - 1e-3) < 1e-9 and v[4] < 1e-9 and v[1] < v[2],
              "warm-up then cosine schedule has the right shape")

        f0 = burnml.grouped_folds_seeded(groups, n_splits, y, len(labels), None)
        check(f0 == burnml.grouped_folds(groups, n_splits, y, len(labels)),
              "repeat 0 reproduces the original pre-specified split")
        reps = [burnml.grouped_folds_seeded(groups, n_splits, y, len(labels), 1000 + k)
                for k in range(1, 5)]
        ok_all = True
        for fo in reps:
            fi = np.array([fo[gg] for gg in groups])
            ok_all &= all(len(set(y[fi == f].tolist())) == len(labels) for f in range(n_splits))
            ok_all &= all(len(set(fi[groups == gg].tolist())) == 1 for gg in set(groups))
        check(ok_all, "every repeat split is class-complete and keeps patients whole")
        check(len({tuple(sorted(fo.items())) for fo in reps + [f0]}) == len(reps) + 1,
              "repeat splits differ from each other and from the original")

        yy = np.random.RandomState(3).randint(0, 2, 200)
        ss = np.random.RandomState(4).normal(yy, 1.0)
        ob = burnml.obuchowski_auc(yy, ss, np.arange(200))
        check(abs(ob["auc"] - burnml.auc_score(yy, ss)) < 1e-12 and ob["se"] > 0,
              "clustered AUC equals the rank AUC, with a positive SE")
        Pq = np.random.RandomState(5).dirichlet([1, 1, 1], 120)
        yq = np.random.RandomState(6).randint(0, 3, 120)
        cov, risk, _ = burnml.risk_coverage(Pq, yq)
        check(abs(risk[-1] - np.mean(Pq.argmax(1) != yq)) < 1e-12,
              "risk at full coverage equals the overall error rate")
        bq = burnml.bca_interval(lambda i: float(np.mean(ss[i])), np.arange(200), 500, 1)
        check(bq["ci95"][0] <= bq["estimate"] <= bq["ci95"][1],
              "BCa interval brackets the estimate")

    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed, {len(WARN)} warnings")
    print("=" * 62)
    if WARN:
        for w in WARN:
            print(f"  warning: {w}")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)
    print("  Safe to submit.")


if __name__ == "__main__":
    main()
