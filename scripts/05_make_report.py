#!/usr/bin/env python3
"""
05_make_report.py — turn the cross-validation output into manuscript-ready
tables and figures.

Produces, in --out-dir/report:
    table1_dataset.csv / .md          dataset composition by class and patient
    table2_performance.csv / .md      per-class metrics with 95% CIs
    fig_confusion_matrix.png
    fig_roc.png                       one-vs-rest ROC from out-of-fold scores
    fig_training_curves.png           training and validation loss per fold
    fig_fold_accuracy.png
    numbers_for_text.txt              the figures to quote in the Results

Usage:
    python 05_make_report.py --results-dir results/classification \
        --manifest results/manifest.csv
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(v, ci=None, dp=3):
    if v != v:
        return "n/a"
    s = f"{v:.{dp}f}"
    if ci:
        s += f" ({ci[0]:.{dp}f}–{ci[1]:.{dp}f})"
    return s


def table1(manifest, outdir):
    rows = list(csv.DictReader(open(manifest, newline="")))
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per[r["patient_group"]][r["class"]] += 1
    cls = burnml.CLASS_DIRS
    body = []
    for g in sorted(per):
        c = per[g]
        body.append([g, sum(c.values())] + [c.get(k, 0) for k in cls])
    tot = [sum(r[1] for r in body)] + [sum(r[2 + i] for r in body)
                                       for i in range(len(cls))]
    body.append(["Total"] + tot)
    header = ["Patient", "Crops"] + [c.replace("_", " ") for c in cls]
    with open(os.path.join(outdir, "table1_dataset.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    with open(os.path.join(outdir, "table1_dataset.md"), "w") as fh:
        fh.write(md_table(header, body) + "\n")
    return len(per), tot


def table2(m, outdir):
    labels = m["design"]["labels"]
    ci = m.get("ci95", {})
    header = ["Class", "n", "TP", "FP", "FN", "TN", "Sensitivity (95% CI)",
              "Specificity (95% CI)", "Precision (95% CI)", "F1 (95% CI)",
              "AUC (95% CI)"]
    body = []
    for c in labels:
        p = m["per_class"][c]
        body.append([c.replace("_", " "), p["TP"] + p["FN"], p["TP"], p["FP"],
                     p["FN"], p["TN"],
                     fmt(p["sensitivity"], ci.get(f"{c}.sensitivity")),
                     fmt(p["specificity"], ci.get(f"{c}.specificity")),
                     fmt(p["precision"], ci.get(f"{c}.precision")),
                     fmt(p["f1"], ci.get(f"{c}.f1")),
                     fmt(p["auc"], ci.get(f"{c}.auc"))])
    for key, name in [("accuracy", "Overall accuracy"),
                      ("balanced_accuracy", "Balanced accuracy"),
                      ("macro_f1", "Macro F1"), ("macro_auc", "Macro AUC")]:
        body.append([name, "", "", "", "", "", fmt(m[key], ci.get(key)),
                     "", "", "", ""])
    with open(os.path.join(outdir, "table2_performance.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    with open(os.path.join(outdir, "table2_performance.md"), "w") as fh:
        fh.write(md_table(header, body) + "\n")


def fig_confusion(m, outdir):
    labels = m["design"]["labels"]
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(labels),
                                    1.4 + 1.3 * len(labels)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [c.replace("_", " ").replace(" thickness", "-thickness")
             for c in labels]
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticklabels(short)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=12)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Out-of-fold confusion matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_confusion_matrix.png"), dpi=300)
    plt.close(fig)


def fig_roc(m, results_dir, outdir):
    labels = m["design"]["labels"]
    path = os.path.join(results_dir, "oof_predictions.csv")
    rows = list(csv.DictReader(open(path, newline="")))
    y = np.array([labels.index(r["true_label"]) for r in rows])
    P = np.array([[float(r[f"p_{c}"]) for c in labels] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 5.4))
    for i, c in enumerate(labels):
        pos = (y == i).astype(int)
        s = P[:, i]
        order = np.argsort(-s)
        tp = np.cumsum(pos[order])
        fp = np.cumsum(1 - pos[order])
        tpr = np.concatenate([[0], tp / max(tp[-1], 1)])
        fpr = np.concatenate([[0], fp / max(fp[-1], 1)])
        a = burnml.auc_score(pos, s)
        ax.plot(fpr, tpr, lw=2,
                label=f"{c.replace('_', ' ')} (AUC = {a:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC, one class versus the rest")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_roc.png"), dpi=300)
    plt.close(fig)


def fig_curves(results_dir, outdir):
    path = os.path.join(results_dir, "training_history.json")
    if not os.path.exists(path):
        return
    hist = json.load(open(path))
    if not hist:
        return
    folds = sorted(hist, key=lambda x: int(x))
    cols = min(3, len(folds))
    rows = int(np.ceil(len(folds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.3 * rows),
                             squeeze=False)
    for k, fold in enumerate(folds):
        ax = axes[k // cols][k % cols]
        h = hist[fold]
        members = ({"": h} if "loss" in h or "probe_loss" in h else h)
        for j, (mname, mh) in enumerate(sorted(members.items())):
            if "probe_loss" in mh:                  # transfer: two phases
                tl = mh.get("probe_loss", []) + mh.get("ft_loss", [])
                vl = mh.get("probe_val_loss", []) + mh.get("ft_val_loss", [])
                split = len(mh.get("probe_loss", []))
            else:                                   # scratch CNN
                tl, vl, split = mh.get("loss", []), mh.get("val_loss", []), None
            lab = f"m{mname} " if mname else ""
            ax.plot(tl, lw=1.2, color=f"C{j}", label=f"{lab}train")
            ax.plot(vl, lw=1.2, ls="--", color=f"C{j}", label=f"{lab}monitor")
            if split:
                ax.axvline(split - 0.5, color="grey", lw=0.8, ls=":")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=6, ncol=2)
    for k in range(len(folds), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Training and monitoring loss (dotted line: probe -> "
                 "fine-tune)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_training_curves.png"), dpi=300)
    plt.close(fig)


def fig_calibration(m, results_dir, outdir):
    """Reliability curves per class, one versus the rest. TRIPOD+AI expects
    calibration plots alongside discrimination."""
    labels = m["design"]["labels"]
    path = os.path.join(results_dir, "oof_predictions.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path, newline="")))
    y = np.array([labels.index(r["true_label"]) for r in rows])
    P = np.array([[float(r[f"p_{c}"]) for c in labels] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for i, c in enumerate(labels):
        p = P[:, i]
        obs = (y == i).astype(float)
        order = np.argsort(p)
        bins = [b for b in np.array_split(order, 8) if len(b)]
        ax.plot([p[b].mean() for b in bins], [obs[b].mean() for b in bins],
                "o-", lw=1.8, ms=5, label=c.replace("_", " "))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration, one class versus the rest")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_calibration.png"), dpi=300)
    plt.close(fig)


def table3_ablation(m, outdir):
    ab = m.get("ablation")
    if not ab:
        return
    header = ["Variant", "Accuracy", "Balanced acc.", "Macro F1", "Macro AUC",
              "Brier", "ECE", "Wrong at >0.9 conf."]
    body = [[r["variant"], f"{r['accuracy']:.3f}",
             f"{r['balanced_accuracy']:.3f}", f"{r['macro_f1']:.3f}",
             f"{r['macro_auc']:.3f}", f"{r['brier']:.3f}", f"{r['ece']:.3f}",
             ("n/a" if r["high_confidence_wrong_rate"] !=
              r["high_confidence_wrong_rate"]
              else f"{100*r['high_confidence_wrong_rate']:.0f}%")]
            for r in ab]
    with open(os.path.join(outdir, "table3_ablation.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    with open(os.path.join(outdir, "table3_ablation.md"), "w") as fh:
        fh.write(md_table(header, body) + "\n")


def fig_folds(m, outdir):
    rows = m.get("per_fold", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(1.3 + 0.9 * len(rows), 3.6))
    x = [r["fold"] for r in rows]
    ax.bar(x, [r["accuracy"] for r in rows], color="#4878A8")
    ax.axhline(m["fold_accuracy_mean"], color="crimson", ls="--", lw=1.2,
               label=f"mean {m['fold_accuracy_mean']:.2f}")
    for r in rows:
        ax.text(r["fold"], r["accuracy"] + 0.02,
                f"{r['n_patients']}p/{r['n_crops']}c", ha="center", fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xlabel("Fold (held-out patients)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on held-out patients by fold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_fold_accuracy.png"), dpi=300)
    plt.close(fig)


def numbers_for_text(m, n_patients, totals, outdir):
    labels = m["design"]["labels"]
    ci = m.get("ci95", {})
    d = defaultdict(lambda: "not recorded", m["design"])
    L = []
    L.append("NUMBERS FOR THE RESULTS SECTION")
    L.append("=" * 60)
    L.append(f"Dataset: {d['crops']} cropped images from {n_patients} patients")
    L.append(f"Design: {d['outer_folds']}-fold patient-grouped cross-validation; "
             f"every patient held out exactly once")
    if d.get("architecture") == "transfer":
        if d.get("weights") == "imagenet":
            L.append(f"Model: {d.get('backbone_keras_name')} pretrained on "
                     f"ImageNet, input {d.get('image_size')}x"
                     f"{d.get('image_size')} px")
        else:
            L.append(f"Model: {d.get('backbone_keras_name')} RANDOMLY "
                     "INITIALISED (weights=none). TEST RUN ONLY: these numbers "
                     "must not be reported.")
        L.append(f"Training: {d.get('strategy')}; {d.get('probe_epochs')} probe "
                 f"epochs at lr {d.get('probe_lr')}, then "
                 f"{d.get('finetune_epochs')} fine-tuning epochs at lr "
                 f"{d.get('finetune_lr')} ({d.get('finetune_schedule')}), last "
                 f"{d.get('unfreeze_layers')} backbone layers unfrozen; "
                 f"{d.get('passes_per_epoch')} augmented passes per epoch; "
                 f"batch {d.get('batch_size')}; label smoothing "
                 f"{d.get('label_smoothing')}; {d.get('optimiser')}")
        L.append(f"Augmentation: {d.get('augmentation')}; applied after the "
                 f"patient split, training data only")
        L.append(f"Inference: {d.get('ensemble_members')}-member "
                 f"{d.get('ensemble_type')}; {d.get('tta_transforms')}-transform "
                 f"test-time augmentation; temperature scaling "
                 f"{'on' if d.get('temperature_scaling') else 'off'} (fitted on "
                 f"inner out-of-fold predictions)")
        L.append("Hyperparameters: pre-specified recipe, no search on these "
                 "patients; no class weighting; no early stopping")
    else:
        L.append(f"Augmentation: {d['rotations_per_training_image']} rotations per "
                 f"training image, applied after splitting, training data only")
        L.append(f"Hyperparameter selection: {d['hyperparameter_selection']}")
    L.append(f"Seed {d['seed']}; {d['bootstrap_draws']} patient-level bootstrap "
             f"draws for 95% CIs")
    L.append("")
    for key, name in [("accuracy", "Overall accuracy"),
                      ("balanced_accuracy", "Balanced accuracy"),
                      ("macro_f1", "Macro F1"), ("macro_auc", "Macro AUC")]:
        L.append(f"  {name}: {fmt(m[key], ci.get(key))}")
    L.append(f"  Per-fold accuracy: mean {m['fold_accuracy_mean']:.3f}, "
             f"SD {m['fold_accuracy_sd']:.3f}")
    L.append("")
    for c in labels:
        p = m["per_class"][c]
        L.append(f"  {c.replace('_', ' ')} (n = {p['TP'] + p['FN']})")
        L.append(f"    sensitivity {fmt(p['sensitivity'], ci.get(f'{c}.sensitivity'))}")
        L.append(f"    specificity {fmt(p['specificity'], ci.get(f'{c}.specificity'))}")
        L.append(f"    precision   {fmt(p['precision'], ci.get(f'{c}.precision'))}")
        L.append(f"    F1          {fmt(p['f1'], ci.get(f'{c}.f1'))}")
        L.append(f"    AUC         {fmt(p['auc'], ci.get(f'{c}.auc'))}")
    L.append("")
    L.append("Per-fold detail")
    for r in m.get("per_fold", []):
        L.append(f"  fold {r['fold']}: patients {r['held_out_patients']} | "
                 f"{r['n_crops']} crops | accuracy {r['accuracy']:.3f} | "
                 f"{r['epochs_run']} epochs")
    ab = m.get("ablation")
    if ab:
        L.append("")
        L.append("Ablation (supplementary table 3)")
        for r in ab:
            L.append(f"  {r['variant']}: accuracy {r['accuracy']:.3f}, "
                     f"macro AUC {r['macro_auc']:.3f}, Brier {r['brier']:.3f}, "
                     f"ECE {r['ece']:.3f}")
        if m.get("temperatures"):
            L.append("  fitted temperatures by fold: "
                     + ", ".join(f"{t:.2f}" for t in m["temperatures"]))
    cal = m.get("calibration")
    if cal:
        L.append("")
        L.append("Calibration (TRIPOD+AI expects this alongside discrimination)")
        L.append(f"  Brier score: {cal['brier']:.3f}")
        L.append(f"  Expected calibration error: {cal['ece']:.3f} "
                 f"(max {cal['mce']:.3f})")
        rate = cal["high_confidence_wrong_rate"]
        L.append(f"  Predictions made with >0.90 confidence: "
                 f"{cal['high_confidence_n']}, of which "
                 f"{cal['high_confidence_wrong']} wrong "
                 + ("(n/a)" if rate != rate else f"({100*rate:.0f}%)"))
        for c in labels:
            pc = cal["per_class"][c]
            L.append(f"  {c.replace('_', ' ')}: calibration intercept "
                     f"{pc['calibration_intercept']:+.3f}, slope "
                     f"{pc['calibration_slope']:.3f} "
                     f"(ideal 0 and 1; slope < 1 means over-confident)")
    L.append("")
    L.append("Reminder: quote confidence intervals alongside every point "
             "estimate, and do not describe any of these as clinical accuracy.")
    txt = "\n".join(L)
    with open(os.path.join(outdir, "numbers_for_text.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    outdir = args.out_dir or os.path.join(args.results_dir, "report")
    os.makedirs(outdir, exist_ok=True)
    m = json.load(open(os.path.join(args.results_dir, "metrics.json")))

    n_patients, totals = table1(args.manifest, outdir)
    table2(m, outdir)
    fig_confusion(m, outdir)
    fig_roc(m, args.results_dir, outdir)
    fig_curves(args.results_dir, outdir)
    fig_folds(m, outdir)
    fig_calibration(m, args.results_dir, outdir)
    table3_ablation(m, outdir)
    numbers_for_text(m, n_patients, totals, outdir)
    print(f"\nWrote tables and figures to {outdir}")


if __name__ == "__main__":
    main()
