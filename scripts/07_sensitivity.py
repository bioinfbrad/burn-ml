#!/usr/bin/env python3
"""
07_sensitivity.py — sensitivity analyses for patient-clustered results.

Computed from the existing out-of-fold predictions; no retraining.

With 18 patients and very unequal crop counts per patient, crop-level metrics
can be dominated by a single patient. This script makes that visible:

  1. per-patient table: crops, class mix and accuracy for every patient
  2. patient-weighted metrics: each patient counts equally, so one patient
     with 30 crops cannot outweigh six patients with a few crops each
  3. leave-dominant-patient-out: metrics recomputed without the patient who
     contributes the most crops of each class
  4. calibration slope and intercept with 95% CIs from the same
     patient-level bootstrap used for the other metrics

Writes report/sensitivity.md and report/sensitivity.json next to the run.

Usage:
    python 07_sensitivity.py --results-dir results/classification/efficientnetb0
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml


def load(results_dir):
    m = json.load(open(os.path.join(results_dir, "metrics.json")))
    labels = m["design"]["labels"]
    rows = list(csv.DictReader(open(os.path.join(results_dir,
                                                 "oof_predictions.csv"))))
    y = np.array([labels.index(r["true_label"]) for r in rows])
    P = np.array([[float(r[f"p_{c}"]) for c in labels] for r in rows])
    g = np.array([r["patient_group"] for r in rows])
    return m, labels, y, P, g


def patient_weighted(y, pred, g, labels):
    """Mean over patients of per-patient sensitivity, per class, using only
    patients that have crops of that class."""
    out = {}
    for i, c in enumerate(labels):
        per = []
        for p in sorted(set(g)):
            sel = (g == p) & (y == i)
            if sel.any():
                per.append(float((pred[sel] == i).mean()))
        out[c] = {"patients": len(per),
                  "mean_sensitivity": float(np.mean(per)) if per else float("nan"),
                  "median_sensitivity": float(np.median(per)) if per else float("nan"),
                  "min": float(np.min(per)) if per else float("nan"),
                  "max": float(np.max(per)) if per else float("nan")}
    acc = [float((pred[g == p] == y[g == p]).mean()) for p in sorted(set(g))]
    out["accuracy"] = {"patients": len(acc), "mean": float(np.mean(acc)),
                       "median": float(np.median(acc)),
                       "min": float(np.min(acc)), "max": float(np.max(acc))}
    return out


def calibration_ci(y, P, g, labels, n_boot, seed):
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(g)))
    idx = {p: np.where(g == p)[0] for p in uniq}
    draws = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[p] for p in pick])
        for i, c in enumerate(labels):
            yb = (y[sel] == i).astype(float)
            if 0 < yb.sum() < len(yb):
                a, b = burnml.calibration_slope_intercept(P[sel, i], yb)
                if a == a and b == b:
                    draws[(c, "slope")].append(b)
                    draws[(c, "intercept")].append(a)
    out = {}
    for i, c in enumerate(labels):
        a, b = burnml.calibration_slope_intercept(P[:, i], (y == i).astype(float))
        rec = {"slope": b, "intercept": a}
        for k in ("slope", "intercept"):
            v = draws.get((c, k))
            if v:
                rec[f"{k}_ci95"] = [float(x) for x in np.percentile(v, [2.5, 97.5])]
        out[c] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m, labels, y, P, g = load(args.results_dir)
    pred = P.argmax(1)
    outdir = os.path.join(args.results_dir, "report")
    os.makedirs(outdir, exist_ok=True)
    res = {"labels": labels, "patients": len(set(g)), "crops": len(y)}

    # 1. per-patient table
    table = []
    for p in sorted(set(g), key=lambda p: -(g == p).sum()):
        sel = g == p
        mix = {labels[i]: int(((y == i) & sel).sum()) for i in range(len(labels))}
        table.append({"patient": p, "crops": int(sel.sum()), "class_mix": mix,
                      "correct": int((pred[sel] == y[sel]).sum()),
                      "accuracy": float((pred[sel] == y[sel]).mean())})
    res["per_patient"] = table

    # 2. patient-weighted
    res["patient_weighted"] = patient_weighted(y, pred, g, labels)

    # 3. leave the dominant patient of each class out
    loo = {}
    for i, c in enumerate(labels):
        counts = Counter(g[y == i])
        if not counts:
            continue
        dom, n_dom = counts.most_common(1)[0]
        keep = g != dom
        mk = burnml.compute_metrics(y[keep], pred[keep], P[keep], labels)
        loo[c] = {"excluded_patient": dom, "excluded_crops_of_class": int(n_dom),
                  "share_of_class": float(n_dom / (y == i).sum()),
                  "sensitivity_all": float(((pred == i) & (y == i)).sum() / (y == i).sum()),
                  "sensitivity_without": mk["per_class"][c]["sensitivity"],
                  "accuracy_without": mk["accuracy"],
                  "balanced_accuracy_without": mk["balanced_accuracy"],
                  "macro_auc_without": mk["macro_auc"]}
    res["leave_dominant_patient_out"] = loo

    # 4. calibration slope / intercept with patient-bootstrap CIs
    res["calibration_ci"] = calibration_ci(y, P, g, labels, args.n_boot, args.seed)

    with open(os.path.join(outdir, "sensitivity.json"), "w") as fh:
        json.dump(burnml.jsonable(res), fh, indent=2)

    L = [f"# Sensitivity analyses: {m['design'].get('task')} / "
         f"{m['design'].get('backbone', m['design'].get('architecture'))}", "",
         f"{len(y)} crops from {len(set(g))} patients. Computed from the "
         "out-of-fold predictions; no retraining.", "",
         "## Patient-weighted sensitivity (each patient counts equally)", "",
         "| Class | Crop-level sensitivity | Patients | Patient-weighted mean | Median | Range |",
         "|---|---|---|---|---|---|"]
    for i, c in enumerate(labels):
        pw = res["patient_weighted"][c]
        crop = m["per_class"][c]["sensitivity"]
        L.append(f"| {c.replace('_', ' ')} | {crop:.3f} | {pw['patients']} | "
                 f"{pw['mean_sensitivity']:.3f} | {pw['median_sensitivity']:.3f} | "
                 f"{pw['min']:.2f}–{pw['max']:.2f} |")
    pa = res["patient_weighted"]["accuracy"]
    L += ["", f"Per-patient accuracy: mean {pa['mean']:.3f}, median "
          f"{pa['median']:.3f}, range {pa['min']:.2f}–{pa['max']:.2f} "
          f"({pa['patients']} patients).", "",
          "## Excluding the patient with the most crops of each class", "",
          "| Class | Excluded patient | Its share of the class | Sensitivity, all | Sensitivity, without |",
          "|---|---|---|---|---|"]
    for c, r in loo.items():
        L.append(f"| {c.replace('_', ' ')} | {r['excluded_patient']} | "
                 f"{r['excluded_crops_of_class']} crops ({100*r['share_of_class']:.0f}%) | "
                 f"{r['sensitivity_all']:.3f} | {r['sensitivity_without']:.3f} |")
    L += ["", "## Calibration slope and intercept (95% CI, patient bootstrap)", "",
          "Ideal: slope 1, intercept 0. Slope > 1 means under-confident; "
          "slope < 1 over-confident. A positive intercept means the class is "
          "under-predicted on average.", "",
          "| Class | Slope (95% CI) | Intercept (95% CI) |", "|---|---|---|"]
    for c in labels:
        r = res["calibration_ci"][c]
        s = r.get("slope_ci95", [float("nan")] * 2)
        it = r.get("intercept_ci95", [float("nan")] * 2)
        L.append(f"| {c.replace('_', ' ')} | {r['slope']:.2f} ({s[0]:.2f}–{s[1]:.2f}) | "
                 f"{r['intercept']:+.2f} ({it[0]:+.2f} to {it[1]:+.2f}) |")
    L += ["", "## Per-patient results", "",
          "| Patient | Crops | " + " | ".join(c.replace('_', ' ') for c in labels)
          + " | Correct | Accuracy |",
          "|---|---|" + "---|" * len(labels) + "---|---|"]
    for r in table:
        L.append(f"| {r['patient']} | {r['crops']} | "
                 + " | ".join(str(r["class_mix"][c]) for c in labels)
                 + f" | {r['correct']} | {r['accuracy']:.2f} |")
    # forest plot: per-patient sensitivity with Wilson intervals, per class
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def wilson(k, n, z=1.96):
        if n == 0:
            return (float("nan"),) * 3
        ph = k / n
        d = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / d
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
        return ph, max(0.0, c - h), min(1.0, c + h)

    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 5), squeeze=False)
    for i, c in enumerate(labels):
        ax = axes[0][i]
        pts = []
        for p in sorted(set(g)):
            sel = (g == p) & (y == i)
            if sel.any():
                k, n = int((pred[sel] == i).sum()), int(sel.sum())
                pts.append((p, n, *wilson(k, n)))
        pts.sort(key=lambda t: -t[1])
        for j, (p, n, ph, lo, hi) in enumerate(pts):
            ax.plot([lo, hi], [j, j], color="0.4", lw=1)
            ax.scatter([ph], [j], s=12 + 3 * n, color="C0", zorder=3)
        yl = [f"{p} (n={n})" for p, n, *_ in pts]
        crop = m["per_class"][c]["sensitivity"]
        pw = res["patient_weighted"][c]["mean_sensitivity"]
        ax.scatter([crop], [len(pts) + 0.6], marker="D", s=60, color="C3", label="crop-pooled")
        ax.scatter([pw], [len(pts) + 1.4], marker="D", s=60, color="C2", label="patient-weighted")
        ax.set_yticks(list(range(len(pts))) + [len(pts) + 0.6, len(pts) + 1.4])
        ax.set_yticklabels(yl + ["crop-pooled", "patient-weighted"], fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.invert_yaxis()
        ax.set_xlabel("Sensitivity (Wilson 95% CI)")
        ax.set_title(c.replace("_", " "), fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Per-patient sensitivity (marker size = number of crops)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_forest_per_patient.png"), dpi=300)
    plt.close(fig)

    with open(os.path.join(outdir, "sensitivity.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:30]))
    print(f"\nWrote sensitivity.md and sensitivity.json to {outdir}")


if __name__ == "__main__":
    main()
