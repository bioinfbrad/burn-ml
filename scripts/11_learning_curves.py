#!/usr/bin/env python3
"""
11_learning_curves.py — performance versus number of training patients.

Reads results/<task>/learning_curve/<model>/n<K>_s<seed>/ and the full-data
run results/<task>/<model>/, and reports mean and range across subsamples.
Descriptive only: with 18 patients no power law is fitted or extrapolated.
Records how often the dominant full-thickness patient was in the training
subsample, because that patient strongly influences full-thickness results.

Usage:
    python 11_learning_curves.py --task-dir results/classification --model efficientnetb0
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--model", default="efficientnetb0")
    ap.add_argument("--dominant-patient", default=None,
                    help="patient group to track; default: the patient with the most "
                         "full-thickness crops in <task-dir>/../manifest.csv")
    args = ap.parse_args()
    if args.dominant_patient is None:
        import csv
        from collections import Counter
        man = os.path.join(os.path.dirname(os.path.abspath(args.task_dir.rstrip("/"))), "manifest.csv")
        rows = list(csv.DictReader(open(man)))
        ft = Counter(r["patient_group"] for r in rows if r["class"] == "Full_thickness_burn")
        args.dominant_patient = ft.most_common(1)[0][0]

    pts = defaultdict(list)
    dom = defaultdict(list)
    for d in glob.glob(os.path.join(args.task_dir, "learning_curve", args.model, "n*_s*")):
        k = int(re.match(r"n(\d+)_s", os.path.basename(d)).group(1))
        mp = os.path.join(d, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        pts[k].append(m)
        folds = glob.glob(os.path.join(d, "folds", "fold_*.json"))
        dom[k].append(float(np.mean([args.dominant_patient in json.load(open(f)).get("training_patients", [])
                                     for f in folds])) if folds else float("nan"))
    full = os.path.join(args.task_dir, args.model, "metrics.json")
    out = {"model": args.model, "points": {}}
    rows = []
    for k in sorted(pts):
        ms = pts[k]
        rec = {key: {"mean": float(np.mean([m[key] for m in ms])),
                     "min": float(np.min([m[key] for m in ms])),
                     "max": float(np.max([m[key] for m in ms]))}
               for key in ("accuracy", "balanced_accuracy", "macro_auc")}
        rec["subsamples"] = len(ms)
        rec["dominant_patient_in_training"] = float(np.nanmean(dom[k]))
        out["points"][k] = rec
        rows.append((k, rec))
    if os.path.exists(full):
        m = json.load(open(full))
        out["full"] = {key: m[key] for key in ("accuracy", "balanced_accuracy", "macro_auc")}
    json.dump(burnml.jsonable(out), open(os.path.join(args.task_dir, f"learning_curve_{args.model}.json"), "w"), indent=2)

    L = [f"# Learning curve: {args.model}", "",
         "| Training patients per fold | Subsamples | Accuracy mean [range] | Balanced acc. mean | Macro AUC mean [range] | Dominant FT patient in training |",
         "|---|---|---|---|---|---|"]
    for k, r in rows:
        L.append(f"| {k} | {r['subsamples']} | {r['accuracy']['mean']:.3f} [{r['accuracy']['min']:.3f}–{r['accuracy']['max']:.3f}] | "
                 f"{r['balanced_accuracy']['mean']:.3f} | {r['macro_auc']['mean']:.3f} [{r['macro_auc']['min']:.3f}–{r['macro_auc']['max']:.3f}] | "
                 f"{100*r['dominant_patient_in_training']:.0f}% of folds |")
    if "full" in out:
        f = out["full"]
        L.append(f"| all (~15) | 1 | {f['accuracy']:.3f} | {f['balanced_accuracy']:.3f} | {f['macro_auc']:.3f} | — |")
    L += ["", "Descriptive only; no extrapolation is attempted with 18 patients."]
    open(os.path.join(args.task_dir, f"learning_curve_{args.model}.md"), "w").write("\n".join(L) + "\n")

    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for key, lab in (("macro_auc", "macro AUC"), ("balanced_accuracy", "balanced accuracy")):
            xs = [k for k, _ in rows]
            ax.errorbar(xs, [r[key]["mean"] for _, r in rows],
                        yerr=[[r[key]["mean"] - r[key]["min"] for _, r in rows],
                              [r[key]["max"] - r[key]["mean"] for _, r in rows]],
                        marker="o", capsize=3, label=lab)
            if "full" in out:
                ax.scatter([15], [out["full"][key]], marker="*", s=120)
        ax.set_xlabel("Training patients per outer fold")
        ax.set_ylabel("Score (mean, range across subsamples)")
        ax.set_title(f"Learning curve: {args.model}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.task_dir, f"learning_curve_{args.model}.png"), dpi=300)
    print("\n".join(L))


if __name__ == "__main__":
    main()
