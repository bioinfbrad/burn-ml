#!/usr/bin/env python3
"""
09_repeats.py — summarise repeated patient-grouped cross-validation.

Repeat 0 is the original pre-specified split (results/<task>/<model>);
repeats k>0 live in results/<task>/repeats/r<k>/<model>. For each model this
reports the mean, SD and range of every headline metric across repeats, and,
within each repeat, the paired difference between the pretrained and the
random-initialisation arm (same folds, patient-bootstrap CI).

The patient bootstrap within a repeat captures patient-sampling variance; the
spread ACROSS repeats captures the variance due to the fold split and training
randomness. Both are reported because neither alone is complete.

Usage:
    python 09_repeats.py --task-dir results/classification \
        --primary efficientnetb0 --comparator efficientnetb0_randominit
"""
import argparse, csv, json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml

KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]


def load(d):
    if not os.path.exists(os.path.join(d, "metrics.json")):
        return None
    m = json.load(open(os.path.join(d, "metrics.json")))
    rows = list(csv.DictReader(open(os.path.join(d, "oof_predictions.csv"))))
    L = m["design"]["labels"]
    by = {r["file"]: r for r in rows}
    return m, L, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--primary", default="efficientnetb0")
    ap.add_argument("--comparator", default="efficientnetb0_randominit")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repeat_dirs = {0: args.task_dir}
    for d in sorted(glob.glob(os.path.join(args.task_dir, "repeats", "r*"))):
        repeat_dirs[int(os.path.basename(d)[1:])] = d
    models = sorted({os.path.basename(x) for d in repeat_dirs.values()
                     for x in glob.glob(os.path.join(d, "*"))
                     if os.path.exists(os.path.join(x, "metrics.json"))})
    out = {"repeats": sorted(repeat_dirs), "models": {}, "paired": []}
    L = ["# Repeated patient-grouped cross-validation", ""]
    L += ["| Model | Repeats | " + " | ".join(f"{k.replace('_',' ')} mean (SD) [range]" for k in KEYS) + " |",
          "|---|---|" + "---|" * len(KEYS)]
    for mdl in models:
        vals = {k: [] for k in KEYS}
        used = []
        for r, d in repeat_dirs.items():
            x = load(os.path.join(d, mdl))
            if x:
                used.append(r)
                for k in KEYS:
                    vals[k].append(x[0][k])
        if not used:
            continue
        summ = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    "min": float(np.min(v)), "max": float(np.max(v)), "values": v}
                for k, v in vals.items()}
        out["models"][mdl] = {"repeats": used, **summ}
        L.append(f"| {mdl} | {len(used)} | " + " | ".join(
            f"{summ[k]['mean']:.3f} ({summ[k]['sd']:.3f}) [{summ[k]['min']:.3f}–{summ[k]['max']:.3f}]"
            for k in KEYS) + " |")

    L += ["", f"## Paired difference within each repeat: {args.primary} minus {args.comparator}", "",
          "| Repeat | Δ macro AUC (95% CI) | Δ balanced accuracy (95% CI) |", "|---|---|---|"]
    for r, d in repeat_dirs.items():
        a, b = load(os.path.join(d, args.primary)), load(os.path.join(d, args.comparator))
        if not (a and b):
            continue
        labels = a[1]
        files = sorted(set(a[2]) & set(b[2]))
        y = np.array([labels.index(a[2][f]["true_label"]) for f in files])
        g = np.array([a[2][f]["patient_group"] for f in files])
        PA = np.array([[float(a[2][f][f"p_{c}"]) for c in labels] for f in files])
        PB = np.array([[float(b[2][f][f"p_{c}"]) for c in labels] for f in files])
        def d_auc(i):
            ma = burnml.compute_metrics(y[i], PA[i].argmax(1), PA[i], labels)
            mb = burnml.compute_metrics(y[i], PB[i].argmax(1), PB[i], labels)
            return ma["macro_auc"] - mb["macro_auc"]
        def d_bal(i):
            ma = burnml.compute_metrics(y[i], PA[i].argmax(1), PA[i], labels)
            mb = burnml.compute_metrics(y[i], PB[i].argmax(1), PB[i], labels)
            return ma["balanced_accuracy"] - mb["balanced_accuracy"]
        ra = burnml.bca_interval(d_auc, g, args.n_boot, args.seed)
        rb = burnml.bca_interval(d_bal, g, args.n_boot, args.seed)
        out["paired"].append({"repeat": r, "d_macro_auc": ra, "d_balanced_accuracy": rb})
        L.append(f"| {r} | {ra['estimate']:+.3f} ({ra['ci95'][0]:+.3f} to {ra['ci95'][1]:+.3f}) | "
                 f"{rb['estimate']:+.3f} ({rb['ci95'][0]:+.3f} to {rb['ci95'][1]:+.3f}) |")
    if out["paired"]:
        da = [p["d_macro_auc"]["estimate"] for p in out["paired"]]
        L += ["", f"Across {len(da)} repeats: Δ macro AUC mean {np.mean(da):+.3f}, "
              f"range {min(da):+.3f} to {max(da):+.3f}. CIs are BCa from the patient bootstrap."]
    json.dump(burnml.jsonable(out), open(os.path.join(args.task_dir, "repeats_summary.json"), "w"), indent=2)
    open(os.path.join(args.task_dir, "repeats_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
