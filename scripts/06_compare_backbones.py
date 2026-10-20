#!/usr/bin/env python3
"""
06_compare_backbones.py — compare backbones evaluated on IDENTICAL patient folds.

Because every backbone is scored on the same patient-grouped folds and the same
held-out crops, differences can be compared as paired data: the bootstrap
resamples patients once and recomputes both models on the same resample, which
is far more sensitive than comparing two independent confidence intervals.

The published burn-image backbone comparisons all split at image level, so
their rankings are confounded by leakage. This is the patient-level version.

Reports, per backbone: accuracy, balanced accuracy, macro F1, macro AUC,
Brier, ECE, calibration slope; and, against the pre-specified primary model,
the paired difference in accuracy and macro AUC with a 95% interval.

Usage:
    python 06_compare_backbones.py --results-dir results/classification \
        --manifest results/manifest.csv --task classification
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml
import backbones as B


def load_run(d, labels):
    mp = os.path.join(d, "metrics.json")
    pp = os.path.join(d, "oof_predictions.csv")
    if not (os.path.exists(mp) and os.path.exists(pp)):
        return None
    m = json.load(open(mp))
    rows = list(csv.DictReader(open(pp, newline="")))
    by_file = {r["file"]: r for r in rows}
    return {"metrics": m, "by_file": by_file}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="results/<task>, containing one directory per backbone")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", required=True, choices=list(burnml.TASKS))
    ap.add_argument("--primary", default=B.PRIMARY)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    labels, folder_to_label = burnml.TASKS[args.task]
    idx = {c: i for i, c in enumerate(labels)}
    outdir = args.out_dir or os.path.join(args.results_dir, "comparison")
    os.makedirs(outdir, exist_ok=True)

    man = list(csv.DictReader(open(args.manifest, newline="")))
    files = [r["file"] for r in man]
    y = np.array([idx[folder_to_label[r["class"]]] for r in man])
    groups = np.array([r["patient_group"] for r in man])

    runs = {}
    for name in sorted(os.listdir(args.results_dir)):
        d = os.path.join(args.results_dir, name)
        if not os.path.isdir(d) or name == "comparison":
            continue
        r = load_run(d, labels)
        if r:
            runs[name] = r
    if not runs:
        sys.exit(f"No completed runs found under {args.results_dir}")

    # align every run to the manifest order
    P, M = {}, {}
    for name, r in runs.items():
        missing = [f for f in files if f not in r["by_file"]]
        if missing:
            print(f"  WARNING {name}: {len(missing)} crops missing, skipped")
            continue
        P[name] = np.array([[float(r["by_file"][f][f"p_{c}"]) for c in labels]
                            for f in files])
        M[name] = r["metrics"]

    print("=" * 94)
    print(f"BACKBONE COMPARISON — {args.task}, identical patient folds, "
          f"{len(set(groups))} patients")
    print("=" * 94)
    hdr = (f"  {'backbone':<18}{'acc':>16}{'bal acc':>10}{'macro F1':>10}"
           f"{'macro AUC':>16}{'Brier':>8}{'ECE':>7}")
    print(hdr)
    table = []
    for name in sorted(P, key=lambda n: -M[n]["accuracy"]):
        m = M[name]
        ci = m.get("ci95", {})
        cal = m.get("calibration") or burnml.calibration_metrics(
            P[name], y, labels)
        a = ci.get("accuracy")
        au = ci.get("macro_auc")
        print(f"  {name:<18}"
              f"{m['accuracy']:.3f}" + (f" [{a[0]:.2f},{a[1]:.2f}]" if a else "")
              + f"{m['balanced_accuracy']:>10.3f}{m['macro_f1']:>10.3f}"
              + f"{m['macro_auc']:>8.3f}" + (f" [{au[0]:.2f},{au[1]:.2f}]" if au else "")
              + f"{cal['brier']:>8.3f}{cal['ece']:>7.3f}")
        row = {"backbone": name, "accuracy": m["accuracy"],
               "accuracy_lo": a[0] if a else None,
               "accuracy_hi": a[1] if a else None,
               "balanced_accuracy": m["balanced_accuracy"],
               "macro_f1": m["macro_f1"], "macro_auc": m["macro_auc"],
               "macro_auc_lo": au[0] if au else None,
               "macro_auc_hi": au[1] if au else None,
               "brier": cal["brier"], "ece": cal["ece"],
               "fold_accuracy_sd": m.get("fold_accuracy_sd")}
        for c in labels:
            row[f"sens_{c}"] = m["per_class"][c]["sensitivity"]
            row[f"calslope_{c}"] = cal["per_class"][c]["calibration_slope"]
        table.append(row)

    # ---- paired comparison against the pre-specified primary -------------- #
    ref = args.primary if args.primary in P else sorted(P)[0]
    print(f"\n  paired difference versus the pre-specified primary "
          f"model ({ref}),\n  bootstrap resampling the same patients for both:")
    rng = np.random.RandomState(args.seed)
    uniq = np.array(sorted(set(groups)))
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    draws = rng.choice(len(uniq), size=(args.n_boot, len(uniq)), replace=True)

    paired = []
    for name in sorted(P):
        if name == ref:
            continue
        d_acc, d_auc = [], []
        for row in draws:
            sel = np.concatenate([idx_by_g[uniq[i]] for i in row])
            ya = y[sel]
            a1 = (P[name][sel].argmax(1) == ya).mean()
            a0 = (P[ref][sel].argmax(1) == ya).mean()
            d_acc.append(a1 - a0)
            try:
                u1 = np.nanmean([burnml.auc_score((ya == i).astype(int),
                                                  P[name][sel][:, i])
                                 for i in range(len(labels))])
                u0 = np.nanmean([burnml.auc_score((ya == i).astype(int),
                                                  P[ref][sel][:, i])
                                 for i in range(len(labels))])
                d_auc.append(u1 - u0)
            except Exception:
                pass
        lo, hi = np.percentile(d_acc, [2.5, 97.5])
        point = M[name]["accuracy"] - M[ref]["accuracy"]
        verdict = "overlaps zero" if lo <= 0 <= hi else (
            "favours " + (name if point > 0 else ref))
        line = (f"    {name:<18} accuracy {point:+.3f} "
                f"[{lo:+.3f}, {hi:+.3f}]   {verdict}")
        rec = {"backbone": name, "vs": ref, "d_accuracy": float(point),
               "d_accuracy_lo": float(lo), "d_accuracy_hi": float(hi),
               "verdict": verdict}
        if d_auc:
            l2, h2 = np.percentile(d_auc, [2.5, 97.5])
            p2 = M[name]["macro_auc"] - M[ref]["macro_auc"]
            line += f"\n{'':<24}macro AUC {p2:+.3f} [{l2:+.3f}, {h2:+.3f}]"
            rec.update({"d_macro_auc": float(p2), "d_macro_auc_lo": float(l2),
                        "d_macro_auc_hi": float(h2)})
        print(line)
        paired.append(rec)

    # ---- confirmatory contrast, decomposition, multiplicity -------------- #
    ri = f"{ref}_randominit"

    def delta(a, b, key):
        def fn(i):
            ma = burnml.compute_metrics(y[i], P[a][i].argmax(1), P[a][i], labels)
            mb = burnml.compute_metrics(y[i], P[b][i].argmax(1), P[b][i], labels)
            return ma[key] - mb[key]
        return burnml.bca_interval(fn, groups, args.n_boot, args.seed)

    confirmatory = None
    if ri in P:
        confirmatory = {"contrast": f"{ref} (ImageNet) minus {ri}",
                        "d_macro_auc": delta(ref, ri, "macro_auc"),
                        "d_balanced_accuracy": delta(ref, ri, "balanced_accuracy")}
        c = confirmatory
        print(f"\n  CONFIRMATORY (pre-specified): {c['contrast']}")
        for k in ("d_macro_auc", "d_balanced_accuracy"):
            r = c[k]
            print(f"     {k:<22} {r['estimate']:+.3f}  BCa 95% CI "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
                  f"(percentile [{r['percentile_ci95'][0]:+.3f}, {r['percentile_ci95'][1]:+.3f}])")

    # secondary and post hoc random-init arms: each against the primary
    sensitivity = []
    for name in sorted(P):
        if name.startswith(f"{ref}_randominit_"):
            kind = ("pre-specified secondary (step-matched)" if name.endswith("_steps")
                    else "POST HOC sensitivity (extended learning-rate grid)"
                    if name.endswith("_widelr")
                    else "POST HOC sensitivity (grid bracketed to 3e-1)"
                    if name.endswith("_bracket") else "secondary")
            sensitivity.append({"arm": name, "kind": kind,
                                "d_macro_auc": delta(ref, name, "macro_auc"),
                                "d_balanced_accuracy": delta(ref, name, "balanced_accuracy")})
    if sensitivity:
        print("\n  other random-init arms (pretrained minus arm):")
        for s_ in sensitivity:
            r = s_["d_macro_auc"]
            print(f"     {s_['arm']:<36} macro AUC {r['estimate']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  ({s_['kind']})")

    decomposition = []
    chain = [m for m in ("scratch", ri, ref) if m in P]
    for a, b in zip(chain[1:], chain[:-1]):
        decomposition.append({"step": f"{b} -> {a}",
                              "d_macro_auc": delta(a, b, "macro_auc"),
                              "d_balanced_accuracy": delta(a, b, "balanced_accuracy")})
    if decomposition:
        print("\n  decomposition (each step adds one change):")
        for d_ in decomposition:
            r = d_["d_macro_auc"]
            print(f"     {d_['step']:<44} macro AUC {r['estimate']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")

    # Holm correction over the descriptive pretrained-backbone contrasts
    desc = [r for r in paired if "randominit" not in r["backbone"]
            and r["backbone"] != "scratch"]
    for r in desc:
        d_ = delta(r["backbone"], ref, "macro_auc")
        r["bootstrap_p_macro_auc"] = d_["bootstrap_p_two_sided"]
    order = sorted(desc, key=lambda r: r["bootstrap_p_macro_auc"])
    running = 0.0
    for i, r in enumerate(order):
        adj = min(1.0, (len(order) - i) * r["bootstrap_p_macro_auc"])
        running = max(running, adj)
        r["holm_p_macro_auc"] = running
    if desc:
        print("\n  descriptive backbone contrasts vs primary (Holm-adjusted bootstrap p, macro AUC):")
        for r in desc:
            print(f"     {r['backbone']:<15} p = {r['bootstrap_p_macro_auc']:.3f}, "
                  f"Holm p = {r['holm_p_macro_auc']:.3f}")

    # ---- outputs ---------------------------------------------------------- #
    with open(os.path.join(outdir, "backbone_comparison.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(os.path.join(outdir, "backbone_comparison.json"), "w") as fh:
        json.dump(burnml.jsonable(
            {"task": args.task, "primary": ref, "patients": len(uniq),
             "crops": len(files), "bootstrap_draws": args.n_boot,
             "table": table, "paired_vs_primary": paired,
             "confirmatory": confirmatory, "decomposition": decomposition,
             "other_random_init_arms": sensitivity}),
            fh, indent=2)

    md = ["| Backbone | Accuracy (95% CI) | Balanced acc. | Macro F1 | "
          "Macro AUC (95% CI) | Brier | ECE |",
          "|---|---|---|---|---|---|---|"]
    for r in table:
        acc = f"{r['accuracy']:.3f}"
        if r["accuracy_lo"] is not None:
            acc += f" ({r['accuracy_lo']:.3f}–{r['accuracy_hi']:.3f})"
        auc = f"{r['macro_auc']:.3f}"
        if r["macro_auc_lo"] is not None:
            auc += f" ({r['macro_auc_lo']:.3f}–{r['macro_auc_hi']:.3f})"
        md.append(f"| {r['backbone']} | {acc} | {r['balanced_accuracy']:.3f} | "
                  f"{r['macro_f1']:.3f} | {auc} | {r['brier']:.3f} | "
                  f"{r['ece']:.3f} |")
    if confirmatory:
        md += ["", "**Confirmatory (pre-specified in the written analysis plan):** " + confirmatory["contrast"], ""]
        for k in ("d_macro_auc", "d_balanced_accuracy"):
            r = confirmatory[k]
            name = {"d_macro_auc": "Δ macro AUC", "d_balanced_accuracy": "Δ balanced accuracy"}[k]
            md.append(f"- {name}: {r['estimate']:+.3f} "
                      f"(BCa 95% CI {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})")
    if sensitivity:
        md += ["", "**Other random-initialisation arms** (pretrained minus arm):", ""]
        for s_ in sensitivity:
            r = s_["d_macro_auc"]; b = s_["d_balanced_accuracy"]
            md.append(f"- {s_['arm']} ({s_['kind']}): Δ macro AUC {r['estimate']:+.3f} "
                      f"(BCa {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f}); Δ balanced accuracy "
                      f"{b['estimate']:+.3f} ({b['ci95'][0]:+.3f} to {b['ci95'][1]:+.3f})")
    if decomposition:
        md += ["", "**Decomposition** (each step adds one change):", ""]
        for d_ in decomposition:
            r = d_["d_macro_auc"]
            md.append(f"- {d_['step']}: Δ macro AUC {r['estimate']:+.3f} "
                      f"({r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})")
    if desc:
        md += ["", "Descriptive backbone contrasts (Holm-adjusted bootstrap p, macro AUC): "
               + "; ".join(f"{r['backbone']} p={r['holm_p_macro_auc']:.3f}" for r in desc)]
    with open(os.path.join(outdir, "backbone_comparison.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"\n  Wrote backbone_comparison.{{csv,json,md}} to {outdir}")
    print("\n  Reporting note: report every backbone in the pre-specified set, "
          "and\n  designate the primary model in advance. With 18 patients the "
          "intervals\n  will overlap; say so rather than declaring a winner.")


if __name__ == "__main__":
    main()
