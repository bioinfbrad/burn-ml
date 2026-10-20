#!/usr/bin/env python3
"""
08_posthoc.py — analyses computed from existing out-of-fold predictions.

  1. selective prediction: risk-coverage curve, AURC, and accuracy, macro F1
     and per-class sensitivity at 100/90/80/70% coverage (least confident
     cases deferred to a clinician), crop-level and patient-weighted, with
     patient-bootstrap CIs
  2. aggregation: photograph x reference-class and patient level ("any
     full-thickness detected"), using mean calibrated probability
  3. clustered AUC (Obuchowski 1997) per class, as an analytical check on
     the patient-bootstrap intervals

Usage:
    python 08_posthoc.py --results-dir results/classification/efficientnetb0 \
        --manifest results/manifest.csv
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml

COVERAGES = [1.0, 0.9, 0.8, 0.7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m = json.load(open(os.path.join(args.results_dir, "metrics.json")))
    labels = m["design"]["labels"]
    rows = list(csv.DictReader(open(os.path.join(args.results_dir, "oof_predictions.csv"))))
    man = {r["file"]: r for r in csv.DictReader(open(args.manifest))}
    y = np.array([labels.index(r["true_label"]) for r in rows])
    P = np.array([[float(r[f"p_{c}"]) for c in labels] for r in rows])
    g = np.array([r["patient_group"] for r in rows])
    photo = np.array([man[r["file"]]["source_id"] for r in rows])
    out = {"labels": labels, "crops": len(y), "patients": len(set(g))}
    rng = np.random.RandomState(args.seed)
    uniq = np.array(sorted(set(g)))
    idx = {p: np.where(g == p)[0] for p in uniq}

    # 1. selective prediction ------------------------------------------------
    cov, risk, aurc = burnml.risk_coverage(P, y)
    out["aurc"] = aurc
    out["risk_coverage_curve"] = {"coverage": cov[::max(1, len(cov)//100)].tolist(),
                                  "risk": risk[::max(1, len(cov)//100)].tolist()}

    def at_cov(sel_idx, c):
        Ps, ys, gs = P[sel_idx], y[sel_idx], g[sel_idx]
        keep = burnml.keep_at_coverage(Ps, c)
        mm = burnml.compute_metrics(ys[keep], Ps[keep].argmax(1), Ps[keep], labels)
        r = {"accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"]}
        for i, cl in enumerate(labels):
            r[f"sens_{cl}"] = mm["per_class"][cl]["sensitivity"]
            per = [float((Ps[keep][(gs[keep] == p) & (ys[keep] == i)].argmax(1) == i).mean())
                   for p in sorted(set(gs[keep])) if ((gs[keep] == p) & (ys[keep] == i)).any()]
            r[f"patient_weighted_sens_{cl}"] = float(np.mean(per)) if per else float("nan")
        r["deferred"] = int((~keep).sum())
        return r

    sel = {}
    full = np.arange(len(y))
    for c in COVERAGES:
        point = at_cov(full, c)
        draws = defaultdict(list)
        for _ in range(args.n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            bi = np.concatenate([idx[p] for p in pick])
            for k, v in at_cov(bi, c).items():
                if k != "deferred" and v == v:
                    draws[k].append(v)
        point["ci95"] = {k: [float(x) for x in np.percentile(v, [2.5, 97.5])]
                         for k, v in draws.items() if v}
        sel[f"{int(c*100)}%"] = point
    out["selective_prediction"] = sel

    # 2. aggregation ----------------------------------------------------------
    keys = defaultdict(list)
    for i in range(len(y)):
        keys[(photo[i], y[i])].append(i)
    py, pp = [], []
    for (ph, cls), ii in keys.items():
        py.append(cls); pp.append(P[ii].mean(0))
    py, pp = np.array(py), np.array(pp)
    agg = {"photograph_x_class": {
        "units": int(len(py)),
        **{k: v for k, v in burnml.compute_metrics(py, pp.argmax(1), pp, labels).items()
           if k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")}}}
    pc = {}
    for i, cl in enumerate(labels):
        has = [p for p in uniq if (y[g == p] == i).any()]
        detected = [p for p in has if (P[g == p].argmax(1) == i).any()]
        false_alarm = [p for p in uniq if not (y[g == p] == i).any()
                       and (P[g == p].argmax(1) == i).any()]
        pc[cl] = {"patients_with_class": len(has),
                  "patients_detected": len(detected),
                  "patient_sensitivity": len(detected) / len(has) if has else float("nan"),
                  "patients_without_class": int(len(uniq) - len(has)),
                  "patients_with_false_alarm": len(false_alarm)}
    agg["patient_any_detected"] = pc
    out["aggregation"] = agg

    # 3. clustered AUC --------------------------------------------------------
    out["obuchowski_auc"] = {cl: burnml.obuchowski_auc((y == i).astype(int), P[:, i], g)
                             for i, cl in enumerate(labels)}

    rep = os.path.join(args.results_dir, "report")
    os.makedirs(rep, exist_ok=True)
    json.dump(burnml.jsonable(out), open(os.path.join(rep, "posthoc.json"), "w"), indent=2)
    L = ["# Post hoc analyses", "", f"{len(y)} crops, {len(uniq)} patients. "
         "Computed from out-of-fold predictions; no retraining.", "",
         "## Selective prediction (least confident crops deferred)", "",
         f"AURC {aurc:.3f} (lower is better).", "",
         "| Coverage | Deferred | Accuracy (95% CI) | Macro F1 | " +
         " | ".join(f"{c.replace('_',' ')} sens. (crop / patient-wt)" for c in labels) + " |",
         "|---|---|---|---|" + "---|" * len(labels)]
    for k, r in sel.items():
        a = r["ci95"].get("accuracy", [float("nan")] * 2)
        L.append(f"| {k} | {r['deferred']} | {r['accuracy']:.3f} ({a[0]:.3f}–{a[1]:.3f}) | "
                 f"{r['macro_f1']:.3f} | " + " | ".join(
                     f"{r['sens_'+c]:.3f} / {r['patient_weighted_sens_'+c]:.3f}" for c in labels) + " |")
    pxc = agg["photograph_x_class"]
    L += ["", "## Aggregation", "",
          f"Photograph x reference class ({pxc['units']} units, mean calibrated "
          f"probability): accuracy {pxc['accuracy']:.3f}, balanced accuracy "
          f"{pxc['balanced_accuracy']:.3f}, macro AUC {pxc['macro_auc']:.3f}.", "",
          "| Class | Patients with class | Detected (any crop) | Patient sensitivity | Patients with a false alarm |",
          "|---|---|---|---|---|"]
    for cl, r in pc.items():
        L.append(f"| {cl.replace('_',' ')} | {r['patients_with_class']} | {r['patients_detected']} | "
                 f"{r['patient_sensitivity']:.3f} | {r['patients_with_false_alarm']} of {r['patients_without_class']} |")
    L += ["", "## Clustered AUC (Obuchowski 1997), one class versus the rest", "",
          "| Class | AUC | 95% CI | Patients |", "|---|---|---|---|"]
    for cl, r in out["obuchowski_auc"].items():
        L.append(f"| {cl.replace('_',' ')} | {r['auc']:.3f} | {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f} | {r.get('clusters','')} |")
    open(os.path.join(rep, "posthoc.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
