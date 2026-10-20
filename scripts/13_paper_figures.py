#!/usr/bin/env python3
"""
13_paper_figures.py: the three figures of the manuscript, drawn from the saved
out-of-fold predictions and comparison files.

Patients are numbered 1..N in sorted group order (as in 12_results_ledger.py);
no accession number, patient group name or file name appears in any figure, so
the output is safe to publish. The pipeline's own report figures (05/07/10)
label patients by group name and must NOT be published.

Usage:
    python scripts/13_paper_figures.py --run1 run1/results --results results --out figures
"""
import argparse, csv, json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600, "pdf.fonttype": 42})
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--run1", required=True, help="results of the run of record (run1/results)")
_ap.add_argument("--results", required=True, help="replicate run and extension (results); used for learning curves")
_ap.add_argument("--out", default="figures")
_a = _ap.parse_args(); R1, EXT, OUT = _a.run1, _a.results, _a.out; os.makedirs(OUT, exist_ok=True)
CB = {"PT": "#E69F00", "FT": "#D55E00", "NS": "#0072B2", "burn": "#CC79A7", "grey": "#555555"}
CLS = ["Partial_thickness_burn", "Normal_skin", "Full_thickness_burn"]; NAME = {"Partial_thickness_burn": "Partial-thickness", "Normal_skin": "Normal skin", "Full_thickness_burn": "Full-thickness"}
COL = {"Partial_thickness_burn": CB["PT"], "Normal_skin": CB["NS"], "Full_thickness_burn": CB["FT"]}

def load(task):
    rows = list(csv.DictReader(open(f"{R1}/{task}/efficientnetb0/oof_predictions.csv")))
    labels = [k[2:] for k in rows[0] if k.startswith("p_")]
    y = np.array([labels.index(r["true_label"]) for r in rows]); P = np.array([[float(r["p_" + c]) for c in labels] for r in rows])
    g = np.array([r["patient_group"] for r in rows]); return labels, y, P, g

def roc(score, pos):
    o = np.argsort(-score); s, t = score[o], pos[o]
    tp = np.cumsum(t); fp = np.cumsum(1 - t); keep = np.r_[np.diff(s) != 0, True]
    tpr = np.r_[0, tp[keep] / t.sum()]; fpr = np.r_[0, fp[keep] / (1 - t).sum()]
    return fpr, tpr, np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

# stable anonymous patient numbers shared by every figure
_, _, _, gc = load("classification"); PID = {p: i + 1 for i, p in enumerate(sorted(set(gc)))}

# ---------------- Figure 1: discrimination and calibration -----------------------
fig, ax = plt.subplots(1, 4, figsize=(7.1, 1.95), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
lab, y, P, g = load("detection"); burn = lab.index([c for c in lab if c != "Normal_skin"][0])
f, t, a = roc(P[:, burn], (y == burn).astype(int))
ax[0].plot(f, t, color=CB["burn"], lw=1.4, label=f"Burnt skin {a:.3f}")
ax[0].set_title("A  Detection", loc="left", fontweight="bold")
lab, y, P, g = load("classification")
for c in CLS:
    k = lab.index(c); f, t, a = roc(P[:, k], (y == k).astype(int)); ax[1].plot(f, t, color=COL[c], lw=1.4, label=f"{NAME[c].split('-')[0].split()[0]} {a:.3f}")
ax[1].set_title("B  Classification", loc="left", fontweight="bold")
for a_ in ax[:2]:
    a_.plot([0, 1], [0, 1], ls=":", color="#999", lw=0.8); a_.set_xlabel("1 − specificity"); a_.set_ylabel("Sensitivity")
    a_.set_xlim(-0.02, 1.02); a_.set_ylim(-0.02, 1.02); a_.set_aspect("equal"); a_.legend(loc="lower right", frameon=False, handlelength=1.0, title="AUC", title_fontsize=6.8, borderaxespad=0.2, fontsize=6.5)
# confusion matrix (rows: reference, columns: predicted), order PT, FT, NS
order = ["Partial_thickness_burn", "Full_thickness_burn", "Normal_skin"]; idx = [lab.index(c) for c in order]
pred = P.argmax(1); M = np.array([[np.sum((y == i) & (pred == j)) for j in idx] for i in idx])
Mn = M / M.sum(1, keepdims=True)
ax[2].imshow(Mn, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax[2].text(j, i, f"{M[i, j]}\n({100*Mn[i, j]:.0f}%)", ha="center", va="center", fontsize=6.5, color="white" if Mn[i, j] > 0.6 else "black")
short = ["Partial", "Full", "Normal"]
ax[2].set_xticks(range(3), short); ax[2].set_yticks(range(3), short); ax[2].set_xlabel("Predicted"); ax[2].set_ylabel("Reference")
ax[2].set_title("C  Classification errors", loc="left", fontweight="bold"); ax[2].spines[:].set_visible(False)
# reliability (top-label confidence vs accuracy, equal-count bins)
for task, colr, nm in (("detection", CB["burn"], "Detection"), ("classification", CB["grey"], "Classification")):
    lab_, y_, P_, _ = load(task); conf = P_.max(1); corr = (P_.argmax(1) == y_).astype(float)
    bins = np.array_split(np.argsort(conf), 8); xs = [conf[b].mean() for b in bins]; ys = [corr[b].mean() for b in bins]
    ax[3].plot(xs, ys, "o-", color=colr, ms=2.8, lw=1.1, label=nm)
ax[3].plot([0, 1], [0, 1], ls=":", color="#999", lw=0.8); ax[3].set_xlim(0.3, 1.02); ax[3].set_ylim(0.3, 1.02)
ax[3].set_xlabel("Predicted probability"); ax[3].set_ylabel("Observed accuracy"); ax[3].set_aspect("equal")
ax[3].set_title("D  Calibration", loc="left", fontweight="bold"); ax[3].legend(loc="lower right", frameon=False)
fig.tight_layout(w_pad=1.0); fig.savefig(f"{OUT}/fig1_performance.pdf"); fig.savefig(f"{OUT}/fig1_performance.png", dpi=200); plt.close(fig)

# ---------------- Figure 2: per-patient sensitivity ---------------------------------
lab, y, P, g = load("classification"); pred = P.argmax(1)
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
summary = {}
for a_, c, ttl in ((ax[0], "Full_thickness_burn", "A  Full-thickness burn"), (ax[1], "Partial_thickness_burn", "B  Partial-thickness burn")):
    k = lab.index(c); rows = []
    for p in sorted(set(g[y == k])):
        m = (g == p) & (y == k); n = int(m.sum()); hit = int((pred[m] == k).sum()); rows.append((PID[p], hit, n))
    rows.sort(key=lambda r: (r[1] / r[2], r[2]))
    crop = sum(r[1] for r in rows) / sum(r[2] for r in rows); pw = np.mean([r[1] / r[2] for r in rows]); summary[c] = (crop, pw, len(rows))
    for i, (pid, hit, n) in enumerate(rows):
        lo, hi = wilson(hit, n); a_.plot([lo, hi], [i, i], color=COL[c], lw=1.2); a_.plot(hit / n, i, "o", color=COL[c], ms=3.5)
        a_.text(1.04, i, f"{hit}/{n}", va="center", fontsize=6.5, transform=a_.get_yaxis_transform())
    a_.set_yticks(range(len(rows)), [f"Patient {r[0]}" for r in rows])
    a_.axvline(crop, color="black", lw=0.9, ls="--"); a_.axvline(pw, color=CB["grey"], lw=0.9, ls=":")
    a_.set_xlim(-0.03, 1.03); a_.set_xlabel("Sensitivity (95% Wilson interval)"); a_.set_title(ttl, loc="left", fontweight="bold")
    a_.set_ylim(-0.6, len(rows) - 0.4)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color="black", lw=0.9, ls="--"), Line2D([], [], color=CB["grey"], lw=0.9, ls=":")],
           ["Crop-level sensitivity (all crops pooled)", "Patient-weighted sensitivity (each patient counts equally)"],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.8)
fig.tight_layout(w_pad=3.0, rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/fig2_patients.pdf"); fig.savefig(f"{OUT}/fig2_patients.png", dpi=200); plt.close(fig)

# ---------------- Figure 3: pretraining effect and learning curves --------------------
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.35, 1]})
names = [("confirmatory", "Planned grid (pre-specified)"), ("_steps", "Step-matched (pre-specified)"), ("_widelr", "Grid to 3×10⁻² (post hoc)"), ("_bracket", "Grid to 3×10⁻¹ (post hoc)")]
yt, yl = [], []; ypos = 0
for task, colr, off in (("classification", CB["grey"], 0.16), ("detection", CB["burn"], -0.16)):
    c = json.load(open(f"{R1}/{task}/comparison/backbone_comparison.json"))
    for i, (key, nm) in enumerate(names):
        d = c["confirmatory"]["d_macro_auc"] if key == "confirmatory" else next(x for x in c["other_random_init_arms"] if x["arm"].endswith(key))["d_macro_auc"]
        yy = len(names) - 1 - i + off
        ax[0].plot(d["ci95"], [yy, yy], color=colr, lw=1.3); ax[0].plot(d["estimate"], yy, "s" if key == "confirmatory" else "o", color=colr, ms=3.6,
                   label=("Classification" if task == "classification" else "Detection") if i == 0 else None)
ax[0].set_yticks(range(len(names)), [n for _, n in names][::-1]); ax[0].axvline(0, color="black", lw=0.8)
ax[0].set_xlabel("Δ macro AUC, pretrained − random init\n(BCa 95% CI)"); ax[0].legend(loc="lower right", frameon=False)
ax[0].set_title("A  Effect of ImageNet pretraining", loc="left", fontweight="bold")
for task, colr, nm in (("classification", CB["grey"], "Classification"), ("detection", CB["burn"], "Detection")):
    lc = json.load(open(f"{EXT}/{task}/learning_curve_efficientnetb0.json")); ks = sorted(lc["points"], key=int)
    xs = [int(k) for k in ks]; m = [lc["points"][k]["macro_auc"]["mean"] for k in ks]
    lo = [lc["points"][k]["macro_auc"]["min"] for k in ks]; hi = [lc["points"][k]["macro_auc"]["max"] for k in ks]
    ax[1].errorbar(xs, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="o-", color=colr, ms=3, lw=1.1, capsize=2, label=nm)
    ax[1].plot([15], [lc["full"]["macro_auc"]], "D", color=colr, ms=3.6, mfc="white")
ax[1].set_xticks([6, 9, 12, 15], ["6", "9", "12", "all\n(~15)"]); ax[1].set_xlabel("Training patients per fold"); ax[1].set_ylabel("Macro AUC")
ax[1].set_ylim(0.85, 1.0); ax[1].legend(loc="lower right", frameon=False); ax[1].set_title("B  Learning curves", loc="left", fontweight="bold")
fig.tight_layout(w_pad=2.0); fig.savefig(f"{OUT}/fig3_pretraining.pdf"); fig.savefig(f"{OUT}/fig3_pretraining.png", dpi=200); plt.close(fig)
print("figures written | per-patient check:", {NAME[c]: f"crop {v[0]:.3f}, patient-weighted {v[1]:.3f}, {v[2]} patients" for c, v in summary.items()})
