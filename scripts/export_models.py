#!/usr/bin/env python3
"""
export_models.py: package the archival ensembles for sharing.

Collects the three-member ensemble trained on all patients for each task
(scripts/04_train_final.py), writes a model card for each, and zips them as
burn-ml-models-v<version>.zip for a GitHub release or Zenodo deposit.

The bundle holds only the networks and a model card written from scratch:
no images, no file names and no patient groups (which are derived from
hospital accession numbers), and no paths from the machine they were trained on.

Usage:
    python scripts/export_models.py --results results --cv-results run1/results
    -> dist/burn-ml-models-v2.0.0.zip   (extract it in the repository root)
"""
import argparse, datetime, glob, hashlib, json, os, shutil, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml                     # noqa: E402

INTENDED_USE = ("Research use only: reproducing and extending the analyses of the associated paper. "
                "Not a medical device; not validated for clinical decision-making.")
LIMITATIONS = [
    "Trained on 284 crops from 18 patient episodes at a single UK burns service; not externally validated.",
    "Inputs were crops selected by a clinician to contain a single tissue type; whole photographs were not tested.",
    "Labels came from clinical assessment at presentation, which is accurate in only 60-75% of cases.",
    "Full-thickness burns were often missed (patient-weighted sensitivity 0.431 in cross-validation).",
    "Skin type (Fitzpatrick) was not recorded; performance across skin tones is unknown.",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cv_summary(metrics_path, labels):
    if not os.path.exists(metrics_path):
        return None
    m = json.load(open(metrics_path)); ci = m.get("ci95", {})
    out = {k: {"estimate": round(m[k], 3), "ci95": [round(x, 3) for x in ci.get(k, [])]}
           for k in ("accuracy", "balanced_accuracy", "macro_auc") if k in m}
    out["per_class"] = {c: {"sensitivity": round(m["per_class"][c]["sensitivity"], 3),
                            "specificity": round(m["per_class"][c]["specificity"], 3)} for c in labels}
    out["method"] = ("pooled out-of-fold predictions of six-fold patient-grouped cross-validation; "
                     "95% CIs from a patient-level cluster bootstrap")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder holding <task>/<backbone>/final/")
    ap.add_argument("--cv-results", default=None, help="folder with the cross-validation metrics to quote "
                    "(default: --results; for the paper's numbers use run1/results)")
    ap.add_argument("--tasks", nargs="+", default=["classification", "detection"])
    ap.add_argument("--backbone", default="efficientnetb0")
    ap.add_argument("--version", default="2.0.0")
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    cvr = a.cv_results or a.results
    stage = os.path.join(a.out, "stage"); shutil.rmtree(stage, ignore_errors=True)
    md = [f"# burn-ml trained models, version {a.version}", "",
          "Ensembles of three EfficientNetB0 networks (ImageNet-pretrained, fine-tuned), trained on all patients "
          "with the recipe of the associated paper. Run them with `python scripts/predict.py`.", "",
          f"**Intended use.** {INTENDED_USE}", "", "**Limitations.**", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    for task in a.tasks:
        fdir = os.path.join(a.results, task, a.backbone, "final")
        rec = json.load(open(os.path.join(fdir, "final_model_record.json")))
        # Class order of the network's outputs, as recorded at training time.
        labels = list(rec.get("labels") or burnml.TASKS[task][0])
        if sorted(labels) != sorted(burnml.TASKS[task][0]):
            sys.exit(f"{task}: recorded labels {labels} do not match the task definition")
        members = sorted(glob.glob(os.path.join(fdir, f"final_{task}_{a.backbone}_m*.model")) +
                         glob.glob(os.path.join(fdir, f"final_{task}_{a.backbone}_m*.keras")))
        if not members:
            sys.exit(f"no ensemble members found in {fdir}")
        tdir = os.path.join(stage, "models", task); os.makedirs(tdir)
        names = []
        for k, m in enumerate(members, 1):
            ext = os.path.splitext(m)[1]; dst = os.path.join(tdir, f"member_{k}{ext}"); names.append(os.path.basename(dst))
            (shutil.copytree if os.path.isdir(m) else shutil.copy2)(m, dst)
        card = {
            "name": f"burn-ml {task} ensemble", "version": a.version, "task": task, "labels": labels,
            "backbone": a.backbone, "pretraining": "ImageNet", "image_size": rec["image_size"],
            "input": "RGB image of a single tissue type (a crop), any size; resized to image_size x image_size",
            "preprocessing": rec.get("preprocessing"),
            "inference": "mean of the members' softmax outputs, each averaged over 8 rotations/reflections; "
                         "then temperature scaling softmax(log(p) / T)",
            "temperature": rec["inference_temperature"], "members": names,
            "training": {"crops": rec.get("training_crops"), "patient_episodes": rec.get("patients"),
                         "recipe": rec.get("recipe")},
            "cross_validated_performance": cv_summary(os.path.join(cvr, task, a.backbone, "metrics.json"), labels),
            "intended_use": INTENDED_USE, "limitations": LIMITATIONS,
            "licence": "AGPL-3.0 (see LICENSE in https://github.com/bioinfbrad/burn-ml)",
            "exported": datetime.date.today().isoformat(),
        }
        json.dump(card, open(os.path.join(tdir, "model_card.json"), "w"), indent=2)
        p = card["cross_validated_performance"] or {}
        md += [f"## {task.capitalize()}", "", f"Classes (output order): {', '.join(labels)}. Members: {len(names)}. "
               f"Temperature: {card['temperature']:.3f}. Trained on {card['training']['crops']} crops from "
               f"{card['training']['patient_episodes']} patient episodes.", ""]
        if p:
            md += [f"Cross-validated performance (patients unseen in training): accuracy {p['accuracy']['estimate']:.3f} "
                   f"(95% CI {p['accuracy']['ci95'][0]:.3f}-{p['accuracy']['ci95'][1]:.3f}), macro AUC "
                   f"{p['macro_auc']['estimate']:.3f} ({p['macro_auc']['ci95'][0]:.3f}-{p['macro_auc']['ci95'][1]:.3f}).", ""]
    open(os.path.join(stage, "models", "MODEL_CARD.md"), "w").write("\n".join(md) + "\n")
    sums = []
    for root, _, files in os.walk(os.path.join(stage, "models")):
        for f in sorted(files):
            fp = os.path.join(root, f); sums.append(f"{sha256(fp)}  {os.path.relpath(fp, stage)}")
    open(os.path.join(stage, "models", "SHA256SUMS"), "w").write("\n".join(sorted(sums)) + "\n")
    zpath = os.path.join(a.out, f"burn-ml-models-v{a.version}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(os.path.join(stage, "models")):
            for f in files:
                fp = os.path.join(root, f); z.write(fp, os.path.relpath(fp, stage))
    shutil.rmtree(stage)
    print(f"Wrote {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)\nSHA-256 of the zip: {sha256(zpath)}")
    print("Attach the zip to the GitHub release and quote the SHA-256 in the release notes.")


if __name__ == "__main__":
    main()
