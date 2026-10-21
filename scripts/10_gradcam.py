#!/usr/bin/env python3
"""
10_gradcam.py — Grad-CAM saliency with a model-randomisation sanity check.

Uses the archived ensemble members of a run (models/fold_<k>/member_<m>), so
every map is produced by models that never saw the crop. Maps are averaged
over the members of the fold that held the crop out.

Sanity check (Adebayo et al., NeurIPS 2018): the weights are randomised in a
cascade from the classifier head downward; if the maps barely change, they
do not reflect what the model learned. Similarity to the original map is
reported as Spearman rank correlation and SSIM at each stage.

Panel, chosen by a fixed seed before looking at maps: 2 correct and 2
incorrect crops per class, plus every error of --all-errors-class.

Saliency shows which pixels influence the output. It is not evidence of
clinical reasoning, and the caption says so.

Usage (GPU node; TensorFlow needs AVX):
    python 10_gradcam.py --run-dir results/classification/repeats/r1/efficientnetb0 \
        --raw-dir raw_Images --task classification
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burnml
import backbones as B


def find_parts(model):
    import tensorflow as tf
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    after = model.layers[model.layers.index(base) + 1:]
    dense = [l for l in after if isinstance(l, tf.keras.layers.Dense)][-1]
    gap = [l for l in after if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)][0]
    return base, gap, dense


def gradcam(model, x, cls):
    import tensorflow as tf
    base, gap, dense = find_parts(model)
    with tf.GradientTape() as tape:
        conv = base(x, training=False)
        tape.watch(conv)
        logit = tf.matmul(gap(conv), dense.kernel) + dense.bias
        score = logit[:, cls]
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0].numpy()
    cam = cv2.resize(cam, (x.shape[2], x.shape[1]))
    return cam / cam.max() if cam.max() > 0 else cam


def ssim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    k = (11, 11)
    mu_a, mu_b = cv2.GaussianBlur(a, k, 1.5), cv2.GaussianBlur(b, k, 1.5)
    va = cv2.GaussianBlur(a * a, k, 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a.ravel(), b.ravel()).correlation
    return float(r) if r == r else 0.0


STAGES = [("head", lambda n: False), ("top", lambda n: n.startswith("top")),
          ("block7", lambda n: n.startswith("block7")),
          ("block6", lambda n: n.startswith("block6")),
          ("block5", lambda n: n.startswith("block5"))]


def randomise(model, upto, rng):
    """Cumulatively re-draw weights for the head and the named backbone
    stages, keeping each tensor's scale."""
    base, gap, dense = find_parts(model)
    def redraw(layer):
        # learned parameters only; BatchNorm running statistics are left as
        # they are (a re-drawn variance can go negative and produce NaNs)
        for v in layer.weights:
            if "moving_mean" in v.name or "moving_variance" in v.name:
                continue
            w = v.numpy()
            v.assign(rng.normal(0, float(np.std(w)) or 0.05, w.shape).astype(w.dtype))
    redraw(dense)
    for name, pred in STAGES[1:upto + 1]:
        for l in base.layers:
            if pred(l.name):
                redraw(l)


NAMES = {"Partial_thickness_burn": "Partial-thickness", "Full_thickness_burn": "Full-thickness",
         "Normal_skin": "Normal skin", "burned_skin": "Burnt skin"}


def patient_numbers(manifest):
    """file -> anonymous patient number (sorted group order, as in 12/13_*.py)."""
    import csv
    rows = list(csv.DictReader(open(manifest)))
    groups = sorted({r["patient_group"] for r in rows})
    return {r["file"]: groups.index(r["patient_group"]) + 1 for r in rows}


def find_manifest(run_dir):
    d = os.path.abspath(run_dir)
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "manifest.csv")):
            return os.path.join(d, "manifest.csv")
        d = os.path.dirname(d)
    return None


def plot_panel(maps, labels, pid, out_dir):
    """Publication panel. Titles carry anonymous patient numbers and class names only;
    never file names or patient group names (derived from accession numbers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(maps); cols = 6; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.7 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (c, im, cam) in enumerate(maps):
        ax = axes[k // cols][k % cols]
        heat = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        ax.imshow((0.6 * im + 0.4 * heat).astype(np.uint8))
        err = c["true"] != c["pred"]
        who = f"Patient {pid[c['file']]}" if c["file"] in pid else "Patient ?"
        ax.set_title(f"{who}\n{NAMES.get(labels[c['true']], labels[c['true']])} \u2192 "
                     f"{NAMES.get(labels[c['pred']], labels[c['pred']])} ({c['conf']:.2f})",
                     fontsize=6.5, color="crimson" if err else "black")
    fig.suptitle("Grad-CAM, averaged over the held-out models of each crop's fold "
                 "(red titles: misclassified). Highlighted regions influenced the output; "
                 "they are not evidence of clinical reasoning.", fontsize=7.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_gradcam_panel.{ext}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--task", required=True, choices=list(burnml.TASKS))
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--all-errors-class", default="Full_thickness_burn")
    ap.add_argument("--max-panel", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default=None,
                    help="manifest.csv used to number patients 1..N as in the paper's figures "
                         "(default: the nearest manifest.csv above --run-dir)")
    args = ap.parse_args()

    burnml.configure_gpu()
    import tensorflow as tf
    labels, _ = burnml.TASKS[args.task]
    folds = {}
    for p in glob.glob(os.path.join(args.run_dir, "folds", "fold_*.json")):
        d = json.load(open(p)); folds[d["fold"]] = d
    backbone = next(iter(folds.values()))["settings"]["backbone"]
    size = next(iter(folds.values()))["settings"]["image_size"]
    pre = B.preprocess_fn(backbone)

    cases = []
    for k, d in folds.items():
        P = np.array(d["probs"])
        for f, t, p in zip(d["files"], d["true_labels"], P):
            cases.append({"file": f, "fold": k, "true": labels.index(t),
                          "pred": int(p.argmax()), "conf": float(p.max())})
    rng = np.random.RandomState(args.seed)
    panel = []
    for i, c in enumerate(labels):
        for correct in (True, False):
            pool = [x for x in cases if x["true"] == i and (x["pred"] == i) == correct]
            rng.shuffle(pool)
            panel += pool[:args.per_cell]
    if args.all_errors_class in labels:
        j = labels.index(args.all_errors_class)
        panel += [x for x in cases if x["true"] == j and x["pred"] != j and x not in panel]
    panel = panel[:args.max_panel]
    print(f"panel: {len(panel)} crops (seed {args.seed}), backbone {backbone}, {size}px")

    def load_members(k):
        paths = sorted(glob.glob(os.path.join(args.run_dir, "models", f"fold_{k}", "member_*")))
        return [tf.keras.models.load_model(p, compile=False) for p in paths]

    cache, maps, sanity = {}, [], {s[0]: {"spearman": [], "ssim": []} for s in STAGES}
    srng = np.random.RandomState(args.seed)
    for c in panel:
        if c["fold"] not in cache:
            cache.clear(); burnml.free_memory()
            cache[c["fold"]] = load_members(c["fold"])
        members = cache[c["fold"]]
        if not members:
            continue
        img = burnml.read_rgb(os.path.join(args.raw_dir, c["file"]))
        x = pre(burnml.stack_uint8([img], size).astype(np.float32))
        cam = np.mean([gradcam(m, x, c["pred"]) for m in members], axis=0)
        maps.append((c, burnml.to_uint8(img, size), cam))
        # sanity cascade on a fresh copy of member 1
        for si in range(len(STAGES)):
            m = tf.keras.models.clone_model(members[0])
            m.set_weights(members[0].get_weights())
            randomise(m, si, srng)
            rc = gradcam(m, x, c["pred"])
            sanity[STAGES[si][0]]["spearman"].append(spearman(cam, rc))
            sanity[STAGES[si][0]]["ssim"].append(ssim(cam, rc))
            del m

    out_dir = os.path.join(args.run_dir, "report")
    os.makedirs(out_dir, exist_ok=True)
    summary = {s: {k: float(np.mean(v)) if v else float("nan") for k, v in d.items()}
               for s, d in sanity.items()}
    json.dump(burnml.jsonable({"panel": [m[0] for m in maps], "sanity_check": summary,
                               "note": "similarity of the trained-model map to maps after "
                                       "cumulative weight randomisation; values near 1 "
                                       "would mean the maps ignore the learned weights"}),
              open(os.path.join(out_dir, "gradcam.json"), "w"), indent=2)

    manifest = args.manifest or find_manifest(args.run_dir)
    pid = patient_numbers(manifest) if manifest else {}
    plot_panel(maps, labels, pid, out_dir)
    # The maps and images themselves, so the figure can be redrawn without a GPU.
    # They contain patient images: keep them on the secure system, never publish them.
    np.savez_compressed(os.path.join(out_dir, "gradcam_maps_CONTAINS_PATIENT_IMAGES.npz"),
                        images=np.stack([m[1] for m in maps]), cams=np.stack([m[2] for m in maps]),
                        true=np.array([m[0]["true"] for m in maps]), pred=np.array([m[0]["pred"] for m in maps]),
                        conf=np.array([m[0]["conf"] for m in maps]),
                        patient=np.array([pid.get(m[0]["file"], 0) for m in maps]))

    print("\nsanity check: mean similarity to the trained-model map after cumulative randomisation")
    for s, v in summary.items():
        print(f"   randomised up to {s:<7} Spearman {v['spearman']:+.3f}   SSIM {v['ssim']:+.3f}")
    print(f"\nWrote fig_gradcam_panel.pdf/.png, gradcam.json and the maps (.npz, patient images: do not publish) to {out_dir}")


if __name__ == "__main__":
    main()
