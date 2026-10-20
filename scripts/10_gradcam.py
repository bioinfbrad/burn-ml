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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--task", required=True, choices=list(burnml.TASKS))
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--all-errors-class", default="Full_thickness_burn")
    ap.add_argument("--max-panel", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(maps); cols = 6; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (c, im, cam) in enumerate(maps):
        ax = axes[k // cols][k % cols]
        heat = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        ax.imshow((0.6 * im + 0.4 * heat).astype(np.uint8))
        ok = "correct" if c["true"] == c["pred"] else "ERROR"
        ax.set_title(f"{labels[c['true']][:4]}->{labels[c['pred']][:4]} {c['conf']:.2f}\n{ok}",
                     fontsize=7, color="black" if ok == "correct" else "crimson")
    fig.suptitle("Grad-CAM (ensemble mean over the fold's held-out models). Regions that "
                 "influence the output; not evidence of clinical reasoning.", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_gradcam_panel.png"), dpi=200)
    plt.close(fig)

    print("\nsanity check: mean similarity to the trained-model map after cumulative randomisation")
    for s, v in summary.items():
        print(f"   randomised up to {s:<7} Spearman {v['spearman']:+.3f}   SSIM {v['ssim']:+.3f}")
    print(f"\nWrote fig_gradcam_panel.png and gradcam.json to {out_dir}")


if __name__ == "__main__":
    main()
