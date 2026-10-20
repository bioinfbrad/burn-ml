#!/usr/bin/env python3
"""
predict.py: run the trained burn-ml ensembles on your own images.

    python scripts/predict.py --task classification path/to/crops/ > /dev/null
    python scripts/predict.py --task detection img1.jpg img2.png --out predictions.csv

Each image should show a single tissue type, as in the training data (a crop of
burnt or normal skin). Output: one row per image with the predicted class, its
probability and the probability of every class, after test-time augmentation
and temperature scaling exactly as in the paper.

RESEARCH USE ONLY. Not a medical device; not validated for clinical decisions.
Full-thickness burns were often missed in cross-validation (see models/MODEL_CARD.md).
"""
import argparse, csv, glob, json, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                 # noqa: E402
import burnml                      # noqa: E402
import backbones as B              # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def list_images(inputs):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            out += sorted(f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True) if f.lower().endswith(EXT))
        elif p.lower().endswith(EXT):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--task", required=True, choices=["classification", "detection"])
    ap.add_argument("--models-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--no-tta", action="store_true", help="skip test-time augmentation (8x faster, slightly different)")
    a = ap.parse_args()
    tdir = os.path.join(a.models_dir, a.task)
    card_path = os.path.join(tdir, "model_card.json")
    if not os.path.exists(card_path):
        sys.exit(f"No models in {tdir}. Download them first:  python scripts/download_models.py")
    card = json.load(open(card_path))
    files = list_images(a.inputs)
    if not files:
        sys.exit("no images found")
    imgs, keep = [], []
    for f in files:
        im = burnml.read_rgb(f)
        if im is None:
            print(f"  skipped (unreadable): {f}")
        else:
            imgs.append(im); keep.append(f)
    import tensorflow as tf
    pre = B.preprocess_fn(card["backbone"]); size = card["image_size"]
    probs = None
    for name in card["members"]:
        model = tf.keras.models.load_model(os.path.join(tdir, name), compile=False)
        p = burnml.predict_tta(model, imgs, size, pre, n_tta=1 if a.no_tta else 8)
        probs = p if probs is None else probs + p
        del model
    probs = burnml.apply_temperature(probs / len(card["members"]), card["temperature"])
    labels = card["labels"]
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "prediction", "probability"] + [f"p_{c}" for c in labels])
        for f, p in zip(keep, probs):
            k = int(np.argmax(p)); w.writerow([f, labels[k], f"{p[k]:.4f}"] + [f"{x:.4f}" for x in p])
    counts = {c: int(np.sum(np.argmax(probs, 1) == i)) for i, c in enumerate(labels)}
    print(f"{len(keep)} image(s) -> {a.out} | predicted: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
    print("Research use only; see models/MODEL_CARD.md for limitations.")


if __name__ == "__main__":
    main()
