#!/usr/bin/env python3
"""
00b_fetch_weights.py — download the ImageNet weights, WITHOUT TensorFlow.

Run once on a LOGIN node (compute nodes have no internet). This script uses
only the Python standard library, deliberately: official TensorFlow wheels
need the CPU's AVX instructions, and many login nodes lack them, so importing
TensorFlow there dies with "Illegal instruction".

It writes the exact files Keras 2.15 looks for into ~/.keras/models. Keras
2.15 always reads weights from ~/.keras/models (KERAS_HOME does not change
that location), and your home directory is mounted on the compute nodes.
Each file is verified against the hash Keras itself checks, so a file that
passes here is guaranteed to be accepted offline.

    python scripts/00b_fetch_weights.py                 # the pre-specified set
    python scripts/00b_fetch_weights.py --backbones efficientnetb0
    python scripts/00b_fetch_weights.py --force         # re-download

Works with any Python >= 3.6, including the Miniconda base environment.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.request

# Taken verbatim from keras 2.15.0 (keras/src/applications/*.py): the file
# name, origin URL and md5 hash that data_utils.get_file() uses. MobileNetV2
# ships no hash in Keras, so its size is sanity-checked instead.
G = "https://storage.googleapis.com/"
WEIGHTS = {
    "efficientnetb0": (
        "efficientnetb0_notop.h5",
        G + "keras-applications/efficientnetb0_notop.h5",
        "50bc09e76180e00e4465e1a485ddc09d"),
    "densenet121": (
        "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
        G + "tensorflow/keras-applications/densenet/"
            "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "30ee3e1110167f948a6b9946edeeb738"),
    "resnet50v2": (
        "resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
        G + "tensorflow/keras-applications/resnet/"
            "resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "fac2f116257151a9d068a22e544a4917"),
    "mobilenetv2": (
        "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
        G + "tensorflow/keras-applications/mobilenet_v2/"
            "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
        None),
}
PRESPECIFIED = ["efficientnetb0", "densenet121", "resnet50v2", "mobilenetv2"]
MIN_BYTES = 1_000_000          # every real weight file is several MB


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def valid(path, expected):
    if not os.path.isfile(path) or os.path.getsize(path) < MIN_BYTES:
        return False
    return expected is None or md5(path) == expected


def download(url, dest, retries=3):
    for attempt in range(1, retries + 1):
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest),
                                       suffix=".part")
            os.close(fd)
            with urllib.request.urlopen(url, timeout=60) as r, \
                    open(tmp, "wb") as out:
                total = int(r.headers.get("Content-Length") or 0)
                done, t0 = 0, time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        sys.stdout.write(f"\r      {done/2**20:6.1f} / "
                                         f"{total/2**20:.1f} MB")
                        sys.stdout.flush()
                sys.stdout.write(f"  ({time.time()-t0:.0f}s)\n")
            os.replace(tmp, dest)            # atomic: no half-written file
            os.chmod(dest, 0o644)
            return True
        except Exception as e:
            print(f"      attempt {attempt} failed: {type(e).__name__}: {e}")
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(2 * attempt)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=PRESPECIFIED)
    ap.add_argument("--dest", default=os.path.join(
        os.path.expanduser("~"), ".keras", "models"),
        help="default ~/.keras/models, where Keras 2.15 looks")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    print(f"Weight cache: {args.dest}\n")
    failed = []
    for name in args.backbones:
        if name == "scratch":
            continue
        if name not in WEIGHTS:
            print(f"  {name}: no entry in this table (add it from the Keras "
                  "source, or fetch it on a node with TensorFlow and internet)")
            failed.append(name)
            continue
        fname, url, h = WEIGHTS[name]
        path = os.path.join(args.dest, fname)
        if not args.force and valid(path, h):
            print(f"  ok        {name:<15} already present and verified")
            continue
        print(f"  fetching  {name:<15} {fname}")
        if not download(url, path) or not valid(path, h):
            print(f"  FAILED    {name}")
            failed.append(name)
            if os.path.exists(path) and not valid(path, h):
                os.remove(path)
            continue
        print(f"  ok        {name:<15} {os.path.getsize(path)/2**20:.1f} MB, "
              + ("md5 matches Keras" if h else "size checked"))

    print()
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        print("  If this node has no internet either, download the URLs above "
              "elsewhere\n  and copy the files into", args.dest)
        sys.exit(1)
    print("  All weights present. The GPU nodes will load them offline;")
    print("  s00_env confirms this before any training starts.")


if __name__ == "__main__":
    main()
