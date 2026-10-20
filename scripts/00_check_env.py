#!/usr/bin/env python3
"""
00_check_env.py — prove the GPU actually works before committing GPU hours.

Reports the driver, the GPU's compute capability, the TensorFlow and CUDA
versions, and then RUNS real kernels to confirm they execute on the GPU rather
than silently falling back to CPU. Writes environment.json, which supplies the
versions for the Methods section.

With --strict, exits non-zero if no GPU is usable, so a SLURM dependency chain
stops instead of running everything on CPU.

Usage:
    python 00_check_env.py --out-dir results
    python 00_check_env.py --out-dir results --strict
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pascal (sm_60) is supported by every CUDA 12 TensorFlow build but by no
# CUDA 13 build: TensorFlow's .bazelrc sets sm_60,sm_70,sm_80,sm_89,compute_90
# for CUDA 12 and sm_75,sm_80,sm_90,sm_100,compute_120 for CUDA 13. Because the
# only PTX shipped is compute_90, a CUDA 13 wheel cannot JIT down to sm_60.
KNOWN_GPUS = {
    (6, 0): "Tesla P100 (Pascal). Supported by CUDA 12 builds; NOT by CUDA 13.",
    (7, 0): "Volta. Tensor Cores available.",
    (7, 5): "Turing. Tensor Cores available.",
    (8, 0): "Ampere A100.",
    (8, 6): "Ampere consumer.",
    (9, 0): "Hopper.",
}


def nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            rows.append({"name": parts[0], "driver": parts[1],
                         "memory": parts[2],
                         "compute_cap": parts[3] if len(parts) > 3 else None})
        return rows
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if no usable GPU")
    ap.add_argument("--check-weights", nargs="*", default=None,
                    help="backbones whose ImageNet weights must load offline")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rec = {}
    problems = []

    print("=" * 66)
    print("  DRIVER AND HARDWARE")
    print("=" * 66)
    smi = nvidia_smi()
    rec["nvidia_smi"] = smi
    if smi:
        for g in smi:
            print(f"  {g['name']} | driver {g['driver']} | {g['memory']}"
                  + (f" | compute capability {g['compute_cap']}"
                     if g["compute_cap"] else ""))
        drv = smi[0]["driver"]
        try:
            major = int(drv.split(".")[0])
            if major >= 525:
                print(f"  driver {drv}: CUDA 12.x capable")
            elif major >= 450:
                print(f"  driver {drv}: CUDA 11.x only, do not install a "
                      "CUDA 12 TensorFlow")
            else:
                problems.append(f"driver {drv} is older than CUDA 11.8 requires")
        except ValueError:
            pass
    else:
        print("  nvidia-smi unavailable (are you on a GPU node?)")
        problems.append("[gpu] nvidia-smi unavailable")

    print()
    print("=" * 66)
    print("  SOFTWARE")
    print("=" * 66)
    import numpy as np
    import cv2
    import tensorflow as tf
    import platform

    rec.update({
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "hostname": platform.node(),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    import burnml
    rec["keras"] = burnml.keras_version()
    try:
        build = tf.sysconfig.get_build_info()
        rec["cuda_version"] = build.get("cuda_version")
        rec["cudnn_version"] = build.get("cudnn_version")
    except Exception:
        rec["cuda_version"] = rec["cudnn_version"] = None
    try:
        import skopt
        rec["scikit_optimize"] = skopt.__version__
    except ImportError:
        rec["scikit_optimize"] = "NOT INSTALLED (needed for the search step)"

    for k in ("python", "tensorflow", "keras", "cuda_version", "cudnn_version",
              "numpy", "opencv", "scikit_optimize", "slurm_cpus"):
        print(f"  {k:<18} {rec.get(k)}")

    cuda = str(rec.get("cuda_version") or "")
    if cuda.startswith("13"):
        problems.append(
            "TensorFlow was built against CUDA 13, whose kernels start at "
            "sm_75. This build cannot run on a Pascal GPU. Reinstall a CUDA 12 "
            "build, e.g. tensorflow[and-cuda]==2.15.1")

    keras_major = burnml.keras_major()
    print(f"\n  Keras {keras_major}.x detected"
          + (" (SavedModel directories, workers= in fit)" if keras_major < 3
             else " (.keras files, no workers= in fit)"))

    print()
    print("=" * 66)
    print("  GPU VISIBILITY")
    print("=" * 66)
    gpus = tf.config.list_physical_devices("GPU")
    rec["gpus"] = [g.name for g in gpus]
    if not gpus:
        print("  No GPU visible to TensorFlow.")
        problems.append("[gpu] TensorFlow sees no GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
        try:
            d = tf.config.experimental.get_device_details(g)
            cc = d.get("compute_capability")
            name = d.get("device_name")
            note = KNOWN_GPUS.get(cc, "")
            print(f"  {g.name} | {name} | compute capability {cc}")
            if note:
                print(f"     {note}")
            rec["compute_capability"] = list(cc) if cc else None
            rec["gpu_name"] = name
            if cc and cc[0] < 7:
                print("     no Tensor Cores: do NOT enable mixed_float16")
                rec["mixed_precision_recommended"] = False
            else:
                rec["mixed_precision_recommended"] = True
        except Exception as e:
            print(f"  {g.name} (details unavailable: {e})")

    print()
    print("=" * 66)
    print("  DO KERNELS ACTUALLY RUN ON THE GPU?")
    print("=" * 66)
    rec["gpu_tests"] = {}
    if gpus:
        try:
            with tf.device("/GPU:0"):
                a = tf.random.normal((1024, 1024))
                c = tf.matmul(a, a)
                _ = float(tf.reduce_sum(c).numpy())
            on_gpu = "GPU" in c.device
            rec["gpu_tests"]["matmul"] = c.device
            print(f"  matmul ran on: {c.device}")
            if not on_gpu:
                problems.append("matmul silently fell back to CPU")
        except Exception as e:
            print(f"  matmul FAILED: {e}")
            problems.append(f"matmul failed: {e}")

        try:
            with tf.device("/GPU:0"):
                x = tf.random.normal((8, 224, 224, 3))
                k = tf.random.normal((3, 3, 3, 32))
                y = tf.nn.conv2d(x, k, strides=1, padding="SAME")
                _ = float(tf.reduce_sum(y).numpy())
            rec["gpu_tests"]["conv2d"] = y.device
            print(f"  conv2d (cuDNN) ran on: {y.device}")
            if "GPU" not in y.device:
                problems.append("conv2d silently fell back to CPU")
        except Exception as e:
            print(f"  conv2d FAILED: {e}")
            problems.append(f"conv2d failed (cuDNN problem): {e}")

        try:
            import time
            base = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_shape=(224, 224, 3))
            xb = tf.random.normal((16, 224, 224, 3))
            base(xb)                       # warm up
            t0 = time.time()
            for _ in range(5):
                base(xb)
            dt = (time.time() - t0) / 5
            rec["gpu_tests"]["efficientnetb0_forward_s"] = round(dt, 4)
            print(f"  EfficientNetB0 forward pass, batch 16: {dt*1000:.0f} ms")
            if dt > 1.0:
                print("     unusually slow: this looks like CPU execution or "
                      "PTX JIT")
        except Exception as e:
            print(f"  EfficientNetB0 test skipped: {e}")

    print()
    print("=" * 66)
    print("  BACKBONES AVAILABLE IN THIS TENSORFLOW")
    print("=" * 66)
    wanted = ["EfficientNetB0", "DenseNet121", "ResNet50V2", "InceptionV3",
              "EfficientNetV2S", "EfficientNetV2B0", "ConvNeXtTiny",
              "ResNetRS50", "RegNetY040"]
    rec["backbones"] = {}
    for w in wanted:
        have = hasattr(tf.keras.applications, w)
        rec["backbones"][w] = have
        print(f"  {'yes' if have else ' no'}  {w}")
    if not rec["backbones"].get("EfficientNetB0"):
        problems.append("EfficientNetB0 missing: TensorFlow is too old")

    if args.check_weights:
        import backbones as BB
        print()
        print("=" * 66)
        print("  IMAGENET WEIGHTS (must already be cached; nodes are offline)")
        print("=" * 66)
        print(f"  cache: {os.path.join(os.path.expanduser('~'), '.keras', 'models')}"
              "  (where Keras 2.15 always looks)")
        rec["weights"] = {}
        for name in args.check_weights:
            if name == "scratch":
                continue
            if not BB.available(name):
                print(f"   no  {name}: not in this TensorFlow")
                problems.append(f"backbone {name} unavailable in TF "
                                f"{tf.__version__}")
                continue
            size = BB.spec(name)["size"]
            try:
                m = BB.build_backbone(name, (size, size, 3), weights="imagenet")
                n = m.count_params()
                del m
                print(f"   ok  {name} ({n/1e6:.1f}M parameters)")
                rec["weights"][name] = True
            except Exception as e:
                print(f"   MISSING  {name}: {type(e).__name__}")
                rec["weights"][name] = False
                problems.append(f"ImageNet weights for {name} not loadable; "
                                "run scripts/00b_fetch_weights.py on a LOGIN "
                                "node")
        tf.keras.backend.clear_session()

    print()
    print("=" * 66)
    if problems:
        print("  PROBLEMS")
        for p in problems:
            print(f"   - {p}")
    else:
        print("  All checks passed. Safe to submit GPU jobs.")
    print("=" * 66)

    if os.environ.get("BURNML_ALLOW_CPU") == "1":
        gpu_only = [p for p in problems if p.startswith("[gpu]")]
        if gpu_only:
            print("\n  BURNML_ALLOW_CPU=1: ignoring GPU problems. TESTING ONLY;")
            print("  never set this for a real run.")
        problems = [p for p in problems if not p.startswith("[gpu]")]
    rec["problems"] = problems
    path = os.path.join(args.out_dir, "environment.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\nWrote {path}")

    if problems and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
