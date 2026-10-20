"""
burnml.py — shared library for the patient-grouped burn CNN pipeline.

Two tasks share this code:
  classification : 3 classes (partial-thickness, normal skin, full-thickness)
  detection      : 2 classes (burned skin = full + partial, normal skin)
Within a task, every model is scored on identical patient folds, so models
can be compared as paired data. The two tasks balance classes differently, so
their folds differ and cross-task comparisons are unpaired.

Memory design
-------------
Augmented images are held as uint8 and normalised lazily inside the input
pipeline, so a fold costs a quarter of what a float32 array would. The
augmentation order is shuffled while the array is filled, so there is never a
second full copy. Above a memory budget the loader switches to streaming,
where rotations are produced per batch from the originals and memory stays
flat however many rotations are requested.

    24 rotations   ~5,600 images   0.16 GB uint8  (1.24 GB as float32 + copy)
   144 rotations  ~33,400 images   0.93 GB uint8, or flat under streaming

Requires only numpy, opencv and tensorflow. scikit-optimize is needed for the
hyperparameter search step only.
"""
import csv
import gc
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np
import cv2

CLASS_DIRS = ["Full_thickness_burn", "Partial_thickness_burn", "Normal_skin"]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

TASKS = {
    "classification": (
        ["Partial_thickness_burn", "Normal_skin", "Full_thickness_burn"],
        {c: c for c in CLASS_DIRS},
    ),
    "detection": (
        ["burned_skin", "Normal_skin"],
        {"Full_thickness_burn": "burned_skin",
         "Partial_thickness_burn": "burned_skin",
         "Normal_skin": "Normal_skin"},
    ),
}

DEFAULT_IMG_SIZE = {"classification": 100, "detection": 100}


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seeds(seed):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu(threads=None):
    """Grow GPU memory on demand instead of reserving the whole card, and cap
    CPU threads to the SLURM allocation. Call once at the start of any script
    that trains."""
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    n = threads or int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if n:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(max(n // 2, 1))
        except RuntimeError:
            pass


def tf_version_tuple():
    import tensorflow as tf
    parts = []
    for p in tf.__version__.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def keras_version():
    """Version of the Keras that tf.keras actually points to.

    tf.keras.__version__ is missing in TF 2.15, so fall back to the keras
    package, then to the TensorFlow version. Versions are compared as numbers:
    as strings, "2.6.0" > "2.16" and TF 2.6 would be misread as Keras 3.
    """
    import tensorflow as tf
    v = getattr(tf.keras, "__version__", None)
    if not v:
        try:
            import keras
            v = getattr(keras, "__version__", None)
        except Exception:
            v = None
    if not v:
        v = "3.0.0" if tf_version_tuple() >= (2, 16) else "2.0.0"
    return str(v)


def keras_major():
    """2 on TF 2.6-2.15 (and on TF >= 2.16 with TF_USE_LEGACY_KERAS=1),
    3 on TF >= 2.16 by default."""
    try:
        return int(keras_version().split(".")[0])
    except ValueError:
        return 3 if tf_version_tuple() >= (2, 16) else 2


def loader_fit_kwargs(workers=None):
    """workers/use_multiprocessing exist on Keras 2 only. Passing them on
    Keras 3 raises, so they are omitted there."""
    if keras_major() >= 3:
        return {}
    n = workers or max(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "4") or 4) - 1, 1)
    return {"workers": n, "use_multiprocessing": False, "max_queue_size": 8}


def save_model(model, path_without_ext):
    """SavedModel directory on Keras 2, .keras file on Keras 3."""
    if keras_major() >= 3:
        p = path_without_ext + ".keras"
    else:
        p = path_without_ext + ".model"
    model.save(p)
    return p


def free_memory():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()


def env_record():
    import tensorflow as tf
    import platform
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "gpus": [g.name for g in gpus],
        "hostname": platform.node(),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def read_rgb(path):
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_array(img, size):
    return cv2.resize(img, (size, size)).astype(np.float32) / 255.0


def rotate_bound(image, angle):
    """Rotate onto an expanded canvas. Reproduces the original pipeline;
    introduces black corners for any angle that is not a multiple of 90."""
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nW / 2 - cX
    M[1, 2] += nH / 2 - cY
    return cv2.warpAffine(image, M, (nW, nH))


def rotate_inscribed(image, angle):
    """Rotate, then crop the largest axis-aligned rectangle containing no
    padding. Removes the black-corner artefact."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    a = np.deg2rad(angle % 180)
    if a > np.pi / 2:
        a = np.pi - a
    sin_a, cos_a = np.sin(a), np.cos(a)
    short, long_ = (w, h) if w <= h else (h, w)
    if short <= 2 * sin_a * cos_a * long_ or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short
        wr, hr = (x / sin_a, x / cos_a) if w <= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    wr, hr = max(int(wr), 1), max(int(hr), 1)
    x0, y0 = max(w // 2 - wr // 2, 0), max(h // 2 - hr // 2, 0)
    return rotated[y0:y0 + hr, x0:x0 + wr]


def to_uint8(img, size):
    return cv2.resize(img, (size, size))


def rotate_to_uint8(img, angle, size, mode):
    out = (rotate_bound if mode == "bound" else rotate_inscribed)(img, angle)
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]
    return to_uint8(out, size)


def augment_uint8(images, labels, n_rot, size, mode="bound", seed=0):
    """Rotated copies as one uint8 array, filled in shuffled order.

    One allocation, no second copy for shuffling. Training data only, always
    applied after the patient-level split.
    """
    angles = np.arange(n_rot) * (360.0 / n_rot)
    pairs = [(i, a) for i in range(len(images)) for a in angles]
    random.Random(seed).shuffle(pairs)
    X = np.empty((len(pairs), size, size, 3), dtype=np.uint8)
    y = np.empty(len(pairs), dtype=np.int64)
    for k, (i, ang) in enumerate(pairs):
        X[k] = rotate_to_uint8(images[i], ang, size, mode)
        y[k] = labels[i]
    return X, y


def stack_uint8(images, size):
    """Unaugmented images as uint8, for validation and evaluation."""
    X = np.empty((len(images), size, size, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        X[i] = to_uint8(img, size)
    return X


def estimate_gb(n_crops, n_rot, size):
    return n_crops * n_rot * size * size * 3 / 2 ** 30


def make_dataset(X_uint8, y_int, n_classes, batch_size, shuffle=False, seed=0):
    """tf.data pipeline that casts and normalises one batch at a time, so a
    float32 copy of the whole array never exists."""
    import tensorflow as tf
    y_onehot = np.eye(n_classes, dtype=np.float32)[y_int]
    ds = tf.data.Dataset.from_tensor_slices((X_uint8, y_onehot))
    if shuffle:
        ds = ds.shuffle(min(len(X_uint8), 4096), seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def rotation_sequence(images, labels, n_rot, size, n_classes, batch_size,
                      mode="bound", seed=0):
    """Keras Sequence that rotates on demand. Memory stays flat regardless of
    rotation count; the cost is CPU work per epoch, covered by SLURM workers."""
    import tensorflow as tf

    class _Seq(tf.keras.utils.Sequence):
        def __init__(self):
            angles = np.arange(n_rot) * (360.0 / n_rot)
            self.pairs = [(i, a) for i in range(len(images)) for a in angles]
            self.rng = random.Random(seed)
            self.eye = np.eye(n_classes, dtype=np.float32)
            self.rng.shuffle(self.pairs)

        def __len__(self):
            return math.ceil(len(self.pairs) / batch_size)

        def __getitem__(self, b):
            chunk = self.pairs[b * batch_size:(b + 1) * batch_size]
            X = np.empty((len(chunk), size, size, 3), dtype=np.float32)
            y = np.empty(len(chunk), dtype=np.int64)
            for k, (i, ang) in enumerate(chunk):
                X[k] = rotate_to_uint8(images[i], ang, size, mode)
                y[k] = labels[i]
            return X / 255.0, self.eye[y]

        def on_epoch_end(self):
            self.rng.shuffle(self.pairs)

    return _Seq()


def load_dataset(manifest_path, raw_dir, task):
    """Return (images, y, groups, files) with y indexed by the task's labels."""
    labels, folder_to_label = TASKS[task]
    idx = {name: i for i, name in enumerate(labels)}
    images, y, groups, files = [], [], [], []
    with open(manifest_path, newline="") as fh:
        for r in csv.DictReader(fh):
            img = read_rgb(os.path.join(raw_dir, r["file"]))
            if img is None:
                print("  WARNING unreadable, skipped:", r["file"])
                continue
            images.append(img)
            y.append(idx[folder_to_label[r["class"]]])
            groups.append(r["patient_group"])
            files.append(r["file"])
    return images, np.array(y), np.array(groups), files


# --------------------------------------------------------------------------- #
# patient-level splitting
# --------------------------------------------------------------------------- #
def grouped_folds(groups, n_splits, y=None, n_classes=None):
    """Assign whole patient groups to folds.

    Whole patients only, so no patient is ever split. When `y` is supplied the
    assignment also balances class counts across folds, which matters here
    because full-thickness crops are concentrated in a few patients: without it
    a fold can end up with no full-thickness case at all.

    Greedy, deterministic, largest patient first, choosing the fold that
    minimises squared deviation from the per-fold class target.
    """
    groups = np.asarray(groups)
    sizes = Counter(groups.tolist())
    order = sorted(sizes, key=lambda g: (-sizes[g], g))

    if y is None:
        fold_of, load = {}, [0] * n_splits
        for g in order:
            f = int(np.argmin(load))
            fold_of[g] = f
            load[f] += sizes[g]
        return fold_of

    y = np.asarray(y)
    k = n_classes or int(y.max()) + 1
    per_group = {g: np.bincount(y[groups == g], minlength=k).astype(float)
                 for g in order}
    total = np.bincount(y, minlength=k).astype(float)
    target = total / n_splits
    counts = np.zeros((n_splits, k), dtype=float)
    fold_of = {}
    for g in order:
        add = per_group[g]
        best, best_cost = 0, None
        for f in range(n_splits):
            trial = counts.copy()
            trial[f] += add
            # squared deviation from the class target, plus a light penalty on
            # overall fold-size imbalance to keep folds comparable
            cost = float(((trial - target) ** 2).sum()
                         + 0.5 * ((trial.sum(axis=1)
                                   - total.sum() / n_splits) ** 2).sum())
            if best_cost is None or cost < best_cost:
                best, best_cost = f, cost
        fold_of[g] = best
        counts[best] += add
    return fold_of


def holdout_groups(train_groups, sizes, frac, rng):
    """Pick whole patient groups totalling roughly `frac` of the images."""
    uniq = sorted(set(train_groups))
    rng.shuffle(uniq)
    target = frac * sum(sizes[g] for g in uniq)
    chosen, total = [], 0
    for g in uniq:
        if total >= target and chosen:
            break
        chosen.append(g)
        total += sizes[g]
    if len(chosen) == len(uniq):
        chosen.pop()
    return set(chosen)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def normalise_params(params):
    """Coerce numpy scalars to Python types.

    scikit-optimize returns numpy integers and floats. Keras 2 accepts them;
    Keras 3 rejects numpy ints for layer sizes. Coercing here keeps the same
    hyperparameter file working on either, and makes the JSON records clean.
    """
    out = {}
    for k, v in dict(params).items():
        if isinstance(v, np.bool_):
            out[k] = bool(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.str_):
            out[k] = str(v)
        else:
            out[k] = v
    for key in ("num_conv_layers", "num_dense_layers", "batch_size"):
        if key in out:
            out[key] = int(out[key])
    for k in list(out):
        if k.endswith("_filters") or k.endswith("_units"):
            out[k] = int(out[k])
        elif k.endswith("_dropout") or k in ("learning_rate", "global_dropout"):
            out[k] = float(out[k])
        elif k.endswith("_batchnorm") or k.endswith("_l2"):
            out[k] = bool(out[k])
    return out


def build_model(params, input_shape, n_classes, loss="categorical_crossentropy"):
    import tensorflow as tf
    params = normalise_params(params)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                         BatchNormalization, Flatten)
    m = Sequential()
    for i in range(params["num_conv_layers"]):
        n = i + 1
        kw = dict(
            filters=params[f"conv{n}_filters"],
            kernel_size=(3, 3),
            padding=params.get(f"conv{n}_padding", "same"),
            activation="relu",
            kernel_regularizer=(tf.keras.regularizers.l2(1e-4)
                                if params.get(f"conv{n}_l2") else None),
        )
        if i == 0:
            kw["input_shape"] = input_shape
        m.add(Conv2D(**kw))
        if params.get(f"conv{n}_batchnorm"):
            m.add(BatchNormalization())
        m.add(MaxPooling2D(pool_size=(2, 2)))
        m.add(Dropout(params["global_dropout"]))
    m.add(Flatten())
    for i in range(params["num_dense_layers"]):
        n = i + 1
        m.add(Dense(units=params[f"dense{n}_units"], activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        if params.get(f"dense{n}_batchnorm"):
            m.add(BatchNormalization())
        m.add(Dropout(params[f"dense{n}_dropout"]))
    m.add(Dense(n_classes, activation="softmax"))
    m.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=params["learning_rate"]),
        loss=loss, metrics=["accuracy"])
    return m


def flatten_features(params, img_size):
    """Width of the flattened layer feeding the first dense layer, following
    the conv/pool stack exactly (3x3 kernels, 2x2 pooling)."""
    d = int(img_size)
    for i in range(1, int(params["num_conv_layers"]) + 1):
        if params.get(f"conv{i}_padding", "same") == "valid":
            d -= 2
        d //= 2
        if d < 1:
            return 0
    return d * d * int(params[f"conv{int(params['num_conv_layers'])}_filters"])


def class_weights(y, n_classes):
    """Balanced weights with a key for EVERY class.

    Keras 2 rejects a class_weight dict whose keys are not exactly
    0..n_classes-1, which happens whenever a small inner fold is missing a
    class. Absent classes get weight 1.0 (they contribute no samples anyway).
    """
    cnt = Counter(np.asarray(y).tolist())
    present = sum(1 for c in range(n_classes) if cnt.get(c, 0))
    return {c: (len(y) / (max(present, 1) * cnt[c]) if cnt.get(c, 0) else 1.0)
            for c in range(n_classes)}


def max_conv_layers(img_size):
    n, d = 0, img_size
    while d > 1:
        d //= 2
        n += 1
    return max(n - 1, 3)


def fit_fold(params, train_images, train_y, val_images, val_y, n_classes,
             size, rotations, rotation_mode, epochs, patience, seed,
             use_class_weight=True, loader="auto", budget_gb=60.0,
             workers=None, verbose=2):
    """Train one model on one fold. Returns (model, history, loader, n_train).

    loader='memory'  materialise the rotations once as uint8
    loader='stream'  rotate per batch, flat memory
    loader='auto'    memory when the estimate fits in budget_gb, else stream
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    n_train = len(train_images) * rotations
    est = estimate_gb(len(train_images), rotations, size)
    used = ("memory" if est <= budget_gb else "stream") if loader == "auto" \
        else loader
    params = normalise_params(params)
    batch = int(params["batch_size"])

    Xva = stack_uint8(val_images, size)
    val_ds = make_dataset(Xva, val_y, n_classes, batch)

    Xtr = ytr = None
    if used == "memory":
        Xtr, ytr = augment_uint8(train_images, train_y, rotations, size,
                                 rotation_mode, seed)
        cw = class_weights(ytr, n_classes) if use_class_weight else None
        train_input = make_dataset(Xtr, ytr, n_classes, batch, shuffle=True,
                                   seed=seed)
        fit_kwargs = {}
    else:
        cw = (class_weights(np.repeat(train_y, rotations), n_classes)
              if use_class_weight else None)
        train_input = rotation_sequence(train_images, train_y, rotations, size,
                                        n_classes, batch, rotation_mode, seed)
        fit_kwargs = loader_fit_kwargs(workers)

    tf.random.set_seed(seed)
    model = build_model(params, (size, size, 3), n_classes)
    history = model.fit(
        train_input, validation_data=val_ds, epochs=epochs, class_weight=cw,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                 restore_best_weights=True)],
        verbose=verbose, **fit_kwargs)

    del Xtr, ytr, Xva, train_input, val_ds
    gc.collect()
    return model, history, used, n_train


def predict_uint8(model, images, size, batch=64):
    """Predict on unaugmented images without building a float32 array of the
    whole set."""
    X = stack_uint8(images, size)
    out = []
    for i in range(0, len(X), batch):
        out.append(model.predict(X[i:i + batch].astype(np.float32) / 255.0,
                                 verbose=0))
    del X
    gc.collect()
    return np.concatenate(out) if out else np.zeros((0, 1))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auc_score(y_bin, scores):
    """One-vs-rest AUC, rank formula with averaged ties."""
    pos = y_bin == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s, i = scores[order], 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_metrics(y_true, y_pred, probs, labels):
    """Macro figures average over classes PRESENT in y_true, so a fold that
    happens to contain no example of a class is not silently scored zero
    for it."""
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    present = [i for i in range(n) if (y_true == i).any()]
    out = {"accuracy": float((y_true == y_pred).mean()),
           "confusion_matrix": cm.tolist(), "per_class": {}}
    for i, name in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        absent = (tp + fn) == 0
        out["per_class"][name] = {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": float("nan") if absent else tp / (tp + fn),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "precision": float("nan") if (absent or not (tp + fp))
            else tp / (tp + fp),
            "f1": float("nan") if absent else (
                2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn)
                else float("nan")),
            "auc": float(auc_score((y_true == i).astype(int), probs[:, i])),
        }

    def macro(key):
        vals = [out["per_class"][labels[i]][key] for i in present]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    out["macro_f1"] = macro("f1")
    out["macro_auc"] = macro("auc")
    out["balanced_accuracy"] = macro("sensitivity")
    out["classes_present"] = [labels[i] for i in present]
    return out


def bootstrap_ci(y_true, y_pred, probs, groups, labels, n_boot=2000, seed=42):
    """Cluster bootstrap: resample PATIENTS with replacement."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(groups)))
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    keys = [(m, None) for m in
            ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")]
    for c in labels:
        for m in ("sensitivity", "specificity", "precision", "f1", "auc"):
            keys.append((m, c))
    draws = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        m = compute_metrics(y_true[idx], y_pred[idx], probs[idx], labels)
        for metric, cls in keys:
            v = m[metric] if cls is None else m["per_class"][cls][metric]
            if v == v:
                draws[(metric, cls)].append(v)
    ci = {}
    for (metric, cls), vals in draws.items():
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ci[metric if cls is None else f"{cls}.{metric}"] = [float(lo), float(hi)]
    return ci


def jsonable(d):
    """numpy scalars -> python scalars, so json.dump works."""
    if isinstance(d, dict):
        return {k: jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [jsonable(v) for v in d]
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


# --------------------------------------------------------------------------- #
# aggregation across fold files
# --------------------------------------------------------------------------- #
def aggregate_folds(fold_dir, files, y, groups, labels, n_boot=2000, seed=42):
    """Pool per-fold prediction files into one out-of-fold result set.

    Each fold is written by 03_run_grouped_cv.py as folds/fold_<k>.json. Folds
    may be produced by separate SLURM array tasks, so this is the only place
    that needs them all present.
    """
    import json
    index = {f: i for i, f in enumerate(files)}
    oof = np.zeros((len(files), len(labels)), dtype=np.float64)
    seen = np.zeros(len(files), dtype=bool)
    fold_idx = np.full(len(files), -1, dtype=int)
    rows, histories, params, settings = [], {}, {}, []
    VARIANTS = ["probs_single_model", "probs_no_tta", "probs_uncalibrated"]
    var_arr = {v: np.zeros((len(files), len(labels))) for v in VARIANTS}
    var_seen = {v: np.zeros(len(files), dtype=bool) for v in VARIANTS}
    temps = []
    n_members = []

    paths = sorted(
        (os.path.join(fold_dir, p) for p in os.listdir(fold_dir)
         if p.startswith("fold_") and p.endswith(".json")),
        key=lambda p: int(os.path.basename(p)[5:-5]))
    if not paths:
        raise SystemExit(f"No fold_*.json files in {fold_dir}")

    for path in paths:
        d = json.load(open(path))
        k = d["fold"]
        for v in VARIANTS:
            if v in d:
                for f_, pr_ in zip(d["files"], d[v]):
                    var_arr[v][index[f_]] = pr_
                    var_seen[v][index[f_]] = True
        if "temperature" in d:
            temps.append(float(d["temperature"]))
        if d.get("ensemble_members"):
            n_members.append(int(d["ensemble_members"]))
        for f, pr in zip(d["files"], d["probs"]):
            i = index[f]
            if seen[i]:
                raise SystemExit(
                    f"{f} appears in more than one fold file. Delete "
                    f"{fold_dir} and rerun; a stale fold is present.")
            oof[i] = pr
            seen[i] = True
            fold_idx[i] = k
        settings.append(d.get("settings", {}))
        rows.append({key: d[key] for key in
                     ("fold", "held_out_patients", "n_patients", "n_crops",
                      "accuracy", "epochs_run", "loader",
                      "augmented_training_images") if key in d})
        histories[k] = d.get("history", {})
        params[k] = d.get("hyperparameters", {})

    missing = int((~seen).sum())
    if missing:
        have = sorted({r["fold"] for r in rows})
        raise SystemExit(
            f"{missing} of {len(files)} crops have no prediction. Folds "
            f"present: {have}. Wait for the remaining array tasks, or rerun "
            f"them, before aggregating.")

    # folds must have been produced with the same settings, or pooling them
    # would mix experiments
    comparable = [{k: v for k, v in s.items() if k != "environment"}
                  for s in settings if s]
    if comparable and any(c != comparable[0] for c in comparable[1:]):
        differing = sorted({k for c in comparable[1:]
                            for k in comparable[0]
                            if c.get(k) != comparable[0].get(k)})
        raise SystemExit(
            "Fold files were produced with different settings "
            f"({', '.join(differing)}). Delete {fold_dir} and rerun so every "
            "fold shares one configuration.")

    y_pred = oof.argmax(axis=1)
    metrics = compute_metrics(y, y_pred, oof, labels)
    metrics["ci95"] = bootstrap_ci(y, y_pred, oof, groups, labels, n_boot, seed)
    acc = [r["accuracy"] for r in rows]
    metrics["per_fold"] = rows
    metrics["fold_accuracy_mean"] = float(np.mean(acc))
    metrics["fold_accuracy_sd"] = float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0
    metrics["fold_hyperparameters"] = params
    if temps:
        metrics["temperatures"] = temps

    # ablation ladder: each step adds one component, so the supplement can
    # show what each contributes. Accuracy is identical for the last two
    # rows because temperature scaling never changes the predicted class.
    k = (f"{min(n_members)}-member" if n_members and min(n_members) ==
         max(n_members) else "cross-validation")
    ladder = [("single model + TTA", "probs_single_model"),
              (f"{k} ensemble, no TTA", "probs_no_tta"),
              (f"{k} ensemble + TTA", "probs_uncalibrated")]
    ablation = []
    for title, key in ladder:
        if var_seen[key].all():
            P = var_arr[key]
            mm = compute_metrics(y, P.argmax(1), P, labels)
            cal = calibration_metrics(P, y, labels)
            ablation.append({"variant": title, "accuracy": mm["accuracy"],
                             "balanced_accuracy": mm["balanced_accuracy"],
                             "macro_f1": mm["macro_f1"],
                             "macro_auc": mm["macro_auc"],
                             "brier": cal["brier"], "ece": cal["ece"],
                             "high_confidence_wrong_rate":
                                 cal["high_confidence_wrong_rate"]})
    if ablation:
        cal = calibration_metrics(oof, y, labels)
        ablation.append({"variant": f"{k} ensemble + TTA + temperature "
                                    "scaling (primary)",
                         "accuracy": metrics["accuracy"],
                         "balanced_accuracy": metrics["balanced_accuracy"],
                         "macro_f1": metrics["macro_f1"],
                         "macro_auc": metrics["macro_auc"],
                         "brier": cal["brier"], "ece": cal["ece"],
                         "high_confidence_wrong_rate":
                             cal["high_confidence_wrong_rate"]})
        metrics["ablation"] = ablation
    # keep the environment of the first fold: it is excluded from the
    # comparison above (hostnames and job ids legitimately differ between
    # array tasks) but the Methods needs the software versions
    keep = dict(comparable[0]) if comparable else {}
    envs = [s.get("environment") for s in settings if s.get("environment")]
    if envs:
        keep["environment"] = envs[0]
        hosts = sorted({e.get("hostname") for e in envs if e.get("hostname")})
        jobs = sorted({str(e.get("slurm_job_id")) for e in envs
                       if e.get("slurm_job_id")})
        keep["environment_nodes"] = hosts
        keep["slurm_job_ids"] = jobs
        tfv = sorted({e.get("tensorflow") for e in envs if e.get("tensorflow")})
        if len(tfv) > 1:
            raise SystemExit(
                f"Folds were run under different TensorFlow versions ({tfv}). "
                "Rerun them under one environment before aggregating.")
    metrics["_settings"] = keep
    return metrics, oof, y_pred, fold_idx, histories


def print_ablation(metrics):
    ab = metrics.get("ablation")
    if not ab:
        return
    print("\n  ablation (each row adds one component)")
    print(f"    {'variant':<58}{'acc':>7}{'mAUC':>7}{'Brier':>7}{'ECE':>7}")
    for r in ab:
        print(f"    {r['variant']:<58}{r['accuracy']:>7.3f}"
              f"{r['macro_auc']:>7.3f}{r['brier']:>7.3f}{r['ece']:>7.3f}")
    t = metrics.get("temperatures")
    if t:
        print(f"    temperatures by fold: "
              + ", ".join(f"{v:.2f}" for v in t))
        stuck = [i + 1 for i, v in enumerate(t) if v <= 0.26 or v >= 9.9]
        if stuck:
            print(f"    WARNING: temperature at its search bound in fold(s) "
                  f"{stuck}. Expected for a smoke test with random weights; "
                  "on a real run it means the model is uninformative there.")


def print_results(metrics, labels, title):
    ci = metrics.get("ci95", {})
    line = "=" * 70
    print("\n" + line)
    print(f"POOLED OUT-OF-FOLD RESULTS  ({title})")
    print(line)
    for name, key in [("accuracy", "accuracy"),
                      ("balanced accuracy", "balanced_accuracy"),
                      ("macro F1", "macro_f1"), ("macro AUC", "macro_auc")]:
        v, c = metrics[key], ci.get(key)
        s = f"  {name:<20} {v:.3f}"
        if c:
            s += f"  95% CI {c[0]:.3f} to {c[1]:.3f}"
        print(s)
    print(f"  {'per-fold accuracy':<20} {metrics['fold_accuracy_mean']:.3f} "
          f"(SD {metrics['fold_accuracy_sd']:.3f})")
    print("\n  confusion matrix (rows true, columns predicted)")
    print("  " + " " * 24 + "".join(f"{c[:14]:>16}" for c in labels))
    for i, c in enumerate(labels):
        print(f"  {c:<24}" + "".join(f"{v:>16}"
                                     for v in metrics["confusion_matrix"][i]))
    print("\n  per class, 95% CI from patient-level bootstrap")
    for c in labels:
        p = metrics["per_class"][c]
        def g(m):
            v = ci.get(f"{c}.{m}")
            return f"[{v[0]:.2f}, {v[1]:.2f}]" if v else "[n/a]"
        print(f"    {c}  (n={p['TP'] + p['FN']})")
        print(f"      sensitivity {p['sensitivity']:.3f} {g('sensitivity')}"
              f"   specificity {p['specificity']:.3f} {g('specificity')}")
        print(f"      precision   {p['precision']:.3f} {g('precision')}"
              f"   AUC         {p['auc']:.3f} {g('auc')}")


# =========================================================================== #
# TRANSFER LEARNING
# =========================================================================== #
def _augment_op(img, size, photometric):
    """Online augmentation, applied only to training data, after the split.

    Replaces the original 24 precomputed rotations: more diverse per epoch,
    no near-duplicate inflation, and no black corners because the image is
    scaled up before a random crop.

    Colour jitter is deliberately gentle. Colour is the diagnostic signal for
    burn depth (pale full-thickness versus pink blanching partial-thickness),
    so strong hue or saturation jitter can erase the very feature that
    separates the classes the model most often confuses.
    """
    import tensorflow as tf
    x = tf.cast(img, tf.float32) / 255.0
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, tf.random.uniform([], 0, 4, dtype=tf.int32))
    if photometric:
        x = tf.image.random_brightness(x, 0.10)
        x = tf.image.random_contrast(x, 0.90, 1.10)
        x = tf.image.random_saturation(x, 0.90, 1.10)
        x = tf.image.random_hue(x, 0.02)
        x = tf.clip_by_value(x, 0.0, 1.0)
    big = tf.cast(tf.round(size * 1.15), tf.int32)
    x = tf.image.resize(x, (big, big))
    x = tf.image.random_crop(x, (size, size, 3))
    return x * 255.0


def make_tf_dataset(X_uint8, y_int, n_classes, batch_size, size, preprocess,
                    training=False, seed=0, photometric=True, passes=1):
    """uint8 in, preprocessed batches out. The float32 copy of the whole array
    never exists; augmentation happens per batch on the fly."""
    import tensorflow as tf
    y = np.eye(n_classes, dtype=np.float32)[y_int]
    ds = tf.data.Dataset.from_tensor_slices((X_uint8, y))
    if training:
        ds = ds.shuffle(min(len(X_uint8), 4096), seed=seed,
                        reshuffle_each_iteration=True)
        # each epoch = `passes` independently augmented passes over the crops,
        # so the optimiser gets enough steps from ~160 training images
        ds = ds.repeat(max(1, int(passes)))
        ds = ds.map(lambda a, b: (_augment_op(a, size, photometric), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda a, b: (preprocess(a), b),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_transfer_model(name, size, n_classes, dropout=0.3, lr=1e-3,
                         label_smoothing=0.05, weights="imagenet"):
    """Frozen backbone plus a small head (the linear-probe phase).

    The backbone is called with training=False so its BatchNormalization
    layers always use their stored ImageNet statistics. This is the single
    most important detail: with batches of 16-32 small, patient-correlated
    crops, letting BatchNorm recompute batch statistics destroys the
    pretrained features. Keras documents this explicitly.
    """
    import tensorflow as tf
    import backbones as B
    base = B.build_backbone(name, (size, size, 3), weights=weights)
    base.trainable = False
    inputs = tf.keras.Input(shape=(size, size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, out, name=f"{name}_head")
    model._burn_base = base                       # kept for the unfreeze step
    compile_model(model, lr, label_smoothing)
    return model


def compile_model(model, lr, label_smoothing=0.05, clipnorm=1.0):
    import tensorflow as tf
    try:
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    except TypeError:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing),
        metrics=["accuracy"])


def cosine_schedule(lr, total_steps):
    """Cosine decay to zero over the fine-tuning phase (SGDR-style, no
    warm-up so that it behaves identically on TF 2.6 through 2.15)."""
    import tensorflow as tf
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=max(1, int(total_steps)),
        alpha=0.0)


def unfreeze_top(model, n_layers, lr, label_smoothing=0.05):
    """Unfreeze the last n_layers of the backbone, BatchNorm always excluded."""
    import tensorflow as tf
    base = getattr(model, "_burn_base", None)
    if base is None:
        return 0
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    frozen_bn = 0
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
    compile_model(model, lr, label_smoothing)
    return sum(1 for l in base.layers if l.trainable)


def fit_transfer(name, train_images, train_y, val_images, val_y, n_classes,
                 size, seed, probe_epochs=15, finetune_epochs=30,
                 probe_lr=1e-3, finetune_lr=1e-5, dropout=0.3,
                 batch_size=16, label_smoothing=0.05, photometric=True,
                 unfreeze=None, verbose=2, weights="imagenet",
                 passes_per_epoch=4, cosine=True):
    """Linear probe, then fine-tune the top block (LP-FT).

    Fixed epoch budgets, no early stopping. With only 2-4 patients in the
    inner validation set, a validation loss curve is dominated by patient
    identity rather than generalisation, and early stopping on it selects
    epoch 1-5 essentially at random. A fixed budget is reproducible and
    removes that lottery. The validation set here is monitored only.

    Returns (model, history, n_trainable_layers, optimiser_steps).
    """
    import tensorflow as tf
    import backbones as B

    sp = B.spec(name)
    pre = B.preprocess_fn(name)
    n_unfreeze = unfreeze if unfreeze is not None else sp["unfreeze_layers"]

    tf.random.set_seed(seed)              # before the pipeline is built
    Xtr = stack_uint8(train_images, size)
    Xva = stack_uint8(val_images, size)
    tr = make_tf_dataset(Xtr, train_y, n_classes, batch_size, size, pre,
                         training=True, seed=seed, photometric=photometric,
                         passes=passes_per_epoch)
    va = make_tf_dataset(Xva, val_y, n_classes, batch_size, size, pre)
    steps_per_epoch = int(np.ceil(len(Xtr) * max(1, passes_per_epoch)
                                  / batch_size))

    model = build_transfer_model(name, size, n_classes, dropout, probe_lr,
                                 label_smoothing, weights=weights)
    h1 = model.fit(tr, validation_data=va, epochs=probe_epochs, verbose=verbose)

    ft_steps = steps_per_epoch * finetune_epochs
    lr2 = cosine_schedule(finetune_lr, ft_steps) if cosine else finetune_lr
    n_trainable = unfreeze_top(model, n_unfreeze, lr2, label_smoothing)
    if verbose:
        print(f"    fine-tuning {n_trainable} backbone layers "
              f"(BatchNorm frozen) at lr {finetune_lr:g}"
              + (" with cosine decay" if cosine else "")
              + f", {ft_steps} steps")
    h2 = model.fit(tr, validation_data=va, epochs=finetune_epochs,
                   verbose=verbose)

    history = {f"probe_{k}": [float(x) for x in v] for k, v in h1.history.items()}
    history.update({f"ft_{k}": [float(x) for x in v]
                    for k, v in h2.history.items()})
    total_steps = steps_per_epoch * (probe_epochs + finetune_epochs)
    del Xtr, Xva, tr, va
    gc.collect()
    return model, history, n_trainable, total_steps


DIHEDRAL = [(k, f) for k in range(4) for f in (False, True)]


def predict_tta(model, images, size, preprocess, n_tta=8, batch=32):
    """Average predictions over up to 8 dihedral transforms.

    Report results with and without TTA: naive TTA can flip correct
    predictions, so it should not be assumed beneficial.
    """
    X = stack_uint8(images, size)
    acc = None
    used = 0
    for k, flip in DIHEDRAL[:max(1, n_tta)]:
        Xt = np.rot90(X, k, axes=(1, 2))
        if flip:
            Xt = Xt[:, :, ::-1, :]
        Xt = np.ascontiguousarray(Xt)
        out = []
        for i in range(0, len(Xt), batch):
            xb = preprocess(Xt[i:i + batch].astype(np.float32))
            out.append(model.predict(xb, verbose=0))
        p = np.concatenate(out)
        acc = p if acc is None else acc + p
        used += 1
    del X
    gc.collect()
    return acc / used


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def fit_temperature(probs, y_true, lo=0.25, hi=10.0, iters=80):
    """One scalar T minimising NLL. Preserves the argmax, so accuracy is
    unchanged; only confidence is corrected. Fitted on inner out-of-fold
    predictions so the outer test patients are never involved."""
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    idx = np.arange(len(y_true))

    def nll(T):
        z = logp / max(T, 1e-6)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.clip(p[idx, y_true], 1e-12, 1.0))))

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(d)
        if abs(b - a) < 1e-4:
            break
    return float((a + b) / 2.0)


def apply_temperature(probs, T):
    z = np.log(np.clip(probs, 1e-12, 1.0)) / max(T, 1e-6)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def brier_score(probs, y_true, n_classes):
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(probs, y_true, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    order = np.argsort(conf)
    bins = np.array_split(order, n_bins)
    ece = mce = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        gap = abs(correct[b].mean() - conf[b].mean())
        ece += len(b) / len(conf) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def calibration_slope_intercept(p, y_bin):
    """Logistic recalibration of one class versus the rest.

    Perfect calibration is intercept 0 and slope 1. Slope below 1 means
    over-confident predictions, which is the usual failure of a small deep
    model. Reported because TRIPOD+AI expects calibration alongside
    discrimination, and clinical reviewers read slope and intercept rather
    than ECE.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(60):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        grad = X.T @ (y_bin - mu)
        H = (X * W[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def calibration_metrics(probs, y_true, labels):
    n = len(labels)
    ece, mce = expected_calibration_error(probs, y_true)
    out = {"brier": brier_score(probs, y_true, n), "ece": ece, "mce": mce,
           "per_class": {}}
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    hi = conf > 0.9
    out["high_confidence_n"] = int(hi.sum())
    out["high_confidence_wrong"] = int((hi & ~correct).sum())
    out["high_confidence_wrong_rate"] = (
        float((hi & ~correct).sum() / hi.sum()) if hi.sum() else float("nan"))
    for i, c in enumerate(labels):
        a, b = calibration_slope_intercept(probs[:, i],
                                           (y_true == i).astype(float))
        out["per_class"][c] = {"calibration_intercept": a,
                               "calibration_slope": b}
    return out


# =========================================================================== #
# EXTENSION: random-initialisation arm, repeats, learning curves, statistics
# =========================================================================== #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def grouped_folds_seeded(groups, n_splits, y, n_classes, seed, tries=200):
    """Class-balanced patient-grouped folds with a RANDOMISED assignment, for
    repeated cross-validation. seed=None reproduces the original split exactly.

    Patients are processed in a seeded random order and each is placed in the
    fold that best balances class counts. The first assignment in which every
    fold contains every class is returned.
    """
    if seed is None:
        return grouped_folds(groups, n_splits, y, n_classes)
    groups = np.asarray(groups)
    y = np.asarray(y)
    k = n_classes
    uniq = sorted(set(groups.tolist()))
    per_group = {g: np.bincount(y[groups == g], minlength=k).astype(float)
                 for g in uniq}
    total = np.bincount(y, minlength=k).astype(float)
    target = total / n_splits
    rng = random.Random(seed)
    for _ in range(tries):
        order = uniq[:]
        rng.shuffle(order)
        counts = np.zeros((n_splits, k))
        fold_of = {}
        for g in order:
            add = per_group[g]
            costs = []
            for f in range(n_splits):
                trial = counts.copy()
                trial[f] += add
                costs.append(float(((trial - target) ** 2).sum()
                                   + 0.5 * ((trial.sum(1) - total.sum() / n_splits) ** 2).sum()
                                   + 1e-9 * rng.random()))
            best = int(np.argmin(costs))
            fold_of[g] = best
            counts[best] += add
        if (counts > 0).all():
            return fold_of
    raise SystemExit(f"Could not find a class-complete fold assignment for seed {seed}")


class WarmupCosine:
    """Linear warm-up then cosine decay to zero. A plain LearningRateSchedule
    so it behaves identically on TF 2.6 through 2.15."""

    def __new__(cls, peak_lr, warmup_steps, total_steps):
        import tensorflow as tf

        class _S(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, peak, warm, total):
                self.peak, self.warm, self.total = float(peak), float(max(1, warm)), float(max(2, total))

            def __call__(self, step):
                s = tf.cast(step, tf.float32)
                warm = self.peak * s / self.warm
                prog = tf.clip_by_value((s - self.warm) / (self.total - self.warm), 0.0, 1.0)
                cos = 0.5 * self.peak * (1.0 + tf.cos(np.pi * prog))
                return tf.where(s < self.warm, warm, cos)

            def get_config(self):
                return {"peak": self.peak, "warm": self.warm, "total": self.total}

        return _S(peak_lr, warmup_steps, total_steps)


def set_imagenet_standardisation(base):
    """With weights=None Keras leaves EfficientNet's Normalization layer as an
    identity, so inputs would reach the stem as x/255 while the pretrained arm
    receives ImageNet-standardised inputs. Setting the layer to the ImageNet
    mean and variance makes the input transform identical in both arms, so
    initialisation is the only difference. Returns True if a layer was set."""
    import tensorflow as tf
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.Normalization):
            w = layer.get_weights()
            if len(w) == 3:
                layer.set_weights([np.array(IMAGENET_MEAN, dtype=w[0].dtype).reshape(w[0].shape),
                                   np.array(IMAGENET_STD, dtype=w[1].dtype).reshape(w[1].shape) ** 2,
                                   w[2]])
                return True
    return False


def stem_input(base, x):
    """Output of the input-standardisation stage (just before the stem conv)."""
    import tensorflow as tf
    names = [l.name for l in base.layers]
    stop = next((n for n in names if n.startswith("stem_conv")), None)
    if stop is None:
        return None
    prev = base.layers[names.index(stop) - 1]
    sub = tf.keras.Model(base.input, prev.output)
    return sub(x, training=False).numpy()


def build_randominit_model(name, size, n_classes, dropout=0.3, label_smoothing=0.05):
    """Same architecture and head as the pretrained arm, randomly initialised,
    every layer trainable, and the backbone called WITHOUT training=False so
    BatchNorm learns its statistics (random-init statistics are meaningless)."""
    import tensorflow as tf
    import backbones as B
    base = B.build_backbone(name, (size, size, 3), weights=None)
    standardised = set_imagenet_standardisation(base)
    base.trainable = True
    inputs = tf.keras.Input(shape=(size, size, 3))
    x = base(inputs)                                   # BatchNorm in training mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, out, name=f"{name}_randominit")
    model._burn_base = base
    model._burn_standardised = standardised
    return model


def fit_randominit(name, train_images, train_y, val_images, val_y, n_classes,
                   size, seed, lr=1e-3, epochs=150, warmup_epochs=5,
                   weight_decay=1e-4, dropout=0.3, batch_size=16,
                   label_smoothing=0.05, photometric=True, passes_per_epoch=4,
                   verbose=2):
    """Full training from random initialisation: AdamW, linear warm-up then
    cosine decay, fixed budget, no early stopping. Everything not forced by
    the initialisation (augmentation, label smoothing, batch, head) is the
    same as the pretrained arm."""
    import tensorflow as tf
    import backbones as B
    pre = B.preprocess_fn(name)
    tf.random.set_seed(seed)
    Xtr = stack_uint8(train_images, size)
    Xva = stack_uint8(val_images, size)
    tr = make_tf_dataset(Xtr, train_y, n_classes, batch_size, size, pre,
                         training=True, seed=seed, photometric=photometric,
                         passes=passes_per_epoch)
    va = make_tf_dataset(Xva, val_y, n_classes, batch_size, size, pre)
    steps_per_epoch = int(np.ceil(len(Xtr) * max(1, passes_per_epoch) / batch_size))
    total = steps_per_epoch * epochs
    model = build_randominit_model(name, size, n_classes, dropout, label_smoothing)
    sched = WarmupCosine(lr, steps_per_epoch * warmup_epochs, total)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=sched,
                                            weight_decay=weight_decay,
                                            clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"])
    h = model.fit(tr, validation_data=va, epochs=epochs, verbose=verbose)
    hist = {k: [float(x) for x in v] for k, v in h.history.items()}
    del Xtr, Xva, tr, va
    gc.collect()
    return model, hist, int(total)


def save_member(model, path):
    """Archive one ensemble member so post hoc analyses (saliency, external
    scoring) can reuse it without retraining."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return save_model(model, path)


def cross_entropy(probs, y):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))


# --------------------------------------------------------------------------- #
# clustered AUC (Obuchowski 1997, Biometrics 53:567-578)
# --------------------------------------------------------------------------- #
def obuchowski_auc(y_bin, scores, groups):
    """Nonparametric AUC with a cluster-robust variance. Each patient is a
    cluster that may contribute positives, negatives, or both."""
    y_bin = np.asarray(y_bin).astype(int)
    s = np.asarray(scores, dtype=float)
    g = np.asarray(groups)
    pos, neg = s[y_bin == 1], s[y_bin == 0]
    M, N = len(pos), len(neg)
    if M == 0 or N == 0:
        return {"auc": float("nan"), "se": float("nan"), "ci95": [float("nan")] * 2}
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    theta = psi.mean()
    v10 = psi.mean(axis=1)                 # per positive
    v01 = psi.mean(axis=0)                 # per negative
    gp, gn = g[y_bin == 1], g[y_bin == 0]
    clusters = sorted(set(g.tolist()))
    I = len(clusters)
    V10 = np.array([v10[gp == c].sum() for c in clusters])
    V01 = np.array([v01[gn == c].sum() for c in clusters])
    m = np.array([(gp == c).sum() for c in clusters])
    n = np.array([(gn == c).sum() for c in clusters])
    S10 = I / ((I - 1) * M) * np.sum((V10 - m * theta) ** 2)
    S01 = I / ((I - 1) * N) * np.sum((V01 - n * theta) ** 2)
    S11 = I / (I - 1) * np.sum((V10 - m * theta) * (V01 - n * theta))
    var = S10 / M + S01 / N + 2 * S11 / (M * N)
    se = float(np.sqrt(max(var, 0.0)))
    return {"auc": float(theta), "se": se,
            "ci95": [float(max(0.0, theta - 1.96 * se)), float(min(1.0, theta + 1.96 * se))],
            "clusters": I, "positives": M, "negatives": N}


# --------------------------------------------------------------------------- #
# selective prediction
# --------------------------------------------------------------------------- #
def risk_coverage(probs, y):
    """Error rate as the least confident predictions are deferred. Returns
    coverages, risks and the area under the risk-coverage curve (AURC)."""
    conf = probs.max(axis=1)
    order = np.argsort(-conf, kind="mergesort")
    wrong = (probs.argmax(1) != y)[order].astype(float)
    n = len(y)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(wrong) / np.arange(1, n + 1)
    # np.trapezoid (NumPy >= 2.0) replaced np.trapz, which NumPy 2.4 removed
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return cov, risk, float(integrate(risk, cov) if n > 1 else risk[0])


def keep_at_coverage(probs, coverage):
    """Boolean mask of the most confident `coverage` fraction of cases."""
    conf = probs.max(axis=1)
    k = max(1, int(round(coverage * len(conf))))
    order = np.argsort(-conf, kind="mergesort")
    keep = np.zeros(len(conf), dtype=bool)
    keep[order[:k]] = True
    return keep


def bca_interval(stat_fn, groups, n_boot=2000, seed=42, alpha=0.05):
    """Bias-corrected and accelerated bootstrap CI, resampling patients.
    stat_fn(index_array) -> float. Acceleration from the patient jackknife."""
    from statistics import NormalDist
    nd = NormalDist()
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    idx = {p: np.where(groups == p)[0] for p in uniq}
    full = np.concatenate([idx[p] for p in uniq])
    theta = stat_fn(full)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = stat_fn(np.concatenate([idx[p] for p in pick]))
        if v == v:
            boots.append(v)
    boots = np.array(boots)
    if len(boots) < 10:
        return {"estimate": theta, "ci95": [float("nan")] * 2}
    prop = np.clip(np.mean(boots < theta), 1e-6, 1 - 1e-6)
    z0 = nd.inv_cdf(prop)
    jack = np.array([stat_fn(np.concatenate([idx[q] for q in uniq if q != p])) for p in uniq])
    jack = jack[jack == jack]
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / den if den > 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = nd.inv_cdf(q)
        adj = nd.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(float(np.percentile(boots, 100 * np.clip(adj, 0, 1))))
    return {"estimate": float(theta), "ci95": out,
            "percentile_ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
            "bootstrap_p_two_sided": float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))),
            "z0": float(z0), "acceleration": float(a)}
