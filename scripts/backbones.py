"""
backbones.py — the pre-specified set of transfer-learning backbones.

Every entry records the three things that are easy to get silently wrong:

  * the correct `preprocess_input`, because each family expects a different
    input convention and double-normalising is a silent accuracy killer;
  * the native input resolution;
  * how many trailing layers to unfreeze for the fine-tuning phase.

Availability differs by TensorFlow version, so `available()` checks rather than
assumes:

    EfficientNetB0 / DenseNet121 / ResNet50V2 / MobileNetV2 / InceptionV3
        present since well before TF 2.6
    EfficientNetV2*      TF 2.8+
    ConvNeXt*            TF 2.10+
    ResNetRS* / RegNet*  TF 2.9+, but REMOVED again in Keras 3 (TF 2.16+)

The 'scratch' entry is the original custom CNN, kept as the comparison
baseline so the paper can quantify what transfer learning actually buys.
"""

BACKBONES = {
    # --- available on every TensorFlow from 2.6 onwards ------------------- #
    "efficientnetb0": {
        "keras_name": "EfficientNetB0",
        "module": "efficientnet",
        "size": 224,
        "unfreeze_layers": 20,
        "params_m": 5.3,
        "note": "Primary model. Best accuracy-per-parameter at small n; "
                "224 px matches the median crop size.",
    },
    "densenet121": {
        "keras_name": "DenseNet121",
        "module": "densenet",
        "size": 224,
        "unfreeze_layers": 32,
        "params_m": 8.1,
        "note": "The most-cited medical-imaging transfer backbone.",
    },
    "resnet50v2": {
        "keras_name": "ResNet50V2",
        "module": "resnet_v2",
        "size": 224,
        "unfreeze_layers": 22,
        "params_m": 25.6,
        "note": "Comparability with the burn literature, which mostly used "
                "ResNet50.",
    },
    "mobilenetv2": {
        "keras_name": "MobileNetV2",
        "module": "mobilenet_v2",
        "size": 224,
        "unfreeze_layers": 20,
        "params_m": 3.5,
        "note": "Deployment-relevant: small enough for a phone, which is the "
                "realistic clinical form factor for burn triage.",
    },
    "inceptionv3": {
        "keras_name": "InceptionV3",
        "module": "inception_v3",
        "size": 299,
        "unfreeze_layers": 30,
        "params_m": 23.9,
        "note": "Optional. Larger native input; slower.",
    },

    # --- TF 2.8+ ---------------------------------------------------------- #
    "efficientnetv2b0": {
        "keras_name": "EfficientNetV2B0",
        "module": "efficientnet_v2",
        "size": 224,
        "unfreeze_layers": 24,
        "params_m": 7.2,
        "note": "TF 2.8+. Improved training recipe over V1.",
    },
    "efficientnetv2s": {
        "keras_name": "EfficientNetV2S",
        "module": "efficientnet_v2",
        "size": 384,
        "unfreeze_layers": 30,
        "params_m": 21.6,
        "note": "TF 2.8+. Strong but heavy; 384 px upscales small crops a lot.",
    },

    # --- TF 2.10+ (and still present in Keras 3) --------------------------- #
    "convnexttiny": {
        "keras_name": "ConvNeXtTiny",
        "module": "convnext",
        "size": 224,
        "unfreeze_layers": 20,
        "params_m": 28.6,
        "note": "TF 2.10+. Modern convolutional design; benefits most from "
                "larger datasets, so a useful upper-bound check.",
    },

    # --- the original model, as the baseline ------------------------------- #
    "scratch": {
        "keras_name": None,
        "module": None,
        "size": 100,
        "unfreeze_layers": None,
        "params_m": None,
        "note": "The original custom CNN trained from random initialisation. "
                "Baseline for quantifying the benefit of transfer learning.",
    },
}

# The pre-specified comparison set. Keep this list short and fixed before
# running anything: a benchmark is defensible, a fishing expedition is not.
PRIMARY = "efficientnetb0"
PRESPECIFIED = ["efficientnetb0", "densenet121", "resnet50v2", "mobilenetv2",
                "scratch"]


def spec(name):
    key = name.lower()
    if key not in BACKBONES:
        raise SystemExit(
            f"Unknown backbone '{name}'. Available: {', '.join(sorted(BACKBONES))}")
    return BACKBONES[key]


def available(name):
    """True if this TensorFlow actually ships the backbone."""
    s = spec(name)
    if s["keras_name"] is None:
        return True                       # scratch is always available
    import tensorflow as tf
    return hasattr(tf.keras.applications, s["keras_name"])


def list_available():
    return [n for n in BACKBONES if available(n)]


def preprocess_fn(name):
    """The family's own preprocess_input.

    Every Keras preprocess_input expects float input in [0, 255]; what it does
    with it differs by family (identity for EfficientNet and ConvNeXt, which
    normalise internally; [-1, 1] for ResNetV2, MobileNetV2 and InceptionV3;
    ImageNet mean/std for DenseNet). Always routing through the family's own
    function is what prevents double-normalisation.
    """
    s = spec(name)
    if s["module"] is None:
        return lambda x: x / 255.0        # the scratch CNN used [0, 1]
    import tensorflow as tf
    mod = getattr(tf.keras.applications, s["module"], None)
    if mod is None:
        import importlib
        mod = importlib.import_module(
            f"tensorflow.keras.applications.{s['module']}")
    return mod.preprocess_input


def build_backbone(name, input_shape, weights="imagenet"):
    """Instantiate the backbone.

    weights="imagenet" requires the weight file to be present in
    ~/.keras/models. Compute nodes usually have no internet access, so run
    00b_fetch_weights.py on a login node first. weights=None is for smoke
    tests only and must never be used for a reported result.
    """
    import tensorflow as tf
    s = spec(name)
    cls = getattr(tf.keras.applications, s["keras_name"])
    return cls(include_top=False, weights=weights, input_shape=input_shape)


def describe(name):
    s = spec(name)
    return (f"{name}: {s['keras_name'] or 'custom CNN'} | {s['size']}px | "
            f"{s['params_m'] or '-'}M params | {s['note']}")
