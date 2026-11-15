"""Load the TFLite model and run predictions (handles INT8 input/output)."""
import json
import numpy as np
from PIL import Image, ImageOps

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter, OpResolverType
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter, OpResolverType

IMG_SIZE = (224, 224)


def load_interpreter(model_path: str):
    """Some runtimes fail to prepare the XNNPACK accelerator for INT8 models
    ('failed to prepare'). If that happens, retry with plain CPU kernels."""
    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except RuntimeError:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
    return interpreter


def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL image -> float32 array (224, 224, 3) with values 0-255.

    Centre-crops to a square first so a 4:3 phone photo is not squashed (the training pipeline
    does the same). The model contains its own preprocessing, so we must NOT divide by 255.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    image = image.crop((left, top, left + s, top + s)).resize(IMG_SIZE, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def predict(interpreter, img_array: np.ndarray) -> np.ndarray:
    """Return class probabilities (float32, sums to ~1)."""
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    x = np.expand_dims(img_array, 0).astype(np.float32)
    if inp["dtype"] == np.int8:
        scale, zero = inp["quantization"]
        x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)

    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])[0]

    if out["dtype"] == np.int8:
        scale, zero = out["quantization"]
        y = (y.astype(np.float32) - zero) * scale
    return y.astype(np.float32)

def predict_tta(interpreter, img_array: np.ndarray) -> np.ndarray:
    """Test-time augmentation: average the prediction for the photo and its mirror image.
    Cheap (2 runs) and makes the answer steadier on real photos."""
    p1 = predict(interpreter, img_array)
    p2 = predict(interpreter, img_array[:, ::-1, :].copy())
    return ((p1 + p2) / 2.0).astype(np.float32)
