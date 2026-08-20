"""Shared pieces: the exact RMS inference preprocessing, and tflite scoring.

The preprocessing contract is RMS/MLFilter.py classifyPNGs (lines 136-156):
PIL open -> PIL-default resize to the model's input size -> float32 -> /255.
No standardization. Training and evaluation must both go through load_crop()
so they can never drift from what the station runs at night.
"""
import os
import glob

import numpy as np
from PIL import Image


def tflite_interpreter(model_path):
    """The same import cascade RMS uses (MLFilter.py:32-54)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def model_input_size(interp):
    _, height, width, _ = interp.get_input_details()[0]["shape"]
    return int(width), int(height)


def load_crop(png_path, size):
    """One crop PNG -> float32 array in [0,1], shape (H, W), exactly as
    MLFilter.classifyPNGs prepares it."""
    image = Image.open(png_path)
    image = image.resize(size)
    image = np.asarray(image, dtype=np.float32)
    return image / 255.0


def score_tflite(interp, image_2d):
    """Single-crop score through a tflite interpreter, mirroring
    MLFilter.set_input_tensor/classify_image."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    tensor = np.expand_dims(image_2d, axis=(0, 3)).astype(np.float32)
    interp.set_tensor(inp["index"], tensor)
    interp.invoke()
    return float(np.squeeze(interp.get_tensor(out["index"])))


def find_crops(data_root):
    """All banked crop PNGs under data/ml_training/<night>/."""
    return sorted(glob.glob(os.path.join(data_root, "*", "FF_*.png")))
