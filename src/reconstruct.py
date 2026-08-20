"""Reconstruct hyper_model.tflite as a trainable Keras model, with a hard
verification gate.

The tflite flatbuffer carries all the weights; the architecture was recovered
by tensor introspection (shapes pin the strides/padding uniquely):

    Input (1,60,60,1)
    Conv2D 15 @ 10x10, stride 3, valid, ReLU   -> (17,17,15)
    Conv2D 16 @ 10x10, stride 3, valid, ReLU   -> (3,3,16)
    MaxPool 3x3                                -> (1,1,16)
    Flatten -> Dense 32 ReLU -> Dense 1 sigmoid

Gate (must pass before any fine-tuning):
  1. Keras twin reproduces the tflite scores on every banked crop, max|d| < 1e-5
  2. Round-trip Keras -> tflite (float32, no quantization) also < 1e-5
Weight layout conversions: tflite conv kernels are OHWI -> Keras HWIO via
transpose(1,2,3,0); tflite FC kernels are (out,in) -> Keras (in,out) via .T.

Usage: python src/reconstruct.py [--model models/hyper_model.tflite]
                                 [--data data/ml_training] [--out runs/reconstructed]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import tflite_interpreter, model_input_size, load_crop, score_tflite, find_crops


def read_weights(interp):
    """Pull the 8 weight tensors out of the flatbuffer, matched by shape.
    Shapes are unique in this model, which avoids hardcoding tensor indices."""
    wanted = {
        "conv1_k": (15, 10, 10, 1), "conv1_b": (15,),
        "conv2_k": (16, 10, 10, 15), "conv2_b": (16,),
        "dense1_k": (32, 16), "dense1_b": (32,),
        "dense2_k": (1, 32), "dense2_b": (1,),
    }
    found = {}
    for det in interp.get_tensor_details():
        shape = tuple(det["shape"].tolist())
        for name, target in wanted.items():
            if shape == target and name not in found:
                try:
                    found[name] = interp.get_tensor(det["index"])
                except ValueError:
                    continue
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError("weight tensors not found by shape: %s" % sorted(missing))
    return found


def build_keras(weights, input_hw):
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.Input(shape=(input_hw[1], input_hw[0], 1)),
        keras.layers.Conv2D(15, 10, strides=3, padding="valid", activation="relu", name="conv1"),
        keras.layers.Conv2D(16, 10, strides=3, padding="valid", activation="relu", name="conv2"),
        keras.layers.MaxPool2D(3, name="pool"),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(32, activation="relu", name="dense1"),
        keras.layers.Dense(1, activation="sigmoid", name="dense2"),
    ])
    model.get_layer("conv1").set_weights([weights["conv1_k"].transpose(1, 2, 3, 0), weights["conv1_b"]])
    model.get_layer("conv2").set_weights([weights["conv2_k"].transpose(1, 2, 3, 0), weights["conv2_b"]])
    model.get_layer("dense1").set_weights([weights["dense1_k"].T, weights["dense1_b"]])
    model.get_layer("dense2").set_weights([weights["dense2_k"].T, weights["dense2_b"]])
    return model


def export_tflite(keras_model, path):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # No optimizations: RMS expects un-quantized float32.
    blob = converter.convert()
    with open(path, "wb") as fh:
        fh.write(blob)
    return path


def max_delta(crops, size, ref_interp, score_fn):
    worst = 0.0
    worst_crop = None
    for png in crops:
        img = load_crop(png, size)
        d = abs(score_tflite(ref_interp, img) - score_fn(img))
        if d > worst:
            worst, worst_crop = d, png
    return worst, worst_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/hyper_model.tflite")
    ap.add_argument("--data", default="data/ml_training")
    ap.add_argument("--out", default="runs/reconstructed")
    ap.add_argument("--tolerance", type=float, default=1e-5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    interp = tflite_interpreter(args.model)
    size = model_input_size(interp)
    print("model input: %dx%d" % size)

    weights = read_weights(interp)
    model = build_keras(weights, size)
    n_params = model.count_params()
    print("keras twin built: %d params" % n_params)

    crops = find_crops(args.data)
    if not crops:
        raise SystemExit("no crops under %s -- sync data/ml_training first" % args.data)
    print("gate corpus: %d crops" % len(crops))

    # Gate 1: keras vs original tflite
    def keras_score(img):
        return float(model.predict(img[None, :, :, None], verbose=0)[0, 0])
    d1, c1 = max_delta(crops, size, interp, keras_score)
    print("gate 1 (keras vs tflite): max|d| = %.2e  (%s)" % (d1, os.path.basename(c1 or "")))

    # Gate 2: round-trip export
    rt_path = os.path.join(args.out, "roundtrip.tflite")
    export_tflite(model, rt_path)
    rt = tflite_interpreter(rt_path)
    d2, c2 = max_delta(crops, size, interp, lambda img: score_tflite(rt, img))
    print("gate 2 (roundtrip tflite vs tflite): max|d| = %.2e  (%s)" % (d2, os.path.basename(c2 or "")))

    ok = d1 < args.tolerance and d2 < args.tolerance
    model.save(os.path.join(args.out, "hyper_model_reconstructed.keras"))
    report = {
        "params": int(n_params), "input": list(size), "crops": len(crops),
        "gate1_max_delta": d1, "gate2_max_delta": d2,
        "tolerance": args.tolerance, "pass": bool(ok),
    }
    with open(os.path.join(args.out, "gate_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("GATE %s -- report in %s" % ("PASS" if ok else "FAIL", args.out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
