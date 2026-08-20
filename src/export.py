"""Export a fine-tuned Keras model to the RMS drop-in tflite.

Float32, no quantization; asserts the tensor contract RMS expects
(NHWC (1,60,60,1) float32 in, (1,1) sigmoid out, high = meteor).

Usage: .venv/bin/python src/export.py --model runs/v1/model.keras \
                                      --out runs/v1/uk00dy_hyper_v1.tflite
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import tflite_interpreter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    model = keras.models.load_model(args.model)
    blob = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    with open(args.out, "wb") as fh:
        fh.write(blob)

    interp = tflite_interpreter(args.out)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    assert list(inp["shape"]) == [1, model.input_shape[1], model.input_shape[2], 1], inp["shape"]
    assert inp["dtype"] == np.float32 and out["dtype"] == np.float32
    assert list(out["shape"]) == [1, 1], out["shape"]
    assert inp["quantization"] == (0.0, 0) and out["quantization"] == (0.0, 0), "must stay un-quantized"

    # keras-vs-tflite agreement spot check on random inputs
    rng = np.random.default_rng(17)
    worst = 0.0
    for _ in range(32):
        x = rng.random((1, model.input_shape[1], model.input_shape[2], 1), dtype=np.float32)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        t = float(np.squeeze(interp.get_tensor(out["index"])))
        k = float(model.predict(x, verbose=0)[0, 0])
        worst = max(worst, abs(t - k))
    assert worst < 1e-5, worst
    print("exported %s (%d bytes), keras-vs-tflite max|d| %.2e" %
          (args.out, os.path.getsize(args.out), worst))


if __name__ == "__main__":
    main()
