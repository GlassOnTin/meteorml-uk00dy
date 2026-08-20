"""Warm-start fine-tune of the reconstructed hyper_model on station-labeled crops.

Design (see README):
- warm start from runs/reconstructed/hyper_model_reconstructed.keras (gate-verified)
- by-NIGHT train/val split: whole held-out nights + a time-contiguous tail of the
  storm night, so correlated frames never straddle the split
- D4 augmentation (flips + 90-degree rotations: zenith camera, streak direction
  carries no class signal) + brightness/contrast jitter + Gaussian noise +
  small translations; no warps/blur (streak straightness and sharpness are
  class-relevant)
- fisheye-era crops train-only, down-weighted (--fisheye-weight, 0 drops them)
- distillation: unlabeled crops enter the loss with the original model's score
  as a soft target at low weight, so local fine-tuning doesn't forget the
  upstream model's general artefact knowledge
- smoke mode (--pseudo-labels): no hand labels needed; teacher scores >0.9/<0.1
  become pseudo labels. Validates the pipeline, not the science.

Usage (workstation):
  .venv/bin/python src/train.py --tag v1 [--val-nights N1 N2] [--freeze-conv1]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import load_crop

VAL_NIGHTS_DEFAULT = ["UK00DY_20260819_205237_784793", "UK00DY_20260819_224710_776426"]
STORM_NIGHT = "UK00DY_20260818_013854_695494"
STORM_VAL_FRACTION = 0.25


def load_rows(root):
    rows = list(csv.DictReader(open(os.path.join(root, "dataset_all.csv"))))
    for r in rows:
        r["pre_ml_score"] = float(r["pre_ml_score"]) if r["pre_ml_score"] else None
    return rows


def split(rows, val_nights):
    """train, val — by night, with a time-contiguous storm tail in val."""
    storm = sorted([r for r in rows if r["night"] == STORM_NIGHT], key=lambda r: r["ff_utc"])
    n_tail = int(len(storm) * STORM_VAL_FRACTION)
    storm_val = {r["crop_png"] for r in storm[-n_tail:]} if n_tail else set()
    train, val = [], []
    for r in rows:
        if r["night"] in val_nights or r["crop_png"] in storm_val:
            val.append(r)
        else:
            train.append(r)
    return train, val


def make_arrays(rows, data_dir, size, args, training):
    """-> images (N,H,W,1), targets (N,), sample weights (N,)."""
    xs, ys, ws = [], [], []
    n_pos = sum(1 for r in rows if r["label"] == "meteor")
    n_neg = sum(1 for r in rows if r["label"] == "artefact")
    # Inverse-frequency weighting explodes at tiny positive counts (14 positives
    # -> ~42x, which taught v1a that any streak is a meteor); cap it.
    w_pos = min((n_pos + n_neg) / (2.0 * n_pos), args.pos_weight_cap) if n_pos else 0.0
    w_neg = (n_pos + n_neg) / (2.0 * n_neg) if n_neg else 0.0
    for r in rows:
        label, teacher = r["label"], r["pre_ml_score"]
        if args.pseudo_labels and label not in ("meteor", "artefact"):
            if teacher is not None and teacher > 0.9:
                label = "meteor"
            elif teacher is not None and teacher < 0.1:
                label = "artefact"
        targets = []
        if label == "meteor":
            targets.append((1.0, w_pos))
        elif label == "artefact":
            w = w_neg
            # Trail-shaped artefacts (satellites/planes; >12 frames, same rule
            # as the harvester's kin_hint) are the confusable class and rare
            # next to storm-cloud negatives -- without extra weight, augmented
            # positives teach "streak = meteor" and satellite scores collapse
            # toward 1.
            try:
                if int(r["n_frames"]) > 12:
                    w *= args.trail_neg_weight
            except (KeyError, ValueError):
                pass
            targets.append((0.0, w))
        elif training and args.distill > 0 and teacher is not None:
            targets.append((teacher, args.distill))
        # --distill-all: artefact-labeled crops ALSO pull toward the teacher's
        # score -- the anti-forgetting term that keeps hyper_model's storm and
        # satellite priors when the labeled set is tiny. (Once every crop is
        # labeled, plain distillation has no unlabeled rows and goes inert.)
        # Meteor rows are exempt: they are exactly where the teacher is wrong,
        # and anchoring them to it caps how far the fine-tune can lift them.
        if (training and args.distill_all and args.distill > 0
                and teacher is not None and label == "artefact"):
            targets.append((teacher, args.distill))
        if not targets:
            continue
        # fisheye down-weighting applies to METEORS only: positive trail
        # morphology is optics-specific, but artefact negatives (thin satellite
        # streaks, clouds, noise) transfer across lenses and are scarce at 4mm.
        if r["lens"] == "fisheye" and label == "meteor":
            if not training or args.fisheye_weight <= 0:
                continue
            targets = [(y, w * args.fisheye_weight) for y, w in targets]
        elif r["lens"] == "fisheye" and not training:
            continue
        png = os.path.join(data_dir, r["night"], r["crop_png"])
        img = load_crop(png, size)
        for y, w in targets:
            xs.append(img)
            ys.append(y)
            ws.append(w)
    x = np.array(xs, dtype=np.float32)[..., None]
    return x, np.array(ys, np.float32), np.array(ws, np.float32)


def augment_dataset(x, y, w, batch, seed=17, photometric=True):
    import tensorflow as tf

    def aug(img, label, weight):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, tf.int32))
        # small translation via pad + random crop
        h, wd = img.shape[0], img.shape[1]
        img = tf.image.resize_with_crop_or_pad(img, h + 6, wd + 6)
        img = tf.image.random_crop(img, (h, wd, 1))
        if photometric:
            # CAUTION: brightness/contrast jitter attacks the very cue that
            # separates thick-bright meteor streaks from thin-faint satellite
            # trails -- measured to collapse satellite rejection. Off for v2+.
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = img + tf.random.normal(tf.shape(img), stddev=0.01)
        return tf.clip_by_value(img, 0.0, 1.0), label, weight

    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    return (ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
              .map(aug, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch).prefetch(tf.data.AUTOTUNE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="data/ml_training")
    ap.add_argument("--base", default="runs/reconstructed/hyper_model_reconstructed.keras")
    ap.add_argument("--val-nights", nargs="*", default=VAL_NIGHTS_DEFAULT,
                    help="empty = deploy mode: train on all nights, val = storm tail only")
    ap.add_argument("--freeze-conv1", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--distill", type=float, default=0.2)
    ap.add_argument("--distill-all", action="store_true")
    ap.add_argument("--pos-weight-cap", type=float, default=5.0)
    ap.add_argument("--trail-neg-weight", type=float, default=3.0)
    ap.add_argument("--photometric-aug", action="store_true")
    ap.add_argument("--fisheye-weight", type=float, default=0.3)
    ap.add_argument("--pseudo-labels", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(args.seed)

    root = os.path.dirname(args.data.rstrip("/")) or "."
    rows = load_rows(root)
    train_rows, val_rows = split(rows, args.val_nights)
    model = keras.models.load_model(args.base)
    size = (model.input_shape[2], model.input_shape[1])

    x_tr, y_tr, w_tr = make_arrays(train_rows, args.data, size, args, training=True)
    x_va, y_va, w_va = make_arrays(val_rows, args.data, size, args, training=False)
    print("train: %d (hard %d) val: %d" % (len(x_tr), int((w_tr != args.distill).sum()), len(x_va)))
    if len(x_va) == 0 or len(x_tr) == 0:
        raise SystemExit("empty split -- label more data or adjust --val-nights")

    if args.freeze_conv1:
        model.get_layer("conv1").trainable = False
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="binary_crossentropy",
                  weighted_metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])

    out = os.path.join("runs", args.tag)
    os.makedirs(out, exist_ok=True)
    # Early stopping needs positives in val to be meaningful; an artefact-only
    # val (deploy mode) is minimized by predicting artefact for everything, so
    # deploy runs train for exactly --epochs, tuned from the CV runs' best epoch.
    cb = []
    if (y_va == 1.0).any():
        cb.append(keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                                patience=15, restore_best_weights=True))
    cb += [
        keras.callbacks.CSVLogger(os.path.join(out, "history.csv")),
    ]
    hist = model.fit(augment_dataset(x_tr, y_tr, w_tr, args.batch, args.seed,
                                     photometric=args.photometric_aug),
                     validation_data=(x_va, y_va, w_va),
                     epochs=args.epochs, verbose=2, callbacks=cb)

    model.save(os.path.join(out, "model.keras"))
    meta = vars(args) | {
        "train_n": len(x_tr), "val_n": len(x_va),
        "best_val_pr_auc": float(max(hist.history.get("val_pr_auc", [0]))),
        "epochs_ran": len(hist.history["loss"]),
    }
    with open(os.path.join(out, "train_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print("saved %s/model.keras  best val PR-AUC %.4f" % (out, meta["best_val_pr_auc"]))


if __name__ == "__main__":
    main()
