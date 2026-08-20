# meteorml-uk00dy

Station-adapted fine-tune of the RMS / Global Meteor Network meteor-vs-artefact
CNN for **UK00DY** — a non-standard station (mono IMX296 global-shutter sensor,
4mm f/0.95 lens, zenith-pointing, Raspberry Pi 5). The stock `hyper_model.tflite`
was trained on IMX291/307 colour cameras and wrongly rejects this station's
faint mono trails (measured: 11 candidates → 2 kept on 2026-08-19, with ~3 real
meteors among the rejects).

The aim is a drop-in replacement `.tflite` for RMS's `MLFilter`, plus a
reproducible recipe others with unusual optics can follow — and eventually a
mono/narrow-FOV variant proposal upstream to GMN.

## Lineage

- Model + inference: [CroatianMeteorNetwork/RMS](https://github.com/CroatianMeteorNetwork/RMS)
  `RMS/MLFilter.py` (Milan Kalina, 2022)
- Original training project: [fiachraf/meteorml](https://github.com/fiachraf/meteorml)
  (Fiachra Feehilly, 2021) and the [satmonkey fork](https://github.com/satmonkey/meteorml)
- `models/hyper_model.tflite` is the RMS-shipped reference model (105 KB float32,
  ~26k params, 60×60×1 input, single sigmoid, high = meteor)

## How it works

1. **`src/reconstruct.py`** — rebuilds the tflite as a trainable Keras twin by
   reading the weight tensors straight out of the flatbuffer, then runs a hard
   verification gate: the twin (and its round-trip tflite re-export) must
   reproduce the original's scores on every banked crop within 1e-5. No
   fine-tuning until the gate is green.
2. **`src/build_dataset.py`** — merges the station's nightly crop manifests
   (banked by an RMS end-of-night hook) into one labeled `dataset.csv`.
3. **`src/label_gmn.py`** — auto-labels confirmed meteors by matching crops
   against GMN's published multi-station trajectory summaries.
4. **`labeler/`** — a small local web page for the residual hand-labeling.
5. **`src/train.py`** — warm-start fine-tune: frozen first conv, D4 augmentation
   (zenith camera ⇒ streak direction uninformative), by-night train/val split,
   distillation against the original model's scores to prevent forgetting.
6. **`src/evaluate.py`** — A/B vs `hyper_model` per night; threshold calibration
   at a fixed false-positive budget on a cloud-storm night.
7. **`src/export.py`** — float32 tflite export + shape/output asserts.

The preprocessing contract (`src/common.py`) mirrors `MLFilter.classifyPNGs`
exactly: PIL open → PIL-default resize to the model input → float32 → /255.

## Data

`data/` is not committed. Crops live on the station under
`~/RMS_data/ml_training/<night>/` (variable-size square grayscale PNGs of
maxpixel−avepixel detections + `manifest.csv` with kinematics, labels, and the
production model's score per crop) and are rsynced here for training.
