"""A/B a candidate model against the shipped hyper_model, per night, and
calibrate the deployment threshold.

Scores every banked crop through both models with the exact inference
preprocessing. Reports per-night score tables, the storm false-pass rate as a
function of threshold, and the two named regression checks from the plan:

  - 205237 fixtures: the confirmed meteors must score above the chosen
    threshold; the satellite pass must stay below it
  - storm night: false-pass rate at the chosen threshold <= budget (default 2%)

The chosen threshold goes RAW into the station .config ml_filter (no 0.5/0.85
rescale for non-default model names, ConfigReader.py:1608-1617).

Usage: .venv/bin/python src/evaluate.py --candidate runs/v1/model.keras
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import tflite_interpreter, model_input_size, load_crop, score_tflite

STORM_NIGHT = "UK00DY_20260818_013854_695494"
FIXTURE_NIGHT = "UK00DY_20260819_205237_784793"


def scorer(path):
    if path.endswith(".tflite"):
        interp = tflite_interpreter(path)
        size = model_input_size(interp)
        return lambda img: score_tflite(interp, img), size
    from tensorflow import keras
    model = keras.models.load_model(path)
    size = (model.input_shape[2], model.input_shape[1])
    return lambda img: float(model.predict(img[None, :, :, None], verbose=0)[0, 0]), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help=".keras or .tflite")
    ap.add_argument("--reference", default="models/hyper_model.tflite")
    ap.add_argument("--data", default="data/ml_training")
    ap.add_argument("--storm-budget", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = os.path.dirname(args.data.rstrip("/")) or "."
    rows = list(csv.DictReader(open(os.path.join(root, "dataset_all.csv"))))
    cand, csize = scorer(args.candidate)
    ref, rsize = scorer(args.reference)

    for r in rows:
        png = os.path.join(args.data, r["night"], r["crop_png"])
        r["cand"] = cand(load_crop(png, csize))
        r["ref"] = ref(load_crop(png, rsize))

    # Threshold sweep on the storm night -> lowest threshold meeting the budget
    storm = [r for r in rows if r["night"] == STORM_NIGHT]
    labeled = [r for r in rows if r["label"] in ("meteor", "artefact")]
    # Keep-rate is judged on current-optics meteors only; the deployed model
    # will never see fisheye trails again.
    meteors = [r for r in labeled if r["label"] == "meteor" and r["lens"] == "4mm"]
    sweep = []
    for thr in np.arange(0.05, 0.96, 0.01):
        fp = sum(1 for r in storm if r["cand"] > thr) / max(len(storm), 1)
        keep = sum(1 for r in meteors if r["cand"] > thr) / max(len(meteors), 1)
        sweep.append({"thr": round(float(thr), 2), "storm_fp": round(fp, 4),
                      "meteor_keep": round(keep, 4)})
    ok = [s for s in sweep if s["storm_fp"] <= args.storm_budget]
    chosen = ok[0]["thr"] if ok else None

    # Per-night comparison table
    nights = {}
    for r in rows:
        n = nights.setdefault(r["night"], {"n": 0, "cand_pass": 0, "ref_pass": 0})
        n["n"] += 1
        if chosen is not None and r["cand"] > chosen:
            n["cand_pass"] += 1
        if r["ref"] > 0.5:  # hyper_model's effective production threshold
            n["ref_pass"] += 1

    # Named regression fixtures
    fixtures = [{"crop": r["crop_png"], "label": r["label"],
                 "ref": round(r["ref"], 4), "cand": round(r["cand"], 4)}
                for r in rows if r["night"] == FIXTURE_NIGHT]
    fx_meteor_ok = all(f["cand"] > (chosen or 1) for f in fixtures if f["label"] == "meteor")
    fx_sat_ok = all(f["cand"] <= (chosen or 0) for f in fixtures if f["label"] == "artefact")

    report = {
        "candidate": args.candidate, "reference": args.reference,
        "chosen_threshold_raw": chosen, "storm_budget": args.storm_budget,
        "labeled": len(labeled), "meteors": len(meteors),
        "meteor_keep_at_thr": next((s["meteor_keep"] for s in sweep if s["thr"] == chosen), None),
        "fixtures_205237": fixtures,
        "fixture_meteors_pass": fx_meteor_ok, "fixture_satellite_reject": fx_sat_ok,
        "per_night": nights, "sweep": sweep,
    }
    out = args.out or os.path.join(os.path.dirname(args.candidate) or ".", "eval_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("threshold (raw ml_filter): %s  meteor keep: %s  fixtures: meteors %s / satellite %s"
          % (chosen, report["meteor_keep_at_thr"],
             "PASS" if fx_meteor_ok else "FAIL", "PASS" if fx_sat_ok else "FAIL"))
    print("report -> %s" % out)


if __name__ == "__main__":
    main()
