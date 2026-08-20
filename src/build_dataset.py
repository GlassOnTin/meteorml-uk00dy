"""Merge the per-night crop manifests into one training table, dataset.csv.

Columns: crop_png, night, ff_utc, n_frames, path_px, label, label_source,
lens, pre_ml_score. Drops unlabeled and 'unsure' rows from the labeled output
but keeps a full sidecar (dataset_all.csv) for distillation soft-targets, which
wants unlabeled crops too.

label_source: gmn (in gmn_matches.csv) > storm (a listed storm night) > hand.
lens: fisheye before the 2026-08-17 lens swap, 4mm after.

Usage: python src/build_dataset.py [--data data/ml_training] [--no-fisheye]
"""
import argparse
import csv
import os

from label_gmn import ff_time

LENS_SWAP = "20260817"
STORM_NIGHTS = {"UK00DY_20260818_013854_695494"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml_training")
    ap.add_argument("--no-fisheye", action="store_true")
    args = ap.parse_args()

    gmn = set()
    gm = os.path.join(os.path.dirname(args.data.rstrip("/")), "gmn_matches.csv")
    if os.path.exists(gm):
        with open(gm, newline="") as fh:
            gmn = {r[0] for r in csv.reader(fh) if r and r[0] != "crop_png"}

    seen = {}
    for night in sorted(os.listdir(args.data)):
        mp = os.path.join(args.data, night, "manifest.csv")
        if not os.path.exists(mp):
            continue
        night_date = night.split("_")[1]
        lens = "fisheye" if night_date < LENS_SWAP else "4mm"
        if args.no_fisheye and lens == "fisheye":
            continue
        with open(mp, newline="") as fh:
            for r in csv.reader(fh):
                if not r or r[0] == "crop_png":
                    continue
                r = (r + [""] * 8)[:8]
                png, ff, num, nf, path_px, hint, label, score = r
                if not os.path.exists(os.path.join(args.data, night, png)):
                    continue
                dt = ff_time(png)
                src = ("gmn" if png in gmn else
                       "storm" if night in STORM_NIGHTS and label == "artefact" else
                       "hand" if label else "")
                # last write wins (manifests are already deduped per night)
                seen[png] = [png, night, dt.isoformat() if dt else "", nf, path_px,
                             label, src, lens, score]

    rows = [seen[k] for k in sorted(seen)]
    hdr = ["crop_png", "night", "ff_utc", "n_frames", "path_px",
           "label", "label_source", "lens", "pre_ml_score"]
    root = os.path.dirname(args.data.rstrip("/"))
    with open(os.path.join(root, "dataset_all.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([hdr] + rows)
    labeled = [r for r in rows if r[5] in ("meteor", "artefact")]
    with open(os.path.join(root, "dataset.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([hdr] + labeled)

    n_m = sum(1 for r in labeled if r[5] == "meteor")
    n_a = len(labeled) - n_m
    by_src = {}
    for r in labeled:
        by_src[r[6]] = by_src.get(r[6], 0) + 1
    print("dataset: %d crops total, %d labeled (%d meteor / %d artefact), sources %s"
          % (len(rows), len(labeled), n_m, n_a, by_src))


if __name__ == "__main__":
    main()
