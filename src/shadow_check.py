#!/usr/bin/env python3
"""Morning shadow-watch report for the deployed station ML model (plan S8).

Gathers, for the most recent archived night: unfiltered vs filtered detection
counts, the deployed model's per-detection scores (from the MLFilter log lines
and the crop manifest), and evaluates the two alarm rules from the deployment
plan. Pure stdlib; safe to run any time after end-of-night processing.

Writes ~/RMS_data/shadow_watch/<night>.md and appends one line to
~/RMS_data/shadow_watch/shadow_watch.log. Run daily (cron/timer) or by hand:
  python3 src/shadow_check.py [--data-root /mnt/nvme/RMS_data]
"""
import argparse
import datetime
import glob
import os
import re


def model_ab(night, v1_thr=0.42, hyper_thr=0.5):
    """Score the night's banked crops with the stock hyper_model and compare
    against the deployed model's manifest scores. Needs the vRMS venv (tflite).
    -> dict(v1_keeps, hyper_keeps, recovered, lost) or raises."""
    import csv
    import shutil
    import sys
    import tempfile
    sys.path.insert(0, "/home/ian/source/RMS")
    from RMS.MLFilter import classifyPNGs
    crops_dir = os.path.join(os.path.expanduser("~"), "RMS_data", "ml_training", night)
    manifest = os.path.join(crops_dir, "manifest.csv")
    if not os.path.isfile(manifest):
        raise RuntimeError("no crop manifest for " + night)
    v1 = {}
    with open(manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("pre_ml_score"):
                v1[r["crop_png"][:-4]] = float(r["pre_ml_score"])
    if not v1:
        raise RuntimeError("no deployed-model scores in manifest")
    with tempfile.TemporaryDirectory() as td:
        for f in glob.glob(os.path.join(crops_dir, "*.png")):
            shutil.copy(f, td)
        hyper = classifyPNGs(td, "/home/ian/source/RMS/share/hyper_model.tflite")
    both = [(k, v1[k], float(hyper[k])) for k in hyper if k in v1]
    return {
        "n": len(both),
        "v1_keeps": sum(1 for _, v, _ in both if v > v1_thr),
        "hyper_keeps": sum(1 for _, _, h in both if h > hyper_thr),
        "recovered": [k for k, v, h in both if v > v1_thr and h <= hyper_thr],
        "lost": [k for k, v, h in both if h > hyper_thr and v <= v1_thr],
    }


def night_report(data_root):
    nights = sorted(glob.glob(os.path.join(data_root, "ArchivedFiles", "UK00DY_*")),
                    key=os.path.getmtime, reverse=True)
    nights = [n for n in nights if os.path.isdir(n)]
    if not nights:
        return None
    nd = nights[0]
    night = os.path.basename(nd)

    def count(path):
        try:
            m = re.search(r"Meteor Count\s*=\s*(\d+)", open(path).read(500))
            return int(m.group(1)) if m else None
        except OSError:
            return None

    ftps = [f for f in glob.glob(os.path.join(nd, "FTPdetectinfo_*.txt"))
            if "backup" not in f and "uncalibrated" not in f]
    filtered = next((count(f) for f in ftps if "unfiltered" not in f), None)
    unfiltered = next((count(f) for f in ftps if "unfiltered" in f), None)

    # Deployed-model scores from the night's log (MLFilter prints one line per crop)
    scores = []
    model_line = ""
    for lg in sorted(glob.glob(os.path.join(data_root, "logs", "log_*.log")),
                     key=os.path.getmtime, reverse=True)[:3]:
        txt = open(lg, errors="replace").read()
        if night.split("_")[1] not in txt:
            continue
        scores += [(m.group(1), float(m.group(2)), m.group(3)) for m in
                   re.finditer(r"(FF_\S+?) - Score:\s+([\d.]+)% - (meteor|artefact)", txt)]
        mm = re.search(r"ml_model_path.*|Using ML model.*", txt)
        if mm:
            model_line = mm.group(0).strip()
        if scores:
            break

    # A/B against the stock model when the night's crops are banked: a raw
    # filtered/unfiltered ratio can't tell over-rejection from an honestly
    # junk-heavy night (night one: 5/131 fired the alarm, yet BOTH models
    # rejected 126 unanimously and v1 kept a meteor hyper missed).
    ab = None
    try:
        ab = model_ab(night)
    except Exception as e:
        print("(A/B unavailable: %s)" % e)

    alarms = []
    if ab is not None:
        if ab["v1_keeps"] < ab["hyper_keeps"]:
            alarms.append("OVER-REJECTION: deployed model kept %d vs stock %d"
                          % (ab["v1_keeps"], ab["hyper_keeps"]))
    elif unfiltered and unfiltered >= 20 and filtered is not None and filtered / unfiltered < 0.10:
        alarms.append("OVER-REJECTION? filtered/unfiltered = %d/%d (<10%%; no crops for A/B)"
                      % (filtered, unfiltered))
    if unfiltered and unfiltered >= 50 and filtered is not None and filtered / unfiltered > 0.90:
        alarms.append("STORM REGRESSION? filtered/unfiltered = %d/%d (>90%% passed on a busy night)"
                      % (filtered, unfiltered))

    lines = ["# Shadow watch — %s" % night,
             "generated %s UTC" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
             "",
             "- filtered meteors: **%s**" % filtered,
             "- unfiltered candidates: **%s**" % unfiltered,
             "- alarms: %s" % ("; ".join(alarms) if alarms else "none"),
             ""]
    if ab is not None:
        lines.append("- A/B on %d crops: deployed keeps **%d**, stock hyper keeps **%d**; "
                     "recovered %s, lost %s" % (ab["n"], ab["v1_keeps"], ab["hyper_keeps"],
                                                [k[20:38] for k in ab["recovered"]] or "none",
                                                [k[20:38] for k in ab["lost"]] or "none"))
        lines.append("")
    if model_line:
        lines.append("- %s" % model_line)
    if scores:
        lines.append("\n## Per-detection scores (deployed model)\n")
        for ff, sc, verdict in scores:
            lines.append("- `%s`  %.1f%%  %s" % (ff, sc, verdict))
    else:
        lines.append("\n(no MLFilter score lines found in recent logs — "
                     "check the night processed and the ML filter ran)")

    out_dir = os.path.join(os.path.expanduser("~"), "RMS_data", "shadow_watch")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, night + ".md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "shadow_watch.log"), "a") as fh:
        fh.write("%s  %s  filtered=%s unfiltered=%s alarms=%s\n"
                 % (datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), night,
                    filtered, unfiltered, len(alarms)))
    print("\n".join(lines))
    return alarms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/nvme/RMS_data")
    args = ap.parse_args()
    night_report(args.data_root)
