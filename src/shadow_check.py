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

    alarms = []
    if unfiltered and unfiltered >= 20 and filtered is not None and filtered / unfiltered < 0.10:
        alarms.append("OVER-REJECTION? filtered/unfiltered = %d/%d (<10%% with substantial candidates)"
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
