"""Auto-label confirmed meteors from GMN's published multi-station trajectories.

A crop whose FF timestamp matches (within tolerance) a GMN trajectory that lists
this station is a triangulation-CONFIRMED meteor -- the strongest positive label
available. Writes label='meteor' into the night manifests and records the match
in data/gmn_matches.csv (which build_dataset.py uses for label_source=gmn).

The station is newly GMN-registered, so early runs may match nothing; re-run
after each publication day. Never overwrites an existing non-empty label except
to promote 'unsure'/'artefact?' style provisional values -- a triangulation
match beats a hand guess.

Usage: python src/label_gmn.py [--months 202608 202609] [--station UK00DY]
                               [--data data/ml_training]
"""
import argparse
import csv
import datetime
import os
import re
import urllib.request

MONTHLY_URL = "https://globalmeteornetwork.org/data/traj_summary_data/monthly/traj_summary_monthly_{month}.txt"
UA = {"User-Agent": "meteorml-uk00dy label_gmn (github.com/GlassOnTin/meteorml-uk00dy)"}
# One FF block is 10.24 s; the trajectory 'beginning' time can precede the FF
# timestamp by up to a block, plus clock slop either side.
TOL_BEFORE = 15.0
TOL_AFTER = 15.0

FF_TIME_RE = re.compile(r"FF_[A-Z0-9]+_(\d{8})_(\d{6})_(\d{3})_")


def ff_time(ff_name):
    m = FF_TIME_RE.search(ff_name)
    if not m:
        return None
    d, t, ms = m.groups()
    return datetime.datetime.strptime(d + t, "%Y%m%d%H%M%S") + datetime.timedelta(milliseconds=int(ms))


def gmn_times(month, station):
    """UTC datetimes of trajectories that include `station` in the given month."""
    url = MONTHLY_URL.format(month=month)
    try:
        text = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=120).read().decode("utf-8", "replace")
    except Exception as e:
        print("label_gmn: fetch failed for %s (%s)" % (month, e))
        return []
    out = []
    for line in text.splitlines():
        if not line[:1].isdigit() or station not in line:
            continue
        # Column 3 is the beginning UTC time, e.g. 2026-08-19 22:19:19.123456
        parts = [p.strip() for p in line.split(";")]
        for p in parts[:4]:
            try:
                out.append(datetime.datetime.strptime(p[:26], "%Y-%m-%d %H:%M:%S.%f"))
                break
            except ValueError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=None,
                    help="YYYYMM list; default = every month present in the crop data")
    ap.add_argument("--station", default="UK00DY")
    ap.add_argument("--data", default="data/ml_training")
    args = ap.parse_args()

    # Gather crops and their times
    crops = []  # (manifest_path, row_index, crop_png, dt)
    manifests = {}
    for night in sorted(os.listdir(args.data)):
        mp = os.path.join(args.data, night, "manifest.csv")
        if not os.path.exists(mp):
            continue
        rows = list(csv.reader(open(mp)))
        manifests[mp] = rows
        for i, r in enumerate(rows[1:], start=1):
            dt = ff_time(r[0])
            if dt:
                crops.append((mp, i, r[0], dt))

    months = args.months or sorted({c[3].strftime("%Y%m") for c in crops})
    times = []
    for month in months:
        t = gmn_times(month, args.station)
        print("label_gmn: %s -> %d %s trajectories" % (month, len(t), args.station))
        times.extend(t)

    matches = []
    for mp, i, png, dt in crops:
        if any(-TOL_BEFORE <= (dt - t).total_seconds() <= TOL_AFTER for t in times):
            rows = manifests[mp]
            hdr = rows[0]
            li = hdr.index("label")
            if rows[i][li] not in ("meteor",):
                rows[i][li] = "meteor"
            matches.append((png, dt.isoformat()))

    for mp, rows in manifests.items():
        with open(mp, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)

    out = os.path.join(os.path.dirname(args.data.rstrip("/")), "gmn_matches.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop_png", "ff_utc"])
        w.writerows(matches)
    print("label_gmn: %d crops confirmed as meteors -> %s" % (len(matches), out))


if __name__ == "__main__":
    main()
