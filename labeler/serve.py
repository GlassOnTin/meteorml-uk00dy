#!/usr/bin/env python3
"""Tiny local web labeler for the banked meteor crops.

Serves every crop with its kinematics + the production model's score; keyboard
m / a / u labels it meteor / artefact / unsure, arrows navigate. Labels are
written straight into the night manifests (the training source of truth).

Run on the station:  python3 labeler/serve.py [--data ~/RMS_data/ml_training] [--port 8750]
Then browse to  http://meteor.local:8750/  (phone works; buttons mirror the keys).
"""
import argparse
import csv
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DATA = os.path.expanduser("~/RMS_data/ml_training")

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UK00DY crop labeler</title>
<style>
 body { background:#131519; color:#e8e6e1; font:16px/1.5 system-ui; margin:0;
        display:flex; flex-direction:column; align-items:center; padding:16px; }
 #crop { image-rendering:pixelated; width:min(88vw,440px); background:#000;
         border:1px solid #2a2e36; border-radius:6px; }
 #meta { font-family:ui-monospace,monospace; font-size:13px; color:#9aa1ab;
         margin:10px 0; text-align:center; }
 #meta b { color:#e8e6e1; }
 .score { color:#c4862c; font-weight:700; }
 #buttons { display:flex; gap:10px; margin-top:6px; }
 button { font:inherit; padding:10px 22px; border-radius:6px; border:1px solid #2a2e36;
          background:#1b1e24; color:#e8e6e1; cursor:pointer; }
 button.m { border-color:#4a9; } button.a { border-color:#a55; }
 #prog { margin-top:14px; color:#9aa1ab; font-size:13px; }
 #done { font-size:22px; margin-top:40px; }
 kbd { background:#2a2e36; border-radius:4px; padding:1px 6px; font-size:12px; }
</style></head><body>
<img id="crop" alt="detection crop"/>
<div id="meta"></div>
<div id="buttons">
 <button class="m" onclick="label('meteor')">meteor <kbd>m</kbd></button>
 <button class="a" onclick="label('artefact')">artefact <kbd>a</kbd></button>
 <button onclick="label('unsure')">unsure <kbd>u</kbd></button>
 <button onclick="move(1)">skip <kbd>&rarr;</kbd></button>
</div>
<div id="prog"></div>
<div id="done" hidden>All labeled &#127881;</div>
<script>
let queue = [], idx = 0;
async function load() {
  queue = await (await fetch('queue')).json();
  idx = 0; show();
}
function show() {
  const remaining = queue.filter(c => !c.done).length;
  document.getElementById('prog').textContent =
    (idx+1) + ' / ' + queue.length + '  (' + remaining + ' unlabeled)  ← back';
  if (!queue.length || idx >= queue.length) {
    document.getElementById('done').hidden = false; return;
  }
  const c = queue[idx];
  document.getElementById('crop').src = 'crop?night=' + c.night + '&png=' + c.png;
  document.getElementById('meta').innerHTML =
    '<b>' + c.png + '</b><br>' + c.night + ' &middot; frames ' + c.n_frames +
    ' &middot; path ' + c.path_px + 'px &middot; ' + c.kin_hint +
    ' &middot; model <span class="score">' + (c.score || '?') + '</span>' +
    (c.label ? ' &middot; labeled: <b>' + c.label + '</b>' : '');
}
function move(d) { idx = Math.min(Math.max(idx + d, 0), queue.length); show(); }
async function label(v) {
  const c = queue[idx]; if (!c) return;
  await fetch('label', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({night:c.night, png:c.png, label:v})});
  c.label = v; c.done = true; move(1);
}
addEventListener('keydown', e => {
  if (e.key === 'm') label('meteor');
  else if (e.key === 'a') label('artefact');
  else if (e.key === 'u') label('unsure');
  else if (e.key === 'ArrowRight' || e.key === ' ') move(1);
  else if (e.key === 'ArrowLeft') move(-1);
});
load();
</script></body></html>"""


def read_manifest(night):
    p = os.path.join(DATA, night, "manifest.csv")
    rows = list(csv.reader(open(p))) if os.path.exists(p) else []
    return p, rows


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/":
            return self._send(200, PAGE.encode())
        if url.path == "/queue":
            items = []
            for night in sorted(os.listdir(DATA)):
                _, rows = read_manifest(night)
                for r in rows[1:]:
                    r = (r + [""] * 8)[:8]
                    items.append({"night": night, "png": r[0], "n_frames": r[3],
                                  "path_px": r[4], "kin_hint": r[5], "label": r[6],
                                  "score": r[7], "done": bool(r[6])})
            # unlabeled first; likely-meteors first within that; newest night first
            items.sort(key=lambda c: c["night"], reverse=True)
            items.sort(key=lambda c: (c["done"], c["kin_hint"] != "meteor?"))
            return self._send(200, json.dumps(items).encode(), "application/json")
        if url.path == "/crop":
            night = os.path.basename(q.get("night", [""])[0])
            png = os.path.basename(q.get("png", [""])[0])
            p = os.path.join(DATA, night, png)
            if not os.path.exists(p):
                return self._send(404, b"missing")
            return self._send(200, open(p, "rb").read(), "image/png")
        return self._send(404, b"nope")

    def do_POST(self):
        if self.path != "/label":
            return self._send(404, b"nope")
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        night = os.path.basename(body["night"])
        png = os.path.basename(body["png"])
        p, rows = read_manifest(night)
        for r in rows[1:]:
            if r and r[0] == png:
                while len(r) < 8:
                    r.append("")
                r[6] = body["label"]
        with open(p + ".tmp", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        os.replace(p + ".tmp", p)
        return self._send(200, b"{}", "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--port", type=int, default=8750)
    args = ap.parse_args()
    DATA = os.path.expanduser(args.data)
    print("labeler on http://0.0.0.0:%d/ over %s" % (args.port, DATA))
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
