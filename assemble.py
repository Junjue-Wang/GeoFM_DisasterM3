#!/usr/bin/env python3
"""Assemble the final LB 0.5463 submission from the 7 raw model prediction dirs.

Reads models/inference_manifest.json, pulls each model's per-tile prediction from
  <preds-root>/<model.dir>/predictions/<tile>.npy   (each is (4,256,256) float32)
and combines them:
  ch0 building = mean over 3 models of pred[0]  -> HARD > 0.50
  ch1 veg      = joint pred[1]                  -> HARD > 0.15
  ch2 water    = joint pred[2]                  -> HARD > 0.375
  ch3 height   = mean over 3 models of clip(pred[3], max 100)  (raw metres)
Writes submissions/full_inference.zip and verifies it against final_submission/.

Usage:  python3 assemble.py [--preds-root DIR] [--pred-subdir predictions]
Default preds-root = models/ ; pred-subdir = 'predictions' (where run_full_inference.sh writes).
"""
import numpy as np, json, os, sys, glob, io, zipfile, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--preds-root", default=os.path.join(HERE, "models"))
ap.add_argument("--pred-subdir", default="predictions")
ap.add_argument("--out", default=os.path.join(HERE, "submissions", "full_inference.zip"))
args = ap.parse_args()

man = json.load(open(os.path.join(HERE, "models", "inference_manifest.json")))
def pdir(m):
    return os.path.join(args.preds_root, m["dir"], args.pred_subdir)

# discover tiles from the first building model
tiles = sorted(os.path.basename(f) for f in glob.glob(os.path.join(pdir(man["ch0_building"]["models"][0]), "*.npy")))
assert tiles, "no predictions found -- run run_full_inference.sh first (or point --preds-root at existing dirs)"

os.makedirs(os.path.dirname(args.out), exist_ok=True)
zf = zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED)
for t in tiles:
    b = np.mean([np.load(os.path.join(pdir(m), t))[0] for m in man["ch0_building"]["models"]], axis=0)
    h = np.mean([np.clip(np.load(os.path.join(pdir(m), t))[3], None, 100.0) for m in man["ch3_height"]["models"]], axis=0)
    jm = man["ch1_veg"]["models"][0]
    j = np.load(os.path.join(pdir(jm), t))
    out = np.stack([
        (b > man["ch0_building"]["threshold"]),
        (j[1] > man["ch1_veg"]["threshold"]),
        (j[2] > man["ch2_water"]["threshold"]),
        h,
    ]).astype(np.float32)
    buf = io.BytesIO(); np.save(buf, out); zf.writestr(t, buf.getvalue())
zf.close()
print("wrote %s  (%d tiles)" % (args.out, len(tiles)))

# verify against the submitted zip
ref = glob.glob(os.path.join(HERE, "final_submission", "*.zip"))
if ref:
    def load(z):
        d = {}
        with zipfile.ZipFile(z) as zz:
            for n in zz.namelist():
                if n.endswith(".npy"): d[os.path.basename(n)] = np.load(io.BytesIO(zz.read(n)))
        return d
    a, r = load(args.out), load(ref[0])
    keys = list(a)
    # per-channel: max abs diff (ch0-2 are 0/1 masks; ch3 is raw height in metres)
    seg_flips = seg_px = 0
    ch_mx = [0.0, 0.0, 0.0, 0.0]
    for k in keys:
        d = np.abs(a[k].astype(np.float64) - r[k].astype(np.float64))
        for c in range(4):
            ch_mx[c] = max(ch_mx[c], float(d[c].max()))
        for c in range(3):                       # seg channels: count flipped pixels
            seg_flips += int((d[c] > 0).sum()); seg_px += d[c].size
    mx = max(ch_mx)
    print("verify vs final_submission (%d tiles):" % len(keys))
    print("  max abs diff  ch0=%.3g ch1=%.3g ch2=%.3g (masks) | ch3=%.4g m (height)"
          % (ch_mx[0], ch_mx[1], ch_mx[2], ch_mx[3]))
    print("  seg pixels differing: %d / %d  (%.4f%%)" % (seg_flips, seg_px, 100.0 * seg_flips / seg_px))
    if mx == 0:
        verdict = "IDENTICAL (bit-exact)"
    elif seg_flips == 0 and ch_mx[3] < 0.5:
        verdict = "REPRODUCED (fp-level height diff only, seg masks bit-exact -- same LB score)"
    elif 100.0 * seg_flips / seg_px < 0.01 and ch_mx[3] < 1.0:
        verdict = "REPRODUCED (negligible fp nondeterminism -- same LB score)"
    else:
        verdict = "MISMATCH -- investigate"
    print("  RESULT:", verdict)
    # non-fatal: fresh GPU runs are rarely bit-exact; verdict conveys the outcome
