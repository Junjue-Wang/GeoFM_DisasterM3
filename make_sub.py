#!/usr/bin/env python3
"""Threshold the pure-joint SOFT master into a submittable zip (0.5463 recipe).

Usage:  python3 make_sub.py [T_veg] [T_water]   (defaults 0.15 0.375 = the LB 0.5463 submission)
Example: python3 make_sub.py 0.15 0.375

soft_master/*.npy holds 946 x (4,256,256) float32:
  ch0 = building (already HARD @0.50, canonical SOTA)  -- passed through untouched
  ch1 = veg      (joint convnext SOFT probability)      -- hardened at T_veg here
  ch2 = water    (joint convnext SOFT probability)      -- hardened at T_water here
  ch3 = height   (raw metres, canonical SOTA)           -- passed through untouched
Only ch1/ch2 are thresholded. Output -> submissions/joint_v<T_veg>_w<T_water>.zip
"""
import numpy as np, zipfile, os, sys, glob, io

TV = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
TW = float(sys.argv[2]) if len(sys.argv) > 2 else 0.375

HERE   = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "soft_master")
OUTDIR = os.path.join(HERE, "submissions")
os.makedirs(OUTDIR, exist_ok=True)
tag = lambda t: ("%.3f" % t).rstrip('0').rstrip('.')
out = os.path.join(OUTDIR, "joint_v%s_w%s.zip" % (tag(TV), tag(TW)))

files = sorted(glob.glob(os.path.join(MASTER, "*.npy")))
assert files, "soft_master/ is empty -- copy the 946 npy into the package first"
zf = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
for f in files:
    a = np.load(f).copy()
    a[1] = (a[1] > TV).astype(np.float32)   # veg   HARD
    a[2] = (a[2] > TW).astype(np.float32)   # water HARD
    buf = io.BytesIO(); np.save(buf, a.astype(np.float32))
    zf.writestr(os.path.basename(f), buf.getvalue())
zf.close()
print("wrote %s  (%d tiles)  T_veg=%.3f  T_water=%.3f" % (out, len(files), TV, TW))
