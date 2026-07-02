#!/bin/bash
# ============================================================================
# GUARANTEED reproduction path (no GPU, no embeddings, ~1 min).
# Rebuilds the exact LB 0.5463 submission from soft_master/ and verifies it
# matches final_submission/ array-for-array (maxdiff 0 over all 946 tiles).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"   # any python w/ numpy; override with PY=/path/to/python if not on PATH

echo "[1/2] Assembling submission from soft_master  (veg>0.15, water>0.375) ..."
"$PY" "$HERE/make_sub.py" 0.15 0.375

echo "[2/2] Verifying against the original submitted zip ..."
"$PY" - "$HERE" <<'PYEOF'
import sys, glob, os, zipfile, io, numpy as np
here = sys.argv[1]
new  = os.path.join(here, "submissions", "joint_v0.15_w0.375.zip")
refs = glob.glob(os.path.join(here, "final_submission", "*.zip"))
assert len(refs) == 1, "expected exactly one reference zip in final_submission/, found %d" % len(refs)
ref  = refs[0]
def load(z):
    d = {}
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.endswith(".npy"):
                d[os.path.basename(n)] = np.load(io.BytesIO(zf.read(n)))
    return d
a, b = load(new), load(ref)
assert set(a) == set(b), "tile set differs"
mx = max(float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in a)
print("  tiles compared:", len(a))
print("  max abs diff  :", mx)
print("  RESULT        :", "IDENTICAL -- reproduction verified" if mx == 0 else "MISMATCH (%.6g)" % mx)
sys.exit(0 if mx == 0 else 1)
PYEOF
echo "Done. Reproduced zip: submissions/joint_v0.15_w0.375.zip  (== final_submission, LB 0.5463)"
