#!/bin/bash
# ============================================================================
# FULL from-weights inference of ALL 7 models behind the LB 0.5463 submission,
# then assemble -> submissions/full_inference.zip (verified vs final_submission).
#
# Runs each model in models/<dir>/ on the 946 test tiles (single-view, no TTA),
# writing per-model predictions to models/<dir>/predictions/, then assemble.py
# combines them (building ENS3 mean>0.50 | joint veg>0.15 water>0.375 | height
# ENS3 clip<=100 mean).
#
# Requires: 1 GPU + conda env with EVER/torch/rasterio, and the ~37 GB test
# embeddings (NOT shipped). Point EMB at the shared .../GeoFM/data/test.
# For the fast, GPU-free path use ../reproduce_from_soft.sh instead.
#
# Run from the package root (activate your env first, then):
#   EMB=/path/to/GeoFM/data/test bash run_full_inference.sh
#   # or override the interpreter too:  PY=/path/to/python EMB=... bash run_full_inference.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/code"
PY="${PY:-python}"                                                    # override with PY=/path/to/python
EMB="${EMB:-/path/to/GeoFM/data/test}"                               # ~37GB test embeddings (NOT shipped) — set this
STATS="$CODE/_stats"
[ -d "$EMB/alphaearth_test_emb" ] || { echo "ERROR: EMB=$EMB has no alphaearth_test_emb/ — set EMB=/path/to/GeoFM/data/test"; exit 1; }

# role | model dir (rel to models/) | model_type | ckpt file
MODELS=(
  "ch0_building/hrnet    | ch0_building_ens3/hrnet         | adapter_fusion_lite_hrnet_token_fusion          | model-last.pth"
  "ch0_building/convnext | ch0_building_ens3/convnext      | adapter_fusion_lite_hrnet_convnext_token_fusion | model-best.pth"
  "ch0_building/mbconv   | ch0_building_ens3/mbconv        | adapter_fusion_lite_hrnet_mbconv_token_fusion   | model-best.pth"
  "ch1ch2_joint/convnext | ch1ch2_vegwater_joint/convnext  | adapter_fusion_lite_hrnet_convnext_token_fusion | model-last.pth"
  "ch3_height/hrnet      | ch3_height_ens3/hrnet           | adapter_fusion_lite_hrnet_token_fusion          | model-last.pth"
  "ch3_height/convnext   | ch3_height_ens3/convnext        | adapter_fusion_lite_hrnet_convnext_token_fusion | model-last.pth"
  "ch3_height/mbconv     | ch3_height_ens3/mbconv          | adapter_fusion_lite_hrnet_mbconv_token_fusion   | model-last.pth"
)

for row in "${MODELS[@]}"; do
  IFS='|' read -r name dir mtype ckpt <<< "$row"
  name="$(echo "$name" | xargs)"; dir="$(echo "$dir" | xargs)"
  mtype="$(echo "$mtype" | xargs)"; ckpt="$(echo "$ckpt" | xargs)"
  OUT="$HERE/models/$dir/predictions"
  echo "=== [$name] $mtype  ($ckpt) -> $OUT ==="
  cd "$CODE"
  PYTHONPATH="$CODE:${PYTHONPATH:-}" "$PY" predict_multi_tta.py \
    --embedding-dirs        "$EMB/alphaearth_test_emb" "$EMB/tessera_test_emb" \
    --token-embedding-dirs  "$EMB/terramind_test_s1_emb" "$EMB/terramind_test_s2_emb" \
                            "$EMB/thor_test_s1_emb" "$EMB/thor_test_s2_emb" \
    --token-stats-paths     "$STATS/terramind_s1_train.npz" "$STATS/terramind_s2_train.npz" \
                            "$STATS/thor_s1_train.npz" "$STATS/thor_s2_train.npz" \
    --checkpoint            "$HERE/models/$dir/$ckpt" \
    --output-dir            "$OUT" \
    --model-type            "$mtype" \
    --in-channels 192 --dense-channels 64 128 --token-channels 768 768 768 768 \
    --adapter-out 64 --token-upsample False --dequantize-ae False \
    --num-views 1 --batch-size 4 --num-workers 2
done

echo "=== assembling final submission ==="
"$PY" "$HERE/assemble.py"
echo "Done -> submissions/full_inference.zip"
