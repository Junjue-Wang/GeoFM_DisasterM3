#!/bin/bash
# ============================================================================
# Train ONE model (example launcher — evidence that the training pipeline runs).
#
# NOTE: the training DATA and teacher pseudo-labels are NOT shipped in this
# package (they are large and not required for the deliverable). This script
# therefore does not reproduce training end-to-end out of the box — point
# GEOFM_DATA_ROOT at a prepared data root to actually train. The code path,
# configs, loss functions and launch command below are exactly those used to
# produce the 7 checkpoints in ../models/.
#
# Usage:
#   GEOFM_DATA_ROOT=/path/to/GeoFM/data \
#   bash run_train.sh configs/geofm/<config>.py <output_model_dir> [nproc_per_node]
#
# Example (the joint veg+water convnext model, 8 GPUs):
#   GEOFM_DATA_ROOT=/path/to/GeoFM/data bash run_train.sh \
#       configs/geofm/vegwater_overfit_test947_noaug_convnext_10k.py runs/vegwater_convnext 8
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/../code"
PY="${PY:-python}"

CONFIG="${1:?usage: run_train.sh <config_path> <model_dir> [nproc_per_node]}"
MODEL_DIR="${2:?usage: run_train.sh <config_path> <model_dir> [nproc_per_node]}"
NPROC="${3:-8}"

# code/ provides module.* (GeoFMNet, losses, metrics) + data.* (loaders);
# this dir (training/) provides the `configs` namespace package.
export PYTHONPATH="$CODE:$HERE:${PYTHONPATH:-}"
: "${GEOFM_DATA_ROOT:?set GEOFM_DATA_ROOT=/path/to/GeoFM/data (training data is not shipped)}"

"$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
    "$HERE/train.py" --config_path "$CONFIG" --model_dir "$MODEL_DIR"
