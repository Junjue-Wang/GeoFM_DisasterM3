# Training pipeline — two stages (for review — full data reproduction NOT required)

This folder documents and ships the **runnable training scripts** behind the 7 checkpoints in
`../models/`. It is provided as evidence that the models are genuinely trained, with the exact
configs and launch command. The training **data and teacher pseudo-labels are not shipped**
(large, and not needed to verify inference / the 0.5463 result — use `../reproduce_from_soft.sh`
or `../run_full_inference.sh` for that). Point `GEOFM_DATA_ROOT` at a prepared data root to
actually run training.

Each backbone (hrnet / convnext / mbconv) is produced in **two stages**:
- **Stage 1** — supervised pretrain on the labelled **TRAIN** split (real IGN-LiDAR GT), region-held-out
  val, augmentation ON, `adamw` LR 2e-4 / wd 1e-4, poly LR. General per-architecture backbone.
- **Stage 2** — pseudo-label distillation on the **TRAIN+TEST** mix: a fresh model is trained **from
  scratch** (NOT resumed from the stage-1 ckpt) on train + the 946 test tiles, where the test tiles are
  pseudo-labelled by the **stage-1 model predictions** (fused into the teacher masks). Pseudo pairs are
  appended to train (`pseudo_oversample=5`), `augment=False`, `weight_decay=0`, near-constant LR 1e-3.
  `train_subsample_n` sets the real-train fraction kept in the mix; the shipped final configs use
  `train_subsample_n=0` (test-pseudo extreme) for max test-calibration.

## Contents
```
training/
├── train.py                       # training entrypoint (EVER th_ddp trainer)
├── run_train.sh                   # example torchrun launcher (one model per call)
└── configs/geofm/
    ├── geofm_base.py                    # shared base (data root, dataset, optim, schedule)
    ├── stage1_trainset_convnext.py      # STAGE 1: supervised train-set pretrain (reference config)
    └── <7 stage-2 model configs>        # STAGE 2: one per shipped checkpoint (see table)
```
Model/loss/loader code is reused from `../code/` (`module/`, `data/`) — training and
inference share the same implementation, so nothing is duplicated.

## Stage 1 config (per architecture)
`stage1_trainset_convnext.py` is the reference stage-1 config (ConvNeXt backbone). The hrnet and
mbconv stage-1 configs are identical except for `model_type`
(`adapter_fusion_lite_hrnet_[token_fusion|convnext_token_fusion|mbconv_token_fusion]`). It trains on
the real train split only (no `pseudo_*` keys, no `train_subsample_n`) and reads token-normalisation
stats from `../../../code/_stats/*.npz` (override with env `GEOFM_TOKEN_STATS`).

## The 7 stage-2 configs ↔ checkpoints
| stage-2 config (configs/geofm/) | → checkpoint (../models/) |
|---|---|
| bldg_specialist_overfit_fuse5222_noaug_10k.py         | ch0_building_ens3/hrnet |
| bldg_overfit_fuse5222_noaug_convnext_10k.py           | ch0_building_ens3/convnext |
| bldg_overfit_fuse5222_noaug_mbconv_10k.py             | ch0_building_ens3/mbconv |
| vegwater_overfit_test947_noaug_convnext_10k.py        | ch1ch2_vegwater_joint/convnext |
| height_specialist_hrnet_testheight_overfit_revfused_noaug_10k.py | ch3_height_ens3/hrnet |
| height_overfit_revfused_noaug_convnext_10k.py         | ch3_height_ens3/convnext |
| height_overfit_revfused_noaug_mbconv_10k.py           | ch3_height_ens3/mbconv |

## How to launch
```bash
# from the training/ dir, with your env active (see ../requirements.txt + vendor/ever wheel):
# Stage 1 — supervised train-set pretrain:
GEOFM_DATA_ROOT=/path/to/GeoFM/data \
bash run_train.sh configs/geofm/stage1_trainset_convnext.py runs/stage1_convnext 8
# Stage 2 — pseudo-label distillation on train+test mix:
GEOFM_DATA_ROOT=/path/to/GeoFM/data \
bash run_train.sh configs/geofm/vegwater_overfit_test947_noaug_convnext_10k.py runs/vegwater_convnext 8
```
`run_train.sh` sets `PYTHONPATH=../code:.` (so `module.*`/`data.*` and the `configs` package
resolve) and launches `torchrun --nproc_per_node=<N> train.py --config_path <cfg> --model_dir <out>`.
Checkpoints are written as `model-best.pth` / `model-last.pth` in `<model_dir>` (inference uses
`model-last` for the overfit runs, `model-best` for building convnext/mbconv — see
`../models/inference_manifest.json`).

## Training recipe

### Stage 1 — supervised pretrain on the TRAIN set
Standard supervised training on the labelled train split (real IGN-LiDAR-derived GT:
band0=bld, 1=veg, 2=water, 3=height), region-held-out val (`split_by="region"`, `val_fraction=0.2`),
augmentation ON, full-256, `adamw` LR 2e-4 / `weight_decay=1e-4`, poly LR schedule. Same model /
loss / loader as stage 2 — see `stage1_trainset_convnext.py` (and its hrnet/mbconv `model_type`
variants). Produces a general per-architecture backbone.

### Stage 2 — pseudo-label distillation on the TRAIN+TEST mix
A fresh model is trained **from scratch** (NOT resumed from the stage-1 ckpt). The 946 **test** tiles,
pseudo-labelled by the **stage-1 model predictions** (fused across models/sources into the teacher
masks below), are appended to the train pairs (`pseudo_oversample=5`), `augment=False`, full-256 crop,
`weight_decay=0`, LR 1e-3 via poly schedule, 10k iters, soft labels, 8×A100 DDP. `train_subsample_n`
controls how many real-train pairs remain in the mix; the shipped final configs use
`train_subsample_n=0` (→ 4730 pseudo samples, the test-pseudo extreme of the mix) for maximal
test-time calibration. LR schedule is per-model: 6 of the 7 use `max_iters=100000` (LR stays
near-constant over the 10k run → maximal overfit); the **hrnet height** config uses `max_iters=10000`,
so its LR decays fully over the run — see each config. Per family, the stage-1-derived teacher (fused
predictions) + stage-2 loss:
- **building** — teacher = building prediction fusion mask (LB iou_b 0.5222); loss `building_baseline`
  (Tversky α=β=0.5).
- **veg+water (joint)** — teacher = merged veg+water prediction masks; loss `vegwater_baseline` =
  per-class `MAE(1.0/bg0.05) + 0.5·SSIM + 0.5·GradDiff + 2.0·Tversky(α=β=0.5)` on ch1 and ch2.
- **height** — teacher = revFused height prediction; loss `bv_height_only` (smooth-L1, supervised on
  bld∪veg only).

The teacher pseudo-labels were built by fusing the stage-1 model predictions across backbones/sources
(`tools/build_vegwater_teacher.py` and per-class equivalents) — not shipped here, as the deliverable
targets inference verification.

## To actually train (paths you must repoint)
Beyond `GEOFM_DATA_ROOT` (data + pseudo-labels): the **stage-1** config reads token-normalisation
stats from `../../../code/_stats/*.npz` automatically (override with env `GEOFM_TOKEN_STATS`). The
**stage-2** configs hardcode `PROJ="/work/.../GeoFM"` and build `token_stats_paths =
f"{PROJ}/runs/_stats/*.npz"` — those 4 `.npz` files **ship at `../code/_stats/`**, so to train
off-cluster edit `PROJ`/`token_stats_paths` in the stage-2 config to point there (or place the npz
under `<PROJ>/runs/_stats/`). This only matters if you actually run training; it does not affect
inference reproduction.

## Validated
`train.py` and all configs (+ `geofm_base.py`) import cleanly with `PYTHONPATH=../code:.` (registry
type references resolve).
- **Stage 1** — a real GPU smoke run of `stage1_trainset_convnext.py` on a small real-embedding train
  subset (RTX 5080, single process, 6 iters) executed the full pipeline end-to-end: 12.68 M-param
  ConvNeXt built, dense-192ch + 4×768 token sources loaded, `adamw` LR 2e-4/wd 1e-4 forward+backward,
  final eval, and `model-best.pth` / `model-last.pth` saved. (The shipped trainer initialises `nccl`;
  on Windows, which has no `nccl`, a single-process run just neutralises the distributed layer — the
  8×A100 launch above is `torchrun` DDP as normal.)
- **Stage 2** — a real 12-iter GPU smoke run of the joint veg+water config confirmed the pseudo-label
  path executes end-to-end (`+pseudo: appended 946 pairs … overfit-mix: real 0 + pseudo 4730 (×5)`,
  exit 0, checkpoint saved).

Full training additionally requires GPUs, `GEOFM_DATA_ROOT`, and the token-stats repoint above.
