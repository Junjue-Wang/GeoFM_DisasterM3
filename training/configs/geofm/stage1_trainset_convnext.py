"""STAGE-1 supervised pretrain on the real TRAIN set (model B / +ConvNeXt 12.7M).

Two-stage pipeline (see ../TRAINING.md):
  * STAGE 1 (this config): supervised training on the labelled TRAIN split
    (real IGN-LiDAR-derived GT: band0=bld, 1=veg, 2=water, 3=height). Standard
    region-held-out val, data augmentation ON, adamw lr 2e-4 / wd 1e-4, poly LR.
    Produces a general backbone per architecture.
  * STAGE 2 (vegwater_overfit_test947_noaug_convnext_10k.py & the other overfit
    configs): test-only self-distillation — the STAGE-1 backbone is further fit on
    the 946 test tiles pseudo-labelled by the stage-1 teachers (train GT dropped via
    train_subsample_n=0, pseudo_oversample=5), no-aug, near-constant LR.

This file is the STAGE-1 counterpart of the ConvNeXt veg+water model: same model_type,
channels and loss (VegWaterBaselineLoss on ch1/ch2), but trained on real train GT with
augmentation instead of the test pseudo-labels. NO pseudo_* keys, NO train_subsample_n.

Token normalisation stats ship at ../../../code/_stats/*.npz (override with
env GEOFM_TOKEN_STATS to point elsewhere).
"""
import os
from copy import deepcopy

from configs.geofm.geofm_base import DATA_ROOT, learning_rate, optimizer, test, train


optimizer = deepcopy(optimizer)
learning_rate = deepcopy(learning_rate)
train = deepcopy(train)
test = deepcopy(test)

# --- real TRAIN-set supervised recipe (stage 1) ---
train["num_iters"] = 30000
train["sync_bn"] = False
train["eval_interval_epoch"] = 5
train["save_ckpt_interval_epoch"] = 5

AE_EMB = os.path.join(DATA_ROOT, "train", "alphaearth_emb")
TES_EMB = os.path.join(DATA_ROOT, "train", "tessera_emb")
LABELS = os.path.join(DATA_ROOT, "train", "labels")
TM_S1 = os.path.join(DATA_ROOT, "train", "terramind_s1_emb")
TM_S2 = os.path.join(DATA_ROOT, "train", "terramind_s2_emb")
TH_S1 = os.path.join(DATA_ROOT, "train", "thor_s1_emb")
TH_S2 = os.path.join(DATA_ROOT, "train", "thor_s2_emb")

# Token normalisation stats: shipped alongside the code at ../../../code/_stats.
_STATS = os.environ.get(
    "GEOFM_TOKEN_STATS",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "code", "_stats")),
)
TM_S1_STATS = os.path.join(_STATS, "terramind_s1_train.npz")
TM_S2_STATS = os.path.join(_STATS, "terramind_s2_train.npz")
TH_S1_STATS = os.path.join(_STATS, "thor_s1_train.npz")
TH_S2_STATS = os.path.join(_STATS, "thor_s2_train.npz")

_common = dict(
    embedding_dirs=[AE_EMB, TES_EMB],
    target_dir=LABELS,
    split_by="region",
    val_fraction=0.2,
    split_seed=42,
    patch_size=256,
    batch_size=16,
    num_workers=4,
    height_norm_constant=30.0,
    dequantize_ae=False,
    token_embedding_dirs=[TM_S1, TM_S2, TH_S1, TH_S2],
    token_stats_paths=[TM_S1_STATS, TM_S2_STATS, TH_S1_STATS, TH_S2_STATS],
    token_clip_z=10.0,
    token_upsample=False,
    harden_labels=False,
    random_crop_size=None,
    random_crop_grid=16,
    # NO pseudo_* keys and NO train_subsample_n → trains purely on real train GT.
)

data = dict(
    train=dict(type="GeoFMMultiEmbeddingLoader",
               params={**_common, "split": "train", "training": True, "augment": True}),
    test=dict(type="GeoFMMultiEmbeddingLoader",
              params={**_common, "split": "val", "training": False, "augment": False}),
)

config = dict(
    model=dict(
        type="GeoFMEmbed2Heights",
        params=dict(
            model_type="adapter_fusion_lite_hrnet_convnext_token_fusion",  # model B; out_channels=4
            in_channels=192,
            dense_channels=[64, 128],
            token_channels=[768, 768, 768, 768],
            adapter_out=64,
            height_activation="softplus",
            height_norm_constant=30.0,
            metric_threshold=0.1,
            loss=dict(
                type="vegwater_baseline",     # VegWaterBaselineLoss (ch1 veg + ch2 water)
                mae_weight=1.0,
                mae_bg_weight=0.05,
                ssim_weight=0.5,
                grad_weight=0.5,
                tversky_weight=2.0,
                tversky_alpha=0.5,
                tversky_beta=0.5,
            ),
        ),
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test,
)
