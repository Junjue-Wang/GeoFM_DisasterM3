"""NON-RGB Building specialist (baseline composite loss) + NOISY TEST building labels in TRAIN.

The building analog of water_specialist_baseline_hrnet_testwater_5k and
veg_specialist_baseline_hrnet_testveg_5k. Same mechanism: the 946 test tiles — with their
fused stage-1 building-prediction teacher masks (fuse5222, LB iou_b 0.5222; soft fractions [0,1]) — are
appended to the TRAIN split via the existing pseudo-label pipeline. Loss is UNCHANGED across
real+pseudo: building_baseline = BaselineSpecialistLoss(class_idx=0) only indexes ch0, so the
pseudo-labels (ch0=noisy building, ch1/2/3=0) feed the exact same loss as train labels.

NON-RGB body (adapter_fusion_lite_hrnet_token_fusion) per user request — same trunk as the
water/veg test-label specialists, 4 GeoFM tokens + dense AE/Tessera, NO DINOv3/RGB.

Design (matches the proven building recipe = all-src building 8960933 / a05 / HRNet specialist):
  - Tversky α=0.5/β=0.5 (SYMMETRIC) — all proven building specialists use symmetric; NOT
    water's FN-heavy 0.3/0.7.
  - NO class_filter — building is present in 941/946 tiles (99.5%), so a tile-level filter
    would drop only ~5 tiles (near no-op); the proven all-src building uses no filter. Keep
    all tiles to retain negatives. So ALL 946 test tiles append.
  - SOFT labels (harden_labels=False) — baseline composite (MAE/SSIM) needs continuous
    building fractions; matches the all-src building recipe.

VAL stays the CLEAN held-out train-region 20% (pseudo never enters val) → ckpt selection by
clean iou_buildings @ T>0.1 is trustworthy, NOT polluted by the noisy test labels.
ckpt train_cls resolves to 0 (None or 0 or 0 → 0; the final `or 0` catches building safely).
"""
import os
from copy import deepcopy

from configs.geofm.geofm_base import DATA_ROOT, learning_rate, optimizer, test, train


optimizer = deepcopy(optimizer)
learning_rate = deepcopy(learning_rate)
train = deepcopy(train)
test = deepcopy(test)

# 7500 iters: train ~2564 (≈1618 region-train + 946 test, NO filter), bs=16 → ~160
# steps/epoch → ~47 epochs.
train["num_iters"] = 10000
train["sync_bn"] = False
train["save_ckpt_interval_epoch"] = 5
train["eval_interval_epoch"] = 3
learning_rate["decoder"]["params"]["max_iters"] = 100000  # near-CONSTANT lr over 10k → maximal overfit (no decay-to-0)

optimizer["decoder"]["params"]["lr"] = 1e-3
learning_rate["decoder"]["params"]["base_lr"] = 1e-3
optimizer["decoder"]["params"]["weight_decay"] = 0.0  # no L2 — max overfit

AE_EMB = os.path.join(DATA_ROOT, "train", "alphaearth_emb")
TES_EMB = os.path.join(DATA_ROOT, "train", "tessera_emb")
LABELS = os.path.join(DATA_ROOT, "train", "labels")
TM_S1 = os.path.join(DATA_ROOT, "train", "terramind_s1_emb")
TM_S2 = os.path.join(DATA_ROOT, "train", "terramind_s2_emb")
TH_S1 = os.path.join(DATA_ROOT, "train", "thor_s1_emb")
TH_S2 = os.path.join(DATA_ROOT, "train", "thor_s2_emb")

PROJ = "/path/to/GeoFM"
TM_S1_STATS = f"{PROJ}/runs/_stats/terramind_s1_train.npz"
TM_S2_STATS = f"{PROJ}/runs/_stats/terramind_s2_train.npz"
TH_S1_STATS = f"{PROJ}/runs/_stats/thor_s1_train.npz"
TH_S2_STATS = f"{PROJ}/runs/_stats/thor_s2_train.npz"

# ---- TEST caches + noisy-building pseudo labels (appended to TRAIN) ----
AE_TEST_EMB = os.path.join(DATA_ROOT, "test", "alphaearth_test_emb")
TES_TEST_EMB = os.path.join(DATA_ROOT, "test", "tessera_test_emb")
TM_S1_TEST = os.path.join(DATA_ROOT, "test", "terramind_test_s1_emb")
TM_S2_TEST = os.path.join(DATA_ROOT, "test", "terramind_test_s2_emb")
TH_S1_TEST = os.path.join(DATA_ROOT, "test", "thor_test_s1_emb")
TH_S2_TEST = os.path.join(DATA_ROOT, "test", "thor_test_s2_emb")
PSEUDO_BLDG_DIR = os.path.join(DATA_ROOT, "train", "pseudo_bldg_labels_fuse5222_4ch")


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
    harden_labels=False,        # SOFT labels — keep building fractions in [0,1]
    random_crop_size=None,
    random_crop_grid=16,
    # NO class_filter: building present in 941/946 tiles, keep all (filter ≈ no-op).
    # ---- NOISY test building labels appended to TRAIN (loss unchanged) ----
    pseudo_embedding_dirs=[AE_TEST_EMB, TES_TEST_EMB],
    pseudo_target_dir=PSEUDO_BLDG_DIR,
    pseudo_token_embedding_dirs=[TM_S1_TEST, TM_S2_TEST, TH_S1_TEST, TH_S2_TEST],
)

data = dict(
    train=dict(type="GeoFMMultiEmbeddingLoader",
               params={**_common, "split": "train", "training": True, "augment": False}),
    test=dict(type="GeoFMMultiEmbeddingLoader",
              params={**_common, "split": "val", "training": False, "augment": False}),
)

config = dict(
    model=dict(
        type="GeoFMEmbed2Heights",
        params=dict(
            model_type="adapter_fusion_lite_hrnet_token_fusion",   # NON-RGB
            in_channels=192,
            dense_channels=[64, 128],
            token_channels=[768, 768, 768, 768],
            adapter_out=64,
            height_activation="softplus",
            height_norm_constant=30.0,
            metric_threshold=0.1,
            loss=dict(
                type="building_baseline",     # → BaselineSpecialistLoss(class_idx=0)
                mae_weight=1.0,
                mae_bg_weight=0.05,
                ssim_weight=0.5,
                grad_weight=0.5,
                tversky_weight=2.0,
                tversky_alpha=0.5,            # SYMMETRIC (proven building recipe)
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

# === TEST-ONLY OVERFIT (fuse5222 0.5222 building teacher), no-aug, single-view+model-last infer ===
config["data"]["train"]["params"]["pseudo_oversample"] = 5
config["data"]["train"]["params"]["train_subsample_n"] = 0   # test-only: drop all real train GT
