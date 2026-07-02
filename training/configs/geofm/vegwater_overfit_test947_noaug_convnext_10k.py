"""JOINT veg+water test-only overfit distillation (model B / +ConvNeXt 12.7M).

Replaces the two SEPARATE single-class specialists (veg ch1 §3.2 / water ch2 §3.3 of
runs/0609/readme.html) with ONE model per backbone that fits BOTH veg+water jointly.
veg & water are conflicting land-cover classes (263k dual-positive px = sub-pixel mixing);
a shared decoder forced to emit mutually-consistent veg+water may calibrate the overlap
better than two independent models, and (grader has veg→water coupling) raise iou_v+iou_w.

STRICTLY identical to runs/0609 recipe EXCEPT two changes:
  1. loss.type = "vegwater_baseline" → VegWaterBaselineLoss = BaselineSpecialistLoss(ch1)
     + BaselineSpecialistLoss(ch2), same symmetric recipe (MAE 1.0/bg0.05 + 0.5 SSIM
     + 0.5 GradDiff + 2.0 Tversky α=β=0.5) on BOTH classes.
  2. pseudo_target_dir = pseudo_vegwater_labels_test947_4ch (ch1=veg teacher mean 0.4196,
     ch2=water teacher mean 0.0134, ch0/ch3=0). Built by tools/build_vegwater_teacher.py.

Model output stays 4-channel (token_fusion model_type → out_channels=4); ch1=veg, ch2=water
indices preserved → predict/ensemble/combo from 0609 reused verbatim.

Frozen overfit recipe (= 0609 §2.1): train_subsample_n=0 + pseudo_oversample=5 (946×5=4730),
augment=False, random_crop_size=None (full 256), weight_decay=0, near-constant LR 1e-3
(poly max_iters=100000, run 10k), SOFT labels, NO class_filter, 8×A100 bs16. Inference:
single-view (--num-views 1), model-LAST, MODEL_TYPE must match ckpt; ENS3 soft mean ch1/ch2.
"""
import os
from copy import deepcopy

from configs.geofm.geofm_base import DATA_ROOT, learning_rate, optimizer, test, train


optimizer = deepcopy(optimizer)
learning_rate = deepcopy(learning_rate)
train = deepcopy(train)
test = deepcopy(test)

train["num_iters"] = 10000
train["sync_bn"] = False
train["save_ckpt_interval_epoch"] = 5
train["eval_interval_epoch"] = 3
learning_rate["decoder"]["params"]["max_iters"] = 100000  # near-CONSTANT lr over 10k → maximal overfit

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

# ---- TEST caches + JOINT veg+water teacher pseudo labels (the ONLY training data) ----
AE_TEST_EMB = os.path.join(DATA_ROOT, "test", "alphaearth_test_emb")
TES_TEST_EMB = os.path.join(DATA_ROOT, "test", "tessera_test_emb")
TM_S1_TEST = os.path.join(DATA_ROOT, "test", "terramind_test_s1_emb")
TM_S2_TEST = os.path.join(DATA_ROOT, "test", "terramind_test_s2_emb")
TH_S1_TEST = os.path.join(DATA_ROOT, "test", "thor_test_s1_emb")
TH_S2_TEST = os.path.join(DATA_ROOT, "test", "thor_test_s2_emb")
PSEUDO_VW_DIR = os.path.join(DATA_ROOT, "train", "pseudo_vegwater_labels_test947_4ch")


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
    harden_labels=False,        # SOFT labels — reproduce the teachers' continuous fields
    random_crop_size=None,
    random_crop_grid=16,
    # NO class_filter: zeros (no-veg / no-water tiles) are part of the teacher field — memorize them.
    pseudo_embedding_dirs=[AE_TEST_EMB, TES_TEST_EMB],
    pseudo_target_dir=PSEUDO_VW_DIR,
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
            model_type="adapter_fusion_lite_hrnet_convnext_token_fusion",  # NON-RGB (model B); out_channels=4
            in_channels=192,
            dense_channels=[64, 128],
            token_channels=[768, 768, 768, 768],
            adapter_out=64,
            height_activation="softplus",
            height_norm_constant=30.0,
            metric_threshold=0.1,
            loss=dict(
                type="vegwater_baseline",     # → VegWaterBaselineLoss (ch1 veg + ch2 water)
                mae_weight=1.0,
                mae_bg_weight=0.05,
                ssim_weight=0.5,
                grad_weight=0.5,
                tversky_weight=2.0,
                tversky_alpha=0.5,            # SYMMETRIC for faithful teacher reproduction (both classes)
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

# === TEST-ONLY OVERFIT (joint veg+water teachers), no-aug, single-view infer ===
config["data"]["train"]["params"]["pseudo_oversample"] = 5
config["data"]["train"]["params"]["train_subsample_n"] = 0   # test-only: drop all real train GT
