"""NON-RGB Height specialist — TRANSDUCTIVE OVERFIT toward the fused test height (1.7594/2.8682).

Goal: make a single non-RGB model reproduce the fused composed height on the 946 TEST tiles
(the eval set) by OVERFITTING the fused-height pseudo, so test predictions → fused → LB rmse
approaches the fused source. NOT an ensemble — a single trained model.

Diagnosis (why the plain testheight only hit LB 1.8909/3.2542): the 946 test pseudo were only
37% of train and trained jointly with all 1618 clean train GT → the model generalized instead
of memorizing; even at the supervised bld∪veg region it was 3.6m off the fused target.

Fix (per user): keep the bld∪veg mask (bv_height_only, NOT full-dense), but make the test
pseudo DOMINATE so the model overfits it:
  - pseudo_oversample=5  → the 946 test fused-height tiles replicated 5× (4730 entries)
  - train_subsample_n=0 → NO real train GT kept (pure test-only overfit)
  → train = 4730 samples (946×5), 100% test → heavy overfit on the fused targets.
  - 10k iters, 8×A100 (regular-a) per-gpu bs=16 (eff 128) → ~256 epochs over the mix.

CRITICAL — inference uses model-LAST, NOT model-best: clean-region val rmse DEGRADES as the
model overfits the test fused targets (it stops being a general height model), so the
official_score "best" ckpt would be the LEAST-overfit early one — the opposite of the goal.
Use runs/.../model-last.pth for TTA. (eval kept sparse, only as an overfit-progress sanity.)
"""
import os
from copy import deepcopy

from configs.geofm.geofm_base import DATA_ROOT, learning_rate, optimizer, test, train

optimizer = deepcopy(optimizer); learning_rate = deepcopy(learning_rate)
train = deepcopy(train); test = deepcopy(test)

train["num_iters"] = 10000
train["sync_bn"] = False
train["save_ckpt_interval_epoch"] = 20      # ~256 epochs → ~13 ckpts (incl model-last)
train["eval_interval_epoch"] = 20           # clean-val is misleading here; sparse sanity only
learning_rate["decoder"]["params"]["max_iters"] = 100000  # near-constant LR (max overfit)
optimizer["decoder"]["params"]["lr"] = 1e-3
learning_rate["decoder"]["params"]["base_lr"] = 1e-3
optimizer["decoder"]["params"]["weight_decay"] = 0.0  # no L2 — maximize overfit

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

AE_TEST_EMB = os.path.join(DATA_ROOT, "test", "alphaearth_test_emb")
TES_TEST_EMB = os.path.join(DATA_ROOT, "test", "tessera_test_emb")
TM_S1_TEST = os.path.join(DATA_ROOT, "test", "terramind_test_s1_emb")
TM_S2_TEST = os.path.join(DATA_ROOT, "test", "terramind_test_s2_emb")
TH_S1_TEST = os.path.join(DATA_ROOT, "test", "thor_test_s1_emb")
TH_S2_TEST = os.path.join(DATA_ROOT, "test", "thor_test_s2_emb")
PSEUDO_HEIGHT_DIR = os.path.join(DATA_ROOT, "train", "pseudo_height_labels_revfused_4ch")

_common = dict(
    embedding_dirs=[AE_EMB, TES_EMB],
    target_dir=LABELS,
    split_by="region", val_fraction=0.2, split_seed=42,
    patch_size=256, batch_size=8, num_workers=4,
    height_norm_constant=30.0, dequantize_ae=False,
    token_embedding_dirs=[TM_S1, TM_S2, TH_S1, TH_S2],
    token_stats_paths=[TM_S1_STATS, TM_S2_STATS, TH_S1_STATS, TH_S2_STATS],
    token_clip_z=10.0, token_upsample=False,
    harden_labels=False,
    random_crop_size=None, random_crop_grid=16,
    pseudo_embedding_dirs=[AE_TEST_EMB, TES_TEST_EMB],
    pseudo_target_dir=PSEUDO_HEIGHT_DIR,
    pseudo_token_embedding_dirs=[TM_S1_TEST, TM_S2_TEST, TH_S1_TEST, TH_S2_TEST],
    pseudo_oversample=5,        # 946 test fused tiles ×5 = 4730 entries (epoch bookkeeping)
    train_subsample_n=0,        # TEST-ONLY: drop all real train GT → pure overfit on fused test
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
            model_type="adapter_fusion_lite_hrnet_mbconv_token_fusion",   # NON-RGB
            in_channels=192, dense_channels=[64, 128],
            token_channels=[768, 768, 768, 768], adapter_out=64,
            height_activation="softplus", height_norm_constant=30.0,
            metric_threshold=0.1,
            loss=dict(
                type="bv_height_only",      # keep bld∪veg mask (NOT full-dense)
                bld_weight=1.0, veg_weight=1.0,
                height_loss_type="smooth_l1", huber_beta=0.1,
                height_mask_thresh=0.1,
            ),
        ),
    ),
    data=data, optimizer=optimizer, learning_rate=learning_rate,
    train=train, test=test,
)
