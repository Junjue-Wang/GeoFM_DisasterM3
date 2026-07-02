import os

# Data root for the embed2heights dataset.
# Override with env var GEOFM_DATA_ROOT to point at your prepared data root.
DATA_ROOT = os.environ.get(
    "GEOFM_DATA_ROOT",
    "/path/to/GeoFM/data",
)
TRAIN_EMB_DIR = os.path.join(DATA_ROOT, "train", "terramind_s2_emb")
TRAIN_LABEL_DIR = os.path.join(DATA_ROOT, "train", "labels")

data = dict(
    train=dict(
        type="GeoFMEmbeddingLoader",
        params=dict(
            embedding_dir=TRAIN_EMB_DIR,
            target_dir=TRAIN_LABEL_DIR,
            split="train",
            split_by="region",
            val_fraction=0.2,
            split_seed=42,
            patch_size=256,
            latent_scale=16,
            batch_size=8,
            num_workers=2,
            training=True,
            augment=True,
            height_norm_constant=30.0,
        ),
    ),
    test=dict(
        type="GeoFMEmbeddingLoader",
        params=dict(
            embedding_dir=TRAIN_EMB_DIR,
            target_dir=TRAIN_LABEL_DIR,
            split="val",
            split_by="region",
            val_fraction=0.2,
            split_seed=42,
            patch_size=256,
            latent_scale=16,
            batch_size=8,
            num_workers=2,
            training=False,
            augment=False,
            height_norm_constant=30.0,
        ),
    ),
)

optimizer = dict(
    decoder=dict(
        type="adamw",
        params=dict(
            lr=2e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        ),
        grad_clip=dict(
            max_norm=1.0,
            norm_type=2,
        ),
    )
)

learning_rate = dict(
    decoder=dict(
        type="poly",
        params=dict(
            base_lr=2e-4,
            power=0.9,
            max_iters=30000,
        ),
    )
)

train = dict(
    forward_times=1,
    num_iters=30000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=False,
    sync_bn=False,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=5,
    eval_interval_epoch=5,
)

test = dict()

