"""Multi-source test inference with Test-Time Augmentation (D4 group, 8 views).

For each test sample, run model on 8 geometric augmentations (identity, hflip,
vflip, hvflip = rot180, rot90, rot270, rot90+hflip, rot270+hflip — the full
dihedral group D4). Predictions are inverted to align with original orientation,
then averaged (mean for seg, mean for height by default).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.geofm import GeoFMMultiEmbeddingDataset
from module.networks import GeoFMNet


# D4 group: 8 geometric transformations. Each entry: (name, forward, inverse).
# `forward` is applied to input image; `inverse` is applied to output prediction
# to map back to original coordinate system.
# All operate on tensors with spatial dims (..., H, W).
def _tta_views():
    # Notation: rotk = torch.rot90(.., k=k, dims=(-2,-1)) is k×90° CCW
    # Inverse of rotk = rot(-k).
    # Flips are self-inverse.
    return [
        ("identity",
         lambda x: x,
         lambda y: y),
        ("hflip",
         lambda x: torch.flip(x, dims=[-1]),
         lambda y: torch.flip(y, dims=[-1])),
        ("vflip",
         lambda x: torch.flip(x, dims=[-2]),
         lambda y: torch.flip(y, dims=[-2])),
        ("hvflip",
         lambda x: torch.flip(x, dims=[-2, -1]),
         lambda y: torch.flip(y, dims=[-2, -1])),
        ("rot90",
         lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
         lambda y: torch.rot90(y, k=-1, dims=[-2, -1])),
        ("rot270",
         lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
         lambda y: torch.rot90(y, k=-3, dims=[-2, -1])),
        ("rot90_hflip",
         lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1]),
         lambda y: torch.rot90(torch.flip(y, dims=[-1]), k=-1, dims=[-2, -1])),
        ("rot270_hflip",
         lambda x: torch.flip(torch.rot90(x, k=3, dims=[-2, -1]), dims=[-1]),
         lambda y: torch.rot90(torch.flip(y, dims=[-1]), k=-3, dims=[-2, -1])),
    ]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-dirs", nargs="+", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-type", required=True)
    p.add_argument("--in-channels", type=int, default=192)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--height-norm-constant", type=float, default=30.0)
    p.add_argument("--dequantize-ae", type=lambda v: str(v).lower() not in ("0", "false", "no"),
                   default=False,
                   help="False (default 2026-05-19): raw int8 AE clipped to ±50 — matches "
                        "training of fusion_adapter_late_* models.")
    p.add_argument("--num-views", type=int, default=8,
                   help="Number of TTA views (1..8). 1 = no TTA, 4 = flips only, 8 = full D4.")
    p.add_argument("--seg-strategy", default="mean", choices=("mean", "median"),
                   help="Aggregation for seg channels across TTA views.")
    p.add_argument("--height-strategy", default="mean", choices=("mean", "median"),
                   help="Aggregation for height channel across TTA views.")
    p.add_argument("--token-embedding-dirs", nargs="*", default=[],
                   help="Token-format embedding dirs (e.g. terramind_test_s1_emb). Auto-upsampled to patch size.")
    p.add_argument("--token-stats-paths", nargs="*", default=[],
                   help="Per-channel mean/std .npz for each token dir (must match length).")
    p.add_argument("--dense-channels", nargs="*", type=int, default=None,
                   help="For adapter_fusion_late_multi: dense source channel sizes (e.g. 64 128)")
    p.add_argument("--token-channels", nargs="*", type=int, default=None,
                   help="For adapter_fusion_late_multi: token source channel sizes (e.g. 768 768 768 768)")
    p.add_argument("--token-upsample", type=lambda v: str(v).lower() in ("1","true","yes"),
                   default=True, help="If False, keep tokens at native 16×16 (late fusion).")
    p.add_argument("--adapter-out", type=int, default=64)
    p.add_argument("--fused-bottleneck", type=int, default=384)
    p.add_argument("--source-channels", nargs="*", type=int, default=None,
                   help="Per-source channel counts in concat order, e.g. 64 128 768 for AE+Tessera+TM_S1.")
    p.add_argument("--use-groupnorm", type=lambda v: str(v).lower() in ("1","true","yes"),
                   default=False)
    p.add_argument("--output-size", type=int, default=None,
                   help="If set, bilinear-downsample model predictions to this size before saving "
                        "(e.g., 256 when training at 512 but LB expects 256x256).")
    p.add_argument("--upsample-mode", default="reflect", choices=("reflect", "bilinear", "nearest"),
                   help="How to handle patch_size > native size at the dataset level "
                        "(bilinear for super-resolution input).")
    # RGB DINOv3-L cache (5th modality, only used by model_type
    # adapter_fusion_lite_hrnet_token_fusion_rgb).
    p.add_argument("--rgb-token-dir", default=None,
                   help="Directory of cached dinov3l_<id>.npy files (fp16, 1024×R×R).")
    p.add_argument("--rgb-token-stats-path", default=None,
                   help="NPZ with per-channel mean/std for z-score, e.g. "
                        "runs/_stats/dinov3l_sat493m_train.npz.")
    p.add_argument("--rgb-token-native-size", type=int, default=160,
                   help="Native spatial extent of cached RGB token grid (160 for 2560/16).")
    p.add_argument("--rgb-token-channels", type=int, default=1024,
                   help="Channel dim of DINOv3-L cache (1024 for ViT-L).")
    p.add_argument("--rgb-modality-dropout", type=float, default=0.10,
                   help="Modality dropout p (only consulted to build model; inference "
                        "uses eval mode so dropout is OFF regardless).")
    p.add_argument("--rgb-clip-z", type=float, default=10.0,
                   help="±σ clip after z-score.")
    p.add_argument("--use-rgb", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                   default=True,
                   help="Master switch: True (default) enables RGB path for "
                        "model_type=*_rgb; False forces no-RGB (only valid if model_type "
                        "doesn't require it).")
    return p.parse_args()


def _extract_state_dict(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional 8-GPU sharding: when LAUNCHER=torchrun, each rank handles a
    # disjoint slice of test samples. No DDP barrier needed — each rank writes
    # its own .npy outputs (filenames are sample-stem unique). When run with
    # LAUNCHER=python, LOCAL_RANK absent → world_size=1 → identity sharding.
    import os as _os
    local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
    world_size = int(_os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        _tag = f"[r{local_rank}/{world_size}]"
    else:
        _tag = ""

    # RGB loader hookup: pass through CLI args. When --rgb-token-dir is None, loader
    # silently skips RGB (no entries in meta). Model_type=*_rgb without --rgb-token-dir
    # is a misconfiguration — assert below after model is built.
    rgb_kwargs = {}
    if args.rgb_token_dir and args.use_rgb:
        rgb_kwargs = dict(
            rgb_token_dir=args.rgb_token_dir,
            rgb_token_stats_path=args.rgb_token_stats_path,
            rgb_token_native_size=args.rgb_token_native_size,
            rgb_token_channels=args.rgb_token_channels,
            rgb_clip_z=args.rgb_clip_z,
        )
    dataset = GeoFMMultiEmbeddingDataset(
        embedding_dirs=args.embedding_dirs,
        target_dir=None,
        split="test",
        patch_size=args.patch_size,
        training=False,
        augment=False,
        height_norm_constant=args.height_norm_constant,
        dequantize_ae=args.dequantize_ae,
        token_embedding_dirs=args.token_embedding_dirs or None,
        token_stats_paths=args.token_stats_paths or None,
        token_upsample=args.token_upsample,
        upsample_mode=args.upsample_mode,
        **rgb_kwargs,
    )
    # Shard the test set by sample index across ranks (data-parallel inference).
    # We modify dataset.samples in-place — bs/num_workers stay as-is, each rank
    # iterates its slice independently.
    if world_size > 1:
        all_samples = list(dataset.samples)
        my_samples = all_samples[local_rank::world_size]
        dataset.samples = my_samples
        # token_paths is a parallel list to samples for late-fusion models;
        # keep it in sync.
        if getattr(dataset, "token_paths", None) is not None:
            all_token_paths = list(dataset.token_paths)
            dataset.token_paths = all_token_paths[local_rank::world_size]
        # _sample_ids parallel list (for diagnostics) — slice if present.
        if getattr(dataset, "_sample_ids", None) is not None:
            all_sids = list(dataset._sample_ids)
            dataset._sample_ids = all_sids[local_rank::world_size]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"{_tag} test size (this rank): {len(dataset)}; in_channels: {dataset.in_channels}")
    print(f"{_tag} dequantize_ae: {args.dequantize_ae}")

    views = _tta_views()[:args.num_views]
    print(f"TTA views ({len(views)}): {[v[0] for v in views]}")
    print(f"seg aggregation: {args.seg_strategy}, height aggregation: {args.height_strategy}")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Single-class specialists output 2 channels (class seg + height); others 4.
    SPECIALIST_CLS_IDX = {
        "adapter_fusion_veg_specialist": 1,
        "adapter_fusion_building_specialist": 0,
        "adapter_fusion_water_specialist": 2,
    }
    spec_cls_idx = SPECIALIST_CLS_IDX.get(args.model_type, None)
    out_channels = 2 if spec_cls_idx is not None else 4
    model = GeoFMNet(
        in_channels=args.in_channels, out_channels=out_channels,
        model_type=args.model_type,
        source_channels=args.source_channels,
        use_groupnorm=args.use_groupnorm,
        # Late-fusion only:
        dense_channels=args.dense_channels,
        token_channels=args.token_channels,
        adapter_out=args.adapter_out,
        fused_bottleneck=args.fused_bottleneck,
        # RGB-fused body only:
        rgb_token_channels=args.rgb_token_channels,
        rgb_modality_dropout=args.rgb_modality_dropout,
    ).to(device).eval()
    miss, unex = model.load_state_dict(_extract_state_dict(args.checkpoint), strict=False)
    if miss:
        print(f"missing keys: {miss[:5]}")
    if unex:
        print(f"unexpected keys: {unex[:5]}")
    # Sanity gate: RGB-fused body needs the loader to deliver rgb_token. Catch a
    # config mismatch (RGB model_type + missing --rgb-token-dir) immediately.
    if getattr(model.body, "requires_rgb_token", False):
        assert args.rgb_token_dir and args.use_rgb, (
            f"model_type={args.model_type} requires_rgb_token=True but "
            f"--rgb-token-dir is unset or --use-rgb=False — provide cache + stats.")

    with torch.no_grad():
        for image, meta in tqdm(loader, desc="TTA predicting"):
            image = image.to(device)
            # Late-fusion models also need tokens, transformed by the same view.
            tokens = meta.get("tokens")
            if tokens is not None:
                tokens = tokens.to(device)
            # RGB DINOv3 token: (B, 1024, R, R) — D4-transformed in TTA loop to stay
            # spatially co-registered with image and 16×16 tokens. has_rgb is per-sample
            # (no spatial extent, no transform needed).
            rgb_token = meta.get("rgb_token")
            has_rgb = meta.get("has_rgb")
            if rgb_token is not None:
                rgb_token = rgb_token.to(device)
            if has_rgb is not None:
                has_rgb = has_rgb.to(device)
                if has_rgb.dtype != torch.bool:
                    has_rgb = has_rgb.to(dtype=torch.bool)
            view_preds = []  # list of (B, 4, H, W) tensors aligned to original

            for _, fwd, inv in views:
                x_aug = fwd(image)
                if tokens is not None:
                    tokens_aug = fwd(tokens)   # same D4 transform on 16×16 tokens
                    if rgb_token is not None:
                        rgb_aug = fwd(rgb_token)
                        y_aug_raw = model(x_aug, tokens=tokens_aug,
                                          rgb_token=rgb_aug, has_rgb=has_rgb)
                    else:
                        y_aug_raw = model(x_aug, tokens=tokens_aug)
                else:
                    y_aug_raw = model(x_aug)
                y_aug = model.activate(y_aug_raw)
                # Single-class specialist outputs (B, 2, H, W) = [class_seg, class_height].
                # Expand to (B, 4, H, W) — other class channels stay 0.
                # Class channel position determined by model_type.
                if y_aug.shape[1] == 2:
                    fake4 = torch.zeros(y_aug.shape[0], 4, y_aug.shape[2], y_aug.shape[3],
                                        device=y_aug.device, dtype=y_aug.dtype)
                    cls = spec_cls_idx if spec_cls_idx is not None else 1
                    fake4[:, cls] = y_aug[:, 0]    # class-specific seg
                    fake4[:, 3] = y_aug[:, 1]      # height
                    y_aug = fake4
                y_unaug = inv(y_aug)  # back to original orientation
                view_preds.append(y_unaug)

            stack = torch.stack(view_preds, dim=0)  # (V, B, 4, H, W)
            # Per-channel aggregation
            seg_op = torch.mean if args.seg_strategy == "mean" else torch.median
            h_op = torch.mean if args.height_strategy == "mean" else torch.median

            if args.seg_strategy == "mean":
                seg_avg = stack[:, :, :3].mean(dim=0)
            else:
                seg_avg = stack[:, :, :3].median(dim=0).values
            if args.height_strategy == "mean":
                h_avg = stack[:, :, 3:4].mean(dim=0)
            else:
                h_avg = stack[:, :, 3:4].median(dim=0).values

            merged_t = torch.cat([seg_avg, h_avg], dim=1)
            # Optional: downsample to LB expected size (256 when model trained at 512)
            if args.output_size is not None and merged_t.shape[-1] != args.output_size:
                merged_t = torch.nn.functional.interpolate(
                    merged_t, size=(args.output_size, args.output_size),
                    mode="bilinear", align_corners=False)
            merged = merged_t.cpu().numpy().astype(np.float32)
            merged[:, :3] = np.clip(merged[:, :3], 0.0, 1.0)
            merged[:, 3] = np.clip(merged[:, 3] * args.height_norm_constant, 0.0, None)

            ids = meta["id"]
            for arr, core_id in zip(merged, ids):
                np.save(output_dir / f"{core_id}.npy", arr)

    print(f"Saved {len(dataset)} TTA-averaged predictions to {output_dir}")


if __name__ == "__main__":
    main()
