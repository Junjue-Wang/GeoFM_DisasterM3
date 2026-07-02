import glob
import os
import pickle
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

try:
    import ever as er
    from ever.api.data import distributed
    from ever.interface import ConfigurableMixin
except ModuleNotFoundError:
    class _Registry:
        def register(self, *args, **kwargs):
            def _decorator(cls):
                return cls
            return _decorator

    class _ER:
        registry = SimpleNamespace(DATALOADER=_Registry())

    class ConfigurableMixin:
        def __init__(self, config):
            self.config = _to_namespace(config)

    class _Distributed:
        StepDistributedSampler = RandomSampler

    er = _ER()
    distributed = _Distributed()


HEIGHT_NORM_CONSTANT = 30.0


def _to_namespace(value):
    if isinstance(value, SimpleNamespace):
        return value
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    return value


def _cfg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def normalize_core_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]

    for prefix in ("label_", "gee_emb_", "tessera_emb_", "emb_", "s2_", "s1_", "dinov3l_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    for suffix in ("_embeddings", "_embedding", "_merged", "_quantized"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    return base


def candidate_core_ids(filename):
    core_id = normalize_core_id(filename)
    no_year = re.sub(r"_\d{4}$", "", core_id)
    return (core_id,) if no_year == core_id else (core_id, no_year)


def parse_core_id(core_id):
    parts = core_id.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return dict(index=parts[0], region=parts[1], year=parts[-1])
    if len(parts) >= 2:
        return dict(index=parts[0], region=parts[1], year=None)
    return dict(index=core_id, region="unknown", year=None)


def list_embedding_files(embedding_dir):
    return sorted(glob.glob(os.path.join(str(embedding_dir), "**", "*.tif"), recursive=True))


def find_file_pairs(embedding_dir, target_dir):
    emb_files = list_embedding_files(embedding_dir)
    label_files = sorted(glob.glob(os.path.join(str(target_dir), "**", "label_*.tif"), recursive=True))

    label_map = {}
    for label_path in label_files:
        for key in candidate_core_ids(label_path):
            label_map[key] = label_path

    pairs = []
    for emb_path in emb_files:
        for key in candidate_core_ids(emb_path):
            if key in label_map:
                pairs.append((emb_path, label_map[key]))
                break

    pairs.sort(key=lambda p: normalize_core_id(p[0]))
    return pairs


def split_pairs(pairs, split, val_fraction=0.2, seed=42, split_by="region"):
    if split in ("all", "full", None):
        return list(pairs)

    rng = random.Random(seed)
    pairs = list(pairs)

    if split_by == "random":
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        val_indices = set(indices[:val_count])
        selected = [p for i, p in enumerate(pairs) if (i in val_indices) == (split == "val")]
        return selected

    groups = {}
    for pair in pairs:
        core_id = normalize_core_id(pair[0])
        parts = parse_core_id(core_id)
        group_key = parts["year"] if split_by == "year" else parts["region"]
        groups.setdefault(group_key, []).append(pair)

    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    target_val = max(1, int(round(len(pairs) * val_fraction)))

    val_groups = set()
    count = 0
    for key in group_keys:
        if count >= target_val:
            break
        val_groups.add(key)
        count += len(groups[key])

    output = []
    for pair in pairs:
        core_id = normalize_core_id(pair[0])
        parts = parse_core_id(core_id)
        group_key = parts["year"] if split_by == "year" else parts["region"]
        is_val = group_key in val_groups
        if (split == "val" and is_val) or (split == "train" and not is_val):
            output.append(pair)
    return output


EMB_CLIP_ABS = 50.0


def _read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=EMB_CLIP_ABS, neginf=-EMB_CLIP_ABS)
    np.clip(arr, -EMB_CLIP_ABS, EMB_CLIP_ABS, out=arr)
    return arr


def _read_token_normalize_upsample(path, mean, std, out_size=256, clip_z=10.0):
    """Read a low-res token-format embedding (C, h, w), z-score normalize per-channel,
    clip, then bilinear-upsample to match a target spatial size.

    Args:
        path: tif file with shape (C, h, w), typically (768, 16, 16) for TerraMind/THOR.
        mean / std: per-channel arrays of shape (C,) — typically loaded from
            runs/_stats/<source>_train.npz computed by tools/compute_token_stats.py.
        out_size: int (square) or (h, w) tuple. Should match the dense source's
            actual H,W in this sample — some dense AE files are 255x255 natively
            and only get padded to patch_size later inside _crop_pixel, so passing
            a fixed 256 here will mis-align the concat.
        clip_z: clip the z-scored values to ±clip_z to bound pathological channels
            (THOR_S2 in particular has ±20k raw values; even after z-score a
            stray outlier can hit ±100, which would dominate GroupNorm downstream).
    """
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (C, h, w)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-channel z-score
    m = mean.reshape(-1, 1, 1).astype(np.float32)
    s = std.reshape(-1, 1, 1).astype(np.float32)
    arr = (arr - m) / np.maximum(s, 1e-6)
    np.clip(arr, -clip_z, clip_z, out=arr)

    if isinstance(out_size, int):
        size = (out_size, out_size)
    else:
        size = tuple(out_size)

    # Bilinear upsample via torch (fast + no extra deps; scipy.ndimage.zoom is much slower)
    t = torch.from_numpy(arr).unsqueeze(0)  # (1, C, h, w)
    t = nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()  # (C, *size)


def _read_tif_dequantize(path, dequantize_ae=True):
    """Read embedding tif and de-quantize per-source.

    Args:
        path: tif file path
        dequantize_ae: if True (default), AlphaEarth values are divided by 127
            (signed int8 → ~[-1, 1]). If False, AlphaEarth values are only
            clipped to ±EMB_CLIP_ABS — matches the OLD (pre-dequantize) data
            distribution, useful when inferring with a model trained before
            dequantization was introduced (e.g. 8906914, 8906928).

    AlphaEarth: signed int8 quantized to [-128, 127]; divide by 127 -> ~[-1, 1].
    Tessera: native float, no transformation.
    THOR/TerraMind: native float, no transformation.
    Source detected by filename pattern.
    """
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)

    path_str = str(path).lower()
    fname = os.path.basename(path_str)
    # Detect AlphaEarth by filename OR by parent path (test files are named
    # "emb_<id>_<region>_<year>_quantized.tif" without "gee_emb"/"alphaearth"
    # in the basename — must check the containing directory).
    is_alphaearth = (
        "gee_emb" in fname
        or "alphaearth" in fname
        or "alphaearth" in path_str
    )

    if is_alphaearth and dequantize_ae:
        # NEW path: dequantize int8 to roughly [-1, 1]
        arr = np.nan_to_num(arr, nan=0.0, posinf=128.0, neginf=-128.0)
        np.clip(arr, -127.0, 127.0, out=arr)
        arr = arr / 127.0
    elif is_alphaearth and not dequantize_ae:
        # LEGACY path: same as _read_tif (matches training of 8906914/8906928)
        arr = np.nan_to_num(arr, nan=0.0, posinf=EMB_CLIP_ABS, neginf=-EMB_CLIP_ABS)
        np.clip(arr, -EMB_CLIP_ABS, EMB_CLIP_ABS, out=arr)
    else:
        # other sources: keep float native, still clip pathological extremes
        arr = np.nan_to_num(arr, nan=0.0, posinf=EMB_CLIP_ABS, neginf=-EMB_CLIP_ABS)
        np.clip(arr, -EMB_CLIP_ABS, EMB_CLIP_ABS, out=arr)
    return arr


def _pad_reflect(arr, target_h, target_w):
    _, h, w = arr.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")


def _augment_pair(image, target=None):
    if random.random() < 0.5:
        image = image[:, :, ::-1]
        if target is not None:
            target = target[:, :, ::-1]
    if random.random() < 0.5:
        image = image[:, ::-1, :]
        if target is not None:
            target = target[:, ::-1, :]
    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k=k, axes=(1, 2))
        if target is not None:
            target = np.rot90(target, k=k, axes=(1, 2))
    return image, target


def _channel_dropout(image, p=0.1):
    """Randomly mask p fraction of input embedding channels.

    Mask shape (C, 1, 1) — channels uniform across all spatial positions.
    Output is rescaled by C/active_count to preserve expected sum.
    Guarantees at least one channel survives.
    """
    if p <= 0.0:
        return image
    C = image.shape[0]
    keep = (np.random.rand(C) > p).astype(np.float32)
    if keep.sum() == 0:
        keep[np.random.randint(C)] = 1.0
    scale = C / keep.sum()
    out = image * (keep[:, None, None] * scale).astype(image.dtype)
    return out


def _mixup_pair(img1, tgt1, img2, tgt2, alpha=0.2):
    """Mixup: linear interpolation of (img, tgt) pairs.

    img1/img2: (C, H, W) float arrays.
    tgt1/tgt2: (4, H, W) float arrays in normalized space.
    Returns (mixed_img, mixed_tgt, lam).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    # Bias toward keeping img1 dominant — convention from Mixup paper variants
    lam = max(lam, 1.0 - lam)
    img = lam * img1 + (1.0 - lam) * img2
    tgt = lam * tgt1 + (1.0 - lam) * tgt2
    return img.astype(np.float32), tgt.astype(np.float32), float(lam)


def _cutmix_pair(img1, tgt1, img2, tgt2, alpha=1.0):
    """CutMix: paste a random rectangle from sample 2 into sample 1.

    Both image AND target get the same rectangular patch replaced.
    Box size derived from Beta-sampled lam: cut_ratio = sqrt(1-lam).
    Returns (mixed_img, mixed_tgt, lam_effective).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    _, H, W = img1.shape
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h = int(round(H * cut_ratio))
    cut_w = int(round(W * cut_ratio))
    if cut_h <= 0 or cut_w <= 0:
        return img1.astype(np.float32), tgt1.astype(np.float32), 1.0

    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + (cut_h - cut_h // 2))
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + (cut_w - cut_w // 2))

    img = img1.copy()
    tgt = tgt1.copy()
    img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
    tgt[:, y1:y2, x1:x2] = tgt2[:, y1:y2, x1:x2]
    # Effective lam = fraction kept from img1
    area_cut = (y2 - y1) * (x2 - x1)
    lam_eff = 1.0 - area_cut / float(H * W)
    return img.astype(np.float32), tgt.astype(np.float32), lam_eff


# Worker-local cache for building bank pickle so each DataLoader worker pays I/O only once.
_BUILDING_BANK_CACHE = {}


def _get_building_bank(path):
    if path is None:
        return None
    key = str(path)
    if key not in _BUILDING_BANK_CACHE:
        with open(key, "rb") as f:
            _BUILDING_BANK_CACHE[key] = pickle.load(f)
    return _BUILDING_BANK_CACHE[key]


def _copypaste_buildings(
    image,
    target,
    bank,
    per_source_channels,
    height_norm_constant=30.0,
    fg_thresh=0.10,
    n_min=1,
    n_max=3,
    max_tries=10,
):
    """Paste building instances from bank onto background-only regions.

    Constraint: only paste where dest pixels are background — i.e., target's
    foreground channels (bld/veg/water) are all below `fg_thresh` for every
    pixel covered by the instance mask. Buildings never overwrite other
    foreground.

    image: (C, H, W) — concatenation of source embeddings in same order as bank.
    target: (4, H, W) — channels [bld, veg, water, height/30].
    bank:  list of dicts from tools/scan_buildings.py
    per_source_channels: e.g. [64, 128] for AE+Tessera, must match bank entry layout.
    """
    if not bank or n_max <= 0 or target is None:
        return image, target, 0

    C, H, W = image.shape
    if sum(per_source_channels) != C or list(per_source_channels) != [64, 128]:
        # Bank is built for AE(64) + Tessera(128). Skip if layout doesn't match.
        return image, target, 0

    n_paste = random.randint(n_min, n_max)
    fg_mask = (target[0] > fg_thresh) | (target[1] > fg_thresh) | (target[2] > fg_thresh)

    image = image.copy()
    target = target.copy()
    n_pasted = 0

    for _ in range(n_paste):
        inst = bank[random.randint(0, len(bank) - 1)]
        ae_crop = inst["ae"]            # (64, h, w)
        tes_crop = inst["tessera"]      # (128, h, w)
        lab_crop = inst["label"]        # (4, h, w), height in raw meters
        mask = inst["mask"]             # (h, w) bool

        # D4 augmentation on instance
        if random.random() < 0.5:
            ae_crop = ae_crop[:, :, ::-1]
            tes_crop = tes_crop[:, :, ::-1]
            lab_crop = lab_crop[:, :, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:
            ae_crop = ae_crop[:, ::-1, :]
            tes_crop = tes_crop[:, ::-1, :]
            lab_crop = lab_crop[:, ::-1, :]
            mask = mask[::-1, :]
        k = random.randint(0, 3)
        if k:
            ae_crop = np.rot90(ae_crop, k=k, axes=(1, 2))
            tes_crop = np.rot90(tes_crop, k=k, axes=(1, 2))
            lab_crop = np.rot90(lab_crop, k=k, axes=(1, 2))
            mask = np.rot90(mask, k=k, axes=(0, 1))

        ae_crop = np.ascontiguousarray(ae_crop)
        tes_crop = np.ascontiguousarray(tes_crop)
        lab_crop = np.ascontiguousarray(lab_crop).astype(np.float32, copy=True)
        mask = np.ascontiguousarray(mask).astype(bool, copy=False)

        h, w = mask.shape
        if h > H or w > W:
            continue

        # Find a paste location where no instance pixel hits existing foreground
        found = False
        for _try in range(max_tries):
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            dest_fg = fg_mask[top:top + h, left:left + w]
            if not bool((dest_fg & mask).any()):
                found = True
                break
        if not found:
            continue

        # Normalize bank height (meters) -> /30 in line with target convention
        lab_crop[3] = np.clip(lab_crop[3] / height_norm_constant, 0.0, 1.5)

        sy = slice(top, top + h)
        sx = slice(left, left + w)
        m3 = mask[None, :, :]

        ae_dest = image[0:per_source_channels[0], sy, sx]
        image[0:per_source_channels[0], sy, sx] = np.where(m3, ae_crop, ae_dest)

        tes_dest = image[per_source_channels[0]:, sy, sx]
        image[per_source_channels[0]:, sy, sx] = np.where(m3, tes_crop, tes_dest)

        lab_dest = target[:, sy, sx]
        target[:, sy, sx] = np.where(m3, lab_crop, lab_dest)

        # Subsequent pastes must avoid the freshly-pasted instance too
        fg_mask = fg_mask.copy()
        fg_mask[sy, sx] |= mask
        n_pasted += 1

    return image, target, n_pasted


class GeoFMEmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_dir,
        target_dir=None,
        split="train",
        val_fraction=0.2,
        split_seed=42,
        split_by="region",
        patch_size=256,
        latent_scale=16,
        training=True,
        augment=True,
        height_norm_constant=HEIGHT_NORM_CONSTANT,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.target_dir = Path(target_dir) if target_dir else None
        self.split = split
        self.patch_size = patch_size
        self.latent_scale = latent_scale
        self.training = training
        self.augment = augment and training
        self.height_norm_constant = float(height_norm_constant)

        if self.target_dir is None:
            self.samples = [(path, None) for path in list_embedding_files(self.embedding_dir)]
        else:
            pairs = find_file_pairs(self.embedding_dir, self.target_dir)
            self.samples = split_pairs(pairs, split, val_fraction, split_seed, split_by)

        if not self.samples:
            raise FileNotFoundError(
                f"No GeoFM samples found for embedding_dir={self.embedding_dir}, target_dir={self.target_dir}, split={split}."
            )

        first = _read_tif(self.samples[0][0])
        self.is_latent = first.shape[0] == 768 and first.shape[1] < patch_size and first.shape[2] < patch_size
        self.in_channels = first.shape[0]

    def __len__(self):
        return len(self.samples)

    def _crop_pixel(self, image, target):
        image = _pad_reflect(image, self.patch_size, self.patch_size)
        if target is not None:
            target = _pad_reflect(target, self.patch_size, self.patch_size)

        _, h, w = image.shape
        if self.training:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        image = image[:, top:top + self.patch_size, left:left + self.patch_size]
        if target is not None:
            target = target[:, top:top + self.patch_size, left:left + self.patch_size]
        return image, target

    def _crop_latent(self, image, target):
        emb_patch = self.patch_size // self.latent_scale
        image = _pad_reflect(image, emb_patch, emb_patch)
        if target is not None:
            target = _pad_reflect(target, self.patch_size, self.patch_size)

        _, h, w = image.shape
        if self.training:
            top = np.random.randint(0, h - emb_patch + 1)
            left = np.random.randint(0, w - emb_patch + 1)
        else:
            top = (h - emb_patch) // 2
            left = (w - emb_patch) // 2

        image = image[:, top:top + emb_patch, left:left + emb_patch]
        if target is not None:
            t_top = top * self.latent_scale
            t_left = left * self.latent_scale
            target = target[:, t_top:t_top + self.patch_size, t_left:t_left + self.patch_size]
        return image, target

    def __getitem__(self, index):
        emb_path, target_path = self.samples[index]
        image = np.nan_to_num(_read_tif(emb_path))

        target = None
        if target_path is not None:
            target = np.nan_to_num(_read_tif(target_path))
            target[3] = np.clip(target[3] / self.height_norm_constant, 0.0, 1.5)

        if self.is_latent:
            image, target = self._crop_latent(image, target)
        else:
            image, target = self._crop_pixel(image, target)

        if self.augment:
            image, target = _augment_pair(image, target)

        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        meta = dict(
            id=normalize_core_id(emb_path),
            embedding_path=str(emb_path),
            is_latent=self.is_latent,
        )
        if target is not None:
            meta["target"] = torch.from_numpy(np.ascontiguousarray(target)).float()
            meta["target_path"] = str(target_path)

        return image, meta


@er.registry.DATALOADER.register()
class GeoFMEmbeddingLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)

        dataset = GeoFMEmbeddingDataset(
            embedding_dir=self.config.embedding_dir,
            target_dir=self.config.target_dir,
            split=self.config.split,
            val_fraction=self.config.val_fraction,
            split_seed=self.config.split_seed,
            split_by=self.config.split_by,
            patch_size=self.config.patch_size,
            latent_scale=self.config.latent_scale,
            training=self.config.training,
            augment=self.config.augment,
            height_norm_constant=self.config.height_norm_constant,
        )

        if self.config.training:
            sampler = distributed.StepDistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        super().__init__(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )

    def set_default_config(self):
        self.config.update(
            dict(
                embedding_dir="",
                target_dir=None,
                split="train",
                val_fraction=0.2,
                split_seed=42,
                split_by="region",
                patch_size=256,
                latent_scale=16,
                batch_size=4,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                training=True,
                augment=True,
                height_norm_constant=HEIGHT_NORM_CONSTANT,
                dequantize_ae=False,
                channel_dropout_p=0.0,
                mixup_p=0.0,
                mixup_alpha=0.2,
                cutmix_p=0.0,
                cutmix_alpha=1.0,
            )
        )


def find_token_paths_for_ids(token_dirs, want_ids):
    """For each (canonical) id we want, look up the file path in each token dir.

    Returns dict: {canonical_id -> tuple_of_token_paths}, only including ids
    present in EVERY token dir.
    """
    per_dir_maps = []
    for d in token_dirs:
        m = {}
        for p in list_embedding_files(d):
            for key in candidate_core_ids(p):
                if key not in m:
                    m[key] = p
        per_dir_maps.append(m)

    out = {}
    for key in want_ids:
        token_paths = []
        ok = True
        for m in per_dir_maps:
            if key in m:
                token_paths.append(m[key])
            else:
                ok = False
                break
        if ok:
            out[key] = tuple(token_paths)
    return out


def find_multi_file_pairs(embedding_dirs, target_dir):
    """Match samples across multiple embedding sources by canonical core_id.

    Returns list of (tuple_of_emb_paths, label_path) where each tuple has the
    embedding path from each dir in order.
    """
    # Per-dir: canonical_id -> emb_path
    per_dir_maps = []
    for d in embedding_dirs:
        m = {}
        for emb_path in list_embedding_files(d):
            for key in candidate_core_ids(emb_path):
                if key not in m:
                    m[key] = emb_path
        per_dir_maps.append(m)

    # Label map
    label_files = sorted(glob.glob(os.path.join(str(target_dir), "**", "label_*.tif"), recursive=True))
    label_map = {}
    for label_path in label_files:
        for key in candidate_core_ids(label_path):
            if key not in label_map:
                label_map[key] = label_path

    # Find common keys: present in every emb_dir AND in labels
    common = set(per_dir_maps[0]) & set(label_map)
    for m in per_dir_maps[1:]:
        common &= set(m)
    common = sorted(common)

    # Dedup by emb_paths tuple — a file can appear under multiple candidate
    # ids (e.g., "3001_BE_2023" and "3001_BE"), which would otherwise emit
    # the same pair twice.
    out = []
    seen = set()
    for key in common:
        emb_paths = tuple(m[key] for m in per_dir_maps)
        if emb_paths in seen:
            continue
        seen.add(emb_paths)
        out.append((emb_paths, label_map[key]))
    return out


class GeoFMMultiEmbeddingDataset(Dataset):
    """Channel-concat fusion of multiple embedding sources for the same patch."""

    def __init__(
        self,
        embedding_dirs,
        target_dir=None,
        split="train",
        val_fraction=0.2,
        split_seed=42,
        split_by="region",
        patch_size=256,
        training=True,
        augment=True,
        height_norm_constant=HEIGHT_NORM_CONSTANT,
        dequantize_ae=False,
        channel_dropout_p=0.0,
        mixup_p=0.0,
        mixup_alpha=0.2,
        cutmix_p=0.0,
        cutmix_alpha=1.0,
        copypaste_bank=None,
        copypaste_p=0.0,
        copypaste_n_min=1,
        copypaste_n_max=3,
        copypaste_fg_thresh=0.10,
        copypaste_max_tries=10,
        token_embedding_dirs=None,
        token_stats_paths=None,
        token_clip_z=10.0,
        token_upsample=True,
        upsample_mode="reflect",
        # RGB DINOv3-L cache (5th modality). Defaults: disabled (backward-compat).
        rgb_token_dir=None,                  # dir with dinov3l_<core_id>.npy fp16 cache
        rgb_token_stats_path=None,           # npz with {mean: (1024,), std: (1024,)}
        rgb_token_native_size=160,           # spatial extent of cached token grid
        rgb_token_channels=1024,             # channel dim of cached token (DINOv3-L = 1024)
        rgb_clip_z=10.0,                     # ±σ clip after z-score (same convention as tokens)
        pseudo_embedding_dirs=None,
        pseudo_target_dir=None,
        pseudo_token_embedding_dirs=None,
        pseudo_oversample=1,
        train_subsample_n=None,
        include_val_in_train=False,
        harden_labels=False,
        harden_thresh=0.1,
        random_crop_size=None,
        random_crop_grid=16,
        building_mask_dir=None,
        building_mask_thresh=0.5,
        mask_channel_idx=0,
        mask_two_channel=False,
        mask_dropout_p=0.0,
        class_filter_idx=None,
        class_filter_thresh=0.1,
        random_scales=None,
    ):
        # class_filter_idx: when set AND training=True, drop train samples whose
        # GT label has ZERO pixels with `label[..., class_filter_idx] > class_filter_thresh`.
        # 0=bld, 1=veg, 2=water. Use for specialist training to skip negative-only tiles.
        # Val/test samples are NEVER filtered (eval distribution must match LB).
        self.class_filter_idx = int(class_filter_idx) if class_filter_idx is not None else None
        self.class_filter_thresh = float(class_filter_thresh)
        if self.class_filter_idx is not None:
            assert self.class_filter_idx in (0, 1, 2), \
                f"class_filter_idx must be 0/1/2, got {self.class_filter_idx}"
        # random_scales validation is below, AFTER self.random_crop_size/grid get set.
        self._pending_random_scales = random_scales  # capture for later validation
        # When True (training split only), hard-set ch0/1/2 labels (bld/veg/water)
        # to 1.0 where soft GT > harden_thresh. ch3 (height) and ch4 (loss_mask)
        # untouched. Val/test never hardened so val metric stays LB-comparable.
        self.harden_labels = bool(harden_labels)
        self.harden_thresh = float(harden_thresh)
        # Random crop augmentation: when training=True AND random_crop_size set,
        # crop both dense (image+target) and tokens to (random_crop_size, random_crop_size).
        # Token crop size = random_crop_size // random_crop_grid (16 for our 256:16 dense:token ratio).
        # Crop origin is snapped to multiples of `random_crop_grid` so dense and tokens stay aligned.
        # Active only during training (val/test always use full patch_size).
        self.random_crop_size = int(random_crop_size) if random_crop_size else None
        self.random_crop_grid = int(random_crop_grid)
        if self.random_crop_size is not None:
            assert self.random_crop_size > 0 and self.random_crop_size <= patch_size, \
                f"random_crop_size must be in (0, patch_size={patch_size}], got {self.random_crop_size}"
            assert self.random_crop_size % self.random_crop_grid == 0, \
                f"random_crop_size ({self.random_crop_size}) must be divisible by random_crop_grid ({self.random_crop_grid})"
            assert patch_size % self.random_crop_grid == 0, \
                f"patch_size ({patch_size}) must be divisible by random_crop_grid ({self.random_crop_grid})"
            if training:
                tok_sz = self.random_crop_size // self.random_crop_grid
                print(f"[GeoFMMultiEmbeddingDataset] random_crop_size={self.random_crop_size}, "
                      f"grid={self.random_crop_grid}, expected_token_size={tok_sz}x{tok_sz}",
                      flush=True)
        # random_scales: list of scale factors for multi-scale augmentation.
        # Each step: pick s ∈ random_scales, upsample dense+tokens+target+mask by s,
        # then _random_crop_aligned takes a random_crop_size crop in scaled space.
        # Scaling happens BEFORE harden (on soft labels) so the post-harden 0/1 is
        # clean at the scaled resolution (avoids the bilinear-of-binary halo bug).
        # All scales s must satisfy patch_size*s % random_crop_grid == 0.
        # Active only when training=True AND augment=True AND random_scales is set.
        _rs = getattr(self, "_pending_random_scales", None)
        self.random_scales = (tuple(float(s) for s in _rs) if _rs else None)
        if hasattr(self, "_pending_random_scales"):
            del self._pending_random_scales
        if self.random_scales is not None:
            assert self.random_crop_size is not None, \
                "random_scales requires random_crop_size to be set"
            # HIGH-severity guard: external building_mask_dir is patch_size-sized
            # and would NOT scale with the MS-scaled dense → silent crop misalignment.
            # Derive mask from target instead (default behavior when dir=None).
            assert building_mask_dir is None, (
                "random_scales is incompatible with building_mask_dir: external "
                "masks are at patch_size and would not scale with dense. Either "
                "disable MS aug or derive mask from target (set building_mask_dir=None)."
            )
            for s in self.random_scales:
                scaled_h = int(round(patch_size * s))
                assert scaled_h % self.random_crop_grid == 0, \
                    (f"random_scales: patch_size*s={patch_size}*{s}={scaled_h} not "
                     f"divisible by random_crop_grid={self.random_crop_grid}")
                assert scaled_h >= self.random_crop_size, \
                    (f"random_scales: scaled_h={scaled_h} < random_crop_size="
                     f"{self.random_crop_size} (need to scale UP, not DOWN)")
            if training:
                print(f"[GeoFMMultiEmbeddingDataset] random_scales={self.random_scales}",
                      flush=True)
        # ----- pseudo-label support -----
        # When set, pseudo pairs (test embeddings + pseudo-labels) are appended to
        # train ONLY (never val). The pseudo source must have a similar (B,H,W,C)
        # format as the real embedding/label dirs.
        self.pseudo_embedding_dirs = ([Path(d) for d in pseudo_embedding_dirs]
                                       if pseudo_embedding_dirs else None)
        self.pseudo_target_dir = Path(pseudo_target_dir) if pseudo_target_dir else None
        # Token version (V4/V7/V8 specialists need test tokens for pseudo IDs).
        self.pseudo_token_embedding_dirs = ([Path(d) for d in pseudo_token_embedding_dirs]
                                             if pseudo_token_embedding_dirs else None)
        # Transductive-overfit knobs (train split only; default no-op):
        #   pseudo_oversample: replicate the appended pseudo (test) pairs N× so they
        #     dominate the batch → model overfits the test pseudo targets.
        #   train_subsample_n: keep only N randomly-chosen real-train pairs (少量 train GT
        #     for feature stability) before oversampling pseudo.
        self.pseudo_oversample = int(pseudo_oversample)
        self.train_subsample_n = int(train_subsample_n) if train_subsample_n is not None else None
        # If True (train split only), use ALL real-labeled samples (no val holdout).
        # Val split still computes via val_fraction so save_best monitoring works.
        # Side effect: val samples leak into train, val miou is inflated.
        self.include_val_in_train = bool(include_val_in_train)
        # upsample_mode: how to handle patch_size > native source size.
        #   "reflect": pad reflection (current default — fine when patch_size matches native ~256)
        #   "bilinear": bilinear interpolate native to patch_size (use for super-resolution training)
        #   "nearest":  nearest-neighbor interpolate
        # When patch_size <= native size, no upsample needed (just crop as usual).
        self.upsample_mode = upsample_mode
        assert upsample_mode in ("reflect", "bilinear", "nearest"), \
            f"upsample_mode must be reflect/bilinear/nearest, got {upsample_mode}"
        self.embedding_dirs = [Path(d) for d in embedding_dirs]
        # Token (low-res, e.g. 16x16) sources — normalized + bilinear-upsampled to patch_size at load time.
        self.token_embedding_dirs = [Path(d) for d in (token_embedding_dirs or [])]
        token_stats_paths = list(token_stats_paths or [])
        if self.token_embedding_dirs and len(token_stats_paths) != len(self.token_embedding_dirs):
            raise ValueError(
                f"token_embedding_dirs has {len(self.token_embedding_dirs)} entries but "
                f"token_stats_paths has {len(token_stats_paths)} entries — must match."
            )
        self.token_clip_z = float(token_clip_z)
        # When True (default), tokens are bilinear-upsampled to patch_size at load
        # time and concatenated into the main image tensor (legacy behavior — works
        # for models that treat tokens as if dense). When False, tokens are kept at
        # native spatial resolution (e.g., 16×16) and returned separately in
        # meta["tokens"] — for late-fusion architectures that consume them at the
        # native scale.
        self.token_upsample = bool(token_upsample)
        # Load per-channel mean/std once (small arrays, host RAM)
        self._token_stats = []
        for sp in token_stats_paths:
            d = np.load(sp)
            self._token_stats.append((d["mean"].astype(np.float32),
                                      d["std"].astype(np.float32)))
        # RGB DINOv3-L cache (5th modality). Off by default — only active when
        # rgb_token_dir is set. Asymmetric: not all tiles have RGB cache (95% train,
        # 96% test). When absent, loader returns zero-token placeholder + has_rgb=False;
        # the model handles via learned absent_token.
        self.rgb_token_dir = Path(rgb_token_dir) if rgb_token_dir else None
        self.rgb_token_native_size = int(rgb_token_native_size)
        self.rgb_token_channels = int(rgb_token_channels)
        self.rgb_clip_z = float(rgb_clip_z)
        self._rgb_stats = None
        if self.rgb_token_dir is not None:
            if rgb_token_stats_path is None:
                raise ValueError(
                    "rgb_token_dir is set but rgb_token_stats_path is None. "
                    "Z-score stats are required for consistent train/test feature space."
                )
            d = np.load(rgb_token_stats_path)
            self._rgb_stats = (d["mean"].astype(np.float32),
                               d["std"].astype(np.float32))
            assert self._rgb_stats[0].shape == (self.rgb_token_channels,), \
                f"rgb stats mean shape {self._rgb_stats[0].shape} != ({self.rgb_token_channels},)"
            assert self._rgb_stats[1].shape == (self.rgb_token_channels,), \
                f"rgb stats std shape {self._rgb_stats[1].shape} != ({self.rgb_token_channels},)"
            # random_scales (multi-scale aug) + rgb_token is unsupported (would require
            # also scaling rgb cache native size — not implemented). Assert disabled.
            assert random_scales is None, (
                "rgb_token_dir is incompatible with random_scales (MS aug): "
                "rgb cache is at fixed native size and won't scale with dense."
            )
        self.target_dir = Path(target_dir) if target_dir else None
        # building_mask_dir: if set, load (4,256,256) HARD .npy from this dir and use
        # ch0 as the binary building mask. When None: mask is derived from GT target
        # at train time (training=True). For test/inference (training=False, target=None),
        # building_mask_dir MUST be set if the model requires a mask input.
        self.building_mask_dir = Path(building_mask_dir) if building_mask_dir else None
        self.building_mask_thresh = float(building_mask_thresh)
        # mask_channel_idx selects WHICH channel from target/cache to use as the
        # binary "mask" input. 0=bld (default, building height specialist), 1=veg
        # (veg height specialist), 2=water. The cached HARD .npy files in
        # building_mask_dir typically have all 3 seg channels (0=bld, 1=veg, 2=water);
        # mask_channel_idx picks which one.
        self.mask_channel_idx = int(mask_channel_idx)
        assert self.mask_channel_idx in (0, 1, 2), \
            f"mask_channel_idx must be 0 (bld) / 1 (veg) / 2 (water), got {self.mask_channel_idx}"
        # When True: load 2-channel mask (bld @ ch0 + veg @ ch1) — for joint bv-height specialist.
        # When False (default): single-channel mask via mask_channel_idx — backward compat.
        self.mask_two_channel = bool(mask_two_channel)
        # mask_dropout_p: probability of dropping a positive mask pixel during TRAINING augmentation.
        # Simulates false-negatives in inference mask (ensemble predicts less than GT). Applied to
        # the building_mask AFTER it's constructed but BEFORE random_crop/augment. Default 0.0 (off).
        # Only active when self.augment is True AND building_mask is not None.
        self.mask_dropout_p = float(mask_dropout_p)
        assert 0.0 <= self.mask_dropout_p < 1.0, \
            f"mask_dropout_p must be in [0, 1), got {self.mask_dropout_p}"
        self.split = split
        self.patch_size = patch_size
        self.training = training
        self.augment = augment and training
        self.height_norm_constant = float(height_norm_constant)
        self.dequantize_ae = bool(dequantize_ae)
        # Advanced augmentations — only active when training AND augment=True
        self.channel_dropout_p = float(channel_dropout_p) if self.augment else 0.0
        self.mixup_p = float(mixup_p) if self.augment else 0.0
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_p = float(cutmix_p) if self.augment else 0.0
        self.cutmix_alpha = float(cutmix_alpha)
        # Copy-Paste augmentation parameters
        self.copypaste_bank_path = str(copypaste_bank) if copypaste_bank else None
        self.copypaste_p = float(copypaste_p) if self.augment else 0.0
        self.copypaste_n_min = int(copypaste_n_min)
        self.copypaste_n_max = int(copypaste_n_max)
        self.copypaste_fg_thresh = float(copypaste_fg_thresh)
        self.copypaste_max_tries = int(copypaste_max_tries)

        if self.target_dir is None:
            # Test/inference mode: build samples from first embedding dir, match other dirs by canonical id.
            per_dir_maps = []
            for d in self.embedding_dirs:
                m = {}
                for p in list_embedding_files(d):
                    for key in candidate_core_ids(p):
                        if key not in m:
                            m[key] = p
                per_dir_maps.append(m)
            common = set(per_dir_maps[0])
            for m in per_dir_maps[1:]:
                common &= set(m)
            sorted_common = sorted(common)
            # Dedup by emb_paths tuple — same file can appear under multiple
            # candidate ids (e.g., "3001_BE_2023" and "3001_BE").
            self.samples = []
            self._sample_ids = []
            seen = set()
            for k in sorted_common:
                emb_paths = tuple(m[k] for m in per_dir_maps)
                if emb_paths in seen:
                    continue
                seen.add(emb_paths)
                self.samples.append((emb_paths, None))
                self._sample_ids.append(k)
        else:
            pairs = find_multi_file_pairs(self.embedding_dirs, self.target_dir)
            # Adapt split_pairs (which works on (emb_path, label) tuples) to multi-source.
            # Trick: pass a "fake" single-source pairing using first emb_dir for grouping.
            proxy = [(p[0][0], p[1]) for p in pairs]  # (first_emb, label)
            if split == "train" and self.include_val_in_train:
                # Use ALL real-labeled samples for training (no region holdout).
                # The val loader (split="val") still gets the original 20% subset for
                # save_best monitoring, but those val samples will leak into train too,
                # so val miou is no longer a clean held-out metric (still useful as
                # a convergence signal).
                kept_ids = set(normalize_core_id(p[0]) for p in proxy)
                print(f"  include_val_in_train=True: using ALL {len(proxy)} real-labeled samples for train")
            else:
                kept_proxy = split_pairs(proxy, split, val_fraction, split_seed, split_by)
                kept_ids = set(normalize_core_id(p[0]) for p in kept_proxy)
            self.samples = [p for p in pairs
                            if normalize_core_id(p[0][0]) in kept_ids]
            self._sample_ids = [normalize_core_id(p[0][0]) for p in self.samples]

            # === Pseudo-label augmentation: append test data with pseudo-labels ===
            # Only when training (split=="train"); never in val to keep eval honest.
            if (split == "train" and self.pseudo_embedding_dirs
                    and self.pseudo_target_dir is not None):
                pseudo_pairs = find_multi_file_pairs(self.pseudo_embedding_dirs,
                                                      self.pseudo_target_dir)
                pseudo_ids = [normalize_core_id(p[0][0]) for p in pseudo_pairs]
                self.samples.extend(pseudo_pairs)
                self._sample_ids.extend(pseudo_ids)
                print(f"  +pseudo: appended {len(pseudo_pairs)} pairs from "
                      f"{self.pseudo_embedding_dirs} + {self.pseudo_target_dir}")

                # Transductive overfit: subsample real train + oversample pseudo (test).
                if self.train_subsample_n is not None or self.pseudo_oversample > 1:
                    import random as _random
                    n_pseudo = len(pseudo_pairs)
                    n_real = len(self.samples) - n_pseudo
                    real_s, real_i = self.samples[:n_real], self._sample_ids[:n_real]
                    pse_s, pse_i = self.samples[n_real:], self._sample_ids[n_real:]
                    if self.train_subsample_n is not None and self.train_subsample_n < len(real_s):
                        rng2 = _random.Random(split_seed)
                        keep = rng2.sample(range(len(real_s)), self.train_subsample_n)
                        real_s = [real_s[i] for i in keep]
                        real_i = [real_i[i] for i in keep]
                    if self.pseudo_oversample > 1:
                        pse_s = pse_s * self.pseudo_oversample
                        pse_i = pse_i * self.pseudo_oversample
                    self.samples = real_s + pse_s
                    self._sample_ids = real_i + pse_i
                    print(f"  overfit-mix: real {len(real_s)} + pseudo {len(pse_s)} "
                          f"(x{self.pseudo_oversample}) = {len(self.samples)} train samples")

        if not self.samples:
            raise FileNotFoundError(
                f"No multi-emb samples found for dirs={self.embedding_dirs}, "
                f"target_dir={self.target_dir}, split={split}."
            )

        # === Optional class-positive filter (training only) ===
        # Drops samples whose label has zero pixels in the target class.
        # Used for specialist training to skip negative-only tiles.
        if self.class_filter_idx is not None and self.training and self.target_dir is not None:
            kept_samples, kept_ids = [], []
            n_before = len(self.samples)
            for (emb_paths, lab_path), cid in zip(self.samples, self._sample_ids):
                lab = _read_tif(lab_path)  # (4, H, W) — bld/veg/water/height
                if (lab[self.class_filter_idx] > self.class_filter_thresh).any():
                    kept_samples.append((emb_paths, lab_path))
                    kept_ids.append(cid)
            self.samples = kept_samples
            self._sample_ids = kept_ids
            cls_name = {0: "bld", 1: "veg", 2: "water"}[self.class_filter_idx]
            print(f"  class_filter_idx={self.class_filter_idx} ({cls_name}>{self.class_filter_thresh}): "
                  f"{n_before} → {len(self.samples)} samples "
                  f"({n_before - len(self.samples)} dropped)", flush=True)

        # === Token sources: pair token paths to each sample by canonical id ===
        # Token paths live in a parallel list indexed by sample idx so the existing
        # (emb_paths, target_path) sample tuple format stays unchanged.
        # Pseudo samples (test IDs) need to look up tokens in TEST token dirs since
        # train token dirs don't contain test IDs — see pseudo_token_embedding_dirs.
        self.token_paths = None
        if self.token_embedding_dirs:
            token_map = find_token_paths_for_ids(self.token_embedding_dirs, set(self._sample_ids))
            # Also search pseudo token dirs for pseudo IDs (test IDs) if provided.
            if self.pseudo_token_embedding_dirs:
                missing_now = [cid for cid in self._sample_ids if cid not in token_map]
                pseudo_token_map = find_token_paths_for_ids(
                    self.pseudo_token_embedding_dirs, set(missing_now))
                token_map.update(pseudo_token_map)
            missing = [cid for cid in self._sample_ids if cid not in token_map]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} samples missing token files in {self.token_embedding_dirs}"
                    + (f" or pseudo dirs {self.pseudo_token_embedding_dirs}" if self.pseudo_token_embedding_dirs else "")
                    + f". first missing: {missing[:3]}"
                )
            self.token_paths = [token_map[cid] for cid in self._sample_ids]

        # Sanity: get total channel count
        first_imgs = [_read_tif(p) for p in self.samples[0][0]]
        self.per_source_channels = [im.shape[0] for im in first_imgs]
        # All DENSE sources must be pixel-level same H,W
        for im in first_imgs[1:]:
            assert im.shape[1] == first_imgs[0].shape[1] and im.shape[2] == first_imgs[0].shape[2], \
                "All dense embedding sources must share spatial dims (use pixel-level only)."

        # Token sources contribute their channel count to per_source_channels (in order).
        # Their spatial dims (e.g. 16x16) differ from the dense ones — that's intentional;
        # _load_sample upsamples them to dense patch size before concat.
        if self.token_paths:
            first_tokens = [_read_tif(p) for p in self.token_paths[0]]
            for ti, im in enumerate(first_tokens):
                self.per_source_channels.append(im.shape[0])
                _ps = self.patch_size
                _route = (f"upsampled to {_ps}x{_ps}" if self.token_upsample
                          else "KEPT NATIVE (meta['tokens'])")
                print(f"  token source [{ti}]: channels={im.shape[0]} "
                      f"native_spatial={im.shape[1]}x{im.shape[2]} → {_route}")
        self.in_channels = sum(self.per_source_channels)

    def __len__(self):
        return len(self.samples)

    def _crop_pixel(self, image, target):
        # Optional: upsample native to patch_size BEFORE pad/crop, when image native
        # spatial size < patch_size and upsample_mode != "reflect". Useful for 256→512
        # super-resolution training where we want bilinear interpolation, not reflection padding.
        _, h0, w0 = image.shape
        if self.upsample_mode in ("bilinear", "nearest") and (h0 < self.patch_size or w0 < self.patch_size):
            mode = self.upsample_mode
            align_corners = False if mode == "bilinear" else None
            t_img = torch.from_numpy(image).unsqueeze(0).float()
            t_img = nn.functional.interpolate(t_img, size=(self.patch_size, self.patch_size),
                                              mode=mode, align_corners=align_corners)
            image = t_img.squeeze(0).numpy()
            if target is not None:
                # Targets are soft probabilities (seg) + continuous heights — bilinear is fine
                t_tgt = torch.from_numpy(target).unsqueeze(0).float()
                t_tgt = nn.functional.interpolate(t_tgt, size=(self.patch_size, self.patch_size),
                                                  mode="bilinear", align_corners=False)
                target = t_tgt.squeeze(0).numpy()
                target[:3] = np.clip(target[:3], 0.0, 1.0)   # keep seg in [0, 1]

        image = _pad_reflect(image, self.patch_size, self.patch_size)
        if target is not None:
            target = _pad_reflect(target, self.patch_size, self.patch_size)
        _, h, w = image.shape
        if self.training:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2
        image = image[:, top:top + self.patch_size, left:left + self.patch_size]
        if target is not None:
            target = target[:, top:top + self.patch_size, left:left + self.patch_size]
        return image, target

    def _random_crop_aligned(self, image, target, native_tokens, building_mask=None,
                             rgb_token=None):
        """Random crop dense (image+target+mask) to (random_crop_size, random_crop_size)
        AND tokens to (cs//grid, cs//grid). Crop origin is on `random_crop_grid` boundary
        so dense/token spatial correspondence stays exact.

        If rgb_token (high-spatial DINOv3 cache, e.g. 1024×160×160) is provided, also
        crops it to a co-registered window. Math: token_anchor = dense_anchor * R / H
        where R is rgb_token native size (e.g. 160 for 2560/16). With patch_size=256,
        random_crop_size=192, random_crop_grid=16, R=160 → token_anchor in {0,10,20,30,40}
        and crop size 120 — exact integers (asserted).

        Only called during training when random_crop_size is set.
        Returns (image, target, native_tokens, building_mask, rgb_token), all cropped.
        """
        cs = self.random_crop_size
        g = self.random_crop_grid
        _, H, W = image.shape
        # Valid crop origins on grid: y1, x1 in {0, g, 2g, ..., H-cs}. When cs==H this
        # collapses to {0} (randint(0,1)=0), correctly degenerating to identity.
        max_y_grid = (H - cs) // g
        max_x_grid = (W - cs) // g
        y1 = np.random.randint(0, max_y_grid + 1) * g
        x1 = np.random.randint(0, max_x_grid + 1) * g

        image = image[:, y1:y1 + cs, x1:x1 + cs]
        if target is not None:
            target = target[:, y1:y1 + cs, x1:x1 + cs]
        if building_mask is not None:
            building_mask = building_mask[:, y1:y1 + cs, x1:x1 + cs]
        if native_tokens is not None:
            # Tokens: each token covers `grid` dense pixels. Native token spatial size = H/grid.
            ty = y1 // g
            tx = x1 // g
            tw = cs // g
            _, tok_h, tok_w = native_tokens.shape
            # Safety: tokens must actually have spatial dim H/g (e.g. 16 for grid=16, H=256).
            # If not, the dense:token ratio assumption is wrong → bail.
            expected_tok = H // g
            assert tok_h == expected_tok and tok_w == expected_tok, (
                f"random_crop_aligned: token spatial ({tok_h}x{tok_w}) != expected "
                f"({expected_tok}x{expected_tok}) for dense H={H}, grid={g}. "
                f"Check token_upsample=False and patch_size matches token native size × grid.")
            native_tokens = native_tokens[:, ty:ty + tw, tx:tx + tw]
        if rgb_token is not None:
            # RGB native spatial R from shape (assumed square H_r == W_r).
            _, R_h, R_w = rgb_token.shape
            assert R_h == R_w, f"rgb_token must be square (got {R_h}×{R_w})"
            R = R_h
            # Integer-exact alignment guard
            assert (y1 * R) % H == 0 and (x1 * R) % H == 0 and (cs * R) % H == 0, (
                f"rgb crop integer-exact math broken: y1={y1} x1={x1} cs={cs} R={R} H={H}. "
                f"Required: y1·R, x1·R, cs·R all divisible by H.")
            y_r = (y1 * R) // H
            x_r = (x1 * R) // H
            s_r = (cs * R) // H
            rgb_token = rgb_token[:, y_r:y_r + s_r, x_r:x_r + s_r]
        return image, target, native_tokens, building_mask, rgb_token

    def _load_sample(self, index):
        """Load one sample's image + target after crop, flip/rot — no mixup/cutmix yet.

        Returns:
          image: (C, H, W) — dense sources (and tokens too if self.token_upsample=True)
          target: (C', H, W) or None
          emb_paths, target_path
          native_tokens: (sum_token_ch, h_tok, w_tok) numpy array if tokens are kept
                         native (self.token_upsample=False), else None.
        """
        emb_paths, target_path = self.samples[index]
        imgs = [np.nan_to_num(_read_tif_dequantize(p, dequantize_ae=self.dequantize_ae)) for p in emb_paths]
        native_tokens = None
        # Match token spatial dims to the FIRST dense source's actual H,W (not patch_size,
        # because some dense files are e.g. 255x255 natively and only get padded to
        # patch_size later inside _crop_pixel).
        if self.token_paths is not None:
            target_h, target_w = imgs[0].shape[1], imgs[0].shape[2]
            token_paths = self.token_paths[index]
            if self.token_upsample:
                # Legacy: bilinear-upsample tokens to dense H,W and concat into image
                for tp, (mean, std) in zip(token_paths, self._token_stats):
                    t = _read_token_normalize_upsample(
                        tp, mean, std, out_size=(target_h, target_w), clip_z=self.token_clip_z)
                    imgs.append(t)
            else:
                # Late-fusion mode: keep tokens at native spatial size, return separately
                tok_parts = []
                for tp, (mean, std) in zip(token_paths, self._token_stats):
                    with rasterio.open(tp) as src:
                        t = src.read().astype(np.float32)   # (C, h_tok, w_tok)
                    t = np.nan_to_num(t, nan=0.0)
                    t = (t - mean[:, None, None]) / np.maximum(std[:, None, None], 1e-6)
                    t = np.clip(t, -self.token_clip_z, self.token_clip_z)
                    tok_parts.append(t)
                native_tokens = np.concatenate(tok_parts, axis=0).astype(np.float32)
        image = np.concatenate(imgs, axis=0).astype(np.float32)

        target = None
        if target_path is not None:
            target = np.nan_to_num(_read_tif(target_path))
            # Detect 5-channel pseudo label: 4 prediction + 1 loss_mask.
            # For uniform downstream handling, ALWAYS make target 5-channel:
            #   ch0-3: bld, veg, water, height_normalized
            #   ch4:   loss_mask (1=use, 0=ignore). Real labels get all-1 mask.
            if target.shape[0] == 4:
                # Real label — append all-1 mask
                ones = np.ones((1, target.shape[1], target.shape[2]), dtype=target.dtype)
                target = np.concatenate([target, ones], axis=0)
            elif target.shape[0] != 5:
                raise ValueError(f"Unexpected label channels {target.shape[0]} (must be 4 or 5)")
            # Normalize height (ch3 only, never touch the mask channel)
            target[3] = np.clip(target[3] / self.height_norm_constant, 0.0, 1.5)
            # NOTE: harden_labels moved DOWN to after random_scales scaling — so that
            # bilinear-upsampling soft labels (in [0,1]) → then harden_thresh produces
            # clean 0/1 binary at the scaled resolution. Prevents the "halo" bug from
            # bilinear-interpolating an already-hardened 0/1 mask.

        image, target = self._crop_pixel(image, target)

        # Multi-scale augmentation (TRAIN only): randomly upsample dense + tokens +
        # target + building_mask by s ∈ self.random_scales. Then the later
        # _random_crop_aligned takes a random_crop_size×random_crop_size crop in
        # scaled space. Tokens MUST scale by same s to keep dense:token ratio = grid.
        if (self.training and self.augment and self.random_scales is not None):
            s = float(np.random.choice(self.random_scales))
            if s != 1.0:
                new_dense_h = int(round(self.patch_size * s))
                new_token_h = new_dense_h // self.random_crop_grid
                import torch
                import torch.nn.functional as _F
                # dense (C, H, W) → bilinear
                t_image = torch.from_numpy(image).unsqueeze(0)
                image = _F.interpolate(t_image, size=(new_dense_h, new_dense_h),
                                       mode='bilinear', align_corners=False
                                       ).squeeze(0).numpy()
                # target (5, H, W) — bilinear OK because still soft (harden not yet applied)
                if target is not None:
                    t_target = torch.from_numpy(target).unsqueeze(0)
                    target = _F.interpolate(t_target, size=(new_dense_h, new_dense_h),
                                            mode='bilinear', align_corners=False
                                            ).squeeze(0).numpy()
                # native_tokens (C, h_tok, w_tok) → bilinear to (new_token_h, new_token_h)
                if native_tokens is not None:
                    t_tok = torch.from_numpy(native_tokens).unsqueeze(0)
                    native_tokens = _F.interpolate(t_tok, size=(new_token_h, new_token_h),
                                                   mode='bilinear', align_corners=False
                                                   ).squeeze(0).numpy()
                # Fix: re-binarize loss_mask (target[4]) after bilinear made it soft.
                # loss_mask must stay 0/1 because the loss multiplies by it (soft values
                # would dilute supervision near boundaries).
                if target is not None and target.shape[0] >= 5:
                    target[4] = (target[4] > 0.5).astype(target.dtype)

        # Optional GT hardening for seg channels (training only). NOW operates on
        # scaled soft labels (still in [0,1]) → clean 0/1 binary at scaled resolution.
        if (self.harden_labels and self.training and target is not None):
            pos_b = target[0] > self.harden_thresh
            pos_v = target[1] > self.harden_thresh
            pos_w = target[2] > self.harden_thresh
            target[0] = np.where(pos_b, 1.0, 0.0).astype(target.dtype)
            target[1] = np.where(pos_v, 1.0, 0.0).astype(target.dtype)
            target[2] = np.where(pos_w, 1.0, 0.0).astype(target.dtype)

        # Build building_mask (1, H, W) — required for mask-conditioned height specialists.
        # Sources (priority order):
        #   1) building_mask_dir set → load (4,H,W) HARD .npy, use ch0 (inference / test)
        #   2) target available → derive from target[:, 0:1] post-hardening (train / val)
        #   3) None → model that doesn't need a mask still works
        building_mask = None
        sample_id = normalize_core_id(emb_paths[0])
        if self.building_mask_dir is not None:
            mask_path = self.building_mask_dir / f"{sample_id}.npy"
            mask_arr = np.load(mask_path)  # expected (>=3, 256, 256) HARD float32
            if self.mask_two_channel:
                # Load ch0 (bld) + ch1 (veg) as 2-ch mask
                assert mask_arr.shape[0] >= 2 and mask_arr.shape[1:] == (self.patch_size, self.patch_size), \
                    f"mask file {mask_path} shape {mask_arr.shape} can't index ch0+ch1 at ({self.patch_size}, {self.patch_size})"
                building_mask = mask_arr[0:2].astype(np.float32)
            else:
                c = self.mask_channel_idx
                assert mask_arr.shape[0] > c and mask_arr.shape[1:] == (self.patch_size, self.patch_size), \
                    f"mask file {mask_path} shape {mask_arr.shape} can't index ch{c} at ({self.patch_size}, {self.patch_size})"
                building_mask = mask_arr[c:c + 1].astype(np.float32)
        elif target is not None:
            # Train/val path: derive from GT class channel (already harden-thresholded
            # at thresh=0.1 in the loader). Apply threshold to be safe (handles both
            # already-binary and soft-label cases).
            if self.mask_two_channel:
                # Derive 2-ch mask: ch0=bld, ch1=veg
                building_mask = (target[0:2] > self.building_mask_thresh).astype(np.float32)
            else:
                c = self.mask_channel_idx
                building_mask = (target[c:c + 1] > self.building_mask_thresh).astype(np.float32)

        # Mask dropout augmentation (TRAIN only). Drops `mask_dropout_p` fraction of
        # positive pixels in building_mask. Simulates false-negatives in the ensemble
        # mask at inference time (predicted mask covers less than GT). Applied BEFORE
        # crop/aug so spatial transforms still see consistent (image, target, mask).
        # Negative pixels (mask=0) are NOT touched (would require separate add_p).
        if (self.augment and self.mask_dropout_p > 0.0 and building_mask is not None):
            # Per-pixel Bernoulli mask: 1 = keep, 0 = drop
            keep = (np.random.random(building_mask.shape) >= self.mask_dropout_p).astype(np.float32)
            building_mask = building_mask * keep

        # === Load RGB DINOv3-L token cache (5th modality), if enabled ===
        # Cache filename: dinov3l_<sample_id>.npy where sample_id is normalized AE core id
        # (e.g. "0000_BE" for train, "3001_BE_2023" for test). 95-96% of tiles have RGB;
        # for the rest, return a zero placeholder + has_rgb=False — model uses absent_token.
        rgb_token = None
        has_rgb = False
        if self.rgb_token_dir is not None:
            R = self.rgb_token_native_size
            C_rgb = self.rgb_token_channels
            rgb_path = self.rgb_token_dir / f"dinov3l_{sample_id}.npy"
            if rgb_path.exists():
                try:
                    arr = np.load(rgb_path)
                except Exception as e:
                    print(f"[WARN] failed to load rgb cache {rgb_path}: {e}", flush=True)
                    arr = None
            else:
                arr = None
            if arr is not None and arr.shape == (C_rgb, R, R):
                rgb_token = arr.astype(np.float32)
                # z-score (per-channel) + clip
                mean, std = self._rgb_stats
                rgb_token = (rgb_token - mean[:, None, None]) / np.maximum(std[:, None, None], 1e-6)
                np.clip(rgb_token, -self.rgb_clip_z, self.rgb_clip_z, out=rgb_token)
                has_rgb = True
            else:
                if arr is not None:
                    print(f"[WARN] rgb cache shape mismatch at {rgb_path}: "
                          f"got {arr.shape}, expected ({C_rgb},{R},{R})", flush=True)
                rgb_token = np.zeros((C_rgb, R, R), dtype=np.float32)
                has_rgb = False

        # Secondary random crop (training only): cut a (random_crop_size)^2 region from
        # the post-_crop_pixel patch and sync tokens accordingly. Keeps grid:1 dense:token
        # ratio so late-fusion stays aligned. Val/test (training=False) skips this.
        if self.training and self.random_crop_size is not None:
            image, target, native_tokens, building_mask, rgb_token = self._random_crop_aligned(
                image, target, native_tokens, building_mask, rgb_token=rgb_token
            )

        if self.augment:
            # Save random state BEFORE image aug; restore for each aligned tensor so
            # all spatial tensors see identical flip/rot decisions.
            rng_state = random.getstate()
            image, target = _augment_pair(image, target)
            if native_tokens is not None:
                random.setstate(rng_state)
                native_tokens, _ = _augment_pair(native_tokens, None)
            if building_mask is not None:
                random.setstate(rng_state)
                building_mask, _ = _augment_pair(building_mask, None)
            if rgb_token is not None:
                random.setstate(rng_state)
                rgb_token, _ = _augment_pair(rgb_token, None)

        # Make contiguous (after possible negative strides from flip/rot)
        image = np.ascontiguousarray(image)
        if target is not None:
            target = np.ascontiguousarray(target)
        if native_tokens is not None:
            native_tokens = np.ascontiguousarray(native_tokens)
        if building_mask is not None:
            building_mask = np.ascontiguousarray(building_mask)
        if rgb_token is not None:
            rgb_token = np.ascontiguousarray(rgb_token)
        return image, target, emb_paths, target_path, native_tokens, building_mask, rgb_token, has_rgb

    def __getitem__(self, index):
        (image, target, emb_paths, target_path, native_tokens, building_mask,
         rgb_token, has_rgb) = self._load_sample(index)

        # === Pair-based augmentations (mixup / cutmix) ===
        # Only mix if training AND target is available (mixup needs labels).
        # NOTE: mixup/cutmix do NOT update building_mask or rgb_token — disable these
        # augs when using mask-conditioned or RGB-fused models (config keeps p=0).
        # Star-unpack `*_` tolerates any future return-arity growth from _load_sample.
        if self.augment and target is not None and len(self.samples) > 1:
            r = random.random()
            if r < self.mixup_p:
                # Mixup with a different random sample
                j = index
                while j == index:
                    j = random.randint(0, len(self.samples) - 1)
                img2, tgt2, *_ = self._load_sample(j)
                image, target, _ = _mixup_pair(image, target, img2, tgt2, alpha=self.mixup_alpha)
            elif r < self.mixup_p + self.cutmix_p:
                # CutMix with a different random sample
                j = index
                while j == index:
                    j = random.randint(0, len(self.samples) - 1)
                img2, tgt2, *_ = self._load_sample(j)
                image, target, _ = _cutmix_pair(image, target, img2, tgt2, alpha=self.cutmix_alpha)

        # === Copy-Paste buildings (background-only paste) ===
        # Applied AFTER mixup/cutmix so pasted buildings keep crisp labels.
        if (self.augment
                and target is not None
                and self.copypaste_p > 0.0
                and self.copypaste_bank_path is not None
                and random.random() < self.copypaste_p):
            bank = _get_building_bank(self.copypaste_bank_path)
            image, target, _ = _copypaste_buildings(
                image, target, bank, self.per_source_channels,
                height_norm_constant=self.height_norm_constant,
                fg_thresh=self.copypaste_fg_thresh,
                n_min=self.copypaste_n_min,
                n_max=self.copypaste_n_max,
                max_tries=self.copypaste_max_tries,
            )

        # === Single-sample augmentations (channel dropout) ===
        # Applied AFTER mixup/cutmix to also affect mixed images.
        if self.augment and self.channel_dropout_p > 0:
            image = _channel_dropout(image, p=self.channel_dropout_p)

        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        meta = dict(
            id=normalize_core_id(emb_paths[0]),
            embedding_paths=[str(p) for p in emb_paths],
            is_latent=False,
        )
        if target is not None:
            # Split off the trailing mask channel (always present after _load_sample)
            assert target.shape[0] == 5, f"target should be 5-ch, got {target.shape[0]}"
            tgt_4ch = target[:4]
            loss_mask = target[4:5]   # keep (1, H, W) for easy broadcasting in loss
            meta["target"] = torch.from_numpy(np.ascontiguousarray(tgt_4ch)).float()
            meta["loss_mask"] = torch.from_numpy(np.ascontiguousarray(loss_mask)).float()
            meta["target_path"] = str(target_path)
        # Late-fusion mode: tokens at native resolution go in meta dict
        if native_tokens is not None:
            meta["tokens"] = torch.from_numpy(np.ascontiguousarray(native_tokens)).float()
        # Mask-conditioned height specialist: (1, H, W) binary building mask
        if building_mask is not None:
            meta["building_mask"] = torch.from_numpy(np.ascontiguousarray(building_mask)).float()
        # RGB DINOv3-L token (when rgb_token_dir is set). Always include in meta
        # (zero placeholder for absent tiles); model uses absent_token via has_rgb gate.
        if rgb_token is not None:
            meta["rgb_token"] = torch.from_numpy(np.ascontiguousarray(rgb_token)).float()
            meta["has_rgb"] = torch.tensor(bool(has_rgb), dtype=torch.bool)
        return image, meta


@er.registry.DATALOADER.register()
class GeoFMMultiEmbeddingLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)

        dataset = GeoFMMultiEmbeddingDataset(
            embedding_dirs=self.config.embedding_dirs,
            target_dir=self.config.target_dir,
            split=self.config.split,
            val_fraction=self.config.val_fraction,
            split_seed=self.config.split_seed,
            split_by=self.config.split_by,
            patch_size=self.config.patch_size,
            training=self.config.training,
            augment=self.config.augment,
            height_norm_constant=self.config.height_norm_constant,
            dequantize_ae=getattr(self.config, "dequantize_ae", False),  # match training raw int8 AE (changed 2026-05-19)
            channel_dropout_p=getattr(self.config, "channel_dropout_p", 0.0),
            mixup_p=getattr(self.config, "mixup_p", 0.0),
            mixup_alpha=getattr(self.config, "mixup_alpha", 0.2),
            cutmix_p=getattr(self.config, "cutmix_p", 0.0),
            cutmix_alpha=getattr(self.config, "cutmix_alpha", 1.0),
            copypaste_bank=getattr(self.config, "copypaste_bank", None),
            copypaste_p=getattr(self.config, "copypaste_p", 0.0),
            copypaste_n_min=getattr(self.config, "copypaste_n_min", 1),
            copypaste_n_max=getattr(self.config, "copypaste_n_max", 3),
            copypaste_fg_thresh=getattr(self.config, "copypaste_fg_thresh", 0.10),
            copypaste_max_tries=getattr(self.config, "copypaste_max_tries", 10),
            token_embedding_dirs=getattr(self.config, "token_embedding_dirs", None),
            token_stats_paths=getattr(self.config, "token_stats_paths", None),
            token_clip_z=getattr(self.config, "token_clip_z", 10.0),
            token_upsample=getattr(self.config, "token_upsample", True),
            upsample_mode=getattr(self.config, "upsample_mode", "reflect"),
            # RGB DINOv3-L cache (5th modality). Default None → backward-compat (off).
            rgb_token_dir=getattr(self.config, "rgb_token_dir", None),
            rgb_token_stats_path=getattr(self.config, "rgb_token_stats_path", None),
            rgb_token_native_size=getattr(self.config, "rgb_token_native_size", 160),
            rgb_token_channels=getattr(self.config, "rgb_token_channels", 1024),
            rgb_clip_z=getattr(self.config, "rgb_clip_z", 10.0),
            pseudo_embedding_dirs=getattr(self.config, "pseudo_embedding_dirs", None),
            pseudo_target_dir=getattr(self.config, "pseudo_target_dir", None),
            pseudo_token_embedding_dirs=getattr(self.config, "pseudo_token_embedding_dirs", None),
            pseudo_oversample=getattr(self.config, "pseudo_oversample", 1),
            train_subsample_n=getattr(self.config, "train_subsample_n", None),
            include_val_in_train=getattr(self.config, "include_val_in_train", False),
            harden_labels=getattr(self.config, "harden_labels", False),
            harden_thresh=getattr(self.config, "harden_thresh", 0.1),
            random_crop_size=getattr(self.config, "random_crop_size", None),
            random_crop_grid=getattr(self.config, "random_crop_grid", 16),
            building_mask_dir=getattr(self.config, "building_mask_dir", None),
            building_mask_thresh=getattr(self.config, "building_mask_thresh", 0.5),
            mask_channel_idx=getattr(self.config, "mask_channel_idx", 0),
            mask_two_channel=getattr(self.config, "mask_two_channel", False),
            mask_dropout_p=getattr(self.config, "mask_dropout_p", 0.0),
            class_filter_idx=getattr(self.config, "class_filter_idx", None),
            class_filter_thresh=getattr(self.config, "class_filter_thresh", 0.1),
            random_scales=getattr(self.config, "random_scales", None),
        )

        if self.config.training:
            sampler = distributed.StepDistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        super().__init__(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )

    def set_default_config(self):
        self.config.update(
            dict(
                embedding_dirs=[],
                target_dir=None,
                split="train",
                val_fraction=0.2,
                split_seed=42,
                split_by="region",
                patch_size=256,
                batch_size=4,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                training=True,
                augment=True,
                height_norm_constant=HEIGHT_NORM_CONSTANT,
            )
        )

