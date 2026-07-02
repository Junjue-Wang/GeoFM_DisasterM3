"""Standalone RGB-only loader for the DPT building specialist.

Input per tile:
  - 1m RGB tif at `<rgb_dir>/<rgb_filename>` (2560×2560×3 uint8)
  - Label tif at `<labels_dir>/label_<full_id>.tif` (256×256×{4,5} float)

Output per __getitem__ (NATIVE-LABEL-RESOLUTION scheme — label is NEVER upsampled):
  TRAIN (training=True):
    image: torch.FloatTensor (3, 640, 640) — normalized RGB, random 640 crop.
           Crop offset (oy,ox) are multiples of 10 (RGB:label ratio 10:1) so the
           label aligns exactly: oy = 10*randint(0,192), ox = 10*randint(0,192).
    target: (4, 64, 64) — native label cropped at [oy//10:oy//10+64, ox//10:...].
    loss_mask: (1, 64, 64) all-1.
  VAL/TEST (training=False): FULL tile, no crop.
    image: (3, 2560, 2560) normalized RGB.
    target: (4, 256, 256) native label (soft, NOT hardened).
    loss_mask: (1, 256, 256) all-1.
    (Model side does sliding-window inference window=640/stride=384.)
  meta: dict with
    target: see above. seg ch0/1/2 hardened @harden_thresh on TRAIN crops only.
    loss_mask: all-1 (no pseudo-label)
    id: full_id string
    embedding_paths: list of RGB path (for downstream id discovery)
    target_path: label path string

This Loader is REGISTERED with EVER (DATALOADER registry) so EVER can dispatch
on `type` in config.
"""
import glob
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

try:
    import ever as er
    from ever.api.data import distributed
    from ever.interface import ConfigurableMixin
except ModuleNotFoundError:
    class _Registry:
        def register(self, *args, **kwargs):
            def _decorator(cls): return cls
            return _decorator
    from types import SimpleNamespace
    class _ER:
        registry = SimpleNamespace(DATALOADER=_Registry())
    class ConfigurableMixin:
        def __init__(self, config): self.config = config
    class _Distributed:
        StepDistributedSampler = RandomSampler
    er = _ER()
    distributed = _Distributed()


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HEIGHT_NORM_CONSTANT = 30.0


def _read_label(path):
    """Read native label tif (256, 256, 4 or 5 channels) → (C, H, W) fp32.
    Returns 4-channel: ch0=bld, ch1=veg, ch2=water, ch3=height_norm (÷30, clip)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)   # (C, H, W)
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.shape[0] == 5:
        arr = arr[:4]
    elif arr.shape[0] != 4:
        raise ValueError(f"unexpected label channels {arr.shape[0]} in {path}")
    # Height normalize (ch3 ÷ 30, clip [0, 1.5])
    arr[3] = np.clip(arr[3] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)
    return arr


def _resize_label_to(label_chw, target_size):
    """Snap a near-256 label (4, h, w) to exactly (4, target_size, target_size).
    seg channels (0/1/2): NEAREST. height (3): bilinear.

    Used only for the ~12% of label tifs that are 255×256 / 256×255 / 255×255
    (see __getitem__). NEAREST is intentional for the seg channels: it preserves
    the soft GT fraction VALUES (no interpolation toward 0/1), keeping labels
    CONTINUOUS under harden_labels=False, and matches how DPT v2 handled non-256
    labels. Bilinear on the (already continuous) height channel is fine."""
    src = torch.from_numpy(label_chw).unsqueeze(0)   # (1, 4, h, w)
    seg = F.interpolate(src[:, :3], size=(target_size, target_size),
                        mode='nearest')
    h = F.interpolate(src[:, 3:4], size=(target_size, target_size),
                      mode='bilinear', align_corners=False)
    out = torch.cat([seg, h], dim=1).squeeze(0).numpy().astype(np.float32)
    return out


def _read_rgb(path, expected_size=2560):
    """Read 1m RGB tif as (3, H, W) uint8 ndarray."""
    arr = np.asarray(Image.open(path))
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"bad RGB {path}: shape={arr.shape} dtype={arr.dtype}")
    if arr.shape[0] != expected_size or arr.shape[1] != expected_size:
        raise ValueError(f"bad RGB size {path}: {arr.shape[:2]} != {expected_size}")
    return arr.transpose(2, 0, 1)   # (3, H, W) uint8


def _normalize_rgb(rgb_chw_u8):
    """uint8 (3, H, W) → fp32 normalized (3, H, W)."""
    x = rgb_chw_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return x


def _augment_d4(image, target):
    """D4 group augmentation, applied identically to image and target.
    image: (C, H, W), target: (4, H, W). Both numpy."""
    if random.random() < 0.5:
        image = image[:, :, ::-1]
        target = target[:, :, ::-1]
    if random.random() < 0.5:
        image = image[:, ::-1, :]
        target = target[:, ::-1, :]
    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k=k, axes=(1, 2))
        target = np.rot90(target, k=k, axes=(1, 2))
    return image, target


class BuildingSpecialistRGBDataset(Dataset):
    """RGB-only single-class building specialist dataset (NATIVE-LABEL scheme).

    Loads (RGB, label) pairs at native resolutions (RGB 2560, label 256; ratio
    10:1). The label is NEVER upsampled.
      TRAIN: random 640 RGB crop with offset a multiple of 10; the matching
             64×64 native-label window is cropped at offset//10. seg labels
             hardened @harden_thresh post-crop. D4 aug applied jointly.
      VAL/TEST: full tile (RGB 2560, label 256, soft) for sliding-window infer.

    Constructor params:
      input_size (int=640): TRAIN random-crop size on RGB. Must be a multiple of
        10 and <= rgb_native_size; the label crop size is input_size//10.
      rgb_native_size (int=2560): native RGB tile size.
      Extra kwargs are accepted and ignored (backward-tolerant; e.g. legacy
      val_center_crop / label_native_size / split_by are no-ops in this scheme).
    """

    def __init__(self,
                 rgb_dir,
                 labels_dir,
                 alignment_json,
                 split="train",
                 val_fraction=0.2,
                 split_by="region",
                 split_seed=42,
                 input_size=640,         # train: random crop size on RGB (multiple of 10)
                 rgb_native_size=2560,
                 training=True,
                 augment=True,
                 harden_labels=True,
                 harden_thresh=0.1,
                 class_filter_idx=0,
                 class_filter_thresh=0.005,
                 **kwargs):              # backward-tolerant: ignore legacy/extra kwargs
        self.rgb_dir = Path(rgb_dir)
        self.labels_dir = Path(labels_dir)
        self.input_size = int(input_size)
        self.rgb_native = int(rgb_native_size)
        # RGB:label ratio is 10:1. Validate crop geometry up front.
        self.rgb_to_lbl_ratio = 10
        if self.input_size % self.rgb_to_lbl_ratio != 0:
            raise ValueError(
                f"input_size {self.input_size} must be a multiple of "
                f"{self.rgb_to_lbl_ratio} (RGB:label ratio)")
        if self.input_size > self.rgb_native:
            raise ValueError(
                f"input_size {self.input_size} > rgb_native {self.rgb_native}")
        self.label_crop = self.input_size // self.rgb_to_lbl_ratio  # train label window
        self.training = bool(training)
        self.augment = bool(augment)
        self.harden_labels = bool(harden_labels)
        self.harden_thresh = float(harden_thresh)
        self.class_filter_idx = int(class_filter_idx) if class_filter_idx is not None else None
        self.class_filter_thresh = float(class_filter_thresh)

        # Build sample list from alignment.json
        with open(alignment_json) as f:
            align = json.load(f)
        # rgb_to_label gives RGB filename -> label filename (per train alignment)
        # If test, rgb_to_full_id gives RGB filename -> full_id (we derive label name)
        if "rgb_to_label" in align:
            # train alignment
            rgb_to_label = align["rgb_to_label"]
            samples_all = [(self.rgb_dir / rgb, self.labels_dir / lbl)
                           for rgb, lbl in sorted(rgb_to_label.items())]
        elif "rgb_to_full_id" in align:
            # test alignment
            rgb_to_full = align["rgb_to_full_id"]
            samples_all = [(self.rgb_dir / rgb, self.labels_dir / f"label_{full}.tif")
                           for rgb, full in sorted(rgb_to_full.items())]
        else:
            raise ValueError(f"unknown alignment schema in {alignment_json}")

        # Region split — replicates GeoFM/data/geofm.py split_pairs(split_by="region")
        # EXACTLY so the RGB specialist sees the SAME val tile set as the canonical
        # GeoFM/HRNet loaders (the 406-pair / ~400-RGB-tile val set), making this a
        # fair retest. We can't reuse split_pairs() directly: its parse_core_id()
        # strips known embedding prefixes (gee_emb_/label_/...) before extracting the
        # region, but RGB filenames are "train_<idx>_<REGION>_<year>.tif" — the
        # "train_" prefix is NOT stripped, so parse_core_id would mis-read the region
        # as the numeric index. We therefore extract the region with the RGB-correct
        # regex but apply GeoFM's IDENTICAL accumulate-until-val_fraction logic:
        #   sorted(regions) -> Random(split_seed).shuffle -> add regions to val until
        #   cumulative tile count >= round(n_total * val_fraction).
        # Because the region SET (e.g. {BE, DH, ...}) and sort order are the same as
        # GeoFM's, the shuffled region ORDER is identical, so the val region set
        # matches GeoFM's 39 regions (val tiles = the 400 of those 406 that have RGB).
        def _region(path):
            m = re.match(r"\w+_\d+_([A-Za-z]+)_\d+\.tif", path.name)
            return m.group(1) if m else "unknown"

        if split in ("train", "val", "test"):
            # Group samples by region (mirrors split_pairs' `groups` dict).
            groups = {}
            for rgb, lbl in samples_all:
                groups.setdefault(_region(rgb), []).append((rgb, lbl))
            group_keys = sorted(groups)
            random.Random(int(split_seed)).shuffle(group_keys)
            # target_val == GeoFM's: round(val_fraction of TOTAL tiles), min 1.
            target_val = max(1, int(round(len(samples_all) * float(val_fraction))))
            val_regions = set()
            count = 0
            for key in group_keys:
                if count >= target_val:
                    break
                val_regions.add(key)
                count += len(groups[key])
            train_regions = set(group_keys) - val_regions
            keep = train_regions if split == "train" else val_regions
        else:
            raise ValueError(f"unknown split={split}")
        samples = [(rgb, lbl) for (rgb, lbl) in samples_all if _region(rgb) in keep]

        # Class filter: drop train tiles with no buildings (or below threshold)
        if self.training and self.class_filter_idx is not None and self.class_filter_thresh > 0:
            kept = []
            for rgb, lbl in samples:
                if not lbl.exists():
                    continue
                lbl_arr = _read_label(lbl)   # (4, 256, 256)
                pos_frac = (lbl_arr[self.class_filter_idx] > 0.1).mean()
                if pos_frac >= self.class_filter_thresh:
                    kept.append((rgb, lbl))
            samples = kept

        self.samples = samples
        print(f"[BuildingSpecialistRGBDataset] split={split} n={len(self.samples)} "
              f"input_size={self.input_size} train_regions={len(train_regions)} val_regions={len(val_regions)}",
              flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_path, lbl_path = self.samples[index]

        # Load RGB at native 2560 (never upsampled)
        rgb = _read_rgb(rgb_path, expected_size=self.rgb_native)   # (3, 2560, 2560) uint8

        # Load label at NATIVE resolution (4, 256, 256); never upsampled.
        lbl_size = self.rgb_native // self.rgb_to_lbl_ratio        # 256
        if lbl_path.exists():
            target = _read_label(lbl_path)                         # (4, H, W) fp32
            # ~12% of label tifs are 255×256 / 256×255 / 255×255 (not exactly 256).
            # Snap to exact lbl_size so the 10:1 crop invariant holds (else edge
            # crops produce 63-wide targets → collate/stack shape mismatch).
            if target.shape[1] != lbl_size or target.shape[2] != lbl_size:
                target = _resize_label_to(target, lbl_size)        # seg nearest / height bilinear
        else:
            # Test-only: no label. Zero placeholder at native label size.
            target = np.zeros((4, lbl_size, lbl_size), dtype=np.float32)

        if self.training:
            # Random 640 RGB crop; offset a multiple of 10 so the label aligns.
            cs = self.input_size                                   # 640
            r = self.rgb_to_lbl_ratio                              # 10
            lc = self.label_crop                                   # 64
            max_off = (self.rgb_native - cs) // r                  # 192
            oy = r * random.randint(0, max_off)                    # in [0, 1920], mult of 10
            ox = r * random.randint(0, max_off)
            rgb = rgb[:, oy:oy + cs, ox:ox + cs]                   # (3, 640, 640)
            ly, lx = oy // r, ox // r                              # exact label offset
            target = target[:, ly:ly + lc, lx:lx + lc]             # (4, 64, 64)

            # HARD labels: binarize seg channels @harden_thresh — ONLY when
            # harden_labels=True AND on the train split. With harden_labels=False
            # (the fair-retest config) the target stays the raw SOFT GT fraction in
            # [0, 1], matching the canonical HRNet/GeoFM soft-label supervision.
            if self.harden_labels and self.training:
                for c in range(3):
                    target[c] = (target[c] > self.harden_thresh).astype(np.float32)

            # D4 augmentation (train only) — same ops applied to both arrays.
            if self.augment:
                rgb, target = _augment_d4(rgb, target)
        # VAL/TEST: full tile, no crop, soft labels (no harden).

        # Make contiguous + normalize
        rgb = np.ascontiguousarray(rgb)
        target = np.ascontiguousarray(target)
        x = _normalize_rgb(rgb)   # fp32

        # .clone() detaches from numpy-owned (non-resizable) storage — required for
        # num_workers>0, else default_collate's shared-memory path raises
        # "Trying to resize storage that is not resizable".
        image = torch.from_numpy(x).float().clone()
        meta = dict(
            id=lbl_path.stem.replace("label_", ""),
            embedding_paths=[str(rgb_path)],
            target_path=str(lbl_path),
            target=torch.from_numpy(target).float().clone(),
            loss_mask=torch.ones(1, target.shape[1], target.shape[2], dtype=torch.float32),
            is_latent=False,
        )
        return image, meta


@er.registry.DATALOADER.register()
class BuildingSpecialistRGBLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)

        dataset = BuildingSpecialistRGBDataset(
            rgb_dir=self.config.rgb_dir,
            labels_dir=self.config.labels_dir,
            alignment_json=self.config.alignment_json,
            split=self.config.split,
            val_fraction=getattr(self.config, "val_fraction", 0.2),
            split_by=getattr(self.config, "split_by", "region"),
            split_seed=getattr(self.config, "split_seed", 42),
            input_size=getattr(self.config, "input_size", 640),
            rgb_native_size=getattr(self.config, "rgb_native_size", 2560),
            training=self.config.training,
            augment=self.config.augment,
            harden_labels=getattr(self.config, "harden_labels", True),
            harden_thresh=getattr(self.config, "harden_thresh", 0.1),
            class_filter_idx=getattr(self.config, "class_filter_idx", 0),
            class_filter_thresh=getattr(self.config, "class_filter_thresh", 0.005),
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
        self.config.update(dict(
            rgb_dir="",
            labels_dir="",
            alignment_json="",
            split="train",
            val_fraction=0.2,
            split_by="region",
            split_seed=42,
            input_size=640,
            rgb_native_size=2560,
            batch_size=2,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            training=True,
            augment=True,
            harden_labels=True,
            harden_thresh=0.1,
            class_filter_idx=0,
            class_filter_thresh=0.005,
        ))
