from types import SimpleNamespace

import torch
import torch.nn as nn

from module.losses import (BaselineSpecialistLoss, BuildingHeightOnlyLoss, BuildingOnlyLoss,
                            BuildingVegHeightOnlyLoss,
                            MultiTaskHRNetLoss,
                            SoftmaxCE4ClassLoss,
                            GeoFMCompositeLoss, VegSpecialistLoss,
                            VegWaterBaselineLoss)
from module.metrics import RunningGeoFMMetrics
from module.networks import GeoFMNet

try:
    import ever as er
    from ever.interface import ERModule
except ModuleNotFoundError:
    class _Registry:
        def register(self, *args, **kwargs):
            def _decorator(cls):
                return cls
            return _decorator

    class _ER:
        registry = SimpleNamespace(MODEL=_Registry())

    class ERModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = _to_namespace(config or {})

    er = _ER()


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


@er.registry.MODEL.register()
class GeoFMEmbed2Heights(ERModule):
    def __init__(self, config):
        super().__init__(config)
        model_type = _cfg(self.config, "model_type", "auto")
        # Single-class specialists (veg/building/water) output 2 channels
        # instead of 4. Target slicing picks (class_idx, height_idx=3).
        # Mapping: veg=ch1, building=ch0, water=ch2 in 4-ch label format.
        _spec_class_idx = {
            "adapter_fusion_veg_specialist": 1,
            "adapter_fusion_building_specialist": 0,
            "adapter_fusion_water_specialist": 2,
        }.get(model_type, None)
        self.is_veg_specialist = (_spec_class_idx is not None)   # keep name for backward compat
        self.specialist_class_idx = _spec_class_idx              # 0 / 1 / 2 / None
        # CE 4-class seg specialist (Plan A): 5-ch output (4 seg + 1 height).
        # Detected here by model_type so the body's out_channels is correct.
        self.is_ce4 = (model_type == "adapter_fusion_late_multi_decoupled_ce")
        if self.is_ce4:
            out_channels = 5
        elif self.is_veg_specialist:
            out_channels = 2
        else:
            out_channels = 4

        self.net = GeoFMNet(
            in_channels=_cfg(self.config, "in_channels", 768),
            out_channels=out_channels,
            model_type=model_type,
            height_activation=_cfg(self.config, "height_activation", "softplus"),
            source_channels=_cfg(self.config, "source_channels", None),
            use_groupnorm=_cfg(self.config, "use_groupnorm", False),
            # Late-fusion params (only used by `adapter_fusion_late_multi`)
            dense_channels=_cfg(self.config, "dense_channels", None),
            token_channels=_cfg(self.config, "token_channels", None),
            adapter_out=_cfg(self.config, "adapter_out", 64),
            fused_bottleneck=_cfg(self.config, "fused_bottleneck", 384),
            # ConvNeXt-UNetFormer params (only used by adapter_fusion_convnext_unetformer)
            encoder_dims=_cfg(self.config, "encoder_dims", (96, 192, 384, 768)),
            encoder_depths=_cfg(self.config, "encoder_depths", (3, 3, 9, 3)),
            drop_path_rate=_cfg(self.config, "drop_path_rate", 0.1),
            # EfficientNet-UNetFormer params (only used by adapter_fusion_efficientnet_unetformer)
            efficientnet_pretrained=_cfg(self.config, "efficientnet_pretrained", True),
            # MiT-B0 pretrained weights (only used by adapter_fusion_mit_b0_token_fusion)
            pretrained_mit_b0=_cfg(self.config, "pretrained_mit_b0", None),
            # RGB-fused body params (only used by adapter_fusion_lite_hrnet_token_fusion_rgb)
            rgb_token_channels=_cfg(self.config, "rgb_token_channels", None),
            rgb_modality_dropout=_cfg(self.config, "rgb_modality_dropout", None),
            # DPT building specialist params (only used by dpt_dinov3l_building_specialist)
            dinov3_weights_path=_cfg(self.config, "dinov3_weights_path", None),
            dpt_input_size=_cfg(self.config, "dpt_input_size", None),
            dpt_out_indices=_cfg(self.config, "dpt_out_indices", None),
            dpt_features=_cfg(self.config, "dpt_features", None),
            dpt_in_shape=_cfg(self.config, "dpt_in_shape", None),
            dpt_label_size=_cfg(self.config, "dpt_label_size", None),
        )

        loss_cfg = _cfg(self.config, "loss", SimpleNamespace())
        # Loss routing:
        #   loss.type="building_only"        → BuildingOnlyLoss (class_idx=0, building)
        #   loss.type="veg_only"             → BuildingOnlyLoss (class_idx=1, vegetation)
        #   loss.type="water_only"           → BuildingOnlyLoss (class_idx=2, water)
        #   loss.type="building_height_only" → BuildingHeightOnlyLoss (mask-conditioned h_b)
        #   is_veg_specialist=True           → VegSpecialistLoss (2-ch preds + targets, legacy)
        #   otherwise                        → GeoFMCompositeLoss (full multi-task)
        loss_type = _cfg(loss_cfg, "type", None)
        # Map loss_type to class_idx for class-specialist runs
        _specialist_map = {"building_only": 0, "veg_only": 1, "water_only": 2}
        self.is_building_only = (loss_type in _specialist_map)
        self.specialist_train_class_idx = _specialist_map.get(loss_type, None)
        # NEW: baseline-style single-class specialist (MAE_split + SSIM + Grad + Tversky)
        _baseline_map = {"building_baseline": 0, "veg_baseline": 1, "water_baseline": 2}
        self.is_baseline_specialist = (loss_type in _baseline_map)
        self.baseline_class_idx = _baseline_map.get(loss_type, None)
        # NEW: JOINT veg+water baseline (supervises ch1 AND ch2 on 4-ch output)
        self.is_vegwater_baseline = (loss_type == "vegwater_baseline")
        # Height-specialist loss map: loss_type → mask_class_idx
        _height_specialist_map = {"building_height_only": 0, "veg_height_only": 1, "water_height_only": 2}
        self.is_height_only = (loss_type in _height_specialist_map)
        self.height_class_idx = _height_specialist_map.get(loss_type, None)
        # Keep old flag for backward compat with existing code paths that check it
        self.is_building_height_only = self.is_height_only
        # NEW: joint bld+veg height regression (ignore_bg + prior-residual arch)
        self.is_bv_height_only = (loss_type == "bv_height_only")
        # NEW: 4-class softmax CE seg specialist (Plan A)
        self.is_ce_loss = (loss_type == "seg_ce_4class")
        # NEW: multi-task HRNet (per-class specialist recipes)
        self.is_mt_hrnet = (loss_type == "mt_hrnet")
        # Sanity: is_ce4 (arch flag) should match is_ce_loss (loss flag).
        if self.is_ce4 ^ self.is_ce_loss:
            raise ValueError(
                f"model_type='{model_type}' (is_ce4={self.is_ce4}) and "
                f"loss.type='{loss_type}' (is_ce_loss={self.is_ce_loss}) must agree: "
                "both must select the CE 4-class path, or neither."
            )

        if self.is_ce_loss:
            assert not self.is_veg_specialist, "seg_ce_4class expects 5-ch model output, not 2-ch specialist arch"
            self.criterion = SoftmaxCE4ClassLoss(
                ce_weight=_cfg(loss_cfg, "ce_weight", 1.0),
                dice_weight=_cfg(loss_cfg, "dice_weight", 0.5),
                height_weight=_cfg(loss_cfg, "height_weight", 1.5),
                class_weights=_cfg(loss_cfg, "class_weights", None),
                label_thresh=_cfg(loss_cfg, "label_thresh", 0.1),
                huber_beta=_cfg(loss_cfg, "huber_beta", 0.1),
                include_dice=tuple(_cfg(loss_cfg, "include_dice", (0, 2))),
            )
        elif self.is_bv_height_only:
            assert not self.is_veg_specialist, "bv_height_only expects 4-ch model output"
            self.criterion = BuildingVegHeightOnlyLoss(
                bld_weight=_cfg(loss_cfg, "bld_weight", 1.0),
                veg_weight=_cfg(loss_cfg, "veg_weight", 1.0),
                height_loss_type=_cfg(loss_cfg, "height_loss_type", "smooth_l1"),
                huber_beta=_cfg(loss_cfg, "huber_beta", 0.1),
                height_mask_thresh=_cfg(loss_cfg, "height_mask_thresh", 0.5),
                under_pen_lambda=_cfg(loss_cfg, "under_pen_lambda", 0.0),
                under_pen_power=_cfg(loss_cfg, "under_pen_power", 2),
                ignore_bg=_cfg(loss_cfg, "ignore_bg", True),
                bld_boost=_cfg(loss_cfg, "bld_boost", 0.0),
                veg_boost=_cfg(loss_cfg, "veg_boost", 0.0),
                per_pixel_weighted=_cfg(loss_cfg, "per_pixel_weighted", False),
            )
        elif self.is_height_only:
            assert not self.is_veg_specialist, "height_only loss expects 4-ch model output"
            self.criterion = BuildingHeightOnlyLoss(
                height_weight=_cfg(loss_cfg, "height_weight", 1.0),
                height_loss_type=_cfg(loss_cfg, "height_loss_type", "smooth_l1"),
                huber_beta=_cfg(loss_cfg, "huber_beta", 1.0),
                build_mask_thresh=_cfg(loss_cfg, "build_mask_thresh", 0.5),
                mask_class_idx=self.height_class_idx,
            )
        elif self.is_building_only:
            assert not self.is_veg_specialist, "class-specialist loss expects 4-ch model output, not 2-ch specialist arch"
            self.criterion = BuildingOnlyLoss(
                tversky_alpha=_cfg(loss_cfg, "tversky_alpha", 0.5),
                tversky_beta=_cfg(loss_cfg, "tversky_beta", 0.5),
                tversky_weight=_cfg(loss_cfg, "tversky_weight", 1.0),
                bce_weight=_cfg(loss_cfg, "bce_weight", 1.0),
                use_bce=_cfg(loss_cfg, "use_bce", True),
                class_idx=self.specialist_train_class_idx,
            )
        elif self.is_baseline_specialist:
            assert not self.is_veg_specialist, "baseline-specialist loss expects 4-ch model output"
            self.criterion = BaselineSpecialistLoss(
                class_idx=self.baseline_class_idx,
                mae_weight=_cfg(loss_cfg, "mae_weight", 1.0),
                mae_bg_weight=_cfg(loss_cfg, "mae_bg_weight", 0.05),
                ssim_weight=_cfg(loss_cfg, "ssim_weight", 0.5),
                grad_weight=_cfg(loss_cfg, "grad_weight", 0.5),
                tversky_weight=_cfg(loss_cfg, "tversky_weight", 2.0),
                tversky_alpha=_cfg(loss_cfg, "tversky_alpha", 0.3),
                tversky_beta=_cfg(loss_cfg, "tversky_beta", 0.7),
            )
        elif self.is_vegwater_baseline:
            assert not self.is_veg_specialist, "vegwater_baseline expects 4-ch model output"
            self.criterion = VegWaterBaselineLoss(
                mae_weight=_cfg(loss_cfg, "mae_weight", 1.0),
                mae_bg_weight=_cfg(loss_cfg, "mae_bg_weight", 0.05),
                ssim_weight=_cfg(loss_cfg, "ssim_weight", 0.5),
                grad_weight=_cfg(loss_cfg, "grad_weight", 0.5),
                tversky_weight=_cfg(loss_cfg, "tversky_weight", 2.0),
                tversky_alpha=_cfg(loss_cfg, "tversky_alpha", 0.5),
                tversky_beta=_cfg(loss_cfg, "tversky_beta", 0.5),
            )
        elif self.is_mt_hrnet:
            assert not self.is_veg_specialist, "mt_hrnet expects 4-ch model output"
            self.criterion = MultiTaskHRNetLoss(
                w_bld=_cfg(loss_cfg, "w_bld", 1.0),
                w_veg=_cfg(loss_cfg, "w_veg", 1.0),
                w_water=_cfg(loss_cfg, "w_water", 1.0),
                w_height=_cfg(loss_cfg, "w_height", 5.0),
                bld_tversky_alpha=_cfg(loss_cfg, "bld_tversky_alpha", 0.5),
                bld_tversky_beta=_cfg(loss_cfg, "bld_tversky_beta", 0.5),
                veg_tversky_alpha=_cfg(loss_cfg, "veg_tversky_alpha", 0.5),
                veg_tversky_beta=_cfg(loss_cfg, "veg_tversky_beta", 0.5),
                water_tversky_alpha=_cfg(loss_cfg, "water_tversky_alpha", 0.3),
                water_tversky_beta=_cfg(loss_cfg, "water_tversky_beta", 0.7),
                bld_bce_weight=_cfg(loss_cfg, "bld_bce_weight", 1.0),
                bld_tversky_weight=_cfg(loss_cfg, "bld_tversky_weight", 1.0),
                bld_use_baseline_style=_cfg(loss_cfg, "bld_use_baseline_style", False),
                mae_weight=_cfg(loss_cfg, "mae_weight", 1.0),
                mae_bg_weight=_cfg(loss_cfg, "mae_bg_weight", 0.05),
                ssim_weight=_cfg(loss_cfg, "ssim_weight", 0.5),
                grad_weight=_cfg(loss_cfg, "grad_weight", 0.5),
                tversky_weight=_cfg(loss_cfg, "tversky_weight", 2.0),
                height_loss_type=_cfg(loss_cfg, "height_loss_type", "smooth_l1"),
                huber_beta=_cfg(loss_cfg, "huber_beta", 0.1),
                height_mask_thresh=_cfg(loss_cfg, "height_mask_thresh", 0.1),
                ignore_bg=_cfg(loss_cfg, "ignore_bg", True),
            )
        elif self.is_veg_specialist:
            self.criterion = VegSpecialistLoss(
                seg_weight=_cfg(loss_cfg, "seg_weight", 1.0),
                height_weight=_cfg(loss_cfg, "height_weight", 10.0),
                tversky_alpha=_cfg(loss_cfg, "tversky_alpha", 0.3),
                tversky_beta=_cfg(loss_cfg, "tversky_beta", 0.7),
                height_mask_thresh=_cfg(loss_cfg, "height_mask_thresh", 0.1),
                ssim_weight=_cfg(loss_cfg, "ssim_weight", 0.0),
                grad_weight=_cfg(loss_cfg, "grad_weight", 0.0),
                loss_type=_cfg(loss_cfg, "loss_type", "l1"),
                huber_beta=_cfg(loss_cfg, "huber_beta", 0.1),
                height_grad_weight=_cfg(loss_cfg, "height_grad_weight", 0.0),
                seg_loss_type=_cfg(loss_cfg, "seg_loss_type", "tversky"),
                bce_dice_ratio=_cfg(loss_cfg, "bce_dice_ratio", 0.5),
                height_mask_mode=_cfg(loss_cfg, "height_mask_mode", "threshold"),
            )
        else:
            self.criterion = GeoFMCompositeLoss(
                lambdas=tuple(_cfg(loss_cfg, "lambdas", (1.0, 0.5, 0.5, 2.0))),
                bg_weight=_cfg(loss_cfg, "bg_weight", 0.05),
                building_height_boost=_cfg(loss_cfg, "building_height_boost", 5.0),
                vegetation_height_boost=_cfg(loss_cfg, "vegetation_height_boost", 0.0),
                height_ignore_bg=_cfg(loss_cfg, "height_ignore_bg", False),
                height_valid_thresh=_cfg(loss_cfg, "height_valid_thresh", 0.1),
                seg_loss_type=_cfg(loss_cfg, "seg_loss_type", "mae"),
                height_loss_type=_cfg(loss_cfg, "height_loss_type", "mae"),
                height_huber_beta=_cfg(loss_cfg, "height_huber_beta", 0.1),
                seg_aux_loss_type=_cfg(loss_cfg, "seg_aux_loss_type", "tversky"),
                tversky_alpha=_cfg(loss_cfg, "tversky_alpha", 0.3),
                tversky_beta=_cfg(loss_cfg, "tversky_beta", 0.7),
                ohem_thresh=_cfg(loss_cfg, "ohem_thresh", 0.7),
                ohem_min_kept_frac=_cfg(loss_cfg, "ohem_min_kept_frac", 0.1),
                per_class_height_weight=_cfg(loss_cfg, "per_class_height_weight", 0.0),
            )
        self.height_norm_constant = _cfg(self.config, "height_norm_constant", 30.0)
        self.metric_threshold = _cfg(self.config, "metric_threshold", 0.1)

    def custom_param_groups(self):
        # DPT building specialist: 3-tier discriminative LR (coarse LLRD).
        # Keys MUST match config optimizer/learning_rate keys (head/enc_top/enc_bot).
        body = getattr(self.net, "body", None)
        if getattr(body, "requires_raw_rgb", False):
            head, enc_top, enc_bot = [], [], []
            for name, p in self.net.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("body.encoder."):
                    if ".blocks." in name:
                        li = int(name.split(".blocks.")[1].split(".")[0])
                        (enc_top if li >= 12 else enc_bot).append(p)
                    elif name.startswith("body.encoder.norm"):
                        enc_top.append(p)           # final encoder norm
                    else:
                        enc_bot.append(p)           # patch_embed / cls / mask / rope params
                else:
                    head.append(p)                  # body.feat_processor.* + body.dpt.*
            return dict(head=head, enc_top=enc_top, enc_bot=enc_bot)
        return dict(decoder=self.net.parameters())

    def _veg_to_fake4(self, preds_native):
        """Expand (B, 2, H, W) single-class specialist preds to (B, 4, H, W) format.
        Other class channels stay 0. Class is picked via self.specialist_class_idx
        (0=building, 1=veg, 2=water)."""
        fake4 = torch.zeros(preds_native.shape[0], 4,
                            preds_native.shape[2], preds_native.shape[3],
                            device=preds_native.device, dtype=preds_native.dtype)
        cls_idx = self.specialist_class_idx if self.specialist_class_idx is not None else 1
        fake4[:, cls_idx] = preds_native[:, 0]   # class-specific seg
        fake4[:, 3] = preds_native[:, 1]         # height
        return fake4

    def _sliding_predict(self, x, window=640, overlap=256, ratio=10, win_chunk=8):
        """Sliding-window eval/inference for the raw-RGB DPT building specialist.

        x: (B, 3, S, S) full RGB tile (S = 2560). Each `window`×`window` RGB crop
        is fed through self.net (→ self.net.activate) producing a
        (B, 4, win_lbl, win_lbl) label-resolution tile (win_lbl = window // ratio).
        Windows are average-stitched onto the full label grid (label_size = S // ratio).

        Anchors per axis = sorted(set(range(0, S-window+1, stride)) ∪ {S-window}),
        with stride = window - overlap, pinning the last anchor at S-window so the
        far edge is always covered.

        Returns preds (B, 4, label_size, label_size) = sum / count.clamp(min=1).
        Windows are batched in chunks of `win_chunk` (over the spatial windows of a
        single tile) to bound memory.
        """
        b, c, S, _ = x.shape
        stride = window - overlap
        win_lbl = window // ratio
        label_size = S // ratio

        anchors = sorted(set(list(range(0, S - window + 1, stride)) + [S - window]))
        coords = [(oy, ox) for oy in anchors for ox in anchors]

        out_ch = getattr(self.net.body, "out_channels_total", 4)
        device = x.device
        acc = torch.zeros(b, out_ch, label_size, label_size, device=device, dtype=torch.float32)
        cnt = torch.zeros(label_size, label_size, device=device, dtype=torch.float32)

        # Iterate windows in chunks to bound memory. Windows of a single tile are
        # stacked along the batch dim, pushed through self.net, then scattered back.
        for i in range(0, len(coords), win_chunk):
            chunk = coords[i:i + win_chunk]
            wins = []
            for (oy, ox) in chunk:
                wins.append(x[:, :, oy:oy + window, ox:ox + window])
            batch = torch.cat(wins, dim=0)                       # (b*k, 3, 640, 640)
            raw = self.net(batch)
            p = self.net.activate(raw)                           # (b*k, 4, 64, 64)
            for j, (oy, ox) in enumerate(chunk):
                pj = p[j * b:(j + 1) * b]                         # (b, 4, 64, 64)
                ly = round(oy / ratio)
                lx = round(ox / ratio)
                acc[:, :, ly:ly + win_lbl, lx:lx + win_lbl] += pj
                cnt[ly:ly + win_lbl, lx:lx + win_lbl] += 1.0

        preds = acc / cnt.clamp(min=1.0).unsqueeze(0).unsqueeze(0)
        return preds

    def forward(self, x, y=None):
        # Late-fusion bodies need tokens too. Extract from y (meta dict) and pass.
        tokens = None
        building_mask = None
        rgb_token = None
        has_rgb = None
        # Raw-RGB specialist bodies (DPT-on-DINOv3) take only x = (B, 3, H, W) RGB.
        # Skip all GeoFM-token / mask / rgb-token routing.
        if getattr(self.net.body, "requires_raw_rgb", False):
            if self.training:
                # TRAIN: x is a (B, 3, 640, 640) RGB crop. Single forward →
                # (B, 4, 64, 64) native-label-resolution preds. Loss/metrics at 64×64.
                raw = self.net(x.float())
                preds = self.net.activate(raw)
                targets = y["target"].to(preds.device).float()
                loss_dict = self.criterion(preds, targets)
                with torch.no_grad():
                    metrics = RunningGeoFMMetrics(
                        threshold=self.metric_threshold,
                        height_norm_constant=self.height_norm_constant,
                    )
                    metrics.update(preds, targets)
                    summary = metrics.summary()
                    for key, value in summary.items():
                        loss_dict[key] = torch.as_tensor(value, device=preds.device)
                return loss_dict
            # EVAL / INFERENCE: x is a FULL (B, 3, 2560, 2560) RGB tile. Run a
            # SLIDING WINDOW of 640 windows (overlap 256 → stride 384), each
            # mapping to a 64×64 label tile, and average-stitch into a full
            # (B, 4, 256, 256) prediction at native label resolution.
            return self._sliding_predict(x.float())
        if getattr(self.net.body, "is_late_fusion", False):
            if y is None or "tokens" not in y:
                raise ValueError("Late-fusion model requires meta['tokens'] in y")
            tokens = y["tokens"].to(x.device).float()
        # RGB-fused body needs rgb_token + has_rgb.
        if getattr(self.net.body, "requires_rgb_token", False):
            if y is None or "rgb_token" not in y or "has_rgb" not in y:
                raise ValueError(
                    "RGB-fused body requires meta['rgb_token'] (B, 1024, h, w) and "
                    "meta['has_rgb'] (B,) bool. Set data.{train,test}.params."
                    "rgb_token_dir/rgb_token_stats_path/rgb_align_json."
                )
            rgb_token = y["rgb_token"].to(x.device).float()
            has_rgb = y["has_rgb"].to(x.device)
            if has_rgb.dtype != torch.bool:
                has_rgb = has_rgb.to(dtype=torch.bool)
        # Mask-conditioned height specialist needs building_mask in meta dict.
        if getattr(self.net.body, "requires_building_mask", False):
            if y is None or "building_mask" not in y:
                raise ValueError(
                    "Mask-conditioned body requires meta['building_mask'] (B, 1, H, W). "
                    "Set data.train.params.building_mask_dir or ensure GT-derived mask is provided."
                )
            building_mask = y["building_mask"].to(x.device).float()

        if tokens is not None:
            raw = self.net(x.float(), tokens=tokens, building_mask=building_mask,
                           rgb_token=rgb_token, has_rgb=has_rgb)
        else:
            raw = self.net(x.float())
        preds = self.net.activate(raw)
        if self.training:
            targets = y["target"].to(preds.device).float()
            # Optional per-pixel loss mask (for pseudo-labels with confidence filtering)
            loss_mask = y.get("loss_mask")
            if loss_mask is not None:
                loss_mask = loss_mask.to(preds.device).float()

            if self.is_veg_specialist:
                # Slice full 4-ch target to (B, 2, H, W) = [class, height]
                # class channel determined by specialist_class_idx (0=bld, 1=veg, 2=water)
                cls_idx = self.specialist_class_idx
                veg_target = targets[:, [cls_idx, 3]]
                loss_dict = self.criterion(preds, veg_target)
                # Use fake-4 for metrics so existing RunningGeoFMMetrics works
                with torch.no_grad():
                    fake4 = self._veg_to_fake4(preds)
                    metrics = RunningGeoFMMetrics(
                        threshold=self.metric_threshold,
                        height_norm_constant=self.height_norm_constant,
                    )
                    metrics.update(fake4, targets)
                    summary = metrics.summary()
                    for key, value in summary.items():
                        loss_dict[key] = torch.as_tensor(value, device=preds.device)
                return loss_dict

            if self.is_ce_loss:
                # Loss needs raw logits at ch0-3 (CE) + softplus-activated height at ch4.
                # raw[:, :4] = raw seg logits; preds[:, 4:5] = softplus-activated height.
                loss_input = torch.cat([raw[:, :4], preds[:, 4:5]], dim=1)
                loss_dict = self.criterion(loss_input, targets)
                # Metrics: argmax over softmax probs → per-class IoU; rmse_b/rmse_v at GT.
                with torch.no_grad():
                    seg_probs = preds[:, :4]                                # softmax probs
                    pred_cls = seg_probs.argmax(dim=1)                       # (B, H, W)
                    th = self.metric_threshold
                    iou_per_class = []
                    for c in range(3):  # bld=0, veg=1, water=2 (skip bg=3)
                        p_hard = (pred_cls == c).float()
                        t_hard = (targets[:, c] > th).float()
                        tp = (p_hard * t_hard).sum()
                        fp = (p_hard * (1 - t_hard)).sum()
                        fn = ((1 - p_hard) * t_hard).sum()
                        iou_per_class.append(tp / (tp + fp + fn + 1e-6))
                    # rmse in meters at GT fg pixels
                    h_pred_m = preds[:, 4] * self.height_norm_constant
                    h_tgt_m = targets[:, 3] * self.height_norm_constant
                    sq_err = (h_pred_m - h_tgt_m) ** 2
                    bld_m = (targets[:, 0] > 0.5).float()
                    veg_m = (targets[:, 1] > 0.5).float()
                    rmse_b = torch.sqrt((sq_err * bld_m).sum() / bld_m.sum().clamp(min=1.0))
                    rmse_v = torch.sqrt((sq_err * veg_m).sum() / veg_m.sum().clamp(min=1.0))
                    # Official-like proxy: mean IoU * 0.5 + rmse score * 0.5 (rough; LB formula)
                    iou_b, iou_v, iou_w = iou_per_class
                    rmse_b_score = torch.clamp(1.0 - rmse_b / 3.0, min=0.0)
                    rmse_v_score = torch.clamp(1.0 - rmse_v / 5.0, min=0.0)
                    official = (0.25 * iou_b + 0.15 * iou_v + 0.15 * iou_w
                                + 0.25 * rmse_b_score + 0.20 * rmse_v_score).detach()
                    loss_dict["miou_buildings"] = iou_b.detach()
                    loss_dict["miou_vegetation"] = iou_v.detach()
                    loss_dict["miou_water"] = iou_w.detach()
                    loss_dict["rmse_buildings"] = rmse_b.detach()
                    loss_dict["rmse_vegetation"] = rmse_v.detach()
                    loss_dict["proxy_score"] = official
                    loss_dict["official_score"] = official
                return loss_dict

            # V2 models expose per-class height heads (_latest_h_b / _latest_h_v)
            # for direct supervision in the loss.
            h_b = getattr(self.net.body, "_latest_h_b", None)
            h_v = getattr(self.net.body, "_latest_h_v", None)
            if h_b is None and hasattr(self.net.body, "body"):
                # AdapterFusion wrapper: per-class heads live on the inner body
                h_b = getattr(self.net.body.body, "_latest_h_b", None)
                h_v = getattr(self.net.body.body, "_latest_h_v", None)
            if self.is_bv_height_only:
                loss_dict = self.criterion(preds, targets)
            elif self.is_building_height_only:
                loss_dict = self.criterion(preds, targets)
            else:
                loss_dict = self.criterion(preds, targets, h_b=h_b, h_v=h_v, loss_mask=loss_mask)
            with torch.no_grad():
                if self.is_bv_height_only:
                    # Joint bld+veg height: compute BOTH rmse_b and rmse_v at GT pixels in meters.
                    bld_mask = (targets[:, 0] > 0.5).float()
                    veg_mask = (targets[:, 1] > 0.5).float()
                    h_pred_m = preds[:, 3] * self.height_norm_constant
                    h_tgt_m = targets[:, 3] * self.height_norm_constant
                    sq_err = (h_pred_m - h_tgt_m) ** 2
                    rmse_b = torch.sqrt((sq_err * bld_mask).sum() / bld_mask.sum().clamp(min=1.0))
                    rmse_v = torch.sqrt((sq_err * veg_mask).sum() / veg_mask.sum().clamp(min=1.0))
                    # Combined proxy following LB formula partial: 0.25*max(0,1-rmse_b/3) + 0.20*max(0,1-rmse_v/5)
                    proxy = (0.25 * torch.clamp(1.0 - rmse_b / 3.0, min=0.0)
                             + 0.20 * torch.clamp(1.0 - rmse_v / 5.0, min=0.0)).detach()
                    loss_dict["miou_buildings"] = torch.zeros_like(rmse_b)
                    loss_dict["miou_vegetation"] = torch.zeros_like(rmse_b)
                    loss_dict["miou_water"] = torch.zeros_like(rmse_b)
                    loss_dict["rmse_buildings"] = rmse_b.detach()
                    loss_dict["rmse_vegetation"] = rmse_v.detach()
                    loss_dict["proxy_score"] = proxy
                    loss_dict["official_score"] = proxy
                elif self.is_height_only:
                    # Height specialist: compute rmse at GT pixels of the masked class in meters.
                    # class_idx: 0=bld → rmse_buildings, 1=veg → rmse_vegetation, 2=water → both 0.
                    c = self.height_class_idx
                    mask = (targets[:, c] > 0.5).float()
                    h_pred_m = preds[:, 3] * self.height_norm_constant
                    h_tgt_m = targets[:, 3] * self.height_norm_constant
                    sq_err = ((h_pred_m - h_tgt_m) ** 2) * mask
                    n_valid = mask.sum().clamp(min=1.0)
                    rmse_c = torch.sqrt(sq_err.sum() / n_valid)
                    # Use 1/(1+rmse) as proxy for "higher is better" (EVER picks best by max)
                    proxy = (1.0 / (1.0 + rmse_c)).detach()
                    loss_dict["miou_buildings"] = torch.zeros_like(rmse_c)
                    loss_dict["miou_vegetation"] = torch.zeros_like(rmse_c)
                    loss_dict["miou_water"] = torch.zeros_like(rmse_c)
                    # Route rmse into the correct class slot
                    loss_dict["rmse_buildings"] = rmse_c.detach() if c == 0 else torch.zeros_like(rmse_c)
                    loss_dict["rmse_vegetation"] = rmse_c.detach() if c == 1 else torch.zeros_like(rmse_c)
                    loss_dict["proxy_score"] = proxy
                    loss_dict["official_score"] = proxy
                elif self.is_building_only or self.is_baseline_specialist:
                    # Single-class specialist training: other channels are untrained,
                    # their metrics would be garbage. Compute ONLY iou for the trained
                    # class and set official_score=iou_c so EVER's "best ckpt" selection
                    # tracks the real target. class_idx: 0=bld, 1=veg, 2=water.
                    c = self.specialist_train_class_idx if self.is_building_only else self.baseline_class_idx
                    cls_pred_hard = (preds[:, c] > self.metric_threshold).float()
                    cls_tgt_hard = (targets[:, c] > self.metric_threshold).float()
                    tp = (cls_pred_hard * cls_tgt_hard).sum()
                    fp = (cls_pred_hard * (1 - cls_tgt_hard)).sum()
                    fn = ((1 - cls_pred_hard) * cls_tgt_hard).sum()
                    iou_c = tp / (tp + fp + fn + 1e-6)
                    # Inject metric keys; route iou into the correct class slot.
                    miou_key = ("miou_buildings", "miou_vegetation", "miou_water")[c]
                    loss_dict["miou_buildings"] = (iou_c if c == 0 else torch.zeros_like(iou_c)).detach()
                    loss_dict["miou_vegetation"] = (iou_c if c == 1 else torch.zeros_like(iou_c)).detach()
                    loss_dict["miou_water"] = (iou_c if c == 2 else torch.zeros_like(iou_c)).detach()
                    loss_dict["rmse_buildings"] = torch.zeros_like(iou_c)
                    loss_dict["rmse_vegetation"] = torch.zeros_like(iou_c)
                    loss_dict["proxy_score"] = iou_c.detach()
                    loss_dict["official_score"] = iou_c.detach()
                elif self.is_vegwater_baseline:
                    # Joint veg+water: compute IoU at BOTH ch1 (veg) and ch2 (water).
                    # ch0 (bld) / ch3 (height) untrained → 0. official = mean(iou_v, iou_w)
                    # (higher=better; inference uses model-LAST regardless, per 0609 §2.1).
                    th = self.metric_threshold
                    def _iou(c):
                        ph = (preds[:, c] > th).float()
                        thd = (targets[:, c] > th).float()
                        tp = (ph * thd).sum(); fp = (ph * (1 - thd)).sum(); fn = ((1 - ph) * thd).sum()
                        return tp / (tp + fp + fn + 1e-6)
                    iou_v = _iou(1); iou_w = _iou(2)
                    proxy = ((iou_v + iou_w) / 2.0).detach()
                    loss_dict["miou_buildings"] = torch.zeros_like(iou_v)
                    loss_dict["miou_vegetation"] = iou_v.detach()
                    loss_dict["miou_water"] = iou_w.detach()
                    loss_dict["rmse_buildings"] = torch.zeros_like(iou_v)
                    loss_dict["rmse_vegetation"] = torch.zeros_like(iou_v)
                    loss_dict["proxy_score"] = proxy
                    loss_dict["official_score"] = proxy
                else:
                    metrics = RunningGeoFMMetrics(
                        threshold=self.metric_threshold,
                        height_norm_constant=self.height_norm_constant,
                    )
                    metrics.update(preds, targets)
                    summary = metrics.summary()
                    for key, value in summary.items():
                        loss_dict[key] = torch.as_tensor(value, device=preds.device)
            return loss_dict

        # Eval / inference path: veg specialist returns 4-ch via fake4 for
        # downstream compatibility (eval metrics, predict_multi_tta, ensembles).
        if self.is_veg_specialist:
            return self._veg_to_fake4(preds)
        # CE 4-class: preds is (B, 5, H, W) = [bld, veg, water, bg, height] as
        # softmax probs + softplus height. Drop bg channel for downstream (B, 4, H, W)
        # compatibility (predict_multi_tta, predict_soft_cache, multi_source_ensemble
        # all expect 4 channels). Bg is implicit (1 - bld - veg - water).
        if self.is_ce4 and preds.shape[1] == 5:
            return torch.cat([preds[:, :3], preds[:, 4:5]], dim=1)
        return preds

    def backward(self, loss_dict, optimizer, amp, **kwargs):
        total_loss = sum(value for value in loss_dict.values() if torch.is_tensor(value) and value.requires_grad)
        if amp:
            scaler = list(kwargs["scaler"].values())[0] if isinstance(kwargs.get("scaler"), dict) else kwargs["scaler"]
            scaler.scale(total_loss).backward()
            for optim in optimizer.values() if isinstance(optimizer, dict) else [optimizer]:
                scaler.unscale_(optim)
                self.clip_grad(optim)
                scaler.step(optim)
                optim.zero_grad()
        else:
            total_loss.backward()
            for optim in optimizer.values() if isinstance(optimizer, dict) else [optimizer]:
                self.clip_grad(optim)
                optim.step()
                optim.zero_grad()

    def set_default_config(self):
        self.config.update(
            dict(
                model_type="auto",
                in_channels=768,
                height_activation="softplus",
                height_norm_constant=30.0,
                metric_threshold=0.1,
                loss=dict(
                    lambdas=(1.0, 0.5, 0.5, 2.0),
                    bg_weight=0.05,
                    building_height_boost=5.0,
                ),
            )
        )
