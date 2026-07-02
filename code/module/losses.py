from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        batch_size = preds.size(0)
        p = preds.reshape(batch_size, -1)
        t = targets.reshape(batch_size, -1)
        tp = torch.sum(p * t, dim=1)
        fp = torch.sum(p * (1 - t), dim=1)
        fn = torch.sum((1 - p) * t, dim=1)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.mean(1.0 - score)


class GradientDifferenceLoss(nn.Module):
    def forward(self, pred, target):
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        return torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.register_buffer("window", self.create_window(window_size, self.channel), persistent=False)

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        one_d = self.gaussian(window_size, 1.5).unsqueeze(1)
        two_d = one_d.mm(one_d.t()).float().unsqueeze(0).unsqueeze(0)
        return two_d.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel != self.channel or self.window.device != img1.device:
            self.window = self.create_window(self.window_size, channel).to(device=img1.device, dtype=img1.dtype)
            self.channel = channel

        window = self.window.type_as(img1)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1 - ssim.mean()


class OhemBCELoss(nn.Module):
    """Online Hard Example Mining BCE for per-channel binary segmentation.

    Inspired by mmsegmentation OhemCrossEntropy (per-pixel hard mining), adapted
    for multi-channel binary GT (post-hardening).

    For each channel independently:
      1. Compute per-pixel BCE.
      2. Compute correct-class probability per pixel (= pred if gt=1 else 1-pred).
      3. Drop "easy" pixels where correct prob > thresh.
      4. Keep at least min_kept_frac of pixels (the hardest by loss) regardless.
      5. Average BCE over kept pixels; mean across channels.

    Args:
      thresh: correct-class probability above which a pixel is "easy" and dropped.
      min_kept_frac: fraction of total pixels per channel to always keep, even
        if they exceed `thresh`.
    """
    def __init__(self, thresh=0.7, min_kept_frac=0.1):
        super().__init__()
        self.thresh = float(thresh)
        self.min_kept_frac = float(min_kept_frac)

    def forward(self, pred, target):
        # pred, target: (B, C, H, W). pred ∈ (0,1) (after sigmoid); target ∈ {0,1}.
        eps = 1e-7
        pred_c = pred.clamp(eps, 1.0 - eps)
        bce = -(target * torch.log(pred_c) + (1 - target) * torch.log(1 - pred_c))
        correct_p = torch.where(target > 0.5, pred_c, 1.0 - pred_c)

        B, C, H, W = pred.shape
        n_per_ch = B * H * W
        min_kept = max(1, int(n_per_ch * self.min_kept_frac))

        per_ch_losses = []
        for c in range(C):
            b_loss = bce[:, c].reshape(-1)
            b_prob = correct_p[:, c].reshape(-1)
            sorted_loss, _ = b_loss.sort(descending=True)
            min_loss_floor = sorted_loss[min_kept - 1]
            keep_mask = (b_prob <= self.thresh) | (b_loss >= min_loss_floor)
            kept = b_loss[keep_mask]
            per_ch_losses.append(kept.mean())
        return torch.stack(per_ch_losses).mean()


class GeoFMCompositeLoss(nn.Module):
    def __init__(self, lambdas=(1.0, 0.5, 0.5, 2.0), bg_weight=0.05,
                 building_height_boost=5.0, vegetation_height_boost=0.0,
                 per_class_height_weight=0.0,
                 height_ignore_bg=False, height_valid_thresh=0.1,
                 seg_loss_type="mae",
                 height_loss_type="mae", height_huber_beta=0.1,
                 seg_aux_loss_type="tversky",
                 tversky_alpha=0.3, tversky_beta=0.7,
                 ohem_thresh=0.7, ohem_min_kept_frac=0.1):
        """
        height_ignore_bg: if True, the height-related loss terms (MAE on ch3 + height_boost)
            ONLY sum over pixels where target_bld > thresh OR target_veg > thresh
            (the "valid" pixels per LB rmse formula). Background pixels' height
            predictions are ignored entirely (no gradient, no penalty).
            Rationale: LB rmse_b/rmse_v are evaluated only at fg pixels, so the
            model doesn't need to learn anything about bg height.
        height_valid_thresh: GT class probability threshold for what counts as a
            "valid" pixel when height_ignore_bg=True.
        """
        super().__init__()
        self.w_mae, self.w_ssim, self.w_grad, self.w_structure = lambdas
        self.bg_weight = bg_weight
        self.building_height_boost = building_height_boost
        self.vegetation_height_boost = vegetation_height_boost
        self.per_class_height_weight = per_class_height_weight  # weight for h_b/h_v direct supervision
        self.height_ignore_bg = bool(height_ignore_bg)
        self.height_valid_thresh = float(height_valid_thresh)
        self.seg_loss_type = str(seg_loss_type)
        assert self.seg_loss_type in ("mae", "bce"), \
            f"seg_loss_type must be 'mae' or 'bce', got {self.seg_loss_type}"
        self.height_loss_type = str(height_loss_type)
        assert self.height_loss_type in ("mae", "smooth_l1"), \
            f"height_loss_type must be 'mae' or 'smooth_l1', got {self.height_loss_type}"
        self.height_huber_beta = float(height_huber_beta)
        if self.seg_loss_type == "bce":
            # bce on hard binary GT requires height_ignore_bg=True so that the
            # mae term (which we now repurpose to hold the bce seg loss) doesn't
            # mix ch3 height MAE with the seg BCE.
            assert self.height_ignore_bg, \
                "seg_loss_type='bce' requires height_ignore_bg=True"
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()
        self.tversky = TverskyLoss(alpha=float(tversky_alpha), beta=float(tversky_beta))
        self.seg_aux_loss_type = str(seg_aux_loss_type)
        assert self.seg_aux_loss_type in ("tversky", "ohem_bce"), \
            f"seg_aux_loss_type must be 'tversky' or 'ohem_bce', got {self.seg_aux_loss_type}"
        self.ohem = OhemBCELoss(thresh=ohem_thresh, min_kept_frac=ohem_min_kept_frac)

    def forward(self, preds, targets, h_b=None, h_v=None, loss_mask=None):
        """
        loss_mask: optional (B, 1, H, W) or (B, H, W). 1.0=use, 0.0=ignore.
        Applied to per-pixel terms (MAE, height_boost, per-class h supervision).
        Region-aggregate terms (Tversky, SSIM, gradient) are left un-masked since
        their spatial-statistic interpretation breaks with masked pixels.
        """
        if loss_mask is not None:
            if loss_mask.dim() == 3:
                loss_mask = loss_mask.unsqueeze(1)

        abs_err = torch.abs(preds - targets)
        fg_mask = (targets > 0).float()
        bg_mask = 1.0 - fg_mask
        if loss_mask is not None:
            fg_mask = fg_mask * loss_mask
            bg_mask = bg_mask * loss_mask
            abs_err = abs_err * loss_mask
        # When height_ignore_bg=True, zero out the ch3 (height) contribution to
        # MAE so only `height_boost` supervises height (which we mask to fg pixels).
        if self.height_ignore_bg:
            abs_err = abs_err.clone()
            abs_err[:, 3] = 0
        fg_sum = torch.sum(fg_mask, dim=(0, 2, 3)) + 1e-6
        bg_sum = torch.sum(bg_mask, dim=(0, 2, 3)) + 1e-6
        mae_fg = torch.sum(abs_err * fg_mask, dim=(0, 2, 3)) / fg_sum
        mae_bg = torch.sum(abs_err * bg_mask, dim=(0, 2, 3)) / bg_sum
        mae = torch.sum(mae_fg + self.bg_weight * mae_bg)

        # When seg_loss_type="bce", replace the seg-channel contribution to `mae`
        # with BCE. height_ignore_bg=True is required (asserted in __init__), so
        # `mae` currently has 0 contribution from ch3 — only ch0-2 (seg) are in
        # there. Recompute that part as BCE for cleaner gradient on binary GT.
        if self.seg_loss_type == "bce":
            bce_seg = F.binary_cross_entropy(
                preds[:, :3].clamp(1e-7, 1.0 - 1e-7),
                targets[:, :3],
                reduction='mean',
            )
            mae = bce_seg  # repurpose slot for log compatibility

        lc_pred = preds[:, :3]
        lc_target = targets[:, :3]
        ssim = self.ssim(lc_pred, lc_target)
        grad = self.gdl(lc_pred, lc_target)

        if self.seg_aux_loss_type == "ohem_bce":
            # OHEM-BCE on 3-ch seg (per-channel hard pixel mining).
            # Logged under the same "tversky" key for compatibility.
            tversky = self.ohem(preds[:, :3], targets[:, :3])
        else:
            tversky = (
                self.tversky(preds[:, 0], targets[:, 0])
                + self.tversky(preds[:, 1], targets[:, 1])
                + self.tversky(preds[:, 2], targets[:, 2])
            ) / 3.0

        build_mask = (targets[:, 0] > 0.1).float()
        veg_mask = (targets[:, 1] > 0.1).float()
        if self.height_loss_type == "smooth_l1":
            height_err = F.smooth_l1_loss(
                preds[:, 3], targets[:, 3],
                beta=self.height_huber_beta, reduction='none',
            )
        else:
            height_err = torch.abs(preds[:, 3] - targets[:, 3])

        # Optional: ignore background pixels for height loss (LB only evaluates rmse
        # at GT bld/veg pixels, so bg height is irrelevant). Constructs a "valid"
        # mask from GT segmentation and only sums over those pixels.
        if self.height_ignore_bg:
            height_valid = ((targets[:, 0] > self.height_valid_thresh) |
                            (targets[:, 1] > self.height_valid_thresh)).float()
        else:
            height_valid = None

        if loss_mask is not None:
            m2 = loss_mask.squeeze(1)   # (B, H, W)
            height_err = height_err * m2
            build_mask = build_mask * m2
            veg_mask = veg_mask * m2
            if height_valid is not None:
                height_valid = height_valid * m2
            n_valid = m2.sum().clamp_min(1.0)
            height_boost = torch.sum(
                height_err
                * (1.0
                   + self.building_height_boost * build_mask
                   + self.vegetation_height_boost * veg_mask)
                * (height_valid if height_valid is not None else 1.0)
            ) / (height_valid.sum().clamp_min(1.0) if height_valid is not None else n_valid)
        else:
            if height_valid is not None:
                # Average only over valid (fg) pixels
                height_err_w = height_err * (
                    1.0
                    + self.building_height_boost * build_mask
                    + self.vegetation_height_boost * veg_mask
                ) * height_valid
                height_boost = height_err_w.sum() / height_valid.sum().clamp_min(1.0)
            else:
                height_boost = torch.mean(
                    height_err
                    * (1.0
                       + self.building_height_boost * build_mask
                       + self.vegetation_height_boost * veg_mask)
                )

        # Optional: per-class direct supervision on internal h_b / h_v heads
        # (only V2 models expose these). Squeeze channel dim if present.
        per_class_h_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
        if (h_b is not None and h_v is not None and self.per_class_height_weight > 0):
            h_b_s = h_b.squeeze(1) if h_b.ndim == 4 else h_b
            h_v_s = h_v.squeeze(1) if h_v.ndim == 4 else h_v
            target_h = targets[:, 3]
            loss_h_b = (torch.abs(h_b_s - target_h) * build_mask).sum() / (build_mask.sum() + 1e-6)
            loss_h_v = (torch.abs(h_v_s - target_h) * veg_mask).sum() / (veg_mask.sum() + 1e-6)
            per_class_h_loss = loss_h_b + loss_h_v

        total = (
            self.w_mae * mae
            + self.w_ssim * ssim
            + self.w_grad * grad
            + self.w_structure * tversky
            + self.w_structure * height_boost
            + self.per_class_height_weight * per_class_h_loss
        )

        return dict(
            total_loss=total,
            mae=mae.detach(),
            ssim=ssim.detach(),
            grad=grad.detach(),
            tversky=tversky.detach(),
            height_boost=height_boost.detach(),
            per_class_h=per_class_h_loss.detach() if isinstance(per_class_h_loss, torch.Tensor) else per_class_h_loss,
        )


class BuildingOnlyLoss(nn.Module):
    """Pure single-class segmentation loss.

    Despite the name (kept for backward compat with older configs), this works
    for ANY single class via the `class_idx` parameter:
      class_idx=0  → building specialist (default)
      class_idx=1  → vegetation specialist
      class_idx=2  → water specialist

    Designed for "specialist" runs that share the late-fusion 4-ch architecture
    but only supervise ONE channel. Other channels get NO gradient — their
    outputs are garbage at the end of training and should be ignored at inference.

    Inputs:
      preds:   (B, 4, H, W) — post-sigmoid/softplus, full 4-ch output
      targets: (B, 4 or 5, H, W) — full target tensor

    Loss components (only on preds[:, class_idx]):
      L_tversky = TverskyLoss(α, β)(preds[:,c], targets[:,c])
      L_bce     = BCE(preds[:,c], targets[:,c])    # only when use_bce=True
      total     = tversky_weight * L_tversky + bce_weight * L_bce
    """

    def __init__(self, tversky_alpha=0.5, tversky_beta=0.5,
                 tversky_weight=1.0, bce_weight=1.0, use_bce=True,
                 class_idx=0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=float(tversky_alpha), beta=float(tversky_beta))
        self.tversky_weight = float(tversky_weight)
        self.bce_weight = float(bce_weight)
        self.use_bce = bool(use_bce)
        self.class_idx = int(class_idx)
        assert self.class_idx in (0, 1, 2), \
            f"class_idx must be 0 (bld) / 1 (veg) / 2 (water), got {self.class_idx}"

    def forward(self, preds, targets, loss_mask=None, **kwargs):
        """Optional per-pixel `loss_mask` (B, 1, H, W) or (B, H, W): pixels with
        mask=0 are excluded from BOTH BCE and Tversky. Used for pseudo-label
        training where low-confidence pixels (conf < thresh) get mask=0 so they
        don't pollute gradients."""
        c = self.class_idx
        cls_pred = preds[:, c].clamp(1e-7, 1.0 - 1e-7)   # (B, H, W)
        cls_tgt = targets[:, c]                           # (B, H, W)

        # Squeeze optional channel dim on mask: (B,1,H,W) → (B,H,W)
        if loss_mask is not None:
            if loss_mask.dim() == 4 and loss_mask.shape[1] == 1:
                loss_mask = loss_mask.squeeze(1)
            mask = loss_mask.to(cls_pred.device, dtype=cls_pred.dtype)
        else:
            mask = None

        # BCE: per-pixel cross entropy, then mask-weighted mean
        if self.use_bce:
            bce_per_px = F.binary_cross_entropy(cls_pred, cls_tgt, reduction='none')
            if mask is not None:
                m_sum = mask.sum().clamp(min=1.0)
                bce_loss = (bce_per_px * mask).sum() / m_sum
            else:
                bce_loss = bce_per_px.mean()
        else:
            bce_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)

        # Tversky: mask by zeroing both preds and targets at mask=0 so they
        # contribute 0 to TP/FP/FN (i.e., as if those pixels were "agree on background")
        if mask is not None:
            cls_pred_m = cls_pred * mask
            cls_tgt_m = cls_tgt * mask
            tversky_loss = self.tversky(cls_pred_m, cls_tgt_m)
        else:
            tversky_loss = self.tversky(cls_pred, cls_tgt)

        total = self.tversky_weight * tversky_loss + self.bce_weight * bce_loss
        zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
        return dict(
            total_loss=total,
            tversky=tversky_loss.detach(),
            mae=bce_loss.detach(),              # reuse "mae" slot for BCE for log compat
            ssim=zero,
            grad=zero,
            height_boost=zero,
            per_class_h=zero,
        )


class BaselineSpecialistLoss(nn.Module):
    """Single-class segmentation specialist using emb2heights-baselines composite recipe.

    Matches the structure of `ImprovedCompositeLoss` from VMarsocci/emb2heights-baselines
    (core/losses.py), but restricted to ONE seg channel (no height_boost, no per-class
    Tversky averaging).

    Components (computed only on preds[:, class_idx] vs targets[:, class_idx]):
      - MAE_split  : split fg/bg L1 → mae_fg + bg_weight * mae_bg   (soft-label friendly)
      - SSIM       : structural similarity (window_size=11)         (boundary smoothness)
      - GradDiff   : |∂pred − ∂target| L1                          (edge sharpness)
      - Tversky    : 1 − TP/(TP + α·FP + β·FN)                     (IoU surrogate)

    Total = w_mae·MAE + w_ssim·SSIM + w_grad·GradDiff + w_tversky·Tversky

    Designed for SOFT labels (targets in [0,1], not binarized). MAE_split's fg mask
    uses (target > 0) so soft positives still count.

    NOT in baseline:
      - BCE (we don't use it here; pure baseline mimic)
      - HeightBoost (specialist seg only; ch3 not supervised)
      - per-class Tversky avg over 3 classes (we have only 1 class)

    Inputs:
      preds:   (B, 4, H, W) — model output post-sigmoid/softplus (cf. BuildingOnlyLoss docstring)
      targets: (B, 4+, H, W) — full target tensor, channel `class_idx` is soft fraction in [0,1]
    """

    def __init__(self, class_idx=2,
                 mae_weight=1.0, mae_bg_weight=0.05,
                 ssim_weight=0.5, grad_weight=0.5,
                 tversky_weight=2.0,
                 tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.class_idx = int(class_idx)
        assert self.class_idx in (0, 1, 2), \
            f"class_idx must be 0 (bld) / 1 (veg) / 2 (water), got {self.class_idx}"
        self.mae_weight = float(mae_weight)
        self.mae_bg_weight = float(mae_bg_weight)
        self.ssim_weight = float(ssim_weight)
        self.grad_weight = float(grad_weight)
        self.tversky_weight = float(tversky_weight)
        self.tversky = TverskyLoss(alpha=float(tversky_alpha), beta=float(tversky_beta))
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()

    def forward(self, preds, targets, **kwargs):
        c = self.class_idx
        cls_pred = preds[:, c].clamp(1e-7, 1.0 - 1e-7)   # (B, H, W) in (0,1)
        cls_tgt = targets[:, c]                           # (B, H, W) soft in [0,1]

        # --- MAE_split: fg (target > 0) gets full weight, bg gets mae_bg_weight ---
        abs_err = torch.abs(cls_pred - cls_tgt)
        fg_mask = (cls_tgt > 0).float()
        bg_mask = 1.0 - fg_mask
        fg_sum = fg_mask.sum().clamp(min=1e-6)
        bg_sum = bg_mask.sum().clamp(min=1e-6)
        mae_fg = (abs_err * fg_mask).sum() / fg_sum
        mae_bg = (abs_err * bg_mask).sum() / bg_sum
        mae_loss = mae_fg + self.mae_bg_weight * mae_bg

        # --- Tversky on soft target (TP/FP/FN computed on continuous values) ---
        tversky_loss = self.tversky(cls_pred, cls_tgt)

        # --- SSIM and GradDiff need a (B, 1, H, W) input ---
        cls_pred_4d = cls_pred.unsqueeze(1)
        cls_tgt_4d = cls_tgt.unsqueeze(1)
        if self.ssim_weight > 0:
            ssim_loss = self.ssim(cls_pred_4d, cls_tgt_4d)
        else:
            ssim_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
        if self.grad_weight > 0:
            grad_loss = self.gdl(cls_pred_4d, cls_tgt_4d)
        else:
            grad_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)

        total = (self.mae_weight * mae_loss
                 + self.ssim_weight * ssim_loss
                 + self.grad_weight * grad_loss
                 + self.tversky_weight * tversky_loss)

        zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
        return dict(
            total_loss=total,
            tversky=tversky_loss.detach(),
            mae=mae_loss.detach(),
            ssim=ssim_loss.detach(),
            grad=grad_loss.detach(),
            height_boost=zero,
            per_class_h=zero,
        )


class VegWaterBaselineLoss(nn.Module):
    """JOINT veg+water baseline specialist: two BaselineSpecialistLoss summed.

    Supervises BOTH ch1 (veg) and ch2 (water) of a 4-channel output with the
    emb2heights-baselines composite recipe (MAE_split + SSIM + GradDiff + Tversky),
    sharing the decoder so the two conflicting land-cover classes are learned
    jointly. Used for the test-only overfit distillation (runs/0609 recipe), but
    with one model fitting both veg+water instead of two separate specialists.

    ch0 (building) and ch3 (height) are NOT supervised (their outputs are garbage,
    ignored downstream — combo reuses 0607 building/height).

    Hyperparams apply to BOTH classes identically (faithful reproduction =
    symmetric Tversky α=β=0.5; same as 0609 veg/water single specialists).

    Inputs:
      preds:   (B, 4, H, W) post-sigmoid (ch1=veg, ch2=water in (0,1))
      targets: (B, 4, H, W) soft teacher: ch1=veg fraction, ch2=water fraction in [0,1]
    """

    def __init__(self, mae_weight=1.0, mae_bg_weight=0.05,
                 ssim_weight=0.5, grad_weight=0.5,
                 tversky_weight=2.0, tversky_alpha=0.5, tversky_beta=0.5):
        super().__init__()
        self.veg = BaselineSpecialistLoss(
            class_idx=1, mae_weight=mae_weight, mae_bg_weight=mae_bg_weight,
            ssim_weight=ssim_weight, grad_weight=grad_weight,
            tversky_weight=tversky_weight, tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta)
        self.water = BaselineSpecialistLoss(
            class_idx=2, mae_weight=mae_weight, mae_bg_weight=mae_bg_weight,
            ssim_weight=ssim_weight, grad_weight=grad_weight,
            tversky_weight=tversky_weight, tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta)

    def forward(self, preds, targets, **kwargs):
        v = self.veg(preds, targets)
        w = self.water(preds, targets)
        total = v["total_loss"] + w["total_loss"]   # only this requires grad
        zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
        return dict(
            total_loss=total,
            tversky=(v["tversky"] + w["tversky"]).detach(),
            mae=(v["mae"] + w["mae"]).detach(),
            ssim=(v["ssim"] + w["ssim"]).detach(),
            grad=(v["grad"] + w["grad"]).detach(),
            veg_total=v["total_loss"].detach(),
            water_total=w["total_loss"].detach(),
            height_boost=zero,
            per_class_h=zero,
        )


class BuildingHeightOnlyLoss(nn.Module):
    """Single-class height regression specialist loss.

    Despite the legacy "Building" name, generalized via `mask_class_idx` to
    supervise height at ANY class's GT pixels:
      mask_class_idx=0 → building height specialist
      mask_class_idx=1 → vegetation height specialist
      mask_class_idx=2 → water height specialist (probably useless since GT water height ~0)

    Supervises ONLY the height channel (ch3), masked to GT pixels of the chosen
    class. Other 3 seg channels (ch0/1/2) are NOT supervised — their outputs are
    garbage. LB metrics rmse_b / rmse_v are computed at GT class pixels only, so
    masking to those pixels matches the LB evaluation.

    Inputs:
      preds:   (B, 4, H, W) post-activation, h_pred at preds[:, 3] (normalized [0,1])
      targets: (B, 4 or 5, H, W) full target; targets[:, mask_class_idx]=binary class,
               targets[:, 3]=normalized height

    Loss formulation (normalized space, smooth_l1 default):
      mask = (targets[:, mask_class_idx] > mask_thresh).float()
      err = smooth_l1(h_pred, h_gt, beta=huber_beta, reduction='none')
      L_height = (err * mask).sum() / max(mask.sum(), 1.0)
      total = height_weight * L_height
    """

    def __init__(self, height_weight=1.0,
                 height_loss_type="smooth_l1", huber_beta=1.0,
                 build_mask_thresh=0.5,
                 mask_class_idx=0):
        super().__init__()
        self.height_weight = float(height_weight)
        self.height_loss_type = str(height_loss_type)
        assert self.height_loss_type in ("l1", "smooth_l1", "mse"), \
            f"height_loss_type must be l1/smooth_l1/mse, got {self.height_loss_type}"
        self.huber_beta = float(huber_beta)
        self.build_mask_thresh = float(build_mask_thresh)
        self.mask_class_idx = int(mask_class_idx)
        assert self.mask_class_idx in (0, 1, 2), \
            f"mask_class_idx must be 0 (bld) / 1 (veg) / 2 (water), got {self.mask_class_idx}"

    def forward(self, preds, targets, **kwargs):
        # h_pred normalized to [0, 1] (height_norm_constant=30 already applied in dataset)
        h_pred = preds[:, 3]
        h_tgt = targets[:, 3]

        # Mask from GT class channel (already binary after harden_labels=True).
        # We use > 0.5 since values are exactly 0 or 1; any threshold in (0,1) works.
        mask = (targets[:, self.mask_class_idx] > self.build_mask_thresh).float()

        if self.height_loss_type == "smooth_l1":
            err = F.smooth_l1_loss(h_pred, h_tgt, beta=self.huber_beta, reduction='none')
        elif self.height_loss_type == "l1":
            err = torch.abs(h_pred - h_tgt)
        else:  # mse
            err = (h_pred - h_tgt).pow(2)

        masked_err = err * mask
        n_valid = mask.sum().clamp(min=1.0)
        L_height = masked_err.sum() / n_valid

        total = self.height_weight * L_height
        zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
        return dict(
            total_loss=total,
            tversky=zero,
            mae=L_height.detach(),       # log L_height in mae slot for log compat
            ssim=zero,
            grad=zero,
            height_boost=zero,
            per_class_h=zero,
        )


class BuildingVegHeightOnlyLoss(nn.Module):
    """Joint Building + Vegetation HEIGHT regression loss.

    Supervises ch3 (height) at BOTH GT building AND GT vegetation pixels
    (ignore_bg strategy). Uses per-class normalization to balance the
    pixel-count imbalance (veg ~40% pixels vs bld ~1.85%) — without this,
    veg error would dominate gradient.

    Model output (h_pred) is already in normalized space; loss computes
    smooth_l1(h_pred, h_gt) with separate normalization for each class.
    The prior-residual is built into the architecture, NOT the loss
    (architecture adds prior to delta before returning; loss supervises
    absolute h_pred against h_gt, which is equivalent to delta supervision
    because the prior is a constant offset).

    Inputs:
      preds:   (B, 4, H, W) post-activation; h_pred at preds[:, 3] (normalized [0, 1.5])
      targets: (B, 4 or 5, H, W); ch0=bld_binary, ch1=veg_binary, ch3=h_gt_normalized

    Loss formulation:
      bld_mask = (targets[:, 0] > thresh).float()
      veg_mask = (targets[:, 1] > thresh).float()
      err = smooth_l1(h_pred, h_gt, beta=huber_beta, reduction='none')
      L_bld = (err * bld_mask).sum() / (bld_mask.sum() + eps)
      L_veg = (err * veg_mask).sum() / (veg_mask.sum() + eps)
      total = bld_weight * L_bld + veg_weight * L_veg
    """

    def __init__(self, bld_weight=1.0, veg_weight=1.0,
                 height_loss_type="smooth_l1", huber_beta=0.1,
                 height_mask_thresh=0.5,
                 under_pen_lambda=0.0, under_pen_power=2,
                 ignore_bg=True, bld_boost=0.0, veg_boost=0.0,
                 per_pixel_weighted=False):
        super().__init__()
        self.bld_weight = float(bld_weight)
        self.veg_weight = float(veg_weight)
        self.height_loss_type = str(height_loss_type)
        assert self.height_loss_type in ("l1", "smooth_l1", "mse"), \
            f"height_loss_type must be l1/smooth_l1/mse, got {self.height_loss_type}"
        self.huber_beta = float(huber_beta)
        self.height_mask_thresh = float(height_mask_thresh)
        # NEW: ignore_bg=False mode (baseline-style).
        # When False, supervise EVERY pixel (including bg/water with target h=0),
        # apply per-pixel weight = 1 + bld_boost·bld_mask + veg_boost·veg_mask.
        # When True (default, backward compat), use per-class mean with ignore_bg.
        self.ignore_bg = bool(ignore_bg)
        self.bld_boost = float(bld_boost)
        self.veg_boost = float(veg_boost)
        # NEW: per_pixel_weighted=True (only meaningful when ignore_bg=True).
        # Replaces the per-class mean aggregation with baseline-style per-pixel
        # weighting where each pixel's err is multiplied by its weight and the
        # final sum is divided by foreground pixel COUNT (not normalized by class).
        #   pixel_weight = bld_boost·bld_mask + veg_boost·veg_mask
        #   total = sum(err * pixel_weight) / sum(bld_mask | veg_mask)
        # Use case: mimic baseline ImprovedCompositeLoss weight pattern
        # (bld_boost=12 + veg_boost=2 → effective bld:veg:bg = 13:3:0 after the
        # implicit 1× term is shifted out for ignore_bg semantics).
        # When ignore_bg=False, this flag is IGNORED.
        self.per_pixel_weighted = bool(per_pixel_weighted)
        # Asymmetric UNDER-prediction penalty.
        # When the model predicts h_pred < h_gt (under-predicts), an extra
        # penalty lambda * max(0, h_gt - h_pred)^power is added on top of the
        # base loss. Default 0.0 = off (backward compat).
        # Motivation: bv-height nomask model systematically under-predicts
        # (test mean 7.20m vs train mean 9.41m). L1 optimizes median; asymmetric
        # term pushes model toward upper estimates.
        self.under_pen_lambda = float(under_pen_lambda)
        self.under_pen_power = int(under_pen_power)
        assert self.under_pen_power in (1, 2), \
            f"under_pen_power must be 1 or 2; got {self.under_pen_power}"
        assert self.under_pen_lambda >= 0, \
            f"under_pen_lambda must be ≥0; got {self.under_pen_lambda}"

    def forward(self, preds, targets, **kwargs):
        h_pred = preds[:, 3]
        h_tgt = targets[:, 3]
        bld_mask = (targets[:, 0] > self.height_mask_thresh).float()
        veg_mask = (targets[:, 1] > self.height_mask_thresh).float()

        if self.height_loss_type == "smooth_l1":
            err = F.smooth_l1_loss(h_pred, h_tgt, beta=self.huber_beta, reduction='none')
        elif self.height_loss_type == "l1":
            err = torch.abs(h_pred - h_tgt)
        else:  # mse
            err = (h_pred - h_tgt).pow(2)

        if self.ignore_bg:
            if self.per_pixel_weighted:
                # NEW: per-pixel weighting on fg pixels only.
                # pixel_weight = bld_boost·bld_mask + veg_boost·veg_mask
                # divide by fg pixel count (bld∪veg) to keep gradient scale stable.
                pixel_weight = self.bld_boost * bld_mask + self.veg_boost * veg_mask
                fg_mask = ((bld_mask + veg_mask) > 0).float()
                total = (err * pixel_weight).sum() / (fg_mask.sum() + 1e-6)
                # Per-class diagnostics for logging only
                L_bld = (err * bld_mask).sum() / (bld_mask.sum() + 1e-6)
                L_veg = (err * veg_mask).sum() / (veg_mask.sum() + 1e-6)
            else:
                # Original path: per-class mean over GT bld/veg pixels only.
                L_bld = (err * bld_mask).sum() / (bld_mask.sum() + 1e-6)
                L_veg = (err * veg_mask).sum() / (veg_mask.sum() + 1e-6)
                total = self.bld_weight * L_bld + self.veg_weight * L_veg
        else:
            # Baseline-style: supervise ALL pixels (incl. bg/water with target h=0).
            # Per-pixel weight = 1 + bld_boost·bld_mask + veg_boost·veg_mask.
            # Per-class means kept for logging only.
            pixel_weight = 1.0 + self.bld_boost * bld_mask + self.veg_boost * veg_mask
            total = (err * pixel_weight).mean()
            # Logging metrics: still per-class L for visibility
            L_bld = (err * bld_mask).sum() / (bld_mask.sum() + 1e-6)
            L_veg = (err * veg_mask).sum() / (veg_mask.sum() + 1e-6)

        # Optional asymmetric under-prediction penalty
        if self.under_pen_lambda > 0:
            under = torch.clamp(h_tgt - h_pred, min=0.0)
            if self.under_pen_power == 2:
                under_err = under ** 2
            else:
                under_err = under
            L_bld_under = (under_err * bld_mask).sum() / (bld_mask.sum() + 1e-6)
            L_veg_under = (under_err * veg_mask).sum() / (veg_mask.sum() + 1e-6)
            total = total + self.under_pen_lambda * (
                self.bld_weight * L_bld_under + self.veg_weight * L_veg_under
            )

        zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
        return dict(
            total_loss=total,
            tversky=zero,
            mae=(L_bld + L_veg).detach() / 2.0,  # avg per-class for log
            ssim=zero,
            grad=zero,
            height_boost=L_bld.detach(),     # reuse slot to log per-class loss
            per_class_h=L_veg.detach(),      # reuse slot to log per-class loss
        )


class VegSpecialistLoss(nn.Module):
    """Single-class (vegetation) specialist loss.

    Inputs:
      preds:   (B, 2, H, W) — [veg_prob (after sigmoid), veg_height (normalized)]
      targets: (B, 2, H, W) — [veg_gt, height_gt_normalized]

    Components:
      - Tversky on veg seg (α=0.3 β=0.7)
      - height loss MASKED to pixels where veg_gt > height_mask_thresh.
        Selectable type: "l1" / "smooth_l1" / "mse".
      - Optional grad regularization on HEIGHT (height_grad_weight) — promotes
        spatially smooth height predictions (canopy is locally continuous).
      - Optional grad/ssim regularization on SEG (grad_weight / ssim_weight).
    """

    def __init__(self, seg_weight=1.0, height_weight=10.0,
                 tversky_alpha=0.3, tversky_beta=0.7,
                 height_mask_thresh=0.1,
                 ssim_weight=0.0, grad_weight=0.0,
                 loss_type="l1", huber_beta=0.1,
                 height_grad_weight=0.0,
                 seg_loss_type="tversky", bce_dice_ratio=0.5,
                 height_mask_mode="threshold"):
        """
        height_mask_mode controls how height loss is masked:
          "threshold": hard binary mask (cls_tgt > height_mask_thresh). Original behavior.
          "soft":      continuous weight = cls_tgt (linear by GT class probability).
                       Bg pixels (cls=0) get 0 weight, edge pixels get partial.
          "none":      no mask — height loss computed at ALL pixels uniformly.
                       Useful when downstream wants the height head to predict
                       reasonable values everywhere (e.g., for blind ensemble merging).
        """
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()
        self.seg_weight = seg_weight
        self.height_weight = height_weight
        self.height_mask_thresh = height_mask_thresh
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.loss_type = loss_type
        self.huber_beta = huber_beta
        self.height_grad_weight = height_grad_weight
        self.seg_loss_type = seg_loss_type
        self.bce_dice_ratio = bce_dice_ratio   # weight on BCE vs Dice when seg_loss_type="bce_dice"
        self.height_mask_mode = height_mask_mode
        assert loss_type in ("l1", "smooth_l1", "mse"), \
            f"loss_type must be one of l1/smooth_l1/mse, got {loss_type}"
        assert seg_loss_type in ("tversky", "dice", "bce", "bce_dice"), \
            f"seg_loss_type must be one of tversky/dice/bce/bce_dice, got {seg_loss_type}"
        assert height_mask_mode in ("threshold", "soft", "none"), \
            f"height_mask_mode must be threshold/soft/none, got {height_mask_mode}"

    def forward(self, preds, targets, **kwargs):
        veg_pred = preds[:, 0]                 # (B, H, W)
        veg_tgt = targets[:, 0]
        h_pred = preds[:, 1]
        h_tgt = targets[:, 1]

        # Vegetation segmentation loss (selectable)
        if self.seg_loss_type == "tversky":
            seg_loss = self.tversky(veg_pred, veg_tgt)
        elif self.seg_loss_type == "dice":
            # Dice = Tversky α=β=0.5, equivalent to F1 loss
            tp = (veg_pred * veg_tgt).sum(dim=[1, 2])
            sum_pred = veg_pred.sum(dim=[1, 2])
            sum_tgt = veg_tgt.sum(dim=[1, 2])
            dice = (2 * tp + 1e-6) / (sum_pred + sum_tgt + 1e-6)
            seg_loss = (1 - dice).mean()
        elif self.seg_loss_type == "bce":
            seg_loss = F.binary_cross_entropy(veg_pred, veg_tgt, reduction='mean')
        elif self.seg_loss_type == "bce_dice":
            bce = F.binary_cross_entropy(veg_pred, veg_tgt, reduction='mean')
            tp = (veg_pred * veg_tgt).sum(dim=[1, 2])
            sum_pred = veg_pred.sum(dim=[1, 2])
            sum_tgt = veg_tgt.sum(dim=[1, 2])
            dice = (2 * tp + 1e-6) / (sum_pred + sum_tgt + 1e-6)
            dice_loss = (1 - dice).mean()
            seg_loss = self.bce_dice_ratio * bce + (1 - self.bce_dice_ratio) * dice_loss

        # Height loss. Per-pixel error first, then weighted/masked.
        if self.loss_type == "l1":
            h_err_raw = torch.abs(h_pred - h_tgt)
        elif self.loss_type == "smooth_l1":
            h_err_raw = F.smooth_l1_loss(h_pred, h_tgt, beta=self.huber_beta, reduction='none')
        elif self.loss_type == "mse":
            h_err_raw = (h_pred - h_tgt).pow(2)

        # Mask mode controls which pixels contribute
        if self.height_mask_mode == "threshold":
            veg_mask = (veg_tgt > self.height_mask_thresh).float()
            height_loss = (h_err_raw * veg_mask).sum() / (veg_mask.sum() + 1e-6)
        elif self.height_mask_mode == "soft":
            # Weight by GT class probability (no threshold)
            height_loss = (h_err_raw * veg_tgt).sum() / (veg_tgt.sum() + 1e-6)
            veg_mask = veg_tgt   # for downstream height_grad masking
        elif self.height_mask_mode == "none":
            # No mask — average over all pixels
            height_loss = h_err_raw.mean()
            veg_mask = torch.ones_like(veg_tgt)

        # Optional regularizers
        if self.ssim_weight > 0:
            ssim_loss = self.ssim(veg_pred.unsqueeze(1), veg_tgt.unsqueeze(1))
        else:
            ssim_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
        if self.grad_weight > 0:
            grad_loss = self.gdl(veg_pred.unsqueeze(1), veg_tgt.unsqueeze(1))
        else:
            grad_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
        if self.height_grad_weight > 0:
            # GDL on HEIGHT (only where veg is present — multiply pred and tgt by mask first)
            h_pred_m = (h_pred * veg_mask).unsqueeze(1)
            h_tgt_m = (h_tgt * veg_mask).unsqueeze(1)
            height_grad_loss = self.gdl(h_pred_m, h_tgt_m)
        else:
            height_grad_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)

        total = (
            self.seg_weight * seg_loss
            + self.height_weight * height_loss
            + self.ssim_weight * ssim_loss
            + self.grad_weight * grad_loss
            + self.height_grad_weight * height_grad_loss
        )

        return dict(
            total_loss=total,
            seg_loss=seg_loss.detach(),
            height_loss=height_loss.detach(),
            ssim_loss=ssim_loss.detach(),
            grad_loss=grad_loss.detach(),
            height_grad_loss=height_grad_loss.detach(),
        )




class SoftmaxCE4ClassLoss(nn.Module):
    """4-class softmax CE on seg + smooth_l1 height (ignore_bg).

    Designed for the CE seg-specialist (Plan A). The model body outputs
    (B, 5, H, W) = [4 raw seg logits, 1 height (post-softplus, in normalized [0,1])].
    Targets are (B, 4, H, W) float with ch0/1/2 = bld/veg/water soft labels
    and ch3 = height. Implicit bg = 1 - bld - veg - water.

    Loss = ce_weight * weighted_CE(seg_logits, class_idx)
         + dice_weight * SoftDice(softmax_probs, class_one_hot)  [bld+water only]
         + height_weight * smooth_l1(h_pred, h_gt) * fg_mask

    fg_mask = (gt_bld > thresh) | (gt_veg > thresh) — same as height_ignore_bg.

    Args:
      ce_weight, dice_weight, height_weight: loss term weights.
      class_weights: list of 4 floats for (bld, veg, water, bg). None → uniform.
      label_thresh: positive threshold (default 0.1, matches harden_thresh).
      huber_beta: smooth_l1 transition point (default 0.1 = 3m at norm 30).
      include_dice: classes to include in Dice term (default [0, 2] = bld + water).
    """

    def __init__(self,
                 ce_weight=1.0,
                 dice_weight=0.5,
                 height_weight=1.5,
                 class_weights=None,
                 label_thresh=0.1,
                 huber_beta=0.1,
                 include_dice=(0, 2)):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.height_weight = float(height_weight)
        self.label_thresh = float(label_thresh)
        self.huber_beta = float(huber_beta)
        self.include_dice = tuple(int(c) for c in include_dice)
        if class_weights is None:
            self.register_buffer("class_weights", None, persistent=False)
        else:
            cw = torch.tensor(list(class_weights), dtype=torch.float32)
            assert cw.numel() == 4, f"class_weights must have 4 entries (bld,veg,water,bg); got {cw.numel()}"
            self.register_buffer("class_weights", cw, persistent=False)

    @staticmethod
    def _soft_argmax_4cls(targets, label_thresh):
        """Derive (B, H, W) class index from (B, 4, H, W) float labels.

        Implicit bg = max(0, 1 - bld - veg - water). Take argmax over [bld, veg, water, bg].
        Works for both soft and hard labels (after harden=True, hard ones are 0/1).
        Background tie-breaks via clamped (1 - sum); if all classes below thresh, bg wins.
        """
        b = targets[:, 0]
        v = targets[:, 1]
        w = targets[:, 2]
        bg = (1.0 - b - v - w).clamp_min(0.0)
        # If all positive classes are below thresh, force bg to dominate. We do this
        # by adding a small epsilon to bg when no class exceeds thresh — but cleaner:
        # just argmax over the 4-way stack including bg. If multiple positives, the
        # one with highest score wins. This is the natural softmax-target behavior.
        stacked = torch.stack([b, v, w, bg], dim=1)
        return stacked.argmax(dim=1).long()

    def forward(self, preds, targets, **kwargs):
        """preds: (B, 5, H, W) where ch0-3 are RAW seg logits, ch4 is softplus height.
        targets: (B, 4, H, W) float — ch0/1/2 soft seg labels, ch3 height (normalized).
        """
        assert preds.shape[1] == 5, f"SoftmaxCE4ClassLoss expects 5-ch preds; got {preds.shape[1]}"
        assert targets.shape[1] == 4, f"SoftmaxCE4ClassLoss expects 4-ch targets; got {targets.shape[1]}"

        seg_logits = preds[:, :4]                      # (B, 4, H, W) raw
        h_pred = preds[:, 4]                            # (B, H, W) softplus'd, normalized
        h_gt = targets[:, 3]                            # (B, H, W) normalized

        # 1. CE on derived class index
        cls_idx = self._soft_argmax_4cls(targets, self.label_thresh)
        ce = F.cross_entropy(
            seg_logits, cls_idx,
            weight=self.class_weights,
            reduction='mean',
        )

        # 2. Soft Dice on selected classes (bld + water by default)
        if self.dice_weight > 0 and self.include_dice:
            probs = F.softmax(seg_logits, dim=1)       # (B, 4, H, W)
            one_hot = F.one_hot(cls_idx, num_classes=4).permute(0, 3, 1, 2).float()
            dice_losses = []
            for c in self.include_dice:
                p = probs[:, c]
                t = one_hot[:, c]
                inter = (p * t).sum()
                denom = p.sum() + t.sum() + 1e-6
                dice = 1.0 - (2.0 * inter / denom)
                dice_losses.append(dice)
            dice_loss = torch.stack(dice_losses).mean()
        else:
            dice_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)

        # 3. Smooth-L1 height with ignore_bg (only at GT bld∪veg pixels)
        fg_mask = ((targets[:, 0] > self.label_thresh) |
                   (targets[:, 1] > self.label_thresh)).float()
        n_valid = fg_mask.sum().clamp_min(1.0)
        h_err = F.smooth_l1_loss(h_pred, h_gt, beta=self.huber_beta, reduction='none')
        height_loss = (h_err * fg_mask).sum() / n_valid

        total = (self.ce_weight * ce
                 + self.dice_weight * dice_loss
                 + self.height_weight * height_loss)

        zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
        return dict(
            total_loss=total,
            tversky=ce.detach(),        # reuse 'tversky' slot for logging compat
            mae=dice_loss.detach(),     # reuse 'mae' slot for Dice logging
            ssim=zero,
            grad=zero,
            height_boost=height_loss.detach(),
            per_class_h=zero,
        )


class MultiTaskHRNetLoss(nn.Module):
    """Multi-task loss combining 4 specialists' EXACT winning recipes.

    Architecture: shared HRNet body, 4-channel output, per-class independent loss.
    No averaging of Tversky across classes — each class gets full specialist gradient.

    Per-class recipes (each operates on its own channel only):
      - ch0 (bld):    BCE + Tversky(α=0.5, β=0.5)
                      (matches building_specialist_hrnet_tokenfusion_10k, LB iou_b 0.4939)
      - ch1 (veg):    MAE_split + 0.5·SSIM + 0.5·Grad + 2.0·Tversky(α=0.5, β=0.5)
                      ("Option β" — water spec structural recipe with veg-symmetric α)
      - ch2 (water):  MAE_split + 0.5·SSIM + 0.5·Grad + 2.0·Tversky(α=0.3, β=0.7)
                      (matches water_specialist_baseline_hrnet_5k, LB iou_w 0.5022)
      - ch3 (height): smooth_l1 + ignore_bg + per-class mean
                      (matches height_hrnet_tokenfusion_10k, LB rmse_b 1.8324)

    Designed for SOFT labels (harden_labels=False) — water specialist breakthrough
    depended on this. height_mask_thresh=0.1 recovers HARD-mode height coverage
    from soft targets (D4 design finding).

    Total = w_bld·L_bld + w_veg·L_veg + w_water·L_water + w_height·L_height

    Default w_height=5.0 because smooth_l1 magnitude ~0.05-0.10 while seg losses ~0.4-0.8.
    Tune after first 100 steps based on gradient norms.
    """

    def __init__(self,
                 w_bld=1.0, w_veg=1.0, w_water=1.0, w_height=5.0,
                 # Tversky α/β per channel
                 bld_tversky_alpha=0.5, bld_tversky_beta=0.5,
                 veg_tversky_alpha=0.5, veg_tversky_beta=0.5,
                 water_tversky_alpha=0.3, water_tversky_beta=0.7,
                 # bld portion (BCE + Tversky only — no MAE/SSIM/Grad)
                 bld_bce_weight=1.0, bld_tversky_weight=1.0,
                 # NEW: switch bld to baseline-style (MAE+SSIM+Grad+Tversky) like veg/water
                 bld_use_baseline_style=False,
                 # veg + water shared portion (MAE+SSIM+Grad+Tversky weights)
                 mae_weight=1.0, mae_bg_weight=0.05,
                 ssim_weight=0.5, grad_weight=0.5,
                 tversky_weight=2.0,
                 # height portion
                 height_loss_type="smooth_l1", huber_beta=0.1,
                 height_mask_thresh=0.1, ignore_bg=True):
        super().__init__()
        self.w_bld = float(w_bld)
        self.w_veg = float(w_veg)
        self.w_water = float(w_water)
        self.w_height = float(w_height)
        self.bld_use_baseline_style = bool(bld_use_baseline_style)

        # ch0 bld: either BCE+Tversky (default) OR baseline-style (MAE+SSIM+Grad+Tversky)
        if self.bld_use_baseline_style:
            self.bld_loss = BaselineSpecialistLoss(
                class_idx=0,
                mae_weight=mae_weight, mae_bg_weight=mae_bg_weight,
                ssim_weight=ssim_weight, grad_weight=grad_weight,
                tversky_weight=tversky_weight,
                tversky_alpha=bld_tversky_alpha, tversky_beta=bld_tversky_beta,
            )
        else:
            self.bld_loss = BuildingOnlyLoss(
                tversky_alpha=bld_tversky_alpha, tversky_beta=bld_tversky_beta,
                tversky_weight=bld_tversky_weight,
                bce_weight=bld_bce_weight, use_bce=True,
                class_idx=0,
            )
        # ch1 veg: composite via BaselineSpecialistLoss(class_idx=1)
        self.veg_loss = BaselineSpecialistLoss(
            class_idx=1,
            mae_weight=mae_weight, mae_bg_weight=mae_bg_weight,
            ssim_weight=ssim_weight, grad_weight=grad_weight,
            tversky_weight=tversky_weight,
            tversky_alpha=veg_tversky_alpha, tversky_beta=veg_tversky_beta,
        )
        # ch2 water: composite via BaselineSpecialistLoss(class_idx=2)
        self.water_loss = BaselineSpecialistLoss(
            class_idx=2,
            mae_weight=mae_weight, mae_bg_weight=mae_bg_weight,
            ssim_weight=ssim_weight, grad_weight=grad_weight,
            tversky_weight=tversky_weight,
            tversky_alpha=water_tversky_alpha, tversky_beta=water_tversky_beta,
        )
        # ch3 height: BuildingVegHeightOnlyLoss with specialist-exact recipe
        self.height_loss = BuildingVegHeightOnlyLoss(
            bld_weight=1.0, veg_weight=1.0,
            height_loss_type=height_loss_type, huber_beta=huber_beta,
            height_mask_thresh=height_mask_thresh,
            under_pen_lambda=0.0, under_pen_power=2,
            ignore_bg=ignore_bg,
            bld_boost=0.0, veg_boost=0.0,
            per_pixel_weighted=False,
        )

    def forward(self, preds, targets, **kwargs):
        l_bld = self.bld_loss(preds, targets)
        l_veg = self.veg_loss(preds, targets)
        l_water = self.water_loss(preds, targets)
        l_h = self.height_loss(preds, targets)

        total = (self.w_bld * l_bld["total_loss"]
                 + self.w_veg * l_veg["total_loss"]
                 + self.w_water * l_water["total_loss"]
                 + self.w_height * l_h["total_loss"])

        zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
        # Aggregate per-component metrics for logging (averages across classes that have them).
        # BuildingOnlyLoss reuses 'mae' slot for BCE — keep separate for clarity.
        return dict(
            total_loss=total,
            # Tversky avg across the 3 seg channels (each computed with own α/β)
            tversky=((l_bld["tversky"] + l_veg["tversky"] + l_water["tversky"]) / 3.0).detach(),
            # MAE_split avg across veg+water (bld has no MAE; bld's 'mae' slot holds BCE)
            mae=((l_veg["mae"] + l_water["mae"]) / 2.0).detach(),
            # SSIM avg across veg+water (bld doesn't compute SSIM)
            ssim=((l_veg["ssim"] + l_water["ssim"]) / 2.0).detach(),
            # GradDiff avg across veg+water
            grad=((l_veg["grad"] + l_water["grad"]) / 2.0).detach(),
            # height_boost slot: store the height loss subtotal
            height_boost=l_h["total_loss"].detach() if torch.is_tensor(l_h["total_loss"]) else zero,
            # per_class_h slot: store bld's BCE for monitoring
            per_class_h=l_bld["mae"].detach(),
        )
