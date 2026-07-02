import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def infer_model_type(n_channels):
    return "decoder_residual" if n_channels == 768 else "lightunet"


# sdpa backend control. PyTorch's automatic selector occasionally picks the
# math backend (which materialises the full (B,H,N_q,N_kv) attention matrix)
# even when memory-efficient or Flash would work. For V4's branch 0 attention
# (N_q = 192² = 36864 at training), the math matrix is ~1.4 GB and the backward
# stash blows up activation memory. Force memory-efficient / Flash explicitly.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    # FLASH+EFFICIENT preferred (don't materialize the attention matrix).
    # MATH is included as a fallback so a per-call shape mismatch (e.g. an
    # unusually small head_dim) doesn't crash the run — math just costs more
    # memory on that one block. With head_dims all multiples of 8 in V4,
    # mem_efficient should be picked on CUDA in practice.
    _SDPA_PREFER_BACKENDS = [SDPBackend.FLASH_ATTENTION,
                             SDPBackend.EFFICIENT_ATTENTION,
                             SDPBackend.MATH]
    _HAS_SDPA_KERNEL = True
except ImportError:
    _SDPA_PREFER_BACKENDS = None
    _HAS_SDPA_KERNEL = False


import math as _math

# Priors computed from training labels (200 sample patches, threshold=0.1):
#   Building heights: mean=3.75m, median=3.11m
#   Vegetation heights: mean=8.82m, median=7.69m
# Normalized to height_norm_constant=30:
PRIOR_H_B_NORMALIZED = 0.125   # 3.75 / 30
PRIOR_H_V_NORMALIZED = 0.294   # 8.82 / 30

# Class frequency priors (soft label means across train set):
#   Building   = 1.24%   (very sparse)
#   Vegetation = 40.36%
#   Water      = 2.12%   (sparse)
# Convert to logits: log(p / (1-p)). Used as bias init so sigmoid(bias) = prior.
SEG_LOGIT_PRIOR_BUILDING   = _math.log(0.0124 / (1 - 0.0124))   # ≈ -4.374
SEG_LOGIT_PRIOR_VEGETATION = _math.log(0.4036 / (1 - 0.4036))   # ≈ -0.391
SEG_LOGIT_PRIOR_WATER      = _math.log(0.0212 / (1 - 0.0212))   # ≈ -3.831


class CascadeDualHeadV2(nn.Module):
    """Cascade dual decoder with 2-channel height head + prior-residual.

    Differences vs CascadeDualDecoderUNet:
      * Height head outputs 2 channels (per-class: h_b, h_v) instead of 1
      * Each channel predicts a RESIDUAL added to a fixed prior (PRIOR_H_*)
      * Final height = soft-mix using detached seg-probs:
            final_h = seg_prob_b * h_b + seg_prob_v * h_v
        Background pixels (both seg low) → near 0 naturally.
      * Output channel 3 is the FINAL normalized height (NOT a logit).
        GeoFMNet.activate must NOT apply softplus to it.

    The model exposes `_latest_h_b` and `_latest_h_v` as buffers/attributes
    so the loss can do per-class direct supervision.
    """

    output_height_pre_activated = True  # tells GeoFMNet.activate to skip softplus on ch 3

    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        assert out_channels == 4

        # === Shared encoder (same as cascade) ===
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # === Seg decoder ===
        self.s_up1 = UpsampleBlock(256, 128)
        self.s_conv1 = DoubleConv(256, 128)
        self.s_up2 = UpsampleBlock(128, 64)
        self.s_conv2 = DoubleConv(128, 64)
        self.s_up3 = UpsampleBlock(64, 32)
        self.s_conv3 = DoubleConv(64, 32)
        self.seg_head = nn.Conv2d(32, 3, kernel_size=1)
        # Class-frequency bias init for seg head (RetinaNet-style trick).
        # Small weight init → initial sigmoid(output) ≈ class prior, not 0.5.
        nn.init.normal_(self.seg_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.seg_head.bias[0], SEG_LOGIT_PRIOR_BUILDING)
        nn.init.constant_(self.seg_head.bias[1], SEG_LOGIT_PRIOR_VEGETATION)
        nn.init.constant_(self.seg_head.bias[2], SEG_LOGIT_PRIOR_WATER)

        # === Height decoder (with cascade seg injection at final scale) ===
        self.h_up1 = UpsampleBlock(256, 128)
        self.h_conv1 = DoubleConv(256, 128)
        self.h_up2 = UpsampleBlock(128, 64)
        self.h_conv2 = DoubleConv(128, 64)
        self.h_up3 = UpsampleBlock(64, 32)
        self.h_conv3 = DoubleConv(64 + 3, 32)   # +3 for detached seg-probs
        # 2-channel head: residual for building height, residual for vegetation height
        self.height_head = nn.Conv2d(32, 2, kernel_size=1)
        # Init residual head near zero so initial prediction ≈ prior
        nn.init.normal_(self.height_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.height_head.bias)

        # Register priors as buffers so they move with .to(device)
        self.register_buffer("prior_h_b", torch.tensor(PRIOR_H_B_NORMALIZED))
        self.register_buffer("prior_h_v", torch.tensor(PRIOR_H_V_NORMALIZED))

        # Latest per-class height predictions (for loss to access)
        self._latest_h_b = None
        self._latest_h_v = None

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Seg decoder
        s = self.s_up1(x4)
        s = self.s_conv1(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_conv2(torch.cat([x2, s], dim=1))
        s = self.s_up3(s)
        s = self.s_conv3(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)  # (B, 3, H, W) — raw logits

        # Detach seg-probs for cascade (no grad flow back to seg from height)
        seg_probs_detached = torch.sigmoid(seg_logits).detach()

        # Height decoder
        h = self.h_up1(x4)
        h = self.h_conv1(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_conv2(torch.cat([x2, h], dim=1))
        h = self.h_up3(h)
        h = self.h_conv3(torch.cat([x1, h, seg_probs_detached], dim=1))
        h_residual = self.height_head(h)   # (B, 2, H, W) — residuals (can be neg)

        # Add prior to get absolute (normalized) heights
        h_b = self.prior_h_b + h_residual[:, 0:1]
        h_v = self.prior_h_v + h_residual[:, 1:2]
        # Clip to valid range (0 to 15 normalized = 450m max)
        h_b = torch.clamp(h_b, min=0.0, max=15.0)
        h_v = torch.clamp(h_v, min=0.0, max=15.0)

        # Soft-mix using detached seg-probs (cascade preserved)
        final_h = (seg_probs_detached[:, 0:1] * h_b +
                   seg_probs_detached[:, 1:2] * h_v)
        # Background pixels (both probs low) → final_h ≈ 0 naturally

        # Stash per-class for loss access
        self._latest_h_b = h_b
        self._latest_h_v = h_v

        # Output (B, 4, H, W): 3 seg LOGITS + 1 final normalized height
        return torch.cat([seg_logits, final_h], dim=1)


class AdapterFusionCascadeV2(nn.Module):
    """Adapter stem + CascadeDualHeadV2 (prior-residual)."""

    # Mark that the height channel of forward() output is already a final value (not logit)
    @property
    def output_height_pre_activated(self):
        return self.body.output_height_pre_activated

    def __init__(self, in_channels, out_channels=4, source_channels=None, adapter_out=64):
        super().__init__()
        assert out_channels == 4
        if source_channels is None:
            assert in_channels == 192
            source_channels = (64, 128)
        else:
            source_channels = tuple(source_channels)
            assert sum(source_channels) == in_channels

        self.source_channels = source_channels
        self.source_offsets = [0]
        for c in source_channels:
            self.source_offsets.append(self.source_offsets[-1] + c)

        self.adapters = nn.ModuleList([
            SourceAdapterStem(c, adapter_out) for c in source_channels
        ])

        fused_in = adapter_out * len(source_channels)
        self.body = CascadeDualHeadV2(in_channels=fused_in, out_channels=out_channels)

    def forward(self, x):
        parts = []
        for i, adapter in enumerate(self.adapters):
            start, end = self.source_offsets[i], self.source_offsets[i + 1]
            parts.append(adapter(x[:, start:end]))
        fused = torch.cat(parts, dim=1)
        return self.body(fused)


class SourceAdapterStem(nn.Module):
    """Per-source adapter for fusion: GroupNorm + 1x1 + 3x3 conv with GroupNorm + GELU.

    Normalizes each source's distribution and projects to common channel width,
    so downstream encoder doesn't have to learn to compensate for cross-source
    scale differences.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups_in = min(8, in_channels)
        groups_out = min(8, out_channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups_in, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups_out, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups_out, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AdapterFusionCascadeUNet(nn.Module):
    """Adapter-fused cascade dual-decoder UNet.

    Layout:
      AE (64) ─→ adapter_ae   ─┐
                                ├──→ concat ─→ shared encoder ─→ cascade dual decoder
      Tessera (128) ─→ adapter_te ┘

    Each source goes through GroupNorm + 1x1 + 3x3 first, then concat at common
    channel width, then the encoder + dual decoder operates on fused features.
    """

    def __init__(self, in_channels, out_channels=4, source_channels=None, adapter_out=64):
        super().__init__()
        assert out_channels == 4
        if source_channels is None:
            # Default to AE(64) + Tessera(128) when in_channels == 192
            assert in_channels == 192, (
                f"AdapterFusionCascadeUNet defaults to AE(64)+Tessera(128) when in_channels=192; "
                f"got in_channels={in_channels}. Specify source_channels explicitly."
            )
            source_channels = (64, 128)
        else:
            source_channels = tuple(source_channels)
            assert sum(source_channels) == in_channels, \
                f"sum(source_channels)={sum(source_channels)} != in_channels={in_channels}"

        self.source_channels = source_channels
        # Offsets for slicing the concatenated input
        self.source_offsets = [0]
        for c in source_channels:
            self.source_offsets.append(self.source_offsets[-1] + c)

        # Per-source adapters
        self.adapters = nn.ModuleList([
            SourceAdapterStem(c, adapter_out) for c in source_channels
        ])

        # Fused channel count
        fused_in = adapter_out * len(source_channels)

        # Body: cascade dual decoder, but operating on fused features
        self.body = CascadeDualDecoderUNet(in_channels=fused_in, out_channels=out_channels)

    def forward(self, x):
        # Split by source, apply each adapter, then concat
        parts = []
        for i, adapter in enumerate(self.adapters):
            start, end = self.source_offsets[i], self.source_offsets[i + 1]
            parts.append(adapter(x[:, start:end]))
        fused = torch.cat(parts, dim=1)
        return self.body(fused)


class ShuffleBlock(nn.Module):
    """Lite-HRNet style shuffle block (split-transform-concat + channel shuffle).

    Halves the channels, applies depthwise + pointwise on one half,
    concats, then shuffles channels to mix information.
    Cheaper than regular conv blocks.
    """

    def __init__(self, channels):
        super().__init__()
        assert channels % 2 == 0
        c2 = channels // 2
        self.right = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),  # depthwise
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, 1, bias=False),                          # pointwise
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c = x.shape[1]
        x1 = x[:, : c // 2]
        x2 = x[:, c // 2 :]
        x2 = self.right(x2)
        out = torch.cat([x1, x2], dim=1)
        # Channel shuffle: interleave groups of 2
        b, c2, h, w = out.shape
        out = out.view(b, 2, c2 // 2, h, w).transpose(1, 2).contiguous().view(b, c2, h, w)
        return out


class HRStage(nn.Module):
    """One HR stage: K parallel branches processed independently then fused
    across resolutions.

    Inputs / outputs: list of tensors at different resolutions.
    """

    def __init__(self, channels, num_blocks=2):
        super().__init__()
        self.num_branches = len(channels)
        self.channels = channels

        # Per-branch processing (sequence of ShuffleBlocks)
        self.branches = nn.ModuleList([
            nn.Sequential(*[ShuffleBlock(c) for _ in range(num_blocks)])
            for c in channels
        ])

        # Cross-resolution fusion modules
        self.fusion = self._build_fusion(channels)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _build_fusion(channels):
        n = len(channels)
        fusion = nn.ModuleList()
        # fusion[i][j] takes branch j and adapts to branch i's resolution
        for i in range(n):
            row = nn.ModuleList()
            for j in range(n):
                if i == j:
                    row.append(nn.Identity())
                elif j > i:
                    # source j has LOWER res (higher index = more downsampled),
                    # upsample to branch i resolution
                    row.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i),
                                    mode="bilinear",
                                    align_corners=False),
                    ))
                else:  # j < i: source j has HIGHER res, downsample to i
                    layers = []
                    cur = channels[j]
                    for k in range(i - j):
                        out_c = channels[i] if k == (i - j - 1) else cur
                        layers.append(nn.Conv2d(cur, out_c, 3, stride=2, padding=1, bias=False))
                        layers.append(nn.BatchNorm2d(out_c))
                        if k < i - j - 1:
                            layers.append(nn.ReLU(inplace=True))
                        cur = out_c
                    row.append(nn.Sequential(*layers))
            fusion.append(row)
        return fusion

    def forward(self, branches):
        # 1) Per-branch processing
        branches = [self.branches[i](branches[i]) for i in range(self.num_branches)]
        # 2) Multi-resolution fusion: each branch gets sum of all transformed branches
        out = []
        for i in range(self.num_branches):
            agg = sum(self.fusion[i][j](branches[j]) for j in range(self.num_branches))
            out.append(self.relu(agg))
        return out


class LiteHRNetBody(nn.Module):
    """Simplified Lite-HRNet style multi-resolution body for dense prediction.

    Structure:
      Input (fused, e.g. 128ch @ 256×256)
        → stem (32ch @ full res)
        → spawn branches: 64ch @ 1/2, 128ch @ 1/4
        → 3 HR stages (ShuffleBlock × 2 per branch + cross-resolution fusion)
        → final fusion at full res (concat + 1×1 conv)
        → dual heads (seg 3 channels, height 1 channel) -- dual_only style
    """

    def __init__(self, in_channels, out_channels=4, branch_channels=(32, 64, 128),
                 num_stages=3, blocks_per_branch=2):
        super().__init__()
        assert out_channels == 4

        # Stem keeps full resolution
        self.stem = DoubleConv(in_channels, branch_channels[0])

        # Build transitions to lower-resolution branches
        self.transitions = nn.ModuleList()
        for k in range(1, len(branch_channels)):
            self.transitions.append(nn.Sequential(
                nn.Conv2d(branch_channels[k - 1], branch_channels[k],
                          3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels[k]),
                nn.ReLU(inplace=True),
            ))

        # HR stages
        self.stages = nn.ModuleList([
            HRStage(channels=list(branch_channels), num_blocks=blocks_per_branch)
            for _ in range(num_stages)
        ])

        # Final fusion: concat all branches at full res
        total_in = sum(branch_channels)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(total_in, branch_channels[0], 1, bias=False),
            nn.BatchNorm2d(branch_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels[0], branch_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Dual heads (dual_only style: no cross-task info flow)
        c_final = branch_channels[0]
        self.seg_head = nn.Conv2d(c_final, 3, kernel_size=1)
        self.height_head = nn.Conv2d(c_final, 1, kernel_size=1)

    def forward(self, x):
        # Initial stem
        b0 = self.stem(x)             # full res, 32ch
        # Spawn lower-resolution branches by successive downsampling
        branches = [b0]
        cur = b0
        for trans in self.transitions:
            cur = trans(cur)
            branches.append(cur)

        # HR stages
        for stage in self.stages:
            branches = stage(branches)

        # Upsample all branches to full res, concat, fuse
        h, w = branches[0].shape[2], branches[0].shape[3]
        upsampled = [branches[0]]
        for i in range(1, len(branches)):
            upsampled.append(F.interpolate(branches[i], size=(h, w),
                                           mode="bilinear", align_corners=False))
        fused = self.final_fuse(torch.cat(upsampled, dim=1))

        # Dual heads
        seg_logits = self.seg_head(fused)      # (B, 3, H, W)
        height_logits = self.height_head(fused)  # (B, 1, H, W)
        return torch.cat([seg_logits, height_logits], dim=1)


class AdapterFusionLiteHRNet(nn.Module):
    """Adapter stem + LiteHRNetBody. Drop-in replacement for
    AdapterFusionDualOnlyUNet — same input/output shape (B, 4, H, W).
    """

    def __init__(self, in_channels, out_channels=4, source_channels=None, adapter_out=64):
        super().__init__()
        assert out_channels == 4
        if source_channels is None:
            assert in_channels == 192, \
                f"AdapterFusionLiteHRNet defaults to AE(64)+Tessera(128); got in={in_channels}"
            source_channels = (64, 128)
        else:
            source_channels = tuple(source_channels)
            assert sum(source_channels) == in_channels

        self.source_channels = source_channels
        self.source_offsets = [0]
        for c in source_channels:
            self.source_offsets.append(self.source_offsets[-1] + c)

        self.adapters = nn.ModuleList([
            SourceAdapterStem(c, adapter_out) for c in source_channels
        ])

        fused_in = adapter_out * len(source_channels)
        self.body = LiteHRNetBody(in_channels=fused_in, out_channels=out_channels)

    def forward(self, x):
        parts = []
        for i, adapter in enumerate(self.adapters):
            start, end = self.source_offsets[i], self.source_offsets[i + 1]
            parts.append(adapter(x[:, start:end]))
        fused = torch.cat(parts, dim=1)
        return self.body(fused)


class LiteHRNetBodyDualDecoder(nn.Module):
    """Lite-HRNet body with TWO independent decoders for seg and height.

    Same multi-res encoder + cross-resolution fusion as LiteHRNetBody.
    Difference: the final stage splits into two parallel decoders, each
    consuming the concatenated multi-resolution features independently.
    Mimics 'adapter_dual_only' philosophy: no shared post-encoder processing.

    Args:
      branch_channels: tuple of channel widths per branch (low→high res index).
      num_stages: number of HR stages (each does shuffle blocks per branch +
        cross-resolution fusion).
      blocks_per_branch: ShuffleBlocks per branch per stage.
      decoder_depth: number of DoubleConv refinement layers in each task decoder.
    """

    def __init__(self, in_channels, out_channels=4, branch_channels=(32, 64, 128),
                 num_stages=3, blocks_per_branch=2, decoder_depth=1):
        super().__init__()
        assert out_channels == 4
        assert decoder_depth >= 1

        # Same encoder as LiteHRNetBody
        self.stem = DoubleConv(in_channels, branch_channels[0])
        self.transitions = nn.ModuleList()
        for k in range(1, len(branch_channels)):
            self.transitions.append(nn.Sequential(
                nn.Conv2d(branch_channels[k - 1], branch_channels[k],
                          3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels[k]),
                nn.ReLU(inplace=True),
            ))
        self.stages = nn.ModuleList([
            HRStage(channels=list(branch_channels), num_blocks=blocks_per_branch)
            for _ in range(num_stages)
        ])

        # Two PARALLEL decoders (each: 1x1 reduce + N×DoubleConv refine + 1x1 head)
        total_in = sum(branch_channels)
        c = branch_channels[0]

        def _build_decoder():
            layers = [
                nn.Conv2d(total_in, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            for _ in range(decoder_depth):
                layers.append(DoubleConv(c, c))
            return nn.Sequential(*layers)

        self.seg_decoder = _build_decoder()
        self.height_decoder = _build_decoder()

        self.seg_head = nn.Conv2d(c, 3, kernel_size=1)
        self.height_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        b0 = self.stem(x)
        branches = [b0]
        cur = b0
        for trans in self.transitions:
            cur = trans(cur)
            branches.append(cur)

        # HR stages
        for stage in self.stages:
            branches = stage(branches)

        # Upsample all to full res, concat (no shared fuse)
        h, w = branches[0].shape[2], branches[0].shape[3]
        upsampled = [branches[0]]
        for i in range(1, len(branches)):
            upsampled.append(F.interpolate(branches[i], size=(h, w),
                                           mode="bilinear", align_corners=False))
        fused_features = torch.cat(upsampled, dim=1)  # (B, sum_channels, H, W)

        # Two independent decoders
        seg_features = self.seg_decoder(fused_features)
        height_features = self.height_decoder(fused_features)

        seg_logits = self.seg_head(seg_features)
        height_logits = self.height_head(height_features)
        return torch.cat([seg_logits, height_logits], dim=1)


class AdapterFusionLiteHRNetDual(nn.Module):
    """Adapter stem + LiteHRNetBodyDualDecoder (true dual decoder)."""

    def __init__(self, in_channels, out_channels=4, source_channels=None, adapter_out=64):
        super().__init__()
        assert out_channels == 4
        if source_channels is None:
            assert in_channels == 192
            source_channels = (64, 128)
        else:
            source_channels = tuple(source_channels)
            assert sum(source_channels) == in_channels

        self.source_channels = source_channels
        self.source_offsets = [0]
        for c in source_channels:
            self.source_offsets.append(self.source_offsets[-1] + c)

        self.adapters = nn.ModuleList([
            SourceAdapterStem(c, adapter_out) for c in source_channels
        ])

        fused_in = adapter_out * len(source_channels)
        self.body = LiteHRNetBodyDualDecoder(in_channels=fused_in, out_channels=out_channels)

    def forward(self, x):
        parts = []
        for i, adapter in enumerate(self.adapters):
            start, end = self.source_offsets[i], self.source_offsets[i + 1]
            parts.append(adapter(x[:, start:end]))
        fused = torch.cat(parts, dim=1)
        return self.body(fused)


class AdapterFusionLiteHRNetHeavy(nn.Module):
    """Adapter stem + Lite-HRNet-Heavy body.

    Heavy variant — strictly more capacity than the default Lite-HRNet:
      * 4 branches at (40, 80, 160, 192) channels (adds a 1/8 resolution branch)
      * 4 HR stages (was 3)
      * 3 ShuffleBlocks per branch (was 2)
      * Deeper dual decoder (decoder_depth=2 — two DoubleConvs)

    Target ~1.5-2M params (vs original Lite-HRNet 0.74M).
    """

    def __init__(self, in_channels, out_channels=4, source_channels=None, adapter_out=64):
        super().__init__()
        assert out_channels == 4
        if source_channels is None:
            assert in_channels == 192
            source_channels = (64, 128)
        else:
            source_channels = tuple(source_channels)
            assert sum(source_channels) == in_channels

        self.source_channels = source_channels
        self.source_offsets = [0]
        for c in source_channels:
            self.source_offsets.append(self.source_offsets[-1] + c)

        self.adapters = nn.ModuleList([
            SourceAdapterStem(c, adapter_out) for c in source_channels
        ])

        fused_in = adapter_out * len(source_channels)
        self.body = LiteHRNetBodyDualDecoder(
            in_channels=fused_in,
            out_channels=out_channels,
            branch_channels=(40, 80, 160, 192),
            num_stages=4,
            blocks_per_branch=3,
            decoder_depth=2,
        )

    def forward(self, x):
        parts = []
        for i, adapter in enumerate(self.adapters):
            start, end = self.source_offsets[i], self.source_offsets[i + 1]
            parts.append(adapter(x[:, start:end]))
        fused = torch.cat(parts, dim=1)
        return self.body(fused)


class LiteHRNetBodyTokenFusion(nn.Module):
    """4-branch heavy Lite-HRNet body with tokens fused INTO the 1/8 branch.

    Plan B design (vs older 1/16 5-branch attempt):
      - 4 dense branches at (1, 1/2, 1/4, 1/8) — same scales as heavy HRNet
      - Tokens (pre-adapted at 1/16, e.g. 256ch by wrapper) bilinear-upsampled 2×
        → 1/8 spatial, then concat with the 1/8 dense branch and reduced via
        DoubleConvLN to branch_channels[3] before HR stages start.
      - 4 HR stages × 4 branches (ShuffleBlock + cross-resolution fusion).
      - Shallow dual decoder (depth=1) → 4-ch output (seg 3 + height 1).

    Inputs (called by AdapterFusionLiteHRNetTokenFusion AFTER per-source adapters):
      dense:  (B, dense_in_channels, H, W) — fused dense feature map
      tokens: (B, token_in_channels, H/16, H/16) — fused token feature map

    Designed for SPECIALIST single-class training (only one of 4 output channels
    supervised via BuildingOnlyLoss). Decoder is shallow (depth=1).

    Spatial constraint: dense H and W must be multiples of 16 (so tokens at 1/16
    upsample×2 cleanly to 1/8). 256 ✓, 192 ✓.
    """

    is_late_fusion = True   # tells GeoFMNet to pass tokens to forward()

    def __init__(self, dense_in_channels=128, token_in_channels=256, out_channels=4,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1):
        super().__init__()
        assert out_channels == 4
        assert len(branch_channels) == 4, \
            f"branch_channels must have 4 entries (1, 1/2, 1/4, 1/8), got {len(branch_channels)}"
        assert all(c % 2 == 0 for c in branch_channels), \
            f"All branch_channels must be even (ShuffleBlock split); got {branch_channels}"

        # Dense path: stem + 3 strided transitions (heavy HRNet style)
        self.stem = DoubleConv(dense_in_channels, branch_channels[0])
        self.transitions = nn.ModuleList()
        for k in range(1, 4):
            self.transitions.append(nn.Sequential(
                nn.Conv2d(branch_channels[k - 1], branch_channels[k],
                          3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels[k]),
                nn.ReLU(inplace=True),
            ))

        # Token fusion at 1/8: tokens come at 1/16 from wrapper, bilinear ×2 → 1/8,
        # then concat with the 1/8 dense branch (branch_channels[3]) and reduce
        # via a LN double-conv back to branch_channels[3] so HRStage channel math
        # stays consistent across stages.
        self.token_fuse = DoubleConvLN(branch_channels[3] + token_in_channels,
                                       branch_channels[3])

        # HR stages — 4 branches, same as existing heavy HRNet
        self.stages = nn.ModuleList([
            HRStage(channels=list(branch_channels), num_blocks=blocks_per_branch)
            for _ in range(num_stages)
        ])

        # Shallow dual decoder (specialist context — depth=1)
        total_in = sum(branch_channels)
        c = branch_channels[0]

        def _build_decoder():
            layers = [
                nn.Conv2d(total_in, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            for _ in range(decoder_depth):
                layers.append(DoubleConv(c, c))
            return nn.Sequential(*layers)

        self.seg_decoder = _build_decoder()
        self.height_decoder = _build_decoder()
        self.seg_head = nn.Conv2d(c, 3, kernel_size=1)
        self.height_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, dense, tokens):
        # dense: (B, dense_in_channels, H, W), W=H=multiple of 16
        # tokens: (B, token_in_channels, H/16, H/16)
        assert dense.dim() == 4 and tokens.dim() == 4
        assert dense.shape[-1] % 16 == 0 and dense.shape[-2] % 16 == 0, \
            f"Dense input must be multiple of 16, got {dense.shape[-2]}x{dense.shape[-1]}"
        expected_tok = dense.shape[-1] // 16
        assert tokens.shape[-1] == expected_tok and tokens.shape[-2] == expected_tok, \
            f"Token spatial {tokens.shape[-2]}x{tokens.shape[-1]} != expected " \
            f"{expected_tok}x{expected_tok} for dense {dense.shape[-2]}x{dense.shape[-1]}"

        # Dense branches (4 of them, scales 1, 1/2, 1/4, 1/8)
        b0 = self.stem(dense)
        b1 = self.transitions[0](b0)
        b2 = self.transitions[1](b1)
        b3_dense = self.transitions[2](b2)

        # Fuse tokens into b3: upsample 2× (1/16 → 1/8), concat, reduce via LN conv
        tokens_at_1_8 = F.interpolate(tokens, scale_factor=2.0,
                                      mode="bilinear", align_corners=False)
        b3 = self.token_fuse(torch.cat([b3_dense, tokens_at_1_8], dim=1))

        branches = [b0, b1, b2, b3]

        # 4 HR stages on 4 branches (cross-resolution fusion as in heavy HRNet)
        for stage in self.stages:
            branches = stage(branches)

        # Final: upsample all to full res, concat
        h, w = branches[0].shape[2], branches[0].shape[3]
        upsampled = [branches[0]]
        for i in range(1, len(branches)):
            upsampled.append(F.interpolate(branches[i], size=(h, w),
                                           mode="bilinear", align_corners=False))
        fused_features = torch.cat(upsampled, dim=1)

        # Dual decoders (shallow)
        seg_features = self.seg_decoder(fused_features)
        height_features = self.height_decoder(fused_features)
        seg_logits = self.seg_head(seg_features)
        height_logits = self.height_head(height_features)
        return torch.cat([seg_logits, height_logits], dim=1)


class AdapterFusionLiteHRNetTokenFusion(nn.Module):
    """LN per-source adapters (matching a05) + LiteHRNetBodyTokenFusion.

    Mirrors AdapterFusionLateFusionUNet's adapter pattern (SourceAdapterStemLN
    for both dense and token sources), differing only in body — HRNet (heavy
    multi-resolution) instead of UNet late-fusion.

    Designed for ensemble diversity with a05 (UNet late-fusion specialist).
    """

    is_late_fusion = True   # GeoFMNet.forward will pass tokens through

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        # Per-source LN adapters (same as a05/AdapterFusionLateFusionUNet)
        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        fused_dense_in = adapter_out * len(self.dense_channels)   # e.g. 64×2 = 128
        fused_token_in = adapter_out * len(self.token_channels)   # e.g. 64×4 = 256

        self.body = LiteHRNetBodyTokenFusion(
            dense_in_channels=fused_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            branch_channels=branch_channels,
            num_stages=num_stages,
            blocks_per_branch=blocks_per_branch,
            decoder_depth=decoder_depth,
        )

    def forward(self, x_dense, tokens):
        # x_dense is concat of dense sources; split + per-source LN adapt + concat
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            start, end = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, start:end]))
        fused_dense = torch.cat(dense_parts, dim=1)

        # tokens is concat of token sources @ 1/16; split + per-source LN adapt + concat
        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            start, end = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, start:end]))
        fused_tokens = torch.cat(token_parts, dim=1)

        return self.body(fused_dense, fused_tokens)


class RGBTokenAdapter(nn.Module):
    """Adapter for DINOv3-L RGB token features (1024 ch, high-spatial 120/160).

    Mirrors SourceAdapterStemLN's LN-based structure but takes 1024 input channels.
    Resolution-agnostic: the same module handles train 120×120 and test 160×160.
    """
    def __init__(self, in_channels=1024, out_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            ChannelLN(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AdapterFusionLiteHRNetTokenFusionRGB(nn.Module):
    """Like AdapterFusionLiteHRNetTokenFusion but with a 5th modality:
    DINOv3-L RGB token features (1024 ch) at high spatial resolution (120 train / 160 test).

    Pipeline:
      Dense adapters (AE+Tessera)   → fused_dense  (B, 128, H, W)
      Token adapters (4 × 768 @ 16) → fused_tokens (B, 256, 16, 16)
      RGB path:
        absent_token + has_rgb gate (multiplicative for DDP grad)
        modality dropout (train only, p=rgb_modality_dropout)
        RGBTokenAdapter (1024→adapter_out)        (B, 64, h_rgb, w_rgb)
        bilinear upsample to (H, W)               (B, 64, H, W)
        concat to fused_dense → (B, 192, H, W)
      body(fused_dense_with_rgb, fused_tokens) → (B, 4, H, W)

    Body dense_in_channels = 128 + 64 = 192 (matches the +RGB-adapter width).
    """

    is_late_fusion = True
    requires_rgb_token = True   # consumed by GeoFMNet.forward and GeoFMEmbed2Heights.forward

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 rgb_token_channels=1024,
                 out_channels=4, adapter_out=64,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1,
                 rgb_modality_dropout=0.10):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.rgb_token_channels = int(rgb_token_channels)
        self.adapter_out = adapter_out
        self.rgb_modality_dropout = float(rgb_modality_dropout)
        assert 0.0 <= self.rgb_modality_dropout < 1.0, \
            f"rgb_modality_dropout out of range: {self.rgb_modality_dropout}"

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        # RGB token adapter (1024 → adapter_out). Resolution-agnostic.
        self.rgb_adapter = RGBTokenAdapter(self.rgb_token_channels, adapter_out)

        # Learned "RGB absent" token, broadcast over spatial dims. Initialized small
        # random so absent-tile gradient is non-degenerate.
        self.rgb_absent_token = nn.Parameter(
            torch.randn(self.rgb_token_channels) * 0.02
        )

        fused_dense_in = adapter_out * len(self.dense_channels)        # 64×2 = 128
        fused_token_in = adapter_out * len(self.token_channels)        # 64×4 = 256
        body_dense_in = fused_dense_in + adapter_out                   # 128 + 64 = 192

        self.body = LiteHRNetBodyTokenFusion(
            dense_in_channels=body_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            branch_channels=branch_channels,
            num_stages=num_stages,
            blocks_per_branch=blocks_per_branch,
            decoder_depth=decoder_depth,
        )

    def forward(self, x_dense, tokens, rgb_token=None, has_rgb=None):
        # x_dense: (B, sum_dense_ch, H, W); tokens: (B, sum_token_ch, 16, 16)
        # rgb_token: (B, 1024, h_rgb, w_rgb); has_rgb: (B,) bool
        if rgb_token is None:
            raise ValueError(
                "AdapterFusionLiteHRNetTokenFusionRGB.forward requires rgb_token. "
                "Loader must populate meta['rgb_token']."
            )
        if has_rgb is None:
            # Conservative: treat as all-absent. The model still works via absent_token.
            has_rgb = torch.zeros(rgb_token.shape[0], dtype=torch.bool, device=rgb_token.device)

        B, _, H, W = x_dense.shape

        # Modality dropout (train only): randomly flip has_rgb→False for some samples.
        # Independent per-rank RNG (CUDA rand_int) — DDP uses different seeds per rank
        # (see train.py + seed_with_rank).
        if self.training and self.rgb_modality_dropout > 0.0:
            drop = torch.rand(B, device=has_rgb.device) < self.rgb_modality_dropout
            has_rgb = has_rgb & (~drop)

        # Build (B, 1024, h, w) using multiplicative gate so absent_token is ALWAYS
        # in autograd graph (DDP-safe; no find_unused_parameters needed).
        absent = self.rgb_absent_token.view(1, -1, 1, 1)                # (1, 1024, 1, 1)
        absent = absent.expand_as(rgb_token)                            # (B, 1024, h, w)
        mask = has_rgb.to(dtype=rgb_token.dtype).view(B, 1, 1, 1)       # (B, 1, 1, 1)
        rgb_in = mask * rgb_token + (1.0 - mask) * absent               # (B, 1024, h, w)

        # 1024 → adapter_out at native h,w
        rgb_feat = self.rgb_adapter(rgb_in)                              # (B, 64, h, w)
        # Upsample to dense H,W (handles both 120→192 train and 160→256 test)
        if rgb_feat.shape[-2:] != (H, W):
            rgb_feat = F.interpolate(rgb_feat, size=(H, W),
                                     mode="bilinear", align_corners=False)

        # Dense adapters
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            start, end = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, start:end]))
        dense_parts.append(rgb_feat)                                     # +RGB feat at 64 ch
        fused_dense = torch.cat(dense_parts, dim=1)                      # (B, 128+64=192, H, W)

        # Token adapters @ 1/16
        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            start, end = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, start:end]))
        fused_tokens = torch.cat(token_parts, dim=1)                     # (B, 256, 16, 16)

        return self.body(fused_dense, fused_tokens)


class AdapterMultiTaskLiteHRNetRGBOnly(nn.Module):
    """RGB-ONLY multi-task model. The DINOv3-L sat493m RGB token features
    (1024 ch @ 120 train / 160 test) are the SOLE input — no AlphaEarth/Tessera
    dense sources, no TerraMind/THOR tokens. Output is the same 4-channel
    multi-task target (seg ch0/1/2 + height ch3) as the token-fusion multitask
    model (`adapter_fusion_lite_hrnet_token_fusion`).

    Pipeline (mirrors the collaborator's RGB handling, then a dense-only body):
      absent_token + has_rgb gate (multiplicative, DDP-safe) on rgb_token
      modality dropout (train only; default 0.0 — RGB is the only modality so
        dropping it just feeds absent_token, kept off by default)
      RGBTokenAdapter (1024 → adapter_out)                  (B, A, h, w)
      bilinear upsample to the dense/label H,W                (B, A, H, W)
      LiteHRNetBodyDualDecoder (dense-only, seg+height heads) (B, 4, H, W)

    The loader still streams x_dense (AE+Tessera) and tokens so the data contract
    is identical to the RGB-fuse run — but this body uses ONLY x_dense.shape[-2:]
    (for the output spatial size) and IGNORES the dense content and the tokens.

    Flags `is_late_fusion`/`requires_rgb_token` make GeoFMNet.forward and
    GeoFMEmbed2Heights route tokens + rgb_token + has_rgb here, same as the
    `_rgb` fusion body.
    """

    is_late_fusion = True
    requires_rgb_token = True

    def __init__(self, rgb_token_channels=1024, out_channels=4, adapter_out=128,
                 branch_channels=(40, 80, 160, 192), num_stages=4,
                 blocks_per_branch=3, decoder_depth=1, rgb_modality_dropout=0.0):
        super().__init__()
        assert out_channels == 4
        self.rgb_token_channels = int(rgb_token_channels)
        self.adapter_out = int(adapter_out)
        self.rgb_modality_dropout = float(rgb_modality_dropout)
        assert 0.0 <= self.rgb_modality_dropout < 1.0, \
            f"rgb_modality_dropout out of range: {self.rgb_modality_dropout}"

        # 1024 → adapter_out at native RGB resolution (resolution-agnostic).
        self.rgb_adapter = RGBTokenAdapter(self.rgb_token_channels, self.adapter_out)
        # Learned "RGB absent" token (small random init) for the ~4-5% tiles with
        # no DINOv3 cache — without it those tiles would have no input at all.
        self.rgb_absent_token = nn.Parameter(
            torch.randn(self.rgb_token_channels) * 0.02
        )

        # Dense-only HRNet with parallel seg/height decoders. Same branch widths
        # / depth as the multitask HRNet specialist (≈ its capacity minus tokens).
        self.body = LiteHRNetBodyDualDecoder(
            in_channels=self.adapter_out,
            out_channels=out_channels,
            branch_channels=branch_channels,
            num_stages=num_stages,
            blocks_per_branch=blocks_per_branch,
            decoder_depth=decoder_depth,
        )

    def forward(self, x_dense, tokens=None, rgb_token=None, has_rgb=None):
        # x_dense: (B, sum_dense_ch, H, W) — used ONLY for output size (H, W).
        # tokens:  ignored.
        # rgb_token: (B, 1024, h, w); has_rgb: (B,) bool.
        if rgb_token is None:
            raise ValueError(
                "AdapterMultiTaskLiteHRNetRGBOnly.forward requires rgb_token. "
                "Loader must populate meta['rgb_token']."
            )
        B = rgb_token.shape[0]
        H, W = x_dense.shape[-2:]
        if has_rgb is None:
            has_rgb = torch.zeros(B, dtype=torch.bool, device=rgb_token.device)

        if self.training and self.rgb_modality_dropout > 0.0:
            drop = torch.rand(B, device=has_rgb.device) < self.rgb_modality_dropout
            has_rgb = has_rgb & (~drop)

        # Multiplicative absent-token gate (keeps absent_token in autograd graph;
        # DDP-safe, no find_unused_parameters needed).
        absent = self.rgb_absent_token.view(1, -1, 1, 1).expand_as(rgb_token)
        mask = has_rgb.to(dtype=rgb_token.dtype).view(B, 1, 1, 1)
        rgb_in = mask * rgb_token + (1.0 - mask) * absent

        feat = self.rgb_adapter(rgb_in)                       # (B, A, h, w)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W),
                                 mode="bilinear", align_corners=False)
        return self.body(feat)


class AdapterFusionLiteHRNetDenseUpsample(nn.Module):
    """Dense-only HRNet that GPU-upsamples token sources to dense H,W.

    Architectural test variant: instead of using tokens as a separate 1/16 branch
    (AdapterFusionLiteHRNetTokenFusion), this class bilinear-upsamples the 16x16
    token sources to the dense H,W ON THE GPU at forward time, concatenates with
    AE+Tessera as if they're additional dense sources, and feeds the unified
    dense input to a dual-decoder LiteHRNet body.

    Why GPU-side upsample (vs DataLoader-side `token_upsample=True`):
      - DataLoader-side: each sample upsampled becomes 855 MB; with prefetch ×
        num_workers × DDP ranks, easily exceeds node CPU RAM → silent OOM kill.
      - GPU-side: each sample stays ~3 MB in DataLoader (16x16 native), upsample
        happens after batch transfer to GPU at ~ms cost.

    Architecture per forward:
      x_dense (B, 192, H, W)        — AE 64 + Tessera 128 (already concatenated)
      tokens (B, 3072, h_tok, w_tok=16)  — 4 token sources × 768 ch (concat in dataloader)
      ↓ F.interpolate(tokens, size=(H, W), bilinear)
      tokens_up (B, 3072, H, W)
      ↓ split per source + per-source LN adapter
      6 adapted parts × adapter_out=64 ch → concat → (B, 384, H, W)
      ↓ LiteHRNetBodyDualDecoder (true dual decoder)
      (B, 4, H, W) = 3 seg + 1 height

    Matches mt_hrnet_bldbase recipe by using dual decoder (same as token_fusion variant),
    just swaps the token integration mechanism. Use to compare separate-branch vs
    upsample-and-merge approaches on identical loss.
    """

    is_late_fusion = True   # tell GeoFMNet.forward to pass tokens through

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 branch_channels=(32, 64, 128),
                 num_stages=3, blocks_per_branch=2, decoder_depth=1):
        super().__init__()
        assert out_channels == 4

        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        # Combined source-channel offsets (dense first, then token sources)
        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        # Per-source LN adapters (same SourceAdapterStemLN as token_fusion variant)
        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        # All 6 sources merged into one dense path
        fused_in = adapter_out * (len(self.dense_channels) + len(self.token_channels))
        # Use LiteHRNetBodyDualDecoder to match mt_hrnet_bldbase's true dual decoder
        self.body = LiteHRNetBodyDualDecoder(
            in_channels=fused_in,
            out_channels=out_channels,
            branch_channels=branch_channels,
            num_stages=num_stages,
            blocks_per_branch=blocks_per_branch,
            decoder_depth=decoder_depth,
        )

    def forward(self, x_dense, tokens):
        # x_dense: (B, sum(dense_channels), H, W)
        # tokens:  (B, sum(token_channels), h_tok, w_tok=16)
        B = x_dense.shape[0]
        H, W = x_dense.shape[-2:]

        # ★ GPU-side bilinear upsample of all token sources to dense H,W
        tokens_up = F.interpolate(tokens, size=(H, W), mode="bilinear", align_corners=False)

        # Per-source LN adapt: dense first, then upsampled tokens
        parts = []
        for i, adapter in enumerate(self.dense_adapters):
            start, end = self.dense_offsets[i], self.dense_offsets[i + 1]
            parts.append(adapter(x_dense[:, start:end]))
        for i, adapter in enumerate(self.token_adapters):
            start, end = self.token_offsets[i], self.token_offsets[i + 1]
            parts.append(adapter(tokens_up[:, start:end]))

        fused = torch.cat(parts, dim=1)  # (B, adapter_out * 6, H, W)
        return self.body(fused)


# ---------------------------------------------------------------------------
# SegFormer MiT-B0 building specialist (transformer-encoder variant of the
# HRNet token-fusion specialist).
#
# Encoder code is a self-contained port of the NVIDIA SegFormer mix_transformer
# reference (segformer/mix_transformer.py in this repo), stripped of mmseg /
# timm registry hooks so it loads without those dependencies. Behaviour
# (weight init, attention, DWConv MLP, overlap patch embed) is preserved.
# ---------------------------------------------------------------------------


def _mit_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= max(m.groups, 1)
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class _MiTDWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class _MiTMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = _MiTDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(_mit_init_weights)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class _MiTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(_mit_init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # Memory-efficient attention via PyTorch sdpa (Flash / mem_eff backend).
        # Mathematically equivalent to the original `softmax(Q @ K^T * scale) @ V`
        # because sdpa defaults to scale = 1/sqrt(head_dim), matching self.scale,
        # and applies dropout post-softmax just like the original. State_dict
        # layout (q/kv/proj/sr/norm) is unchanged so existing ckpts still load.
        # Explicit kernel selection (excluding math backend) is required to
        # avoid materialising the full (B,H,N_q,N_kv) attention matrix on
        # high-resolution branches (V4 branch 0: N_q = 192² = 36864).
        dropout_p = self.attn_drop.p if self.training else 0.0
        if _HAS_SDPA_KERNEL:
            with sdpa_kernel(_SDPA_PREFER_BACKENDS):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class _MiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _MiTAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop,
                                  proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _MiTMlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                           act_layer=act_layer, drop=drop)
        self.apply(_mit_init_weights)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class _MiTOverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_mit_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MitB0BodyTokenFusion(nn.Module):
    """MiT-B0 (SegFormer) backbone + 1/16 token fusion + OS=1 progressive decoder.

    V2 design: addresses the V1 boundary-quality problem where the All-MLP
    decoder operated at OS=4 then bilinearly upsampled ×4 to full res, leaving
    no learnable refinement at OS=2 or OS=1 (building edges came out as 4-pixel
    blocks). Now mirrors the HRNet specialist's "decoder stays at OS=1" idea:

      - Two parallel high-resolution conv stems on the dense input keep fine
        spatial detail alive (encoder otherwise loses everything finer than OS=4):
          stem_full @ OS=1, 32 ch   (small DoubleConvLN)
          stem_half @ OS=2, 64 ch   (strided from stem_full)
      - MiT encoder runs at OS=4/8/16/32 (with token fusion injected at OS=16
        before stage 4) and the All-MLP fuse produces 256ch @ OS=4.
      - **Progressive UNet-style decoder** (replaces the single bilinear ×4):
          OS=4 (256) → bilinear×2 → concat stem_half (+64) → DoubleConvLN → 128 @ OS=2
                    → bilinear×2 → concat stem_full (+32) → DoubleConvLN →  64 @ OS=1
      - Dual shallow heads (seg / height) operate on the OS=1 features.

    Input / output contract is unchanged from V1:
      dense (B, dense_in_channels, H, W),  H,W multiples of 16
      tokens (B, token_in_channels, H/16, W/16)
      → (B, 4, H, W)  (3 seg + 1 height)
    """

    is_late_fusion = True

    def __init__(self, dense_in_channels=128, token_in_channels=256, out_channels=4,
                 embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 qkv_bias=True,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 decoder_dim=256,
                 norm_layer=None,
                 stem_full_dim=32,
                 stem_half_dim=64,
                 dec_os2_dim=128,
                 dec_os1_dim=64):
        super().__init__()
        assert out_channels == 4
        assert len(embed_dims) == 4 and len(num_heads) == 4
        assert len(mlp_ratios) == 4 and len(depths) == 4 and len(sr_ratios) == 4
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # ----- High-resolution detail stems (parallel to MiT encoder) -----
        # These do NOT learn semantics; they keep OS=1/OS=2 spatial detail
        # available as skip features for the decoder so the heads work on
        # truly-OS=1 features, not bilinearly-upsampled OS=4 features.
        self.stem_full = DoubleConvLN(dense_in_channels, stem_full_dim)        # OS=1
        self.stem_half = StridedDoubleConvLN(stem_full_dim, stem_half_dim)     # OS=2

        # ----- Patch embeds (stride 4 / 2 / 2 / 2) -----
        self.patch_embed1 = _MiTOverlapPatchEmbed(
            patch_size=7, stride=4, in_chans=dense_in_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # ----- Transformer blocks -----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        def _stage(idx, depth):
            blocks = nn.ModuleList([_MiTBlock(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx],
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=0.,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[idx])
                for i in range(depth)])
            return blocks

        self.block1 = _stage(0, depths[0]); self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]
        self.block2 = _stage(1, depths[1]); self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]
        self.block3 = _stage(2, depths[2]); self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]
        self.block4 = _stage(3, depths[3]); self.norm4 = norm_layer(embed_dims[3])

        # ----- 1/16 token fusion: concat onto stage-3 feature map and reduce -----
        # tokens arrive at 1/16, stage-3 features are at 1/16 → no spatial resize.
        self.token_fuse = DoubleConvLN(embed_dims[2] + token_in_channels, embed_dims[2])

        # ----- All-MLP fuse @ OS=4 (unchanged from V1) -----
        self.linear_c1 = nn.Conv2d(embed_dims[0], decoder_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(embed_dims[1], decoder_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(embed_dims[2], decoder_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(embed_dims[3], decoder_dim, kernel_size=1)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            ChannelLN(decoder_dim),
            nn.GELU(),
        )

        # ----- Progressive OS=4 → OS=2 → OS=1 decoder with hi-res skips -----
        # OS=4 (decoder_dim) → bilinear×2 → concat stem_half → conv → dec_os2_dim @ OS=2
        self.conv_os2 = DoubleConvLN(decoder_dim + stem_half_dim, dec_os2_dim)
        # OS=2 → bilinear×2 → concat stem_full → conv → dec_os1_dim @ OS=1
        self.conv_os1 = DoubleConvLN(dec_os2_dim + stem_full_dim, dec_os1_dim)

        # Dual shallow seg / height decoders + heads, all operating at OS=1
        # (mirrors HRNet specialist's depth-1 dual decoder design).
        self.seg_decoder = DoubleConvLN(dec_os1_dim, dec_os1_dim // 2)
        self.height_decoder = DoubleConvLN(dec_os1_dim, dec_os1_dim // 2)
        self.seg_head = nn.Conv2d(dec_os1_dim // 2, 3, kernel_size=1)
        self.height_head = nn.Conv2d(dec_os1_dim // 2, 1, kernel_size=1)

    def _run_stage(self, x_img, patch_embed, blocks, norm):
        x, H, W = patch_embed(x_img)
        for blk in blocks:
            x = blk(x, H, W)
        x = norm(x)
        B = x.shape[0]
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, dense, tokens):
        assert dense.dim() == 4 and tokens.dim() == 4
        H_in, W_in = dense.shape[-2], dense.shape[-1]
        assert H_in % 16 == 0 and W_in % 16 == 0, \
            f"Dense input must be multiple of 16, got {H_in}x{W_in}"
        expected_tok_h, expected_tok_w = H_in // 16, W_in // 16
        assert tokens.shape[-2] == expected_tok_h and tokens.shape[-1] == expected_tok_w, \
            f"Token spatial {tokens.shape[-2]}x{tokens.shape[-1]} != expected " \
            f"{expected_tok_h}x{expected_tok_w} for dense {H_in}x{W_in}"

        # ----- High-res detail skip features (parallel to MiT encoder) -----
        s_full = self.stem_full(dense)         # (B, stem_full_dim, H,   W  )
        s_half = self.stem_half(s_full)        # (B, stem_half_dim, H/2, W/2)

        # ----- MiT encoder stages 1-3 -----
        c1 = self._run_stage(dense, self.patch_embed1, self.block1, self.norm1)  # 1/4
        c2 = self._run_stage(c1,    self.patch_embed2, self.block2, self.norm2)  # 1/8
        c3 = self._run_stage(c2,    self.patch_embed3, self.block3, self.norm3)  # 1/16

        # ----- Token fusion at 1/16 -----
        c3 = self.token_fuse(torch.cat([c3, tokens], dim=1))

        # ----- Stage 4 -----
        c4 = self._run_stage(c3, self.patch_embed4, self.block4, self.norm4)     # 1/32

        # ----- All-MLP fuse @ OS=4 -----
        h4, w4 = c1.shape[-2], c1.shape[-1]
        f1 = self.linear_c1(c1)
        f2 = F.interpolate(self.linear_c2(c2), size=(h4, w4), mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.linear_c3(c3), size=(h4, w4), mode="bilinear", align_corners=False)
        f4 = F.interpolate(self.linear_c4(c4), size=(h4, w4), mode="bilinear", align_corners=False)
        fused_os4 = self.linear_fuse(torch.cat([f4, f3, f2, f1], dim=1))         # (B, D, H/4, W/4)

        # ----- Progressive OS=4 → OS=2 → OS=1 with hi-res skips -----
        h2, w2 = s_half.shape[-2], s_half.shape[-1]
        up_os2 = F.interpolate(fused_os4, size=(h2, w2), mode="bilinear", align_corners=False)
        x_os2 = self.conv_os2(torch.cat([up_os2, s_half], dim=1))                # OS=2, dec_os2_dim

        up_os1 = F.interpolate(x_os2, size=(H_in, W_in), mode="bilinear", align_corners=False)
        x_os1 = self.conv_os1(torch.cat([up_os1, s_full], dim=1))                # OS=1, dec_os1_dim

        # ----- Dual shallow heads at OS=1 -----
        seg_logits = self.seg_head(self.seg_decoder(x_os1))
        height_logits = self.height_head(self.height_decoder(x_os1))
        return torch.cat([seg_logits, height_logits], dim=1)


def load_mit_b0_pretrained(body, ckpt_path, strict_keys=False, verbose=True):
    """Load NVIDIA-style MiT-B0 ImageNet weights into a MitB0BodyTokenFusion.

    The supplied checkpoint key prefixes (`patch_embed{1-4}`, `block{1-4}`,
    `norm{1-4}`) match the names exposed by our `_MiT*` modules, so most
    weights drop in directly.

    Skipped on purpose:
      - `patch_embed1.proj.*`  : pretrained is (32, 3, 7, 7), our model uses
        (32, dense_in_channels, 7, 7). Channel count differs → random init.
      - `head.*` : ImageNet classification head, not used here.

    Anything else with a shape mismatch is skipped with a warning.
    Decoder / token_fuse / stem_full / stem_half / heads stay at their original
    random init since they have no counterpart in the pretrained checkpoint.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    own = body.state_dict()
    loaded, skipped_missing, skipped_shape = [], [], []

    for k, v in sd.items():
        if k.startswith("head."):
            continue
        if k.startswith("patch_embed1.proj."):
            # channel mismatch (dense_in != 3) — keep random init
            skipped_shape.append((k, tuple(v.shape)))
            continue
        if k not in own:
            skipped_missing.append(k)
            continue
        if own[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(own[k].shape)))
            continue
        own[k] = v
        loaded.append(k)

    body.load_state_dict(own, strict=strict_keys)
    if verbose:
        print(f"[mit_b0 pretrained] loaded {len(loaded)} tensors from {ckpt_path}")
        if skipped_missing:
            print(f"[mit_b0 pretrained] {len(skipped_missing)} ckpt keys not in model (sample: {skipped_missing[:3]})")
        if skipped_shape:
            print(f"[mit_b0 pretrained] {len(skipped_shape)} keys skipped due to shape mismatch")
            for entry in skipped_shape[:5]:
                print(f"    {entry}")
    return len(loaded), len(skipped_missing), len(skipped_shape)


class AdapterFusionMitB0TokenFusion(nn.Module):
    """LN per-source adapters + MitB0BodyTokenFusion.

    Mirrors AdapterFusionLiteHRNetTokenFusion's wrapper exactly: dense sources
    are split, each goes through a SourceAdapterStemLN, then concatenated to
    form the MiT input; token sources likewise concat post-adapter and are
    handed to the body as the 1/16 token feature map.

    Goal: transformer-encoder cousin of the HRNet building specialist for
    architectural-diversity ensembling.
    """

    is_late_fusion = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 drop_path_rate=0.1,
                 decoder_dim=256,
                 pretrained_mit_b0=None):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        fused_dense_in = adapter_out * len(self.dense_channels)
        fused_token_in = adapter_out * len(self.token_channels)

        self.body = MitB0BodyTokenFusion(
            dense_in_channels=fused_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
            decoder_dim=decoder_dim,
        )
        if pretrained_mit_b0:
            load_mit_b0_pretrained(self.body, pretrained_mit_b0)

    def forward(self, x_dense, tokens):
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        fused_dense = torch.cat(dense_parts, dim=1)

        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)

        return self.body(fused_dense, fused_tokens)


# ---------------------------------------------------------------------------
# V3: MiT-B0 encoder + HRNet's EXACT decoder (strict encoder-only swap).
#
# Rationale: V2-A changed encoder AND decoder AND added hi-res conv stems,
# which mixed three independent variables. V3 keeps the decoder byte-for-byte
# identical to LiteHRNetBodyTokenFusion (BN+ReLU+DoubleConv, no LN, no stems,
# no progressive upsample) so the delta vs the HRNet specialist (LB 0.4939)
# is *purely* the encoder swap.
# ---------------------------------------------------------------------------


class MitB0BodyHRNetDecoder(nn.Module):
    """MiT-B0 encoder + LiteHRNet-style decoder.

    Encoder is the same MiT-B0 stack as MitB0BodyTokenFusion (patch_embed
    1-4 + block 1-4 + norm 1-4 + token fusion at 1/16), so pretrained MiT-B0
    weight loading works unchanged (key prefixes match).

    Decoder is a verbatim copy of LiteHRNetBodyTokenFusion's:
      - bilinearly upsample all 4 stage outputs (c1..c4) to the input
        spatial size (in HRNet branch0 was already at OS=1; for MiT all four
        stages are at OS≥4 so all four need upsampling — c1 by 4×)
      - concat along channels: (B, 32+64+160+256=512, H, W)
      - dual decoder:
            Conv1x1(512 → 32, bias=False)  +  BN  +  ReLU
          + DoubleConv(32, 32)             (BN-based, kernel=3 padding=1)
      - 1×1 heads → 3 seg / 1 height

    No hi-res stems, no progressive upsample, no SegFormer-style 1×1
    projections — everything off-encoder mirrors HRNet exactly. The DELTA
    vs the HRNet specialist run is therefore *only* the encoder.
    """

    is_late_fusion = True

    def __init__(self, dense_in_channels=128, token_in_channels=256, out_channels=4,
                 embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 qkv_bias=True,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 decoder_depth=1,
                 norm_layer=None):
        super().__init__()
        assert out_channels == 4
        assert len(embed_dims) == 4 and len(num_heads) == 4
        assert len(mlp_ratios) == 4 and len(depths) == 4 and len(sr_ratios) == 4
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # ----- MiT-B0 encoder (identical to V2 minus the stems) -----
        self.patch_embed1 = _MiTOverlapPatchEmbed(
            patch_size=7, stride=4, in_chans=dense_in_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        def _stage(idx, depth):
            return nn.ModuleList([_MiTBlock(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx],
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=0.,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[idx])
                for i in range(depth)])

        self.block1 = _stage(0, depths[0]); self.norm1 = norm_layer(embed_dims[0]); cur += depths[0]
        self.block2 = _stage(1, depths[1]); self.norm2 = norm_layer(embed_dims[1]); cur += depths[1]
        self.block3 = _stage(2, depths[2]); self.norm3 = norm_layer(embed_dims[2]); cur += depths[2]
        self.block4 = _stage(3, depths[3]); self.norm4 = norm_layer(embed_dims[3])

        # 1/16 token fusion (same DoubleConvLN as V2 — this is encoder-side glue,
        # not part of HRNet decoder; LiteHRNetBodyTokenFusion uses DoubleConvLN here too).
        self.token_fuse = DoubleConvLN(embed_dims[2] + token_in_channels, embed_dims[2])

        # ----- HRNet decoder (byte-for-byte copy of LiteHRNetBodyTokenFusion) -----
        total_in = sum(embed_dims)   # 32+64+160+256 = 512
        c = embed_dims[0]            # 32, matches HRNet using branch_channels[0]

        def _build_decoder():
            layers = [
                nn.Conv2d(total_in, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            for _ in range(decoder_depth):
                layers.append(DoubleConv(c, c))
            return nn.Sequential(*layers)

        self.seg_decoder = _build_decoder()
        self.height_decoder = _build_decoder()
        self.seg_head = nn.Conv2d(c, 3, kernel_size=1)
        self.height_head = nn.Conv2d(c, 1, kernel_size=1)

    def _run_stage(self, x_img, patch_embed, blocks, norm):
        x, H, W = patch_embed(x_img)
        for blk in blocks:
            x = blk(x, H, W)
        x = norm(x)
        B = x.shape[0]
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, dense, tokens):
        assert dense.dim() == 4 and tokens.dim() == 4
        H_in, W_in = dense.shape[-2], dense.shape[-1]
        assert H_in % 16 == 0 and W_in % 16 == 0, \
            f"Dense input must be multiple of 16, got {H_in}x{W_in}"
        expected_tok_h, expected_tok_w = H_in // 16, W_in // 16
        assert tokens.shape[-2] == expected_tok_h and tokens.shape[-1] == expected_tok_w, \
            f"Token spatial {tokens.shape[-2]}x{tokens.shape[-1]} != expected " \
            f"{expected_tok_h}x{expected_tok_w}"

        # MiT encoder
        c1 = self._run_stage(dense, self.patch_embed1, self.block1, self.norm1)  # OS=4
        c2 = self._run_stage(c1,    self.patch_embed2, self.block2, self.norm2)  # OS=8
        c3 = self._run_stage(c2,    self.patch_embed3, self.block3, self.norm3)  # OS=16
        c3 = self.token_fuse(torch.cat([c3, tokens], dim=1))                     # OS=16
        c4 = self._run_stage(c3, self.patch_embed4, self.block4, self.norm4)     # OS=32

        # HRNet decoder: bilinear ↑ all to OS=1, concat, dual decoder
        c1_up = F.interpolate(c1, size=(H_in, W_in), mode="bilinear", align_corners=False)
        c2_up = F.interpolate(c2, size=(H_in, W_in), mode="bilinear", align_corners=False)
        c3_up = F.interpolate(c3, size=(H_in, W_in), mode="bilinear", align_corners=False)
        c4_up = F.interpolate(c4, size=(H_in, W_in), mode="bilinear", align_corners=False)
        fused = torch.cat([c1_up, c2_up, c3_up, c4_up], dim=1)   # (B, 512, H, W)

        seg_logits = self.seg_head(self.seg_decoder(fused))
        height_logits = self.height_head(self.height_decoder(fused))
        return torch.cat([seg_logits, height_logits], dim=1)


class MitB0BodyUNetPlusPlus(nn.Module):
    """MiT-B0 (SegFormer) backbone + 1/16 token fusion + UNet++ NESTED decoder.

    Variant of MitB0BodyTokenFusion that swaps the single-path progressive
    OS=4→OS=2→OS=1 decoder for a UNet++ nested-skip decoder. The nested skip
    connections allow each output to draw context from every higher-resolution
    encoder feature (vs UNet's single skip per level), often improving boundary
    quality and reducing distribution-shift sensitivity (per Zhou et al. 2018).

    Encoder (unchanged from MitB0BodyTokenFusion):
      patch_embed1 stride=4 → c1 @ OS=4   (embed_dims[0]=32 ch)
      patch_embed2 stride=2 → c2 @ OS=8   (embed_dims[1]=64 ch)
      patch_embed3 stride=2 → c3 @ OS=16  (embed_dims[2]=160 ch, token-fused)
      patch_embed4 stride=2 → c4 @ OS=32  (embed_dims[3]=256 ch)

    UNet++ nested decoder (j = nested column, i = depth):
      Row 0 (OS=4):  X[0,0]=c1 → X[0,1] → X[0,2] → X[0,3]    (final OS=4 feature)
      Row 1 (OS=8):  X[1,0]=c2 → X[1,1] → X[1,2]
      Row 2 (OS=16): X[2,0]=c3 → X[2,1]
      Row 3 (OS=32): X[3,0]=c4

      X[i,j] = DoubleConvLN(concat(X[i,0..j-1], Up(X[i+1,j-1])))
      6 nested DoubleConvLN blocks total.

    OS=4 → OS=2 → OS=1 progressive with hi-res stem skips (same as parent):
      X[0,3] @ OS=4 (32ch) → bilinear×2 → concat stem_half (+64) → DoubleConvLN → OS=2 (dec_os2_dim)
                          → bilinear×2 → concat stem_full (+32) → DoubleConvLN → OS=1 (dec_os1_dim)

    Dual shallow seg + height decoders + 1×1 heads at OS=1 → (B, 4, H, W).
    """

    is_late_fusion = True

    def __init__(self, dense_in_channels=128, token_in_channels=256, out_channels=4,
                 embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 qkv_bias=True,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=None,
                 stem_full_dim=32,
                 stem_half_dim=64,
                 dec_os2_dim=128,
                 dec_os1_dim=64):
        super().__init__()
        assert out_channels == 4
        assert len(embed_dims) == 4 and len(num_heads) == 4
        assert len(mlp_ratios) == 4 and len(depths) == 4 and len(sr_ratios) == 4
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        e0, e1, e2, e3 = embed_dims  # (32, 64, 160, 256)

        # ----- High-resolution detail stems (parallel to MiT encoder) -----
        self.stem_full = DoubleConvLN(dense_in_channels, stem_full_dim)        # OS=1
        self.stem_half = StridedDoubleConvLN(stem_full_dim, stem_half_dim)     # OS=2

        # ----- Patch embeds (stride 4 / 2 / 2 / 2) -----
        self.patch_embed1 = _MiTOverlapPatchEmbed(
            patch_size=7, stride=4, in_chans=dense_in_channels, embed_dim=e0)
        self.patch_embed2 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=e0, embed_dim=e1)
        self.patch_embed3 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=e1, embed_dim=e2)
        self.patch_embed4 = _MiTOverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=e2, embed_dim=e3)

        # ----- Transformer blocks -----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        def _stage(idx, depth):
            blocks = nn.ModuleList([_MiTBlock(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx],
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=0.,
                drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[idx])
                for i in range(depth)])
            return blocks

        self.block1 = _stage(0, depths[0]); self.norm1 = norm_layer(e0)
        cur += depths[0]
        self.block2 = _stage(1, depths[1]); self.norm2 = norm_layer(e1)
        cur += depths[1]
        self.block3 = _stage(2, depths[2]); self.norm3 = norm_layer(e2)
        cur += depths[2]
        self.block4 = _stage(3, depths[3]); self.norm4 = norm_layer(e3)

        # ----- 1/16 token fusion (same pattern as MitB0BodyTokenFusion) -----
        self.token_fuse = DoubleConvLN(e2 + token_in_channels, e2)

        # ----- UNet++ nested decoder blocks -----
        # Row 2 (OS=16): X[2,1] = DoubleConvLN([X[2,0]=e2, Up(X[3,0]=e3)])
        self.x21 = DoubleConvLN(e2 + e3, e2)
        # Row 1 (OS=8):
        #   X[1,1] = DoubleConvLN([X[1,0]=e1, Up(X[2,0]=e2)])
        self.x11 = DoubleConvLN(e1 + e2, e1)
        #   X[1,2] = DoubleConvLN([X[1,0]=e1, X[1,1]=e1, Up(X[2,1]=e2)])
        self.x12 = DoubleConvLN(e1 * 2 + e2, e1)
        # Row 0 (OS=4):
        #   X[0,1] = DoubleConvLN([X[0,0]=e0, Up(X[1,0]=e1)])
        self.x01 = DoubleConvLN(e0 + e1, e0)
        #   X[0,2] = DoubleConvLN([X[0,0]=e0, X[0,1]=e0, Up(X[1,1]=e1)])
        self.x02 = DoubleConvLN(e0 * 2 + e1, e0)
        #   X[0,3] = DoubleConvLN([X[0,0]=e0, X[0,1]=e0, X[0,2]=e0, Up(X[1,2]=e1)])
        self.x03 = DoubleConvLN(e0 * 3 + e1, e0)

        # ----- Progressive OS=4 → OS=2 → OS=1 with hi-res stem skips -----
        self.conv_os2 = DoubleConvLN(e0 + stem_half_dim, dec_os2_dim)   # OS=2
        self.conv_os1 = DoubleConvLN(dec_os2_dim + stem_full_dim, dec_os1_dim)   # OS=1

        # ----- Dual shallow seg + height decoders + 1×1 heads at OS=1 -----
        self.seg_decoder = DoubleConvLN(dec_os1_dim, dec_os1_dim // 2)
        self.height_decoder = DoubleConvLN(dec_os1_dim, dec_os1_dim // 2)
        self.seg_head = nn.Conv2d(dec_os1_dim // 2, 3, kernel_size=1)
        self.height_head = nn.Conv2d(dec_os1_dim // 2, 1, kernel_size=1)

    def _run_stage(self, x_img, patch_embed, blocks, norm):
        x, H, W = patch_embed(x_img)
        for blk in blocks:
            x = blk(x, H, W)
        x = norm(x)
        B = x.shape[0]
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _up_to(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, dense, tokens):
        assert dense.dim() == 4 and tokens.dim() == 4
        H_in, W_in = dense.shape[-2], dense.shape[-1]
        assert H_in % 32 == 0 and W_in % 32 == 0, \
            f"Dense input must be multiple of 32 (UNet++ depth=4), got {H_in}x{W_in}"
        expected_tok_h, expected_tok_w = H_in // 16, W_in // 16
        assert tokens.shape[-2] == expected_tok_h and tokens.shape[-1] == expected_tok_w, \
            f"Token spatial {tokens.shape[-2]}x{tokens.shape[-1]} != expected " \
            f"{expected_tok_h}x{expected_tok_w} for dense {H_in}x{W_in}"

        # ----- High-res detail skip features -----
        s_full = self.stem_full(dense)         # OS=1
        s_half = self.stem_half(s_full)        # OS=2

        # ----- MiT encoder stages 1-3 -----
        c1 = self._run_stage(dense, self.patch_embed1, self.block1, self.norm1)  # 1/4
        c2 = self._run_stage(c1,    self.patch_embed2, self.block2, self.norm2)  # 1/8
        c3 = self._run_stage(c2,    self.patch_embed3, self.block3, self.norm3)  # 1/16

        # ----- Token fusion at 1/16 -----
        c3 = self.token_fuse(torch.cat([c3, tokens], dim=1))

        # ----- Stage 4 -----
        c4 = self._run_stage(c3, self.patch_embed4, self.block4, self.norm4)     # 1/32

        # ----- UNet++ nested decoder -----
        s1, s2, s3 = c1.shape[-2:], c2.shape[-2:], c3.shape[-2:]
        # Row 2: X[2,1]
        x21 = self.x21(torch.cat([c3, self._up_to(c4, s3)], dim=1))
        # Row 1: X[1,1] and X[1,2]
        x11 = self.x11(torch.cat([c2, self._up_to(c3, s2)], dim=1))
        x12 = self.x12(torch.cat([c2, x11, self._up_to(x21, s2)], dim=1))
        # Row 0: X[0,1], X[0,2], X[0,3]
        x01 = self.x01(torch.cat([c1, self._up_to(c2, s1)], dim=1))
        x02 = self.x02(torch.cat([c1, x01, self._up_to(x11, s1)], dim=1))
        x03 = self.x03(torch.cat([c1, x01, x02, self._up_to(x12, s1)], dim=1))  # OS=4, e0 ch

        # ----- Progressive OS=4 → OS=2 → OS=1 with hi-res stem skips -----
        x_os2 = self.conv_os2(torch.cat([self._up_to(x03, s_half.shape[-2:]), s_half], dim=1))
        x_os1 = self.conv_os1(torch.cat([self._up_to(x_os2, (H_in, W_in)), s_full], dim=1))

        seg_logits = self.seg_head(self.seg_decoder(x_os1))
        height_logits = self.height_head(self.height_decoder(x_os1))
        return torch.cat([seg_logits, height_logits], dim=1)


class AdapterFusionMitB0UNetPlusPlus(nn.Module):
    """LN per-source adapters + MitB0BodyUNetPlusPlus.

    Mirrors AdapterFusionMitB0HRNetDecoder exactly, only the inner body swaps
    to MitB0BodyUNetPlusPlus (UNet++ nested decoder). `pretrained_mit_b0` loads
    the NVIDIA ImageNet checkpoint into the MiT encoder; decoder stays random.
    """

    is_late_fusion = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 drop_path_rate=0.1,
                 pretrained_mit_b0=None):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        fused_dense_in = adapter_out * len(self.dense_channels)
        fused_token_in = adapter_out * len(self.token_channels)

        self.body = MitB0BodyUNetPlusPlus(
            dense_in_channels=fused_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
        )
        if pretrained_mit_b0:
            load_mit_b0_pretrained(self.body, pretrained_mit_b0)

    def forward(self, x_dense, tokens):
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        fused_dense = torch.cat(dense_parts, dim=1)

        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)

        return self.body(fused_dense, fused_tokens)


class AdapterFusionMitB0HRNetDecoder(nn.Module):
    """V3 wrapper: LN per-source adapters + MitB0BodyHRNetDecoder.

    Mirrors AdapterFusionMitB0TokenFusion exactly except the inner body
    swaps to MitB0BodyHRNetDecoder. `pretrained_mit_b0` still loads the
    NVIDIA ImageNet checkpoint into the encoder; decoder stays random.
    """

    is_late_fusion = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 drop_path_rate=0.1,
                 decoder_depth=1,
                 pretrained_mit_b0=None):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        fused_dense_in = adapter_out * len(self.dense_channels)
        fused_token_in = adapter_out * len(self.token_channels)

        self.body = MitB0BodyHRNetDecoder(
            dense_in_channels=fused_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
            decoder_depth=decoder_depth,
        )
        if pretrained_mit_b0:
            load_mit_b0_pretrained(self.body, pretrained_mit_b0)

    def forward(self, x_dense, tokens):
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        fused_dense = torch.cat(dense_parts, dim=1)

        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)

        return self.body(fused_dense, fused_tokens)


# ---------------------------------------------------------------------------
# V4: HRNet-pipeline body with MiT blocks replacing the ShuffleBlock per-branch
# processor. Everything else (stem, transitions, token_fuse @ 1/8,
# cross-resolution fusion, decoder, heads) stays byte-for-byte identical to
# LiteHRNetBodyTokenFusion. Only the per-branch conv unit changes.
# ---------------------------------------------------------------------------


class _MiTBranchStage(nn.Module):
    """Stack of N MiT blocks operating on a single resolution branch.

    Drop-in replacement for `nn.Sequential(*[ShuffleBlock(c) for _ in range(N)])`:
    accepts and returns (B, C, H, W). Internally reshapes once into (B, N, C),
    runs all MiT blocks (each gets H, W for its DWConv-MLP and SR-attention),
    then reshapes back.
    """

    def __init__(self, dim, num_blocks, num_heads, sr_ratio,
                 mlp_ratio=4.0, drop_path=0.0,
                 qkv_bias=True, norm_layer=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if isinstance(drop_path, (list, tuple)):
            assert len(drop_path) == num_blocks
            dpr = list(drop_path)
        else:
            dpr = [float(drop_path)] * num_blocks
        self.blocks = nn.ModuleList([
            _MiTBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, drop_path=dpr[i],
                      norm_layer=norm_layer, sr_ratio=sr_ratio)
            for i in range(num_blocks)
        ])

    def forward(self, x):
        # (B, C, H, W) → (B, N, C) → blocks → (B, C, H, W)
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            seq = blk(seq, H, W)
        return seq.transpose(1, 2).reshape(B, C, H, W).contiguous()


class HRStageMiT(nn.Module):
    """Drop-in replacement for `HRStage` with MiT per-branch processing.

    Cross-resolution fusion (`_build_fusion` + post-fusion ReLU) is shared
    verbatim with HRStage; only the per-branch processor swaps from
    `nn.Sequential(*[ShuffleBlock(c)])` to `_MiTBranchStage`.
    """

    def __init__(self, channels, num_blocks=3,
                 num_heads=(5, 5, 5, 6),
                 sr_ratios=(8, 4, 2, 1),
                 mlp_ratio=4.0,
                 drop_path_per_branch=None,
                 qkv_bias=True,
                 norm_layer=None):
        super().__init__()
        self.num_branches = len(channels)
        self.channels = channels
        assert len(num_heads) == self.num_branches
        assert len(sr_ratios) == self.num_branches
        if drop_path_per_branch is None:
            drop_path_per_branch = [[0.0] * num_blocks for _ in range(self.num_branches)]

        self.branches = nn.ModuleList([
            _MiTBranchStage(
                dim=channels[i], num_blocks=num_blocks,
                num_heads=num_heads[i], sr_ratio=sr_ratios[i],
                mlp_ratio=mlp_ratio, drop_path=drop_path_per_branch[i],
                qkv_bias=qkv_bias, norm_layer=norm_layer)
            for i in range(self.num_branches)
        ])
        # Reuse HRStage's fusion builder verbatim
        self.fusion = HRStage._build_fusion(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):
        # 1) Per-branch processing (MiT blocks)
        branches = [self.branches[i](branches[i]) for i in range(self.num_branches)]
        # 2) Multi-resolution fusion (identical to HRStage)
        out = []
        for i in range(self.num_branches):
            agg = sum(self.fusion[i][j](branches[j]) for j in range(self.num_branches))
            out.append(self.relu(agg))
        return out


class LiteHRNetMiTBodyTokenFusion(nn.Module):
    """HRNet 4-branch body with per-branch MiT blocks instead of ShuffleBlocks.

    Strict ablation against LiteHRNetBodyTokenFusion: every non-encoder piece
    (stem, transitions, token_fuse @ 1/8, cross-resolution fusion in HRStage,
    HRNet decoder, heads) is byte-for-byte identical. The only change is
    `nn.Sequential(*[ShuffleBlock(c) for _ in range(blocks_per_branch)])`
    → `_MiTBranchStage(c, blocks_per_branch, num_heads, sr_ratio)` inside
    `HRStageMiT`.

    SR ratios are scaled to the branch resolution so each branch's attention
    has the same K,V token count (~24² at H=192 training). Memory-efficient
    sdpa attention (Flash backend) is required for branch 0 (192² queries).
    """

    is_late_fusion = True

    def __init__(self, dense_in_channels=128, token_in_channels=256, out_channels=4,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1,
                 num_heads=(5, 5, 5, 6),
                 sr_ratios=(8, 4, 2, 1),
                 mlp_ratio=4.0,
                 drop_path_rate=0.1,
                 qkv_bias=True,
                 use_gradient_checkpointing=True):
        super().__init__()
        assert out_channels == 4
        assert len(branch_channels) == 4
        assert all(c % h == 0 for c, h in zip(branch_channels, num_heads)), \
            f"branch channels {branch_channels} must be divisible by num_heads {num_heads}"
        # Branch 0 at OS=1 has 192² = 36864 tokens; with blocks_per_branch=3 ×
        # num_stages=4 = 12 MiT blocks on branch 0 alone, MLP activations
        # (B, 36864, 160) × 12 = ~25GB even at bs=4/gpu. Gradient checkpointing
        # each HR stage cuts forward activation memory ~4× at the cost of one
        # extra forward in backward (≈ 1.5× wall-clock per step).
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ----- Stem + transitions (IDENTICAL to LiteHRNetBodyTokenFusion) -----
        self.stem = DoubleConv(dense_in_channels, branch_channels[0])
        self.transitions = nn.ModuleList()
        for k in range(1, 4):
            self.transitions.append(nn.Sequential(
                nn.Conv2d(branch_channels[k - 1], branch_channels[k],
                          3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels[k]),
                nn.ReLU(inplace=True),
            ))

        # ----- Token fusion at 1/8 (IDENTICAL to LiteHRNetBodyTokenFusion) -----
        self.token_fuse = DoubleConvLN(branch_channels[3] + token_in_channels,
                                       branch_channels[3])

        # ----- Drop-path schedule: linspace across all blocks, in stage→branch→block order -----
        total_blocks = num_stages * len(branch_channels) * blocks_per_branch
        dpr_flat = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        idx = 0

        # ----- 4 HR stages with MiT per-branch processing -----
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            drop_path_per_branch = []
            for _ in range(len(branch_channels)):
                drop_path_per_branch.append(dpr_flat[idx:idx + blocks_per_branch])
                idx += blocks_per_branch
            self.stages.append(HRStageMiT(
                channels=list(branch_channels),
                num_blocks=blocks_per_branch,
                num_heads=num_heads,
                sr_ratios=sr_ratios,
                mlp_ratio=mlp_ratio,
                drop_path_per_branch=drop_path_per_branch,
                qkv_bias=qkv_bias,
            ))

        # ----- HRNet decoder (IDENTICAL to LiteHRNetBodyTokenFusion) -----
        total_in = sum(branch_channels)
        c = branch_channels[0]

        def _build_decoder():
            layers = [
                nn.Conv2d(total_in, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            for _ in range(decoder_depth):
                layers.append(DoubleConv(c, c))
            return nn.Sequential(*layers)

        self.seg_decoder = _build_decoder()
        self.height_decoder = _build_decoder()
        self.seg_head = nn.Conv2d(c, 3, kernel_size=1)
        self.height_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, dense, tokens):
        # IDENTICAL to LiteHRNetBodyTokenFusion.forward — only the
        # branch-block ops inside self.stages differ (MiT vs Shuffle).
        assert dense.dim() == 4 and tokens.dim() == 4
        assert dense.shape[-1] % 16 == 0 and dense.shape[-2] % 16 == 0, \
            f"Dense input must be multiple of 16, got {dense.shape[-2]}x{dense.shape[-1]}"
        expected_tok = dense.shape[-1] // 16
        assert tokens.shape[-1] == expected_tok and tokens.shape[-2] == expected_tok, \
            f"Token spatial {tokens.shape[-2]}x{tokens.shape[-1]} != expected {expected_tok}"

        # Dense branches at scales (1, 1/2, 1/4, 1/8)
        b0 = self.stem(dense)
        b1 = self.transitions[0](b0)
        b2 = self.transitions[1](b1)
        b3_dense = self.transitions[2](b2)

        # Fuse tokens (1/16) into b3 (1/8) — bilinear ×2 + concat + DoubleConvLN reduce
        tokens_at_1_8 = F.interpolate(tokens, scale_factor=2.0,
                                      mode="bilinear", align_corners=False)
        b3 = self.token_fuse(torch.cat([b3_dense, tokens_at_1_8], dim=1))

        branches = [b0, b1, b2, b3]

        # 4 HR stages on 4 branches (cross-resolution fusion identical to HRNet).
        # Gradient checkpointing each stage saves ~75% of forward activation
        # memory (each stage's per-block MLP/attention intermediates are not
        # retained for backward — they're recomputed by re-running forward).
        for stage in self.stages:
            if self.use_gradient_checkpointing and self.training:
                # checkpoint expects positional-arg input/output but our stage
                # forward takes a list. Wrap input via tuple and unpack inside.
                branches = torch.utils.checkpoint.checkpoint(
                    stage, branches, use_reentrant=False)
            else:
                branches = stage(branches)

        # Upsample all to OS=1, concat
        h, w = branches[0].shape[2], branches[0].shape[3]
        upsampled = [branches[0]]
        for i in range(1, len(branches)):
            upsampled.append(F.interpolate(branches[i], size=(h, w),
                                           mode="bilinear", align_corners=False))
        fused_features = torch.cat(upsampled, dim=1)

        seg_features = self.seg_decoder(fused_features)
        height_features = self.height_decoder(fused_features)
        seg_logits = self.seg_head(seg_features)
        height_logits = self.height_head(height_features)
        return torch.cat([seg_logits, height_logits], dim=1)


class AdapterFusionLiteHRNetMiTTokenFusion(nn.Module):
    """V4 wrapper: per-source LN adapters + LiteHRNetMiTBodyTokenFusion.

    Identical adapter pattern to AdapterFusionLiteHRNetTokenFusion (per-source
    SourceAdapterStemLN, concat into 128ch dense + 256ch token), differing
    only in the body's per-branch processor (MiT block vs ShuffleBlock).
    """

    is_late_fusion = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1,
                 num_heads=(5, 5, 5, 6),
                 sr_ratios=(8, 4, 2, 1),
                 mlp_ratio=4.0,
                 drop_path_rate=0.1):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        fused_dense_in = adapter_out * len(self.dense_channels)
        fused_token_in = adapter_out * len(self.token_channels)

        self.body = LiteHRNetMiTBodyTokenFusion(
            dense_in_channels=fused_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            branch_channels=branch_channels,
            num_stages=num_stages,
            blocks_per_branch=blocks_per_branch,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x_dense, tokens):
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        fused_dense = torch.cat(dense_parts, dim=1)

        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)

        return self.body(fused_dense, fused_tokens)


# ---------------------------------------------------------------------------
# V7 / V8: same HRNet pipeline as V4 — only the per-branch ShuffleBlock is
# replaced. V7 uses ConvNeXt blocks (DwConv 7x7 + LN + 4x MLP + LayerScale),
# V8 uses EfficientNet MBConv blocks (expand 1x1 + DwConv + SE + project).
# Everything else (stem, transitions, token_fuse, cross-resolution fusion,
# decoder, heads) is byte-for-byte identical to LiteHRNetBodyTokenFusion.
# Reuses pre-existing `ConvNeXtBlock` (~line 2698 in this file) and a new
# `_MBConvBlock` defined below.
# ---------------------------------------------------------------------------


class _MBConvBlock(nn.Module):
    """EfficientNet-B0 style MBConv block (stride=1 only, residual when
    in/out channels match).

    Pipeline: 1x1 expand → BN+SiLU → DwConv k×k → BN+SiLU → SE → 1x1 project → BN
    + identity residual + stochastic depth (drop_path).

    Defaults match the most common MBConv6_k3 block in EfficientNet-B0:
      expand_ratio = 6, kernel_size = 3, se_ratio = 0.25
    """

    def __init__(self, dim, expand_ratio=6, kernel_size=3, se_ratio=0.25,
                 drop_path=0.0):
        super().__init__()
        hidden = dim * expand_ratio
        # 1x1 expand (skipped when expand_ratio == 1, MBConv1 style)
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(dim, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
        else:
            self.expand = nn.Identity()
        # Depthwise conv
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size,
                      padding=kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        # Squeeze-Excitation
        se_hidden = max(1, int(dim * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, se_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_hidden, hidden, 1),
            nn.Sigmoid(),
        )
        # 1x1 project (no activation; followed by residual)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        # Stochastic depth (drop_path equivalent for EfficientNet)
        from torchvision.ops import StochasticDepth
        self.drop_path = StochasticDepth(drop_path, "row") if drop_path > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.dwconv(out)
        out = out * self.se(out)
        out = self.project(out)
        return identity + self.drop_path(out)


class _ConvNeXtBranchStage(nn.Module):
    """Stack of N ConvNeXt blocks operating on a single HRNet resolution
    branch. Drop-in replacement for `nn.Sequential(*[ShuffleBlock(c) for _])`.
    """

    def __init__(self, dim, num_blocks, drop_path=0.0, layer_scale_init=1.0):
        super().__init__()
        if isinstance(drop_path, (list, tuple)):
            assert len(drop_path) == num_blocks
            dpr = list(drop_path)
        else:
            dpr = [float(drop_path)] * num_blocks
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=dim, drop_path=dpr[i],
                          layer_scale_init=layer_scale_init)
            for i in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


class _MBConvBranchStage(nn.Module):
    """Stack of N MBConv blocks operating on a single HRNet resolution
    branch. Drop-in replacement for `nn.Sequential(*[ShuffleBlock(c) for _])`.
    """

    def __init__(self, dim, num_blocks, expand_ratio=6, kernel_size=3,
                 se_ratio=0.25, drop_path=0.0):
        super().__init__()
        if isinstance(drop_path, (list, tuple)):
            assert len(drop_path) == num_blocks
            dpr = list(drop_path)
        else:
            dpr = [float(drop_path)] * num_blocks
        self.blocks = nn.Sequential(*[
            _MBConvBlock(dim=dim, expand_ratio=expand_ratio,
                         kernel_size=kernel_size, se_ratio=se_ratio,
                         drop_path=dpr[i])
            for i in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


class HRStageConvNeXt(nn.Module):
    """HRStage clone with per-branch ConvNeXt block processor.
    Cross-resolution fusion machinery is identical to `HRStage`.
    """

    def __init__(self, channels, num_blocks=3,
                 drop_path_per_branch=None,
                 layer_scale_init=1.0):
        super().__init__()
        self.num_branches = len(channels)
        self.channels = channels
        if drop_path_per_branch is None:
            drop_path_per_branch = [[0.0] * num_blocks for _ in range(self.num_branches)]

        self.branches = nn.ModuleList([
            _ConvNeXtBranchStage(
                dim=channels[i], num_blocks=num_blocks,
                drop_path=drop_path_per_branch[i],
                layer_scale_init=layer_scale_init)
            for i in range(self.num_branches)
        ])
        self.fusion = HRStage._build_fusion(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):
        branches = [self.branches[i](branches[i]) for i in range(self.num_branches)]
        out = []
        for i in range(self.num_branches):
            agg = sum(self.fusion[i][j](branches[j]) for j in range(self.num_branches))
            out.append(self.relu(agg))
        return out


class HRStageMBConv(nn.Module):
    """HRStage clone with per-branch MBConv block processor.
    Cross-resolution fusion machinery is identical to `HRStage`.
    """

    def __init__(self, channels, num_blocks=3,
                 expand_ratio=6, kernel_size=3, se_ratio=0.25,
                 drop_path_per_branch=None):
        super().__init__()
        self.num_branches = len(channels)
        self.channels = channels
        if drop_path_per_branch is None:
            drop_path_per_branch = [[0.0] * num_blocks for _ in range(self.num_branches)]

        self.branches = nn.ModuleList([
            _MBConvBranchStage(
                dim=channels[i], num_blocks=num_blocks,
                expand_ratio=expand_ratio, kernel_size=kernel_size,
                se_ratio=se_ratio,
                drop_path=drop_path_per_branch[i])
            for i in range(self.num_branches)
        ])
        self.fusion = HRStage._build_fusion(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):
        branches = [self.branches[i](branches[i]) for i in range(self.num_branches)]
        out = []
        for i in range(self.num_branches):
            agg = sum(self.fusion[i][j](branches[j]) for j in range(self.num_branches))
            out.append(self.relu(agg))
        return out


def _build_hrnet_body_variant(stage_cls, stage_kwargs, dense_in_channels,
                              token_in_channels, branch_channels, num_stages,
                              blocks_per_branch, decoder_depth, drop_path_rate):
    """Shared body skeleton for V4/V7/V8 — stem, transitions, token_fuse @ 1/8,
    HR stages (with the supplied per-branch stage class), HRNet decoder.

    Returns a tuple of nn.Module objects that the wrapper class registers.
    """
    raise NotImplementedError("Use class-level construction (see body classes)")


class LiteHRNetConvNeXtBodyTokenFusion(nn.Module):
    """V7-A body: HRNet 4-branch with per-branch ConvNeXt blocks.

    Strict ablation against LiteHRNetBodyTokenFusion: every non-encoder piece
    identical. Only the ShuffleBlock per-branch processor is swapped for
    ConvNeXtBlock (DwConv 7x7 + LN + 4x MLP + LayerScale=1.0).
    """

    is_late_fusion = True

    def __init__(self, dense_in_channels=128, token_in_channels=256, out_channels=4,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1,
                 drop_path_rate=0.1,
                 layer_scale_init=1.0):
        super().__init__()
        assert out_channels == 4
        assert len(branch_channels) == 4

        # Stem + transitions (IDENTICAL to LiteHRNetBodyTokenFusion)
        self.stem = DoubleConv(dense_in_channels, branch_channels[0])
        self.transitions = nn.ModuleList()
        for k in range(1, 4):
            self.transitions.append(nn.Sequential(
                nn.Conv2d(branch_channels[k - 1], branch_channels[k],
                          3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels[k]),
                nn.ReLU(inplace=True),
            ))

        # Token fusion at 1/8
        self.token_fuse = DoubleConvLN(branch_channels[3] + token_in_channels,
                                       branch_channels[3])

        # Drop-path schedule (linspace over all blocks, stage→branch→block order)
        total_blocks = num_stages * len(branch_channels) * blocks_per_branch
        dpr_flat = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        idx = 0

        # HR stages with ConvNeXt per-branch processing
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            drop_path_per_branch = []
            for _ in range(len(branch_channels)):
                drop_path_per_branch.append(dpr_flat[idx:idx + blocks_per_branch])
                idx += blocks_per_branch
            self.stages.append(HRStageConvNeXt(
                channels=list(branch_channels),
                num_blocks=blocks_per_branch,
                drop_path_per_branch=drop_path_per_branch,
                layer_scale_init=layer_scale_init,
            ))

        # HRNet decoder
        total_in = sum(branch_channels)
        c = branch_channels[0]

        def _build_decoder():
            layers = [
                nn.Conv2d(total_in, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            for _ in range(decoder_depth):
                layers.append(DoubleConv(c, c))
            return nn.Sequential(*layers)

        self.seg_decoder = _build_decoder()
        self.height_decoder = _build_decoder()
        self.seg_head = nn.Conv2d(c, 3, kernel_size=1)
        self.height_head = nn.Conv2d(c, 1, kernel_size=1)

        # Gradient checkpointing per HR stage (huge activation memory savings;
        # ConvNeXt MLP intermediate at branch 0 = bs × 36864 × 160 × 4B is large)
        self.use_gradient_checkpointing = True

    def forward(self, dense, tokens):
        assert dense.dim() == 4 and tokens.dim() == 4
        assert dense.shape[-1] % 16 == 0 and dense.shape[-2] % 16 == 0
        expected_tok = dense.shape[-1] // 16
        assert tokens.shape[-1] == expected_tok and tokens.shape[-2] == expected_tok

        b0 = self.stem(dense)
        b1 = self.transitions[0](b0)
        b2 = self.transitions[1](b1)
        b3_dense = self.transitions[2](b2)
        tokens_at_1_8 = F.interpolate(tokens, scale_factor=2.0,
                                      mode="bilinear", align_corners=False)
        b3 = self.token_fuse(torch.cat([b3_dense, tokens_at_1_8], dim=1))

        branches = [b0, b1, b2, b3]
        for stage in self.stages:
            if self.use_gradient_checkpointing and self.training:
                branches = torch.utils.checkpoint.checkpoint(
                    stage, branches, use_reentrant=False)
            else:
                branches = stage(branches)

        h, w = branches[0].shape[2], branches[0].shape[3]
        upsampled = [branches[0]]
        for i in range(1, len(branches)):
            upsampled.append(F.interpolate(branches[i], size=(h, w),
                                           mode="bilinear", align_corners=False))
        fused_features = torch.cat(upsampled, dim=1)
        # Two-step decoder/head (matches LiteHRNetBodyTokenFusion byte-for-byte).
        seg_features = self.seg_decoder(fused_features)
        height_features = self.height_decoder(fused_features)
        seg_logits = self.seg_head(seg_features)
        height_logits = self.height_head(height_features)
        return torch.cat([seg_logits, height_logits], dim=1)


class LiteHRNetMBConvBodyTokenFusion(nn.Module):
    """V8-A body: HRNet 4-branch with per-branch MBConv (EfficientNet) blocks.

    Strict ablation against LiteHRNetBodyTokenFusion. Only the ShuffleBlock
    per-branch processor is swapped for MBConv (expand 1x1 + DwConv + SE +
    project, expand_ratio=6, kernel=3, se=0.25 — most common EfficientNet-B0
    block recipe).
    """

    is_late_fusion = True

    def __init__(self, dense_in_channels=128, token_in_channels=256, out_channels=4,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1,
                 drop_path_rate=0.1,
                 expand_ratio=6, kernel_size=3, se_ratio=0.25):
        super().__init__()
        assert out_channels == 4
        assert len(branch_channels) == 4

        # Stem + transitions (IDENTICAL to LiteHRNetBodyTokenFusion)
        self.stem = DoubleConv(dense_in_channels, branch_channels[0])
        self.transitions = nn.ModuleList()
        for k in range(1, 4):
            self.transitions.append(nn.Sequential(
                nn.Conv2d(branch_channels[k - 1], branch_channels[k],
                          3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels[k]),
                nn.ReLU(inplace=True),
            ))

        self.token_fuse = DoubleConvLN(branch_channels[3] + token_in_channels,
                                       branch_channels[3])

        total_blocks = num_stages * len(branch_channels) * blocks_per_branch
        dpr_flat = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        idx = 0

        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            drop_path_per_branch = []
            for _ in range(len(branch_channels)):
                drop_path_per_branch.append(dpr_flat[idx:idx + blocks_per_branch])
                idx += blocks_per_branch
            self.stages.append(HRStageMBConv(
                channels=list(branch_channels),
                num_blocks=blocks_per_branch,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                se_ratio=se_ratio,
                drop_path_per_branch=drop_path_per_branch,
            ))

        total_in = sum(branch_channels)
        c = branch_channels[0]

        def _build_decoder():
            layers = [
                nn.Conv2d(total_in, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            for _ in range(decoder_depth):
                layers.append(DoubleConv(c, c))
            return nn.Sequential(*layers)

        self.seg_decoder = _build_decoder()
        self.height_decoder = _build_decoder()
        self.seg_head = nn.Conv2d(c, 3, kernel_size=1)
        self.height_head = nn.Conv2d(c, 1, kernel_size=1)

        # Gradient checkpointing per HR stage — MBConv expand×6 makes branch 0
        # activations heavy (bs × 240 × 192² × 4B at OS=1).
        self.use_gradient_checkpointing = True

    def forward(self, dense, tokens):
        assert dense.dim() == 4 and tokens.dim() == 4
        assert dense.shape[-1] % 16 == 0 and dense.shape[-2] % 16 == 0
        expected_tok = dense.shape[-1] // 16
        assert tokens.shape[-1] == expected_tok and tokens.shape[-2] == expected_tok

        b0 = self.stem(dense)
        b1 = self.transitions[0](b0)
        b2 = self.transitions[1](b1)
        b3_dense = self.transitions[2](b2)
        tokens_at_1_8 = F.interpolate(tokens, scale_factor=2.0,
                                      mode="bilinear", align_corners=False)
        b3 = self.token_fuse(torch.cat([b3_dense, tokens_at_1_8], dim=1))

        branches = [b0, b1, b2, b3]
        for stage in self.stages:
            if self.use_gradient_checkpointing and self.training:
                branches = torch.utils.checkpoint.checkpoint(
                    stage, branches, use_reentrant=False)
            else:
                branches = stage(branches)

        h, w = branches[0].shape[2], branches[0].shape[3]
        upsampled = [branches[0]]
        for i in range(1, len(branches)):
            upsampled.append(F.interpolate(branches[i], size=(h, w),
                                           mode="bilinear", align_corners=False))
        fused_features = torch.cat(upsampled, dim=1)
        # Two-step decoder/head (matches LiteHRNetBodyTokenFusion byte-for-byte).
        seg_features = self.seg_decoder(fused_features)
        height_features = self.height_decoder(fused_features)
        seg_logits = self.seg_head(seg_features)
        height_logits = self.height_head(height_features)
        return torch.cat([seg_logits, height_logits], dim=1)


class AdapterFusionLiteHRNetConvNeXtTokenFusion(nn.Module):
    """V7-A wrapper: per-source LN adapters + LiteHRNetConvNeXtBodyTokenFusion."""

    is_late_fusion = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1,
                 drop_path_rate=0.1,
                 layer_scale_init=1.0):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        self.body = LiteHRNetConvNeXtBodyTokenFusion(
            dense_in_channels=adapter_out * len(self.dense_channels),
            token_in_channels=adapter_out * len(self.token_channels),
            out_channels=out_channels,
            branch_channels=branch_channels,
            num_stages=num_stages,
            blocks_per_branch=blocks_per_branch,
            decoder_depth=decoder_depth,
            drop_path_rate=drop_path_rate,
            layer_scale_init=layer_scale_init,
        )

    def forward(self, x_dense, tokens):
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        fused_dense = torch.cat(dense_parts, dim=1)
        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)
        return self.body(fused_dense, fused_tokens)


class AdapterFusionLiteHRNetMBConvTokenFusion(nn.Module):
    """V8-A wrapper: per-source LN adapters + LiteHRNetMBConvBodyTokenFusion."""

    is_late_fusion = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 branch_channels=(40, 80, 160, 192),
                 num_stages=4, blocks_per_branch=3, decoder_depth=1,
                 drop_path_rate=0.1,
                 expand_ratio=6, kernel_size=3, se_ratio=0.25):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        self.body = LiteHRNetMBConvBodyTokenFusion(
            dense_in_channels=adapter_out * len(self.dense_channels),
            token_in_channels=adapter_out * len(self.token_channels),
            out_channels=out_channels,
            branch_channels=branch_channels,
            num_stages=num_stages,
            blocks_per_branch=blocks_per_branch,
            decoder_depth=decoder_depth,
            drop_path_rate=drop_path_rate,
            expand_ratio=expand_ratio,
            kernel_size=kernel_size,
            se_ratio=se_ratio,
        )

    def forward(self, x_dense, tokens):
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        fused_dense = torch.cat(dense_parts, dim=1)
        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)
        return self.body(fused_dense, fused_tokens)


class AdapterFusionDualOnlyUNet(nn.Module):
    """Ablation 2x2: adapter stem + dual decoder WITHOUT cascade injection.

    Compare to AdapterFusionCascadeUNet to isolate cascade's contribution
    once adapters are in place.
    """

    def __init__(self, in_channels, out_channels=4, source_channels=None, adapter_out=64):
        super().__init__()
        assert out_channels == 4
        if source_channels is None:
            assert in_channels == 192
            source_channels = (64, 128)
        else:
            source_channels = tuple(source_channels)
            assert sum(source_channels) == in_channels

        self.source_channels = source_channels
        self.source_offsets = [0]
        for c in source_channels:
            self.source_offsets.append(self.source_offsets[-1] + c)

        self.adapters = nn.ModuleList([
            SourceAdapterStem(c, adapter_out) for c in source_channels
        ])

        fused_in = adapter_out * len(source_channels)
        self.body = DualDecoderOnlyUNet(in_channels=fused_in, out_channels=out_channels)

    def forward(self, x):
        parts = []
        for i, adapter in enumerate(self.adapters):
            start, end = self.source_offsets[i], self.source_offsets[i + 1]
            parts.append(adapter(x[:, start:end]))
        fused = torch.cat(parts, dim=1)
        return self.body(fused)


class VegSpecialistDualUNet(nn.Module):
    """Vegetation-only dual-decoder UNet (single-class specialist).

    Shared encoder + two parallel decoders:
      - seg decoder: outputs 1 channel (vegetation probability logit)
      - height decoder: outputs 1 channel (vegetation height logit, normalized)
    Height decoder's FINAL DoubleConv receives sigmoid(seg_logits).detach()
    concatenated to features — gives height head a "veg presence" signal
    without polluting seg gradients.

    Output: (B, 2, H, W) = [veg_seg_logit, veg_height_logit].
    Activate (in GeoFMNet.activate): seg → sigmoid, height → softplus.
    """

    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        assert out_channels == 2

        # Shared encoder (same as DualDecoderOnlyUNet for consistency)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Seg decoder (veg only, 1 output channel)
        self.s_up1 = UpsampleBlock(256, 128); self.s_conv1 = DoubleConv(256, 128)
        self.s_up2 = UpsampleBlock(128, 64);  self.s_conv2 = DoubleConv(128, 64)
        self.s_up3 = UpsampleBlock(64, 32);   self.s_conv3 = DoubleConv(64, 32)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

        # Height decoder (veg height), with seg-gate injection at final stage
        # Final DoubleConv takes height_feat (32 ch) + seg_gate (1 ch) = 33 ch
        self.h_up1 = UpsampleBlock(256, 128); self.h_conv1 = DoubleConv(256, 128)
        self.h_up2 = UpsampleBlock(128, 64);  self.h_conv2 = DoubleConv(128, 64)
        self.h_up3 = UpsampleBlock(64, 32)
        self.h_conv3 = DoubleConv(33, 32)
        self.height_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Shared encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Seg decoder (vegetation probability)
        s = self.s_up1(x4)
        s = self.s_conv1(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_conv2(torch.cat([x2, s], dim=1))
        s = self.s_up3(s)
        s = self.s_conv3(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)  # (B, 1, H, W)

        # Height decoder, gated by seg confidence at final stage
        h = self.h_up1(x4)
        h = self.h_conv1(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_conv2(torch.cat([x2, h], dim=1))
        h = self.h_up3(h)                                       # (B, 32, H, W)
        seg_gate = torch.sigmoid(seg_logits.detach())            # (B, 1, H, W)
        h = self.h_conv3(torch.cat([h, seg_gate], dim=1))        # (B, 32, H, W)
        height_logits = self.height_head(h)                      # (B, 1, H, W)

        return torch.cat([seg_logits, height_logits], dim=1)     # (B, 2, H, W)


class AdapterFusionVegSpecialist(nn.Module):
    """SourceAdapterStem fan-in + VegSpecialistDualUNet body."""

    def __init__(self, in_channels, out_channels=2, source_channels=None, adapter_out=64):
        super().__init__()
        assert out_channels == 2
        if source_channels is None:
            assert in_channels == 192
            source_channels = (64, 128)
        else:
            source_channels = tuple(source_channels)
            assert sum(source_channels) == in_channels

        self.source_channels = source_channels
        self.source_offsets = [0]
        for c in source_channels:
            self.source_offsets.append(self.source_offsets[-1] + c)

        self.adapters = nn.ModuleList([
            SourceAdapterStem(c, adapter_out) for c in source_channels
        ])

        fused_in = adapter_out * len(source_channels)
        self.body = VegSpecialistDualUNet(in_channels=fused_in, out_channels=2)

    def forward(self, x):
        parts = []
        for i, adapter in enumerate(self.adapters):
            start, end = self.source_offsets[i], self.source_offsets[i + 1]
            parts.append(adapter(x[:, start:end]))
        fused = torch.cat(parts, dim=1)
        return self.body(fused)


class DualDecoderOnlyUNet(nn.Module):
    """Ablation control: shared encoder + dual parallel decoders, NO seg→height
    injection. Each decoder is fully independent. Compare against
    CascadeDualDecoderUNet to isolate the contribution of cascade injection.
    """

    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        assert out_channels == 4

        # === Shared encoder (same as cascade) ===
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # === Seg decoder ===
        self.s_up1 = UpsampleBlock(256, 128)
        self.s_conv1 = DoubleConv(256, 128)
        self.s_up2 = UpsampleBlock(128, 64)
        self.s_conv2 = DoubleConv(128, 64)
        self.s_up3 = UpsampleBlock(64, 32)
        self.s_conv3 = DoubleConv(64, 32)
        self.seg_head = nn.Conv2d(32, 3, kernel_size=1)

        # === Height decoder — NO seg injection ===
        self.h_up1 = UpsampleBlock(256, 128)
        self.h_conv1 = DoubleConv(256, 128)
        self.h_up2 = UpsampleBlock(128, 64)
        self.h_conv2 = DoubleConv(128, 64)
        self.h_up3 = UpsampleBlock(64, 32)
        self.h_conv3 = DoubleConv(64, 32)   # NOTE: 64 not 64+3 — no seg injection
        self.height_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Seg decoder
        s = self.s_up1(x4)
        s = self.s_conv1(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_conv2(torch.cat([x2, s], dim=1))
        s = self.s_up3(s)
        s = self.s_conv3(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)

        # Height decoder (fully independent)
        h = self.h_up1(x4)
        h = self.h_conv1(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_conv2(torch.cat([x2, h], dim=1))
        h = self.h_up3(h)
        h = self.h_conv3(torch.cat([x1, h], dim=1))   # No seg injected here
        height_logits = self.height_head(h)

        return torch.cat([seg_logits, height_logits], dim=1)


class CascadeDualDecoderUNet(nn.Module):
    """Shared encoder + dual parallel decoders. Cascade design:
    seg decoder runs first; its sigmoid(logits).detach() is injected into the
    height decoder at the final upsample scale. Height-loss gradients NEVER
    flow back to the seg decoder.

    Output: (B, 4, H, W) where channels [0:3] are seg logits and [3] is height
    logit (will be passed through softplus by GeoFMNet.activate).
    """

    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        assert out_channels == 4, "CascadeDualDecoderUNet only supports out_channels=4"

        # === Shared encoder (3 downsamples → 32x32 bottleneck) ===
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # === Seg decoder ===
        self.s_up1 = UpsampleBlock(256, 128)
        self.s_conv1 = DoubleConv(256, 128)
        self.s_up2 = UpsampleBlock(128, 64)
        self.s_conv2 = DoubleConv(128, 64)
        self.s_up3 = UpsampleBlock(64, 32)
        self.s_conv3 = DoubleConv(64, 32)
        self.seg_head = nn.Conv2d(32, 3, kernel_size=1)

        # === Height decoder (parallel, only sees encoder features + detached seg at final) ===
        self.h_up1 = UpsampleBlock(256, 128)
        self.h_conv1 = DoubleConv(256, 128)
        self.h_up2 = UpsampleBlock(128, 64)
        self.h_conv2 = DoubleConv(128, 64)
        self.h_up3 = UpsampleBlock(64, 32)
        # Inject detached seg-probs (3 channels) at full resolution
        self.h_conv3 = DoubleConv(64 + 3, 32)
        self.height_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Seg decoder
        s = self.s_up1(x4)
        s = self.s_conv1(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_conv2(torch.cat([x2, s], dim=1))
        s = self.s_up3(s)
        s = self.s_conv3(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)  # (B, 3, H, W)

        # Detach seg probs before feeding into height decoder
        seg_probs_detached = torch.sigmoid(seg_logits).detach()

        # Height decoder (parallel through encoder features)
        h = self.h_up1(x4)
        h = self.h_conv1(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_conv2(torch.cat([x2, h], dim=1))
        h = self.h_up3(h)
        # Inject detached seg here, at full resolution
        h = self.h_conv3(torch.cat([x1, h, seg_probs_detached], dim=1))
        height_logits = self.height_head(h)  # (B, 1, H, W)

        return torch.cat([seg_logits, height_logits], dim=1)


# =============================================================================
# LayerNorm-based blocks (ConvNeXt / transformer style) — used by late-fusion
# multi-source UNet that fuses dense + token features at the 16² bottleneck.
# Rationale: all FM embeddings come from transformers whose output is LN-normalized
# per-token. Re-using LN at the adapter respects this and avoids re-normalizing
# with cross-sample statistics (which BN would do).
# =============================================================================

class ChannelLN(nn.Module):
    """LayerNorm on the channel dimension at each spatial location (ConvNeXt style).

    For input (B, C, H, W), normalizes the C-dim feature vector at each (b, h, w),
    matching what transformers do per-token.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        # (B, C, H, W) -> (B, H, W, C) -> LayerNorm on last dim -> back
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class DoubleConvLN(nn.Module):
    """Same shape as DoubleConv but uses ChannelLN + GELU (no batch dependency)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlockLN(nn.Module):
    """Bilinear ×2 + Conv + ChannelLN + GELU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class StridedDoubleConvLN(nn.Module):
    """Replacement for `MaxPool(2) + DoubleConvLN`: a single block that downsamples
    2× via a learnable strided conv (kernel=3, stride=2, padding=1) and then
    applies another Conv+LN+GELU. Total compute is roughly equivalent to
    MaxPool+DoubleConv but the downsample is learnable.

    Sequence:  Conv(k=3, s=2, p=1) -> LN -> GELU -> Conv(k=3, s=1, p=1) -> LN -> GELU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SourceAdapterStemLN(nn.Module):
    """Per-source adapter using LayerNorm throughout (matches FM transformer-style
    output normalization). Drop-in replacement for SourceAdapterStem.

    Structure:  LN -> Conv 1×1 -> LN -> GELU -> Conv 3×3 -> LN -> GELU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ChannelLN(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AdapterFusionLateFusionUNet(nn.Module):
    """Multi-source late-fusion UNet (all LN-based, ConvNeXt style) +
    V2-style prior-residual height head with soft seg-gated mix.

    Inputs:
      x_dense: (B, sum(dense_channels), 256, 256) — concat of dense FM outputs
               (e.g., AlphaEarth 64ch + Tessera 128ch)
      tokens:  (B, sum(token_channels), 16, 16)   — concat of token FM outputs
               at native spatial resolution (NOT upsampled), e.g., TerraMind/THOR 4×768

    Architecture:
      Dense  --(per-source LN adapter)--> 128ch @ 256² --UNet encoder (4-level)--> 384ch @ 16²
      Tokens --(per-source LN adapter)--> 256ch @ 16²
      Fuse at 16²: concat → DoubleConvLN → 384ch @ 16²
      Dual decoder (seg + height, no cascade injection during decoding) → up to 256²
        - seg_head: 3ch with class-frequency bias init (RetinaNet style)
        - height_head: 2ch residual (h_b, h_v), added to fixed train priors, then
          soft-mixed via detached seg_probs (V2 design).

    Output: (B, 4, 256, 256). Channel 3 (height) is PRE-ACTIVATED — GeoFMNet.activate
    must skip softplus on it (we set output_height_pre_activated=True).
    """

    # Flags consumed by GeoFMEmbed2Heights and GeoFMNet wrappers
    is_late_fusion = True
    output_height_pre_activated = True   # tells activate() to skip softplus on ch 3

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64, fused_bottleneck=384):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        # ----- Per-source LN adapters -----
        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in token_channels
        ])
        fused_dense = adapter_out * len(dense_channels)   # 64×2 = 128
        fused_token = adapter_out * len(token_channels)   # 64×4 = 256

        self.dense_offsets = [0]
        for c in dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        # ----- Dense encoder (4-level, 256² → 16²) -----
        # Each `down*` uses StridedDoubleConvLN (learnable strided conv k=3 s=2 p=1
        # + LN + GELU + Conv k=3 + LN + GELU), replacing the classic MaxPool+DoubleConv.
        self.inc   = DoubleConvLN(fused_dense, 32)              # 256²
        self.down1 = StridedDoubleConvLN(32, 64)                # 128²
        self.down2 = StridedDoubleConvLN(64, 128)               # 64²
        self.down3 = StridedDoubleConvLN(128, 256)              # 32²
        self.down4 = StridedDoubleConvLN(256, fused_bottleneck) # 16²

        # ----- Late fusion at 16² (concat dense bottleneck + token features) -----
        self.fuse = DoubleConvLN(fused_bottleneck + fused_token, fused_bottleneck)

        # ----- Seg decoder (4-level mirror, with skip connections) -----
        self.s_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.s_c4  = DoubleConvLN(256 + 256, 256)
        self.s_up3 = UpsampleBlockLN(256, 128)
        self.s_c3  = DoubleConvLN(128 + 128, 128)
        self.s_up2 = UpsampleBlockLN(128, 64)
        self.s_c2  = DoubleConvLN(64 + 64, 64)
        self.s_up1 = UpsampleBlockLN(64, 32)
        self.s_c1  = DoubleConvLN(32 + 32, 32)
        self.seg_head = nn.Conv2d(32, 3, kernel_size=1)
        # Class-frequency bias init (RetinaNet style): initial sigmoid(output) ≈ prior
        nn.init.normal_(self.seg_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.seg_head.bias[0], SEG_LOGIT_PRIOR_BUILDING)
        nn.init.constant_(self.seg_head.bias[1], SEG_LOGIT_PRIOR_VEGETATION)
        nn.init.constant_(self.seg_head.bias[2], SEG_LOGIT_PRIOR_WATER)

        # ----- Height decoder (mirror; outputs RESIDUAL h_b, h_v) -----
        self.h_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.h_c4  = DoubleConvLN(256 + 256, 256)
        self.h_up3 = UpsampleBlockLN(256, 128)
        self.h_c3  = DoubleConvLN(128 + 128, 128)
        self.h_up2 = UpsampleBlockLN(128, 64)
        self.h_c2  = DoubleConvLN(64 + 64, 64)
        self.h_up1 = UpsampleBlockLN(64, 32)
        self.h_c1  = DoubleConvLN(32 + 32, 32)
        # 2-channel residual head: residual_b, residual_v
        self.height_head = nn.Conv2d(32, 2, kernel_size=1)
        nn.init.normal_(self.height_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.height_head.bias)
        # Fixed priors (normalized to height_norm_constant=30)
        self.register_buffer("prior_h_b", torch.tensor(PRIOR_H_B_NORMALIZED))
        self.register_buffer("prior_h_v", torch.tensor(PRIOR_H_V_NORMALIZED))
        # Per-class height for optional per_class_height_weight loss
        self._latest_h_b = None
        self._latest_h_v = None

    def forward(self, x_dense, tokens):
        # Adapt dense sources independently
        dense_parts = []
        for i, ad in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(ad(x_dense[:, s:e]))
        dense_in = torch.cat(dense_parts, dim=1)   # (B, fused_dense, 256, 256)

        # Adapt token sources independently (at native 16² spatial)
        token_parts = []
        for i, ad in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(ad(tokens[:, s:e]))
        token_feats = torch.cat(token_parts, dim=1)   # (B, fused_token, 16, 16)

        # Dense encoder
        x1 = self.inc(dense_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)                          # (B, fused_bottleneck, 16, 16)

        # Late fusion at 16²
        fused = torch.cat([x5, token_feats], dim=1)
        fused = self.fuse(fused)                     # (B, fused_bottleneck, 16, 16)

        # Seg decoder
        s = self.s_up4(fused)
        s = self.s_c4(torch.cat([x4, s], dim=1))
        s = self.s_up3(s)
        s = self.s_c3(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_c2(torch.cat([x2, s], dim=1))
        s = self.s_up1(s)
        s = self.s_c1(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)

        # Detach seg-probs for cascade (no grad flow back through seg head from height)
        seg_probs_detached = torch.sigmoid(seg_logits).detach()

        # Height decoder produces per-class RESIDUAL
        h = self.h_up4(fused)
        h = self.h_c4(torch.cat([x4, h], dim=1))
        h = self.h_up3(h)
        h = self.h_c3(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_c2(torch.cat([x2, h], dim=1))
        h = self.h_up1(h)
        h = self.h_c1(torch.cat([x1, h], dim=1))
        h_residual = self.height_head(h)   # (B, 2, 256, 256)

        # Add fixed train-prior to residual → absolute (normalized) height per class
        h_b = self.prior_h_b + h_residual[:, 0:1]
        h_v = self.prior_h_v + h_residual[:, 1:2]
        h_b = torch.clamp(h_b, min=0.0, max=15.0)
        h_v = torch.clamp(h_v, min=0.0, max=15.0)

        # Soft-mix using detached seg-probs.
        # Background pixels (both seg low) → final_h ≈ 0 naturally.
        final_h = (seg_probs_detached[:, 0:1] * h_b +
                   seg_probs_detached[:, 1:2] * h_v)

        # Stash per-class for optional direct supervision in loss
        self._latest_h_b = h_b
        self._latest_h_v = h_v

        # Output: 3 seg LOGITS + 1 final normalized height (NOT a logit;
        # output_height_pre_activated=True tells GeoFMNet.activate to skip softplus)
        return torch.cat([seg_logits, final_h], dim=1)


class AdapterFusionLateFusionUNetDecoupled(nn.Module):
    """Late-fusion UNet with FULLY DECOUPLED height + seg decoders (no soft-mix).

    Same encoder + adapters + fusion as AdapterFusionLateFusionUNet,
    but the height path:
      - Single-channel direct output (no per-class h_b/h_v split)
      - NO interaction with seg probs (no soft-mix, no detach gating)
      - GeoFMNet.activate applies standard softplus → [0, +)

    Rationale: V2 soft-mix multiplies h_b by seg_prob_b at every pixel. For
    sparse classes (building 1%), seg_prob at most pixels is near 0, which
    means h_b's gradient is dominated by very few pixels. A fully decoupled
    head learns "the height of THIS pixel" directly, with gradient flowing
    from every pixel.

    Trade-off:
      + Stronger gradient at ALL pixels → better convergence on sparse classes
      - Loses prior-residual + automatic background-≈-0 trick
      - Model must implicitly learn "background → 0m"

    Output (B, 4, 256, 256): 3 seg logits + 1 height logit.
    Caller (GeoFMNet.activate) applies sigmoid to seg and softplus to height
    (NOT pre-activated — same as adapter_fusion_dual_only behavior).
    """

    is_late_fusion = True   # GeoFMEmbed2Heights routes y["tokens"] for us

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64, fused_bottleneck=384):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in token_channels
        ])
        fused_dense = adapter_out * len(dense_channels)
        fused_token = adapter_out * len(token_channels)

        self.dense_offsets = [0]
        for c in dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        # Encoder (4-level, 256² → 16²)
        self.inc   = DoubleConvLN(fused_dense, 32)
        self.down1 = StridedDoubleConvLN(32, 64)
        self.down2 = StridedDoubleConvLN(64, 128)
        self.down3 = StridedDoubleConvLN(128, 256)
        self.down4 = StridedDoubleConvLN(256, fused_bottleneck)

        # Late fusion at 16²
        self.fuse = DoubleConvLN(fused_bottleneck + fused_token, fused_bottleneck)

        # Seg decoder
        self.s_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.s_c4  = DoubleConvLN(256 + 256, 256)
        self.s_up3 = UpsampleBlockLN(256, 128)
        self.s_c3  = DoubleConvLN(128 + 128, 128)
        self.s_up2 = UpsampleBlockLN(128, 64)
        self.s_c2  = DoubleConvLN(64 + 64, 64)
        self.s_up1 = UpsampleBlockLN(64, 32)
        self.s_c1  = DoubleConvLN(32 + 32, 32)
        self.seg_head = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.normal_(self.seg_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.seg_head.bias[0], SEG_LOGIT_PRIOR_BUILDING)
        nn.init.constant_(self.seg_head.bias[1], SEG_LOGIT_PRIOR_VEGETATION)
        nn.init.constant_(self.seg_head.bias[2], SEG_LOGIT_PRIOR_WATER)

        # Height decoder — FULLY DECOUPLED (no seg interaction)
        self.h_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.h_c4  = DoubleConvLN(256 + 256, 256)
        self.h_up3 = UpsampleBlockLN(256, 128)
        self.h_c3  = DoubleConvLN(128 + 128, 128)
        self.h_up2 = UpsampleBlockLN(128, 64)
        self.h_c2  = DoubleConvLN(64 + 64, 64)
        self.h_up1 = UpsampleBlockLN(64, 32)
        self.h_c1  = DoubleConvLN(32 + 32, 32)
        # Single channel direct output (no per-class, no soft-mix)
        self.height_head = nn.Conv2d(32, 1, kernel_size=1)
        # Standard init — let model learn from scratch (softplus → exp-like at init,
        # so output is ~0.7, post-clamp ~0.7 norm = 21m. Not great but model will learn.)

    def forward(self, x_dense, tokens):
        # Adapt dense
        dense_parts = []
        for i, ad in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(ad(x_dense[:, s:e]))
        dense_in = torch.cat(dense_parts, dim=1)

        # Adapt tokens
        token_parts = []
        for i, ad in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(ad(tokens[:, s:e]))
        token_feats = torch.cat(token_parts, dim=1)

        # Encoder
        x1 = self.inc(dense_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Late fusion
        fused = torch.cat([x5, token_feats], dim=1)
        fused = self.fuse(fused)

        # Seg decoder
        s = self.s_up4(fused)
        s = self.s_c4(torch.cat([x4, s], dim=1))
        s = self.s_up3(s)
        s = self.s_c3(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_c2(torch.cat([x2, s], dim=1))
        s = self.s_up1(s)
        s = self.s_c1(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)

        # Height decoder — FULLY INDEPENDENT (no seg interaction)
        h = self.h_up4(fused)
        h = self.h_c4(torch.cat([x4, h], dim=1))
        h = self.h_up3(h)
        h = self.h_c3(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_c2(torch.cat([x2, h], dim=1))
        h = self.h_up1(h)
        h = self.h_c1(torch.cat([x1, h], dim=1))
        height_logit = self.height_head(h)   # (B, 1, 256, 256) — raw logit, softplus applied by activate()

        return torch.cat([seg_logits, height_logit], dim=1)


class AdapterFusionLateFusionUNetDecoupledRawHeight(AdapterFusionLateFusionUNetDecoupled):
    """Raw-height variant: skips softplus, allows negative raw output during training.

    Matches official emb2heights baseline design: head outputs raw (no clamp,
    no activation). Loss sees raw values directly. Negative outputs are pushed
    upward by smooth_l1 gradient against positive GT.

    Flags consumed by GeoFMNet.activate():
      - output_height_pre_activated=True: skip softplus.
      - raw_height_allow_negative=True:   skip min=0 clamp (allow negative at training).
                                          Only clamp max=15 for upper safety.

    This is different from V2 models which set only `output_height_pre_activated`
    and DO want min=0 clamp (their internal softplus already guarantees ≥0).
    """
    output_height_pre_activated = True
    raw_height_allow_negative = True

    def __init__(self, *args, height_bias_init=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        # Bias init = 0.3 (≈ train mean 9m / 30) gives the model a head start near
        # the typical fg-pixel height. Not strictly necessary (gradient flows even
        # from negative now), but speeds convergence.
        import torch.nn as _nn
        _nn.init.normal_(self.height_head.weight, mean=0.0, std=0.001)
        _nn.init.constant_(self.height_head.bias, float(height_bias_init))


class AdapterFusionLateFusionUNetDecoupledCE(nn.Module):
    """4-class softmax-CE variant of AdapterFusionLateFusionUNetDecoupled.

    Same encoder + adapters + fusion + decoders as a05, but seg_head outputs
    4 channels (bld, veg, water, bg) for softmax CE training instead of 3
    independent sigmoid channels.

    Output (B, 5, H, W): 4 RAW seg logits (no activation) + 1 height logit.
    GeoFMNet.activate() detects `is_ce_seg=True` and applies:
      - seg ch0-3: softmax along dim=1 (only at inference)
      - height ch4: softplus + clamp
    For training, the loss (SoftmaxCE4ClassLoss) consumes raw logits directly
    — see GeoFMEmbed2Heights forward path for is_ce4 branch.

    Bias init for softmax (4-class log-prior, NOT logit/sigmoid version):
      bld: log(0.0124) ≈ -4.39
      veg: log(0.4036) ≈ -0.91
      water: log(0.0212) ≈ -3.85
      bg: log(1 - 0.0124 - 0.4036 - 0.0212) = log(0.5628) ≈ -0.575
    These set the model's initial output distribution to match the empirical
    class prior, avoiding the "model collapses to all-bg" failure mode.
    """

    is_late_fusion = True
    is_ce_seg = True   # marker for GeoFMNet.activate() to route to softmax

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=5, adapter_out=64, fused_bottleneck=384):
        super().__init__()
        assert out_channels == 5, f"CE body expects out_channels=5 (4 seg + 1 height); got {out_channels}"
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in token_channels
        ])
        fused_dense = adapter_out * len(dense_channels)
        fused_token = adapter_out * len(token_channels)

        self.dense_offsets = [0]
        for c in dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        # Encoder (4-level, 256² → 16²) — IDENTICAL to a05
        self.inc   = DoubleConvLN(fused_dense, 32)
        self.down1 = StridedDoubleConvLN(32, 64)
        self.down2 = StridedDoubleConvLN(64, 128)
        self.down3 = StridedDoubleConvLN(128, 256)
        self.down4 = StridedDoubleConvLN(256, fused_bottleneck)

        # Late fusion at 16²
        self.fuse = DoubleConvLN(fused_bottleneck + fused_token, fused_bottleneck)

        # Seg decoder — IDENTICAL to a05
        self.s_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.s_c4  = DoubleConvLN(256 + 256, 256)
        self.s_up3 = UpsampleBlockLN(256, 128)
        self.s_c3  = DoubleConvLN(128 + 128, 128)
        self.s_up2 = UpsampleBlockLN(128, 64)
        self.s_c2  = DoubleConvLN(64 + 64, 64)
        self.s_up1 = UpsampleBlockLN(64, 32)
        self.s_c1  = DoubleConvLN(32 + 32, 32)
        # 4-class seg head (bld, veg, water, bg)
        self.seg_head = nn.Conv2d(32, 4, kernel_size=1)
        nn.init.normal_(self.seg_head.weight, mean=0.0, std=0.001)
        # log-prior bias for softmax (not log-odds; just log p)
        _LOG_P_BLD   = _math.log(0.0124)
        _LOG_P_VEG   = _math.log(0.4036)
        _LOG_P_WATER = _math.log(0.0212)
        _LOG_P_BG    = _math.log(1.0 - 0.0124 - 0.4036 - 0.0212)   # ≈ log(0.5628)
        nn.init.constant_(self.seg_head.bias[0], _LOG_P_BLD)
        nn.init.constant_(self.seg_head.bias[1], _LOG_P_VEG)
        nn.init.constant_(self.seg_head.bias[2], _LOG_P_WATER)
        nn.init.constant_(self.seg_head.bias[3], _LOG_P_BG)

        # Height decoder — FULLY DECOUPLED (no seg interaction), IDENTICAL to a05
        self.h_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.h_c4  = DoubleConvLN(256 + 256, 256)
        self.h_up3 = UpsampleBlockLN(256, 128)
        self.h_c3  = DoubleConvLN(128 + 128, 128)
        self.h_up2 = UpsampleBlockLN(128, 64)
        self.h_c2  = DoubleConvLN(64 + 64, 64)
        self.h_up1 = UpsampleBlockLN(64, 32)
        self.h_c1  = DoubleConvLN(32 + 32, 32)
        self.height_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_dense, tokens):
        # Adapt dense
        dense_parts = []
        for i, ad in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(ad(x_dense[:, s:e]))
        dense_in = torch.cat(dense_parts, dim=1)

        # Adapt tokens
        token_parts = []
        for i, ad in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(ad(tokens[:, s:e]))
        token_feats = torch.cat(token_parts, dim=1)

        # Encoder
        x1 = self.inc(dense_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Late fusion
        fused = torch.cat([x5, token_feats], dim=1)
        fused = self.fuse(fused)

        # Seg decoder (4-class logits, NO activation here)
        s = self.s_up4(fused)
        s = self.s_c4(torch.cat([x4, s], dim=1))
        s = self.s_up3(s)
        s = self.s_c3(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_c2(torch.cat([x2, s], dim=1))
        s = self.s_up1(s)
        s = self.s_c1(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)   # (B, 4, 256, 256) raw

        # Height decoder
        h = self.h_up4(fused)
        h = self.h_c4(torch.cat([x4, h], dim=1))
        h = self.h_up3(h)
        h = self.h_c3(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_c2(torch.cat([x2, h], dim=1))
        h = self.h_up1(h)
        h = self.h_c1(torch.cat([x1, h], dim=1))
        height_logit = self.height_head(h)   # (B, 1, 256, 256) — raw, softplus by activate()

        return torch.cat([seg_logits, height_logit], dim=1)   # (B, 5, 256, 256)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================================
# ConvNeXt-T + UNetFormer (architectural diversity for specialist ensemble)
# ----------------------------------------------------------------------------
# Encoder: ConvNeXt-Tiny (depths=3,3,9,3, dims=96,192,384,768) with custom stem
#   accepting fused-dense (128ch) instead of RGB. Returns 4-stage pyramid at
#   strides (4, 8, 16, 32).
# Token fusion: at the 1/16 stage (c3), concat 256ch fused-tokens + reduce.
# Decoder: UNetFormer-style — at each upsample stage, fuse skip and run a
#   GlobalLocalTransformerBlock (window MSA + DwConv 3x3 local path + MLP).
# Output: 4-ch (3 seg + 1 height), same interface as other specialist bodies.
# ============================================================================


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: DwConv 7x7 -> LN -> 1x1 -> GELU -> 1x1 -> residual (+drop_path).

    LayerScale init=1.0 (NOT 1e-6 from ConvNeXt-v1 paper). The 1e-6 init is for
    fine-tuning ImageNet-pretrained weights — it kills from-scratch training on
    small datasets. Pilot run 2026-05-21: with init=1e-6 the gamma grew to only
    ~0.022 over 10k iters, output collapsed to constant, val miou_b stuck at 0.
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        from torchvision.ops import StochasticDepth
        self.drop_path = StochasticDepth(drop_path, "row") if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2).contiguous()
        return residual + self.drop_path(x)


class ConvNeXtTinyEncoder(nn.Module):
    """ConvNeXt-Tiny with custom in_channels stem. Returns 5-stage pyramid.

    Stem is SPLIT into two stride-2 stages (instead of one 4x4-stride-4 stem)
    to expose a 1/2-resolution c0 feature for high-res skip in the decoder.
    Building edges are 1-2 pixels thin; without 1/2 skip the decoder relies
    on bilinear upsampling from 1/4, losing edge detail.

    Depths/dims default to ConvNeXt-T (depths=3,3,9,3; dims=96,192,384,768).
    For lighter variants pass smaller (depths, dims).

    Output pyramid: (c0@1/2 [dims[0]//2 ch], c1@1/4, c2@1/8, c3@1/16, c4@1/32)
    """
    def __init__(self, in_channels=128, depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768), drop_path_rate=0.1):
        super().__init__()
        assert len(depths) == len(dims) == 4
        # Two-stage stem: in_channels → dims[0]//2 (1/2) → dims[0] (1/4)
        self.stem_p1 = nn.Sequential(
            nn.Conv2d(in_channels, dims[0] // 2, 3, stride=2, padding=1),
            ChannelLN(dims[0] // 2),
        )
        self.stem_p2 = nn.Sequential(
            nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1),
            ChannelLN(dims[0]),
        )
        # c0 channel count for downstream use
        self.c0_channels = dims[0] // 2

        self.downsample_layers = nn.ModuleList([
            nn.Sequential(ChannelLN(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2))
            for i in range(3)
        ])
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            blocks = nn.Sequential(*[
                ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])
            ])
            self.stages.append(blocks)
            cur += depths[i]

    def forward(self, x):
        c0 = self.stem_p1(x)                                  # 1/2, dims[0]//2
        x = self.stem_p2(c0)                                  # 1/4, dims[0]
        c1 = self.stages[0](x)                                # 1/4
        c2 = self.stages[1](self.downsample_layers[0](c1))   # 1/8
        c3 = self.stages[2](self.downsample_layers[1](c2))   # 1/16
        c4 = self.stages[3](self.downsample_layers[2](c3))   # 1/32
        return c0, c1, c2, c3, c4


def _window_partition(x, ws):
    """(B, H, W, C) -> (B*num_windows, ws*ws, C). H, W must divide ws."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)


def _window_reverse(x, ws, H, W):
    B = x.shape[0] // ((H // ws) * (W // ws))
    x = x.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class GlobalLocalTransformerBlock(nn.Module):
    """UNetFormer-style block: window-MSA + DwConv 3x3 local + MLP.

    Uses F.scaled_dot_product_attention for the window self-attention.
    Padding handles arbitrary H, W (not just multiples of window_size).
    """
    def __init__(self, dim, num_heads=8, window_size=8, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.local_dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        from torchvision.ops import StochasticDepth
        self.drop_path = StochasticDepth(drop_path, "row") if drop_path > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size

        # Pre-norm + window attention path
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_ln = self.norm1(x_perm)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_ln = F.pad(x_ln, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x_ln.shape[1], x_ln.shape[2]

        x_w = _window_partition(x_ln, ws)  # (B*nw, ws*ws, C)
        qkv = self.qkv(x_w).reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nw, h, ws*ws, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)  # (B*nw, h, ws*ws, hd)
        attn = attn.transpose(1, 2).reshape(-1, ws * ws, C)
        attn = self.proj(attn)
        x_attn = _window_reverse(attn, ws, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            x_attn = x_attn[:, :H, :W, :].contiguous()

        # Local DwConv path on attention output (now back to (B, C, H, W))
        x_local = self.local_dwconv(x_attn.permute(0, 3, 1, 2).contiguous())
        x = x + self.drop_path(x_local)

        # MLP path
        x_mlp = x.permute(0, 2, 3, 1).contiguous()
        x_mlp = self.norm2(x_mlp)
        x_mlp = self.mlp(x_mlp).permute(0, 3, 1, 2).contiguous()
        return x + self.drop_path(x_mlp)


class UNetFormerDecoder(nn.Module):
    """Decoder for ConvNeXt pyramid (c1..c4) with token-fused c3.

    Stages:
      c4 (1/32)  -> upsample 2x -> add c3 (1/16, token-fused) -> GLTB
       -> upsample 2x -> add c2 (1/8)  -> GLTB
       -> upsample 2x -> add c1 (1/4)  -> DoubleConvLN
       -> upsample 4x -> final 1x1 conv head
    """
    def __init__(self, encoder_dims=(96, 192, 384, 768), out_channels=4,
                 num_heads=(4, 8), drop_path=0.0):
        super().__init__()
        c1, c2, c3, c4 = encoder_dims
        self.up_c4 = nn.Conv2d(c4, c3, 1)
        self.gltb_c3 = GlobalLocalTransformerBlock(c3, num_heads=num_heads[1], drop_path=drop_path)

        self.up_c3 = nn.Conv2d(c3, c2, 1)
        self.gltb_c2 = GlobalLocalTransformerBlock(c2, num_heads=num_heads[0], drop_path=drop_path)

        self.up_c2 = nn.Conv2d(c2, c1, 1)
        self.fuse_c1 = DoubleConvLN(c1, c1)

        self.final_up = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, padding=1),
            ChannelLN(c1 // 2),
            nn.GELU(),
        )
        self.head = nn.Conv2d(c1 // 2, out_channels, 1)

        # Class-prior bias init — CRITICAL for sparse-class training from scratch.
        # Matches a05 (LateFusionUNetDecoupled, line 1655-1659) and CascadeV2.
        # RetinaNet-style: weight std=0.001 + bias=log-prior keeps initial output
        # near class prior (e.g. 0.0124 for building) so BCE gradient is balanced
        # and Tversky has signal from spatial structure of decoder features.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        if self.head.bias is not None:
            if out_channels >= 3:
                nn.init.constant_(self.head.bias[0], SEG_LOGIT_PRIOR_BUILDING)
                nn.init.constant_(self.head.bias[1], SEG_LOGIT_PRIOR_VEGETATION)
                nn.init.constant_(self.head.bias[2], SEG_LOGIT_PRIOR_WATER)
            if out_channels >= 4:
                nn.init.zeros_(self.head.bias[3:])  # height logit — softplus(0)≈0.69m start

    def forward(self, c1, c2, c3, c4):
        x = F.interpolate(self.up_c4(c4), scale_factor=2, mode="bilinear", align_corners=False)
        x = x + c3
        x = self.gltb_c3(x)

        x = F.interpolate(self.up_c3(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = x + c2
        x = self.gltb_c2(x)

        x = F.interpolate(self.up_c2(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = x + c1
        x = self.fuse_c1(x)

        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.final_up(x)
        return self.head(x)


class ConvNeXtUNetFormerBody(nn.Module):
    """ConvNeXt-T encoder (split-stem) + token fusion at 1/16 + 5-level conv decoder.

    Decoder is SimpleUNetDecoder (no GLTB). Stem split into 2 stride-2 stages to
    expose c0@1/2 for high-res skip. Keeping the class name for backward compat
    with model_type='adapter_fusion_convnext_unetformer'.

    Spatial constraint: dense H, W must be multiples of 32. 256 ✓, 192 ✓.
    """
    is_late_fusion = True

    def __init__(self, dense_in_channels=128, token_in_channels=256,
                 out_channels=4, encoder_dims=(96, 192, 384, 768),
                 encoder_depths=(3, 3, 9, 3), drop_path_rate=0.1):
        super().__init__()
        assert out_channels == 4
        self.encoder = ConvNeXtTinyEncoder(
            in_channels=dense_in_channels, depths=encoder_depths,
            dims=encoder_dims, drop_path_rate=drop_path_rate,
        )
        # Token fusion at 1/16 (encoder c3 = dims[2])
        self.token_fuse = DoubleConvLN(encoder_dims[2] + token_in_channels, encoder_dims[2])
        # 5-level decoder: c0 channel = dims[0]//2 from split stem
        c0_ch = encoder_dims[0] // 2
        self.decoder = SimpleUNetDecoder(
            encoder_dims=(c0_ch,) + tuple(encoder_dims),
            out_channels=out_channels,
        )

    def forward(self, x_dense, tokens):
        c0, c1, c2, c3, c4 = self.encoder(x_dense)
        if c3.shape[2:] != tokens.shape[2:]:
            raise RuntimeError(
                f"Token-fusion shape mismatch: encoder c3 {tuple(c3.shape[2:])} != "
                f"tokens {tuple(tokens.shape[2:])}. Expect both at H/16."
            )
        c3 = self.token_fuse(torch.cat([c3, tokens], dim=1))
        return self.decoder(c0, c1, c2, c3, c4)


class AdapterFusionConvNeXtUNetFormer(nn.Module):
    """LN per-source adapters (same pattern as a05/HRNet) + ConvNeXt-T UNetFormer body.

    Designed for specialist single-class training. 3rd architectural family
    after UNet late-fusion (a05) and Lite-HRNet token-fusion. Provides
    transformer-decoder inductive bias for ensemble diversity.
    """
    is_late_fusion = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 encoder_dims=(96, 192, 384, 768),
                 encoder_depths=(3, 3, 9, 3), drop_path_rate=0.1):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        fused_dense_in = adapter_out * len(self.dense_channels)
        fused_token_in = adapter_out * len(self.token_channels)

        self.body = ConvNeXtUNetFormerBody(
            dense_in_channels=fused_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            encoder_dims=encoder_dims,
            encoder_depths=encoder_depths,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x_dense, tokens):
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        fused_dense = torch.cat(dense_parts, dim=1)

        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)

        return self.body(fused_dense, fused_tokens)


# ============================================================================
# EfficientNet-B0 (ImageNet pretrained) + UNetFormer decoder
# ----------------------------------------------------------------------------
# Replacement for ConvNeXt-UNetFormer after 3 failed from-scratch attempts.
# EfficientNet-B0 brings:
#   - ImageNet 1.3M pretrained features for features[1:] (only stem is new)
#   - BatchNorm throughout (3 BN per MBConv × 7 stages) — running stats
#     absorb sparse-class bias automatically, unlike LayerNorm
#   - Spatial inductive bias via MBConv (depthwise-separable conv)
# Pyramid taps: features[2]@1/4=24ch, [3]@1/8=40ch, [5]@1/16=112ch, [7]@1/32=320ch
# ============================================================================


class EfficientNetB0PretrainedEncoder(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 backbone, custom 128ch stem.

    Loads `efficientnet_b0(weights=IMAGENET1K_V1)` then REPLACES features[0][0]
    (the 3-ch RGB stem Conv2d) with a Conv2d(in_channels, 32, ...) initialized
    via Kaiming default. All other layers (features[1] through features[7])
    keep their pretrained weights.

    Pyramid output (4-stage):
      c1 @ 1/4  (24 ch)  — output of features[2]
      c2 @ 1/8  (40 ch)  — output of features[3]
      c3 @ 1/16 (112 ch) — output of features[5]
      c4 @ 1/32 (320 ch) — output of features[7]

    Spatial constraint: dense H, W must be multiples of 32. 256 ✓, 192 ✓.
    """
    def __init__(self, in_channels=128, pretrained=True):
        super().__init__()
        import os
        from torchvision.models import efficientnet_b0
        # Build skeleton with no weights first (avoid torch.hub network/cache).
        net = efficientnet_b0(weights=None)
        if pretrained:
            # Optionally load a pre-downloaded efficientnet_b0 state_dict to skip
            # torch.hub network access (useful offline or on read-only-$HOME nodes):
            # set env EFFICIENTNET_B0_WEIGHTS to the checkpoint path. Otherwise
            # torchvision downloads IMAGENET1K_V1 to the default cache.
            weights_path = os.environ.get("EFFICIENTNET_B0_WEIGHTS", "")
            if weights_path and os.path.exists(weights_path):
                import torch
                sd = torch.load(weights_path, map_location="cpu", weights_only=True)
                miss, unex = net.load_state_dict(sd, strict=True)
                assert len(miss) == 0 and len(unex) == 0, \
                    f"EfficientNet-B0 state_dict load mismatch: miss={miss[:3]} unex={unex[:3]}"
            else:
                from torchvision.models import EfficientNet_B0_Weights
                net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Replace stem first conv to accept arbitrary in_channels.
        # features[0] is Conv2dNormActivation = [Conv2d, BN, SiLU]. The Conv2d
        # is at index [0]. Keep BN + SiLU; only re-init the Conv2d.
        old_stem_conv = net.features[0][0]
        new_stem_conv = nn.Conv2d(
            in_channels, old_stem_conv.out_channels,
            kernel_size=old_stem_conv.kernel_size,
            stride=old_stem_conv.stride,
            padding=old_stem_conv.padding,
            bias=False,
        )
        # Kaiming default init from nn.Conv2d.__init__ is fine for from-scratch stem.
        net.features[0][0] = new_stem_conv

        # Keep stem (features[0]) and 7 MB-block stages (features[1..7]).
        # features[8] is a final 1x1 conv that boosts to 1280ch — drop it; we
        # don't need it for feature pyramid extraction.
        self.features = net.features[:8]

    # 5-stage pyramid channel counts (c0 at 1/2 from stem, others at 1/4..1/32)
    PYRAMID_CHANNELS = (16, 24, 40, 112, 320)

    def forward(self, x):
        # Trace native strides: features[0]=stem stride 2 → 1/2,
        # [1] stride 1, [2] stride 2 → 1/4, [3] stride 2 → 1/8, [4] stride 1,
        # [5] stride 2 → 1/16, [6] stride 1, [7] stride 2 → 1/32.
        x = self.features[0](x)              # 1/2,  32 ch (post-stem-conv + BN + SiLU)
        c0 = self.features[1](x)             # 1/2,  16 ch  ← tap (NEW high-res skip)
        c1 = self.features[2](c0)            # 1/4,  24 ch  ← tap
        c2 = self.features[3](c1)            # 1/8,  40 ch  ← tap
        x = self.features[4](c2)             # 1/8,  80 ch (stride-1 inside)
        c3 = self.features[5](x)             # 1/16, 112 ch ← tap
        x = self.features[6](c3)             # 1/16, 192 ch (stride-1 inside)
        c4 = self.features[7](x)             # 1/32, 320 ch ← tap
        return c0, c1, c2, c3, c4


class SimpleUNetDecoder(nn.Module):
    """Conv-only UNet decoder for 5-stage pyramid (c0..c4). NO attention.

    Replaces UNetFormerDecoder (which had GLTB / window MSA killing sparse-class
    gradient). a05 and HRNet use pure conv decoders and train fine on the same
    data with 1.85% building prior.

    NEW (2026-05-21): added c0@1/2 high-res skip + 4th upsample stage. Building
    edges are 1-2px thin; without 1/2 skip the decoder relied on a single 4×
    bilinear upsample from 1/4 → 1/1, losing edge detail. Now decoder spans
    full encoder pyramid: 1/32 → 1/16 → 1/8 → 1/4 → 1/2 → 1/1.

    Stages:
      c4 (1/32) → up → concat c3 (1/16, token-fused) → DoubleConvLN
                → up → concat c2 (1/8)              → DoubleConvLN
                → up → concat c1 (1/4)              → DoubleConvLN
                → up → concat c0 (1/2)              → DoubleConvLN
                → up 2× → final 3×3 conv → 1×1 head → 4ch
    """
    def __init__(self, encoder_dims=(16, 24, 40, 112, 320), out_channels=4):
        super().__init__()
        # 5-level pyramid: (c0@1/2, c1@1/4, c2@1/8, c3@1/16, c4@1/32)
        assert len(encoder_dims) == 5, \
            f"encoder_dims must be 5-tuple (c0..c4), got {len(encoder_dims)}: {encoder_dims}"
        c0, c1, c2, c3, c4 = encoder_dims

        self.up_c4 = UpsampleBlockLN(c4, c3)
        self.fuse_c3 = DoubleConvLN(c3 + c3, c3)
        self.up_c3 = UpsampleBlockLN(c3, c2)
        self.fuse_c2 = DoubleConvLN(c2 + c2, c2)
        self.up_c2 = UpsampleBlockLN(c2, c1)
        self.fuse_c1 = DoubleConvLN(c1 + c1, c1)
        self.up_c1 = UpsampleBlockLN(c1, c0)
        self.fuse_c0 = DoubleConvLN(c0 + c0, c0)

        self.final_up = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1, bias=False),
            ChannelLN(c0),
            nn.GELU(),
        )
        self.head = nn.Conv2d(c0, out_channels, 1)

        # Class-prior bias init — same RetinaNet-style trick as a05.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        if self.head.bias is not None:
            if out_channels >= 3:
                nn.init.constant_(self.head.bias[0], SEG_LOGIT_PRIOR_BUILDING)
                nn.init.constant_(self.head.bias[1], SEG_LOGIT_PRIOR_VEGETATION)
                nn.init.constant_(self.head.bias[2], SEG_LOGIT_PRIOR_WATER)
            if out_channels >= 4:
                nn.init.zeros_(self.head.bias[3:])

    def forward(self, c0, c1, c2, c3, c4):
        x = self.up_c4(c4)                              # 1/16
        x = self.fuse_c3(torch.cat([x, c3], dim=1))
        x = self.up_c3(x)                               # 1/8
        x = self.fuse_c2(torch.cat([x, c2], dim=1))
        x = self.up_c2(x)                               # 1/4
        x = self.fuse_c1(torch.cat([x, c1], dim=1))
        x = self.up_c1(x)                               # 1/2
        x = self.fuse_c0(torch.cat([x, c0], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 1/1
        x = self.final_up(x)
        return self.head(x)


class EfficientNetUNetFormerBody(nn.Module):
    """EfficientNet-B0 encoder + token fusion at 1/16 + conv-only UNet decoder.

    Note: name kept as "UNetFormer" for backward compat with model_type registry,
    but DECODER is now SimpleUNetDecoder (no window attention). Window MSA was
    drowning out the sparse-class gradient signal at init (4 pilot failures).

    Token fusion: c3 (112ch @ 1/16) + tokens (256ch @ 1/16) → DoubleConvLN
    → 112ch @ 1/16. Decoder operates on (24, 40, 112, 320) pyramid.

    Spatial constraint: dense H, W must be multiples of 32. 256 ✓, 192 ✓.
    """
    is_late_fusion = True

    # EfficientNet-B0 5-stage pyramid (c0@1/2 from features[1], rest from native taps)
    ENCODER_CHANNELS = (16, 24, 40, 112, 320)

    def __init__(self, dense_in_channels=128, token_in_channels=256,
                 out_channels=4, pretrained=True, drop_path_rate=0.1):
        super().__init__()
        assert out_channels == 4
        self.encoder = EfficientNetB0PretrainedEncoder(
            in_channels=dense_in_channels, pretrained=pretrained,
        )
        c0, c1, c2, c3, c4 = self.ENCODER_CHANNELS
        # Token fusion at 1/16 (encoder c3 = 112ch)
        self.token_fuse = DoubleConvLN(c3 + token_in_channels, c3)
        # 5-level conv decoder with high-res c0 skip
        self.decoder = SimpleUNetDecoder(
            encoder_dims=self.ENCODER_CHANNELS,
            out_channels=out_channels,
        )

    def forward(self, x_dense, tokens):
        c0, c1, c2, c3, c4 = self.encoder(x_dense)
        if c3.shape[2:] != tokens.shape[2:]:
            raise RuntimeError(
                f"Token-fusion shape mismatch: encoder c3 {tuple(c3.shape[2:])} != "
                f"tokens {tuple(tokens.shape[2:])}. Expect both at H/16."
            )
        c3 = self.token_fuse(torch.cat([c3, tokens], dim=1))
        return self.decoder(c0, c1, c2, c3, c4)


class AdapterFusionEfficientNetUNetFormer(nn.Module):
    """LN per-source adapters (same as a05/HRNet/ConvNeXt) + EfficientNet-B0
    pretrained encoder + UNetFormer decoder.

    Architectural family #4 for specialist ensemble — adds BN-heavy + ImageNet
    pretrained inductive bias on top of conv (a05), multi-resolution (HRNet),
    and (failed) transformer (ConvNeXt).
    """
    is_late_fusion = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 pretrained=True, drop_path_rate=0.1):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])

        fused_dense_in = adapter_out * len(self.dense_channels)
        fused_token_in = adapter_out * len(self.token_channels)

        self.body = EfficientNetUNetFormerBody(
            dense_in_channels=fused_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x_dense, tokens):
        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        fused_dense = torch.cat(dense_parts, dim=1)

        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)

        return self.body(fused_dense, fused_tokens)


class MaskAdapter(nn.Module):
    """Simple adapter for binary mask input (1 or 2 channels).

    Skips the leading LN-on-1ch step that's present in SourceAdapterStemLN.
    Rationale: LayerNorm over a single channel computes mean over 1 scalar
    and returns ~0 everywhere (mean=x, var=0 → (x-x)/sqrt(eps)≈0), DESTROYING
    the binary mask information at the first layer. This adapter starts with
    Conv1x1 to expand in_channels→out_channels FIRST, then LN/GELU pipeline.

    Structure: Conv1x1(in→out) → LN(out) → GELU → Conv3x3(out→out) → LN(out) → GELU
    """
    def __init__(self, out_channels, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ChannelLN(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AdapterFusionEfficientNetUNetFormerHeightMask(nn.Module):
    """Building height regression specialist with binary mask input.

    Architecturally identical to AdapterFusionEfficientNetUNetFormer EXCEPT:
      - Extra `mask_adapter`: SourceAdapterStemLN(1, adapter_out) for the 1-ch
        binary building mask. Mask is the 3rd dense source.
      - forward() takes additional `building_mask` arg.
      - Otherwise: same dense (AE+Tessera) + token adapters, same EfficientNet
        body, same conv UNet decoder.

    Mask provenance:
      - Train: GT building binary, derived from target[:, 0:1] after harden_labels=True
        (harden_thresh=0.1) — so mask is exact {0, 1} matching real building pixels.
      - Inference: HARD output from avg(a05+HRNet) ensemble (T=0.5 binary) at
        ch0. Provided via dataset's building_mask_dir.

    Output: 4-ch (compatibility with existing 4-ch architecture). Only ch3
    (height) is supervised — others are garbage at convergence.
    """
    is_late_fusion = True
    requires_building_mask = True   # flag for GeoFMEmbed2Heights routing

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64,
                 pretrained=True, drop_path_rate=0.1):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_offsets = [0]
        for c in self.dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in self.token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.dense_channels
        ])
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in self.token_channels
        ])
        # New: binary mask adapter (1ch → adapter_out)
        # MaskAdapter (NOT SourceAdapterStemLN) — see MaskAdapter docstring for why.
        # LN-on-1ch destroys mask info; MaskAdapter expands 1→out first.
        self.mask_adapter = MaskAdapter(adapter_out)

        # fused_dense_in adds 1 more source (the mask) on top of dense_channels
        fused_dense_in = adapter_out * (len(self.dense_channels) + 1)
        fused_token_in = adapter_out * len(self.token_channels)

        self.body = EfficientNetUNetFormerBody(
            dense_in_channels=fused_dense_in,
            token_in_channels=fused_token_in,
            out_channels=out_channels,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x_dense, tokens, building_mask=None):
        if building_mask is None:
            raise ValueError(
                "AdapterFusionEfficientNetUNetFormerHeightMask requires "
                "`building_mask` (B, 1, H, W). Pass via GeoFMNet.forward kwargs."
            )

        dense_parts = []
        for i, adapter in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(adapter(x_dense[:, s:e]))
        # 3rd dense source: mask
        dense_parts.append(self.mask_adapter(building_mask))
        fused_dense = torch.cat(dense_parts, dim=1)

        token_parts = []
        for i, adapter in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(adapter(tokens[:, s:e]))
        fused_tokens = torch.cat(token_parts, dim=1)

        return self.body(fused_dense, fused_tokens)


class AdapterFusionLateFusionUNetDecoupledHeightMask(nn.Module):
    """Building height regression specialist with a05's UNet encoder + binary mask input.

    Architecture: SAME as AdapterFusionLateFusionUNetDecoupled (a05) EXCEPT:
      - Adds MaskAdapter (1ch binary mask → adapter_out channels) as the 3rd
        dense source. fused_dense input to encoder is 64*3 = 192ch.
      - Output is still 4-ch (seg_head 3 + height_head 1) for activate() compat.
      - Only ch3 (height) supervised — seg_head outputs are garbage.

    Why a05 encoder? After EfficientNet showed slow convergence (early step 500
    tversky 0.640 vs HRNet 0.505), the user requested switching to a05's proven
    UNet encoder (LN-based, 5-level, 32→64→128→256→bottleneck=384) which trains
    well on this dataset. Same encoder as a05/HRNet specialists — known good.
    """
    is_late_fusion = True
    requires_building_mask = True

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64, fused_bottleneck=384):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in dense_channels
        ])
        # Mask adapter (NOT SourceAdapterStemLN — see MaskAdapter docstring)
        self.mask_adapter = MaskAdapter(adapter_out)
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in token_channels
        ])
        # Fused dense includes mask as 3rd source: 64 * 3 = 192
        fused_dense = adapter_out * (len(dense_channels) + 1)
        fused_token = adapter_out * len(token_channels)

        self.dense_offsets = [0]
        for c in dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        # === Encoder (same as a05): 4-level 256² → 16² ===
        self.inc   = DoubleConvLN(fused_dense, 32)
        self.down1 = StridedDoubleConvLN(32, 64)
        self.down2 = StridedDoubleConvLN(64, 128)
        self.down3 = StridedDoubleConvLN(128, 256)
        self.down4 = StridedDoubleConvLN(256, fused_bottleneck)

        # Late fusion at 16²
        self.fuse = DoubleConvLN(fused_bottleneck + fused_token, fused_bottleneck)

        # === Seg decoder (kept for output shape compat — but NOT supervised) ===
        self.s_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.s_c4  = DoubleConvLN(256 + 256, 256)
        self.s_up3 = UpsampleBlockLN(256, 128)
        self.s_c3  = DoubleConvLN(128 + 128, 128)
        self.s_up2 = UpsampleBlockLN(128, 64)
        self.s_c2  = DoubleConvLN(64 + 64, 64)
        self.s_up1 = UpsampleBlockLN(64, 32)
        self.s_c1  = DoubleConvLN(32 + 32, 32)
        self.seg_head = nn.Conv2d(32, 3, kernel_size=1)
        # Class-prior bias init — keeps seg output at prior (sparse, near 0)
        # so the model doesn't waste capacity on unsupervised seg
        nn.init.normal_(self.seg_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.seg_head.bias[0], SEG_LOGIT_PRIOR_BUILDING)
        nn.init.constant_(self.seg_head.bias[1], SEG_LOGIT_PRIOR_VEGETATION)
        nn.init.constant_(self.seg_head.bias[2], SEG_LOGIT_PRIOR_WATER)

        # === Height decoder (FULLY DECOUPLED, same as a05) ===
        self.h_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.h_c4  = DoubleConvLN(256 + 256, 256)
        self.h_up3 = UpsampleBlockLN(256, 128)
        self.h_c3  = DoubleConvLN(128 + 128, 128)
        self.h_up2 = UpsampleBlockLN(128, 64)
        self.h_c2  = DoubleConvLN(64 + 64, 64)
        self.h_up1 = UpsampleBlockLN(64, 32)
        self.h_c1  = DoubleConvLN(32 + 32, 32)
        self.height_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_dense, tokens, building_mask=None):
        if building_mask is None:
            raise ValueError(
                "AdapterFusionLateFusionUNetDecoupledHeightMask requires "
                "`building_mask` (B, 1, H, W)."
            )

        # Adapt dense + mask as 3 sources
        dense_parts = []
        for i, ad in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(ad(x_dense[:, s:e]))
        dense_parts.append(self.mask_adapter(building_mask))
        dense_in = torch.cat(dense_parts, dim=1)

        # Adapt tokens
        token_parts = []
        for i, ad in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(ad(tokens[:, s:e]))
        token_feats = torch.cat(token_parts, dim=1)

        # Encoder
        x1 = self.inc(dense_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Late fusion at 1/16
        fused = torch.cat([x5, token_feats], dim=1)
        fused = self.fuse(fused)

        # Seg decoder (output NOT supervised — keeps prior values via bias init)
        s = self.s_up4(fused)
        s = self.s_c4(torch.cat([x4, s], dim=1))
        s = self.s_up3(s)
        s = self.s_c3(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_c2(torch.cat([x2, s], dim=1))
        s = self.s_up1(s)
        s = self.s_c1(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)

        # Height decoder
        h = self.h_up4(fused)
        h = self.h_c4(torch.cat([x4, h], dim=1))
        h = self.h_up3(h)
        h = self.h_c3(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_c2(torch.cat([x2, h], dim=1))
        h = self.h_up1(h)
        h = self.h_c1(torch.cat([x1, h], dim=1))
        height_logit = self.height_head(h)

        return torch.cat([seg_logits, height_logit], dim=1)


class AdapterFusionLateFusionUNetBVHeight(nn.Module):
    """Joint Building + Vegetation HEIGHT regression specialist with prior-residual.

    Differences vs AdapterFusionLateFusionUNetDecoupledHeightMask (single-class):
      1. **2-ch mask input** (bld_mask, veg_mask). MaskAdapter is 2→adapter_out.
      2. **Prior-residual height**: model's height_head outputs a raw delta
         (can be ±). At forward time, the prior is added based on input mask:
              h_pred_norm = (delta + prior_h_b*bld_m + prior_h_v*veg_m).clamp(0, 1.5)
         where bld_m/veg_m are the INPUT mask channels (binary 0/1).
      3. **output_height_pre_activated=True**: GeoFMNet.activate() skips softplus
         (the body already added priors and clipped).
      4. **Loss**: BuildingVegHeightOnlyLoss supervises h at BOTH GT bld AND
         GT veg pixels (ignore_bg). Background height unsupervised.

    Why this design:
      - Building height (mean 3.75m) and veg height (mean 8.82m) have very
        different baselines. Single h_head learning both classes jointly
        is hard without prior. With prior, model only learns delta from class
        mean, smaller-range target → easier convergence.
      - ignore_bg matches LB metric (rmse only at GT fg pixels).
      - Joint training reuses encoder/decoder capacity efficiently.

    Output ch3 semantics:
      - At input bld_m=1 pixels: prior_h_b + delta — building height
      - At input veg_m=1 pixels: prior_h_v + delta — veg height
      - At input bg pixels (mask=0): delta only (clamped to 0) — bg height
    """
    is_late_fusion = True
    requires_building_mask = True
    output_height_pre_activated = True   # tells GeoFMNet.activate() to skip softplus

    def __init__(self, dense_channels=(64, 128), token_channels=(768, 768, 768, 768),
                 out_channels=4, adapter_out=64, fused_bottleneck=384):
        super().__init__()
        assert out_channels == 4
        self.dense_channels = tuple(dense_channels)
        self.token_channels = tuple(token_channels)
        self.adapter_out = adapter_out

        self.dense_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in dense_channels
        ])
        # 2-ch mask adapter (bld_mask + veg_mask)
        self.mask_adapter = MaskAdapter(adapter_out, in_channels=2)
        self.token_adapters = nn.ModuleList([
            SourceAdapterStemLN(c, adapter_out) for c in token_channels
        ])
        fused_dense = adapter_out * (len(dense_channels) + 1)
        fused_token = adapter_out * len(token_channels)

        self.dense_offsets = [0]
        for c in dense_channels:
            self.dense_offsets.append(self.dense_offsets[-1] + c)
        self.token_offsets = [0]
        for c in token_channels:
            self.token_offsets.append(self.token_offsets[-1] + c)

        # Same encoder as a05 Decoupled
        self.inc   = DoubleConvLN(fused_dense, 32)
        self.down1 = StridedDoubleConvLN(32, 64)
        self.down2 = StridedDoubleConvLN(64, 128)
        self.down3 = StridedDoubleConvLN(128, 256)
        self.down4 = StridedDoubleConvLN(256, fused_bottleneck)

        # Late fusion at 16²
        self.fuse = DoubleConvLN(fused_bottleneck + fused_token, fused_bottleneck)

        # Seg decoder (NOT supervised — kept for output shape compat)
        self.s_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.s_c4  = DoubleConvLN(256 + 256, 256)
        self.s_up3 = UpsampleBlockLN(256, 128)
        self.s_c3  = DoubleConvLN(128 + 128, 128)
        self.s_up2 = UpsampleBlockLN(128, 64)
        self.s_c2  = DoubleConvLN(64 + 64, 64)
        self.s_up1 = UpsampleBlockLN(64, 32)
        self.s_c1  = DoubleConvLN(32 + 32, 32)
        self.seg_head = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.normal_(self.seg_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.seg_head.bias[0], SEG_LOGIT_PRIOR_BUILDING)
        nn.init.constant_(self.seg_head.bias[1], SEG_LOGIT_PRIOR_VEGETATION)
        nn.init.constant_(self.seg_head.bias[2], SEG_LOGIT_PRIOR_WATER)

        # Height decoder (decoupled — same as a05)
        self.h_up4 = UpsampleBlockLN(fused_bottleneck, 256)
        self.h_c4  = DoubleConvLN(256 + 256, 256)
        self.h_up3 = UpsampleBlockLN(256, 128)
        self.h_c3  = DoubleConvLN(128 + 128, 128)
        self.h_up2 = UpsampleBlockLN(128, 64)
        self.h_c2  = DoubleConvLN(64 + 64, 64)
        self.h_up1 = UpsampleBlockLN(64, 32)
        self.h_c1  = DoubleConvLN(32 + 32, 32)
        # height_head outputs delta (raw, ±) — prior added in forward
        self.height_head = nn.Conv2d(32, 1, kernel_size=1)
        # Init bias near 0 so initial delta ≈ 0 → h_pred ≈ prior_at_pixel
        nn.init.constant_(self.height_head.bias, 0.0)
        nn.init.normal_(self.height_head.weight, mean=0.0, std=0.001)

        # Priors as buffers (normalized space: real height / height_norm_constant=30)
        self.register_buffer("prior_h_b", torch.tensor(PRIOR_H_B_NORMALIZED))
        self.register_buffer("prior_h_v", torch.tensor(PRIOR_H_V_NORMALIZED))

    def forward(self, x_dense, tokens, building_mask=None):
        if building_mask is None:
            raise ValueError("AdapterFusionLateFusionUNetBVHeight requires 2-ch building_mask.")
        if building_mask.shape[1] != 2:
            raise ValueError(
                f"building_mask must be (B, 2, H, W) [bld, veg]; got channel count {building_mask.shape[1]}."
            )

        # Adapt dense + mask
        dense_parts = []
        for i, ad in enumerate(self.dense_adapters):
            s, e = self.dense_offsets[i], self.dense_offsets[i + 1]
            dense_parts.append(ad(x_dense[:, s:e]))
        dense_parts.append(self.mask_adapter(building_mask))
        dense_in = torch.cat(dense_parts, dim=1)

        # Adapt tokens
        token_parts = []
        for i, ad in enumerate(self.token_adapters):
            s, e = self.token_offsets[i], self.token_offsets[i + 1]
            token_parts.append(ad(tokens[:, s:e]))
        token_feats = torch.cat(token_parts, dim=1)

        # Encoder
        x1 = self.inc(dense_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Late fusion at 1/16
        fused = torch.cat([x5, token_feats], dim=1)
        fused = self.fuse(fused)

        # Seg decoder (unsupervised; produces prior-init outputs)
        s = self.s_up4(fused)
        s = self.s_c4(torch.cat([x4, s], dim=1))
        s = self.s_up3(s)
        s = self.s_c3(torch.cat([x3, s], dim=1))
        s = self.s_up2(s)
        s = self.s_c2(torch.cat([x2, s], dim=1))
        s = self.s_up1(s)
        s = self.s_c1(torch.cat([x1, s], dim=1))
        seg_logits = self.seg_head(s)

        # Height decoder
        h = self.h_up4(fused)
        h = self.h_c4(torch.cat([x4, h], dim=1))
        h = self.h_up3(h)
        h = self.h_c3(torch.cat([x3, h], dim=1))
        h = self.h_up2(h)
        h = self.h_c2(torch.cat([x2, h], dim=1))
        h = self.h_up1(h)
        h = self.h_c1(torch.cat([x1, h], dim=1))
        delta = self.height_head(h)   # (B, 1, H, W) — RAW raw delta, can be negative

        # Prior-residual: add prior_h_b * bld_m + prior_h_v * veg_m at each pixel
        bld_m = building_mask[:, 0:1]
        veg_m = building_mask[:, 1:2]
        prior_at_pixel = self.prior_h_b * bld_m + self.prior_h_v * veg_m
        h_pred_norm = (delta + prior_at_pixel).clamp(min=0.0, max=1.5)

        # Pre-activated: GeoFMNet.activate() will only clamp (no softplus)
        return torch.cat([seg_logits, h_pred_norm], dim=1)


class LightUNet(nn.Module):
    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = UpsampleBlock(256, 128)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.conv3 = DoubleConv(64, 32)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = self.conv1(torch.cat([x3, x], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x2, x], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x1, x], dim=1))
        return self.head(x)


class StandardUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class EfficientDecoder256Fast(nn.Module):
    def __init__(self, in_channels=768, out_channels=4):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.up1 = StandardUpsampleBlock(256, 128)
        self.up2 = StandardUpsampleBlock(128, 64)
        self.up3 = StandardUpsampleBlock(64, 32)
        self.up4 = StandardUpsampleBlock(32, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


class DPTBuildingSpecialistDINOv3L(nn.Module):
    """Standalone building specialist: DINOv3-L (sat493m) + DPT head.

    - Encoder: DINOv3-L ViT-L/16, sat493m pretrained, NOT frozen (fine-tuned).
    - Decoder: DPT (4-tap from layers `out_indices` of 24-layer ViT-L, default
      [4, 11, 17, 23] 0-indexed). ReassembleLayer scales [×4, ×2, ×1, ×0.5].
    - Input: 640×640 RGB window (40×40 tokens). Train = random 640 crop; eval =
      sliding 640 windows over the full 2560 tile (stitched in GeoFMEmbed2Heights).
    - Output: NATIVE-LABEL-RESOLUTION (B, 4, label_size, label_size) where
      label_size = dpt_label_size (default 64 = 640/10). DPT native res is
      interpolated DIRECTLY to the label grid (the only resize). ch0 = building
      seg logit, ch1/2/3 = zeros.
      Padded to 4-ch so it plugs into the existing GeoFMNet activate path
      and BuildingOnlyLoss(class_idx=0) routing.

    Flags consumed by GeoFMNet/GeoFMEmbed2Heights:
      requires_raw_rgb = True   — wrapper passes raw 3-ch RGB (skip token/mask routing)
      is_late_fusion   = False
      requires_rgb_token = False
      requires_building_mask = False
    """

    is_late_fusion = False
    requires_rgb_token = False
    requires_building_mask = False
    requires_raw_rgb = True

    def __init__(self,
                 dinov3_weights_path,
                 input_size=640,
                 out_indices=(4, 11, 17, 23),
                 patch_size=16,
                 embed_dim=1024,
                 dpt_features=256,
                 dpt_in_shape=(256, 512, 1024, 1024),
                 dpt_label_size=64,
                 out_channels=4):
        super().__init__()
        # Import dinov3 + DPT primitives at construction time (symlinked into module/)
        from module.dinov3.models.vision_transformer import vit_large, load_dinov3_pretrain
        from module.dpt.dpt import DPTFeatureProcessor, DPT

        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.out_indices = tuple(int(i) for i in out_indices)
        self.embed_dim = int(embed_dim)
        self.out_channels_total = int(out_channels)
        # Native-label-resolution output: model always emits a (B, C, label_size,
        # label_size) prediction. With a 640 window → 40×40 tokens → DPT native
        # res, then interpolate down to label_size (64 = 640/10). Hardcoded for the
        # fixed sliding pipeline (every window is 640, every output is 64).
        self.dpt_label_size = int(dpt_label_size)

        # Encoder (fine-tuned)
        self.encoder = vit_large(
            img_size=self.input_size,
            patch_size=self.patch_size,
            layerscale_init=1e-5,
            mask_k_bias=True,
            norm_layer="layernorm",
            ffn_layer="mlp",
        )
        self.encoder.init_weights()
        ok = load_dinov3_pretrain(self.encoder, dinov3_weights_path, verbose=False)
        if not ok:
            raise RuntimeError(f"Failed to load DINOv3 weights from {dinov3_weights_path}")
        # Encoder fine-tuned: keep requires_grad=True (default)

        # DPT decoder
        self.feat_processor = DPTFeatureProcessor(
            in_channels=self.embed_dim,
            out_channels=list(dpt_in_shape),
            image_size=self.input_size,
            patch_size=self.patch_size,
            readout='project',
        )
        self.dpt = DPT(
            num_classes=1,
            features=int(dpt_features),
            channels_last=False,
            use_bn=False,
            in_shape=list(dpt_in_shape),
        )

    def encoder_params(self):
        return list(self.encoder.parameters())

    def decoder_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def forward(self, x):
        # x: (B, 3, H, W) — assume H == W == self.input_size (640).
        # During DDP train, all ranks see the same H == self.input_size (640).
        # At sliding-window inference, each window is also self.input_size (640).
        b, c, h, w = x.shape
        # Encoder: extract intermediate layers WITH cls token (ReassembleLayer's
        # ProjectReadout handles cls→patch fusion internally).
        features = self.encoder.get_intermediate_layers(
            x=x, n=self.out_indices, reshape=False, return_class_token=True,
        )
        # features: tuple of 4 × (patches:(B,N,D), cls:(B,D))
        token_feats = []
        for patches, cls in features:
            tf = torch.cat([cls.unsqueeze(1), patches], dim=1)   # (B, N+1, D)
            token_feats.append(tf)
        feature_maps = self.feat_processor(token_feats)          # 4 maps at different scales
        out = self.dpt(feature_maps)                              # (B, 1, ~H/4, ~W/4)
        # NATIVE-LABEL-RESOLUTION supervision: interpolate DPT native res directly
        # to label_size (64), NOT to the 640 input res. A 640 RGB window maps to a
        # 64×64 label tile (10:1). This is the only resize and it lands on the GT grid.
        ls = self.dpt_label_size
        out = F.interpolate(out, size=(ls, ls), mode='bilinear', align_corners=False)
        # Pad to (B, out_channels_total, ls, ls): ch0 = building logit, ch1+ = zeros.
        if self.out_channels_total > 1:
            zeros = torch.zeros(b, self.out_channels_total - 1, ls, ls,
                                device=out.device, dtype=out.dtype)
            out = torch.cat([out, zeros], dim=1)
        return out


class DPTMultiTaskDINOv3L(nn.Module):
    """MULTI-TASK RGB-only: DINOv3-L (sat493m) FULLY FINE-TUNED + DPT head → 4 real channels.

    Sibling of DPTBuildingSpecialistDINOv3L, but the DPT emits ALL 4 multi-task
    channels (ch0 bld / ch1 veg / ch2 water seg logits + ch3 height logit) instead
    of a single building logit + zero pad. Trained with the `mt_hrnet` multi-task
    loss; eval via the same sliding-window stitch in GeoFMEmbed2Heights.

    - Encoder: DINOv3-L ViT-L/16, sat493m pretrained, NOT frozen — FULL fine-tune.
      3-tier discriminative LR (head / enc_top / enc_bot) is applied automatically
      by GeoFMEmbed2Heights.custom_param_groups (keys on requires_raw_rgb +
      `encoder.` prefix), so the backbone trains at a low LR while the DPT head
      trains fast. This is the "全量微调 DINOv3" route (contrast: the frozen-cache
      RGB-only HRNet model AdapterMultiTaskLiteHRNetRGBOnly).
    - Decoder: DPT 4-tap (layers `out_indices` of the 24-layer ViT-L), num_classes=4.
    - Input 640×640 RGB window → 40×40 tokens → DPT native res → interpolate to
      dpt_label_size (64 = 640/10, native label grid).
    - Output (B, 4, label_size, label_size), ALL channels real (no zero pad).

    Flags: requires_raw_rgb=True → wrapper passes raw 3-ch RGB and uses the
    sliding-window eval path. GeoFMNet.activate does sigmoid(ch0/1/2)+softplus(ch3).
    """

    is_late_fusion = False
    requires_rgb_token = False
    requires_building_mask = False
    requires_raw_rgb = True

    def __init__(self,
                 dinov3_weights_path,
                 input_size=640,
                 out_indices=(4, 11, 17, 23),
                 patch_size=16,
                 embed_dim=1024,
                 dpt_features=256,
                 dpt_in_shape=(256, 512, 1024, 1024),
                 dpt_label_size=64,
                 out_channels=4):
        super().__init__()
        from module.dinov3.models.vision_transformer import vit_large, load_dinov3_pretrain
        from module.dpt.dpt import DPTFeatureProcessor, DPT

        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.out_indices = tuple(int(i) for i in out_indices)
        self.embed_dim = int(embed_dim)
        self.out_channels_total = int(out_channels)   # 4 — all real (used by sliding eval)
        self.dpt_label_size = int(dpt_label_size)

        # Encoder (fine-tuned — requires_grad stays True)
        self.encoder = vit_large(
            img_size=self.input_size,
            patch_size=self.patch_size,
            layerscale_init=1e-5,
            mask_k_bias=True,
            norm_layer="layernorm",
            ffn_layer="mlp",
        )
        self.encoder.init_weights()
        ok = load_dinov3_pretrain(self.encoder, dinov3_weights_path, verbose=False)
        if not ok:
            raise RuntimeError(f"Failed to load DINOv3 weights from {dinov3_weights_path}")

        # DPT decoder — num_classes = 4 (the full multi-task output).
        self.feat_processor = DPTFeatureProcessor(
            in_channels=self.embed_dim,
            out_channels=list(dpt_in_shape),
            image_size=self.input_size,
            patch_size=self.patch_size,
            readout='project',
        )
        self.dpt = DPT(
            num_classes=self.out_channels_total,        # 4 real channels
            features=int(dpt_features),
            channels_last=False,
            use_bn=False,
            in_shape=list(dpt_in_shape),
        )

    def encoder_params(self):
        return list(self.encoder.parameters())

    def decoder_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def forward(self, x):
        # x: (B, 3, H, W), H == W == input_size (640) for both train crop and eval window.
        b, c, h, w = x.shape
        features = self.encoder.get_intermediate_layers(
            x=x, n=self.out_indices, reshape=False, return_class_token=True,
        )
        token_feats = []
        for patches, cls in features:
            tf = torch.cat([cls.unsqueeze(1), patches], dim=1)   # (B, N+1, D)
            token_feats.append(tf)
        feature_maps = self.feat_processor(token_feats)
        out = self.dpt(feature_maps)                              # (B, 4, ~H/4, ~W/4)
        ls = self.dpt_label_size
        out = F.interpolate(out, size=(ls, ls), mode='bilinear', align_corners=False)
        return out                                                # (B, 4, ls, ls) all real


def _bn_to_gn(model, num_groups=32):
    """Swap all nn.BatchNorm2d modules in `model` for nn.GroupNorm in-place.

    Reason: bs=2 makes BN's batch statistics noisy, and BN.eval() switches to
    running mean/var that may have drifted from val/test distribution. GN is
    per-sample, batch-independent, and has no train/eval mode shift.
    `num_groups` is capped to min(num_groups, channels) — if num_features<32 we
    fall back to a smaller group count.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            c = module.num_features
            g = min(num_groups, c)
            # Ensure c % g == 0; walk down until divisible
            while c % g != 0 and g > 1:
                g -= 1
            new_norm = nn.GroupNorm(num_groups=g, num_channels=c)
            if "." in name:
                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name
            setattr(parent, child_name, new_norm)
    return model


class GeoFMNet(nn.Module):
    def __init__(self, in_channels, out_channels=4, model_type="auto", height_activation="softplus",
                 source_channels=None, use_groupnorm=False,
                 dense_channels=None, token_channels=None,
                 adapter_out=64, fused_bottleneck=384,
                 encoder_dims=(96, 192, 384, 768),
                 encoder_depths=(3, 3, 9, 3),
                 drop_path_rate=0.1,
                 efficientnet_pretrained=True,
                 pretrained_mit_b0=None,
                 rgb_token_channels=None,
                 rgb_modality_dropout=None,
                 dinov3_weights_path=None,
                 dpt_input_size=None,
                 dpt_out_indices=None,
                 dpt_features=None,
                 dpt_in_shape=None,
                 dpt_label_size=None):
        super().__init__()
        selected = infer_model_type(in_channels) if model_type == "auto" else model_type
        # Pass source_channels only to adapter-fusion bodies (others ignore it).
        kw = {"source_channels": tuple(source_channels)} if source_channels is not None else {}
        if selected == "lightunet":
            self.body = LightUNet(in_channels, out_channels)
        elif selected == "decoder_residual":
            self.body = EfficientDecoder256Fast(in_channels, out_channels)
        elif selected == "cascade_dual":
            self.body = CascadeDualDecoderUNet(in_channels, out_channels)
        elif selected == "dual_only":
            self.body = DualDecoderOnlyUNet(in_channels, out_channels)
        elif selected == "adapter_fusion_cascade":
            self.body = AdapterFusionCascadeUNet(in_channels, out_channels, **kw)
        elif selected == "adapter_fusion_dual_only":
            self.body = AdapterFusionDualOnlyUNet(in_channels, out_channels, **kw)
        elif selected == "cascade_dual_v2":
            self.body = CascadeDualHeadV2(in_channels, out_channels)
        elif selected == "adapter_fusion_cascade_v2":
            self.body = AdapterFusionCascadeV2(in_channels, out_channels, **kw)
        elif selected == "adapter_fusion_lite_hrnet":
            self.body = AdapterFusionLiteHRNet(in_channels, out_channels, **kw)
        elif selected == "adapter_fusion_lite_hrnet_dual":
            self.body = AdapterFusionLiteHRNetDual(in_channels, out_channels, **kw)
        elif selected == "adapter_fusion_lite_hrnet_heavy":
            self.body = AdapterFusionLiteHRNetHeavy(in_channels, out_channels, **kw)
        elif selected == "adapter_fusion_lite_hrnet_token_fusion":
            # 5-branch Lite-HRNet that ingests token data at 1/16.
            # Requires `dense_channels` and `token_channels` (via top-level kwargs)
            # just like adapter_fusion_late_multi. GeoFMEmbed2Heights routes y["tokens"].
            self.body = AdapterFusionLiteHRNetTokenFusion(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
            )
        elif selected == "adapter_fusion_lite_hrnet_token_fusion_rgb":
            # Same as adapter_fusion_lite_hrnet_token_fusion + a 5th modality:
            # DINOv3-L sat493m RGB token features (1024 ch) at 120/160 spatial.
            # GeoFMEmbed2Heights routes y["rgb_token"] and y["has_rgb"].
            self.body = AdapterFusionLiteHRNetTokenFusionRGB(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                rgb_token_channels=rgb_token_channels if rgb_token_channels is not None else 1024,
                out_channels=out_channels,
                adapter_out=adapter_out,
                rgb_modality_dropout=rgb_modality_dropout if rgb_modality_dropout is not None else 0.10,
            )
        elif selected == "adapter_mt_lite_hrnet_rgb_only":
            # RGB-ONLY multi-task: DINOv3-L sat493m RGB token features are the sole
            # input (no AE/Tessera dense, no TerraMind/THOR tokens). Same 4-ch MT
            # output + same loader contract as the `_rgb` fusion body (it ignores
            # the dense/token content, uses only x_dense.shape for output size).
            self.body = AdapterMultiTaskLiteHRNetRGBOnly(
                rgb_token_channels=rgb_token_channels if rgb_token_channels is not None else 1024,
                out_channels=out_channels,
                adapter_out=adapter_out,
                rgb_modality_dropout=rgb_modality_dropout if rgb_modality_dropout is not None else 0.0,
            )
        elif selected == "dpt_dinov3l_building_specialist":
            # Standalone building specialist: DINOv3-L (sat493m, fine-tuned) + DPT head.
            # Pure RGB input, no GeoFM tokens. Output 4-ch (ch0 = building seg logit).
            # GeoFMEmbed2Heights routes raw image as RGB to body (skip tokens/mask/rgb_token).
            if dinov3_weights_path is None:
                raise ValueError("dpt_dinov3l_building_specialist requires dinov3_weights_path")
            self.body = DPTBuildingSpecialistDINOv3L(
                dinov3_weights_path=dinov3_weights_path,
                input_size=int(dpt_input_size) if dpt_input_size is not None else 640,
                out_indices=tuple(dpt_out_indices) if dpt_out_indices is not None else (4, 11, 17, 23),
                patch_size=16,
                embed_dim=1024,
                dpt_features=int(dpt_features) if dpt_features is not None else 256,
                dpt_in_shape=tuple(dpt_in_shape) if dpt_in_shape is not None else (256, 512, 1024, 1024),
                dpt_label_size=int(dpt_label_size) if dpt_label_size is not None else 64,
                out_channels=out_channels,
            )
        elif selected == "dpt_dinov3l_multitask":
            # MULTI-TASK RGB-only: DINOv3-L (sat493m) FULLY FINE-TUNED + DPT head,
            # all 4 channels real (bld/veg/water seg + height). Raw RGB input,
            # sliding-window eval. 3-tier discriminative LR via custom_param_groups.
            if dinov3_weights_path is None:
                raise ValueError("dpt_dinov3l_multitask requires dinov3_weights_path")
            self.body = DPTMultiTaskDINOv3L(
                dinov3_weights_path=dinov3_weights_path,
                input_size=int(dpt_input_size) if dpt_input_size is not None else 640,
                out_indices=tuple(dpt_out_indices) if dpt_out_indices is not None else (4, 11, 17, 23),
                patch_size=16,
                embed_dim=1024,
                dpt_features=int(dpt_features) if dpt_features is not None else 256,
                dpt_in_shape=tuple(dpt_in_shape) if dpt_in_shape is not None else (256, 512, 1024, 1024),
                dpt_label_size=int(dpt_label_size) if dpt_label_size is not None else 64,
                out_channels=out_channels,
            )
        elif selected == "adapter_fusion_lite_hrnet_dense_upsample":
            # Dense-only HRNet that GPU-upsamples token sources to dense H,W in
            # forward. Tokens received via meta["tokens"] (is_late_fusion=True).
            # Avoids CPU OOM that DataLoader-side upsample triggered (855 MB/sample
            # × prefetch × workers × ranks > node RAM).
            self.body = AdapterFusionLiteHRNetDenseUpsample(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
            )
        elif selected == "adapter_fusion_mit_b0_token_fusion":
            # SegFormer MiT-B0 transformer-encoder cousin of the HRNet specialist.
            # Same input contract (dense_channels, token_channels, adapter_out);
            # token fusion happens at the 1/16 stage-3 output of the MiT encoder.
            # V2: OS=1 progressive decoder with hi-res skip stems (see
            # MitB0BodyTokenFusion docstring). Optional `pretrained_mit_b0`
            # path loads ImageNet MiT-B0 weights into stages 2-4 and blocks/norms
            # (patch_embed1 stays random-init because dense_in_channels != 3).
            self.body = AdapterFusionMitB0TokenFusion(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                pretrained_mit_b0=pretrained_mit_b0,
            )
        elif selected == "adapter_fusion_mit_b0_hrnet_decoder":
            # V3: MiT-B0 encoder + byte-for-byte copy of LiteHRNetBodyTokenFusion's
            # decoder (no stems, no progressive upsample, no LN — BN+ReLU+DoubleConv).
            # Strict encoder-only swap of the HRNet specialist for clean ablation.
            self.body = AdapterFusionMitB0HRNetDecoder(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                pretrained_mit_b0=pretrained_mit_b0,
            )
        elif selected == "adapter_fusion_mit_b0_unetplusplus":
            # NEW: MiT-B0 encoder + UNet++ nested decoder + token fusion at 1/16.
            # 6 nested DoubleConvLN blocks (X[2,1], X[1,1], X[1,2], X[0,1..3])
            # + progressive OS=4 → OS=2 → OS=1 with hi-res stem skips.
            # `pretrained_mit_b0` loads NVIDIA ImageNet MiT-B0 weights into encoder.
            self.body = AdapterFusionMitB0UNetPlusPlus(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                pretrained_mit_b0=pretrained_mit_b0,
            )
        elif selected == "adapter_fusion_lite_hrnet_mit_token_fusion":
            # V4: full HRNet pipeline (stem, transitions, token_fuse, 4-branch HR
            # stages with cross-resolution fusion, decoder) unchanged — only the
            # per-branch ShuffleBlock processor is swapped for a stack of MiT
            # blocks (SR attention, DWConv-MLP). Per-branch num_heads/sr_ratio
            # scale with branch resolution so each branch's K,V token count is
            # ~24² at H=192 training. Uses sdpa for branch 0's 192² queries.
            self.body = AdapterFusionLiteHRNetMiTTokenFusion(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
            )
        elif selected == "adapter_fusion_lite_hrnet_convnext_token_fusion":
            # V7: same HRNet pipeline as V4 — only the per-branch ShuffleBlock
            # is swapped for ConvNeXt blocks (DwConv 7×7 + LN + 4× MLP +
            # LayerScale=1.0 for scratch). 4-branch cross-resolution fusion
            # and HRNet decoder kept identical.
            self.body = AdapterFusionLiteHRNetConvNeXtTokenFusion(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
            )
        elif selected == "adapter_fusion_lite_hrnet_mbconv_token_fusion":
            # V8: same HRNet pipeline as V4 — only the per-branch ShuffleBlock
            # is swapped for EfficientNet-B0 MBConv blocks (1×1 expand×6 +
            # DwConv k=3 + SE r=0.25 + 1×1 project + StochasticDepth).
            self.body = AdapterFusionLiteHRNetMBConvTokenFusion(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
            )
        elif selected == "adapter_fusion_convnext_unetformer":
            # 3rd architecture family: ConvNeXt-T encoder + UNetFormer decoder
            # with Global-Local Transformer blocks. Token fusion at 1/16 stage.
            # Designed for ensemble diversity vs UNet late-fusion (a05) and HRNet
            # token-fusion. Default depths/dims = ConvNeXt-Tiny.
            # NOTE: from-scratch failed 3 times. Use only with pretrained weights.
            self.body = AdapterFusionConvNeXtUNetFormer(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                encoder_dims=tuple(encoder_dims),
                encoder_depths=tuple(encoder_depths),
                drop_path_rate=float(drop_path_rate),
            )
        elif selected == "adapter_fusion_efficientnet_unetformer":
            # 4th architecture family: EfficientNet-B0 (ImageNet PRETRAINED) encoder
            # + UNetFormer decoder. Replaces failed from-scratch ConvNeXt with a
            # BN-heavy, MBConv-based encoder that has ImageNet semantic priors.
            # Token fusion at 1/16. Pyramid channels are fixed (24,40,112,320 — B0).
            self.body = AdapterFusionEfficientNetUNetFormer(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                pretrained=bool(efficientnet_pretrained),
                drop_path_rate=float(drop_path_rate),
            )
        elif selected == "adapter_fusion_efficientnet_unetformer_height_mask":
            # Building HEIGHT regression specialist with binary mask input.
            # Same backbone as adapter_fusion_efficientnet_unetformer but body
            # takes 3rd dense source (mask 1ch). Forward requires `building_mask`
            # kwarg (B, 1, H, W). See AdapterFusionEfficientNetUNetFormerHeightMask.
            self.body = AdapterFusionEfficientNetUNetFormerHeightMask(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                pretrained=bool(efficientnet_pretrained),
                drop_path_rate=float(drop_path_rate),
            )
        elif selected == "adapter_fusion_late_multi_decoupled_height_mask":
            # Building HEIGHT regression specialist with a05's UNet encoder + mask.
            # Same encoder/decoder as a05 (proven to converge fast) + MaskAdapter
            # as 3rd dense source. Forward requires `building_mask` kwarg.
            self.body = AdapterFusionLateFusionUNetDecoupledHeightMask(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                fused_bottleneck=fused_bottleneck,
            )
        elif selected == "adapter_fusion_late_multi_decoupled_bv_height":
            # Joint Building + Vegetation HEIGHT regression specialist (prior-residual + ignore_bg).
            # 2-ch mask input (bld_mask, veg_mask), prior_h_b/prior_h_v added per-pixel,
            # supervises height at both class GT pixels (ignore_bg).
            self.body = AdapterFusionLateFusionUNetBVHeight(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                fused_bottleneck=fused_bottleneck,
            )
        elif selected == "adapter_fusion_veg_specialist":
            # Single-class (vegetation) specialist with dual decoders.
            # Output is (B, 2, H, W) instead of (B, 4, H, W). out_channels=2 required.
            self.body = AdapterFusionVegSpecialist(in_channels, out_channels, **kw)
        elif selected == "adapter_fusion_building_specialist":
            # Single-class (building) specialist — same dual decoder architecture as veg.
            # Differs only in target slicing (handled by GeoFMEmbed2Heights).
            self.body = AdapterFusionVegSpecialist(in_channels, out_channels, **kw)
        elif selected == "adapter_fusion_water_specialist":
            # Single-class (water) specialist — same dual decoder architecture as veg/bld.
            # water is even sparser than building (1.1% positive),
            # and water height GT is trivially 0 (water surface = ground).
            self.body = AdapterFusionVegSpecialist(in_channels, out_channels, **kw)
        elif selected == "adapter_fusion_late_multi":
            # Multi-source late-fusion: dense (AE, Tessera) + token (TerraMind, THOR)
            # fuse at 16² bottleneck. Requires `dense_channels` and `token_channels`
            # to be passed via top-level kwargs (not via source_channels). The
            # wrapper GeoFMEmbed2Heights checks `is_late_fusion=True` and routes
            # y["tokens"] as a separate input.
            self.body = AdapterFusionLateFusionUNet(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                fused_bottleneck=fused_bottleneck,
            )
        elif selected == "adapter_fusion_late_multi_decoupled":
            # Same as late_multi but height path is FULLY decoupled from seg
            # (no soft-mix, no per-class h_b/h_v, no detach gating).
            # Tests whether the V2 soft-mix design is actually beneficial vs.
            # a simple 1-channel direct height head.
            self.body = AdapterFusionLateFusionUNetDecoupled(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                fused_bottleneck=fused_bottleneck,
            )
        elif selected == "adapter_fusion_late_multi_decoupled_ce":
            # CE seg-specialist (Plan A): same architecture as decoupled UNet,
            # but seg_head outputs 4 classes (bld/veg/water/bg) for softmax CE.
            # Output: (B, 5, H, W) = [4 raw seg logits, 1 height logit].
            # Loss: SoftmaxCE4ClassLoss (see module/losses.py).
            # Activate: at inference, softmax over seg + softplus on height.
            self.body = AdapterFusionLateFusionUNetDecoupledCE(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                fused_bottleneck=fused_bottleneck,
            )
        elif selected == "adapter_fusion_late_multi_decoupled_raw_height":
            # Raw-height variant of a05: identical arch but height_head outputs
            # raw (no softplus). Matches official baseline design.
            # `output_height_pre_activated=True` set on body → activate() clamps
            # height to [0, 15] without softplus, allows gradient flow at h=0.
            self.body = AdapterFusionLateFusionUNetDecoupledRawHeight(
                dense_channels=tuple(dense_channels) if dense_channels is not None else (64, 128),
                token_channels=tuple(token_channels) if token_channels is not None else (768, 768, 768, 768),
                out_channels=out_channels,
                adapter_out=adapter_out,
                fused_bottleneck=fused_bottleneck,
            )
        else:
            raise ValueError(
                f"Unknown model_type={model_type}. Use auto, lightunet, "
                f"decoder_residual, cascade_dual, dual_only, adapter_fusion_cascade, "
                f"adapter_fusion_dual_only, cascade_dual_v2, or adapter_fusion_cascade_v2."
            )

        # Optional: swap all BatchNorm2d → GroupNorm. Done AFTER body construction
        # so the swap covers every nn.BatchNorm2d the body created (DoubleConv,
        # ShuffleBlock, HRStage fusion, ...) without per-class plumbing.
        if use_groupnorm:
            _bn_to_gn(self)

        self.model_type = selected
        self.height_activation = height_activation
        self.softplus = nn.Softplus()

    def forward(self, x, tokens=None, building_mask=None, rgb_token=None, has_rgb=None):
        """For late-fusion models, pass tokens too. For mask-conditioned height
        specialists, also pass building_mask=(B, 1, H, W). For RGB-fused bodies
        (requires_rgb_token=True), pass rgb_token=(B, 1024, h, w) and has_rgb=(B,)
        bool. For others, ignore.

        For raw-RGB specialist bodies (requires_raw_rgb=True), the wrapper passes
        the 3-channel RGB directly via `x` and skips all other modalities.
        """
        if getattr(self.body, "requires_raw_rgb", False):
            return self.body(x)
        if getattr(self.body, "is_late_fusion", False):
            if tokens is None:
                raise ValueError(
                    "Late-fusion body needs tokens=... (a B×C×16×16 tensor). "
                    "Caller (e.g. GeoFMEmbed2Heights) must extract from meta['tokens']."
                )
            if getattr(self.body, "requires_rgb_token", False):
                if rgb_token is None:
                    raise ValueError(
                        "RGB-fused body requires rgb_token=(B, 1024, h, w). "
                        "Caller must extract from meta['rgb_token']."
                    )
                return self.body(x, tokens, rgb_token=rgb_token, has_rgb=has_rgb)
            if getattr(self.body, "requires_building_mask", False):
                if building_mask is None:
                    raise ValueError(
                        "Body requires building_mask=(B, 1, H, W). Caller must "
                        "extract from meta['building_mask']."
                    )
                return self.body(x, tokens, building_mask=building_mask)
            return self.body(x, tokens)
        return self.body(x)

    def activate(self, raw):
        # Veg specialist outputs (B, 2, H, W) = [seg_logit, height_logit].
        # Activate same way: seg→sigmoid, height→softplus, then clamp.
        if raw.shape[1] == 2:
            seg = torch.sigmoid(raw[:, :1])
            height_raw = torch.clamp(raw[:, 1:2], max=30.0)
            height = self.softplus(height_raw) if self.height_activation != "relu" else torch.relu(height_raw)
            return torch.cat([seg, height], dim=1)

        # CE 4-class specialist outputs (B, 5, H, W) = [4 raw seg logits, 1 height logit].
        # Activate: softmax over 4 seg classes, softplus on height. Return (B, 5, H, W)
        # — caller decides whether to drop bg (eval path drops it for (B, 4, H, W) compat).
        if getattr(self.body, "is_ce_seg", False) and raw.shape[1] == 5:
            seg = torch.softmax(raw[:, :4], dim=1)
            height_raw = torch.clamp(raw[:, 4:5], max=30.0)
            height = self.softplus(height_raw) if self.height_activation != "relu" else torch.relu(height_raw)
            return torch.cat([seg, height], dim=1)

        landcover = torch.sigmoid(raw[:, :3])

        # V2 models: forward() already produced final normalized height via
        # prior+residual+soft-mix (NOT a logit). Just clip to valid range.
        # NEW: raw_height_allow_negative=True (e.g. raw_height variant) skips
        # the min=0 clamp so negative outputs preserve gradient flow during
        # training (matches official baseline). Inference clamping is the
        # downstream tool's responsibility.
        if getattr(self.body, "output_height_pre_activated", False):
            if getattr(self.body, "raw_height_allow_negative", False):
                height = torch.clamp(raw[:, 3:4], max=15.0)   # no min clamp
            else:
                height = torch.clamp(raw[:, 3:4], min=0.0, max=15.0)
            return torch.cat([landcover, height], dim=1)

        # Original models: forward() returned a logit. Apply softplus.
        height_raw = torch.clamp(raw[:, 3:4], max=30.0)
        if self.height_activation == "relu":
            height = torch.relu(height_raw)
        else:
            height = self.softplus(height_raw)
        return torch.cat([landcover, height], dim=1)

