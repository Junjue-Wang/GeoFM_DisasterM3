import torch


class RunningGeoFMMetrics:
    def __init__(self, threshold=0.1, height_norm_constant=30.0, eps=1e-6):
        self.threshold = threshold
        self.height_norm_constant = height_norm_constant
        self.eps = eps
        self.intersections = torch.zeros(3, dtype=torch.float64)
        self.unions = torch.zeros(3, dtype=torch.float64)
        self.height_sse = torch.zeros(2, dtype=torch.float64)
        self.height_count = torch.zeros(2, dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds, targets):
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        pred_masks = preds[:, :3] > self.threshold
        true_masks = targets[:, :3] > self.threshold
        for idx in range(3):
            inter = torch.logical_and(pred_masks[:, idx], true_masks[:, idx]).sum()
            union = torch.logical_or(pred_masks[:, idx], true_masks[:, idx]).sum()
            self.intersections[idx] += inter.double()
            self.unions[idx] += union.double()

        pred_h = preds[:, 3] * self.height_norm_constant
        true_h = targets[:, 3] * self.height_norm_constant
        for out_idx, mask_idx in enumerate((0, 1)):
            mask = true_masks[:, mask_idx]
            if mask.any():
                err = pred_h[mask] - true_h[mask]
                self.height_sse[out_idx] += torch.sum(err.double() ** 2)
                self.height_count[out_idx] += mask.sum().double()

    def summary(self):
        iou = self.intersections / torch.clamp(self.unions, min=self.eps)
        rmse = torch.sqrt(self.height_sse / torch.clamp(self.height_count, min=self.eps))
        score = (
            0.25 * iou[0]
            + 0.15 * iou[1]
            + 0.15 * iou[2]
            + 0.25 * (1.0 / (1.0 + rmse[0]))
            + 0.20 * (1.0 / (1.0 + rmse[1]))
        )
        # Official AI4EO 2026 LB score (Ferdinand Schenck, forum 2026-04-18,
        # verified against 3 LB rows 2026-05-18).
        official = (
            0.25 * iou[0]
            + 0.15 * iou[1]
            + 0.15 * iou[2]
            + 0.25 * torch.clamp(1.0 - rmse[0] / 3.0, min=0.0)
            + 0.20 * torch.clamp(1.0 - rmse[1] / 5.0, min=0.0)
        )
        return {
            "miou_buildings": float(iou[0]),
            "miou_vegetation": float(iou[1]),
            "miou_water": float(iou[2]),
            "rmse_buildings": float(rmse[0]),
            "rmse_vegetation": float(rmse[1]),
            "proxy_score": float(score),
            "official_score": float(official),
        }

