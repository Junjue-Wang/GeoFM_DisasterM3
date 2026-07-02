import os

import torch
from tqdm import tqdm

import data.geofm
import module.geofm
from module.metrics import RunningGeoFMMetrics

try:
    import ever as er
except ModuleNotFoundError as exc:
    raise SystemExit("EVER is not installed. Install requirements first, then rerun train.py.") from exc


er.registry.register_all()


# Track the best val score seen so far for "save best ckpt" logic.
# Keyed by model_dir so multiple training runs in the same Python process don't
# stomp on each other (rare, but safe). Score is the metric printed on the
# `GeoFM validation` line — for specialists, the single-class IoU; for
# multi-task models, official_score.
_BEST_VAL = {}


def _load_persisted_best(model_dir):
    """Read best_val_info.txt from a prior run/cycle so _BEST_VAL survives
    process restarts (e.g. walltime → pjsub resume). Returns dict or None."""
    info_path = os.path.join(model_dir, "best_val_info.txt")
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path) as f:
            kv = dict(line.strip().split("=") for line in f if "=" in line)
        return {"score": float(kv["best_val_score"]), "step": int(kv["best_val_step"])}
    except Exception:
        return None


def _save_best_ckpt(launcher, score, step):
    """Save current model state to <model_dir>/model-best.pth if this is a
    new high water mark. Overwrites the previous best each time it improves.
    Only rank 0 writes (to avoid concurrent writes on multi-GPU DDP).

    Cross-cycle persistence: on first call after process restart, seed
    _BEST_VAL from <model_dir>/best_val_info.txt if present, so a resumed
    job doesn't overwrite the historical best with a transient lower val."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    model_dir = launcher.model_dir
    key = model_dir
    if key not in _BEST_VAL:
        # First eval in this process — try to seed from disk (resume case)
        persisted = _load_persisted_best(model_dir)
        if persisted is not None:
            _BEST_VAL[key] = persisted
            launcher.logger.info(
                f"[best-ckpt] Resumed prev best val={persisted['score']:.4f} "
                f"at step {persisted['step']} from disk")
    prev = _BEST_VAL.get(key, {"score": float("-inf"), "step": -1})
    if score <= prev["score"]:
        return
    _BEST_VAL[key] = {"score": float(score), "step": int(step)}
    inner = launcher.model.module if hasattr(launcher.model, "module") else launcher.model
    sd = inner.state_dict()
    # Save in same shape as launcher's regular ckpts (top-level state_dict + meta).
    payload = {
        "state_dict": sd,
        "global_step": int(step),
        "val_score": float(score),
        "best_so_far": True,
    }
    best_path = os.path.join(model_dir, "model-best.pth")
    tmp_path = best_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, best_path)  # atomic
    # Also write a tiny info file alongside so it's easy to grep later.
    info_path = os.path.join(model_dir, "best_val_info.txt")
    with open(info_path, "w") as f:
        f.write(f"best_val_score={score:.6f}\nbest_val_step={step}\n")
    launcher.logger.info(
        f"[best-ckpt] NEW BEST val={score:.4f} at step {step}, saved → {best_path}")


def _save_last_ckpt(launcher, score, step):
    """Always overwrite <model_dir>/model-last.pth with the current model state.
    Useful when val data leaks into training (e.g. include_val_in_train=True)
    so save_best can't be trusted — caller should use model-last.pth instead.
    Only rank 0 writes."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    model_dir = launcher.model_dir
    inner = launcher.model.module if hasattr(launcher.model, "module") else launcher.model
    sd = inner.state_dict()
    payload = {
        "state_dict": sd,
        "global_step": int(step),
        "val_score": float(score),
        "last": True,
    }
    last_path = os.path.join(model_dir, "model-last.pth")
    tmp_path = last_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, last_path)
    info_path = os.path.join(model_dir, "last_val_info.txt")
    with open(info_path, "w") as f:
        f.write(f"last_val_score={score:.6f}\nlast_val_step={step}\n")


def evaluate(self, test_dataloader, config=None):
    torch.cuda.empty_cache()
    self.model.eval()
    metric = RunningGeoFMMetrics(threshold=0.1, height_norm_constant=30.0)

    # Single-class specialist models only supervise ONE channel — others random.
    # Detect and switch to single-class IoU instead of the full official_score.
    # Class index from specialist_train_class_idx (0=bld, 1=veg, 2=water).
    inner = self.model.module if hasattr(self.model, "module") else self.model
    is_specialist = bool(getattr(inner, "is_building_only", False) or
                          getattr(inner, "is_baseline_specialist", False))
    train_cls = int(
        getattr(inner, "specialist_train_class_idx", None)
        or getattr(inner, "baseline_class_idx", None)
        or 0
    )
    cls_tp = cls_fp = cls_fn = 0

    with torch.no_grad():
        for image, meta in tqdm(test_dataloader, total=len(test_dataloader), desc="GeoFM val"):
            if "target" not in meta:
                continue
            device = next(self.model.parameters()).device
            image = image.to(device)
            target = meta["target"].to(device)
            # Pass full meta — late-fusion models extract tokens from meta["tokens"].
            # Other models accept y=meta and ignore extra keys.
            pred = self.model(image, meta)
            if is_specialist:
                p = (pred[:, train_cls] > 0.1)
                g = (target[:, train_cls] > 0.1)
                cls_tp += int((p & g).sum().item())
                cls_fp += int((p & ~g).sum().item())
                cls_fn += int((~p & g).sum().item())
            else:
                metric.update(pred, target)

    if is_specialist:
        iou_c = cls_tp / max(cls_tp + cls_fp + cls_fn, 1)
        miou_keys = ("miou_buildings", "miou_vegetation", "miou_water")
        message = f"{miou_keys[train_cls]}: {iou_c:.4f} | official_score: {iou_c:.4f}"
        current_score = iou_c
    else:
        summary = metric.summary()
        message = " | ".join(f"{k}: {v:.4f}" for k, v in summary.items())
        current_score = float(summary.get("official_score", 0.0))
    self.logger.info(f"\nGeoFM validation | {message}")

    # Save best-by-val ckpt to <model_dir>/model-best.pth, AND always overwrite
    # model-last.pth with the current ckpt (use latter when val isn't trustworthy,
    # e.g. include_val_in_train=True).
    try:
        step = int(getattr(self, "_ckpt", None).global_step) if getattr(self, "_ckpt", None) is not None else -1
    except Exception:
        step = -1
    _save_best_ckpt(self, current_score, step)
    _save_last_ckpt(self, current_score, step)

    torch.cuda.empty_cache()


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    seed = int(os.environ.get("GEOFM_SEED", "2333"))
    # Per-rank seed shift so RGB modality dropout (and any future per-sample RNG)
    # draws independent patterns across DDP ranks. LOCAL_RANK env var is set by torchrun.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    seed_rank = seed + local_rank
    torch.manual_seed(seed_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_rank)
    trainer = er.trainer.get_trainer("th_ddp")()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])

