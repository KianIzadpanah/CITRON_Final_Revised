"""Pixel-level metrics for overlap mask prediction."""

import torch
import numpy as np


def compute_metrics(pred_prob: torch.Tensor, target: torch.Tensor,
                    threshold: float = 0.5) -> dict[str, float]:
    """
    Args:
        pred_prob: sigmoid probabilities (B, 1, H, W) or (B, H, W)
        target:    binary ground truth (B, 1, H, W) or (B, H, W)
    Returns dict with: dice, iou, precision, recall, f1
    """
    pred_bin = (pred_prob >= threshold).float()
    pred_f = pred_bin.view(-1)
    tgt_f = target.float().view(-1)

    tp = (pred_f * tgt_f).sum()
    fp = (pred_f * (1 - tgt_f)).sum()
    fn = ((1 - pred_f) * tgt_f).sum()
    tn = ((1 - pred_f) * (1 - tgt_f)).sum()

    eps = 1e-6
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def aggregate_metrics(metric_list: list[dict]) -> dict[str, float]:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}
