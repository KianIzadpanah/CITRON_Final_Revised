"""
scene_metrics.py — AP / precision / recall computation at scene level.

Uses a straightforward matching approach consistent with COCO/VOC:
    - Sort detections by confidence descending
    - Greedy match to GT by IoU
    - Compute precision and recall at scene level
    - mAP50 = mean AP across classes at IoU=0.5
    - mAP50_95 = mean AP averaged over IoU thresholds 0.5:0.05:0.95
"""

from __future__ import annotations
import numpy as np
from src.detection.detection_utils import box_iou


def _match_dets_to_gt(
    dets: list[dict], gts: list[dict], iou_thresh: float = 0.5, cls_id: int | None = None
) -> tuple[list[int], list[int]]:
    """
    Match detections to ground-truths for a single class.
    Returns (tp_flags, fp_flags) parallel to sorted dets.
    """
    if cls_id is not None:
        dets = [d for d in dets if d["cls"] == cls_id]
        gts = [g for g in gts if g["cls"] == cls_id]

    dets_sorted = sorted(dets, key=lambda d: -d["conf"])
    matched = [False] * len(gts)
    tp_flags, fp_flags = [], []

    for d in dets_sorted:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            iou = box_iou(d, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp_flags.append(1)
            fp_flags.append(0)
            matched[best_j] = True
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    return tp_flags, fp_flags


def compute_ap(tp_flags: list[int], fp_flags: list[int], n_gt: int) -> float:
    """VOC 11-point interpolation AP."""
    if n_gt == 0:
        return 0.0
    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / n_gt
    ap = 0.0
    for thresh in np.linspace(0, 1, 11):
        p = precisions[recalls >= thresh]
        ap += p.max() if len(p) > 0 else 0.0
    return ap / 11.0


def scene_level_metrics(
    dets: list[dict],
    gts: list[dict],
    class_ids: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute mAP50, mAP50-95, precision, recall for one scene.
    """
    if class_ids is None:
        class_ids = sorted(set([g["cls"] for g in gts] + [d["cls"] for d in dets]))

    iou_thresholds_50 = [0.5]
    iou_thresholds_5095 = np.arange(0.5, 1.0, 0.05).tolist()

    per_class_ap50 = []
    per_class_ap5095 = []

    tp_all, fp_all = [], []
    n_gt_all = 0

    for cls in class_ids:
        gts_cls = [g for g in gts if g["cls"] == cls]
        dets_cls = [d for d in dets if d["cls"] == cls]
        n_gt_all += len(gts_cls)

        # AP@50
        tp, fp = _match_dets_to_gt(dets_cls, gts_cls, 0.5)
        ap50 = compute_ap(tp, fp, len(gts_cls))
        per_class_ap50.append(ap50)
        tp_all.extend(tp)
        fp_all.extend(fp)

        # AP@50:95
        aps = []
        for thr in iou_thresholds_5095:
            tp_t, fp_t = _match_dets_to_gt(dets_cls, gts_cls, thr)
            aps.append(compute_ap(tp_t, fp_t, len(gts_cls)))
        per_class_ap5095.append(float(np.mean(aps)))

    tp_cum = np.cumsum(tp_all)
    fp_cum = np.cumsum(fp_all)
    precision = float(tp_cum[-1] / max(tp_cum[-1] + fp_cum[-1], 1)) if tp_all else 0.0
    recall = float(tp_cum[-1] / max(n_gt_all, 1)) if tp_all else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "map50": float(np.mean(per_class_ap50)) if per_class_ap50 else 0.0,
        "map50_95": float(np.mean(per_class_ap5095)) if per_class_ap5095 else 0.0,
    }


def per_class_metrics(
    dets: list[dict],
    gts: list[dict],
    class_names: dict[int, str] | None = None,
) -> list[dict]:
    """Per-class AP50 and AP50:95."""
    class_ids = sorted(set([g["cls"] for g in gts] + [d["cls"] for d in dets]))
    rows = []
    for cls in class_ids:
        gts_cls = [g for g in gts if g["cls"] == cls]
        dets_cls = [d for d in dets if d["cls"] == cls]
        tp, fp = _match_dets_to_gt(dets_cls, gts_cls, 0.5)
        ap50 = compute_ap(tp, fp, len(gts_cls))
        aps_5095 = []
        for thr in np.arange(0.5, 1.0, 0.05):
            tp_t, fp_t = _match_dets_to_gt(dets_cls, gts_cls, thr)
            aps_5095.append(compute_ap(tp_t, fp_t, len(gts_cls)))
        rows.append({
            "cls_id": cls,
            "cls_name": class_names.get(cls, str(cls)) if class_names else str(cls),
            "n_gt": len(gts_cls),
            "n_det": len(dets_cls),
            "ap50": ap50,
            "ap50_95": float(np.mean(aps_5095)),
        })
    return rows
