"""
detection_utils.py — coordinate conversion, NMS, and YOLO helpers.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Box format helpers
# ---------------------------------------------------------------------------

def yolo_to_pixels(cx: float, cy: float, w: float, h: float,
                   img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def pixels_to_yolo(x1: float, y1: float, x2: float, y2: float,
                   img_w: int, img_h: int) -> tuple[float, float, float, float]:
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def crop_box_to_scene(
    cx: float, cy: float, bw: float, bh: float,
    crop_w: int, crop_h: int,
    crop_x_start: int,
    pano_w: int, pano_h: int,
) -> tuple[float, float, float, float]:
    """Convert YOLO crop-space box to YOLO scene (panorama) space."""
    x1, y1, x2, y2 = yolo_to_pixels(cx, cy, bw, bh, crop_w, crop_h)
    x1 += crop_x_start
    x2 += crop_x_start
    return pixels_to_yolo(x1, y1, x2, y2, pano_w, pano_h)


def stitched_box_to_scene(
    cx: float, cy: float, bw: float, bh: float,
    stitched_w: int, stitched_h: int,
    pano_w: int, pano_h: int,
    resize_factor: float,
) -> tuple[float, float, float, float]:
    """Convert YOLO stitched-image box to YOLO panorama space."""
    x1, y1, x2, y2 = yolo_to_pixels(cx, cy, bw, bh, stitched_w, stitched_h)
    scale = 1.0 / resize_factor
    return pixels_to_yolo(x1 * scale, y1 * scale, x2 * scale, y2 * scale, pano_w, pano_h)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def box_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def class_nms(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Per-class NMS. Each det must have keys: cls, conf, x1, y1, x2, y2."""
    if not dets:
        return []
    classes = set(d["cls"] for d in dets)
    kept = []
    for cls in classes:
        cls_dets = sorted([d for d in dets if d["cls"] == cls], key=lambda d: -d["conf"])
        suppressed = [False] * len(cls_dets)
        for i in range(len(cls_dets)):
            if suppressed[i]:
                continue
            kept.append(cls_dets[i])
            for j in range(i + 1, len(cls_dets)):
                if not suppressed[j] and box_iou(cls_dets[i], cls_dets[j]) >= iou_thresh:
                    suppressed[j] = True
    return kept


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_scene_gt(label_path: str | Path, pano_w: int, pano_h: int) -> list[dict]:
    """Load YOLO-format scene GT and return pixel xyxy dicts."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1, y1, x2, y2 = yolo_to_pixels(cx, cy, bw, bh, pano_w, pano_h)
            boxes.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


# ---------------------------------------------------------------------------
# YOLO result parsing
# ---------------------------------------------------------------------------

def parse_yolo_results(results, conf_thresh: float = 0.25) -> list[dict]:
    """Extract detections from ultralytics YOLO result object."""
    dets = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            xyxy = box.xyxy[0].tolist()
            dets.append({
                "cls": int(box.cls[0]),
                "conf": conf,
                "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
            })
    return dets
