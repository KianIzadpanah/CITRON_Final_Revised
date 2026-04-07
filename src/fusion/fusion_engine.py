"""
fusion_engine.py — Phase C

Mask-guided image fusion engine for CITRON.

Exposed functions:
    fuse_concat_only(scene, crop_meta)
    fuse_oracle_mask(scene, crop_meta, gt_mask_dir)
    fuse_predicted_mask(scene, crop_meta, overlap_model, device, size)
    fuse_predicted_mask_resized(scene, crop_meta, overlap_model, device, size, resize_factor)

Each returns a FusionResult dataclass.

Seam policy: mask-guided keep-reference.
  - In the overlap zone of adjacent crops (k, k+1), the LEFT crop pixels are kept
    (reference), and the RIGHT crop's overlapping pixels are masked out.
  - This avoids double-writing and preserves object continuity near seams.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.dataset.geometry_utils import compute_crop_geometries, build_overlap_mask


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    scene_id: str
    variant: str
    stitched_image: np.ndarray          # BGR numpy array
    stitched_bytes: int                 # JPEG-compressed size estimate
    canvas_width: int
    canvas_height: int
    resize_factor: float = 1.0
    mask_dice_scores: list[float] = field(default_factory=list)
    mask_iou_scores: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_crops(crop_rows: list[dict]) -> list[tuple[np.ndarray, int, int]]:
    """Returns list of (img_bgr, x_start, x_end) sorted by crop_index."""
    result = []
    for row in sorted(crop_rows, key=lambda r: r["crop_index"]):
        img = cv2.imread(row["image_path"])
        if img is None:
            h = row["crop_height"]
            w = row["crop_width"]
            img = np.zeros((h, w, 3), np.uint8)
        result.append((img, row["crop_x_start"], row["crop_x_end"]))
    return result


def _build_canvas(pano_h: int, pano_w: int) -> np.ndarray:
    return np.zeros((pano_h, pano_w, 3), dtype=np.uint8)


def _place_crops_on_canvas(
    canvas: np.ndarray,
    crops: list[tuple[np.ndarray, int, int]],
    masks: list[np.ndarray | None],
) -> np.ndarray:
    """
    Place crops onto canvas left-to-right using mask-guided keep-reference.
    masks[i] = binary mask (H, W) indicating overlap zone of crop i with crop i+1.
    Pixels in crop i that are marked as overlap (mask=255) are NOT placed —
    unless they haven't been written yet (canvas is black there).
    """
    written = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=bool)
    out = canvas.copy()

    for i, (crop_img, xs, xe) in enumerate(crops):
        ch, cw = crop_img.shape[:2]
        region_w = xe - xs
        if cw != region_w:
            crop_img = cv2.resize(crop_img, (region_w, ch), interpolation=cv2.INTER_LINEAR)

        # Build per-pixel keep mask for this crop
        keep = np.ones((ch, region_w), dtype=bool)

        # Exclude overlap with NEXT crop (right side of this crop)
        if masks[i] is not None:
            overlap_zone = (masks[i] > 127)
            if overlap_zone.shape != (ch, region_w):
                overlap_zone_r = cv2.resize(
                    masks[i], (region_w, ch), interpolation=cv2.INTER_NEAREST
                )
                overlap_zone = (overlap_zone_r > 127)
            # Only exclude if the canvas region has already been written (keep-reference)
            # For first crop (i=0) nothing written yet; for later crops, respect written mask
            if i > 0:
                already_written = written[:, xs:xe]
                keep[overlap_zone & already_written] = False

        crop_region = crop_img[:ch, :region_w]
        for c in range(3):
            ch_canvas = out[:ch, xs:xe, c]
            ch_canvas[keep] = crop_region[:, :, c][keep]
            out[:ch, xs:xe, c] = ch_canvas

        written[:ch, xs:xe] |= keep

    return out


def _mask_from_gt_file(mask_path: str | Path, target_h: int, target_w: int) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    return cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _mask_from_model(
    img_l_bgr: np.ndarray, img_r_bgr: np.ndarray,
    model, device: torch.device, size: int, threshold: float = 0.5,
) -> np.ndarray:
    to_tensor = transforms.ToTensor()

    def prep(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size))
        t = to_tensor(rgb)
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return t.unsqueeze(0).to(device)

    tl = prep(img_l_bgr)
    tr_ = prep(img_r_bgr)
    with torch.no_grad():
        prob = model(tl, tr_).cpu().squeeze().numpy()
    bin_mask = (prob >= threshold).astype(np.uint8) * 255
    return bin_mask


def _compute_mask_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Returns (dice, iou) between binary masks."""
    p = (pred > 127).astype(float).ravel()
    g = (gt > 127).astype(float).ravel()
    eps = 1e-6
    inter = (p * g).sum()
    dice = (2 * inter + eps) / (p.sum() + g.sum() + eps)
    union = p.sum() + g.sum() - inter
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def _estimate_bytes(img: np.ndarray, quality: int = 85) -> int:
    """Estimate compressed JPEG size in bytes."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return len(buf) if ok else int(img.nbytes * 0.3)


def _resize_image(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_scene_geometry(crop_rows: list[dict]):
    row0 = crop_rows[0]
    pano_h = row0["panorama_height"]
    pano_w = row0["panorama_width"]
    return pano_h, pano_w


def fuse_concat_only(scene_id: str, crop_rows: list[dict]) -> FusionResult:
    """A1: concatenate crops with no redundancy removal."""
    crops = _load_crops(crop_rows)
    pano_h, pano_w = _get_scene_geometry(crop_rows)
    canvas = _build_canvas(pano_h, pano_w)
    masks = [None] * len(crops)
    stitched = _place_crops_on_canvas(canvas, crops, masks)
    return FusionResult(
        scene_id=scene_id, variant="A1_concat_only",
        stitched_image=stitched, stitched_bytes=_estimate_bytes(stitched),
        canvas_width=pano_w, canvas_height=pano_h,
    )


def fuse_oracle_mask(scene_id: str, crop_rows: list[dict]) -> FusionResult:
    """A2: fusion using ground-truth overlap masks."""
    crops = _load_crops(crop_rows)
    pano_h, pano_w = _get_scene_geometry(crop_rows)
    canvas = _build_canvas(pano_h, pano_w)

    sorted_rows = sorted(crop_rows, key=lambda r: r["crop_index"])
    masks = []
    for i, row in enumerate(sorted_rows):
        mp = row.get("mask_path_right", "")
        if mp and Path(mp).exists():
            ch = row["crop_height"]
            cw = row["crop_x_end"] - row["crop_x_start"]
            masks.append(_mask_from_gt_file(mp, ch, cw))
        else:
            masks.append(None)

    stitched = _place_crops_on_canvas(canvas, crops, masks)
    return FusionResult(
        scene_id=scene_id, variant="A2_oracle_mask",
        stitched_image=stitched, stitched_bytes=_estimate_bytes(stitched),
        canvas_width=pano_w, canvas_height=pano_h,
    )


def fuse_predicted_mask(
    scene_id: str, crop_rows: list[dict],
    overlap_model, device: torch.device, model_input_size: int = 256,
    threshold: float = 0.5,
) -> FusionResult:
    """A3: fusion using predicted overlap masks (ResNet-50 or MobileNet)."""
    crops = _load_crops(crop_rows)
    pano_h, pano_w = _get_scene_geometry(crop_rows)
    canvas = _build_canvas(pano_h, pano_w)

    sorted_rows = sorted(crop_rows, key=lambda r: r["crop_index"])
    dice_scores, iou_scores = [], []

    masks = []
    for i in range(len(crops)):
        if i < len(crops) - 1:
            img_l, xs_l, xe_l = crops[i]
            img_r, xs_r, xe_r = crops[i + 1]
            pred_mask = _mask_from_model(img_l, img_r, overlap_model,
                                         device, model_input_size, threshold)
            # resize to crop dimensions
            cw = xe_l - xs_l
            ch = img_l.shape[0]
            pred_mask = cv2.resize(pred_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

            # Compute quality vs GT if available
            mp = sorted_rows[i].get("mask_path_right", "")
            if mp and Path(mp).exists():
                gt_m = _mask_from_gt_file(mp, ch, cw)
                d, iou = _compute_mask_metrics(pred_mask, gt_m)
                dice_scores.append(d)
                iou_scores.append(iou)

            masks.append(pred_mask)
        else:
            masks.append(None)

    stitched = _place_crops_on_canvas(canvas, crops, masks)
    return FusionResult(
        scene_id=scene_id, variant="A3_pred_mask",
        stitched_image=stitched, stitched_bytes=_estimate_bytes(stitched),
        canvas_width=pano_w, canvas_height=pano_h,
        mask_dice_scores=dice_scores, mask_iou_scores=iou_scores,
    )


def fuse_predicted_mask_resized(
    scene_id: str, crop_rows: list[dict],
    overlap_model, device: torch.device, model_input_size: int = 256,
    resize_factor: float = 0.6, threshold: float = 0.5,
) -> FusionResult:
    """A4: fusion with predicted masks + resize by factor (paper CITRON full pipeline)."""
    result = fuse_predicted_mask(
        scene_id, crop_rows, overlap_model, device, model_input_size, threshold
    )
    resized = _resize_image(result.stitched_image, resize_factor)
    return FusionResult(
        scene_id=scene_id, variant="A4_pred_mask_resized",
        stitched_image=resized, stitched_bytes=_estimate_bytes(resized),
        canvas_width=result.canvas_width, canvas_height=result.canvas_height,
        resize_factor=resize_factor,
        mask_dice_scores=result.mask_dice_scores,
        mask_iou_scores=result.mask_iou_scores,
    )


def save_fusion_result(
    result: FusionResult,
    out_dir: str | Path,
    crop_rows: list[dict] | None = None,
) -> Path:
    """Save stitched image and seam visualization to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{result.scene_id}_{result.variant}.jpg"
    cv2.imwrite(str(img_path), result.stitched_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return img_path
