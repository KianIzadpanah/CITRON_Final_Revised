"""
Geometric utilities for CITRON dataset:
- Crop coordinate computation (paper eq. 11-13)
- Bounding box clipping to crop windows
- KITTI <-> YOLO label conversion
- Crop-to-scene coordinate mapping for evaluation
"""

import numpy as np
from typing import NamedTuple


class CropGeometry(NamedTuple):
    crop_index: int
    x_start: int
    x_end: int
    width: int
    height: int
    overlap_ratio: float
    stride: int


def compute_crop_geometries(
    panorama_width: int,
    panorama_height: int,
    n_vehicles: int,
    overlap_ratio: float,
) -> list[CropGeometry]:
    """
    Compute crop boundaries for N vehicles given an overlap ratio rho.
    Paper equations 11-13:
        wc = W / N
        overlap_px = rho * wc
        stride = wc - overlap_px
        crop_i: x_start = i * stride, x_end = x_start + wc
    """
    wc = panorama_width / n_vehicles
    overlap_px = overlap_ratio * wc
    stride = wc - overlap_px

    geometries = []
    for i in range(n_vehicles):
        x_start = int(round(i * stride))
        x_end = min(int(round(x_start + wc)), panorama_width)
        geometries.append(CropGeometry(
            crop_index=i,
            x_start=x_start,
            x_end=x_end,
            width=x_end - x_start,
            height=panorama_height,
            overlap_ratio=overlap_ratio,
            stride=int(round(stride)),
        ))
    return geometries


def build_overlap_mask(
    panorama_width: int,
    panorama_height: int,
    geom_left: CropGeometry,
    geom_right: CropGeometry,
    target_crop: str = "left",
) -> np.ndarray:
    """
    Build a binary overlap mask for an adjacent pair (left, right).

    The overlap region in panorama coordinates is [geom_right.x_start, geom_left.x_end].
    The mask is relative to the target crop image:
        - target_crop='left' : mask is in left-crop coordinate space
        - target_crop='right': mask is in right-crop coordinate space

    Returns uint8 binary mask (0=unique, 255=overlap) of shape (height, width).
    """
    overlap_x_start_pano = geom_right.x_start
    overlap_x_end_pano = geom_left.x_end

    if overlap_x_end_pano <= overlap_x_start_pano:
        # No actual overlap
        if target_crop == "left":
            return np.zeros((geom_left.height, geom_left.width), dtype=np.uint8)
        else:
            return np.zeros((geom_right.height, geom_right.width), dtype=np.uint8)

    if target_crop == "left":
        local_start = overlap_x_start_pano - geom_left.x_start
        local_end = overlap_x_end_pano - geom_left.x_start
        mask = np.zeros((geom_left.height, geom_left.width), dtype=np.uint8)
        local_start = max(0, local_start)
        local_end = min(geom_left.width, local_end)
    else:
        local_start = overlap_x_start_pano - geom_right.x_start
        local_end = overlap_x_end_pano - geom_right.x_start
        mask = np.zeros((geom_right.height, geom_right.width), dtype=np.uint8)
        local_start = max(0, local_start)
        local_end = min(geom_right.width, local_end)

    if local_end > local_start:
        mask[:, local_start:local_end] = 255
    return mask


def clip_boxes_to_crop(
    boxes_kitti: list[tuple],
    crop_x_start: int,
    crop_x_end: int,
    crop_height: int,
    min_visibility: float = 0.1,
) -> list[tuple]:
    """
    Clip KITTI-format boxes [x1, y1, x2, y2] to a crop window.
    Returns only boxes with visibility >= min_visibility.
    Each returned box is shifted to crop-local coordinates.
    """
    clipped = []
    for box in boxes_kitti:
        cls, x1, y1, x2, y2 = box
        cx1 = max(x1, crop_x_start)
        cx2 = min(x2, crop_x_end)
        cy1 = max(y1, 0)
        cy2 = min(y2, crop_height)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        orig_area = max((x2 - x1) * (y2 - y1), 1e-6)
        clip_area = (cx2 - cx1) * (cy2 - cy1)
        if clip_area / orig_area < min_visibility:
            continue
        # shift to crop-local x
        lx1 = cx1 - crop_x_start
        lx2 = cx2 - crop_x_start
        clipped.append((cls, lx1, cy1, lx2, cy2))
    return clipped


def kitti_box_to_yolo(x1: float, y1: float, x2: float, y2: float,
                       img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert pixel [x1,y1,x2,y2] to YOLO normalized [cx, cy, w, h]."""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def yolo_box_to_pixels(cx: float, cy: float, w: float, h: float,
                        img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert YOLO normalized [cx,cy,w,h] to pixel [x1,y1,x2,y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def crop_to_scene_coords(
    cx_norm: float, cy_norm: float, w_norm: float, h_norm: float,
    crop_w: int, crop_h: int,
    crop_x_start: int,
    pano_w: int, pano_h: int,
) -> tuple[float, float, float, float]:
    """
    Convert YOLO-normalized crop-space box to YOLO-normalized scene (panorama) space.
    Used in ODO evaluation and CITRON box projection.
    """
    x1_crop, y1_crop, x2_crop, y2_crop = yolo_box_to_pixels(
        cx_norm, cy_norm, w_norm, h_norm, crop_w, crop_h
    )
    x1_scene = x1_crop + crop_x_start
    x2_scene = x2_crop + crop_x_start
    y1_scene = y1_crop
    y2_scene = y2_crop
    cx_s, cy_s, w_s, h_s = kitti_box_to_yolo(
        x1_scene, y1_scene, x2_scene, y2_scene, pano_w, pano_h
    )
    return cx_s, cy_s, w_s, h_s


def stitched_to_scene_coords(
    cx_norm: float, cy_norm: float, w_norm: float, h_norm: float,
    stitched_w: int, stitched_h: int,
    pano_w: int, pano_h: int,
    resize_factor: float,
) -> tuple[float, float, float, float]:
    """
    Convert YOLO-normalized stitched-image box back to panorama coordinate space.
    The stitched image was resized by `resize_factor` from the fused canvas.
    Canvas width ~ pano_w; after resize: stitched_w = pano_w * resize_factor.
    """
    x1_s, y1_s, x2_s, y2_s = yolo_box_to_pixels(
        cx_norm, cy_norm, w_norm, h_norm, stitched_w, stitched_h
    )
    # scale back to canvas/pano space
    scale = 1.0 / resize_factor
    x1_p = x1_s * scale
    x2_p = x2_s * scale
    y1_p = y1_s * scale
    y2_p = y2_s * scale
    cx_p, cy_p, w_p, h_p = kitti_box_to_yolo(x1_p, y1_p, x2_p, y2_p, pano_w, pano_h)
    return cx_p, cy_p, w_p, h_p
