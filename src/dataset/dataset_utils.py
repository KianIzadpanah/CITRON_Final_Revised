"""
Utilities for loading KITTI tracking labels and images.
KITTI tracking format (one file per sequence):
  frame_id  track_id  class  truncated  occluded  alpha  x1 y1 x2 y2  h w l  x y z  ry  [score]
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator


KITTI_FIELDS = [
    "frame_id", "track_id", "type", "truncated", "occluded", "alpha",
    "left", "top", "right", "bottom",
    "height", "width", "length", "x", "y", "z", "rotation_y"
]

IGNORE_CLASSES = {"DontCare"}


def load_kitti_tracking_labels(label_path: str | Path, ignore_classes: set | None = None) -> dict[int, list]:
    """
    Load KITTI tracking label file.
    Returns dict mapping frame_id -> list of (class_str, x1, y1, x2, y2).
    """
    if ignore_classes is None:
        ignore_classes = IGNORE_CLASSES
    labels: dict[int, list] = {}
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            frame_id = int(parts[0])
            obj_class = parts[2]
            if obj_class in ignore_classes:
                continue
            x1 = float(parts[6])
            y1 = float(parts[7])
            x2 = float(parts[8])
            y2 = float(parts[9])
            if frame_id not in labels:
                labels[frame_id] = []
            labels[frame_id].append((obj_class, x1, y1, x2, y2))
    return labels


def list_sequence_frames(img_root: str | Path, seq_id: int) -> list[int]:
    """Return sorted list of frame IDs available for a sequence."""
    seq_dir = Path(img_root) / f"{seq_id:04d}"
    if not seq_dir.exists():
        return []
    frames = []
    for p in sorted(seq_dir.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            try:
                frames.append(int(p.stem))
            except ValueError:
                pass
    return sorted(frames)


def load_image(img_root: str | Path, seq_id: int, frame_id: int) -> np.ndarray | None:
    """Load a KITTI tracking image (BGR, numpy array)."""
    seq_dir = Path(img_root) / f"{seq_id:04d}"
    for ext in (".png", ".jpg", ".jpeg"):
        p = seq_dir / f"{frame_id:06d}{ext}"
        if p.exists():
            img = cv2.imread(str(p))
            return img
    return None


def augment_crop(
    img: np.ndarray,
    mask: np.ndarray,
    boxes_local: list[tuple],
    aug_probs: dict,
    scale_range: tuple,
    rotate_range: tuple,
    blur_kernel_range: tuple,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """
    Apply synchronized augmentation to (crop image, overlap mask, local boxes).
    Returns (aug_img, aug_mask, aug_boxes).

    Augmentations (paper Section IV-E):
        - Motion/Gaussian blur  (p=0.25)
        - Scale jitter           (p=0.25)
        - Rotation noise ±3°     (p=0.25)
    """
    h, w = img.shape[:2]

    # --- Blur ---
    if rng.random() < aug_probs.get("blur", 0.25):
        k = int(rng.integers(blur_kernel_range[0], blur_kernel_range[1] + 1))
        k = k if k % 2 == 1 else k + 1
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (k, k), 0)
        else:
            angle = float(rng.integers(0, 360))
            kernel = _motion_blur_kernel(k, angle)
            img = cv2.filter2D(img, -1, kernel)
        # blur does not affect geometry

    # --- Scale ---
    if rng.random() < aug_probs.get("scale", 0.25):
        s = float(rng.uniform(scale_range[0], scale_range[1]))
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        boxes_local = _scale_boxes(boxes_local, w, h, new_w, new_h)
        w, h = new_w, new_h

    # --- Rotation ---
    if rng.random() < aug_probs.get("rotate", 0.25):
        angle = float(rng.uniform(rotate_range[0], rotate_range[1]))
        img, mask, boxes_local = _rotate_all(img, mask, boxes_local, angle)

    return img, mask, boxes_local


def _motion_blur_kernel(size: int, angle: float) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    return k / k.sum()


def _scale_boxes(boxes, old_w, old_h, new_w, new_h):
    sx, sy = new_w / old_w, new_h / old_h
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        out.append((cls, x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return out


def _rotate_all(img, mask, boxes, angle):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img_r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    mask_r = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    boxes_r = []
    for cls, x1, y1, x2, y2 in boxes:
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((4, 1))])
        pts_r = (M @ pts_h.T).T
        nx1, ny1 = pts_r[:, 0].min(), pts_r[:, 1].min()
        nx2, ny2 = pts_r[:, 0].max(), pts_r[:, 1].max()
        nx1, nx2 = max(0, nx1), min(w, nx2)
        ny1, ny2 = max(0, ny1), min(h, ny2)
        if nx2 > nx1 and ny2 > ny1:
            boxes_r.append((cls, nx1, ny1, nx2, ny2))
    return img_r, mask_r, boxes_r
