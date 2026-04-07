import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Sequence


COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 165, 0), (128, 0, 128), (0, 255, 255),
    (255, 20, 147), (139, 69, 19),
]


def draw_boxes_on_image(img: np.ndarray, boxes_xyxy: np.ndarray,
                        labels: Sequence[str] | None = None,
                        color: tuple = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    out = img.copy()
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(out, str(labels[i]), (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def draw_crop_boundaries(img: np.ndarray, crop_x_starts: list[int],
                          crop_x_ends: list[int]) -> np.ndarray:
    out = img.copy()
    for i, (xs, xe) in enumerate(zip(crop_x_starts, crop_x_ends)):
        color = COLORS[i % len(COLORS)]
        cv2.line(out, (xs, 0), (xs, img.shape[0]), color, 2)
        cv2.line(out, (xe, 0), (xe, img.shape[0]), color, 2)
    return out


def overlay_mask(img: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.4,
                 color: tuple = (255, 0, 0)) -> np.ndarray:
    out = img.copy().astype(np.float32)
    overlay = np.zeros_like(out)
    overlay[mask > 0] = color
    out = (1 - alpha) * out + alpha * overlay
    return np.clip(out, 0, 255).astype(np.uint8)


def save_qc_panel(panorama: np.ndarray,
                  crops: list[np.ndarray],
                  masks: list[np.ndarray],
                  crop_x_starts: list[int],
                  crop_x_ends: list[int],
                  scene_id: str,
                  out_path: str | Path) -> None:
    N = len(crops)
    fig, axes = plt.subplots(2, max(N, 2), figsize=(4 * max(N, 2), 8))
    axes = np.array(axes)

    pano_vis = draw_crop_boundaries(panorama, crop_x_starts, crop_x_ends)
    axes[0, 0].imshow(cv2.cvtColor(pano_vis, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"{scene_id}\nPanorama + boundaries")
    axes[0, 0].axis("off")

    for i in range(1, max(N, 2)):
        axes[0, i].axis("off")

    for i, crop in enumerate(crops):
        axes[1, i].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        title = f"Crop {i}"
        if i < len(masks) and masks[i] is not None:
            axes[1, i].set_title(title + " + mask")
            m = masks[i]
            axes[1, i].imshow(m, alpha=0.3, cmap="Reds")
        else:
            axes[1, i].set_title(title)
        axes[1, i].axis("off")

    for i in range(N, axes.shape[1]):
        axes[1, i].axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def save_detection_panel(panorama: np.ndarray,
                         gt_boxes: np.ndarray,
                         odo_boxes: np.ndarray,
                         citron_img: np.ndarray,
                         citron_boxes: np.ndarray,
                         scene_id: str,
                         out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    gt_vis = draw_boxes_on_image(panorama, gt_boxes, color=(0, 255, 0))
    axes[0].imshow(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"{scene_id}\nGround Truth")
    axes[0].axis("off")

    odo_vis = draw_boxes_on_image(panorama, odo_boxes, color=(255, 165, 0))
    axes[1].imshow(cv2.cvtColor(odo_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title("ODO predictions")
    axes[1].axis("off")

    citron_vis = draw_boxes_on_image(citron_img, citron_boxes, color=(0, 0, 255))
    axes[2].imshow(cv2.cvtColor(citron_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title("CITRON stitched + predictions")
    axes[2].axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
