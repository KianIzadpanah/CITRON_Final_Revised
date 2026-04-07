"""
run_fusion_ablation.py — Phase C ablation runner

Evaluates all fusion variants on scene_test.csv:
    A0  ODO             (separate crops, merged detections)
    A1  Concat-only
    A2  Oracle-mask
    A3  Predicted-mask (ResNet-50)
    A4  A3 + resize 0.6
    A5  Predicted-mask (MobileNet)

Usage:
    python src/fusion/run_fusion_ablation.py \
        --detector_weights outputs/detection/crop_mode/weights/best.pt \
        --resnet_ckpt outputs/overlap/overlap_resnet50_best.pt \
        --mobilenet_ckpt outputs/overlap/overlap_mobilenet_best.pt \
        --scene_csv data/processed/metadata/scene_test.csv \
        --crop_meta data/processed/metadata/crop_metadata.csv \
        --out_dir outputs/ablation

Outputs:
    fusion_ablation_results.csv
    fusion_ablation_summary.json
    figures/ablation_examples/  (one panel per variant)
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.common.seed_utils import set_all_seeds
from src.common.config_utils import load_yaml, save_json
from src.common.io_utils import get_logger, log_failure
from src.overlap.overlap_model import build_overlap_model
from src.fusion.fusion_engine import (
    fuse_concat_only, fuse_oracle_mask,
    fuse_predicted_mask, fuse_predicted_mask_resized,
    save_fusion_result, _estimate_bytes, _load_crops,
)

LOG = get_logger("run_fusion_ablation")
SEED = 42
RESIZE_FACTOR = 0.6
MODEL_SIZE = 256


def load_yolo(weights_path: str, device: torch.device):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return model


def run_yolo_on_image(yolo_model, img_bgr: np.ndarray) -> list[dict]:
    """Run YOLO on a BGR image, return list of {cls, conf, xyxy}."""
    results = yolo_model(img_bgr, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
            })
    return detections


def load_gt_boxes(gt_label_path: str, pano_w: int, pano_h: int) -> list[dict]:
    """Load YOLO-format GT and convert to pixel xyxy."""
    boxes = []
    if not Path(gt_label_path).exists():
        return boxes
    with open(gt_label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * pano_w
            y1 = (cy - bh / 2) * pano_h
            x2 = (cx + bw / 2) * pano_w
            y2 = (cy + bh / 2) * pano_h
            boxes.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def compute_simple_ap(dets: list[dict], gts: list[dict],
                       iou_thresh: float = 0.5) -> dict[str, float]:
    """Simplified AP computation for one scene and one IoU threshold."""
    if not gts:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0}
    if not dets:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0}

    dets_sorted = sorted(dets, key=lambda d: -d["conf"])
    matched = [False] * len(gts)
    tp, fp = 0, 0

    for d in dets_sorted:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            if d["cls"] != g["cls"]:
                continue
            iou = _box_iou(d, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched[best_j] = True
        else:
            fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(len(gts), 1)
    return {"precision": precision, "recall": recall, "map50": precision * recall}


def _box_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def odo_nms(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Class-wise NMS to remove duplicate ODO detections from overlapping crops."""
    if not dets:
        return []
    classes = set(d["cls"] for d in dets)
    kept = []
    for cls in classes:
        cls_dets = [d for d in dets if d["cls"] == cls]
        cls_dets = sorted(cls_dets, key=lambda d: -d["conf"])
        suppressed = [False] * len(cls_dets)
        for i in range(len(cls_dets)):
            if suppressed[i]:
                continue
            kept.append(cls_dets[i])
            for j in range(i + 1, len(cls_dets)):
                if not suppressed[j] and _box_iou(cls_dets[i], cls_dets[j]) >= iou_thresh:
                    suppressed[j] = True
    return kept


def crop_dets_to_scene(dets: list[dict], xs: int, pano_w: int, pano_h: int,
                        crop_w: int, crop_h: int) -> list[dict]:
    """Shift crop-coordinate detections to scene coordinates."""
    out = []
    for d in dets:
        out.append({
            "cls": d["cls"],
            "conf": d["conf"],
            "x1": d["x1"] + xs,
            "y1": d["y1"],
            "x2": d["x2"] + xs,
            "y2": d["y2"],
        })
    return out


def run_ablation(
    scene_df: pd.DataFrame,
    crop_meta: pd.DataFrame,
    yolo, resnet_model, mobilenet_model,
    device: torch.device,
    out_dir: Path,
    fig_dir: Path,
    failure_log: Path,
) -> pd.DataFrame:

    variants = ["A0_ODO", "A1_concat", "A2_oracle", "A3_pred_resnet50",
                "A4_pred_resnet50_resized", "A5_pred_mobilenet"]

    rows = []
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    for _, srow in scene_df.iterrows():
        scene_id = str(srow["scene_id"])
        pano_h = int(crop_meta[crop_meta["scene_id"] == scene_id].iloc[0]["panorama_height"])
        pano_w = int(crop_meta[crop_meta["scene_id"] == scene_id].iloc[0]["panorama_width"])
        n_veh = int(srow["vehicle_count"])

        scene_crops = crop_meta[crop_meta["scene_id"] == scene_id].sort_values("crop_index")
        crop_rows = scene_crops.to_dict("records")

        gt_path = str(srow["scene_gt_label_path"])
        gt_boxes = load_gt_boxes(gt_path, pano_w, pano_h)

        for variant in variants:
            try:
                row = _evaluate_variant(
                    variant, scene_id, crop_rows, gt_boxes,
                    pano_w, pano_h, n_veh,
                    yolo, resnet_model, mobilenet_model, device,
                    out_dir, fig_dir,
                )
                rows.append(row)
            except Exception as e:
                log_failure(f"{scene_id}_{variant}", str(e), failure_log)
                LOG.error(f"[{scene_id}] {variant} failed: {e}")

    return pd.DataFrame(rows)


def _evaluate_variant(
    variant, scene_id, crop_rows, gt_boxes,
    pano_w, pano_h, n_veh,
    yolo, resnet_model, mobilenet_model, device,
    out_dir, fig_dir,
) -> dict:

    mask_dice, mask_iou = [], []
    payload_bytes = 0
    v2i_bytes = 0

    if variant == "A0_ODO":
        # Run YOLO on each crop, merge in scene coords
        all_dets = []
        crops_loaded = _load_crops(crop_rows)
        for (crop_img, xs, xe), row in zip(crops_loaded, sorted(crop_rows, key=lambda r: r["crop_index"])):
            dets = run_yolo_on_image(yolo, crop_img)
            scene_dets = crop_dets_to_scene(dets, xs, pano_w, pano_h,
                                             crop_img.shape[1], crop_img.shape[0])
            all_dets.extend(scene_dets)
            payload_bytes += _estimate_bytes(crop_img)
        all_dets = odo_nms(all_dets)
        v2i_bytes = payload_bytes
        eval_img = None
        eval_dets = all_dets

    else:
        # Fusion variants
        if variant == "A1_concat":
            result = fuse_concat_only(scene_id, crop_rows)
        elif variant == "A2_oracle":
            result = fuse_oracle_mask(scene_id, crop_rows)
        elif variant == "A3_pred_resnet50":
            result = fuse_predicted_mask(scene_id, crop_rows, resnet_model, device, MODEL_SIZE)
        elif variant == "A4_pred_resnet50_resized":
            result = fuse_predicted_mask_resized(scene_id, crop_rows, resnet_model, device,
                                                  MODEL_SIZE, RESIZE_FACTOR)
        elif variant == "A5_pred_mobilenet":
            if mobilenet_model is None:
                return {"scene_id": scene_id, "variant": variant, "skipped": True}
            result = fuse_predicted_mask_resized(scene_id, crop_rows, mobilenet_model, device,
                                                  MODEL_SIZE, RESIZE_FACTOR)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Run YOLO on stitched image
        dets = run_yolo_on_image(yolo, result.stitched_image)
        # Convert stitched detections back to scene space if resized
        rf = result.resize_factor
        if rf != 1.0:
            dets = _scale_dets(dets, 1.0 / rf)
        eval_dets = dets
        eval_img = result.stitched_image
        payload_bytes = result.stitched_bytes
        v2i_bytes = result.stitched_bytes
        mask_dice = result.mask_dice_scores
        mask_iou = result.mask_iou_scores

        # Save one example per variant (first scene only)
        if len(list(fig_dir.glob(f"*{variant}*"))) == 0:
            save_path = fig_dir / f"{scene_id}_{variant}.jpg"
            cv2.imwrite(str(save_path), result.stitched_image)

    # Compute metrics vs GT
    ap = compute_simple_ap(eval_dets, gt_boxes, iou_thresh=0.5)

    return {
        "scene_id": scene_id,
        "variant": variant,
        "vehicle_count": n_veh,
        "n_gt": len(gt_boxes),
        "n_det": len(eval_dets),
        "precision": ap["precision"],
        "recall": ap["recall"],
        "map50": ap["map50"],
        "v2i_bytes": v2i_bytes,
        "payload_bytes": payload_bytes,
        "mask_dice_mean": float(np.mean(mask_dice)) if mask_dice else None,
        "mask_iou_mean": float(np.mean(mask_iou)) if mask_iou else None,
    }


def _scale_dets(dets: list[dict], scale: float) -> list[dict]:
    out = []
    for d in dets:
        out.append({**d, "x1": d["x1"] * scale, "y1": d["y1"] * scale,
                    "x2": d["x2"] * scale, "y2": d["y2"] * scale})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_weights", required=True)
    parser.add_argument("--resnet_ckpt", required=True)
    parser.add_argument("--mobilenet_ckpt", default="")
    parser.add_argument("--scene_csv", required=True)
    parser.add_argument("--crop_meta", required=True)
    parser.add_argument("--out_dir", default="outputs/ablation")
    parser.add_argument("--fig_dir", default="outputs/figures/ablation_examples")
    args = parser.parse_args()

    set_all_seeds(SEED)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    failure_log = out_dir / "failures.log"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG.info("Loading YOLO detector...")
    yolo = load_yolo(args.detector_weights, device)

    LOG.info("Loading ResNet-50 overlap model...")
    resnet_model = build_overlap_model("resnet50", pretrained=False)
    ckpt = torch.load(args.resnet_ckpt, map_location=device)
    resnet_model.load_state_dict(ckpt["model_state"])
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    mobilenet_model = None
    if args.mobilenet_ckpt and Path(args.mobilenet_ckpt).exists():
        LOG.info("Loading MobileNet overlap model...")
        mobilenet_model = build_overlap_model("mobilenet_v3_large", pretrained=False)
        ckpt_m = torch.load(args.mobilenet_ckpt, map_location=device)
        mobilenet_model.load_state_dict(ckpt_m["model_state"])
        mobilenet_model = mobilenet_model.to(device)
        mobilenet_model.eval()

    scene_df = pd.read_csv(args.scene_csv)
    crop_meta = pd.read_csv(args.crop_meta)
    LOG.info(f"Scenes: {len(scene_df)} | Crops: {len(crop_meta)}")

    results_df = run_ablation(
        scene_df, crop_meta, yolo, resnet_model, mobilenet_model,
        device, out_dir, fig_dir, failure_log,
    )

    results_path = out_dir / "fusion_ablation_results.csv"
    results_df.to_csv(results_path, index=False)
    LOG.info(f"Ablation results saved: {results_path}")

    # Summary JSON
    summary = {}
    for variant, grp in results_df.groupby("variant"):
        summary[variant] = {
            "precision_mean": round(float(grp["precision"].mean()), 4),
            "recall_mean": round(float(grp["recall"].mean()), 4),
            "map50_mean": round(float(grp["map50"].mean()), 4),
            "v2i_bytes_mean": round(float(grp["v2i_bytes"].mean()), 0),
            "n_scenes": len(grp),
        }
    save_json(summary, out_dir / "fusion_ablation_summary.json")
    LOG.info("fusion_ablation_summary.json saved")


if __name__ == "__main__":
    main()
