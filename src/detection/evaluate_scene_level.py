"""
evaluate_scene_level.py — Phase D

Scene-level comparison of ODO vs CITRON object detection.

Both methods use the SAME YOLOv8n weights (fair comparison).
ODO runs YOLO per-crop then merges with NMS.
CITRON fuses crops with predicted masks then runs YOLO once.

Usage:
    python src/detection/evaluate_scene_level.py \
        --detector_weights outputs/detection/crop_mode/train/weights/best.pt \
        --overlap_ckpt outputs/overlap/overlap_resnet50_best.pt \
        --scene_csv data/processed/metadata/scene_test.csv \
        --crop_meta data/processed/metadata/crop_metadata.csv \
        --out_dir outputs/detection \
        --fig_dir outputs/figures/scene_examples

Outputs:
    scene_level_results.csv
    scene_level_per_class_results.csv
    figures/scene_examples/  (qualitative panels)
"""

import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.common.seed_utils import set_all_seeds
from src.common.config_utils import load_yaml
from src.common.io_utils import get_logger, log_failure
from src.common.vis_utils import save_detection_panel
from src.overlap.overlap_model import build_overlap_model
from src.fusion.fusion_engine import (
    fuse_predicted_mask_resized, _load_crops, _estimate_bytes,
)
from src.detection.detection_utils import (
    crop_box_to_scene, stitched_box_to_scene,
    class_nms, load_scene_gt, parse_yolo_results,
    yolo_to_pixels,
)
from src.detection.scene_metrics import scene_level_metrics, per_class_metrics

LOG = get_logger("evaluate_scene_level")
SEED = 42
RESIZE_FACTOR = 0.6
MODEL_SIZE = 256
CONF_THRESH = 0.25
NMS_IOU = 0.5

CLASS_NAMES = {
    0: "Car", 1: "Pedestrian", 2: "Van", 3: "Cyclist",
    4: "Truck", 5: "Misc", 6: "Tram", 7: "Person_sitting",
}

# Scenes selected for qualitative panels
QUALITATIVE_TAGS = ["easy", "medium", "hard_seam", "failure"]


def run_odo(yolo, crop_rows: list[dict], pano_w: int, pano_h: int) -> list[dict]:
    """Run YOLO on each crop independently and merge detections to scene space."""
    all_dets = []
    for row in sorted(crop_rows, key=lambda r: r["crop_index"]):
        img = cv2.imread(row["image_path"])
        if img is None:
            continue
        crop_h, crop_w = img.shape[:2]
        xs = int(row["crop_x_start"])

        results = yolo(img, verbose=False)
        dets = parse_yolo_results(results, conf_thresh=CONF_THRESH)

        for d in dets:
            # shift x to scene coordinates
            d_scene = {
                **d,
                "x1": d["x1"] + xs,
                "x2": d["x2"] + xs,
            }
            all_dets.append(d_scene)

    # Suppress duplicates from overlapping crops
    all_dets = class_nms(all_dets, iou_thresh=NMS_IOU)
    return all_dets


def run_citron(yolo, overlap_model, device, crop_rows: list[dict],
               pano_w: int, pano_h: int) -> tuple[list[dict], np.ndarray]:
    """Run CITRON fusion + single YOLO inference."""
    fusion_result = fuse_predicted_mask_resized(
        scene_id="eval", crop_rows=crop_rows,
        overlap_model=overlap_model, device=device,
        model_input_size=MODEL_SIZE, resize_factor=RESIZE_FACTOR,
    )
    stitched = fusion_result.stitched_image
    sh, sw = stitched.shape[:2]

    results = yolo(stitched, verbose=False)
    dets = parse_yolo_results(results, conf_thresh=CONF_THRESH)

    # Convert stitched-space boxes to scene space
    scene_dets = []
    for d in dets:
        x1_p, y1_p, x2_p, y2_p = (
            d["x1"] / RESIZE_FACTOR,
            d["y1"] / RESIZE_FACTOR,
            d["x2"] / RESIZE_FACTOR,
            d["y2"] / RESIZE_FACTOR,
        )
        scene_dets.append({**d, "x1": x1_p, "y1": y1_p, "x2": x2_p, "y2": y2_p})

    return scene_dets, stitched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_weights", required=True)
    parser.add_argument("--overlap_ckpt", required=True)
    parser.add_argument("--scene_csv", required=True)
    parser.add_argument("--crop_meta", required=True)
    parser.add_argument("--out_dir", default="outputs/detection")
    parser.add_argument("--fig_dir", default="outputs/figures/scene_examples")
    parser.add_argument("--n_qual", type=int, default=4,
                        help="Number of qualitative example scenes to save")
    args = parser.parse_args()

    set_all_seeds(SEED)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    failure_log = out_dir / "eval_failures.log"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG.info("Loading YOLO detector...")
    from ultralytics import YOLO
    yolo = YOLO(args.detector_weights)

    LOG.info("Loading overlap model...")
    overlap_model = build_overlap_model("resnet50", pretrained=False)
    ckpt = torch.load(args.overlap_ckpt, map_location=device)
    overlap_model.load_state_dict(ckpt["model_state"])
    overlap_model = overlap_model.to(device)
    overlap_model.eval()

    scene_df = pd.read_csv(args.scene_csv)
    crop_meta = pd.read_csv(args.crop_meta)
    LOG.info(f"Evaluating {len(scene_df)} scenes")

    scene_rows = []
    per_class_rows = []
    odo_metrics_by_scene = []
    citron_metrics_by_scene = []

    # Track scenes for qualitative selection
    scene_odo_map50 = []

    for i, srow in scene_df.iterrows():
        scene_id = str(srow["scene_id"])
        n_veh = int(srow["vehicle_count"])

        crops_sub = crop_meta[crop_meta["scene_id"] == scene_id].sort_values("crop_index")
        if len(crops_sub) == 0:
            continue
        crop_rows = crops_sub.to_dict("records")

        pano_w = int(crops_sub.iloc[0]["panorama_width"])
        pano_h = int(crops_sub.iloc[0]["panorama_height"])

        gt_path = str(srow["scene_gt_label_path"])
        gt_boxes = load_scene_gt(gt_path, pano_w, pano_h)
        if not gt_boxes:
            continue

        try:
            # ODO
            odo_dets = run_odo(yolo, crop_rows, pano_w, pano_h)
            odo_m = scene_level_metrics(odo_dets, gt_boxes)

            # CITRON
            citron_dets, stitched_img = run_citron(
                yolo, overlap_model, device, crop_rows, pano_w, pano_h
            )
            citron_m = scene_level_metrics(citron_dets, gt_boxes)

            scene_rows.append({
                "scene_id": scene_id, "vehicle_count": n_veh,
                "method": "ODO", "n_gt": len(gt_boxes), "n_det": len(odo_dets),
                **{f"odo_{k}": v for k, v in odo_m.items()},
                **{f"citron_{k}": v for k, v in citron_m.items()},
            })

            odo_per_cls = per_class_metrics(odo_dets, gt_boxes, CLASS_NAMES)
            for r in odo_per_cls:
                per_class_rows.append({
                    "scene_id": scene_id, "method": "ODO", "vehicle_count": n_veh, **r
                })
            citron_per_cls = per_class_metrics(citron_dets, gt_boxes, CLASS_NAMES)
            for r in citron_per_cls:
                per_class_rows.append({
                    "scene_id": scene_id, "method": "CITRON", "vehicle_count": n_veh, **r
                })

            scene_odo_map50.append((scene_id, odo_m["map50"], stitched_img, crop_rows,
                                     gt_boxes, odo_dets, citron_dets, pano_h, pano_w))

            if i % 50 == 0:
                LOG.info(
                    f"[{i}/{len(scene_df)}] {scene_id} | "
                    f"ODO mAP50={odo_m['map50']:.3f} | CITRON mAP50={citron_m['map50']:.3f}"
                )

        except Exception as e:
            log_failure(scene_id, str(e), failure_log)
            LOG.error(f"Failed {scene_id}: {e}")
            traceback.print_exc()

    # Save main results
    results_df = pd.DataFrame(scene_rows)
    results_df.to_csv(out_dir / "scene_level_results.csv", index=False)
    LOG.info(f"scene_level_results.csv: {len(results_df)} rows")

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(out_dir / "scene_level_per_class_results.csv", index=False)
    LOG.info(f"scene_level_per_class_results.csv: {len(per_class_df)} rows")

    # Aggregate summary
    if len(results_df) > 0:
        for col_prefix in ["odo", "citron"]:
            for metric in ["precision", "recall", "map50", "map50_95"]:
                col = f"{col_prefix}_{metric}"
                if col in results_df.columns:
                    mean_val = results_df[col].mean()
                    LOG.info(f"{col}: {mean_val:.4f}")

    # Qualitative panels
    if scene_odo_map50:
        # Sort by ODO mAP50 to pick easy/medium/hard
        scene_odo_map50.sort(key=lambda x: x[1])
        n = len(scene_odo_map50)
        picks = []
        if n >= 4:
            picks = [
                scene_odo_map50[-1],    # best (easy)
                scene_odo_map50[n // 2],  # median
                scene_odo_map50[n // 4],  # hard
                scene_odo_map50[0],     # worst (failure)
            ]
        else:
            picks = scene_odo_map50[:min(4, n)]

        for j, (sid, map50, stitched_img, crop_rows, gt_boxes, odo_dets, citron_dets,
                pano_h, pano_w) in enumerate(picks):
            # Load panorama from first crop's x_start=0 crop (approximate)
            first_row = sorted(crop_rows, key=lambda r: r["crop_index"])[0]
            pano = cv2.imread(first_row["image_path"])
            if pano is None:
                continue

            def dets_to_xyxy(dets, clip_w, clip_h):
                arr = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets]
                                ) if dets else np.zeros((0, 4))
                return arr

            gt_arr = dets_to_xyxy(gt_boxes, pano_w, pano_h)
            odo_arr = dets_to_xyxy(odo_dets, pano_w, pano_h)
            citron_arr = dets_to_xyxy(citron_dets, stitched_img.shape[1], stitched_img.shape[0])
            tag = QUALITATIVE_TAGS[j] if j < len(QUALITATIVE_TAGS) else f"scene{j}"
            save_detection_panel(
                pano, gt_arr, odo_arr, stitched_img, citron_arr,
                f"{sid}_{tag}", fig_dir / f"{sid}_{tag}.png"
            )

    LOG.info("Evaluation complete.")


if __name__ == "__main__":
    main()
