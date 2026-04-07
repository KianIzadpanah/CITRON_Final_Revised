"""
build_citron_dataset.py — Phase A dataset builder

Usage:
    python src/dataset/build_citron_dataset.py --config configs/dataset.yaml --mode overlap [--force]
    python src/dataset/build_citron_dataset.py --config configs/dataset.yaml --mode scene  [--force]

Modes:
    overlap  — Mode 1: overlap-model training data (with per-crop augmentation)
    scene    — Mode 2: scene-evaluation data (no independent geometric augmentation)

Outputs (under data/processed/):
    crops/          {scene_id}_crop{k}.png / .txt
    crops/          {scene_id}_mask{k}_{k+1}.png
    scenes/         {scene_id}_gt.txt
    metadata/       crop_metadata.csv, scene_metadata.csv
    metadata/       overlap_{train,val,test}.csv, detect_{train,val}.csv, scene_test.csv
outputs/dataset_summary/
    dataset_split_summary.csv
    qc_report.json
    qc_panels/      {scene_id}_qc.png  (20 random panels)
"""

import sys
import os
import argparse
import json
import shutil
import traceback
from pathlib import Path

import numpy as np
import cv2
import pandas as pd

# Allow running from repo root
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.common.seed_utils import set_all_seeds
from src.common.config_utils import load_yaml, save_json, copy_config
from src.common.io_utils import get_logger, guard_overwrite, log_failure
from src.common.vis_utils import save_qc_panel
from src.dataset.geometry_utils import (
    compute_crop_geometries, build_overlap_mask,
    clip_boxes_to_crop, kitti_box_to_yolo,
)
from src.dataset.dataset_utils import (
    load_kitti_tracking_labels, list_sequence_frames, load_image,
    augment_crop,
)
from src.dataset.split_utils import (
    scene_level_split, save_split_csvs,
    build_dataset_split_summary,
)


LOG = get_logger("build_citron_dataset")


# ---------------------------------------------------------------------------
# Scene ID convention
# ---------------------------------------------------------------------------

def make_scene_id(seq_id: int, frame_id: int, n_veh: int) -> str:
    return f"seq{seq_id:04d}_frame{frame_id:06d}_veh{n_veh}"


# ---------------------------------------------------------------------------
# Core per-frame processor
# ---------------------------------------------------------------------------

def process_frame(
    seq_id: int,
    frame_id: int,
    n_veh: int,
    img: np.ndarray,
    kitti_labels: list[tuple],  # [(cls, x1, y1, x2, y2), ...]
    cfg: dict,
    mode: str,
    crops_dir: Path,
    scenes_dir: Path,
    rng: np.random.Generator,
    failure_log: Path,
) -> dict | None:
    """Process one KITTI frame and return metadata dict."""
    scene_id = make_scene_id(seq_id, frame_id, n_veh)
    pano_h, pano_w = img.shape[:2]
    ignore_classes = set(cfg.get("ignore_classes", ["DontCare"]))
    class_map: dict = cfg["class_map"]

    # Sample overlap ratio
    rho_min, rho_max = cfg["overlap_ratio_range"]
    rho = float(rng.uniform(rho_min, rho_max))

    # Compute crop geometries
    geoms = compute_crop_geometries(pano_w, pano_h, n_veh, rho)

    crop_rows = []
    saved_crops = []

    for geom in geoms:
        k = geom.crop_index
        xs, xe = geom.x_start, geom.x_end
        crop_img = img[:, xs:xe].copy()

        # Clip boxes to this crop
        valid_boxes = clip_boxes_to_crop(kitti_labels, xs, xe, pano_h)

        # Build masks for adjacent pairs (left neighbor)
        mask_left = None
        mask_path_left = ""
        if k > 0:
            mask_left = build_overlap_mask(pano_w, pano_h, geoms[k - 1], geom,
                                           target_crop="right")
        mask_right = None
        mask_path_right = ""
        if k < n_veh - 1:
            mask_right = build_overlap_mask(pano_w, pano_h, geom, geoms[k + 1],
                                            target_crop="left")

        # Mode 1: apply augmentation
        rotation_deg, scale_factor, blur_type, blur_applied = 0.0, 1.0, "none", False
        if mode == "overlap":
            aug_mask = mask_right if mask_right is not None else (
                mask_left if mask_left is not None else
                np.zeros((crop_img.shape[0], crop_img.shape[1]), dtype=np.uint8)
            )
            aug_img, aug_mask_out, aug_boxes = augment_crop(
                crop_img, aug_mask, valid_boxes,
                cfg["augmentation_probs"],
                tuple(cfg["scale_range"]),
                tuple(cfg["rotate_range_deg"]),
                tuple(cfg["blur_kernel_range"]),
                rng,
            )
            crop_img = aug_img
            valid_boxes = aug_boxes
            # (simplified: mask_right updated with aug)
            if mask_right is not None:
                mask_right = aug_mask_out

        crop_h, crop_w = crop_img.shape[:2]

        # Save crop image
        img_name = f"{scene_id}_crop{k}.png"
        img_path = crops_dir / img_name
        cv2.imwrite(str(img_path), crop_img)

        # Save YOLO labels for this crop
        lbl_name = f"{scene_id}_crop{k}.txt"
        lbl_path = crops_dir / lbl_name
        with open(lbl_path, "w") as f:
            for cls, x1, y1, x2, y2 in valid_boxes:
                if cls not in class_map:
                    continue
                cls_id = class_map[cls]
                cx, cy, bw, bh = kitti_box_to_yolo(x1, y1, x2, y2, crop_w, crop_h)
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                bw = max(0, min(1, bw))
                bh = max(0, min(1, bh))
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # Save masks
        if mask_right is not None:
            m_name = f"{scene_id}_mask{k}_{k+1}.png"
            m_path = crops_dir / m_name
            cv2.imwrite(str(m_path), mask_right)
            mask_path_right = str(m_path)

        if mask_left is not None:
            m_name = f"{scene_id}_mask{k-1}_{k}.png"
            m_path = crops_dir / m_name
            if not m_path.exists():
                cv2.imwrite(str(m_path), mask_left)
            mask_path_left = str(m_path)

        crop_rows.append({
            "scene_id": scene_id,
            "sequence_id": seq_id,
            "frame_id": frame_id,
            "vehicle_count": n_veh,
            "crop_index": k,
            "crop_x_start": xs,
            "crop_x_end": xe,
            "crop_width": crop_w,
            "crop_height": crop_h,
            "panorama_width": pano_w,
            "panorama_height": pano_h,
            "overlap_ratio": round(rho, 4),
            "stride": geom.stride,
            "rotation_deg": rotation_deg,
            "scale_factor": scale_factor,
            "blur_type": blur_type,
            "blur_applied": blur_applied,
            "image_path": str(img_path),
            "label_path": str(lbl_path),
            "mask_path_left": mask_path_left,
            "mask_path_right": mask_path_right,
        })
        saved_crops.append(crop_img)

    # Save scene-level GT labels (panorama coordinate system)
    gt_path = scenes_dir / f"{scene_id}_gt.txt"
    pano_img_path = ""
    with open(gt_path, "w") as f:
        for cls, x1, y1, x2, y2 in kitti_labels:
            if cls not in class_map:
                continue
            cls_id = class_map[cls]
            cx, cy, bw, bh = kitti_box_to_yolo(x1, y1, x2, y2, pano_w, pano_h)
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0, min(1, bw))
            bh = max(0, min(1, bh))
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    adjacent_pairs = [(k, k + 1) for k in range(n_veh - 1)]

    scene_row = {
        "scene_id": scene_id,
        "sequence_id": seq_id,
        "frame_id": frame_id,
        "vehicle_count": n_veh,
        "num_crops": n_veh,
        "crop_indices": list(range(n_veh)),
        "adjacent_pairs": adjacent_pairs,
        "panorama_path": "",  # raw KITTI path (filled below)
        "scene_gt_label_path": str(gt_path),
        "generation_mode": mode,
    }

    return {
        "scene_id": scene_id,
        "scene_row": scene_row,
        "crop_rows": crop_rows,
        "crops": saved_crops,
        "geoms": geoms,
        "pano": img,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--mode", choices=["overlap", "scene"], default="overlap")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = cfg["seed"]
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    mode = args.mode
    force = args.force

    output_root = Path(cfg["output_root"])
    summary_root = Path(cfg["summary_root"])
    crops_dir = output_root / "crops"
    scenes_dir = output_root / "scenes"
    meta_dir = output_root / "metadata"
    qc_dir = summary_root / "qc_panels"
    for d in [crops_dir, scenes_dir, meta_dir, qc_dir, summary_root]:
        d.mkdir(parents=True, exist_ok=True)

    failure_log = summary_root / "failures.log"
    LOG.info(f"Mode: {mode} | Seed: {seed} | Force: {force}")

    img_root = Path(cfg["kitti_img_root"])
    lbl_root = Path(cfg["kitti_label_root"])
    sequences = cfg["sequences"]
    vehicle_counts = cfg["vehicle_counts"]
    class_map = cfg["class_map"]
    ignore_classes = set(cfg.get("ignore_classes", ["DontCare"]))

    all_crop_rows = []
    all_scene_rows = []
    scene_ids_by_veh: dict[int, list] = {n: [] for n in vehicle_counts}

    total_processed = 0

    for seq_id in sequences:
        lbl_file = lbl_root / f"{seq_id:04d}.txt"
        if not lbl_file.exists():
            LOG.warning(f"No label file for seq {seq_id:04d}: {lbl_file}")
            continue

        frame_labels = load_kitti_tracking_labels(lbl_file, ignore_classes)
        frames = list_sequence_frames(img_root, seq_id)

        LOG.info(f"Seq {seq_id:04d}: {len(frames)} frames, {len(frame_labels)} labeled frames")

        for frame_id in frames:
            if frame_id not in frame_labels:
                continue  # skip frames with no valid objects

            img = load_image(img_root, seq_id, frame_id)
            if img is None:
                log_failure(f"seq{seq_id:04d}_frame{frame_id:06d}", "image not found", failure_log)
                continue

            kitti_boxes = frame_labels[frame_id]
            if len(kitti_boxes) == 0:
                continue

            for n_veh in vehicle_counts:
                scene_id = make_scene_id(seq_id, frame_id, n_veh)
                try:
                    result = process_frame(
                        seq_id, frame_id, n_veh,
                        img, kitti_boxes, cfg, mode,
                        crops_dir, scenes_dir, rng, failure_log,
                    )
                    if result is None:
                        continue
                    all_crop_rows.extend(result["crop_rows"])
                    all_scene_rows.append(result["scene_row"])
                    scene_ids_by_veh[n_veh].append(scene_id)
                    total_processed += 1
                except Exception as e:
                    log_failure(scene_id, str(e), failure_log)
                    LOG.error(f"Failed scene {scene_id}: {e}")
                    traceback.print_exc()

    LOG.info(f"Total scenes processed: {total_processed}")

    # Save metadata CSVs
    crop_meta = pd.DataFrame(all_crop_rows)
    scene_meta = pd.DataFrame(all_scene_rows)

    crop_meta_path = meta_dir / "crop_metadata.csv"
    scene_meta_path = meta_dir / "scene_metadata.csv"

    crop_meta.to_csv(crop_meta_path, index=False)
    scene_meta.to_csv(scene_meta_path, index=False)
    LOG.info(f"crop_metadata.csv: {len(crop_meta)} rows")
    LOG.info(f"scene_metadata.csv: {len(scene_meta)} rows")

    # Build splits
    # Overlap splits use all scenes; detect splits use same scene pool
    all_scene_ids = scene_meta["scene_id"].tolist()
    overlap_ratios = cfg["split_ratios"]["overlap"]
    detect_ratios = cfg["split_ratios"]["detect"]

    overlap_splits = scene_level_split(all_scene_ids, overlap_ratios, seed)
    detect_splits = scene_level_split(all_scene_ids, detect_ratios, seed + 1)

    save_split_csvs(crop_meta, scene_meta, overlap_splits, detect_splits,
                    meta_dir, force=force)

    # Dataset split summary
    split_summary = build_dataset_split_summary(scene_meta, overlap_splits, detect_splits)
    split_summary.to_csv(summary_root / "dataset_split_summary.csv", index=False)
    LOG.info("dataset_split_summary.csv saved")

    # QC report
    masks_with_overlap = 0
    total_masks = 0
    for _, row in crop_meta.iterrows():
        for mp in [row["mask_path_right"], row["mask_path_left"]]:
            if mp and Path(mp).exists():
                total_masks += 1
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None and m.max() > 0:
                    masks_with_overlap += 1
                break

    file_sizes = []
    for _, row in crop_meta.iterrows():
        p = Path(row["image_path"])
        if p.exists():
            file_sizes.append(p.stat().st_size)

    qc = {
        "seed": seed,
        "mode": mode,
        "total_scenes": len(all_scene_rows),
        "scenes_by_vehicle_count": {
            str(n): len(ids) for n, ids in scene_ids_by_veh.items()
        },
        "total_crops": len(all_crop_rows),
        "total_masks_checked": total_masks,
        "masks_with_overlap_pct": round(
            100 * masks_with_overlap / max(total_masks, 1), 2
        ),
        "overlap_ratio_min": float(crop_meta["overlap_ratio"].min()) if len(crop_meta) else 0,
        "overlap_ratio_max": float(crop_meta["overlap_ratio"].max()) if len(crop_meta) else 0,
        "overlap_ratio_mean": float(crop_meta["overlap_ratio"].mean()) if len(crop_meta) else 0,
        "crop_size_bytes_min": int(min(file_sizes)) if file_sizes else 0,
        "crop_size_bytes_max": int(max(file_sizes)) if file_sizes else 0,
        "crop_size_bytes_mean": float(np.mean(file_sizes)) if file_sizes else 0,
    }
    save_json(qc, summary_root / "qc_report.json")
    LOG.info(f"QC report: {qc}")

    # 20 random QC panels
    if len(all_scene_rows) > 0:
        sample_size = min(20, len(all_scene_rows))
        sample_indices = rng.choice(len(all_scene_rows), size=sample_size, replace=False)
        for idx in sample_indices:
            srow = all_scene_rows[idx]
            sid = srow["scene_id"]
            seq_id = srow["sequence_id"]
            frame_id = srow["frame_id"]
            n_veh = srow["vehicle_count"]
            pano = load_image(img_root, seq_id, frame_id)
            if pano is None:
                continue
            crop_subset = crop_meta[crop_meta["scene_id"] == sid].sort_values("crop_index")
            crops_imgs, masks_imgs, xs_list, xe_list = [], [], [], []
            for _, crow in crop_subset.iterrows():
                c = cv2.imread(crow["image_path"])
                crops_imgs.append(c if c is not None else np.zeros((100, 100, 3), np.uint8))
                mp = crow["mask_path_right"]
                if mp and Path(mp).exists():
                    masks_imgs.append(cv2.imread(mp, cv2.IMREAD_GRAYSCALE))
                else:
                    masks_imgs.append(None)
                xs_list.append(int(crow["crop_x_start"]))
                xe_list.append(int(crow["crop_x_end"]))
            save_qc_panel(pano, crops_imgs, masks_imgs, xs_list, xe_list,
                          sid, qc_dir / f"{sid}_qc.png")
        LOG.info(f"Saved {sample_size} QC panels to {qc_dir}")

    # Save run config
    copy_config(args.config, summary_root)
    LOG.info("Dataset build complete.")


if __name__ == "__main__":
    main()
