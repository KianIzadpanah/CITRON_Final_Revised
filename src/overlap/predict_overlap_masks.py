"""
predict_overlap_masks.py — Phase B inference

For every adjacent pair in scene_test.csv, saves:
    {scene_id}_mask{k}_{k+1}_pred_prob.png
    {scene_id}_mask{k}_{k+1}_pred_bin.png
    {scene_id}_mask{k}_{k+1}_overlay.png

Usage:
    python src/overlap/predict_overlap_masks.py \
        --config configs/overlap_resnet50.yaml \
        --checkpoint outputs/overlap/overlap_resnet50_best.pt \
        --scene_csv data/processed/metadata/scene_test.csv \
        --out_dir outputs/overlap/predicted_masks
"""

from src.overlap.overlap_model import build_overlap_model
from src.common.io_utils import get_logger
from src.common.config_utils import load_yaml
from src.common.seed_utils import set_all_seeds
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))


LOG = get_logger("predict_overlap_masks")
SEED = 42


def load_img_tensor(path: str, size: int) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        return torch.zeros(3, size, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    t = transforms.ToTensor()(img)
    t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overlap_resnet50.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene_csv", required=True)
    parser.add_argument("--out_dir", default="outputs/overlap/predicted_masks")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    set_all_seeds(SEED)
    cfg = load_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_overlap_model(cfg["model"], pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    LOG.info(f"Loaded model from {args.checkpoint}")

    scene_df = pd.read_csv(args.scene_csv)
    crop_meta_path = Path(cfg["data"]["split_dir"]) / "crop_metadata.csv"
    if not crop_meta_path.exists():
        # Try relative path
        crop_meta_path = Path(cfg["data"]["split_dir"]
                              ).parent / "crop_metadata.csv"
    crop_meta = pd.read_csv(crop_meta_path)

    # Remap stored absolute paths to the configured crop_dir (handles stale paths
    # from a previous session, e.g. /kaggle/working/ → /kaggle/input/...).
    crop_dir = cfg["data"].get("crop_dir")
    if crop_dir:
        crop_dir = Path(crop_dir)
        crop_meta["image_path"] = crop_meta["image_path"].apply(
            lambda p: str(
                crop_dir / Path(p).name) if isinstance(p, str) and p else p
        )

    size = cfg["input_size"]
    threshold = args.threshold

    processed = 0
    for _, srow in scene_df.iterrows():
        scene_id = srow["scene_id"]
        crops = crop_meta[crop_meta["scene_id"]
                          == scene_id].sort_values("crop_index")
        if len(crops) < 2:
            continue
        for i in range(len(crops) - 1):
            row_l = crops.iloc[i]
            row_r = crops.iloc[i + 1]
            k_l = int(row_l["crop_index"])
            k_r = int(row_r["crop_index"])

            img_l = load_img_tensor(row_l["image_path"], size)
            img_r = load_img_tensor(row_r["image_path"], size)

            with torch.no_grad():
                pred = model(img_l.unsqueeze(0).to(device),
                             img_r.unsqueeze(0).to(device)).cpu().squeeze().numpy()

            pred_bin = (pred >= threshold).astype(np.uint8) * 255

            # Save prob map
            prob_vis = (pred * 255).astype(np.uint8)
            prob_colored = cv2.applyColorMap(prob_vis, cv2.COLORMAP_HOT)
            cv2.imwrite(
                str(out_dir / f"{scene_id}_mask{k_l}_{k_r}_pred_prob.png"), prob_colored)

            # Save binary mask
            cv2.imwrite(
                str(out_dir / f"{scene_id}_mask{k_l}_{k_r}_pred_bin.png"), pred_bin)

            # Overlay on left crop
            crop_img = cv2.imread(row_l["image_path"])
            if crop_img is not None:
                crop_resized = cv2.resize(crop_img, (size, size))
                mask_3ch = cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(crop_resized, 0.6, mask_3ch, 0.4, 0)
                cv2.imwrite(
                    str(out_dir / f"{scene_id}_mask{k_l}_{k_r}_overlay.png"), overlay)

            processed += 1

    LOG.info(f"Predicted {processed} pairs -> {out_dir}")


if __name__ == "__main__":
    main()
