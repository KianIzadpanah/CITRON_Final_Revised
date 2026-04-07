"""
train_overlap_resnet50.py — Phase B

Trains the Siamese ResNet-50 overlap predictor.

Usage:
    python src/overlap/train_overlap_resnet50.py --config configs/overlap_resnet50.yaml [--force]

Outputs (under outputs/overlap/):
    overlap_resnet50_best.pt
    overlap_resnet50_train_curves.csv
    overlap_resnet50_test_metrics.csv
    qc_panels/  (20 test panels)
    run_config_resnet50.yaml
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.common.seed_utils import set_all_seeds
from src.common.config_utils import load_yaml, save_yaml, copy_config
from src.common.io_utils import get_logger, guard_overwrite
from src.overlap.overlap_model import build_overlap_model, count_parameters
from src.overlap.overlap_losses import BCEDiceLoss
from src.overlap.overlap_metrics import compute_metrics, aggregate_metrics

LOG = get_logger("train_overlap_resnet50")
SEED = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OverlapPairDataset(Dataset):
    """
    Loads adjacent crop pairs for overlap mask training.
    Each row in the CSV must have: image_path, mask_path_right
    (i.e., the left crop's right-side mask, which is the mask for crop k
     and its right neighbor k+1 in left-crop coordinates).
    """

    def __init__(self, csv_path: str | Path, input_size: int = 256, augment: bool = False):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {csv_path}\n"
                "Run Phase A first: python src/dataset/build_citron_dataset.py "
                "--config configs/dataset.yaml --mode overlap"
            )
        self.df = pd.read_csv(csv_path)
        if len(self.df) == 0:
            raise ValueError(
                f"Split CSV is empty: {csv_path}\n"
                "Phase A may have processed 0 scenes. Check that the KITTI paths in "
                "configs/dataset.yaml are correct and that Phase A completed successfully."
            )

        # Keep only crops that have a right neighbour (all except the last crop per scene)
        mask_col = self.df["mask_path_right"]
        self.df = self.df[mask_col.notna() & (mask_col != "")].reset_index(drop=True)

        self.input_size = input_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        # Build pairs: (left_crop, right_crop, mask)
        self.pairs = self._build_pairs()

        if len(self.pairs) == 0:
            # Diagnose why
            n_missing = sum(
                1 for p in self._build_pairs_raw()
                if not Path(p["mask"]).exists()
            )
            raise ValueError(
                f"OverlapPairDataset built 0 pairs from {csv_path}\n"
                f"  Rows with valid mask_path_right in CSV: {len(self.df)}\n"
                f"  Mask files missing on disk: {n_missing}\n"
                "Possible causes:\n"
                "  1. Phase A was run in a previous Kaggle session and /kaggle/working/ was wiped. "
                "Re-run Phase A.\n"
                "  2. The KITTI image/label directories in configs/dataset.yaml point to a wrong path "
                "so Phase A processed 0 frames.\n"
                "  3. Masks were not written during Phase A (check for errors in Phase A output)."
            )

    def _build_pairs_raw(self):
        """Same as _build_pairs but without existence check — used only for diagnostics."""
        pairs = []
        for _, group in self.df.groupby("scene_id"):
            group = group.sort_values("crop_index").reset_index(drop=True)
            for i in range(len(group) - 1):
                row_l = group.iloc[i]
                row_r = group.iloc[i + 1]
                pairs.append({"img_l": row_l["image_path"],
                               "img_r": row_r["image_path"],
                               "mask": row_l["mask_path_right"]})
        return pairs

    def _build_pairs(self):
        pairs = []
        missing_masks = 0
        grouped = self.df.groupby("scene_id")
        for scene_id, group in grouped:
            group = group.sort_values("crop_index").reset_index(drop=True)
            for i in range(len(group) - 1):
                row_l = group.iloc[i]
                row_r = group.iloc[i + 1]
                mask_path = row_l["mask_path_right"]
                # Do NOT skip on missing file — _load_mask returns zeros gracefully.
                # Log missing masks but still train so the session does not abort.
                if mask_path and not Path(mask_path).exists():
                    missing_masks += 1
                pairs.append({
                    "img_l": row_l["image_path"],
                    "img_r": row_r["image_path"],
                    "mask": mask_path,
                    "scene_id": scene_id,
                    "crop_l": int(row_l["crop_index"]),
                    "crop_r": int(row_r["crop_index"]),
                })
        if missing_masks > 0:
            import logging
            logging.getLogger("OverlapPairDataset").warning(
                f"{missing_masks}/{len(pairs)} mask files not found on disk — "
                "zero masks will be used for those pairs. "
                "Re-run Phase A to regenerate masks."
            )
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img_l = self._load_img(p["img_l"])
        img_r = self._load_img(p["img_r"])
        mask = self._load_mask(p["mask"])
        return img_l, img_r, mask

    def _load_img(self, path: str) -> torch.Tensor:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        t = self.to_tensor(img)
        # Normalize with ImageNet stats
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t

    def _load_mask(self, path: str) -> torch.Tensor:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((self.input_size, self.input_size), np.uint8)
        m = cv2.resize(m, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.float32)
        return torch.from_numpy(m).unsqueeze(0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_metrics = []

    with torch.set_grad_enabled(training):
        for img1, img2, mask in loader:
            img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)
            pred = model(img1, img2)
            loss = criterion(pred, mask)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * img1.size(0)
            all_metrics.append(compute_metrics(pred.detach(), mask.detach()))

    avg_loss = total_loss / max(len(loader.dataset), 1)
    avg_m = aggregate_metrics(all_metrics)
    return avg_loss, avg_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overlap_resnet50.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_all_seeds(SEED)

    out_dir = Path(cfg["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "qc_panels_resnet50"
    qc_dir.mkdir(exist_ok=True)

    ckpt_path = out_dir / cfg["checkpoint_name"]
    curves_path = out_dir / "overlap_resnet50_train_curves.csv"
    test_metrics_path = out_dir / "overlap_resnet50_test_metrics.csv"

    if not guard_overwrite(ckpt_path, args.force):
        LOG.info("Checkpoint exists, skipping training.")
    else:
        # Build datasets
        split_dir = Path(cfg["data"]["split_dir"])
        train_ds = OverlapPairDataset(split_dir / cfg["data"]["train_csv"],
                                      input_size=cfg["input_size"])
        val_ds = OverlapPairDataset(split_dir / cfg["data"]["val_csv"],
                                    input_size=cfg["input_size"])

        train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)

        LOG.info(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_overlap_model(cfg["model"], pretrained=cfg["encoder_pretrained"])
        model = model.to(device)
        LOG.info(f"Model: {cfg['model']} | Params: {count_parameters(model):,}")

        criterion = BCEDiceLoss(
            bce_weight=cfg["loss"]["bce_weight"],
            dice_weight=cfg["loss"]["dice_weight"],
            eps=cfg["loss"]["dice_eps"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=cfg["training"]["scheduler_patience"],
            factor=cfg["training"]["scheduler_factor"],
            min_lr=cfg["training"]["min_lr"],
        )

        best_val_loss = float("inf")
        patience_counter = 0
        records = []

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_m = run_epoch(model, train_dl, criterion, optimizer, device, True)
            vl_loss, vl_m = run_epoch(model, val_dl, criterion, optimizer, device, False)
            scheduler.step(vl_loss)
            elapsed = time.time() - t0

            records.append({
                "epoch": epoch,
                "train_loss": tr_loss, "val_loss": vl_loss,
                **{f"train_{k}": v for k, v in tr_m.items()},
                **{f"val_{k}": v for k, v in vl_m.items()},
            })
            LOG.info(
                f"Epoch {epoch:03d}/{cfg['training']['epochs']} | "
                f"tr_loss={tr_loss:.4f} vl_loss={vl_loss:.4f} | "
                f"vl_dice={vl_m.get('dice', 0):.4f} | {elapsed:.1f}s"
            )

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "val_loss": vl_loss, "val_metrics": vl_m}, ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["training"]["early_stop_patience"]:
                    LOG.info(f"Early stop at epoch {epoch}")
                    break

        pd.DataFrame(records).to_csv(curves_path, index=False)
        LOG.info(f"Training curves saved: {curves_path}")

    # --- Evaluation on test set ---
    split_dir = Path(cfg["data"]["split_dir"])
    test_ds = OverlapPairDataset(split_dir / cfg["data"]["test_csv"],
                                  input_size=cfg["input_size"])
    test_dl = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                         shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_overlap_model(cfg["model"], pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    criterion = BCEDiceLoss()
    tst_loss, tst_m = run_epoch(model, test_dl, criterion, None, device, False)
    LOG.info(f"Test loss={tst_loss:.4f} | {tst_m}")

    tst_df = pd.DataFrame([{"model": cfg["model"], "test_loss": tst_loss, **tst_m}])
    tst_df.to_csv(test_metrics_path, index=False)
    LOG.info(f"Test metrics saved: {test_metrics_path}")

    # --- QC panels (20 random test pairs) ---
    model.eval()
    pairs_to_show = min(20, len(test_ds))
    indices = np.random.choice(len(test_ds), pairs_to_show, replace=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i, idx in enumerate(indices):
        img1_t, img2_t, mask_t = test_ds[idx]
        with torch.no_grad():
            pred = model(img1_t.unsqueeze(0).to(device),
                         img2_t.unsqueeze(0).to(device)).cpu().squeeze()
        pred_bin = (pred >= cfg["training"]["threshold"]).float()

        def denorm(t):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(denorm(img1_t)); axes[0].set_title("Crop A"); axes[0].axis("off")
        axes[1].imshow(denorm(img2_t)); axes[1].set_title("Crop B"); axes[1].axis("off")
        axes[2].imshow(mask_t.squeeze(), cmap="gray"); axes[2].set_title("GT mask"); axes[2].axis("off")
        axes[3].imshow(pred.numpy(), cmap="hot"); axes[3].set_title("Pred prob"); axes[3].axis("off")
        diff = (pred_bin.numpy() != mask_t.squeeze().numpy()).astype(float)
        axes[4].imshow(diff, cmap="bwr"); axes[4].set_title("Diff (pred vs GT)"); axes[4].axis("off")
        plt.tight_layout()
        plt.savefig(qc_dir / f"resnet50_qc_{i:03d}.png", dpi=80)
        plt.close(fig)

    LOG.info(f"QC panels saved to {qc_dir}")
    copy_config(args.config, out_dir)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
