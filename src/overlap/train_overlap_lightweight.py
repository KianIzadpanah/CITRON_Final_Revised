"""
train_overlap_lightweight.py — Phase F

Trains the Siamese MobileNetV3-Large overlap predictor.
Identical training setup to ResNet-50; only the encoder changes.

Usage:
    python src/overlap/train_overlap_lightweight.py --config configs/overlap_mobilenet.yaml [--force]

Outputs (under outputs/overlap/):
    overlap_mobilenet_best.pt
    overlap_mobilenet_train_curves.csv
    overlap_mobilenet_test_metrics.csv
    qc_panels_mobilenet/
    overlap_backbone_results.csv  (comparison table ResNet-50 vs MobileNet)
"""

from src.overlap.train_overlap_resnet50 import OverlapPairDataset, run_epoch
from src.overlap.overlap_metrics import compute_metrics, aggregate_metrics
from src.overlap.overlap_losses import BCEDiceLoss
from src.overlap.overlap_model import build_overlap_model, count_parameters
from src.common.io_utils import get_logger, guard_overwrite
from src.common.config_utils import load_yaml, save_yaml, copy_config
from src.common.seed_utils import set_all_seeds
import sys
import argparse
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))


LOG = get_logger("train_overlap_lightweight")
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overlap_mobilenet.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_all_seeds(SEED)

    out_dir = Path(cfg["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "qc_panels_mobilenet"
    qc_dir.mkdir(exist_ok=True)

    ckpt_path = out_dir / cfg["checkpoint_name"]
    curves_path = out_dir / "overlap_mobilenet_train_curves.csv"
    test_metrics_path = out_dir / "overlap_mobilenet_test_metrics.csv"

    if not guard_overwrite(ckpt_path, args.force):
        LOG.info("Checkpoint exists, skipping training.")
    else:
        split_dir = Path(cfg["data"]["split_dir"])
        crop_dir = cfg["data"].get("crop_dir")
        train_ds = OverlapPairDataset(split_dir / cfg["data"]["train_csv"],
                                      input_size=cfg["input_size"],
                                      crop_dir=crop_dir)
        val_ds = OverlapPairDataset(split_dir / cfg["data"]["val_csv"],
                                    input_size=cfg["input_size"],
                                    crop_dir=crop_dir)
        train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)

        LOG.info(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_overlap_model(
            cfg["model"], pretrained=cfg["encoder_pretrained"])
        model = model.to(device)
        LOG.info(
            f"Model: {cfg['model']} | Params: {count_parameters(model):,}")

        criterion = BCEDiceLoss(
            bce_weight=cfg["loss"]["bce_weight"],
            dice_weight=cfg["loss"]["dice_weight"],
            eps=cfg["loss"]["dice_eps"],
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["training"]["lr"])
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
            tr_loss, tr_m = run_epoch(
                model, train_dl, criterion, optimizer, device, True)
            vl_loss, vl_m = run_epoch(
                model, val_dl, criterion, optimizer, device, False)
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

    # Test evaluation
    split_dir = Path(cfg["data"]["split_dir"])
    crop_dir = cfg["data"].get("crop_dir")
    test_ds = OverlapPairDataset(split_dir / cfg["data"]["test_csv"],
                                 input_size=cfg["input_size"],
                                 crop_dir=crop_dir)
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

    # Measure inference time
    import time
    model.eval()
    dummy_l = torch.randn(
        1, 3, cfg["input_size"], cfg["input_size"]).to(device)
    dummy_r = torch.randn(
        1, 3, cfg["input_size"], cfg["input_size"]).to(device)
    times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.time()
            _ = model(dummy_l, dummy_r)
            times.append(time.time() - t0)
    avg_infer_ms = float(np.mean(times[10:])) * 1000  # skip warmup

    n_params = count_parameters(model)
    model_size_mb = ckpt_path.stat().st_size / 1e6

    tst_row = {
        "model": cfg["model"],
        "test_loss": tst_loss,
        **tst_m,
        "n_params_M": round(n_params / 1e6, 2),
        "model_size_mb": round(model_size_mb, 2),
        "avg_infer_time_ms": round(avg_infer_ms, 2),
    }
    pd.DataFrame([tst_row]).to_csv(test_metrics_path, index=False)
    LOG.info(f"Test metrics: {tst_row}")

    # Compile backbone comparison table
    resnet_path = out_dir / "overlap_resnet50_test_metrics.csv"
    rows = [tst_row]
    if resnet_path.exists():
        rn_df = pd.read_csv(resnet_path)
        rows = [rn_df.iloc[0].to_dict(), tst_row]

    backbone_df = pd.DataFrame(rows)
    backbone_df.to_csv(out_dir / "overlap_backbone_results.csv", index=False)
    LOG.info(
        f"Backbone comparison saved: {out_dir / 'overlap_backbone_results.csv'}")

    copy_config(args.config, out_dir)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
