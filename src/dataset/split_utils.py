"""
Deterministic train/val/test split utilities.
Split at scene level to prevent data leakage between crops.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def scene_level_split(
    scene_ids: list[str],
    ratios: dict,
    seed: int,
) -> dict[str, list[str]]:
    """
    Split scene_ids deterministically.
    ratios = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    Returns {'train': [...], 'val': [...], 'test': [...]}
    """
    rng = np.random.default_rng(seed)
    ids = sorted(scene_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    n_test = n - n_train - n_val

    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def filter_crop_rows_by_scene(
    crop_meta_df: pd.DataFrame,
    scene_ids: list[str],
) -> pd.DataFrame:
    return crop_meta_df[crop_meta_df["scene_id"].isin(set(scene_ids))].reset_index(drop=True)


def filter_scene_rows(
    scene_meta_df: pd.DataFrame,
    scene_ids: list[str],
) -> pd.DataFrame:
    return scene_meta_df[scene_meta_df["scene_id"].isin(set(scene_ids))].reset_index(drop=True)


def save_split_csvs(
    crop_meta: pd.DataFrame,
    scene_meta: pd.DataFrame,
    overlap_splits: dict,
    detect_splits: dict,
    out_dir: str | Path,
    force: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(df, name):
        p = out_dir / name
        if p.exists() and not force:
            print(f"  [split_utils] Exists, skipping: {p}")
            return
        df.to_csv(p, index=False)
        print(f"  [split_utils] Saved {len(df)} rows -> {p}")

    for split_name in ["train", "val", "test"]:
        ids = overlap_splits.get(split_name, [])
        df = filter_crop_rows_by_scene(crop_meta, ids)
        _save(df, f"overlap_{split_name}.csv")

    for split_name in ["train", "val"]:
        ids = detect_splits.get(split_name, [])
        df = filter_crop_rows_by_scene(crop_meta, ids)
        _save(df, f"detect_{split_name}.csv")

    # scene_test: use the overlap test split (consistent eval set)
    scene_test_ids = overlap_splits.get("test", [])
    df = filter_scene_rows(scene_meta, scene_test_ids)
    _save(df, "scene_test.csv")


def build_dataset_split_summary(
    scene_meta: pd.DataFrame,
    overlap_splits: dict,
    detect_splits: dict,
) -> pd.DataFrame:
    rows = []
    for veh in sorted(scene_meta["vehicle_count"].unique()):
        sub = scene_meta[scene_meta["vehicle_count"] == veh]
        ids = set(sub["scene_id"])
        for split_type, splits in [("overlap", overlap_splits), ("detect", detect_splits)]:
            for split_name, split_ids in splits.items():
                n = len(ids & set(split_ids))
                rows.append({
                    "vehicle_count": veh,
                    "split_type": split_type,
                    "split_name": split_name,
                    "n_scenes": n,
                })
    return pd.DataFrame(rows)
