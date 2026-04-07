"""
leader_delay_estimator.py — Phase G

Implements the transparent leader-selection heuristic that addresses
Reviewer 2's question: how can the system estimate processing delay
BEFORE the leader is selected and BEFORE images are received?

Answer: each vehicle maintains rolling averages of recent overlap-inference
time and fusion time from previous cloudlet formations. It broadcasts these
estimates in the leader-advertisement beacon.

Formula:
    D_hat_proc(v) = q_v * t_pair_avg(v) + (N-1) * t_pair_avg(v) + t_fuse_avg(v)

Leader selection:
    v* = argmin_v [D_hat_proc(v) + D_V2I_estimated(v)]

Usage:
    python src/simulation/leader_delay_estimator.py --config configs/network.yaml

Outputs:
    outputs/simulation/leader_delay_estimation_results.csv
"""

import sys
import argparse
import json
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.common.config_utils import load_yaml, save_json
from src.common.io_utils import get_logger
from src.simulation.simulation_utils import (
    V2ILink, ScenarioConstants, bytes_to_mb, propagation_delay,
)

LOG = get_logger("leader_delay_estimator")


# ---------------------------------------------------------------------------
# Vehicle state model
# ---------------------------------------------------------------------------

class VehicleState:
    """
    Maintains rolling averages for a single vehicle candidate.
    In practice these would be computed from recent CloudletEvent logs.
    """

    def __init__(self, vehicle_id: int, window: int = 10):
        self.vid = vehicle_id
        self._pair_times = deque(maxlen=window)
        self._fuse_times = deque(maxlen=window)
        self.queue_length = 0  # pending overlap tasks

    def record_pair_inference(self, t: float):
        self._pair_times.append(t)

    def record_fusion(self, t: float):
        self._fuse_times.append(t)

    @property
    def t_pair_avg(self) -> float:
        return float(np.mean(self._pair_times)) if self._pair_times else 0.00305  # fallback to measured

    @property
    def t_fuse_avg(self) -> float:
        return float(np.mean(self._fuse_times)) if self._fuse_times else 0.001

    def estimate_proc_delay(self, n_vehicles: int) -> float:
        """
        D_hat_proc(v) = q_v * t_pair_avg + (N-1) * t_pair_avg + t_fuse_avg
        """
        return (self.queue_length * self.t_pair_avg
                + (n_vehicles - 1) * self.t_pair_avg
                + self.t_fuse_avg)


def estimate_v2i_delay(
    v2i: V2ILink, sc: ScenarioConstants,
    n_vehicles: int, overlap_fraction: float, resize_factor: float,
) -> float:
    S_crop = sc.avg_crop_size_bytes
    P = n_vehicles - 1
    S_overlap = overlap_fraction * S_crop
    S_fused = resize_factor * (n_vehicles * S_crop - P * S_overlap)
    S_fused_mb = bytes_to_mb(S_fused)
    prop = propagation_delay(sc.d_vehicle_to_cloud_km, sc.speed_of_light_fiber_km_s)
    return S_fused_mb / v2i.bandwidth_mbps + v2i.latency_s + prop


def select_leader(
    vehicles: list[VehicleState],
    n_vehicles: int,
    v2i: V2ILink,
    sc: ScenarioConstants,
) -> tuple[int, list[dict]]:
    """
    Select leader that minimises D_hat_proc + D_V2I_estimated.
    Returns (chosen_leader_vid, estimates_per_candidate).
    """
    D_v2i = estimate_v2i_delay(v2i, sc, n_vehicles,
                                sc.avg_overlap_fraction, sc.resize_factor)
    estimates = []
    for v in vehicles:
        D_proc = v.estimate_proc_delay(n_vehicles)
        D_total = D_proc + D_v2i
        estimates.append({
            "vehicle_id": v.vid,
            "queue_length": v.queue_length,
            "t_pair_avg_s": v.t_pair_avg,
            "t_fuse_avg_s": v.t_fuse_avg,
            "D_hat_proc_s": D_proc,
            "D_hat_v2i_s": D_v2i,
            "D_hat_total_s": D_total,
        })
    best = min(estimates, key=lambda e: e["D_hat_total_s"])
    return best["vehicle_id"], estimates


# ---------------------------------------------------------------------------
# Scenario simulation
# ---------------------------------------------------------------------------

def run_estimation_scenarios(
    cfg: dict,
) -> pd.DataFrame:
    """
    Simulate leader estimation for multiple network/vehicle combinations.
    Returns DataFrame with results.
    """
    mc = cfg["measured_constants"]
    sc = ScenarioConstants(
        avg_crop_size_bytes=mc["avg_crop_size_bytes"],
        avg_overlap_fraction=mc["avg_overlap_fraction"],
        avg_cloud_inference_time_s=mc["avg_cloud_inference_time_s"],
        avg_leader_inference_time_s=mc["avg_leader_inference_time_s"],
        avg_cpu_self_inference_time_s=mc["avg_cpu_self_inference_time_s"],
        avg_fusion_time_s=mc.get("avg_fusion_time_s", 0.001),
        resize_factor=mc["resize_factor"],
        d_vehicle_to_cloud_km=mc["d_vehicle_to_cloud_km"],
        d_vehicle_to_leader_km=mc["d_vehicle_to_leader_km"],
        speed_of_light_fiber_km_s=mc["speed_of_light_fiber_km_s"],
    )

    rng = np.random.default_rng(42)
    rows = []

    for v2i_name, v2i_cfg in cfg["v2i_networks"].items():
        v2i = V2ILink(
            name=v2i_name,
            bandwidth_mbps=v2i_cfg["bandwidth_mbps"],
            energy_per_mb=v2i_cfg["energy_per_mb"],
            latency_s=v2i_cfg["latency_s"],
        )

        for n_veh in [3, 5]:
            # Simulate 20 cloudlet formations per scenario
            for scene_idx in range(20):
                # Each vehicle has slightly different history (simulate heterogeneity)
                vehicles = []
                for vid in range(n_veh):
                    v = VehicleState(vid)
                    # Populate rolling window with synthetic history
                    base_t = sc.avg_leader_inference_time_s
                    for _ in range(5):
                        jitter = rng.normal(0, base_t * 0.1)
                        v.record_pair_inference(max(0.001, base_t + jitter))
                        v.record_fusion(max(0.0005, sc.avg_fusion_time_s + rng.normal(0, 0.0002)))
                    v.queue_length = int(rng.integers(0, 3))
                    vehicles.append(v)

                leader_id, estimates = select_leader(vehicles, n_veh, v2i, sc)

                for e in estimates:
                    rows.append({
                        "scene_idx": scene_idx,
                        "v2i_network": v2i_name,
                        "n_vehicles": n_veh,
                        "chosen_leader": leader_id,
                        "is_leader": e["vehicle_id"] == leader_id,
                        **e,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--out_dir", default="outputs/simulation")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Running leader delay estimation scenarios...")
    df = run_estimation_scenarios(cfg)

    out_path = out_dir / "leader_delay_estimation_results.csv"
    df.to_csv(out_path, index=False)
    LOG.info(f"Saved {len(df)} rows -> {out_path}")

    # Summary statistics
    summary = {}
    for (v2i, n_veh), grp in df[df["is_leader"]].groupby(["v2i_network", "n_vehicles"]):
        summary[f"{v2i}_N{n_veh}"] = {
            "mean_proc_delay_s": round(float(grp["D_hat_proc_s"].mean()), 5),
            "mean_total_delay_s": round(float(grp["D_hat_total_s"].mean()), 5),
            "mean_queue_length": round(float(grp["queue_length"].mean()), 2),
        }
    save_json(summary, out_dir / "leader_delay_summary.json")
    LOG.info(f"leader_delay_summary.json saved")
    LOG.info("Done.")


if __name__ == "__main__":
    main()
