"""
network_simulator.py — Phase E (complete rewrite)

Simulates delay, bandwidth, and energy for:
    ODO    — all vehicles send raw crops to cloud via V2I
    SOD    — each vehicle detects locally on CPU, no transmission
    CITRON — V2V to leader -> leader fuses -> V2I unified image to cloud

Fixes vs original network_calculations.py:
    1. 4G replaces 6G (paper Table II)
    2. Correct CITRON V2I payload: S_fused = r*(N*S_crop - (N-1)*S_overlap)
    3. N parameterized (3 and 5 vehicles)
    4. Staggered ODO transmissions with explicit cloud queueing
    5. Decomposed energy output (V2V, V2I, leader compute, cloud compute)
    6. Sensitivity sweeps over overlap fraction, resize factor, leader power, N

Usage:
    python src/simulation/network_simulator.py --config configs/network.yaml

Outputs:
    outputs/simulation/network_results.csv
    outputs/simulation/network_sensitivity_results.csv
    outputs/simulation/assumptions.json
    outputs/figures/network_plots/  (PNG plots)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.common.config_utils import load_yaml, save_json
from src.common.io_utils import get_logger
from src.simulation.simulation_utils import (
    V2ILink, V2VLink, HardwareConfig, ScenarioConstants, SimResult,
    bytes_to_mb, propagation_delay,
)

LOG = get_logger("network_simulator")


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------

def simulate_odo(
    n: int, v2i: V2ILink, hw: HardwareConfig, sc: ScenarioConstants,
) -> SimResult:
    """
    ODO: N vehicles each send one raw crop to cloud via V2I.
    Staggered: vehicle k starts at t0 + k * offset * T_single_tx.
    Cloud queues arrivals and processes sequentially.
    """
    S_crop = sc.avg_crop_size_bytes
    S_mb = bytes_to_mb(S_crop)
    bw_mbps = v2i.bandwidth_mbps
    prop = propagation_delay(sc.d_vehicle_to_cloud_km, sc.speed_of_light_fiber_km_s)

    # Single transmission time (tx duration only)
    T_tx_single = S_mb / bw_mbps  # seconds

    # Staggered start offsets
    offset = sc.staggered_tx_offset_fraction
    T_finish = []  # time each crop finishes transmission at cloud
    for k in range(n):
        t_start = k * offset * T_tx_single
        t_finish = t_start + T_tx_single + v2i.latency_s + prop
        T_finish.append(t_finish)

    # Queue at cloud: process in order of arrival
    T_process_start = T_finish[0]
    T_complete = []
    for k in range(n):
        t_start = max(T_finish[k], T_process_start)
        t_end = t_start + sc.avg_cloud_inference_time_s
        T_complete.append(t_end)
        T_process_start = t_end

    total_delay = T_complete[-1]
    tx_delay = T_finish[-1]  # last crop arrives
    compute_delay = n * sc.avg_cloud_inference_time_s
    queue_delay = total_delay - tx_delay - compute_delay

    v2i_bytes = S_crop * n
    v2i_energy = bytes_to_mb(v2i_bytes) * v2i.energy_per_mb
    cloud_compute_energy = compute_delay * hw.cloud_gpu_power_w

    r = SimResult(
        method="ODO", v2i_name=v2i.name, v2v_name="N/A", n_vehicles=n,
        total_delay_s=total_delay,
        tx_delay_s=tx_delay,
        queue_delay_s=max(0.0, queue_delay),
        compute_delay_s=compute_delay,
        v2i_bytes=v2i_bytes, v2v_bytes=0.0, total_bytes=v2i_bytes,
        v2i_energy_j=v2i_energy, cloud_compute_energy_j=cloud_compute_energy,
        total_energy_j=v2i_energy + cloud_compute_energy,
    )
    return r


def simulate_sod(
    n: int, hw: HardwareConfig, sc: ScenarioConstants,
) -> SimResult:
    """
    SOD: each vehicle runs local CPU inference independently (perfect parallelism).
    Total delay = max(inference times) = one self-inference time.
    """
    T_self = sc.avg_cpu_self_inference_time_s
    # parallel: all n vehicles run simultaneously
    total_delay = T_self

    self_compute_energy = T_self * hw.self_cpu_power_w * n

    return SimResult(
        method="SOD", v2i_name="N/A", v2v_name="N/A", n_vehicles=n,
        total_delay_s=total_delay,
        tx_delay_s=0.0, queue_delay_s=0.0, compute_delay_s=total_delay,
        v2i_bytes=0.0, v2v_bytes=0.0, total_bytes=0.0,
        self_compute_energy_j=self_compute_energy,
        total_energy_j=self_compute_energy,
    )


def simulate_citron(
    n: int, v2i: V2ILink, v2v: V2VLink, hw: HardwareConfig, sc: ScenarioConstants,
    overlap_fraction: float | None = None,
    resize_factor: float | None = None,
    leader_power_w: float | None = None,
) -> SimResult:
    """
    CITRON: (N-1) members send crops to leader via V2V,
            leader runs Siamese inference + fusion,
            leader sends unified fused image via V2I to cloud.

    Fused payload: S_fused = resize * (N * S_crop - (N-1) * S_overlap)
    """
    if overlap_fraction is None:
        overlap_fraction = sc.avg_overlap_fraction
    if resize_factor is None:
        resize_factor = sc.resize_factor
    if leader_power_w is None:
        leader_power_w = hw.leader_gpu_power_w

    S_crop = sc.avg_crop_size_bytes
    P = n - 1  # number of adjacent pairs
    S_overlap = overlap_fraction * S_crop

    # V2V: (N-1) members each send one crop to leader
    S_v2v = S_crop * P
    S_v2v_mb = bytes_to_mb(S_v2v)
    prop_v2v = propagation_delay(sc.d_vehicle_to_leader_km, sc.speed_of_light_fiber_km_s)
    T_v2v = S_v2v_mb / v2v.bandwidth_mbps + v2v.latency_s + prop_v2v

    # Leader processing: P overlap inferences + fusion
    T_proc = P * sc.avg_leader_inference_time_s + sc.avg_fusion_time_s

    # V2I: send fused image
    S_fused = resize_factor * (n * S_crop - P * S_overlap)
    S_fused_mb = bytes_to_mb(S_fused)
    prop_v2i = propagation_delay(sc.d_vehicle_to_cloud_km, sc.speed_of_light_fiber_km_s)
    T_v2i = S_fused_mb / v2i.bandwidth_mbps + v2i.latency_s + prop_v2i

    # Cloud inference
    T_cloud = sc.avg_cloud_inference_time_s

    total_delay = T_v2v + T_proc + T_v2i + T_cloud
    tx_delay = T_v2v + T_v2i
    compute_delay = T_proc + T_cloud

    v2v_energy = S_v2v_mb * v2v.energy_per_mb
    v2i_energy = S_fused_mb * v2i.energy_per_mb
    leader_compute_energy = T_proc * leader_power_w
    cloud_compute_energy = T_cloud * hw.cloud_gpu_power_w

    return SimResult(
        method="CITRON", v2i_name=v2i.name, v2v_name=v2v.name, n_vehicles=n,
        total_delay_s=total_delay,
        tx_delay_s=tx_delay, queue_delay_s=0.0, compute_delay_s=compute_delay,
        v2i_bytes=S_fused, v2v_bytes=S_v2v, total_bytes=S_fused + S_v2v,
        v2v_energy_j=v2v_energy, v2i_energy_j=v2i_energy,
        leader_compute_energy_j=leader_compute_energy,
        cloud_compute_energy_j=cloud_compute_energy,
        total_energy_j=v2v_energy + v2i_energy + leader_compute_energy + cloud_compute_energy,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(cfg: dict):
    v2i_links = {
        name: V2ILink(
            name=name,
            bandwidth_mbps=v["bandwidth_mbps"],
            energy_per_mb=v["energy_per_mb"],
            latency_s=v["latency_s"],
        )
        for name, v in cfg["v2i_networks"].items()
    }
    v2v_links = {
        name: V2VLink(
            name=name,
            bandwidth_mbps=v["bandwidth_mbps"],
            energy_per_mb=v["energy_per_mb"],
            latency_s=v["latency_s"],
        )
        for name, v in cfg["v2v_networks"].items()
    }
    hw_cfg = cfg["hardware"]
    hw = HardwareConfig(
        cloud_gpu_power_w=hw_cfg["cloud_gpu_power_w"],
        leader_gpu_power_w=hw_cfg["leader_gpu_power_w"],
        leader_cpu_power_w=hw_cfg["leader_cpu_power_w"],
        self_cpu_power_w=hw_cfg["self_cpu_power_w"],
    )
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
        staggered_tx_offset_fraction=cfg.get("staggered_tx_offset_fraction", 0.05),
    )
    return v2i_links, v2v_links, hw, sc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(results_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")
    palette = {"ODO": "#e74c3c", "SOD": "#3498db", "CITRON": "#2ecc71"}

    # 1. Total delay vs V2I network (per N)
    for n_veh in results_df["n_vehicles"].unique():
        sub = results_df[results_df["n_vehicles"] == n_veh]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = sorted(sub["v2i_network"].unique(), key=lambda x: ["3G", "4G", "5G"].index(x)
                   if x in ["3G", "4G", "5G"] else 99)
        for method in ["ODO", "SOD", "CITRON"]:
            vals = []
            for net in x:
                row = sub[(sub["method"] == method) & (sub["v2i_network"] == net)]
                vals.append(float(row["total_delay_s"].mean()) if len(row) else 0)
            ax.plot(x, vals, marker="o", label=method, color=palette.get(method))
        ax.set_title(f"Total Delay vs V2I Network (N={n_veh} vehicles)")
        ax.set_ylabel("Delay (s)")
        ax.set_xlabel("V2I Network")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"delay_vs_network_veh{n_veh}.png", dpi=120)
        plt.close(fig)

    # 2. Total energy vs V2I network
    for n_veh in results_df["n_vehicles"].unique():
        sub = results_df[results_df["n_vehicles"] == n_veh]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = sorted(sub["v2i_network"].unique(), key=lambda x: ["3G", "4G", "5G"].index(x)
                   if x in ["3G", "4G", "5G"] else 99)
        for method in ["ODO", "SOD", "CITRON"]:
            vals = []
            for net in x:
                row = sub[(sub["method"] == method) & (sub["v2i_network"] == net)]
                vals.append(float(row["total_energy_j"].mean()) if len(row) else 0)
            ax.plot(x, vals, marker="s", label=method, color=palette.get(method))
        ax.set_title(f"Total Energy vs V2I Network (N={n_veh} vehicles)")
        ax.set_ylabel("Energy (J)")
        ax.set_xlabel("V2I Network")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"energy_vs_network_veh{n_veh}.png", dpi=120)
        plt.close(fig)

    # 3. Stacked energy bars (one per network, for N=3 and N=5)
    for n_veh in results_df["n_vehicles"].unique():
        sub = results_df[(results_df["n_vehicles"] == n_veh) &
                         (results_df["v2v_network"].isin(["DSRC", "N/A"]))]
        x_nets = sorted(sub["v2i_network"].unique(), key=lambda x: ["3G", "4G", "5G"].index(x)
                        if x in ["3G", "4G", "5G"] else 99)
        methods = ["ODO", "SOD", "CITRON"]
        n_m = len(methods)
        x_pos = np.arange(len(x_nets))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        for mi, method in enumerate(methods):
            v2v_e, v2i_e, ldr_e, cld_e, slf_e = [], [], [], [], []
            for net in x_nets:
                row = sub[(sub["method"] == method) & (sub["v2i_network"] == net)]
                if len(row) == 0:
                    v2v_e.append(0); v2i_e.append(0); ldr_e.append(0); cld_e.append(0); slf_e.append(0)
                    continue
                v2v_e.append(float(row["v2v_energy_j"].mean()))
                v2i_e.append(float(row["v2i_energy_j"].mean()))
                ldr_e.append(float(row["leader_compute_energy_j"].mean()))
                cld_e.append(float(row["cloud_compute_energy_j"].mean()))
                slf_e.append(float(row["self_compute_energy_j"].mean()))
            pos = x_pos + (mi - 1) * width
            ax.bar(pos, v2i_e, width, label=f"{method} V2I" if mi == 0 else "", color=palette.get(method), alpha=0.9)
            ax.bar(pos, v2v_e, width, bottom=v2i_e, label=f"{method} V2V" if mi == 0 else "", color=palette.get(method), alpha=0.5)
            bottom2 = [a + b for a, b in zip(v2i_e, v2v_e)]
            ax.bar(pos, cld_e, width, bottom=bottom2, label=f"{method} Compute" if mi == 0 else "", color=palette.get(method), alpha=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_nets)
        ax.set_ylabel("Energy (J)")
        ax.set_title(f"Stacked Energy by Component (N={n_veh})")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"energy_stacked_veh{n_veh}.png", dpi=120)
        plt.close(fig)

    # 4. V2I bytes by method
    for n_veh in results_df["n_vehicles"].unique():
        sub = results_df[(results_df["n_vehicles"] == n_veh) &
                         (results_df["v2v_network"].isin(["DSRC", "N/A"]))]
        x_nets = sorted(sub["v2i_network"].unique(), key=lambda x: ["3G", "4G", "5G"].index(x)
                        if x in ["3G", "4G", "5G"] else 99)
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in ["ODO", "CITRON"]:
            vals = []
            for net in x_nets:
                row = sub[(sub["method"] == method) & (sub["v2i_network"] == net)]
                vals.append(float(row["v2i_bytes"].mean()) / 1e6 if len(row) else 0)
            ax.bar([f"{net}\n{method}" for net in x_nets], vals,
                   color=palette.get(method), alpha=0.8, label=method)
        ax.set_ylabel("V2I Upload (MB)")
        ax.set_title(f"V2I Payload by Method (N={n_veh})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"v2i_bytes_veh{n_veh}.png", dpi=120)
        plt.close(fig)

    # 5. Delay vs N vehicles
    x_ns = sorted(results_df["n_vehicles"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    net_fixed = "4G"
    for method in ["ODO", "SOD", "CITRON"]:
        vals = []
        for n_v in x_ns:
            sub = results_df[(results_df["method"] == method) &
                             (results_df["n_vehicles"] == n_v) &
                             (results_df["v2i_network"].isin([net_fixed, "N/A"]))]
            vals.append(float(sub["total_delay_s"].mean()) if len(sub) else 0)
        ax.plot(x_ns, vals, marker="o", label=method, color=palette.get(method))
    ax.set_xlabel("Number of Vehicles")
    ax.set_ylabel("Total Delay (s)")
    ax.set_title(f"Delay vs Number of Vehicles ({net_fixed})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "delay_vs_n_vehicles.png", dpi=120)
    plt.close(fig)

    LOG.info(f"Plots saved to {fig_dir}")


def make_sensitivity_plots(sens_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Sensitivity to overlap fraction
    if "overlap_fraction" in sens_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in ["total_delay_s", "total_energy_j"]:
            sub = sens_df[sens_df["sweep_param"] == "overlap_fraction"]
            if len(sub) == 0:
                continue
            ax.plot(sub["sweep_value"], sub[metric], marker="o", label=metric)
        ax.set_xlabel("Overlap Fraction")
        ax.set_title("CITRON Sensitivity to Overlap Fraction (4G, N=3)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "sensitivity_overlap.png", dpi=120)
        plt.close(fig)

    # Sensitivity to resize factor
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = sens_df[sens_df["sweep_param"] == "resize_factor"]
    if len(sub) > 0:
        ax.plot(sub["sweep_value"], sub["v2i_bytes"] / 1e6, marker="s", color="teal")
        ax.set_xlabel("Resize Factor")
        ax.set_ylabel("V2I Upload (MB)")
        ax.set_title("V2I Upload vs Resize Factor (CITRON, 4G, N=3)")
        plt.tight_layout()
        plt.savefig(fig_dir / "sensitivity_resize.png", dpi=120)
        plt.close(fig)

    # Sensitivity to leader GPU power
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = sens_df[sens_df["sweep_param"] == "leader_power_w"]
    if len(sub) > 0:
        ax.plot(sub["sweep_value"], sub["total_energy_j"], marker="^", color="purple")
        ax.set_xlabel("Leader GPU Power (W)")
        ax.set_ylabel("Total Energy (J)")
        ax.set_title("Total Energy vs Leader GPU Power (CITRON, 4G, N=3)")
        plt.tight_layout()
        plt.savefig(fig_dir / "sensitivity_leader_power.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--out_dir", default="outputs/simulation")
    parser.add_argument("--fig_dir", default="outputs/figures/network_plots")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    v2i_links, v2v_links, hw, sc = load_configs(cfg)

    vehicle_counts = [3, 5]
    rows = []

    for n_veh in vehicle_counts:
        for v2i_name, v2i in v2i_links.items():
            # ODO
            r = simulate_odo(n_veh, v2i, hw, sc)
            rows.append(r.to_dict())

            # SOD (no V2I/V2V network dependency for delay/energy, still vary for record)
            r = simulate_sod(n_veh, hw, sc)
            r_dict = r.to_dict()
            r_dict["v2i_network"] = v2i_name
            rows.append(r_dict)

            # CITRON with each V2V option
            for v2v_name, v2v in v2v_links.items():
                r = simulate_citron(n_veh, v2i, v2v, hw, sc)
                rows.append(r.to_dict())

    results_df = pd.DataFrame(rows)
    results_path = out_dir / "network_results.csv"
    results_df.to_csv(results_path, index=False)
    LOG.info(f"network_results.csv: {len(results_df)} rows -> {results_path}")

    # Sensitivity analysis (fixed: 4G, DSRC, N=3)
    v2i_4g = v2i_links["4G"]
    v2v_dsrc = v2v_links["DSRC"]
    sens_rows = []
    sensitivity = cfg.get("sensitivity", {})

    for ov in sensitivity.get("overlap_fractions", []):
        r = simulate_citron(3, v2i_4g, v2v_dsrc, hw, sc, overlap_fraction=ov)
        d = r.to_dict()
        d["sweep_param"] = "overlap_fraction"
        d["sweep_value"] = ov
        sens_rows.append(d)

    for rf in sensitivity.get("resize_factors", []):
        r = simulate_citron(3, v2i_4g, v2v_dsrc, hw, sc, resize_factor=rf)
        d = r.to_dict()
        d["sweep_param"] = "resize_factor"
        d["sweep_value"] = rf
        sens_rows.append(d)

    for pw in sensitivity.get("leader_gpu_powers_w", []):
        r = simulate_citron(3, v2i_4g, v2v_dsrc, hw, sc, leader_power_w=pw)
        d = r.to_dict()
        d["sweep_param"] = "leader_power_w"
        d["sweep_value"] = pw
        sens_rows.append(d)

    for n_v in sensitivity.get("vehicle_counts", []):
        for method_fn, method_name in [
            (lambda n: simulate_odo(n, v2i_4g, hw, sc), "ODO"),
            (lambda n: simulate_sod(n, hw, sc), "SOD"),
            (lambda n: simulate_citron(n, v2i_4g, v2v_dsrc, hw, sc), "CITRON"),
        ]:
            r = method_fn(n_v)
            d = r.to_dict()
            d["sweep_param"] = "n_vehicles"
            d["sweep_value"] = n_v
            sens_rows.append(d)

    sens_df = pd.DataFrame(sens_rows)
    sens_path = out_dir / "network_sensitivity_results.csv"
    sens_df.to_csv(sens_path, index=False)
    LOG.info(f"network_sensitivity_results.csv: {len(sens_df)} rows -> {sens_path}")

    # Save assumptions
    assumptions = {
        "v2i_networks": cfg["v2i_networks"],
        "v2v_networks": cfg["v2v_networks"],
        "hardware": cfg["hardware"],
        "measured_constants": cfg["measured_constants"],
        "staggered_tx_offset_fraction": cfg.get("staggered_tx_offset_fraction", 0.05),
        "note_4g": "4G added; 6G removed to match paper Table II (3G/4G/5G)",
        "note_citron_payload": (
            "S_fused = resize_factor * (N * S_crop - (N-1) * S_overlap); "
            "fix from original script that returned ODO payload for all methods"
        ),
    }
    save_json(assumptions, out_dir / "assumptions.json")
    LOG.info(f"assumptions.json saved")

    # Plots
    make_plots(results_df, fig_dir)
    make_sensitivity_plots(sens_df, fig_dir)

    LOG.info("Simulation complete.")


if __name__ == "__main__":
    main()
