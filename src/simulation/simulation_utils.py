"""
simulation_utils.py — shared dataclasses and helper functions for the simulator.
"""

from dataclasses import dataclass, field


@dataclass
class V2ILink:
    name: str
    bandwidth_mbps: float
    energy_per_mb: float          # J/MB
    latency_s: float              # fixed network latency


@dataclass
class V2VLink:
    name: str
    bandwidth_mbps: float
    energy_per_mb: float
    latency_s: float


@dataclass
class HardwareConfig:
    cloud_gpu_power_w: float = 200.0
    leader_gpu_power_w: float = 200.0
    leader_cpu_power_w: float = 80.0
    self_cpu_power_w: float = 80.0


@dataclass
class ScenarioConstants:
    avg_crop_size_bytes: float = 257903.25
    avg_overlap_fraction: float = 0.4
    avg_cloud_inference_time_s: float = 0.01287
    avg_leader_inference_time_s: float = 0.00305
    avg_cpu_self_inference_time_s: float = 0.1287
    avg_fusion_time_s: float = 0.001
    resize_factor: float = 0.6
    d_vehicle_to_cloud_km: float = 1000.0
    d_vehicle_to_leader_km: float = 0.18778
    speed_of_light_fiber_km_s: float = 200000.0
    staggered_tx_offset_fraction: float = 0.05


@dataclass
class SimResult:
    method: str
    v2i_name: str
    v2v_name: str
    n_vehicles: int
    # Delay breakdown (seconds)
    total_delay_s: float = 0.0
    tx_delay_s: float = 0.0
    queue_delay_s: float = 0.0
    compute_delay_s: float = 0.0
    # Bytes
    v2i_bytes: float = 0.0
    v2v_bytes: float = 0.0
    total_bytes: float = 0.0
    # Energy breakdown (Joules)
    v2v_energy_j: float = 0.0
    v2i_energy_j: float = 0.0
    leader_compute_energy_j: float = 0.0
    cloud_compute_energy_j: float = 0.0
    self_compute_energy_j: float = 0.0
    total_energy_j: float = 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "v2i_network": self.v2i_name,
            "v2v_network": self.v2v_name,
            "n_vehicles": self.n_vehicles,
            "total_delay_s": self.total_delay_s,
            "tx_delay_s": self.tx_delay_s,
            "queue_delay_s": self.queue_delay_s,
            "compute_delay_s": self.compute_delay_s,
            "v2i_bytes": self.v2i_bytes,
            "v2v_bytes": self.v2v_bytes,
            "total_bytes": self.total_bytes,
            "v2v_energy_j": self.v2v_energy_j,
            "v2i_energy_j": self.v2i_energy_j,
            "leader_compute_energy_j": self.leader_compute_energy_j,
            "cloud_compute_energy_j": self.cloud_compute_energy_j,
            "self_compute_energy_j": self.self_compute_energy_j,
            "total_energy_j": self.total_energy_j,
        }


def bytes_to_mb(b: float) -> float:
    return b / 1e6


def propagation_delay(distance_km: float, speed_km_s: float) -> float:
    return distance_km / speed_km_s
