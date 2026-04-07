"""
control_overhead.py — Phase G

Models the communication overhead of cloudlet formation handshakes and
dynamic mobility events (join, leave, leader_failure).

Addresses Reviewer 2's concern about signaling overhead not being quantified.

Usage:
    python src/simulation/control_overhead.py --config configs/network.yaml

Outputs:
    outputs/simulation/control_overhead_results.csv
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.common.config_utils import load_yaml, save_json
from src.common.io_utils import get_logger
from src.simulation.simulation_utils import bytes_to_mb

LOG = get_logger("control_overhead")


# ---------------------------------------------------------------------------
# Packet size model (bytes)
# ---------------------------------------------------------------------------

@dataclass
class ControlPackets:
    leader_advertisement: int = 64
    member_status: int = 128        # per member
    ack_selection: int = 32
    heartbeat_beacon: int = 32      # per member, optional
    re_election_trigger: int = 48   # broadcast on leader failure/leave
    join_request: int = 96
    join_ack: int = 32
    leave_notification: int = 32


PKT = ControlPackets()


def formation_overhead(
    n_vehicles: int,
    include_heartbeat: bool = True,
) -> dict:
    """
    One complete cloudlet formation handshake:
        1. Leader advertisement (broadcast)
        2. Member status messages (N-1 members -> leader)
        3. ACK / selection confirmation (leader -> N-1 members)
        4. [Optional] heartbeat beacons
    """
    bytes_sent = (
        PKT.leader_advertisement
        + PKT.member_status * (n_vehicles - 1)
        + PKT.ack_selection * (n_vehicles - 1)
    )
    if include_heartbeat:
        bytes_sent += PKT.heartbeat_beacon * (n_vehicles - 1)
    return {"event_type": "formation", "control_bytes": bytes_sent}


def join_overhead(n_vehicles: int) -> dict:
    """New vehicle joins existing cloudlet."""
    bytes_sent = (
        PKT.join_request
        + PKT.join_ack
        + PKT.leader_advertisement   # updated membership broadcast
    )
    return {"event_type": "join", "control_bytes": bytes_sent}


def leave_overhead(n_vehicles: int) -> dict:
    """Vehicle leaves gracefully (not leader)."""
    bytes_sent = PKT.leave_notification + PKT.ack_selection
    return {"event_type": "leave", "control_bytes": bytes_sent}


def leader_failure_overhead(n_vehicles: int) -> dict:
    """
    Leader fails unexpectedly.
    Triggers re-election: all remaining members broadcast status,
    new leader selected, acknowledgements sent.
    """
    remaining = n_vehicles - 1
    if remaining <= 0:
        return {"event_type": "leader_failure", "control_bytes": 0}
    bytes_sent = (
        PKT.re_election_trigger                # detected/timeout -> broadcast
        + PKT.member_status * remaining        # all candidates broadcast state
        + PKT.leader_advertisement             # new leader announces
        + PKT.ack_selection * (remaining - 1) # new leader notifies others
    )
    return {"event_type": "leader_failure", "control_bytes": bytes_sent}


def compute_control_metrics(
    event_dict: dict,
    n_vehicles: int,
    link_name: str,
    bandwidth_mbps: float,
    energy_per_mb: float,
    latency_s: float,
) -> dict:
    B = event_dict["control_bytes"]
    B_mb = bytes_to_mb(B)
    delay_s = B_mb / bandwidth_mbps + latency_s
    energy_j = B_mb * energy_per_mb
    return {
        "event_type": event_dict["event_type"],
        "n_vehicles": n_vehicles,
        "link": link_name,
        "control_bytes": B,
        "control_delay_s": round(delay_s, 6),
        "control_energy_j": round(energy_j, 6),
    }


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

    # Use V2V links for control messages (local broadcast)
    v2v_links = cfg["v2v_networks"]

    rows = []
    for n_veh in [3, 5]:
        for link_name, link_cfg in v2v_links.items():
            bw = link_cfg["bandwidth_mbps"]
            e_mb = link_cfg["energy_per_mb"]
            lat = link_cfg["latency_s"]

            events = [
                formation_overhead(n_veh, include_heartbeat=True),
                join_overhead(n_veh),
                leave_overhead(n_veh),
                leader_failure_overhead(n_veh),
            ]

            for ev in events:
                row = compute_control_metrics(ev, n_veh, link_name, bw, e_mb, lat)
                rows.append(row)

    df = pd.DataFrame(rows)
    out_path = out_dir / "control_overhead_results.csv"
    df.to_csv(out_path, index=False)
    LOG.info(f"control_overhead_results.csv: {len(df)} rows -> {out_path}")

    # Summary
    LOG.info("\n" + df.to_string(index=False))

    # Save packet size model
    pkt_model = {
        "leader_advertisement_bytes": PKT.leader_advertisement,
        "member_status_bytes_per_member": PKT.member_status,
        "ack_selection_bytes_per_member": PKT.ack_selection,
        "heartbeat_beacon_bytes_per_member": PKT.heartbeat_beacon,
        "re_election_trigger_bytes": PKT.re_election_trigger,
        "join_request_bytes": PKT.join_request,
        "join_ack_bytes": PKT.join_ack,
        "leave_notification_bytes": PKT.leave_notification,
        "note": (
            "Packet sizes are conservative estimates for DSRC/V2V control frames. "
            "Formation overhead is dominated by member status exchange and scales linearly with N."
        ),
    }
    save_json(pkt_model, out_dir / "control_packet_model.json")
    LOG.info("control_packet_model.json saved")
    LOG.info("Done.")


if __name__ == "__main__":
    main()
