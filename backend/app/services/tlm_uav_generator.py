"""
TLM-UAV Demo Data Generator for UAIE

Generates realistic UAV (Unmanned Aerial Vehicle) telemetry data modeled after
the TLM:UAV Anomaly Detection Dataset (Kaggle: luyucwnu/tlmuav-anomaly-detection-datasets).

The dataset simulates ArduPilot SITL flight logs across four sensor groups:
  - GPS: position, speed, satellite count
  - IMU: accelerometer and gyroscope readings
  - RATE: desired vs achieved attitude rates (roll, pitch, yaw)
  - VIBE: processed acceleration vibration levels

Four fault types are injected following the Time Line Modeling (TLM) method:
  1. GPS fault     – position drift / satellite loss
  2. Accelerometer fault – IMU bias and noise spike
  3. Engine fault  – thrust loss, altitude drop, vibration surge
  4. RC System fault – control input freeze / erratic commands

Reference:
  Yang et al., "Acquisition and Processing of UAV Fault Data Based on
  Time Line Modeling Method", Applied Sciences 13(7):4301, 2023.
"""

import math
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple


# ── Fault window definitions (record index ranges) ──────────────────────
# Each fault is anchored at a point and stretched into a window, per TLM.
FAULT_WINDOWS = {
    "gps_fault":          (150, 200),
    "accelerometer_fault": (350, 410),
    "engine_fault":        (550, 620),
    "rc_system_fault":     (750, 820),
}


def generate_tlm_uav_data(
    num_records: int = 1000,
    include_anomalies: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate synthetic UAV telemetry data with TLM-style fault injection.

    Returns:
        Tuple of (records, metadata)
    """
    random.seed(42)

    base_time = datetime.now() - timedelta(seconds=num_records * 0.1)
    records: List[Dict[str, Any]] = []

    # Flight plan: 4 waypoints forming a rectangle at ~50 m altitude
    waypoints = [
        (35.6812, 139.7671),  # Start / WP1
        (35.6820, 139.7671),  # WP2 – north
        (35.6820, 139.7681),  # WP3 – north-east
        (35.6812, 139.7681),  # WP4 – east
    ]

    for i in range(num_records):
        t = i / num_records
        timestamp = base_time + timedelta(seconds=i * 0.1)

        # ── Flight trajectory (smooth waypoint interpolation) ────────
        segment = int(t * 4) % 4
        seg_t = (t * 4) % 1.0
        wp_a = waypoints[segment]
        wp_b = waypoints[(segment + 1) % 4]
        lat = wp_a[0] + (wp_b[0] - wp_a[0]) * seg_t
        lon = wp_a[1] + (wp_b[1] - wp_a[1]) * seg_t

        # Altitude: climb to 50m, hold, descend at end
        if t < 0.05:
            alt = 50.0 * (t / 0.05)
        elif t > 0.95:
            alt = 50.0 * ((1.0 - t) / 0.05)
        else:
            alt = 50.0

        # Ground speed varies with phase
        base_speed = 5.0 + 2.0 * math.sin(t * math.pi * 8)

        # ── GPS fields ───────────────────────────────────────────────
        gps_lat = round(lat + random.gauss(0, 0.000002), 7)
        gps_lon = round(lon + random.gauss(0, 0.000002), 7)
        gps_alt = round(alt + random.gauss(0, 0.15), 2)
        gps_speed = round(max(0, base_speed + random.gauss(0, 0.2)), 2)
        gps_num_sats = random.randint(10, 14)
        gps_hdop = round(0.8 + random.gauss(0, 0.05), 2)

        # ── IMU fields (accelerometer + gyroscope) ───────────────────
        # Gravity-compensated body-frame accelerations (m/s^2)
        imu_acc_x = round(random.gauss(0.0, 0.08), 4)
        imu_acc_y = round(random.gauss(0.0, 0.08), 4)
        imu_acc_z = round(-9.81 + random.gauss(0, 0.1), 4)
        # Gyroscope (rad/s)
        imu_gyro_x = round(random.gauss(0.0, 0.005), 5)
        imu_gyro_y = round(random.gauss(0.0, 0.005), 5)
        imu_gyro_z = round(random.gauss(0.0, 0.003), 5)

        # ── RATE fields (desired vs achieved attitude rates, deg/s) ──
        # Smooth desired rates from flight plan
        des_roll_rate = round(2.0 * math.sin(t * math.pi * 12) + random.gauss(0, 0.3), 3)
        des_pitch_rate = round(1.5 * math.cos(t * math.pi * 10) + random.gauss(0, 0.3), 3)
        des_yaw_rate = round(0.5 * math.sin(t * math.pi * 4) + random.gauss(0, 0.2), 3)
        # Achieved tracks desired with small lag/error
        ach_roll_rate = round(des_roll_rate + random.gauss(0, 0.15), 3)
        ach_pitch_rate = round(des_pitch_rate + random.gauss(0, 0.15), 3)
        ach_yaw_rate = round(des_yaw_rate + random.gauss(0, 0.1), 3)

        # ── VIBE fields (vibration levels, m/s^2) ────────────────────
        vibe_x = round(abs(random.gauss(0.5, 0.15)), 3)
        vibe_y = round(abs(random.gauss(0.5, 0.15)), 3)
        vibe_z = round(abs(random.gauss(0.8, 0.2)), 3)
        vibe_clipping_0 = 0
        vibe_clipping_1 = 0
        vibe_clipping_2 = 0

        # ── Throttle / engine ────────────────────────────────────────
        throttle_pct = round(min(100, max(0, 55 + 10 * math.sin(t * math.pi * 6) + random.gauss(0, 2))), 1)

        # Label: 0 = normal
        fault_label = 0
        fault_type = "normal"

        record = {
            "timestamp": timestamp.isoformat(),
            "flight_mode": "AUTO",
            # GPS
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "gps_alt_m": gps_alt,
            "gps_speed_m_s": gps_speed,
            "gps_num_sats": gps_num_sats,
            "gps_hdop": gps_hdop,
            # IMU
            "imu_acc_x_m_s2": imu_acc_x,
            "imu_acc_y_m_s2": imu_acc_y,
            "imu_acc_z_m_s2": imu_acc_z,
            "imu_gyro_x_rad_s": imu_gyro_x,
            "imu_gyro_y_rad_s": imu_gyro_y,
            "imu_gyro_z_rad_s": imu_gyro_z,
            # RATE
            "rate_des_roll_deg_s": des_roll_rate,
            "rate_des_pitch_deg_s": des_pitch_rate,
            "rate_des_yaw_deg_s": des_yaw_rate,
            "rate_ach_roll_deg_s": ach_roll_rate,
            "rate_ach_pitch_deg_s": ach_pitch_rate,
            "rate_ach_yaw_deg_s": ach_yaw_rate,
            # VIBE
            "vibe_x_m_s2": vibe_x,
            "vibe_y_m_s2": vibe_y,
            "vibe_z_m_s2": vibe_z,
            "vibe_clipping_0": vibe_clipping_0,
            "vibe_clipping_1": vibe_clipping_1,
            "vibe_clipping_2": vibe_clipping_2,
            # Engine
            "throttle_pct": throttle_pct,
            # Labels
            "fault_label": fault_label,
            "fault_type": fault_type,
        }

        if include_anomalies:
            record = _inject_faults(record, i)

        records.append(record)

    metadata = {
        "system_name": "UAV Copter - TLM Flight Test",
        "system_type": "uav",
        "description": (
            "Simulated multi-rotor UAV telemetry data based on ArduPilot SITL, "
            "featuring GPS, IMU, RATE, and VIBE sensor groups with four injected "
            "fault types following the Time Line Modeling (TLM) method."
        ),
        "confidence": 0.95,
        "record_count": num_records,
        "field_count": len(records[0]) if records else 0,
        "demo_anomalies_injected": include_anomalies,
        "anomaly_types": [
            "gps_fault",
            "accelerometer_fault",
            "engine_fault",
            "rc_system_fault",
        ] if include_anomalies else [],
        "dataset_reference": (
            "TLM:UAV Anomaly Detection Datasets — "
            "https://www.kaggle.com/datasets/luyucwnu/tlmuav-anomaly-detection-datasets"
        ),
    }

    return records, metadata


def _inject_faults(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Inject TLM-style faults into a single record."""

    # ─────────────────────────────────────────────────────────────────────
    # FAULT 1: GPS fault — satellite loss + position drift
    # Triggers: Statistical Analyst, Cross-Sensor Sync, Domain Expert
    # ─────────────────────────────────────────────────────────────────────
    start, end = FAULT_WINDOWS["gps_fault"]
    if start <= index < end:
        progress = (index - start) / (end - start)
        record["gps_num_sats"] = max(2, int(14 - 12 * progress))
        record["gps_hdop"] = round(0.8 + 8.0 * progress, 2)
        # Position drifts increasingly
        record["gps_lat"] += random.gauss(0, 0.0001 * progress)
        record["gps_lon"] += random.gauss(0, 0.0001 * progress)
        record["gps_alt_m"] += random.gauss(0, 3.0 * progress)
        record["gps_speed_m_s"] = round(abs(record["gps_speed_m_s"] + random.gauss(0, 2.0 * progress)), 2)
        record["fault_label"] = 1
        record["fault_type"] = "gps_fault"

    # ─────────────────────────────────────────────────────────────────────
    # FAULT 2: Accelerometer fault — bias offset + noise amplification
    # Triggers: Stagnation Sentinel, Micro-Drift Tracker, Safety Auditor
    # ─────────────────────────────────────────────────────────────────────
    start, end = FAULT_WINDOWS["accelerometer_fault"]
    if start <= index < end:
        progress = (index - start) / (end - start)
        # Growing bias on X and Y axes
        bias = 0.5 * progress
        record["imu_acc_x_m_s2"] = round(record["imu_acc_x_m_s2"] + bias + random.gauss(0, 0.3 * progress), 4)
        record["imu_acc_y_m_s2"] = round(record["imu_acc_y_m_s2"] - bias + random.gauss(0, 0.3 * progress), 4)
        # Z-axis noise increases
        record["imu_acc_z_m_s2"] = round(record["imu_acc_z_m_s2"] + random.gauss(0, 0.8 * progress), 4)
        # Gyro becomes noisy too (cross-axis coupling)
        record["imu_gyro_x_rad_s"] = round(record["imu_gyro_x_rad_s"] + random.gauss(0, 0.03 * progress), 5)
        record["imu_gyro_y_rad_s"] = round(record["imu_gyro_y_rad_s"] + random.gauss(0, 0.03 * progress), 5)
        record["fault_label"] = 2
        record["fault_type"] = "accelerometer_fault"

    # ─────────────────────────────────────────────────────────────────────
    # FAULT 3: Engine fault — thrust degradation, altitude drop, vibration
    # Triggers: Harmonic Distortion, Efficiency Analyst, Cross-Sensor Sync
    # ─────────────────────────────────────────────────────────────────────
    start, end = FAULT_WINDOWS["engine_fault"]
    if start <= index < end:
        progress = (index - start) / (end - start)
        # Throttle commands rise but altitude drops (engine not responding)
        record["throttle_pct"] = round(min(100, record["throttle_pct"] + 30 * progress), 1)
        record["gps_alt_m"] = round(record["gps_alt_m"] - 15 * progress + random.gauss(0, 0.5), 2)
        # Vibration spikes from mechanical failure
        record["vibe_x_m_s2"] = round(record["vibe_x_m_s2"] + 3.0 * progress + random.gauss(0, 0.5), 3)
        record["vibe_y_m_s2"] = round(record["vibe_y_m_s2"] + 3.0 * progress + random.gauss(0, 0.5), 3)
        record["vibe_z_m_s2"] = round(record["vibe_z_m_s2"] + 5.0 * progress + random.gauss(0, 0.8), 3)
        record["vibe_clipping_0"] = random.randint(0, int(20 * progress))
        record["vibe_clipping_1"] = random.randint(0, int(15 * progress))
        record["vibe_clipping_2"] = random.randint(0, int(25 * progress))
        # Rate tracking degrades
        record["rate_ach_roll_deg_s"] += random.gauss(0, 2.0 * progress)
        record["rate_ach_pitch_deg_s"] += random.gauss(0, 2.0 * progress)
        record["fault_label"] = 3
        record["fault_type"] = "engine_fault"

    # ─────────────────────────────────────────────────────────────────────
    # FAULT 4: RC System fault — frozen / erratic control inputs
    # Triggers: Stagnation Sentinel, Logic State Conflict, Human-Context
    # ─────────────────────────────────────────────────────────────────────
    start, end = FAULT_WINDOWS["rc_system_fault"]
    if start <= index < end:
        progress = (index - start) / (end - start)
        if progress < 0.5:
            # Phase A: Control inputs freeze (stagnation)
            record["rate_des_roll_deg_s"] = 0.0
            record["rate_des_pitch_deg_s"] = 0.0
            record["rate_des_yaw_deg_s"] = 0.0
        else:
            # Phase B: Erratic commands
            record["rate_des_roll_deg_s"] = round(random.uniform(-15, 15), 3)
            record["rate_des_pitch_deg_s"] = round(random.uniform(-15, 15), 3)
            record["rate_des_yaw_deg_s"] = round(random.uniform(-10, 10), 3)
        # Achieved rates diverge from desired
        record["rate_ach_roll_deg_s"] = round(record["rate_des_roll_deg_s"] * 0.3 + random.gauss(0, 1.5), 3)
        record["rate_ach_pitch_deg_s"] = round(record["rate_des_pitch_deg_s"] * 0.3 + random.gauss(0, 1.5), 3)
        record["rate_ach_yaw_deg_s"] = round(record["rate_des_yaw_deg_s"] * 0.3 + random.gauss(0, 1.0), 3)
        record["flight_mode"] = "STABILIZE"  # Failsafe mode switch
        record["fault_label"] = 4
        record["fault_type"] = "rc_system_fault"

    return record


def generate_tlm_uav_description_file() -> str:
    """Generate a documentation markdown string for the TLM-UAV demo system."""
    return """# TLM-UAV Telemetry System Documentation

## System Overview
Multi-rotor UAV (quadcopter) telemetry data collected from an ArduPilot
Software-In-The-Loop (SITL) simulation environment. The flight follows a
rectangular waypoint mission at approximately 50 m altitude.

## Sensor Groups

### GPS Information
- **gps_lat / gps_lon**: WGS-84 latitude and longitude (degrees)
- **gps_alt_m**: GPS-reported altitude above home (m). Normal: 45-55 m during cruise.
- **gps_speed_m_s**: Ground speed (m/s). Normal cruise: 3-8 m/s.
- **gps_num_sats**: Number of visible satellites. Healthy: >= 8. Critical: < 5.
- **gps_hdop**: Horizontal dilution of precision. Good: < 1.5. Poor: > 4.0.

### IMU (Inertial Measurement Unit)
- **imu_acc_{x,y,z}_m_s2**: Body-frame accelerometer (m/s^2). Z should be ~-9.81 in hover.
- **imu_gyro_{x,y,z}_rad_s**: Body-frame gyroscope (rad/s). Should be near zero in stable hover.

### RATE (Attitude Rate Controller)
- **rate_des_{roll,pitch,yaw}_deg_s**: Desired attitude rates from the flight controller (deg/s).
- **rate_ach_{roll,pitch,yaw}_deg_s**: Achieved attitude rates measured by sensors (deg/s).
- Divergence between desired and achieved indicates control system issues.

### VIBE (Vibration Monitoring)
- **vibe_{x,y,z}_m_s2**: RMS vibration acceleration (m/s^2). Healthy: < 3.0. Critical: > 6.0.
- **vibe_clipping_{0,1,2}**: Accelerometer clipping counts per axis. Should be 0 in normal flight.

### Engine / Throttle
- **throttle_pct**: Throttle output (0-100%). Hover is typically 45-65%.

### Flight Status
- **flight_mode**: AUTO (normal mission), STABILIZE (manual/failsafe).
- **fault_label**: 0=normal, 1=GPS, 2=accelerometer, 3=engine, 4=RC system.
- **fault_type**: Human-readable fault category.

## Fault Types (TLM Method)

| Fault | Label | Symptoms |
|-------|-------|----------|
| GPS fault | 1 | Satellite loss, HDOP increase, position drift |
| Accelerometer fault | 2 | IMU bias growth, noise amplification, gyro coupling |
| Engine fault | 3 | Altitude drop despite high throttle, vibration surge, clipping |
| RC System fault | 4 | Control freeze then erratic commands, mode switch to STABILIZE |

## Safety Limits
| Parameter | Warning | Critical |
|-----------|---------|----------|
| GPS Satellites | 6 | 4 |
| GPS HDOP | 3.0 | 5.0 |
| Vibration (any axis) | 3.0 m/s^2 | 6.0 m/s^2 |
| Altitude deviation | +/- 5 m | +/- 15 m |
| Throttle | > 85% sustained | 100% for > 5 s |

## Reference
Yang et al., "Acquisition and Processing of UAV Fault Data Based on Time
Line Modeling Method", Applied Sciences 13(7):4301, 2023.
Dataset: https://www.kaggle.com/datasets/luyucwnu/tlmuav-anomaly-detection-datasets
"""


def generate_full_tlm_uav_package() -> Dict[str, Any]:
    """
    Generate a complete TLM-UAV demo package with data, schema, and metadata.

    Returns a dict matching the structure expected by the demo/create endpoint.
    """
    records, metadata = generate_tlm_uav_data(num_records=1000, include_anomalies=True)
    description = generate_tlm_uav_description_file()

    discovered_fields = [
        {
            "name": "timestamp",
            "display_name": "Timestamp",
            "type": "datetime",
            "category": "temporal",
            "description": "Telemetry sample timestamp (10 Hz)",
            "confidence": 0.99,
        },
        {
            "name": "flight_mode",
            "display_name": "Flight Mode",
            "type": "string",
            "category": "content",
            "description": "ArduPilot flight mode (AUTO, STABILIZE, etc.)",
            "confidence": 0.98,
        },
        # ── GPS ──────────────────────────────────────────────────────
        {
            "name": "gps_lat",
            "display_name": "GPS Latitude",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "GPS Receiver",
            "description": "WGS-84 latitude",
            "confidence": 0.97,
        },
        {
            "name": "gps_lon",
            "display_name": "GPS Longitude",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "GPS Receiver",
            "description": "WGS-84 longitude",
            "confidence": 0.97,
        },
        {
            "name": "gps_alt_m",
            "display_name": "GPS Altitude",
            "type": "float",
            "physical_unit": "m",
            "category": "content",
            "component": "GPS Receiver",
            "description": "Altitude above home position",
            "engineering_context": {
                "typical_range": {"min": 0, "max": 55},
                "what_high_means": "UAV climbing or altitude sensor error",
                "what_low_means": "Descending or engine/thrust loss",
                "safety_critical": True,
            },
            "confidence": 0.96,
        },
        {
            "name": "gps_speed_m_s",
            "display_name": "Ground Speed",
            "type": "float",
            "physical_unit": "m/s",
            "category": "content",
            "component": "GPS Receiver",
            "description": "GPS-derived ground speed",
            "engineering_context": {
                "typical_range": {"min": 0, "max": 10},
            },
            "confidence": 0.95,
        },
        {
            "name": "gps_num_sats",
            "display_name": "GPS Satellites",
            "type": "integer",
            "category": "content",
            "component": "GPS Receiver",
            "description": "Number of visible GPS satellites",
            "engineering_context": {
                "typical_range": {"min": 8, "max": 14},
                "what_low_means": "Poor GPS fix — position unreliable",
                "design_limit_hint": {"min": 4, "max": 20},
                "safety_critical": True,
            },
            "confidence": 0.97,
        },
        {
            "name": "gps_hdop",
            "display_name": "GPS HDOP",
            "type": "float",
            "category": "content",
            "component": "GPS Receiver",
            "description": "Horizontal dilution of precision (lower = better)",
            "engineering_context": {
                "typical_range": {"min": 0.5, "max": 1.5},
                "what_high_means": "Poor satellite geometry — position accuracy degraded",
                "safety_critical": True,
            },
            "confidence": 0.93,
        },
        # ── IMU ──────────────────────────────────────────────────────
        {
            "name": "imu_acc_x_m_s2",
            "display_name": "IMU Accel X",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "IMU / Accelerometer",
            "description": "Body-frame X-axis acceleration",
            "engineering_context": {
                "typical_range": {"min": -1.0, "max": 1.0},
            },
            "confidence": 0.94,
        },
        {
            "name": "imu_acc_y_m_s2",
            "display_name": "IMU Accel Y",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "IMU / Accelerometer",
            "description": "Body-frame Y-axis acceleration",
            "engineering_context": {
                "typical_range": {"min": -1.0, "max": 1.0},
            },
            "confidence": 0.94,
        },
        {
            "name": "imu_acc_z_m_s2",
            "display_name": "IMU Accel Z",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "IMU / Accelerometer",
            "description": "Body-frame Z-axis acceleration (includes gravity)",
            "engineering_context": {
                "typical_range": {"min": -10.5, "max": -9.0},
                "what_high_means": "Upward acceleration or sensor bias",
                "what_low_means": "Downward acceleration beyond gravity",
            },
            "confidence": 0.94,
        },
        {
            "name": "imu_gyro_x_rad_s",
            "display_name": "IMU Gyro X",
            "type": "float",
            "physical_unit": "rad/s",
            "category": "content",
            "component": "IMU / Gyroscope",
            "description": "Body-frame roll rate",
            "confidence": 0.93,
        },
        {
            "name": "imu_gyro_y_rad_s",
            "display_name": "IMU Gyro Y",
            "type": "float",
            "physical_unit": "rad/s",
            "category": "content",
            "component": "IMU / Gyroscope",
            "description": "Body-frame pitch rate",
            "confidence": 0.93,
        },
        {
            "name": "imu_gyro_z_rad_s",
            "display_name": "IMU Gyro Z",
            "type": "float",
            "physical_unit": "rad/s",
            "category": "content",
            "component": "IMU / Gyroscope",
            "description": "Body-frame yaw rate",
            "confidence": 0.93,
        },
        # ── RATE ─────────────────────────────────────────────────────
        {
            "name": "rate_des_roll_deg_s",
            "display_name": "Desired Roll Rate",
            "type": "float",
            "physical_unit": "deg/s",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Commanded roll rate from flight controller",
            "confidence": 0.92,
        },
        {
            "name": "rate_des_pitch_deg_s",
            "display_name": "Desired Pitch Rate",
            "type": "float",
            "physical_unit": "deg/s",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Commanded pitch rate from flight controller",
            "confidence": 0.92,
        },
        {
            "name": "rate_des_yaw_deg_s",
            "display_name": "Desired Yaw Rate",
            "type": "float",
            "physical_unit": "deg/s",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Commanded yaw rate from flight controller",
            "confidence": 0.92,
        },
        {
            "name": "rate_ach_roll_deg_s",
            "display_name": "Achieved Roll Rate",
            "type": "float",
            "physical_unit": "deg/s",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Measured actual roll rate",
            "confidence": 0.92,
        },
        {
            "name": "rate_ach_pitch_deg_s",
            "display_name": "Achieved Pitch Rate",
            "type": "float",
            "physical_unit": "deg/s",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Measured actual pitch rate",
            "confidence": 0.92,
        },
        {
            "name": "rate_ach_yaw_deg_s",
            "display_name": "Achieved Yaw Rate",
            "type": "float",
            "physical_unit": "deg/s",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Measured actual yaw rate",
            "confidence": 0.92,
        },
        # ── VIBE ─────────────────────────────────────────────────────
        {
            "name": "vibe_x_m_s2",
            "display_name": "Vibration X",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "Vibration Monitor",
            "description": "X-axis vibration RMS acceleration",
            "engineering_context": {
                "typical_range": {"min": 0, "max": 3.0},
                "what_high_means": "Mechanical imbalance, propeller damage, or motor failure",
                "safety_critical": True,
            },
            "confidence": 0.91,
        },
        {
            "name": "vibe_y_m_s2",
            "display_name": "Vibration Y",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "Vibration Monitor",
            "description": "Y-axis vibration RMS acceleration",
            "engineering_context": {
                "typical_range": {"min": 0, "max": 3.0},
                "what_high_means": "Mechanical imbalance, propeller damage, or motor failure",
                "safety_critical": True,
            },
            "confidence": 0.91,
        },
        {
            "name": "vibe_z_m_s2",
            "display_name": "Vibration Z",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "Vibration Monitor",
            "description": "Z-axis vibration RMS acceleration",
            "engineering_context": {
                "typical_range": {"min": 0, "max": 3.0},
                "what_high_means": "Mechanical imbalance or motor failure",
                "safety_critical": True,
            },
            "confidence": 0.91,
        },
        {
            "name": "vibe_clipping_0",
            "display_name": "Accel Clipping X",
            "type": "integer",
            "category": "content",
            "component": "Vibration Monitor",
            "description": "Accelerometer X-axis clipping event count (should be 0)",
            "confidence": 0.90,
        },
        {
            "name": "vibe_clipping_1",
            "display_name": "Accel Clipping Y",
            "type": "integer",
            "category": "content",
            "component": "Vibration Monitor",
            "description": "Accelerometer Y-axis clipping event count (should be 0)",
            "confidence": 0.90,
        },
        {
            "name": "vibe_clipping_2",
            "display_name": "Accel Clipping Z",
            "type": "integer",
            "category": "content",
            "component": "Vibration Monitor",
            "description": "Accelerometer Z-axis clipping event count (should be 0)",
            "confidence": 0.90,
        },
        # ── Engine ───────────────────────────────────────────────────
        {
            "name": "throttle_pct",
            "display_name": "Throttle Output",
            "type": "float",
            "physical_unit": "%",
            "category": "content",
            "component": "Motor Controller",
            "description": "Throttle percentage output to motors",
            "engineering_context": {
                "typical_range": {"min": 40, "max": 70},
                "what_high_means": "UAV is working hard — possible payload, wind, or thrust loss",
                "design_limit_hint": {"min": 0, "max": 100},
            },
            "confidence": 0.95,
        },
        # ── Labels ───────────────────────────────────────────────────
        {
            "name": "fault_label",
            "display_name": "Fault Label",
            "type": "integer",
            "category": "content",
            "description": "Numeric fault class (0=normal, 1=GPS, 2=accel, 3=engine, 4=RC)",
            "confidence": 0.99,
        },
        {
            "name": "fault_type",
            "display_name": "Fault Type",
            "type": "string",
            "category": "content",
            "description": "Human-readable fault category name",
            "confidence": 0.99,
        },
    ]

    relationships = [
        {
            "fields": ["gps_num_sats", "gps_hdop"],
            "relationship": "inverse_correlation",
            "description": "More satellites generally means lower HDOP (better accuracy)",
            "expected_correlation": "negative",
            "diagnostic_value": "Both degrading simultaneously confirms GPS environment issue",
        },
        {
            "fields": ["throttle_pct", "gps_alt_m"],
            "relationship": "control_target",
            "description": "Throttle controls altitude; high throttle with dropping altitude indicates engine fault",
            "diagnostic_value": "Divergence is the primary engine-fault signature",
        },
        {
            "fields": ["vibe_x_m_s2", "vibe_y_m_s2", "vibe_z_m_s2"],
            "relationship": "proportional",
            "description": "Vibration axes tend to increase together during mechanical failures",
            "expected_correlation": "positive",
        },
        {
            "fields": ["rate_des_roll_deg_s", "rate_ach_roll_deg_s"],
            "relationship": "control_target",
            "description": "Achieved roll rate should track desired; divergence means control degradation",
            "diagnostic_value": "Gap between desired and achieved rates diagnoses actuator or RC faults",
        },
        {
            "fields": ["rate_des_pitch_deg_s", "rate_ach_pitch_deg_s"],
            "relationship": "control_target",
            "description": "Achieved pitch rate should track desired",
            "diagnostic_value": "Used alongside roll tracking to isolate axis-specific failures",
        },
        {
            "fields": ["imu_acc_x_m_s2", "imu_acc_y_m_s2"],
            "relationship": "correlation",
            "description": "Body-frame lateral accelerations should be near zero in stable hover",
            "diagnostic_value": "Simultaneous bias growth indicates accelerometer calibration failure",
        },
    ]

    blind_spots = [
        "No battery voltage/current sensor — cannot detect power supply degradation",
        "No barometric altimeter — cannot cross-validate GPS altitude",
        "No magnetometer data — cannot assess compass interference or yaw accuracy",
        "No motor RPM telemetry — cannot isolate which motor is failing during engine faults",
        "No wind speed sensor — cannot distinguish wind disturbances from control faults",
    ]

    return {
        "records": records,
        "metadata": metadata,
        "description_content": description,
        "discovered_fields": discovered_fields,
        "relationships": relationships,
        "blind_spots": blind_spots,
        "analysis_summary": {
            "files_analyzed": 2,
            "total_records": len(records),
            "unique_fields": len(discovered_fields),
            "ai_powered": True,
        },
        "recommendation": {
            "suggested_name": "UAV Copter - TLM Flight Test",
            "suggested_type": "uav",
            "suggested_description": (
                "Multi-rotor UAV telemetry with GPS, IMU, attitude rate, and vibration "
                "monitoring. Data contains four distinct fault scenarios (GPS, accelerometer, "
                "engine, RC system) injected using the Time Line Modeling method."
            ),
            "confidence": 0.95,
            "system_subtype": "Multi-Rotor Quadcopter",
            "domain": "Aerospace / UAV",
            "detected_components": [
                {"name": "GPS Receiver", "role": "Position and velocity sensing", "fields": ["gps_lat", "gps_lon", "gps_alt_m", "gps_speed_m_s", "gps_num_sats", "gps_hdop"]},
                {"name": "IMU", "role": "Inertial measurement (accel + gyro)", "fields": ["imu_acc_x_m_s2", "imu_acc_y_m_s2", "imu_acc_z_m_s2", "imu_gyro_x_rad_s", "imu_gyro_y_rad_s", "imu_gyro_z_rad_s"]},
                {"name": "Attitude Controller", "role": "Rate control loop (desired vs achieved)", "fields": ["rate_des_roll_deg_s", "rate_des_pitch_deg_s", "rate_des_yaw_deg_s", "rate_ach_roll_deg_s", "rate_ach_pitch_deg_s", "rate_ach_yaw_deg_s"]},
                {"name": "Vibration Monitor", "role": "Mechanical health monitoring", "fields": ["vibe_x_m_s2", "vibe_y_m_s2", "vibe_z_m_s2", "vibe_clipping_0", "vibe_clipping_1", "vibe_clipping_2"]},
                {"name": "Motor Controller", "role": "Throttle and propulsion", "fields": ["throttle_pct"]},
            ],
            "probable_use_case": "UAV flight testing and anomaly detection research",
            "data_characteristics": {
                "temporal_resolution": "100 ms (10 Hz)",
                "duration_estimate": "~100 seconds",
                "completeness": "100%",
            },
        },
    }
