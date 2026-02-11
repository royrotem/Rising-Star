"""
TLM-UAV Demo Data Loader for UAIE

Loads REAL UAV telemetry data from the TLM:UAV Anomaly Detection Dataset
(Kaggle: luyucwnu/tlmuav-anomaly-detection-datasets).

The dataset contains ArduPilot SITL flight logs with four fault types
injected using the Time Line Modeling (TLM) method:
  1. GPS fault (label=1)
  2. Accelerometer fault (label=2)
  3. Engine fault (label=3)
  4. RC System fault (label=4)
  0. Normal (label=0)

The Fusion_Data.csv merges ATT, MAG, and IMU sensor groups into a single
18-feature dataset with 12,253 labeled records.

Reference:
  Yang et al., "Acquisition and Processing of UAV Fault Data Based on
  Time Line Modeling Method", Applied Sciences 13(7):4301, 2023.
"""

import csv
import io
import logging
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Path to the bundled Kaggle dataset
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ARCHIVE_PATH = _DATA_DIR / "tlmuav_kaggle.zip"

# Label mapping from integer to fault name
LABEL_MAP = {
    0: "normal",
    1: "gps_fault",
    2: "accelerometer_fault",
    3: "engine_fault",
    4: "rc_system_fault",
}

# In-memory cache so we only parse the zip once per process
_cache: Dict[str, Any] = {}


def _load_csv_from_zip(zip_path: Path, csv_inner_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file from inside a zip archive and return rows as dicts."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(csv_inner_path) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            return [row for row in reader]


def _load_kaggle_data() -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Load real data from the Kaggle archive.

    Returns:
        Tuple of (records_without_labels, ground_truth_labels)
        - records: list of dicts with sensor fields only (NO labels)
        - ground_truth: parallel list of integer labels (0-4)
    """
    if "records" in _cache and "ground_truth" in _cache:
        return _cache["records"], _cache["ground_truth"]

    if not _ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at {_ARCHIVE_PATH}. "
            "Please place tlmuav_kaggle.zip in backend/app/data/"
        )

    logger.info("Loading TLM-UAV Kaggle dataset from %s", _ARCHIVE_PATH)

    # ── Load Fusion_Data.csv (merged multi-sensor dataset) ──────────
    fusion_rows = _load_csv_from_zip(_ARCHIVE_PATH, "dataset/Fusion_Data.csv")

    # ── Load GPS data for position/speed/satellites ─────────────────
    gps_rows = _load_csv_from_zip(_ARCHIVE_PATH, "dataset/GPS/ALL_FAIL_LOG_GPS_0.csv")

    # ── Load RATE data for attitude rate controller ─────────────────
    rate_rows = _load_csv_from_zip(_ARCHIVE_PATH, "dataset/RATE/ALL_FAIL_LOG_RATE.csv")

    # ── Load VIBE data for vibration monitoring ─────────────────────
    vibe_rows = _load_csv_from_zip(_ARCHIVE_PATH, "dataset/VIBE/ALL_FAIL_LOG_VIBE_0_Random.csv")

    # ── Load BARO data ──────────────────────────────────────────────
    baro_rows = _load_csv_from_zip(_ARCHIVE_PATH, "dataset/BARO/ALL_FAIL_LOG_BARO.csv")

    # ── Load BAT data ───────────────────────────────────────────────
    bat_rows = _load_csv_from_zip(_ARCHIVE_PATH, "dataset/BAT/ALL_FAIL_LOG_BAT_0.csv")

    # ── Load MAG data ───────────────────────────────────────────────
    mag_rows = _load_csv_from_zip(_ARCHIVE_PATH, "dataset/MAG/ALL_FAIL_LOG_MAG_0.csv")

    # Build records from Fusion_Data (primary source — richest merged data)
    records: List[Dict[str, Any]] = []
    ground_truth: List[int] = []

    base_time = datetime.now() - timedelta(seconds=len(fusion_rows) * 0.1)

    for idx, row in enumerate(fusion_rows):
        ts = base_time + timedelta(seconds=idx * 0.1)

        label = int(float(row.get("labels", "0")))
        ground_truth.append(label)

        record = {
            "timestamp": ts.isoformat(),
            "row_index": idx,
            # ATT — Attitude (from Fusion)
            "att_DesRoll": float(row.get("DesRoll", 0)),
            "att_Roll": float(row.get("Roll", 0)),
            "att_DesPitch": float(row.get("DesPitch", 0)),
            "att_Pitch": float(row.get("Pitch", 0)),
            "att_DesYaw": float(row.get("DesYaw", 0)),
            "att_Yaw": float(row.get("Yaw", 0)),
            "att_ErrRP": float(row.get("ErrRP", 0)),
            "att_ErrYaw": float(row.get("ErrYaw", 0)),
            # MAG — Magnetometer (from Fusion)
            "mag_X": float(row.get("MagX", 0)),
            "mag_Y": float(row.get("MagY", 0)),
            "mag_Z": float(row.get("MagZ", 0)),
            # IMU — Inertial Measurement Unit (from Fusion)
            "imu_GyrX": float(row.get("abGyrX", 0)),
            "imu_GyrY": float(row.get("abGyrY", 0)),
            "imu_GyrZ": float(row.get("abGyrZ", 0)),
            "imu_AccX": float(row.get("abAccX", 0)),
            "imu_AccY": float(row.get("abAccY", 0)),
            "imu_AccZ": float(row.get("abAccZ", 0)),
        }

        # NOTE: We intentionally do NOT include "labels" in the record.
        # Ground truth is stored separately for post-analysis comparison.

        records.append(record)

    logger.info(
        "Loaded %d records from Fusion_Data.csv (labels: %s)",
        len(records),
        {LABEL_MAP.get(k, k): v for k, v in sorted(
            {l: ground_truth.count(l) for l in set(ground_truth)}.items()
        )},
    )

    # ── Also prepare per-sensor-group data for multi-file view ──────
    # GPS records (smaller: 2450 rows)
    gps_records: List[Dict[str, Any]] = []
    gps_ground_truth: List[int] = []
    for idx, row in enumerate(gps_rows):
        label = int(float(row.get("labels", "0")))
        gps_ground_truth.append(label)
        ts = base_time + timedelta(seconds=idx * 0.5)
        gps_records.append({
            "timestamp": ts.isoformat(),
            "row_index": idx,
            "gps_Status": float(row.get("Status", 0)),
            "gps_NSats": float(row.get("NSats", 0)),
            "gps_HDop": float(row.get("HDop", 0)),
            "gps_Lat": float(row.get("Lat", 0)),
            "gps_Lng": float(row.get("Lng", 0)),
            "gps_Alt": float(row.get("Alt", 0)),
            "gps_Spd": float(row.get("Spd", 0)),
            "gps_GCrs": float(row.get("GCrs", 0)),
            "gps_VZ": float(row.get("VZ", 0)),
            "gps_Yaw": float(row.get("Yaw", 0)),
            "gps_U": float(row.get("U", 0)),
        })

    # RATE records (4900 rows)
    rate_records: List[Dict[str, Any]] = []
    rate_ground_truth: List[int] = []
    for idx, row in enumerate(rate_rows):
        label = int(float(row.get("labels", "0")))
        rate_ground_truth.append(label)
        ts = base_time + timedelta(seconds=idx * 0.25)
        rate_records.append({
            "timestamp": ts.isoformat(),
            "row_index": idx,
            "rate_RDes": float(row.get("RDes", 0)),
            "rate_R": float(row.get("R", 0)),
            "rate_Rout": float(row.get("Rout", 0)),
            "rate_PDes": float(row.get("PDes", 0)),
            "rate_P": float(row.get("P", 0)),
            "rate_POut": float(row.get("POut", 0)),
            "rate_YDes": float(row.get("YDes", 0)),
            "rate_Y": float(row.get("Y", 0)),
            "rate_YOut": float(row.get("YOut", 0)),
            "rate_ADes": float(row.get("ADes", 0)),
            "rate_A": float(row.get("A", 0)),
            "rate_AOut": float(row.get("AOut", 0)),
        })

    # Cache everything
    _cache["records"] = records
    _cache["ground_truth"] = ground_truth
    _cache["gps_records"] = gps_records
    _cache["gps_ground_truth"] = gps_ground_truth
    _cache["rate_records"] = rate_records
    _cache["rate_ground_truth"] = rate_ground_truth
    _cache["sensor_file_stats"] = {
        "fusion": {"rows": len(fusion_rows), "cols": len(fusion_rows[0]) if fusion_rows else 0},
        "gps": {"rows": len(gps_rows), "cols": len(gps_rows[0]) if gps_rows else 0},
        "rate": {"rows": len(rate_rows), "cols": len(rate_rows[0]) if rate_rows else 0},
        "vibe": {"rows": len(vibe_rows), "cols": len(vibe_rows[0]) if vibe_rows else 0},
        "baro": {"rows": len(baro_rows), "cols": len(baro_rows[0]) if baro_rows else 0},
        "bat": {"rows": len(bat_rows), "cols": len(bat_rows[0]) if bat_rows else 0},
        "mag": {"rows": len(mag_rows), "cols": len(mag_rows[0]) if mag_rows else 0},
    }

    return records, ground_truth


def get_ground_truth(system_id: str = None) -> Dict[str, Any]:
    """
    Return ground-truth labels for the loaded Kaggle dataset.

    Returns dict with:
      - labels: list of int (0-4) parallel to the records
      - label_map: mapping from int to fault name
      - distribution: count per label
    """
    _, ground_truth = _load_kaggle_data()
    dist = {}
    for lbl in ground_truth:
        name = LABEL_MAP.get(lbl, f"unknown_{lbl}")
        dist[name] = dist.get(name, 0) + 1

    return {
        "labels": ground_truth,
        "label_map": LABEL_MAP,
        "distribution": dist,
        "total_records": len(ground_truth),
        "total_anomalous": sum(1 for l in ground_truth if l != 0),
        "total_normal": sum(1 for l in ground_truth if l == 0),
    }


def generate_tlm_uav_description_file() -> str:
    """Generate a documentation markdown string for the TLM-UAV demo system."""
    return """# TLM-UAV Telemetry System — Real Kaggle Dataset

## System Overview
Multi-rotor UAV (quadcopter) telemetry data from the TLM:UAV Anomaly Detection
Dataset (Kaggle). Collected via ArduPilot Software-In-The-Loop (SITL) simulation
with real fault injection using the Time Line Modeling (TLM) method.

**This is REAL data from academic research, not synthetic.**

## Data Source
- **Dataset**: TLM:UAV Anomaly Detection Datasets
- **Source**: https://www.kaggle.com/datasets/luyucwnu/tlmuav-anomaly-detection-datasets
- **Paper**: Yang et al., "Acquisition and Processing of UAV Fault Data Based on
  Time Line Modeling Method", Applied Sciences 13(7):4301, 2023.
- **Primary file**: Fusion_Data.csv (12,253 records, 18 sensor features + label)

## Sensor Groups (Fusion_Data.csv)

### ATT — Attitude
- **att_DesRoll / att_Roll**: Desired vs actual roll angle (deg)
- **att_DesPitch / att_Pitch**: Desired vs actual pitch angle (deg)
- **att_DesYaw / att_Yaw**: Desired vs actual yaw heading (deg)
- **att_ErrRP**: Roll/Pitch error magnitude
- **att_ErrYaw**: Yaw error magnitude

### MAG — Magnetometer
- **mag_X / mag_Y / mag_Z**: 3-axis magnetic field readings

### IMU — Inertial Measurement Unit
- **imu_GyrX / imu_GyrY / imu_GyrZ**: Gyroscope rates (rad/s)
- **imu_AccX / imu_AccY / imu_AccZ**: Accelerometer readings (m/s^2)
  - Z-axis should be ~-9.81 in stable hover

## Fault Types (Ground Truth Labels)

| Label | Fault | Symptoms |
|-------|-------|----------|
| 0 | Normal | Nominal flight |
| 1 | GPS fault | Position drift, satellite loss |
| 2 | Accelerometer fault | IMU bias growth, noise amplification |
| 3 | Engine fault | Thrust loss, altitude drop, vibration surge |
| 4 | RC System fault | Control freeze / erratic commands |

## Ground Truth Distribution
- Normal (0): ~53.6% (6,567 records)
- GPS fault (1): ~14.9% (1,823 records)
- Accelerometer fault (2): ~15.2% (1,863 records)
- Engine fault (3): ~11.8% (1,449 records)
- RC System fault (4): ~4.5% (551 records)

## Evaluation Note
Labels are available for post-analysis comparison but are NOT provided
to the anomaly detection system. This enables fair benchmarking of the
UAIE platform against known ground truth.
"""


def generate_full_tlm_uav_package() -> Dict[str, Any]:
    """
    Load the real TLM-UAV Kaggle dataset and package it for the demo endpoint.

    Returns a dict with:
      - records: list of dicts (NO labels — these go to the analysis engine)
      - ground_truth: dict with labels and distribution (stored separately)
      - metadata, discovered_fields, relationships, etc.
    """
    records, ground_truth_labels = _load_kaggle_data()
    description = generate_tlm_uav_description_file()
    gt_info = get_ground_truth()

    sample = records[0] if records else {}
    sensor_fields = [k for k in sample.keys() if k not in ("timestamp", "row_index")]

    metadata = {
        "system_name": "UAV Copter - TLM Kaggle Dataset",
        "system_type": "uav",
        "description": (
            "REAL UAV telemetry data from the TLM:UAV Anomaly Detection Dataset (Kaggle). "
            "12,253 records across ATT, MAG, and IMU sensor groups with four fault types. "
            "Ground truth labels available for post-analysis evaluation."
        ),
        "confidence": 0.98,
        "record_count": len(records),
        "field_count": len(sensor_fields) + 2,  # +timestamp +row_index
        "demo_anomalies_injected": True,
        "anomaly_types": ["gps_fault", "accelerometer_fault", "engine_fault", "rc_system_fault"],
        "dataset_reference": (
            "TLM:UAV Anomaly Detection Datasets — "
            "https://www.kaggle.com/datasets/luyucwnu/tlmuav-anomaly-detection-datasets"
        ),
        "is_real_data": True,
    }

    discovered_fields = [
        {
            "name": "timestamp",
            "display_name": "Timestamp",
            "type": "datetime",
            "category": "temporal",
            "description": "Telemetry sample timestamp (10 Hz, reconstructed)",
            "confidence": 0.99,
        },
        {
            "name": "row_index",
            "display_name": "Row Index",
            "type": "integer",
            "category": "identifier",
            "description": "Sequential row number from the original Kaggle CSV",
            "confidence": 0.99,
        },
        # ── ATT — Attitude ──────────────────────────────────────────
        {
            "name": "att_DesRoll",
            "display_name": "Desired Roll",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Commanded roll angle from flight controller",
            "confidence": 0.95,
        },
        {
            "name": "att_Roll",
            "display_name": "Actual Roll",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Measured actual roll angle",
            "engineering_context": {
                "typical_range": {"min": -35, "max": 35},
                "what_high_means": "Large roll angle — aggressive maneuver or loss of control",
                "safety_critical": True,
            },
            "confidence": 0.95,
        },
        {
            "name": "att_DesPitch",
            "display_name": "Desired Pitch",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Commanded pitch angle",
            "confidence": 0.95,
        },
        {
            "name": "att_Pitch",
            "display_name": "Actual Pitch",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Measured actual pitch angle",
            "engineering_context": {
                "typical_range": {"min": -35, "max": 35},
                "safety_critical": True,
            },
            "confidence": 0.95,
        },
        {
            "name": "att_DesYaw",
            "display_name": "Desired Yaw",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Commanded yaw heading (0-360)",
            "confidence": 0.94,
        },
        {
            "name": "att_Yaw",
            "display_name": "Actual Yaw",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Measured yaw heading",
            "confidence": 0.94,
        },
        {
            "name": "att_ErrRP",
            "display_name": "Roll/Pitch Error",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Combined roll/pitch error magnitude",
            "engineering_context": {
                "typical_range": {"min": 0, "max": 0.1},
                "what_high_means": "Attitude control is struggling — possible IMU or motor issue",
                "safety_critical": True,
            },
            "confidence": 0.93,
        },
        {
            "name": "att_ErrYaw",
            "display_name": "Yaw Error",
            "type": "float",
            "physical_unit": "deg",
            "category": "content",
            "component": "Attitude Controller",
            "description": "Yaw error magnitude",
            "engineering_context": {
                "typical_range": {"min": 0, "max": 0.1},
                "what_high_means": "Yaw tracking failure — compass interference or motor asymmetry",
            },
            "confidence": 0.93,
        },
        # ── MAG — Magnetometer ──────────────────────────────────────
        {
            "name": "mag_X",
            "display_name": "Magnetometer X",
            "type": "float",
            "physical_unit": "mGauss",
            "category": "content",
            "component": "Magnetometer",
            "description": "X-axis magnetic field reading",
            "confidence": 0.92,
        },
        {
            "name": "mag_Y",
            "display_name": "Magnetometer Y",
            "type": "float",
            "physical_unit": "mGauss",
            "category": "content",
            "component": "Magnetometer",
            "description": "Y-axis magnetic field reading",
            "confidence": 0.92,
        },
        {
            "name": "mag_Z",
            "display_name": "Magnetometer Z",
            "type": "float",
            "physical_unit": "mGauss",
            "category": "content",
            "component": "Magnetometer",
            "description": "Z-axis magnetic field reading",
            "confidence": 0.92,
        },
        # ── IMU — Inertial Measurement Unit ─────────────────────────
        {
            "name": "imu_GyrX",
            "display_name": "Gyroscope X",
            "type": "float",
            "physical_unit": "rad/s",
            "category": "content",
            "component": "IMU / Gyroscope",
            "description": "Body-frame roll rate",
            "engineering_context": {
                "typical_range": {"min": -2, "max": 2},
            },
            "confidence": 0.94,
        },
        {
            "name": "imu_GyrY",
            "display_name": "Gyroscope Y",
            "type": "float",
            "physical_unit": "rad/s",
            "category": "content",
            "component": "IMU / Gyroscope",
            "description": "Body-frame pitch rate",
            "confidence": 0.94,
        },
        {
            "name": "imu_GyrZ",
            "display_name": "Gyroscope Z",
            "type": "float",
            "physical_unit": "rad/s",
            "category": "content",
            "component": "IMU / Gyroscope",
            "description": "Body-frame yaw rate",
            "confidence": 0.94,
        },
        {
            "name": "imu_AccX",
            "display_name": "Accelerometer X",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "IMU / Accelerometer",
            "description": "Body-frame X-axis acceleration",
            "engineering_context": {
                "typical_range": {"min": -2, "max": 2},
            },
            "confidence": 0.94,
        },
        {
            "name": "imu_AccY",
            "display_name": "Accelerometer Y",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "IMU / Accelerometer",
            "description": "Body-frame Y-axis acceleration",
            "confidence": 0.94,
        },
        {
            "name": "imu_AccZ",
            "display_name": "Accelerometer Z",
            "type": "float",
            "physical_unit": "m/s^2",
            "category": "content",
            "component": "IMU / Accelerometer",
            "description": "Body-frame Z-axis acceleration (includes gravity)",
            "engineering_context": {
                "typical_range": {"min": -10.5, "max": -9.0},
                "what_high_means": "Upward acceleration or sensor bias",
                "what_low_means": "Downward acceleration beyond gravity",
                "safety_critical": True,
            },
            "confidence": 0.94,
        },
    ]

    relationships = [
        {
            "fields": ["att_DesRoll", "att_Roll"],
            "relationship": "control_target",
            "description": "Actual roll should track desired roll; divergence indicates control failure",
            "diagnostic_value": "Gap diagnoses actuator or attitude control faults",
        },
        {
            "fields": ["att_DesPitch", "att_Pitch"],
            "relationship": "control_target",
            "description": "Actual pitch should track desired pitch",
            "diagnostic_value": "Used alongside roll tracking to isolate axis-specific failures",
        },
        {
            "fields": ["att_DesYaw", "att_Yaw"],
            "relationship": "control_target",
            "description": "Yaw heading should match commanded heading",
            "diagnostic_value": "Yaw divergence may indicate compass interference or RC fault",
        },
        {
            "fields": ["att_ErrRP", "att_ErrYaw"],
            "relationship": "correlation",
            "description": "Both errors rising together indicates systemic attitude control degradation",
        },
        {
            "fields": ["imu_AccX", "imu_AccY"],
            "relationship": "correlation",
            "description": "Lateral accelerations should be near zero in stable hover",
            "diagnostic_value": "Simultaneous bias growth indicates accelerometer calibration failure",
        },
        {
            "fields": ["imu_GyrX", "imu_GyrY", "imu_GyrZ"],
            "relationship": "proportional",
            "description": "Gyro rates should be small in stable flight; all elevated means vibration",
        },
        {
            "fields": ["mag_X", "mag_Y", "mag_Z"],
            "relationship": "magnitude_constraint",
            "description": "Total magnetic field magnitude should be roughly constant (~500 mGauss); "
                           "sudden changes indicate interference",
        },
    ]

    blind_spots = [
        "No GPS position data in Fusion — cannot directly detect position drift",
        "No barometric altimeter — cannot cross-validate altitude",
        "No battery voltage/current — cannot detect power supply degradation",
        "No vibration (VIBE) data in Fusion — cannot assess mechanical health",
        "No throttle/motor data — cannot detect engine thrust loss directly",
    ]

    return {
        "records": records,
        "ground_truth": gt_info,
        "metadata": metadata,
        "description_content": description,
        "discovered_fields": discovered_fields,
        "relationships": relationships,
        "blind_spots": blind_spots,
        "analysis_summary": {
            "files_analyzed": 7,
            "total_records": len(records),
            "unique_fields": len(discovered_fields),
            "ai_powered": True,
            "data_source": "Kaggle TLM:UAV (real data)",
        },
        "recommendation": {
            "suggested_name": "UAV Copter - TLM Kaggle Dataset",
            "suggested_type": "uav",
            "suggested_description": (
                "REAL UAV telemetry from the TLM:UAV Anomaly Detection Dataset. "
                "12,253 records with ATT, MAG, and IMU sensors. Contains four fault "
                "types for benchmarking anomaly detection accuracy."
            ),
            "confidence": 0.98,
            "system_subtype": "Multi-Rotor Quadcopter",
            "domain": "Aerospace / UAV",
            "detected_components": [
                {
                    "name": "Attitude Controller",
                    "role": "Desired vs actual attitude (roll/pitch/yaw)",
                    "fields": ["att_DesRoll", "att_Roll", "att_DesPitch", "att_Pitch",
                               "att_DesYaw", "att_Yaw", "att_ErrRP", "att_ErrYaw"],
                },
                {
                    "name": "Magnetometer",
                    "role": "Compass heading and magnetic field sensing",
                    "fields": ["mag_X", "mag_Y", "mag_Z"],
                },
                {
                    "name": "IMU",
                    "role": "Inertial measurement (accelerometer + gyroscope)",
                    "fields": ["imu_GyrX", "imu_GyrY", "imu_GyrZ",
                               "imu_AccX", "imu_AccY", "imu_AccZ"],
                },
            ],
            "probable_use_case": "UAV anomaly detection benchmarking with ground truth",
            "data_characteristics": {
                "temporal_resolution": "~50 ms (reconstructed)",
                "duration_estimate": "~20 minutes of flight",
                "completeness": "100%",
                "data_source": "Kaggle TLM:UAV (real SITL telemetry)",
            },
        },
    }
