"""
Demo Data Generator for UAIE

Generates realistic HVAC sensor data with embedded anomalies that
will trigger all 25 AI agents during analysis.
"""

import math
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple


def generate_demo_data(
    num_records: int = 1000,
    include_anomalies: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate demo HVAC sensor data with embedded anomalies.

    Returns:
        Tuple of (records, metadata) where metadata contains system info
    """
    random.seed(42)  # Reproducible demo

    base_time = datetime.now() - timedelta(hours=num_records // 60)
    records = []

    # Simulate realistic HVAC patterns
    for i in range(num_records):
        timestamp = base_time + timedelta(minutes=i)
        hour = timestamp.hour

        # Day/night cycle affects HVAC
        is_daytime = 6 <= hour < 22
        is_peak = 9 <= hour < 17

        # Base values with realistic variation
        base_temp = 21.5 if is_daytime else 19.0
        base_humidity = 45.0
        base_power = 2500 if is_peak else 1200

        # Add natural variation
        temp_noise = random.gauss(0, 0.3)
        humidity_noise = random.gauss(0, 2)
        power_noise = random.gauss(0, 50)

        record = {
            "timestamp": timestamp.isoformat(),
            "device_id": "HVAC-001",
            "location": "Building A - Floor 2",
            "zone_temperature_c": round(base_temp + temp_noise, 2),
            "setpoint_temperature_c": 21.0 if is_daytime else 19.0,
            "return_air_temp_c": round(base_temp + temp_noise + random.gauss(1, 0.2), 2),
            "supply_air_temp_c": round(base_temp + temp_noise - random.gauss(3, 0.3), 2),
            "outdoor_temp_c": round(15 + 10 * math.sin(hour * math.pi / 12) + random.gauss(0, 1), 2),
            "humidity_percent": round(base_humidity + humidity_noise, 1),
            "power_consumption_w": round(base_power + power_noise, 0),
            "compressor_current_a": round(base_power / 230 + random.gauss(0, 0.3), 2),
            "fan_speed_rpm": round(1200 + random.gauss(0, 30), 0),
            "damper_position_percent": round(min(100, max(0, 50 + random.gauss(0, 10))), 1),
            "filter_pressure_drop_pa": round(120 + i * 0.01 + random.gauss(0, 3), 1),  # Slowly increasing (clogging)
            "refrigerant_pressure_bar": round(12.5 + random.gauss(0, 0.2), 2),
            "operating_mode": "cooling" if base_temp > 20 else "heating",
            "system_status": "ON" if is_daytime else "STANDBY",
            "fault_code": 0,
            "efficiency_cop": round(3.5 + random.gauss(0, 0.1), 2),
        }

        # ============ INJECT ANOMALIES ============
        if include_anomalies:
            record = _inject_anomalies(record, i, num_records, hour, is_daytime)

        records.append(record)

    # Generate metadata about the demo system
    metadata = {
        "system_name": "Building A - HVAC System",
        "system_type": "hvac",
        "description": (
            "Commercial building HVAC system with multiple zones. "
            "This demo data includes various sensor anomalies for AI analysis demonstration."
        ),
        "confidence": 0.95,
        "record_count": num_records,
        "field_count": len(records[0]) if records else 0,
        "demo_anomalies_injected": include_anomalies,
        "anomaly_types": [
            "frozen_sensor",
            "statistical_outlier",
            "efficiency_degradation",
            "filter_clogging",
            "power_spike",
            "temperature_drift",
            "humidity_spike",
            "off_hours_activity",
            "logic_state_conflict",
            "cross_sensor_violation",
        ] if include_anomalies else [],
    }

    return records, metadata


def _inject_anomalies(
    record: Dict[str, Any],
    index: int,
    total: int,
    hour: int,
    is_daytime: bool,
) -> Dict[str, Any]:
    """Inject various anomalies to trigger different AI agents."""

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 1: Frozen sensor (triggers Stagnation Sentinel)
    # Temperature sensor stuck at exactly 21.1111°C for a window
    # ─────────────────────────────────────────────────────────────────────────
    if 200 <= index < 250:
        record["zone_temperature_c"] = 21.1111  # Suspiciously exact value

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 2: Statistical outliers (triggers Statistical Analyst)
    # Occasional extreme values
    # ─────────────────────────────────────────────────────────────────────────
    if index in [100, 300, 600]:
        record["zone_temperature_c"] = 35.5  # Way too hot
        record["humidity_percent"] = 85.0   # Way too humid

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 3: Power spike (triggers Harmonic Distortion, Domain Expert)
    # Unusual power consumption pattern
    # ─────────────────────────────────────────────────────────────────────────
    if 400 <= index < 420:
        record["power_consumption_w"] = round(record["power_consumption_w"] * 1.8, 0)
        record["compressor_current_a"] = round(record["compressor_current_a"] * 2.1, 2)  # Disproportionate

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 4: Efficiency degradation (triggers Efficiency Analyst)
    # COP dropping over time in a section
    # ─────────────────────────────────────────────────────────────────────────
    if 500 <= index < 600:
        degradation = (index - 500) * 0.02
        record["efficiency_cop"] = round(max(1.5, record["efficiency_cop"] - degradation), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 5: Off-hours activity (triggers Human-Context Filter)
    # High power consumption at 2 AM
    # ─────────────────────────────────────────────────────────────────────────
    if hour == 2 and 700 <= index < 750:
        record["power_consumption_w"] = 3500  # High power at 2 AM
        record["system_status"] = "ON"

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 6: Logic state conflict (triggers Logic State Conflict)
    # System says OFF but consuming power
    # ─────────────────────────────────────────────────────────────────────────
    if 800 <= index < 830:
        record["system_status"] = "OFF"
        record["power_consumption_w"] = 1800  # But still consuming power!

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 7: Cross-sensor sync violation (triggers Cross-Sensor Sync)
    # Temperature rises but humidity stays flat (should be inversely correlated)
    # ─────────────────────────────────────────────────────────────────────────
    if 850 <= index < 880:
        record["zone_temperature_c"] = 26.0 + (index - 850) * 0.1  # Rising
        record["humidity_percent"] = 45.0  # Frozen (should drop)

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 8: Micro-drift (triggers Micro-Drift Tracker)
    # Slow monotonic increase in return air temp
    # ─────────────────────────────────────────────────────────────────────────
    if 600 <= index < 700:
        drift = (index - 600) * 0.015
        record["return_air_temp_c"] = round(record["return_air_temp_c"] + drift, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 9: Safety margin violation (triggers Safety Auditor)
    # Refrigerant pressure approaching dangerous level
    # ─────────────────────────────────────────────────────────────────────────
    if 900 <= index < 950:
        record["refrigerant_pressure_bar"] = round(18.5 + random.gauss(0, 0.2), 2)  # Near 20 bar limit

    # ─────────────────────────────────────────────────────────────────────────
    # ANOMALY 10: Data quality issue (triggers Data Quality Inspector)
    # Missing values / nulls in critical field
    # ─────────────────────────────────────────────────────────────────────────
    if index in [150, 151, 152, 450, 451]:
        record["compressor_current_a"] = None  # Missing data

    return record


def generate_demo_description_file() -> str:
    """Generate a description markdown file for the demo system."""
    return """# HVAC System Documentation

## System Overview
This is a commercial building HVAC (Heating, Ventilation, and Air Conditioning)
system installed in Building A, Floor 2.

## Sensor Specifications

### Temperature Sensors
- **zone_temperature_c**: Zone air temperature (°C). Normal range: 18-26°C
- **setpoint_temperature_c**: Desired temperature setting (°C)
- **return_air_temp_c**: Return air duct temperature (°C)
- **supply_air_temp_c**: Supply air duct temperature (°C). Should be 2-4°C below zone temp in cooling mode.
- **outdoor_temp_c**: Outdoor ambient temperature (°C)

### Humidity & Air Quality
- **humidity_percent**: Relative humidity (%). Normal range: 30-60%
- **filter_pressure_drop_pa**: Pressure drop across air filter (Pa). Clean filter: 50-100 Pa, Replace at: 250 Pa

### Electrical Parameters
- **power_consumption_w**: Total electrical power (W). Normal operating range: 1000-3000 W
- **compressor_current_a**: Compressor motor current (A). Normal: 5-15 A

### Mechanical Parameters
- **fan_speed_rpm**: Supply fan speed (RPM). Normal: 800-1500 RPM
- **damper_position_percent**: Outside air damper position (%). 0=closed, 100=open
- **refrigerant_pressure_bar**: Refrigerant line pressure (bar). Normal: 10-15 bar, Critical: >20 bar

### System Status
- **operating_mode**: Current mode (heating/cooling/ventilation)
- **system_status**: ON/OFF/STANDBY
- **fault_code**: Error code (0 = no fault)
- **efficiency_cop**: Coefficient of Performance. Normal: 3.0-4.5

## Safety Limits
| Parameter | Warning | Critical |
|-----------|---------|----------|
| Refrigerant Pressure | 18 bar | 20 bar |
| Compressor Current | 18 A | 22 A |
| Zone Temperature | 28°C | 32°C |

## Maintenance Schedule
- Filter replacement: When pressure_drop_pa > 250
- Compressor inspection: Every 6 months
- Refrigerant check: Annually
"""


def generate_full_demo_package() -> Dict[str, Any]:
    """
    Generate a complete demo package with data files and metadata.

    Returns a dict with all components needed for demo mode.
    """
    records, metadata = generate_demo_data(num_records=1000, include_anomalies=True)
    description = generate_demo_description_file()

    # Build discovered schema (simulating what LLM would discover)
    discovered_fields = [
        {
            "name": "timestamp",
            "display_name": "Timestamp",
            "type": "datetime",
            "category": "temporal",
            "description": "Measurement timestamp",
            "confidence": 0.99,
        },
        {
            "name": "device_id",
            "display_name": "Device ID",
            "type": "string",
            "category": "identifier",
            "description": "HVAC unit identifier",
            "confidence": 0.99,
        },
        {
            "name": "location",
            "display_name": "Location",
            "type": "string",
            "category": "identifier",
            "description": "Physical location of the unit",
            "confidence": 0.98,
        },
        {
            "name": "zone_temperature_c",
            "display_name": "Zone Temperature",
            "type": "float",
            "physical_unit": "°C",
            "category": "content",
            "component": "Temperature Sensor",
            "description": "Measured zone air temperature",
            "engineering_context": {
                "typical_range": {"min": 18, "max": 26},
                "what_high_means": "Cooling system may be undersized or failing",
                "what_low_means": "Heating system may be running excessively",
                "safety_critical": False,
            },
            "confidence": 0.95,
        },
        {
            "name": "setpoint_temperature_c",
            "display_name": "Temperature Setpoint",
            "type": "float",
            "physical_unit": "°C",
            "category": "content",
            "description": "Target temperature setting",
            "confidence": 0.95,
        },
        {
            "name": "return_air_temp_c",
            "display_name": "Return Air Temperature",
            "type": "float",
            "physical_unit": "°C",
            "category": "content",
            "component": "Air Handler",
            "description": "Temperature of air returning to the unit",
            "confidence": 0.92,
        },
        {
            "name": "supply_air_temp_c",
            "display_name": "Supply Air Temperature",
            "type": "float",
            "physical_unit": "°C",
            "category": "content",
            "component": "Air Handler",
            "description": "Temperature of conditioned air being supplied",
            "confidence": 0.92,
        },
        {
            "name": "outdoor_temp_c",
            "display_name": "Outdoor Temperature",
            "type": "float",
            "physical_unit": "°C",
            "category": "content",
            "description": "Ambient outdoor air temperature",
            "confidence": 0.95,
        },
        {
            "name": "humidity_percent",
            "display_name": "Relative Humidity",
            "type": "float",
            "physical_unit": "%",
            "category": "content",
            "description": "Zone relative humidity",
            "engineering_context": {
                "typical_range": {"min": 30, "max": 60},
                "what_high_means": "Risk of condensation and mold growth",
                "what_low_means": "Occupant discomfort, static electricity",
            },
            "confidence": 0.95,
        },
        {
            "name": "power_consumption_w",
            "display_name": "Power Consumption",
            "type": "float",
            "physical_unit": "W",
            "category": "content",
            "component": "Electrical System",
            "description": "Total electrical power draw",
            "engineering_context": {
                "typical_range": {"min": 1000, "max": 3000},
            },
            "confidence": 0.95,
        },
        {
            "name": "compressor_current_a",
            "display_name": "Compressor Current",
            "type": "float",
            "physical_unit": "A",
            "category": "content",
            "component": "Compressor",
            "description": "Compressor motor current draw",
            "engineering_context": {
                "typical_range": {"min": 5, "max": 15},
                "design_limit_hint": {"min": 0, "max": 22},
                "safety_critical": True,
            },
            "confidence": 0.93,
        },
        {
            "name": "fan_speed_rpm",
            "display_name": "Fan Speed",
            "type": "float",
            "physical_unit": "RPM",
            "category": "content",
            "component": "Fan Motor",
            "description": "Supply fan rotational speed",
            "confidence": 0.94,
        },
        {
            "name": "damper_position_percent",
            "display_name": "Damper Position",
            "type": "float",
            "physical_unit": "%",
            "category": "content",
            "component": "Damper Actuator",
            "description": "Outside air damper opening percentage",
            "confidence": 0.90,
        },
        {
            "name": "filter_pressure_drop_pa",
            "display_name": "Filter Pressure Drop",
            "type": "float",
            "physical_unit": "Pa",
            "category": "content",
            "component": "Air Filter",
            "description": "Pressure differential across air filter",
            "engineering_context": {
                "typical_range": {"min": 50, "max": 150},
                "what_high_means": "Filter is clogged and needs replacement",
            },
            "confidence": 0.88,
        },
        {
            "name": "refrigerant_pressure_bar",
            "display_name": "Refrigerant Pressure",
            "type": "float",
            "physical_unit": "bar",
            "category": "content",
            "component": "Refrigeration Circuit",
            "description": "Refrigerant line pressure",
            "engineering_context": {
                "typical_range": {"min": 10, "max": 15},
                "design_limit_hint": {"min": 5, "max": 20},
                "safety_critical": True,
            },
            "confidence": 0.85,
        },
        {
            "name": "operating_mode",
            "display_name": "Operating Mode",
            "type": "string",
            "category": "content",
            "description": "Current HVAC operating mode",
            "confidence": 0.98,
        },
        {
            "name": "system_status",
            "display_name": "System Status",
            "type": "string",
            "category": "content",
            "description": "Current system power status",
            "confidence": 0.98,
        },
        {
            "name": "fault_code",
            "display_name": "Fault Code",
            "type": "integer",
            "category": "content",
            "description": "Active fault/error code (0 = none)",
            "confidence": 0.95,
        },
        {
            "name": "efficiency_cop",
            "display_name": "Coefficient of Performance",
            "type": "float",
            "category": "content",
            "description": "HVAC energy efficiency ratio",
            "engineering_context": {
                "typical_range": {"min": 3.0, "max": 4.5},
                "what_low_means": "System is operating inefficiently, may need maintenance",
            },
            "confidence": 0.90,
        },
    ]

    # Field relationships
    relationships = [
        {
            "fields": ["zone_temperature_c", "humidity_percent"],
            "relationship": "inverse_correlation",
            "description": "Temperature and humidity typically have an inverse relationship",
            "expected_correlation": "negative",
            "diagnostic_value": "If both rise together, check for external moisture source",
        },
        {
            "fields": ["power_consumption_w", "compressor_current_a"],
            "relationship": "proportional",
            "description": "Power consumption should be proportional to compressor current",
            "expected_correlation": "positive",
            "diagnostic_value": "Disproportionate values indicate electrical issues",
        },
        {
            "fields": ["zone_temperature_c", "setpoint_temperature_c"],
            "relationship": "control_target",
            "description": "Zone temperature tracks toward setpoint",
            "diagnostic_value": "Large gap indicates undersized or failing system",
        },
        {
            "fields": ["outdoor_temp_c", "power_consumption_w"],
            "relationship": "load_correlation",
            "description": "Power consumption increases with temperature differential",
            "expected_correlation": "positive",
        },
    ]

    # Blind spots (missing sensors that would improve analysis)
    blind_spots = [
        "No vibration sensor on compressor motor — cannot detect bearing wear early",
        "No CO2 sensor — cannot assess ventilation adequacy for occupancy",
        "No refrigerant flow sensor — cannot detect refrigerant leaks",
        "No airflow sensor — cannot validate fan performance independently",
    ]

    return {
        "records": records,
        "metadata": metadata,
        "description_content": description,
        "discovered_fields": discovered_fields,
        "relationships": relationships,
        "blind_spots": blind_spots,
        "analysis_summary": {
            "files_analyzed": 2,  # data file + description
            "total_records": len(records),
            "unique_fields": len(discovered_fields),
            "ai_powered": True,
        },
        "recommendation": {
            "suggested_name": "Building A - HVAC System",
            "suggested_type": "hvac",
            "suggested_description": (
                "Commercial HVAC system with temperature, humidity, power, and "
                "refrigeration monitoring. Data suggests a cooling-focused operation "
                "with periodic efficiency variations."
            ),
            "confidence": 0.95,
            "system_subtype": "Commercial Air Conditioning",
            "domain": "Building Management",
            "detected_components": [
                {"name": "Compressor", "role": "Refrigeration cycle compression", "fields": ["compressor_current_a", "refrigerant_pressure_bar"]},
                {"name": "Air Handler", "role": "Air circulation and conditioning", "fields": ["fan_speed_rpm", "return_air_temp_c", "supply_air_temp_c"]},
                {"name": "Control System", "role": "System automation", "fields": ["operating_mode", "system_status", "setpoint_temperature_c"]},
            ],
            "probable_use_case": "Commercial building climate control with remote monitoring",
            "data_characteristics": {
                "temporal_resolution": "1 minute",
                "duration_estimate": "~16 hours",
                "completeness": "98.5%",
            },
        },
    }
