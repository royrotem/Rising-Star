"""
LLM-Powered Schema Discovery Service

Makes ONE structured call to Claude to interpret all fields in a dataset,
identify the system type, describe the system, and explain field relationships.

Returns a rich structured output that the rest of the system can use
for anomaly detection, reports, chat, and more.

Falls back to rule-based discovery when no API key is available.
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uaie.llm_discovery")


# ─── System type catalogue ───────────────────────────────────────────

SYSTEM_TYPES: Dict[str, str] = {
    "vehicle": "Vehicle & Automotive",
    "aerospace": "Aerospace & Aviation",
    "robot": "Robotics & Automation",
    "medical_device": "Medical & Biomedical",
    "industrial": "Industrial Manufacturing",
    "energy": "Energy & Power Systems",
    "solar_energy": "Solar & Photovoltaic",
    "wind_energy": "Wind Energy",
    "battery_system": "Battery & Energy Storage",
    "hvac": "HVAC & Building Systems",
    "water_treatment": "Water & Fluid Systems",
    "telecom": "Telecommunications",
    "marine": "Marine & Maritime",
    "rail": "Rail & Railway",
    "agriculture": "Agriculture & AgTech",
    "semiconductor": "Semiconductor & Cleanroom",
    "oil_gas": "Oil, Gas & Pipeline",
    "generic_iot": "Generic IoT / Sensor Network",
}

SYSTEM_TYPE_LIST = "\n".join(f"  - {k}: {v}" for k, v in SYSTEM_TYPES.items())


# ─── Prompt construction ─────────────────────────────────────────────

def _build_prompt(
    field_profiles: List[Dict],
    dataset_summary: Dict,
    description_context: str,
    file_classification: Dict,
) -> str:
    """Build the prompt that will be sent to Claude."""

    # Format field profiles concisely
    fields_text = ""
    for fp in field_profiles:
        lines = [f"  Field: \"{fp['name']}\""]
        lines.append(f"    type: {fp.get('detected_type', 'unknown')}")

        if fp.get("detected_type") == "numeric":
            lines.append(f"    range: [{fp.get('min')}, {fp.get('max')}]")
            lines.append(f"    mean: {fp.get('mean')}, std: {fp.get('std')}, median: {fp.get('median')}")
            if fp.get("p5") is not None:
                lines.append(f"    percentiles: p5={fp.get('p5')}, p25={fp.get('p25')}, p75={fp.get('p75')}, p95={fp.get('p95')}")
            if fp.get("is_integer"):
                lines.append("    integer-valued: yes")
            if fp.get("is_monotonic_increasing"):
                lines.append("    monotonically increasing: yes")
            if fp.get("interval_median") is not None:
                lines.append(f"    interval: median={fp.get('interval_median')}, regular={fp.get('interval_regular')}")
            if fp.get("likely_unix_timestamp"):
                lines.append("    NOTE: values look like Unix timestamps")
            if fp.get("likely_unix_timestamp_ms"):
                lines.append("    NOTE: values look like Unix timestamps (milliseconds)")
            if fp.get("is_constant"):
                lines.append("    NOTE: constant value (zero variance)")
            if fp.get("is_binary_numeric"):
                lines.append("    NOTE: only 2 unique values")
            if fp.get("skewness") is not None:
                lines.append(f"    skewness: {fp.get('skewness')}, kurtosis: {fp.get('kurtosis')}")

        elif fp.get("detected_type") in ("categorical", "string"):
            if fp.get("top_values"):
                top_items = list(fp["top_values"].items())[:6]
                top_str = ", ".join(f'"{k}"({v})' for k, v in top_items)
                lines.append(f"    top values: {top_str}")
            lines.append(f"    unique: {fp.get('unique_count', '?')}")

        elif fp.get("detected_type") in ("timestamp_string", "timestamp"):
            pass  # sample values will show the format

        if fp.get("sample_values"):
            sv = fp["sample_values"][:6]
            lines.append(f"    samples: {sv}")

        lines.append(f"    nulls: {fp.get('null_pct', 0)}%")
        fields_text += "\n".join(lines) + "\n\n"

    # Dataset summary
    summary_text = (
        f"Records: {dataset_summary.get('record_count', '?')}\n"
        f"Fields: {dataset_summary.get('field_count', '?')}\n"
        f"Type breakdown: {dataset_summary.get('type_breakdown', {})}\n"
    )

    # File info
    files_text = ""
    data_files = file_classification.get("data_files", [])
    desc_files = file_classification.get("description_files", [])
    if data_files:
        files_text += f"Data files: {', '.join(data_files)}\n"
    if desc_files:
        files_text += f"Documentation files: {', '.join(desc_files)}\n"

    # Description context
    ctx_section = ""
    if description_context:
        ctx_section = f"""
## Additional Context from Documentation Files
```
{description_context[:3000]}
```
"""

    return f"""You are an expert engineering data analyst. You are given a dataset from a hardware/engineering system. Your job is to fully interpret every field, identify the system, and describe what the data represents.

## Dataset Overview
{summary_text}
{files_text}

## Field Profiles (statistical analysis)
{fields_text}
{ctx_section}

## Available System Types
{SYSTEM_TYPE_LIST}

## Instructions

Analyze ALL fields and the dataset as a whole. Return a JSON object with EXACTLY this structure (no markdown, no extra text — ONLY valid JSON):

{{
  "system_identification": {{
    "system_type": "<one of the system types listed above>",
    "system_type_confidence": <0.0-1.0>,
    "system_subtype": "<free-form specific subtype, e.g. 'photovoltaic_inverter_array'>",
    "system_description": "<2-4 sentence natural language description of what this system is, what it monitors, its approximate scale/capacity, and what the data captures>",
    "domain": "<broad domain, e.g. 'renewable_energy', 'automotive', 'manufacturing'>",
    "detected_components": [
      {{
        "name": "<component name>",
        "role": "<what this component does>",
        "fields": ["<field names belonging to this component>"]
      }}
    ],
    "probable_use_case": "<what this data is likely used for>",
    "data_characteristics": {{
      "temporal_resolution": "<e.g. '1 sample per 60 seconds' or 'irregular'>",
      "duration_estimate": "<e.g. '~24 hours'>",
      "completeness": "<high/medium/low — overall data quality>"
    }}
  }},
  "fields": [
    {{
      "name": "<exact field name from input>",
      "display_name": "<human-readable name, e.g. 'Motor Temperature'>",
      "description": "<1-2 sentence description of what this field measures and why it matters>",
      "type": "<numeric|categorical|boolean|timestamp|identifier|text>",
      "physical_unit": "<SI or common unit abbreviation: V, A, °C, rpm, Pa, m/s, %, etc. null if dimensionless or not applicable>",
      "physical_unit_full": "<full unit name, e.g. 'volts (DC)', 'degrees Celsius'. null if not applicable>",
      "category": "<content|temporal|identifier|auxiliary>",
      "component": "<which detected_component this belongs to, or null>",
      "engineering_context": {{
        "typical_range": {{"min": <number>, "max": <number>}} or null,
        "operating_range_description": "<1 sentence describing what range is normal>",
        "what_high_means": "<what it means when values are high>",
        "what_low_means": "<what it means when values are low>",
        "safety_critical": <true|false>,
        "design_limit_hint": {{"min": <number>, "max": <number>}} or null
      }},
      "value_interpretation": {{
        "assessment": "<1-2 sentences: are the observed values normal? any concerns?>"
      }},
      "confidence": {{
        "type": <0.0-1.0>,
        "unit": <0.0-1.0>,
        "meaning": <0.0-1.0>,
        "overall": <0.0-1.0>
      }},
      "reasoning": "<1 sentence: why you interpreted this field this way>"
    }}
  ],
  "field_relationships": [
    {{
      "fields": ["<field_a>", "<field_b>"],
      "relationship": "<parallel_measurement|causal|derived|inverse|covariate>",
      "description": "<how these fields relate to each other>",
      "expected_correlation": "<e.g. 'high positive (>0.85)'>",
      "diagnostic_value": "<what it means when this relationship breaks>"
    }}
  ],
  "blind_spots": [
    "<missing measurement or gap in the data that limits analysis>"
  ],
  "recommended_confirmation_fields": [
    {{
      "field": "<field name>",
      "reason": "<why this needs human confirmation>",
      "question": "<specific question to ask the engineer>"
    }}
  ]
}}

IMPORTANT RULES:
- Return ONLY the JSON object, no markdown code fences, no explanation before or after.
- Every field from the input MUST appear in the "fields" array.
- Be specific about units — use actual unit symbols (V, A, °C, kPa, rpm), not generic categories.
- For engineering_context, use your domain knowledge to estimate typical/design ranges for the identified system type.
- Only put fields in recommended_confirmation_fields if you are genuinely uncertain (overall confidence < 0.7).
- For field_relationships, focus on the most important diagnostic relationships (max 10).
- The system_description should sound like an expert engineer wrote it — specific, technical, and useful.
"""


# ─── LLM Call ─────────────────────────────────────────────────────────

async def discover_with_llm(
    field_profiles: List[Dict],
    dataset_summary: Dict,
    description_context: str,
    file_classification: Dict,
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Make a single Claude call to interpret the entire dataset.

    Returns the parsed structured output, or None on failure.
    """
    logger.info("discover_with_llm() called with %d field_profiles, api_key_len=%d",
                len(field_profiles), len(api_key) if api_key else 0)

    try:
        from anthropic import AsyncAnthropic
        logger.info("anthropic SDK imported successfully")
    except ImportError:
        logger.error("anthropic SDK NOT INSTALLED — pip install anthropic")
        return None

    if not api_key:
        logger.warning("No API key provided — returning None")
        return None

    prompt = _build_prompt(
        field_profiles, dataset_summary, description_context, file_classification,
    )
    logger.info("Prompt built: %d characters", len(prompt))
    logger.debug("Prompt preview (first 500 chars): %s", prompt[:500])

    try:
        client = AsyncAnthropic(api_key=api_key)
        logger.info("Sending request to Claude (model=claude-sonnet-4-20250514, max_tokens=8192)...")

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )

        logger.info("Claude response received: stop_reason=%s, usage=%s",
                    response.stop_reason,
                    {"input": response.usage.input_tokens, "output": response.usage.output_tokens} if response.usage else "?")

        raw_text = response.content[0].text.strip()
        logger.info("Raw response length: %d chars", len(raw_text))
        logger.debug("Raw response preview (first 300 chars): %s", raw_text[:300])

        # Strip markdown fences if the model wrapped them anyway
        if raw_text.startswith("```"):
            logger.info("Stripping markdown code fences from response")
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].rstrip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].lstrip()

        result = json.loads(raw_text)
        sys_id = result.get("system_identification", {})
        logger.info(
            "LLM discovery SUCCEEDED: system_type=%s, confidence=%.2f, fields=%d, relationships=%d",
            sys_id.get("system_type", "?"),
            sys_id.get("system_type_confidence", 0),
            len(result.get("fields", [])),
            len(result.get("field_relationships", [])),
        )
        return result

    except json.JSONDecodeError as e:
        logger.error("LLM returned INVALID JSON: %s", e)
        logger.error("Raw text that failed to parse (first 500 chars): %s", raw_text[:500] if raw_text else "(empty)")
        return None
    except Exception as e:
        logger.error("LLM discovery call FAILED: %s: %s", type(e).__name__, e)
        import traceback
        logger.error(traceback.format_exc())
        return None


# ─── Cross-Validation ─────────────────────────────────────────────────

def cross_validate(
    llm_result: Dict[str, Any],
    field_profiles: List[Dict],
) -> Dict[str, Any]:
    """
    Validate LLM interpretations against statistical profiles.

    Fixes obvious inconsistencies and adjusts confidence scores.
    Returns the (possibly modified) llm_result.
    """
    profile_map = {fp["name"]: fp for fp in field_profiles}

    for field in llm_result.get("fields", []):
        fname = field.get("name", "")
        fp = profile_map.get(fname)
        if not fp:
            continue

        conf = field.get("confidence", {})

        # ── Validate type ─────────────────────────────────────
        llm_type = field.get("type", "")
        stat_type = fp.get("detected_type", "")

        # If stats say it's a timestamp but LLM says numeric — fix
        if stat_type in ("timestamp", "timestamp_string") and llm_type == "numeric":
            field["type"] = "timestamp"
            field["category"] = "temporal"
            conf["type"] = max(conf.get("type", 0.5), 0.9)

        # If stats detected Unix timestamp pattern
        if fp.get("likely_unix_timestamp") or fp.get("likely_unix_timestamp_ms"):
            if llm_type != "timestamp":
                field["type"] = "timestamp"
                field["category"] = "temporal"
                if fp.get("likely_unix_timestamp"):
                    field["physical_unit"] = "unix_epoch_s"
                    field["physical_unit_full"] = "seconds since 1970-01-01 UTC"
                else:
                    field["physical_unit"] = "unix_epoch_ms"
                    field["physical_unit_full"] = "milliseconds since 1970-01-01 UTC"

        # ── Validate unit vs value range ──────────────────────
        unit = (field.get("physical_unit") or "").lower()
        fmin = fp.get("min")
        fmax = fp.get("max")

        if unit and fmin is not None and fmax is not None:
            issues = _check_unit_range(unit, fmin, fmax)
            if issues:
                # Reduce unit confidence and add a note
                conf["unit"] = max(conf.get("unit", 0.5) - 0.3, 0.1)
                existing_assessment = field.get("value_interpretation", {}).get("assessment", "")
                field.setdefault("value_interpretation", {})["assessment"] = (
                    f"{existing_assessment} [VALIDATION WARNING: {issues}]"
                )

        # ── Validate constant fields ──────────────────────────
        if fp.get("is_constant") and field.get("category") == "content":
            field["category"] = "auxiliary"
            conf["meaning"] = min(conf.get("meaning", 0.5), 0.3)

        # ── Recalculate overall confidence ────────────────────
        type_c = conf.get("type", 0.5)
        unit_c = conf.get("unit", 0.5)
        meaning_c = conf.get("meaning", 0.5)
        conf["overall"] = round((type_c * 0.2 + unit_c * 0.3 + meaning_c * 0.5), 3)
        field["confidence"] = conf

    return llm_result


def _check_unit_range(unit: str, fmin: float, fmax: float) -> str:
    """Check if value range is plausible for the stated unit. Returns issue string or empty."""

    checks = {
        # unit_pattern: (reasonable_min, reasonable_max, description)
        "°c": (-80, 500, "Celsius temperature"),
        "celsius": (-80, 500, "Celsius temperature"),
        "°f": (-100, 1000, "Fahrenheit temperature"),
        "k": (0, 5000, "Kelvin temperature"),
        "v": (-1500, 1500, "Voltage"),
        "volt": (-1500, 1500, "Voltage"),
        "a": (-1000, 1000, "Current (amps)"),
        "amp": (-1000, 1000, "Current (amps)"),
        "%": (-1, 101, "Percentage"),
        "percent": (-1, 101, "Percentage"),
        "rpm": (0, 200000, "RPM"),
        "pa": (-200000, 100000000, "Pressure (Pascals)"),
        "kpa": (-200, 100000, "Pressure (kPa)"),
        "bar": (-3, 1000, "Pressure (bar)"),
        "psi": (-15, 50000, "Pressure (PSI)"),
    }

    for pattern, (rmin, rmax, desc) in checks.items():
        if pattern in unit:
            if fmin < rmin * 2 or fmax > rmax * 2:
                return f"Values [{fmin:.1f}, {fmax:.1f}] unusual for {desc} ({rmin} to {rmax})"
            break

    return ""


# ─── Fallback: Rule-Based Discovery ──────────────────────────────────

def fallback_rule_based(
    field_profiles: List[Dict],
    dataset_summary: Dict,
) -> Dict[str, Any]:
    """
    Generate a discovery result using only rule-based heuristics.
    Used when no API key is available.
    """
    fields = []
    for fp in field_profiles:
        name = fp["name"]
        detected_type = fp.get("detected_type", "string")
        category = fp.get("detected_category", "auxiliary")

        # Map detected_type to output type
        type_map = {
            "numeric": "numeric",
            "boolean": "boolean",
            "categorical": "categorical",
            "timestamp": "timestamp",
            "timestamp_string": "timestamp",
            "identifier_string": "identifier",
            "text": "text",
            "string": "string",
            "empty": "string",
        }
        out_type = type_map.get(detected_type, "string")

        unit = _infer_unit_from_name(name)
        meaning = _infer_meaning_from_name(name)
        display_name = _make_display_name(name)

        # Build confidence
        base_meaning_conf = 0.3 if "Unknown" in meaning else 0.6
        base_unit_conf = 0.6 if unit else 0.2

        field_entry = {
            "name": name,
            "display_name": display_name,
            "description": meaning,
            "type": out_type,
            "physical_unit": unit,
            "physical_unit_full": None,
            "category": category,
            "component": None,
            "engineering_context": {
                "typical_range": None,
                "operating_range_description": None,
                "what_high_means": None,
                "what_low_means": None,
                "safety_critical": False,
                "design_limit_hint": None,
            },
            "value_interpretation": {
                "assessment": _basic_value_assessment(fp),
            },
            "confidence": {
                "type": 0.9 if detected_type != "string" else 0.5,
                "unit": base_unit_conf,
                "meaning": base_meaning_conf,
                "overall": round((0.9 * 0.2 + base_unit_conf * 0.3 + base_meaning_conf * 0.5), 3),
            },
            "reasoning": f"Rule-based inference from field name '{name}' and value statistics.",
        }
        fields.append(field_entry)

    # Guess system type from field names
    sys_type, sys_conf = _guess_system_type([fp["name"] for fp in field_profiles])

    return {
        "system_identification": {
            "system_type": sys_type,
            "system_type_confidence": sys_conf,
            "system_subtype": None,
            "system_description": f"System with {len(fields)} monitored parameters and {dataset_summary.get('record_count', 0)} data records. AI-powered analysis unavailable — configure an Anthropic API key for detailed system identification.",
            "domain": sys_type,
            "detected_components": [],
            "probable_use_case": "Data monitoring and analysis",
            "data_characteristics": {
                "temporal_resolution": _guess_temporal_resolution(field_profiles),
                "duration_estimate": "unknown",
                "completeness": _guess_completeness(field_profiles),
            },
        },
        "fields": fields,
        "field_relationships": [],
        "blind_spots": [
            "AI-powered analysis unavailable — detailed blind spot detection requires an Anthropic API key."
        ],
        "recommended_confirmation_fields": [
            {
                "field": f["name"],
                "reason": "Rule-based inference only — AI could not verify this field's meaning.",
                "question": f"What does the field '{f['name']}' represent in your system?",
            }
            for f in fields
            if f["confidence"]["overall"] < 0.5
        ],
        "_ai_powered": False,
    }


# ─── Fallback helpers ─────────────────────────────────────────────────

_UNIT_PATTERNS = {
    "temperature": (["temp", "celsius", "fahrenheit", "thermal", "heat"], "°C"),
    "voltage": (["volt", "voltage", "vbat", "vcc", "vdd", "vbus"], "V"),
    "current": (["current", "amp", "amps", "ibat"], "A"),
    "pressure": (["pressure", "psi", "bar", "pascal", "kpa"], "Pa"),
    "speed": (["speed", "velocity"], "m/s"),
    "rpm": (["rpm", "rps"], "rpm"),
    "acceleration": (["accel", "acceleration", "g_force"], "m/s²"),
    "position": (["lat", "lon", "altitude", "alt"], "°"),
    "angle": (["angle", "yaw", "pitch", "roll", "heading"], "°"),
    "distance": (["distance", "range", "dist", "odometer"], "m"),
    "frequency": (["freq", "frequency", "hz"], "Hz"),
    "power": (["power", "watt", "watts"], "W"),
    "energy": (["energy", "kwh", "joule"], "kWh"),
    "percentage": (["percent", "pct", "soc", "soh", "level", "efficiency"], "%"),
    "flow": (["flow", "flow_rate"], "L/min"),
    "torque": (["torque"], "Nm"),
}


def _infer_unit_from_name(name: str) -> Optional[str]:
    low = name.lower()
    for _, (keywords, unit) in _UNIT_PATTERNS.items():
        if any(k in low for k in keywords):
            return unit
    return None


_MEANING_PATTERNS = [
    (["temp", "thermal"], "Temperature measurement"),
    (["battery", "batt", "soc"], "Battery-related metric"),
    (["motor", "engine"], "Motor/engine parameter"),
    (["error", "fault", "warn", "alarm"], "Error or warning indicator"),
    (["gps", "lat", "lon", "position"], "Location/positioning data"),
    (["time", "stamp", "date", "epoch"], "Timestamp or time reference"),
    (["speed", "velocity"], "Speed measurement"),
    (["pressure", "psi", "bar"], "Pressure measurement"),
    (["volt", "voltage", "vbus"], "Voltage measurement"),
    (["current", "amp"], "Current measurement"),
    (["power", "watt"], "Power measurement"),
    (["flow", "flow_rate"], "Flow rate measurement"),
    (["vibration", "accel"], "Vibration/acceleration measurement"),
    (["humidity", "rh"], "Humidity measurement"),
    (["rpm", "rps"], "Rotational speed measurement"),
    (["torque"], "Torque measurement"),
    (["level", "tank"], "Level measurement"),
    (["status", "state", "mode"], "Status/state indicator"),
]


def _infer_meaning_from_name(name: str) -> str:
    low = name.lower()
    for keywords, meaning in _MEANING_PATTERNS:
        if any(k in low for k in keywords):
            return meaning
    return "Unknown — requires engineer confirmation"


def _make_display_name(name: str) -> str:
    """Turn 'motor_temp_c' into 'Motor Temp C'."""
    return name.replace("_", " ").replace("-", " ").title()


def _basic_value_assessment(fp: Dict) -> str:
    """Basic assessment from stats alone."""
    if fp.get("is_constant"):
        return "Constant value — no variation observed."
    if fp.get("null_pct", 0) > 50:
        return f"High missing data ({fp['null_pct']}% null)."
    if fp.get("detected_type") == "numeric":
        return f"Range [{fp.get('min')}, {fp.get('max')}], mean={fp.get('mean')}."
    return "No specific concerns."


_SYSTEM_TYPE_KEYWORDS_FALLBACK = {
    "vehicle": ["speed", "velocity", "rpm", "engine", "motor", "battery", "fuel",
                 "odometer", "gps", "latitude", "longitude", "steering", "brake",
                 "throttle", "gear", "wheel", "tire", "can_bus", "obd"],
    "aerospace": ["altitude", "airspeed", "heading", "pitch", "roll", "yaw",
                   "thrust", "fuel_flow", "flap", "rudder", "aileron", "flight"],
    "robot": ["joint", "axis", "torque", "servo", "gripper", "end_effector",
              "pose", "orientation", "actuator", "encoder", "dof"],
    "medical_device": ["patient", "heart", "ecg", "ekg", "blood", "pulse",
                       "oxygen", "saturation", "respiration", "dose"],
    "solar_energy": ["pv", "solar", "irradiance", "ghi", "mppt", "inverter",
                     "string", "panel", "module"],
    "wind_energy": ["turbine", "nacelle", "blade", "rotor", "wind_speed",
                    "gearbox", "yaw"],
    "battery_system": ["soc", "soh", "cell", "bms", "charge", "discharge",
                       "capacity", "cycle"],
    "hvac": ["hvac", "ahu", "chiller", "compressor", "damper", "thermostat",
             "setpoint", "supply_air", "return_air"],
    "water_treatment": ["pump", "filter", "chlorine", "turbidity", "ph",
                        "dissolved", "membrane", "backwash"],
    "oil_gas": ["wellhead", "pipeline", "drill", "crude", "separator",
                "compressor", "flare"],
    "marine": ["hull", "propeller", "rudder", "ballast", "draught",
               "navigation", "knot"],
    "rail": ["traction", "pantograph", "bogie", "signal", "axle", "rail"],
    "agriculture": ["soil", "irrigation", "moisture", "greenhouse", "crop",
                    "fertigation", "ph"],
    "semiconductor": ["wafer", "chamber", "vacuum", "plasma", "deposition",
                      "etch", "cleanroom"],
    "telecom": ["rssi", "snr", "bandwidth", "latency", "antenna", "rf",
                "base_station"],
    "energy": ["generator", "transformer", "grid", "load", "frequency",
               "reactive", "active_power"],
    "industrial": ["pump", "valve", "flow", "pressure", "level", "tank",
                   "conveyor", "plc", "scada", "process", "production",
                   "machine", "vibration", "sensor"],
    "generic_iot": [],
}


def _guess_system_type(field_names: List[str]) -> tuple:
    """Guess system type from field names. Returns (type, confidence)."""
    all_lower = [f.lower() for f in field_names]

    scores = {}
    for sys_type, keywords in _SYSTEM_TYPE_KEYWORDS_FALLBACK.items():
        if not keywords:
            continue
        score = sum(1 for f in all_lower if any(k in f for k in keywords))
        scores[sys_type] = score

    if not scores or max(scores.values()) == 0:
        return "generic_iot", 0.3

    best = max(scores, key=scores.get)
    confidence = min(0.85, scores[best] / max(len(field_names), 1) + 0.4)
    return best, round(confidence, 2)


def _guess_temporal_resolution(profiles: List[Dict]) -> str:
    """Guess temporal resolution from field profiles."""
    for fp in profiles:
        if fp.get("likely_unix_timestamp") and fp.get("interval_median"):
            interval = fp["interval_median"]
            if interval < 1:
                return f"~{interval*1000:.0f}ms"
            elif interval < 60:
                return f"~{interval:.0f} seconds"
            elif interval < 3600:
                return f"~{interval/60:.0f} minutes"
            else:
                return f"~{interval/3600:.1f} hours"
    return "unknown"


def _guess_completeness(profiles: List[Dict]) -> str:
    """Guess overall data completeness."""
    if not profiles:
        return "unknown"
    avg_null = sum(fp.get("null_pct", 0) for fp in profiles) / len(profiles)
    if avg_null < 2:
        return "high"
    elif avg_null < 10:
        return "medium"
    else:
        return "low"
