"""
Shared utility functions for UAIE backend.

Consolidates helpers that were duplicated across multiple modules:
  - sanitize_for_json: numpy/pandas → native Python conversion
  - anomaly_to_dict: Anomaly dataclass → API dict
  - merge_ai_anomalies: deduplicated merging of AI findings
  - get_data_dir: resolve the DATA_DIR path
  - get_analyses_dir: resolve the analyses sub-directory
  - build_field_statistics: compute field-level stats from records
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ─── Data directory resolution ──────────────────────────────────────

def get_data_dir() -> Path:
    """Return the resolved data directory (works in Docker and locally)."""
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return Path(data_dir)


def get_analyses_dir() -> Path:
    """Return the analyses sub-directory, creating it if needed."""
    d = get_data_dir() / "analyses"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_saved_analysis(system_id: str) -> Optional[Dict[str, Any]]:
    """Load saved analysis JSON for a system, or None if absent."""
    path = get_analyses_dir() / f"{system_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_analysis(system_id: str, analysis: Dict[str, Any]) -> None:
    """Persist an analysis result to disk."""
    path = get_analyses_dir() / f"{system_id}.json"
    try:
        path.write_text(json.dumps(
            sanitize_for_json(analysis), indent=2, default=str
        ))
    except OSError:
        pass


# ─── JSON serialization ────────────────────────────────────────────

def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ─── Anomaly helpers ────────────────────────────────────────────────

def anomaly_to_dict(a) -> Dict[str, Any]:
    """Convert a rule-based Anomaly dataclass to API dict format."""
    return {
        "id": a.id,
        "type": a.anomaly_type.value,
        "severity": a.severity.value,
        "title": a.title,
        "description": a.description,
        "affected_fields": [a.field_name] + a.related_fields,
        "natural_language_explanation": a.natural_language_explanation,
        "possible_causes": a.possible_causes,
        "recommendations": a.recommendations,
        "impact_score": a.impact_score,
        "confidence": a.confidence,
        "value": a.value,
        "expected_range": a.expected_range,
        "contributing_agents": ["Rule Engine"],
        "web_references": [],
        "agent_perspectives": [],
    }


def merge_ai_anomalies(anomalies: List[Dict], ai_result: Optional[Dict]) -> None:
    """Merge AI-found anomalies into the existing list, deduplicating by title overlap."""
    from .services.recommendation import titles_overlap

    if not ai_result or not ai_result.get("anomalies"):
        return
    for ai_anomaly in ai_result["anomalies"]:
        is_duplicate = False
        for existing in anomalies:
            if titles_overlap(existing.get("title", ""), ai_anomaly.get("title", "")):
                existing.setdefault("contributing_agents", []).extend(
                    ai_anomaly.get("contributing_agents", [])
                )
                existing.setdefault("agent_perspectives", []).extend(
                    ai_anomaly.get("agent_perspectives", [])
                )
                existing.setdefault("web_references", []).extend(
                    ai_anomaly.get("web_references", [])
                )
                if ai_anomaly.get("confidence", 0) > existing.get("confidence", 0):
                    existing["confidence"] = ai_anomaly["confidence"]
                is_duplicate = True
                break
        if not is_duplicate:
            anomalies.append(ai_anomaly)


# ─── Statistics helpers ─────────────────────────────────────────────

def build_field_statistics(
    records: List[Dict],
    source_count: int = 1,
) -> Optional[Dict[str, Any]]:
    """Build field-level statistics dict from raw records."""
    if not records:
        return None
    try:
        df = pd.DataFrame(records)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        fields = []
        for col in df.columns:
            info: Dict[str, Any] = {
                "name": col,
                "type": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
            }
            if col in numeric_cols:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    info["min"] = float(non_null.min())
                    info["max"] = float(non_null.max())
                    info["mean"] = float(non_null.mean())
                    info["std"] = float(non_null.std())
                else:
                    info["min"] = info["max"] = info["mean"] = info["std"] = None
            fields.append(info)

        return {
            "total_records": len(df),
            "total_sources": source_count,
            "field_count": len(df.columns),
            "fields": fields,
        }
    except Exception:
        return None
