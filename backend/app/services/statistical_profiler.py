"""
Statistical Profiler — Rich Field Profiling Before LLM

Builds a comprehensive statistical profile for each field in the dataset.
This profile is passed to the LLM so it can make informed decisions about
field meanings, units, and relationships.

Runs entirely locally — no API calls.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("uaie.profiler")


def build_field_profiles(records: List[Dict]) -> List[Dict[str, Any]]:
    """
    Build a rich statistical profile for every field in the dataset.

    For each field produces:
      - type, distribution shape, percentiles, cardinality
      - pattern detection for strings (timestamps, IDs, enums, codes)
      - temporal analysis (monotonicity, interval regularity)
      - null/missing pattern analysis
      - value clustering hints
    """
    logger.info("build_field_profiles: %d records", len(records))
    if not records:
        logger.warning("build_field_profiles: no records — returning empty")
        return []

    # Sanitize records — stringify unhashable types
    clean = []
    for rec in records:
        row = {}
        for k, v in rec.items():
            if isinstance(v, (dict, list)):
                row[k] = str(v)[:200] if v else None
            else:
                row[k] = v
        clean.append(row)

    df = pd.DataFrame(clean)
    logger.info("build_field_profiles: DataFrame created with %d cols: %s", len(df.columns), list(df.columns))
    profiles: List[Dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        profile = _profile_field(col, series, len(df))
        logger.debug("  profiled '%s' → type=%s, category=%s", col, profile.get("detected_type"), profile.get("detected_category"))
        profiles.append(profile)

    logger.info("build_field_profiles: done — %d profiles", len(profiles))
    return profiles


def build_dataset_summary(
    records: List[Dict],
    field_profiles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a dataset-level summary from the field profiles.
    """
    n_records = len(records)
    n_fields = len(field_profiles)

    type_counts = {}
    for fp in field_profiles:
        t = fp.get("detected_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    category_counts = {}
    for fp in field_profiles:
        c = fp.get("detected_category", "unknown")
        category_counts[c] = category_counts.get(c, 0) + 1

    return {
        "record_count": n_records,
        "field_count": n_fields,
        "type_breakdown": type_counts,
        "category_breakdown": category_counts,
    }


# ─── Internal helpers ────────────────────────────────────────────────────


def _profile_field(name: str, series: pd.Series, total_rows: int) -> Dict[str, Any]:
    """Create a rich profile for a single field."""

    profile: Dict[str, Any] = {
        "name": name,
        "total_rows": total_rows,
    }

    # ── Null analysis ────────────────────────────────────────────
    null_count = int(series.isna().sum())
    profile["null_count"] = null_count
    profile["null_pct"] = round(null_count / max(total_rows, 1) * 100, 2)

    non_null = series.dropna()
    if len(non_null) == 0:
        profile["detected_type"] = "empty"
        profile["detected_category"] = "auxiliary"
        return profile

    # ── Type detection ───────────────────────────────────────────
    if pd.api.types.is_bool_dtype(series):
        profile["detected_type"] = "boolean"
        profile["detected_category"] = "auxiliary"
        profile["true_pct"] = round(non_null.sum() / len(non_null) * 100, 2)
        return profile

    if pd.api.types.is_numeric_dtype(series):
        return _profile_numeric(name, non_null, profile)

    if pd.api.types.is_datetime64_any_dtype(series):
        return _profile_timestamp_native(name, non_null, profile)

    # String / object — deeper analysis needed
    return _profile_string(name, non_null, profile)


def _profile_numeric(
    name: str, series: pd.Series, profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Profile a numeric field."""

    profile["detected_type"] = "numeric"

    vals = series.to_numpy(dtype=float, na_value=np.nan)
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        profile["detected_category"] = "auxiliary"
        return profile

    # Core statistics
    profile["min"] = _safe(float(np.min(vals)))
    profile["max"] = _safe(float(np.max(vals)))
    profile["mean"] = _safe(float(np.mean(vals)))
    profile["median"] = _safe(float(np.median(vals)))
    profile["std"] = _safe(float(np.std(vals)))

    # Percentiles
    try:
        pcts = np.percentile(vals, [5, 25, 75, 95])
        profile["p5"] = _safe(float(pcts[0]))
        profile["p25"] = _safe(float(pcts[1]))
        profile["p75"] = _safe(float(pcts[2]))
        profile["p95"] = _safe(float(pcts[3]))
    except Exception:
        pass

    # Cardinality
    unique = len(set(vals))
    profile["unique_count"] = unique

    # Is integer-valued?
    is_int = bool(np.all(vals == np.floor(vals)))
    profile["is_integer"] = is_int

    # Monotonicity (important for timestamp / index detection)
    if len(vals) > 2:
        diffs = np.diff(vals)
        mono_inc = bool(np.all(diffs >= 0))
        mono_dec = bool(np.all(diffs <= 0))
        profile["is_monotonic_increasing"] = mono_inc
        profile["is_monotonic_decreasing"] = mono_dec

        # Regular interval detection
        if mono_inc and len(diffs) > 1:
            median_diff = float(np.median(diffs))
            if median_diff > 0:
                diff_std = float(np.std(diffs))
                profile["interval_median"] = _safe(median_diff)
                profile["interval_std"] = _safe(diff_std)
                profile["interval_regular"] = diff_std / median_diff < 0.1 if median_diff > 0 else False
    else:
        profile["is_monotonic_increasing"] = False
        profile["is_monotonic_decreasing"] = False

    # Distribution shape
    if len(vals) >= 20:
        try:
            from scipy import stats as sp_stats
            skew = float(sp_stats.skew(vals))
            kurt = float(sp_stats.kurtosis(vals))
            profile["skewness"] = _safe(skew)
            profile["kurtosis"] = _safe(kurt)
        except Exception:
            pass

    # Constant / near-constant detection
    if profile.get("std", 0) == 0:
        profile["is_constant"] = True
    elif unique <= 2:
        profile["is_binary_numeric"] = True

    # Timestamp heuristic: 10-digit integers in Unix epoch range
    if (is_int
            and profile.get("min", 0) > 1_000_000_000
            and profile.get("max", 0) < 2_000_000_000
            and profile.get("is_monotonic_increasing", False)):
        profile["likely_unix_timestamp"] = True

    # Epoch millis
    if (is_int
            and profile.get("min", 0) > 1_000_000_000_000
            and profile.get("max", 0) < 2_000_000_000_000
            and profile.get("is_monotonic_increasing", False)):
        profile["likely_unix_timestamp_ms"] = True

    # Sample values (first 8 non-null)
    profile["sample_values"] = [_safe(float(v)) for v in vals[:8]]

    # ── Category guess ──────────────────────────────────────────
    profile["detected_category"] = _guess_numeric_category(name, profile)

    return profile


def _profile_timestamp_native(
    name: str, series: pd.Series, profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Profile a native datetime field."""
    profile["detected_type"] = "timestamp"
    profile["detected_category"] = "temporal"

    try:
        profile["min"] = str(series.min())
        profile["max"] = str(series.max())
        profile["sample_values"] = [str(v) for v in series.head(5).tolist()]
    except Exception:
        pass

    return profile


def _profile_string(
    name: str, series: pd.Series, profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Profile a string / object field."""

    str_series = series.astype(str)
    unique_count = int(str_series.nunique())
    total = len(str_series)
    profile["unique_count"] = unique_count

    # Sample values — up to 10 most common
    top = str_series.value_counts().head(10)
    profile["top_values"] = {str(k): int(v) for k, v in top.items()}
    profile["sample_values"] = list(top.index[:8])

    # Average length
    lengths = str_series.str.len()
    profile["avg_length"] = _safe(float(lengths.mean()))
    profile["max_length"] = int(lengths.max()) if len(lengths) > 0 else 0

    # ── Detect if it looks like a timestamp string ──────────────
    timestamp_patterns = [
        r"\d{4}[-/]\d{2}[-/]\d{2}",                 # 2024-01-15
        r"\d{2}[-/]\d{2}[-/]\d{4}",                 # 15/01/2024
        r"\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}",  # ISO datetime
    ]
    sample = str_series.head(20).tolist()
    ts_matches = sum(
        1 for v in sample
        if any(re.search(p, str(v)) for p in timestamp_patterns)
    )
    if ts_matches > len(sample) * 0.7:
        profile["detected_type"] = "timestamp_string"
        profile["detected_category"] = "temporal"
        return profile

    # ── Detect enum / categorical ───────────────────────────────
    cardinality_ratio = unique_count / max(total, 1)
    if cardinality_ratio < 0.05 or unique_count <= 20:
        profile["detected_type"] = "categorical"
        profile["detected_category"] = _guess_string_category(name, profile)
        return profile

    # ── Detect ID-like strings ──────────────────────────────────
    if cardinality_ratio > 0.9:
        profile["detected_type"] = "identifier_string"
        profile["detected_category"] = "identifier"
        return profile

    # ── Long text (description) ─────────────────────────────────
    if profile.get("avg_length", 0) > 50:
        profile["detected_type"] = "text"
        profile["detected_category"] = "auxiliary"
        return profile

    # Default
    profile["detected_type"] = "string"
    profile["detected_category"] = _guess_string_category(name, profile)
    return profile


# ─── Category guessing ───────────────────────────────────────────────


_TEMPORAL_HINTS = {
    "time", "timestamp", "date", "datetime", "epoch", "ts",
    "created", "updated", "modified",
}

_IDENTIFIER_HINTS = {
    "id", "uuid", "serial", "name", "label", "tag", "key",
    "index", "row", "record", "line_number",
}


def _guess_numeric_category(name: str, profile: Dict) -> str:
    """Guess category for a numeric field."""
    low = name.lower()

    # Temporal
    if any(low == h or low.endswith(f"_{h}") or low.startswith(f"{h}_") for h in _TEMPORAL_HINTS):
        return "temporal"
    if profile.get("likely_unix_timestamp") or profile.get("likely_unix_timestamp_ms"):
        return "temporal"

    # Identifier
    if any(low == h or low.endswith(f"_{h}") or low.startswith(f"{h}_") for h in _IDENTIFIER_HINTS):
        return "identifier"

    # Constant / binary → auxiliary
    if profile.get("is_constant"):
        return "auxiliary"

    return "content"


def _guess_string_category(name: str, profile: Dict) -> str:
    """Guess category for a string field."""
    low = name.lower()

    if any(low == h or low.endswith(f"_{h}") or low.startswith(f"{h}_") for h in _TEMPORAL_HINTS):
        return "temporal"
    if any(low == h or low.endswith(f"_{h}") or low.startswith(f"{h}_") for h in _IDENTIFIER_HINTS):
        return "identifier"

    # Status / state / mode fields → content (useful for analysis)
    status_hints = {"status", "state", "mode", "fault", "error", "alarm", "warning", "level", "grade"}
    if any(h in low for h in status_hints):
        return "content"

    return "auxiliary"


def _safe(value: float) -> Optional[float]:
    """Convert to float, returning None for NaN/inf."""
    if value is None:
        return None
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None
