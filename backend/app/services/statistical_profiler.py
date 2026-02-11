"""
Statistical Profiler — Rich Field Profiling Before LLM

Builds a comprehensive statistical profile for each field in the dataset.
This profile is passed to the LLM so it can make informed decisions about
field meanings, units, and relationships.

Runs entirely locally — no API calls.

Supports two modes:
1. In-memory profiling (original) — for small datasets
2. Incremental/streaming profiling — for large datasets (50GB+)
"""

import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("uaie.profiler")


# ═══════════════════════════════════════════════════════════════════════════
# Incremental Field Profiler — For Large Datasets
# ═══════════════════════════════════════════════════════════════════════════


class IncrementalFieldProfiler:
    """
    Incremental field profiler using Welford's algorithm for online statistics.

    Processes data chunk-by-chunk without keeping all records in memory.
    Suitable for datasets up to 50GB+.

    Usage:
        profiler = IncrementalFieldProfiler()
        for chunk in data_chunks:
            profiler.update(chunk)
        profiles = profiler.finalize()
    """

    # Reservoir size for sampling
    RESERVOIR_SIZE = 1000

    def __init__(self):
        self._field_stats: Dict[str, Dict[str, Any]] = {}
        self._total_records = 0
        self._chunk_count = 0

    def update(self, chunk: pd.DataFrame) -> None:
        """Update statistics with a new chunk of data."""
        if chunk.empty:
            return

        chunk_size = len(chunk)
        self._chunk_count += 1

        for col in chunk.columns:
            if col not in self._field_stats:
                self._field_stats[col] = self._init_field_stats(col)

            series = chunk[col]
            self._update_field_stats(col, series, self._total_records)

        self._total_records += chunk_size
        logger.debug(
            "IncrementalFieldProfiler: chunk %d processed, total records: %d",
            self._chunk_count, self._total_records
        )

    def _init_field_stats(self, name: str) -> Dict[str, Any]:
        """Initialize statistics tracking for a field."""
        return {
            "name": name,
            # Welford's algorithm state
            "count": 0,
            "mean": 0.0,
            "M2": 0.0,  # Sum of squares of differences from mean
            # Min/max tracking
            "min": None,
            "max": None,
            # Null tracking
            "null_count": 0,
            # Type tracking
            "numeric_count": 0,
            "string_count": 0,
            "bool_count": 0,
            "datetime_count": 0,
            # Cardinality (approximate using set up to limit)
            "unique_values": set(),
            "unique_overflow": False,
            "unique_limit": 10000,
            # Reservoir sampling for examples
            "reservoir": [],
            "reservoir_index": 0,
            # String length stats (Welford's for lengths)
            "str_len_count": 0,
            "str_len_mean": 0.0,
            "str_len_M2": 0.0,
            "str_len_max": 0,
            # Monotonicity tracking
            "last_value": None,
            "is_monotonic_inc": True,
            "is_monotonic_dec": True,
            "monotonic_checked": 0,
            # Value counts for top values (limited)
            "value_counts": {},
            "value_counts_limit": 100,
            # Integer check
            "all_integers": True,
            # Timestamp pattern detection
            "timestamp_pattern_matches": 0,
            "timestamp_pattern_checks": 0,
        }

    def _update_field_stats(
        self,
        name: str,
        series: pd.Series,
        chunk_start: int
    ) -> None:
        """Update statistics for a single field with new data."""
        stats = self._field_stats[name]

        # Count nulls
        null_mask = series.isna()
        null_count = int(null_mask.sum())
        stats["null_count"] += null_count

        # Work with non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return

        # Type detection and stats update
        if pd.api.types.is_bool_dtype(series):
            stats["bool_count"] += len(non_null)
            self._update_categorical_stats(stats, non_null)

        elif pd.api.types.is_numeric_dtype(series):
            stats["numeric_count"] += len(non_null)
            self._update_numeric_stats(stats, non_null, chunk_start)

        elif pd.api.types.is_datetime64_any_dtype(series):
            stats["datetime_count"] += len(non_null)
            self._update_datetime_stats(stats, non_null)

        else:
            stats["string_count"] += len(non_null)
            self._update_string_stats(stats, non_null, chunk_start)

        # Reservoir sampling for examples
        self._update_reservoir(stats, non_null, chunk_start)

    def _update_numeric_stats(
        self,
        stats: Dict,
        series: pd.Series,
        chunk_start: int
    ) -> None:
        """Update numeric statistics using Welford's algorithm."""
        vals = series.to_numpy(dtype=float, na_value=np.nan)
        vals = vals[~np.isnan(vals)]

        if len(vals) == 0:
            return

        # Update min/max
        chunk_min = float(np.min(vals))
        chunk_max = float(np.max(vals))
        if stats["min"] is None or chunk_min < stats["min"]:
            stats["min"] = chunk_min
        if stats["max"] is None or chunk_max > stats["max"]:
            stats["max"] = chunk_max

        # Welford's online algorithm for mean and variance
        for val in vals:
            stats["count"] += 1
            delta = val - stats["mean"]
            stats["mean"] += delta / stats["count"]
            delta2 = val - stats["mean"]
            stats["M2"] += delta * delta2

        # Check if all integers
        if stats["all_integers"]:
            stats["all_integers"] = bool(np.all(vals == np.floor(vals)))

        # Update unique values (with overflow protection)
        if not stats["unique_overflow"]:
            for val in vals:
                stats["unique_values"].add(val)
                if len(stats["unique_values"]) > stats["unique_limit"]:
                    stats["unique_overflow"] = True
                    break

        # Monotonicity check (sample-based for efficiency)
        if stats["monotonic_checked"] < 10000:
            for val in vals[:100]:  # Check first 100 per chunk
                if stats["last_value"] is not None:
                    if val < stats["last_value"]:
                        stats["is_monotonic_inc"] = False
                    if val > stats["last_value"]:
                        stats["is_monotonic_dec"] = False
                stats["last_value"] = val
                stats["monotonic_checked"] += 1

    def _update_string_stats(
        self,
        stats: Dict,
        series: pd.Series,
        chunk_start: int
    ) -> None:
        """Update string statistics."""
        str_series = series.astype(str)

        # String length stats using Welford's
        lengths = str_series.str.len()
        for length in lengths:
            stats["str_len_count"] += 1
            delta = length - stats["str_len_mean"]
            stats["str_len_mean"] += delta / stats["str_len_count"]
            delta2 = length - stats["str_len_mean"]
            stats["str_len_M2"] += delta * delta2
            if length > stats["str_len_max"]:
                stats["str_len_max"] = length

        # Update value counts (limited)
        if len(stats["value_counts"]) < stats["value_counts_limit"]:
            for val in str_series:
                if val in stats["value_counts"]:
                    stats["value_counts"][val] += 1
                elif len(stats["value_counts"]) < stats["value_counts_limit"]:
                    stats["value_counts"][val] = 1

        # Update unique values
        if not stats["unique_overflow"]:
            for val in str_series:
                stats["unique_values"].add(val)
                if len(stats["unique_values"]) > stats["unique_limit"]:
                    stats["unique_overflow"] = True
                    break

        # Timestamp pattern detection (sample-based)
        if stats["timestamp_pattern_checks"] < 200:
            timestamp_patterns = [
                r"\d{4}[-/]\d{2}[-/]\d{2}",
                r"\d{2}[-/]\d{2}[-/]\d{4}",
                r"\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}",
            ]
            sample = str_series.head(20).tolist()
            for val in sample:
                stats["timestamp_pattern_checks"] += 1
                if any(re.search(p, str(val)) for p in timestamp_patterns):
                    stats["timestamp_pattern_matches"] += 1

    def _update_datetime_stats(self, stats: Dict, series: pd.Series) -> None:
        """Update datetime statistics."""
        try:
            chunk_min = series.min()
            chunk_max = series.max()

            if stats["min"] is None or chunk_min < stats["min"]:
                stats["min"] = str(chunk_min)
            if stats["max"] is None or chunk_max > stats["max"]:
                stats["max"] = str(chunk_max)
        except Exception:
            pass

    def _update_categorical_stats(self, stats: Dict, series: pd.Series) -> None:
        """Update categorical/boolean statistics."""
        for val in series:
            str_val = str(val)
            if str_val in stats["value_counts"]:
                stats["value_counts"][str_val] += 1
            elif len(stats["value_counts"]) < stats["value_counts_limit"]:
                stats["value_counts"][str_val] = 1

    def _update_reservoir(
        self,
        stats: Dict,
        series: pd.Series,
        chunk_start: int
    ) -> None:
        """Update reservoir sample using Algorithm R."""
        for i, val in enumerate(series):
            global_idx = chunk_start + i
            stats["reservoir_index"] += 1

            if len(stats["reservoir"]) < self.RESERVOIR_SIZE:
                stats["reservoir"].append(val)
            else:
                # Random replacement
                j = random.randint(0, stats["reservoir_index"] - 1)
                if j < self.RESERVOIR_SIZE:
                    stats["reservoir"][j] = val

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Finalize and return field profiles.

        Converts internal statistics to the same format as build_field_profiles().
        """
        logger.info(
            "IncrementalFieldProfiler.finalize: %d fields, %d total records",
            len(self._field_stats), self._total_records
        )

        profiles = []
        for name, stats in self._field_stats.items():
            profile = self._build_profile_from_stats(name, stats)
            profiles.append(profile)

        return profiles

    def _build_profile_from_stats(
        self,
        name: str,
        stats: Dict
    ) -> Dict[str, Any]:
        """Convert accumulated statistics to a field profile."""
        profile: Dict[str, Any] = {
            "name": name,
            "total_rows": self._total_records,
            "null_count": stats["null_count"],
            "null_pct": round(
                stats["null_count"] / max(self._total_records, 1) * 100, 2
            ),
        }

        # Determine primary type
        type_counts = {
            "numeric": stats["numeric_count"],
            "string": stats["string_count"],
            "boolean": stats["bool_count"],
            "datetime": stats["datetime_count"],
        }
        primary_type = max(type_counts, key=type_counts.get)
        total_non_null = sum(type_counts.values())

        if total_non_null == 0:
            profile["detected_type"] = "empty"
            profile["detected_category"] = "auxiliary"
            return profile

        # Build type-specific profile
        if primary_type == "boolean":
            profile["detected_type"] = "boolean"
            profile["detected_category"] = "auxiliary"
            if stats["value_counts"]:
                true_count = stats["value_counts"].get("True", 0)
                total = sum(stats["value_counts"].values())
                profile["true_pct"] = round(true_count / max(total, 1) * 100, 2)

        elif primary_type == "datetime":
            profile["detected_type"] = "timestamp"
            profile["detected_category"] = "temporal"
            profile["min"] = stats["min"]
            profile["max"] = stats["max"]
            profile["sample_values"] = [
                str(v) for v in stats["reservoir"][:5]
            ]

        elif primary_type == "numeric":
            profile = self._build_numeric_profile(name, stats, profile)

        else:  # string
            profile = self._build_string_profile(name, stats, profile)

        return profile

    def _build_numeric_profile(
        self,
        name: str,
        stats: Dict,
        profile: Dict
    ) -> Dict:
        """Build numeric field profile from accumulated stats."""
        profile["detected_type"] = "numeric"

        if stats["count"] == 0:
            profile["detected_category"] = "auxiliary"
            return profile

        # Core statistics
        profile["min"] = _safe(stats["min"])
        profile["max"] = _safe(stats["max"])
        profile["mean"] = _safe(stats["mean"])

        # Variance and std from Welford's M2
        if stats["count"] > 1:
            variance = stats["M2"] / (stats["count"] - 1)
            profile["std"] = _safe(float(np.sqrt(variance)))
        else:
            profile["std"] = 0.0

        # Unique count
        if stats["unique_overflow"]:
            profile["unique_count"] = f">{stats['unique_limit']}"
        else:
            profile["unique_count"] = len(stats["unique_values"])

        profile["is_integer"] = stats["all_integers"]
        profile["is_monotonic_increasing"] = stats["is_monotonic_inc"]
        profile["is_monotonic_decreasing"] = stats["is_monotonic_dec"]

        # Constant detection
        if profile.get("std", 0) == 0:
            profile["is_constant"] = True

        # Timestamp heuristics
        if (stats["all_integers"]
                and stats["min"] is not None
                and stats["min"] > 1_000_000_000
                and stats["max"] < 2_000_000_000
                and stats["is_monotonic_inc"]):
            profile["likely_unix_timestamp"] = True

        if (stats["all_integers"]
                and stats["min"] is not None
                and stats["min"] > 1_000_000_000_000
                and stats["max"] < 2_000_000_000_000
                and stats["is_monotonic_inc"]):
            profile["likely_unix_timestamp_ms"] = True

        # Sample values from reservoir
        profile["sample_values"] = [
            _safe(float(v)) for v in stats["reservoir"][:8]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))
        ]

        # Category guess
        profile["detected_category"] = _guess_numeric_category(name, profile)

        return profile

    def _build_string_profile(
        self,
        name: str,
        stats: Dict,
        profile: Dict
    ) -> Dict:
        """Build string field profile from accumulated stats."""
        # Check for timestamp strings first
        if (stats["timestamp_pattern_checks"] > 0
                and stats["timestamp_pattern_matches"] / stats["timestamp_pattern_checks"] > 0.7):
            profile["detected_type"] = "timestamp_string"
            profile["detected_category"] = "temporal"
            profile["sample_values"] = [str(v) for v in stats["reservoir"][:8]]
            return profile

        # Unique count
        if stats["unique_overflow"]:
            profile["unique_count"] = f">{stats['unique_limit']}"
            unique_count = stats["unique_limit"]
        else:
            profile["unique_count"] = len(stats["unique_values"])
            unique_count = len(stats["unique_values"])

        # Top values
        sorted_counts = sorted(
            stats["value_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        profile["top_values"] = {k: v for k, v in sorted_counts}
        profile["sample_values"] = [k for k, v in sorted_counts[:8]]

        # String length stats
        profile["avg_length"] = _safe(stats["str_len_mean"])
        profile["max_length"] = stats["str_len_max"]

        # Type classification
        total_non_null = stats["string_count"]
        cardinality_ratio = unique_count / max(total_non_null, 1)

        if cardinality_ratio < 0.05 or unique_count <= 20:
            profile["detected_type"] = "categorical"
            profile["detected_category"] = _guess_string_category(name, profile)
        elif cardinality_ratio > 0.9:
            profile["detected_type"] = "identifier_string"
            profile["detected_category"] = "identifier"
        elif stats["str_len_mean"] > 50:
            profile["detected_type"] = "text"
            profile["detected_category"] = "auxiliary"
        else:
            profile["detected_type"] = "string"
            profile["detected_category"] = _guess_string_category(name, profile)

        return profile

    def get_progress(self) -> Dict[str, Any]:
        """Get current profiling progress."""
        return {
            "total_records": self._total_records,
            "chunk_count": self._chunk_count,
            "field_count": len(self._field_stats),
        }


async def build_field_profiles_streaming(
    chunk_iterator,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    total_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build field profiles from a streaming chunk iterator.

    Args:
        chunk_iterator: Async iterator yielding pandas DataFrames
        progress_callback: Optional callback(processed_records, total_records)
        total_records: Optional total record count for progress reporting

    Returns:
        List of field profiles (same format as build_field_profiles)
    """
    profiler = IncrementalFieldProfiler()
    processed = 0

    async for chunk in chunk_iterator:
        profiler.update(chunk)
        processed += len(chunk)

        if progress_callback and total_records:
            progress_callback(processed, total_records)

    return profiler.finalize()


# ═══════════════════════════════════════════════════════════════════════════
# Original In-Memory Profiler — For Small Datasets
# ═══════════════════════════════════════════════════════════════════════════


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
