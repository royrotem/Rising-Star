"""
Complex Type Detector — Advanced Data Type Recognition

Detects non-trivial data types that require special handling:
- Geospatial data (coordinates, GeoJSON, WKT)
- Time series patterns (regular intervals, seasonal)
- Nested/hierarchical structures (JSON objects, arrays)
- State machines (categorical sequences, transitions)
- Signal data (waveforms, sensor readings)
- Encoded data (base64, hex, binary patterns)

These detections inform the AI agents about special analysis requirements.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger("uaie.complex_types")


# ═══════════════════════════════════════════════════════════════════════════
# Complex Type Definitions
# ═══════════════════════════════════════════════════════════════════════════


class ComplexType(Enum):
    """Types of complex/non-trivial data patterns."""

    # Geospatial
    COORDINATES_LAT_LON = "coordinates_lat_lon"
    GEOJSON = "geojson"
    WKT_GEOMETRY = "wkt_geometry"
    UTM_COORDINATES = "utm_coordinates"

    # Time Series
    REGULAR_TIME_SERIES = "regular_time_series"
    IRREGULAR_TIME_SERIES = "irregular_time_series"
    SEASONAL_PATTERN = "seasonal_pattern"
    EVENT_SEQUENCE = "event_sequence"

    # Nested Structures
    JSON_OBJECT = "json_object"
    JSON_ARRAY = "json_array"
    NESTED_HIERARCHY = "nested_hierarchy"
    KEY_VALUE_PAIRS = "key_value_pairs"

    # State/Categorical
    STATE_MACHINE = "state_machine"
    CATEGORICAL_SEQUENCE = "categorical_sequence"
    ENUM_WITH_TRANSITIONS = "enum_with_transitions"
    ERROR_CODES = "error_codes"

    # Signal/Waveform
    SIGNAL_WAVEFORM = "signal_waveform"
    FFT_SPECTRUM = "fft_spectrum"
    VIBRATION_DATA = "vibration_data"
    AUDIO_SAMPLES = "audio_samples"

    # Encoded
    BASE64_ENCODED = "base64_encoded"
    HEX_ENCODED = "hex_encoded"
    BINARY_BLOB = "binary_blob"
    COMPRESSED_DATA = "compressed_data"

    # Identifiers
    UUID_PATTERN = "uuid_pattern"
    HASH_PATTERN = "hash_pattern"
    SERIAL_NUMBER = "serial_number"
    HIERARCHICAL_ID = "hierarchical_id"


@dataclass
class ComplexTypeDetection:
    """Result of complex type detection for a field."""
    field_name: str
    detected_type: ComplexType
    confidence: float  # 0.0 - 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_hints: List[str] = field(default_factory=list)


@dataclass
class FieldPairRelation:
    """Detected relationship between two fields."""
    field_a: str
    field_b: str
    relation_type: str  # e.g., "latitude_longitude", "timestamp_value", "id_reference"
    confidence: float
    description: str


# ═══════════════════════════════════════════════════════════════════════════
# Detection Patterns
# ═══════════════════════════════════════════════════════════════════════════


# Geospatial patterns
GEOJSON_PATTERN = re.compile(r'^\s*\{\s*"type"\s*:\s*"(Point|LineString|Polygon|MultiPoint|MultiLineString|MultiPolygon|GeometryCollection|Feature|FeatureCollection)"', re.IGNORECASE)
WKT_PATTERN = re.compile(r'^\s*(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*\(', re.IGNORECASE)

# Encoded data patterns
BASE64_PATTERN = re.compile(r'^[A-Za-z0-9+/]{20,}={0,2}$')
HEX_PATTERN = re.compile(r'^(0x)?[0-9A-Fa-f]{16,}$')
UUID_PATTERN = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
HASH_PATTERNS = {
    'md5': re.compile(r'^[0-9a-fA-F]{32}$'),
    'sha1': re.compile(r'^[0-9a-fA-F]{40}$'),
    'sha256': re.compile(r'^[0-9a-fA-F]{64}$'),
}

# Common field name patterns
LAT_NAMES = {'lat', 'latitude', 'y', 'lat_deg', 'latitude_deg', 'gps_lat', 'position_lat'}
LON_NAMES = {'lon', 'lng', 'long', 'longitude', 'x', 'lon_deg', 'longitude_deg', 'gps_lon', 'position_lon'}
TIMESTAMP_NAMES = {'time', 'timestamp', 'ts', 'datetime', 'date', 'created_at', 'updated_at', 'epoch', 't'}
STATE_NAMES = {'state', 'status', 'mode', 'phase', 'stage', 'condition', 'level'}
ERROR_NAMES = {'error', 'error_code', 'fault', 'fault_code', 'alarm', 'warning', 'dtc'}


# ═══════════════════════════════════════════════════════════════════════════
# Complex Type Detector
# ═══════════════════════════════════════════════════════════════════════════


class ComplexTypeDetector:
    """
    Detects complex/non-trivial data types in datasets.

    Used during ingestion to identify special data patterns that require
    specific handling or analysis approaches.
    """

    def __init__(self, sample_size: int = 1000):
        """
        Initialize detector.

        Args:
            sample_size: Number of samples to analyze per field
        """
        self.sample_size = sample_size

    def detect_complex_types(
        self,
        df: pd.DataFrame,
        field_profiles: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Detect complex types in a DataFrame.

        Args:
            df: DataFrame to analyze
            field_profiles: Optional pre-computed field profiles

        Returns:
            Dictionary with detections, relations, and recommendations
        """
        detections: List[ComplexTypeDetection] = []
        field_pairs: List[FieldPairRelation] = []

        # Sample data if too large
        if len(df) > self.sample_size:
            sample_df = df.sample(n=self.sample_size, random_state=42)
        else:
            sample_df = df

        # Detect types for each field
        for col in sample_df.columns:
            series = sample_df[col]
            field_detections = self._detect_field_types(col, series)
            detections.extend(field_detections)

        # Detect field pair relationships
        field_pairs = self._detect_field_relations(sample_df, detections)

        # Generate analysis recommendations
        recommendations = self._generate_recommendations(detections, field_pairs)

        return {
            "complex_types_detected": len(detections) > 0,
            "detections": [self._detection_to_dict(d) for d in detections],
            "field_relations": [self._relation_to_dict(r) for r in field_pairs],
            "recommendations": recommendations,
            "summary": self._generate_summary(detections, field_pairs),
        }

    def _detect_field_types(
        self,
        field_name: str,
        series: pd.Series,
    ) -> List[ComplexTypeDetection]:
        """Detect complex types for a single field."""
        detections = []

        # Skip empty series
        non_null = series.dropna()
        if len(non_null) == 0:
            return detections

        field_lower = field_name.lower()

        # Check for geospatial
        geo_detection = self._check_geospatial(field_name, field_lower, non_null)
        if geo_detection:
            detections.append(geo_detection)

        # Check for nested structures
        nested_detection = self._check_nested_structure(field_name, non_null)
        if nested_detection:
            detections.append(nested_detection)

        # Check for encoded data
        encoded_detection = self._check_encoded_data(field_name, non_null)
        if encoded_detection:
            detections.append(encoded_detection)

        # Check for state machine / categorical sequence
        state_detection = self._check_state_pattern(field_name, field_lower, non_null)
        if state_detection:
            detections.append(state_detection)

        # Check for signal/waveform data
        signal_detection = self._check_signal_data(field_name, field_lower, non_null)
        if signal_detection:
            detections.append(signal_detection)

        # Check for special identifiers
        id_detection = self._check_identifier_patterns(field_name, non_null)
        if id_detection:
            detections.append(id_detection)

        return detections

    def _check_geospatial(
        self,
        field_name: str,
        field_lower: str,
        series: pd.Series,
    ) -> Optional[ComplexTypeDetection]:
        """Check for geospatial data patterns."""

        # Check field name for lat/lon hints
        if field_lower in LAT_NAMES or any(n in field_lower for n in ['lat', '_y']):
            if pd.api.types.is_numeric_dtype(series):
                vals = series.to_numpy(dtype=float, na_value=np.nan)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    min_val, max_val = np.min(vals), np.max(vals)
                    if -90 <= min_val and max_val <= 90:
                        return ComplexTypeDetection(
                            field_name=field_name,
                            detected_type=ComplexType.COORDINATES_LAT_LON,
                            confidence=0.85,
                            evidence=[
                                f"Field name suggests latitude: {field_name}",
                                f"Values in valid latitude range: [{min_val:.4f}, {max_val:.4f}]",
                            ],
                            metadata={"coordinate_type": "latitude", "range": [min_val, max_val]},
                            analysis_hints=["Look for corresponding longitude field", "Consider geospatial clustering analysis"],
                        )

        if field_lower in LON_NAMES or any(n in field_lower for n in ['lon', 'lng', '_x']):
            if pd.api.types.is_numeric_dtype(series):
                vals = series.to_numpy(dtype=float, na_value=np.nan)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    min_val, max_val = np.min(vals), np.max(vals)
                    if -180 <= min_val and max_val <= 180:
                        return ComplexTypeDetection(
                            field_name=field_name,
                            detected_type=ComplexType.COORDINATES_LAT_LON,
                            confidence=0.85,
                            evidence=[
                                f"Field name suggests longitude: {field_name}",
                                f"Values in valid longitude range: [{min_val:.4f}, {max_val:.4f}]",
                            ],
                            metadata={"coordinate_type": "longitude", "range": [min_val, max_val]},
                            analysis_hints=["Look for corresponding latitude field", "Consider geospatial trajectory analysis"],
                        )

        # Check for GeoJSON
        if series.dtype == object:
            sample = series.head(10).astype(str)
            geojson_matches = sum(1 for v in sample if GEOJSON_PATTERN.match(str(v)))
            if geojson_matches > len(sample) * 0.5:
                return ComplexTypeDetection(
                    field_name=field_name,
                    detected_type=ComplexType.GEOJSON,
                    confidence=0.9,
                    evidence=[f"{geojson_matches}/{len(sample)} samples match GeoJSON pattern"],
                    metadata={"format": "geojson"},
                    analysis_hints=["Parse GeoJSON for spatial analysis", "Consider map visualization"],
                )

        # Check for WKT
        if series.dtype == object:
            sample = series.head(10).astype(str)
            wkt_matches = sum(1 for v in sample if WKT_PATTERN.match(str(v)))
            if wkt_matches > len(sample) * 0.5:
                return ComplexTypeDetection(
                    field_name=field_name,
                    detected_type=ComplexType.WKT_GEOMETRY,
                    confidence=0.9,
                    evidence=[f"{wkt_matches}/{len(sample)} samples match WKT pattern"],
                    metadata={"format": "wkt"},
                    analysis_hints=["Parse WKT for spatial analysis", "Convert to GeoJSON for visualization"],
                )

        return None

    def _check_nested_structure(
        self,
        field_name: str,
        series: pd.Series,
    ) -> Optional[ComplexTypeDetection]:
        """Check for nested/hierarchical data structures."""

        if series.dtype != object:
            return None

        sample = series.dropna().head(20)
        if len(sample) == 0:
            return None

        json_object_count = 0
        json_array_count = 0

        for val in sample:
            str_val = str(val).strip()
            if str_val.startswith('{') and str_val.endswith('}'):
                try:
                    parsed = json.loads(str_val)
                    if isinstance(parsed, dict):
                        json_object_count += 1
                except json.JSONDecodeError:
                    pass
            elif str_val.startswith('[') and str_val.endswith(']'):
                try:
                    parsed = json.loads(str_val)
                    if isinstance(parsed, list):
                        json_array_count += 1
                except json.JSONDecodeError:
                    pass

        total = len(sample)
        if json_object_count > total * 0.5:
            return ComplexTypeDetection(
                field_name=field_name,
                detected_type=ComplexType.JSON_OBJECT,
                confidence=json_object_count / total,
                evidence=[f"{json_object_count}/{total} samples are JSON objects"],
                metadata={"nested_type": "object"},
                analysis_hints=["Flatten nested fields for analysis", "Extract key metrics from objects"],
            )

        if json_array_count > total * 0.5:
            return ComplexTypeDetection(
                field_name=field_name,
                detected_type=ComplexType.JSON_ARRAY,
                confidence=json_array_count / total,
                evidence=[f"{json_array_count}/{total} samples are JSON arrays"],
                metadata={"nested_type": "array"},
                analysis_hints=["Analyze array lengths", "Consider exploding arrays into rows"],
            )

        return None

    def _check_encoded_data(
        self,
        field_name: str,
        series: pd.Series,
    ) -> Optional[ComplexTypeDetection]:
        """Check for encoded data patterns."""

        if series.dtype != object:
            return None

        sample = series.dropna().head(50).astype(str)
        if len(sample) == 0:
            return None

        # Check Base64
        base64_matches = sum(1 for v in sample if BASE64_PATTERN.match(v) and len(v) > 30)
        if base64_matches > len(sample) * 0.7:
            return ComplexTypeDetection(
                field_name=field_name,
                detected_type=ComplexType.BASE64_ENCODED,
                confidence=base64_matches / len(sample),
                evidence=[f"{base64_matches}/{len(sample)} samples match Base64 pattern"],
                metadata={"encoding": "base64"},
                analysis_hints=["Decode Base64 to analyze content", "Check if binary or text data"],
            )

        # Check Hex
        hex_matches = sum(1 for v in sample if HEX_PATTERN.match(v))
        if hex_matches > len(sample) * 0.7:
            return ComplexTypeDetection(
                field_name=field_name,
                detected_type=ComplexType.HEX_ENCODED,
                confidence=hex_matches / len(sample),
                evidence=[f"{hex_matches}/{len(sample)} samples match hex pattern"],
                metadata={"encoding": "hex"},
                analysis_hints=["Decode hex for binary analysis", "May contain sensor readings or protocol data"],
            )

        return None

    def _check_state_pattern(
        self,
        field_name: str,
        field_lower: str,
        series: pd.Series,
    ) -> Optional[ComplexTypeDetection]:
        """Check for state machine / categorical sequence patterns."""

        # Check if field name suggests state
        is_state_field = field_lower in STATE_NAMES or any(n in field_lower for n in STATE_NAMES)
        is_error_field = field_lower in ERROR_NAMES or any(n in field_lower for n in ERROR_NAMES)

        if not (is_state_field or is_error_field):
            return None

        unique_values = series.nunique()
        total_values = len(series)

        # State machines typically have limited unique values
        if unique_values < 2 or unique_values > 50:
            return None

        cardinality_ratio = unique_values / total_values
        if cardinality_ratio > 0.1:  # Too many unique values
            return None

        # Analyze transitions
        if len(series) > 1:
            transitions = set()
            prev = None
            for val in series:
                if prev is not None and val != prev:
                    transitions.add((str(prev), str(val)))
                prev = val

            if is_error_field:
                return ComplexTypeDetection(
                    field_name=field_name,
                    detected_type=ComplexType.ERROR_CODES,
                    confidence=0.8,
                    evidence=[
                        f"Field name suggests error/fault codes",
                        f"{unique_values} unique values, {len(transitions)} transitions observed",
                    ],
                    metadata={
                        "unique_values": unique_values,
                        "transition_count": len(transitions),
                        "sample_values": list(series.value_counts().head(10).index),
                    },
                    analysis_hints=[
                        "Analyze error frequency and patterns",
                        "Look for error sequences and root causes",
                        "Consider time-to-failure analysis",
                    ],
                )
            else:
                return ComplexTypeDetection(
                    field_name=field_name,
                    detected_type=ComplexType.STATE_MACHINE,
                    confidence=0.75,
                    evidence=[
                        f"Field name suggests state/mode tracking",
                        f"{unique_values} states, {len(transitions)} transitions observed",
                    ],
                    metadata={
                        "state_count": unique_values,
                        "transition_count": len(transitions),
                        "states": list(series.value_counts().head(10).index),
                    },
                    analysis_hints=[
                        "Build state transition diagram",
                        "Analyze dwell time in each state",
                        "Look for unexpected state transitions",
                    ],
                )

        return None

    def _check_signal_data(
        self,
        field_name: str,
        field_lower: str,
        series: pd.Series,
    ) -> Optional[ComplexTypeDetection]:
        """Check for signal/waveform data patterns."""

        if not pd.api.types.is_numeric_dtype(series):
            return None

        # Signal-related field name hints
        signal_hints = {'signal', 'wave', 'vibration', 'accel', 'velocity', 'amplitude', 'freq', 'fft', 'spectrum'}
        is_signal_field = any(h in field_lower for h in signal_hints)

        vals = series.to_numpy(dtype=float, na_value=np.nan)
        vals = vals[~np.isnan(vals)]

        if len(vals) < 100:
            return None

        # Check for high-frequency oscillation (many zero crossings)
        mean_val = np.mean(vals)
        centered = vals - mean_val
        zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
        crossing_rate = zero_crossings / len(vals)

        # High crossing rate suggests oscillating signal
        if crossing_rate > 0.3 or is_signal_field:
            # Calculate additional signal characteristics
            std_val = np.std(vals)
            if std_val > 0:
                return ComplexTypeDetection(
                    field_name=field_name,
                    detected_type=ComplexType.SIGNAL_WAVEFORM,
                    confidence=0.7 if is_signal_field else 0.6,
                    evidence=[
                        f"Zero crossing rate: {crossing_rate:.2%}",
                        f"Signal characteristics: mean={mean_val:.4f}, std={std_val:.4f}",
                    ],
                    metadata={
                        "zero_crossing_rate": crossing_rate,
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    },
                    analysis_hints=[
                        "Consider FFT analysis for frequency components",
                        "Look for anomalous amplitude patterns",
                        "Check sampling rate for Nyquist compliance",
                    ],
                )

        return None

    def _check_identifier_patterns(
        self,
        field_name: str,
        series: pd.Series,
    ) -> Optional[ComplexTypeDetection]:
        """Check for special identifier patterns (UUID, hash, etc.)."""

        if series.dtype != object:
            return None

        sample = series.dropna().head(50).astype(str)
        if len(sample) == 0:
            return None

        # Check UUID
        uuid_matches = sum(1 for v in sample if UUID_PATTERN.match(v))
        if uuid_matches > len(sample) * 0.8:
            return ComplexTypeDetection(
                field_name=field_name,
                detected_type=ComplexType.UUID_PATTERN,
                confidence=uuid_matches / len(sample),
                evidence=[f"{uuid_matches}/{len(sample)} samples are valid UUIDs"],
                metadata={"id_type": "uuid"},
                analysis_hints=["Use for record linkage", "May be generated sequentially or randomly"],
            )

        # Check hashes
        for hash_type, pattern in HASH_PATTERNS.items():
            hash_matches = sum(1 for v in sample if pattern.match(v))
            if hash_matches > len(sample) * 0.8:
                return ComplexTypeDetection(
                    field_name=field_name,
                    detected_type=ComplexType.HASH_PATTERN,
                    confidence=hash_matches / len(sample),
                    evidence=[f"{hash_matches}/{len(sample)} samples match {hash_type.upper()} pattern"],
                    metadata={"hash_type": hash_type},
                    analysis_hints=["Likely content hash or identifier", "Check for duplicates"],
                )

        return None

    def _detect_field_relations(
        self,
        df: pd.DataFrame,
        detections: List[ComplexTypeDetection],
    ) -> List[FieldPairRelation]:
        """Detect relationships between fields."""
        relations = []

        # Build detection map
        detection_map = {d.field_name: d for d in detections}

        # Find lat/lon pairs
        lat_fields = [d.field_name for d in detections
                     if d.detected_type == ComplexType.COORDINATES_LAT_LON
                     and d.metadata.get("coordinate_type") == "latitude"]
        lon_fields = [d.field_name for d in detections
                     if d.detected_type == ComplexType.COORDINATES_LAT_LON
                     and d.metadata.get("coordinate_type") == "longitude"]

        for lat in lat_fields:
            for lon in lon_fields:
                relations.append(FieldPairRelation(
                    field_a=lat,
                    field_b=lon,
                    relation_type="latitude_longitude",
                    confidence=0.9,
                    description=f"Geographic coordinate pair: {lat} (lat) + {lon} (lon)",
                ))

        # Find timestamp + value pairs (for time series)
        timestamp_fields = [col for col in df.columns
                          if col.lower() in TIMESTAMP_NAMES or 'time' in col.lower()]
        numeric_fields = [col for col in df.columns
                        if pd.api.types.is_numeric_dtype(df[col]) and col not in timestamp_fields]

        for ts_field in timestamp_fields[:1]:  # Use first timestamp field
            for num_field in numeric_fields[:5]:  # Limit to 5 numeric fields
                relations.append(FieldPairRelation(
                    field_a=ts_field,
                    field_b=num_field,
                    relation_type="timestamp_value",
                    confidence=0.7,
                    description=f"Time series: {num_field} over {ts_field}",
                ))

        return relations

    def _generate_recommendations(
        self,
        detections: List[ComplexTypeDetection],
        relations: List[FieldPairRelation],
    ) -> List[Dict[str, Any]]:
        """Generate analysis recommendations based on detections."""
        recommendations = []

        # Group detections by type
        type_counts = {}
        for d in detections:
            t = d.detected_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        # Geospatial recommendations
        if ComplexType.COORDINATES_LAT_LON.value in type_counts:
            recommendations.append({
                "category": "geospatial",
                "priority": "high",
                "title": "Geospatial Analysis Available",
                "description": "Dataset contains geographic coordinates. Consider spatial clustering, trajectory analysis, or map visualization.",
                "suggested_analyses": ["spatial_clustering", "trajectory_analysis", "geofencing"],
            })

        # Time series recommendations
        ts_relations = [r for r in relations if r.relation_type == "timestamp_value"]
        if ts_relations:
            recommendations.append({
                "category": "time_series",
                "priority": "high",
                "title": "Time Series Data Detected",
                "description": f"Found {len(ts_relations)} time series relationships. Consider trend analysis and anomaly detection.",
                "suggested_analyses": ["trend_analysis", "seasonality_detection", "anomaly_detection"],
            })

        # State machine recommendations
        if ComplexType.STATE_MACHINE.value in type_counts or ComplexType.ERROR_CODES.value in type_counts:
            recommendations.append({
                "category": "state_analysis",
                "priority": "medium",
                "title": "State/Mode Tracking Detected",
                "description": "Dataset contains state machine or error code fields. Consider transition analysis.",
                "suggested_analyses": ["state_transition_analysis", "dwell_time_analysis", "error_pattern_detection"],
            })

        # Signal recommendations
        if ComplexType.SIGNAL_WAVEFORM.value in type_counts:
            recommendations.append({
                "category": "signal_processing",
                "priority": "medium",
                "title": "Signal Data Detected",
                "description": "Dataset contains waveform/signal data. Consider frequency analysis.",
                "suggested_analyses": ["fft_analysis", "envelope_detection", "vibration_analysis"],
            })

        # Nested structure recommendations
        if ComplexType.JSON_OBJECT.value in type_counts or ComplexType.JSON_ARRAY.value in type_counts:
            recommendations.append({
                "category": "data_transformation",
                "priority": "low",
                "title": "Nested Data Structures Found",
                "description": "Some fields contain nested JSON. Consider flattening for analysis.",
                "suggested_analyses": ["schema_extraction", "nested_field_analysis"],
            })

        return recommendations

    def _generate_summary(
        self,
        detections: List[ComplexTypeDetection],
        relations: List[FieldPairRelation],
    ) -> Dict[str, Any]:
        """Generate a summary of all detections."""
        type_counts = {}
        for d in detections:
            t = d.detected_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_complex_fields": len(detections),
            "total_relations": len(relations),
            "type_breakdown": type_counts,
            "has_geospatial": any(
                d.detected_type in [ComplexType.COORDINATES_LAT_LON, ComplexType.GEOJSON, ComplexType.WKT_GEOMETRY]
                for d in detections
            ),
            "has_time_series": any(r.relation_type == "timestamp_value" for r in relations),
            "has_state_machine": any(
                d.detected_type in [ComplexType.STATE_MACHINE, ComplexType.ERROR_CODES]
                for d in detections
            ),
            "has_signal_data": any(d.detected_type == ComplexType.SIGNAL_WAVEFORM for d in detections),
            "has_nested_data": any(
                d.detected_type in [ComplexType.JSON_OBJECT, ComplexType.JSON_ARRAY]
                for d in detections
            ),
        }

    def _detection_to_dict(self, detection: ComplexTypeDetection) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            "field_name": detection.field_name,
            "detected_type": detection.detected_type.value,
            "confidence": detection.confidence,
            "evidence": detection.evidence,
            "metadata": detection.metadata,
            "analysis_hints": detection.analysis_hints,
        }

    def _relation_to_dict(self, relation: FieldPairRelation) -> Dict[str, Any]:
        """Convert relation to dictionary."""
        return {
            "field_a": relation.field_a,
            "field_b": relation.field_b,
            "relation_type": relation.relation_type,
            "confidence": relation.confidence,
            "description": relation.description,
        }


# Global instance
complex_type_detector = ComplexTypeDetector()
