"""
Advanced Analysis Engine for UAIE

Multi-layered anomaly detection and AI-powered analysis with
natural language explanations and root cause analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class AnomalyType(Enum):
    STATISTICAL_OUTLIER = "statistical_outlier"
    THRESHOLD_BREACH = "threshold_breach"
    TREND_CHANGE = "trend_change"
    CORRELATION_BREAK = "correlation_break"
    PATTERN_ANOMALY = "pattern_anomaly"
    DISTRIBUTION_SHIFT = "distribution_shift"
    RATE_OF_CHANGE = "rate_of_change"
    PREDICTIVE_WARNING = "predictive_warning"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Anomaly:
    id: str
    anomaly_type: AnomalyType
    severity: Severity
    field_name: str
    title: str
    description: str
    value: Any
    expected_range: Tuple[float, float] = None
    confidence: float = 0.0
    timestamp: str = None
    related_fields: List[str] = field(default_factory=list)
    natural_language_explanation: str = ""
    possible_causes: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, str]] = field(default_factory=list)
    impact_score: float = 0.0


@dataclass
class AnalysisResult:
    system_id: str
    system_type: str
    health_score: float
    anomalies: List[Anomaly]
    insights: List[str]
    engineering_margins: List[Dict]
    blind_spots: List[Dict]
    correlation_matrix: Dict[str, Dict[str, float]]
    trend_analysis: Dict[str, Dict]
    summary: str
    recommendations: List[Dict]
    analyzed_at: str


class AnalysisEngine:
    """
    Multi-layered analysis engine that combines statistical methods,
    pattern recognition, and domain-specific knowledge for comprehensive
    anomaly detection and explanation.
    """

    def __init__(self):
        # Domain knowledge for different system types
        self.domain_knowledge = self._load_domain_knowledge()

        # Thresholds for different severity levels
        self.severity_thresholds = {
            "z_score": {"critical": 4.0, "high": 3.0, "medium": 2.5, "low": 2.0},
            "margin_percentage": {"critical": 5, "high": 10, "medium": 20, "low": 30},
        }

    def _load_domain_knowledge(self) -> Dict[str, Dict]:
        """Load domain-specific knowledge for different system types."""
        return {
            "industrial": {
                "critical_parameters": ["vibration", "temperature", "pressure", "current", "acoustic"],
                "normal_ranges": {
                    "temperature": (20, 85),  # Celsius
                    "vibration": (0, 10),     # mm/s RMS
                    "pressure": (0, 150),     # PSI
                    "current": (0, 50),       # Amps
                },
                "correlations": {
                    ("temperature", "current"): "positive",  # Higher current = more heat
                    ("vibration", "acoustic"): "positive",   # Vibration causes noise
                    ("temperature", "vibration"): "positive", # Heat can indicate friction
                },
                "failure_modes": {
                    "high_vibration": [
                        "Bearing wear or damage",
                        "Shaft misalignment",
                        "Imbalance in rotating components",
                        "Loose mounting bolts",
                        "Worn gears or belts",
                    ],
                    "high_temperature": [
                        "Insufficient lubrication",
                        "Overloading",
                        "Cooling system failure",
                        "Friction from worn components",
                        "Electrical issues causing excess heat",
                    ],
                    "abnormal_current": [
                        "Motor winding issues",
                        "Power supply problems",
                        "Mechanical binding or obstruction",
                        "Phase imbalance",
                        "Insulation breakdown",
                    ],
                    "high_acoustic": [
                        "Cavitation in pumps",
                        "Bearing defects",
                        "Gear mesh problems",
                        "Loose components",
                        "Flow turbulence",
                    ],
                },
                "maintenance_recommendations": {
                    "vibration": "Schedule vibration analysis and bearing inspection",
                    "temperature": "Check lubrication levels and cooling system",
                    "current": "Inspect motor windings and electrical connections",
                    "acoustic": "Perform acoustic emission testing",
                },
            },
            "vehicle": {
                "critical_parameters": ["speed", "rpm", "temperature", "battery", "fuel", "brake"],
                "normal_ranges": {
                    "engine_temp": (70, 105),
                    "battery_voltage": (12.4, 14.7),
                    "oil_pressure": (25, 65),
                    "rpm": (600, 6500),
                },
                "failure_modes": {
                    "high_engine_temp": [
                        "Coolant leak or low level",
                        "Thermostat failure",
                        "Water pump failure",
                        "Radiator blockage",
                        "Head gasket issues",
                    ],
                    "low_battery": [
                        "Alternator failure",
                        "Battery degradation",
                        "Parasitic drain",
                        "Corroded connections",
                    ],
                },
            },
            "robot": {
                "critical_parameters": ["joint", "torque", "position", "velocity", "temperature"],
                "normal_ranges": {
                    "joint_torque": (0, 100),
                    "joint_temp": (20, 60),
                    "position_error": (0, 0.1),
                },
                "failure_modes": {
                    "high_torque": [
                        "Mechanical obstruction",
                        "Joint wear",
                        "Payload exceeds capacity",
                        "Lubrication issues",
                    ],
                    "position_error": [
                        "Encoder failure",
                        "Calibration drift",
                        "Mechanical backlash",
                        "Control system issues",
                    ],
                },
            },
            "medical_device": {
                "critical_parameters": ["heart_rate", "blood_pressure", "oxygen", "temperature"],
                "normal_ranges": {
                    "heart_rate": (60, 100),
                    "spo2": (95, 100),
                    "body_temp": (36.1, 37.2),
                },
                "failure_modes": {
                    "abnormal_reading": [
                        "Sensor malfunction",
                        "Patient movement artifact",
                        "Calibration drift",
                        "Environmental interference",
                    ],
                },
            },
            "aerospace": {
                "critical_parameters": ["altitude", "airspeed", "engine", "fuel", "hydraulic"],
                "normal_ranges": {
                    "engine_temp": (200, 900),
                    "oil_pressure": (40, 100),
                    "fuel_flow": (100, 5000),
                },
                "failure_modes": {
                    "engine_anomaly": [
                        "Compressor stall",
                        "Fuel system contamination",
                        "Turbine blade damage",
                        "Oil system degradation",
                    ],
                },
            },
        }

    async def analyze(
        self,
        system_id: str,
        system_type: str,
        records: List[Dict],
        discovered_schema: List[Dict] = None,
        metadata: Dict = None,
    ) -> AnalysisResult:
        """
        Perform comprehensive multi-layered analysis on the data.
        """
        if not records:
            return self._empty_result(system_id, system_type)

        df = pd.DataFrame(records)
        anomalies = []
        insights = []

        # Get domain knowledge for this system type
        domain = self.domain_knowledge.get(system_type, self.domain_knowledge.get("industrial", {}))

        # Layer 1: Statistical Anomaly Detection
        statistical_anomalies = self._detect_statistical_anomalies(df, domain)
        anomalies.extend(statistical_anomalies)

        # Layer 2: Threshold-Based Detection
        threshold_anomalies = self._detect_threshold_anomalies(df, domain)
        anomalies.extend(threshold_anomalies)

        # Layer 3: Trend Analysis
        trend_analysis, trend_anomalies = self._analyze_trends(df, domain)
        anomalies.extend(trend_anomalies)

        # Layer 4: Correlation Analysis
        correlation_matrix, correlation_anomalies = self._analyze_correlations(df, domain)
        anomalies.extend(correlation_anomalies)

        # Layer 5: Pattern Detection
        pattern_anomalies = self._detect_pattern_anomalies(df, domain)
        anomalies.extend(pattern_anomalies)

        # Layer 6: Rate of Change Analysis
        rate_anomalies = self._detect_rate_of_change_anomalies(df, domain)
        anomalies.extend(rate_anomalies)

        # Generate natural language explanations for all anomalies
        for anomaly in anomalies:
            self._enrich_anomaly_explanation(anomaly, system_type, domain, df)

        # Calculate engineering margins
        engineering_margins = self._calculate_engineering_margins(df, domain)

        # Identify blind spots
        blind_spots = self._identify_blind_spots(df, domain, discovered_schema)

        # Calculate health score
        health_score = self._calculate_health_score(anomalies, engineering_margins)

        # Generate insights
        insights = self._generate_insights(anomalies, trend_analysis, correlation_matrix, domain)

        # Generate overall summary
        summary = self._generate_summary(system_type, anomalies, health_score, len(records), domain)

        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies, engineering_margins, domain)

        return AnalysisResult(
            system_id=system_id,
            system_type=system_type,
            health_score=health_score,
            anomalies=anomalies,
            insights=insights,
            engineering_margins=engineering_margins,
            blind_spots=blind_spots,
            correlation_matrix=correlation_matrix,
            trend_analysis=trend_analysis,
            summary=summary,
            recommendations=recommendations,
            analyzed_at=datetime.utcnow().isoformat(),
        )

    def _detect_statistical_anomalies(self, df: pd.DataFrame, domain: Dict) -> List[Anomaly]:
        """Layer 1: Detect anomalies using statistical methods (Z-score, IQR)."""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isna().all():
                continue

            series = df[col].dropna()
            if len(series) < 10:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            # Z-score analysis
            z_scores = np.abs((series - mean) / std)

            for severity_name, threshold in self.severity_thresholds["z_score"].items():
                outlier_mask = z_scores > threshold
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    outlier_values = series[outlier_mask]
                    severity = Severity[severity_name.upper()]

                    anomaly = Anomaly(
                        id=f"stat_{col}_{severity_name}_{datetime.utcnow().timestamp()}",
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=severity,
                        field_name=col,
                        title=f"Statistical outliers detected in {col}",
                        description=f"Found {outlier_count} values with z-score > {threshold}",
                        value={"outlier_count": int(outlier_count), "max_z_score": float(z_scores.max())},
                        expected_range=(float(mean - 2*std), float(mean + 2*std)),
                        confidence=min(0.95, 0.7 + (threshold / 10)),
                        impact_score=min(100, outlier_count / len(series) * 100 * threshold),
                    )
                    anomalies.append(anomaly)
                    break  # Only report highest severity

        return anomalies

    def _detect_threshold_anomalies(self, df: pd.DataFrame, domain: Dict) -> List[Anomaly]:
        """Layer 2: Detect anomalies based on domain-specific thresholds."""
        anomalies = []
        normal_ranges = domain.get("normal_ranges", {})

        for col in df.columns:
            col_lower = col.lower()

            # Try to match column to known parameter
            for param, (min_val, max_val) in normal_ranges.items():
                if param in col_lower:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        continue

                    series = df[col].dropna()
                    if len(series) == 0:
                        continue

                    above_max = (series > max_val).sum()
                    below_min = (series < min_val).sum()

                    if above_max > 0:
                        max_value = series.max()
                        pct_over = ((max_value - max_val) / max_val) * 100
                        severity = self._get_severity_from_percentage(pct_over)

                        anomalies.append(Anomaly(
                            id=f"thresh_high_{col}_{datetime.utcnow().timestamp()}",
                            anomaly_type=AnomalyType.THRESHOLD_BREACH,
                            severity=severity,
                            field_name=col,
                            title=f"{col} exceeds safe operating range",
                            description=f"{above_max} readings above maximum threshold of {max_val}",
                            value={"max_value": float(max_value), "threshold": max_val, "count": int(above_max)},
                            expected_range=(min_val, max_val),
                            confidence=0.9,
                            impact_score=min(100, pct_over * 2),
                        ))

                    if below_min > 0:
                        min_value = series.min()
                        pct_under = ((min_val - min_value) / min_val) * 100 if min_val != 0 else 50
                        severity = self._get_severity_from_percentage(pct_under)

                        anomalies.append(Anomaly(
                            id=f"thresh_low_{col}_{datetime.utcnow().timestamp()}",
                            anomaly_type=AnomalyType.THRESHOLD_BREACH,
                            severity=severity,
                            field_name=col,
                            title=f"{col} below minimum operating range",
                            description=f"{below_min} readings below minimum threshold of {min_val}",
                            value={"min_value": float(min_value), "threshold": min_val, "count": int(below_min)},
                            expected_range=(min_val, max_val),
                            confidence=0.9,
                            impact_score=min(100, pct_under * 2),
                        ))
                    break

        return anomalies

    def _analyze_trends(self, df: pd.DataFrame, domain: Dict) -> Tuple[Dict, List[Anomaly]]:
        """Layer 3: Analyze trends and detect trend changes."""
        trend_analysis = {}
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 20:
                continue

            # Calculate rolling statistics
            window = min(50, len(series) // 4)
            if window < 5:
                continue

            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()

            # Detect trend direction
            first_half_mean = series[:len(series)//2].mean()
            second_half_mean = series[len(series)//2:].mean()

            if first_half_mean != 0:
                change_pct = ((second_half_mean - first_half_mean) / abs(first_half_mean)) * 100
            else:
                change_pct = 0

            trend_direction = "increasing" if change_pct > 5 else ("decreasing" if change_pct < -5 else "stable")

            # Detect volatility change
            first_half_std = series[:len(series)//2].std()
            second_half_std = series[len(series)//2:].std()
            volatility_change = "increasing" if second_half_std > first_half_std * 1.5 else (
                "decreasing" if second_half_std < first_half_std * 0.5 else "stable"
            )

            trend_analysis[col] = {
                "direction": trend_direction,
                "change_percentage": float(change_pct),
                "volatility": volatility_change,
                "current_mean": float(series.tail(window).mean()),
                "current_std": float(series.tail(window).std()),
            }

            # Generate anomalies for significant trends
            critical_params = domain.get("critical_parameters", [])
            is_critical = any(p in col.lower() for p in critical_params)

            if abs(change_pct) > 20 and is_critical:
                severity = Severity.HIGH if abs(change_pct) > 50 else Severity.MEDIUM

                anomalies.append(Anomaly(
                    id=f"trend_{col}_{datetime.utcnow().timestamp()}",
                    anomaly_type=AnomalyType.TREND_CHANGE,
                    severity=severity,
                    field_name=col,
                    title=f"Significant {trend_direction} trend in {col}",
                    description=f"{col} shows {abs(change_pct):.1f}% {trend_direction} trend over the data period",
                    value={"change_percentage": float(change_pct), "direction": trend_direction},
                    confidence=0.85,
                    impact_score=min(100, abs(change_pct)),
                ))

            if volatility_change == "increasing" and is_critical:
                anomalies.append(Anomaly(
                    id=f"volatility_{col}_{datetime.utcnow().timestamp()}",
                    anomaly_type=AnomalyType.PATTERN_ANOMALY,
                    severity=Severity.MEDIUM,
                    field_name=col,
                    title=f"Increasing instability in {col}",
                    description=f"{col} shows increasing volatility, indicating potential system instability",
                    value={"volatility_ratio": float(second_half_std / first_half_std) if first_half_std > 0 else 0},
                    confidence=0.8,
                    impact_score=50,
                ))

        return trend_analysis, anomalies

    def _analyze_correlations(self, df: pd.DataFrame, domain: Dict) -> Tuple[Dict, List[Anomaly]]:
        """Layer 4: Analyze correlations and detect correlation breaks."""
        anomalies = []
        correlation_matrix = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return correlation_matrix, anomalies

        # Calculate correlation matrix
        corr_df = df[numeric_cols].corr()
        correlation_matrix = corr_df.to_dict()

        # Check expected correlations from domain knowledge
        expected_correlations = domain.get("correlations", {})

        for (field_a, field_b), expected_direction in expected_correlations.items():
            # Find matching columns
            col_a = next((c for c in numeric_cols if field_a in c.lower()), None)
            col_b = next((c for c in numeric_cols if field_b in c.lower()), None)

            if col_a and col_b and col_a in corr_df.columns and col_b in corr_df.columns:
                actual_corr = corr_df.loc[col_a, col_b]

                expected_positive = expected_direction == "positive"
                actual_positive = actual_corr > 0.3
                actual_negative = actual_corr < -0.3

                # Check for correlation break
                if expected_positive and actual_negative:
                    anomalies.append(Anomaly(
                        id=f"corr_break_{col_a}_{col_b}_{datetime.utcnow().timestamp()}",
                        anomaly_type=AnomalyType.CORRELATION_BREAK,
                        severity=Severity.HIGH,
                        field_name=col_a,
                        title=f"Unexpected correlation between {col_a} and {col_b}",
                        description=f"Expected positive correlation, but found negative ({actual_corr:.2f}). This may indicate sensor issues or abnormal operating conditions.",
                        value={"actual_correlation": float(actual_corr), "expected": expected_direction},
                        related_fields=[col_b],
                        confidence=0.85,
                        impact_score=abs(actual_corr) * 70,
                    ))
                elif not expected_positive and actual_positive:
                    anomalies.append(Anomaly(
                        id=f"corr_break_{col_a}_{col_b}_{datetime.utcnow().timestamp()}",
                        anomaly_type=AnomalyType.CORRELATION_BREAK,
                        severity=Severity.MEDIUM,
                        field_name=col_a,
                        title=f"Unexpected correlation between {col_a} and {col_b}",
                        description=f"Expected negative correlation, but found positive ({actual_corr:.2f}).",
                        value={"actual_correlation": float(actual_corr), "expected": expected_direction},
                        related_fields=[col_b],
                        confidence=0.8,
                        impact_score=abs(actual_corr) * 50,
                    ))

        return correlation_matrix, anomalies

    def _detect_pattern_anomalies(self, df: pd.DataFrame, domain: Dict) -> List[Anomaly]:
        """Layer 5: Detect pattern-based anomalies (sudden changes, plateaus, etc.)."""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 30:
                continue

            # Detect sudden jumps
            diff = series.diff().abs()
            mean_diff = diff.mean()
            std_diff = diff.std()

            if std_diff > 0:
                jump_threshold = mean_diff + 4 * std_diff
                jumps = (diff > jump_threshold).sum()

                if jumps > 0 and jumps < len(series) * 0.01:  # Rare but present
                    critical_params = domain.get("critical_parameters", [])
                    is_critical = any(p in col.lower() for p in critical_params)

                    if is_critical:
                        anomalies.append(Anomaly(
                            id=f"jump_{col}_{datetime.utcnow().timestamp()}",
                            anomaly_type=AnomalyType.PATTERN_ANOMALY,
                            severity=Severity.MEDIUM,
                            field_name=col,
                            title=f"Sudden value changes detected in {col}",
                            description=f"Found {jumps} sudden jumps in {col} that exceed normal variation by 4 standard deviations",
                            value={"jump_count": int(jumps), "threshold": float(jump_threshold)},
                            confidence=0.75,
                            impact_score=min(100, jumps * 20),
                        ))

            # Detect stuck values (sensor failure indicator)
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.01 and len(series) > 100:  # Less than 1% unique values
                anomalies.append(Anomaly(
                    id=f"stuck_{col}_{datetime.utcnow().timestamp()}",
                    anomaly_type=AnomalyType.PATTERN_ANOMALY,
                    severity=Severity.HIGH,
                    field_name=col,
                    title=f"Possible sensor failure in {col}",
                    description=f"{col} shows very low variation ({unique_ratio*100:.2f}% unique values), which may indicate a stuck or failed sensor",
                    value={"unique_ratio": float(unique_ratio), "unique_count": int(series.nunique())},
                    confidence=0.9,
                    impact_score=80,
                ))

        return anomalies

    def _detect_rate_of_change_anomalies(self, df: pd.DataFrame, domain: Dict) -> List[Anomaly]:
        """Layer 6: Detect anomalies in rate of change."""
        anomalies = []
        critical_params = domain.get("critical_parameters", [])
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if not any(p in col.lower() for p in critical_params):
                continue

            series = df[col].dropna()
            if len(series) < 50:
                continue

            # Calculate rate of change
            rate = series.diff()
            rate_mean = rate.mean()
            rate_std = rate.std()

            if rate_std == 0:
                continue

            # Detect accelerating changes
            second_derivative = rate.diff()
            accel_outliers = (second_derivative.abs() > 3 * second_derivative.std()).sum()

            if accel_outliers > len(series) * 0.02:  # More than 2% accelerating changes
                anomalies.append(Anomaly(
                    id=f"accel_{col}_{datetime.utcnow().timestamp()}",
                    anomaly_type=AnomalyType.RATE_OF_CHANGE,
                    severity=Severity.MEDIUM,
                    field_name=col,
                    title=f"Unusual rate of change in {col}",
                    description=f"{col} shows {accel_outliers} instances of rapid acceleration/deceleration, indicating unstable behavior",
                    value={"acceleration_events": int(accel_outliers)},
                    confidence=0.7,
                    impact_score=min(100, accel_outliers / len(series) * 500),
                ))

        return anomalies

    def _enrich_anomaly_explanation(self, anomaly: Anomaly, system_type: str, domain: Dict, df: pd.DataFrame):
        """Generate natural language explanation and possible causes for an anomaly."""
        field_lower = anomaly.field_name.lower()

        # Find matching failure modes
        failure_modes = domain.get("failure_modes", {})
        possible_causes = []

        for mode_key, causes in failure_modes.items():
            if any(keyword in field_lower for keyword in mode_key.split("_")):
                possible_causes.extend(causes[:3])  # Top 3 causes
                break

        # If no specific causes found, generate generic ones
        if not possible_causes:
            if anomaly.anomaly_type == AnomalyType.STATISTICAL_OUTLIER:
                possible_causes = [
                    "Sensor calibration drift",
                    "Environmental interference",
                    "Actual system anomaly requiring investigation",
                ]
            elif anomaly.anomaly_type == AnomalyType.THRESHOLD_BREACH:
                possible_causes = [
                    "Operating conditions outside design parameters",
                    "Component degradation",
                    "Control system malfunction",
                ]
            elif anomaly.anomaly_type == AnomalyType.TREND_CHANGE:
                possible_causes = [
                    "Gradual component wear",
                    "Process drift",
                    "Environmental changes affecting operation",
                ]

        anomaly.possible_causes = possible_causes

        # Generate natural language explanation
        explanation = self._generate_explanation(anomaly, system_type, domain)
        anomaly.natural_language_explanation = explanation

        # Generate recommendations
        recommendations = self._generate_anomaly_recommendations(anomaly, domain)
        anomaly.recommendations = recommendations

    def _generate_explanation(self, anomaly: Anomaly, system_type: str, domain: Dict) -> str:
        """Generate a natural language explanation for an anomaly."""
        field = anomaly.field_name
        severity = anomaly.severity.value

        # System type context
        system_context = {
            "industrial": "industrial equipment",
            "vehicle": "vehicle systems",
            "robot": "robotic system",
            "medical_device": "medical device",
            "aerospace": "aerospace system",
        }.get(system_type, "system")

        # Build explanation based on anomaly type
        if anomaly.anomaly_type == AnomalyType.STATISTICAL_OUTLIER:
            count = anomaly.value.get("outlier_count", 0)
            z_score = anomaly.value.get("max_z_score", 0)
            explanation = (
                f"Analysis of {field} in your {system_context} has detected {count} statistically unusual readings "
                f"that deviate significantly from normal patterns (up to {z_score:.1f} standard deviations). "
                f"This level of deviation is classified as {severity} severity because "
            )
            if severity == "critical":
                explanation += "it represents extreme outliers that rarely occur under normal operating conditions and often indicate serious issues."
            elif severity == "high":
                explanation += "values this far from normal typically indicate a problem that needs attention soon."
            else:
                explanation += "while not immediately dangerous, these values warrant monitoring and investigation."

        elif anomaly.anomaly_type == AnomalyType.THRESHOLD_BREACH:
            max_val = anomaly.value.get("max_value") or anomaly.value.get("min_value")
            threshold = anomaly.value.get("threshold")
            count = anomaly.value.get("count", 0)
            explanation = (
                f"Your {system_context}'s {field} has exceeded safe operating limits {count} times. "
                f"Values reached {max_val:.2f} compared to the recommended limit of {threshold}. "
                f"Operating outside these parameters can lead to accelerated wear, reduced efficiency, "
                f"or potential safety issues. Immediate investigation is recommended."
            )

        elif anomaly.anomaly_type == AnomalyType.TREND_CHANGE:
            direction = anomaly.value.get("direction", "changing")
            pct = abs(anomaly.value.get("change_percentage", 0))
            explanation = (
                f"A significant {direction} trend has been detected in {field}, with values shifting by approximately {pct:.1f}% "
                f"over the analysis period. In {system_context}s, this type of gradual change often indicates "
                f"developing issues such as component wear, degradation, or changing operating conditions. "
                f"Early detection of these trends allows for proactive maintenance before failures occur."
            )

        elif anomaly.anomaly_type == AnomalyType.CORRELATION_BREAK:
            actual_corr = anomaly.value.get("actual_correlation", 0)
            expected = anomaly.value.get("expected", "positive")
            related = anomaly.related_fields[0] if anomaly.related_fields else "related parameters"
            explanation = (
                f"The expected relationship between {field} and {related} has broken down. "
                f"Normally, these parameters should have a {expected} correlation, but the data shows "
                f"a correlation of {actual_corr:.2f}. This unexpected behavior in your {system_context} "
                f"may indicate sensor malfunction, unusual operating conditions, or a developing fault "
                f"that is disrupting normal system behavior."
            )

        elif anomaly.anomaly_type == AnomalyType.PATTERN_ANOMALY:
            if "stuck" in anomaly.title.lower() or "sensor" in anomaly.title.lower():
                explanation = (
                    f"The {field} sensor appears to be malfunctioning. The readings show almost no variation, "
                    f"which is highly unusual for an active {system_context}. This could indicate a failed sensor, "
                    f"disconnected wiring, or a frozen value in the data acquisition system. "
                    f"Sensor integrity should be verified immediately as this creates a blind spot in monitoring."
                )
            else:
                explanation = (
                    f"Unusual patterns have been detected in {field}. The data shows sudden jumps or changes "
                    f"that don't match normal operating patterns for your {system_context}. These discontinuities "
                    f"may indicate intermittent faults, loose connections, or actual rapid changes in the monitored parameter."
                )

        elif anomaly.anomaly_type == AnomalyType.RATE_OF_CHANGE:
            events = anomaly.value.get("acceleration_events", 0)
            explanation = (
                f"The rate at which {field} is changing shows unusual behavior, with {events} instances of "
                f"rapid acceleration or deceleration detected. In a healthy {system_context}, parameters typically "
                f"change smoothly and predictably. These rapid fluctuations may indicate control system issues, "
                f"mechanical problems, or unstable operating conditions."
            )

        else:
            explanation = f"An anomaly has been detected in {field} that requires investigation."

        return explanation

    def _generate_anomaly_recommendations(self, anomaly: Anomaly, domain: Dict) -> List[Dict[str, str]]:
        """Generate specific recommendations for addressing an anomaly."""
        recommendations = []
        field_lower = anomaly.field_name.lower()

        # Priority based on severity
        priority_map = {
            Severity.CRITICAL: "immediate",
            Severity.HIGH: "high",
            Severity.MEDIUM: "medium",
            Severity.LOW: "low",
            Severity.INFO: "low",
        }
        priority = priority_map.get(anomaly.severity, "medium")

        # Get domain-specific recommendations
        maint_recs = domain.get("maintenance_recommendations", {})
        for param, rec in maint_recs.items():
            if param in field_lower:
                recommendations.append({
                    "type": "maintenance",
                    "priority": priority,
                    "action": rec,
                })
                break

        # Add general recommendations based on anomaly type
        if anomaly.anomaly_type == AnomalyType.STATISTICAL_OUTLIER:
            recommendations.append({
                "type": "investigation",
                "priority": priority,
                "action": f"Review historical data for {anomaly.field_name} to identify when anomalies started",
            })
        elif anomaly.anomaly_type == AnomalyType.THRESHOLD_BREACH:
            recommendations.append({
                "type": "immediate_action",
                "priority": priority,
                "action": f"Verify {anomaly.field_name} sensor calibration and check for environmental factors",
            })
            recommendations.append({
                "type": "process",
                "priority": priority,
                "action": "Review operating procedures to prevent threshold breaches",
            })
        elif anomaly.anomaly_type == AnomalyType.TREND_CHANGE:
            recommendations.append({
                "type": "predictive",
                "priority": priority,
                "action": f"Schedule inspection of components related to {anomaly.field_name} before degradation worsens",
            })
        elif anomaly.anomaly_type == AnomalyType.PATTERN_ANOMALY:
            if "sensor" in anomaly.title.lower():
                recommendations.append({
                    "type": "immediate_action",
                    "priority": "high",
                    "action": f"Inspect and test {anomaly.field_name} sensor for proper operation",
                })

        return recommendations

    def _calculate_engineering_margins(self, df: pd.DataFrame, domain: Dict) -> List[Dict]:
        """Calculate how close parameters are to their design limits."""
        margins = []
        normal_ranges = domain.get("normal_ranges", {})

        for col in df.columns:
            col_lower = col.lower()

            for param, (min_val, max_val) in normal_ranges.items():
                if param in col_lower:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        continue

                    series = df[col].dropna()
                    if len(series) == 0:
                        continue

                    current_value = series.iloc[-10:].mean() if len(series) >= 10 else series.mean()

                    # Calculate margin to upper limit
                    if max_val > min_val:
                        margin_upper = ((max_val - current_value) / (max_val - min_val)) * 100
                        margin_lower = ((current_value - min_val) / (max_val - min_val)) * 100

                        # Use the smaller margin
                        margin_pct = min(margin_upper, margin_lower)

                        # Determine trend
                        if len(series) >= 20:
                            recent = series.tail(len(series)//2).mean()
                            older = series.head(len(series)//2).mean()
                            if recent > older * 1.05:
                                trend = "degrading"
                            elif recent < older * 0.95:
                                trend = "improving"
                            else:
                                trend = "stable"
                        else:
                            trend = "unknown"

                        margins.append({
                            "component": col,
                            "parameter": param,
                            "current_value": float(current_value),
                            "design_limit": float(max_val),
                            "lower_limit": float(min_val),
                            "margin_percentage": float(max(0, min(100, margin_pct))),
                            "trend": trend,
                            "safety_critical": param in domain.get("critical_parameters", []),
                        })
                    break

        return sorted(margins, key=lambda x: x["margin_percentage"])

    def _identify_blind_spots(self, df: pd.DataFrame, domain: Dict, discovered_schema: List[Dict] = None) -> List[Dict]:
        """Identify monitoring gaps and suggest additional sensors."""
        blind_spots = []
        available_params = set(col.lower() for col in df.columns)
        critical_params = domain.get("critical_parameters", [])

        # Check for missing critical parameters
        for param in critical_params:
            if not any(param in col for col in available_params):
                blind_spots.append({
                    "title": f"No {param} monitoring detected",
                    "description": f"Critical parameter '{param}' is not being monitored. This creates a blind spot in fault detection.",
                    "recommended_sensor": {
                        "type": f"{param} sensor",
                        "specification": f"Industrial-grade {param} sensor with appropriate range",
                        "estimated_cost": 500,
                    },
                    "diagnostic_coverage_improvement": 15,
                })

        # Check for low-quality data
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                null_pct = df[col].isna().sum() / len(df) * 100
                if null_pct > 20:
                    blind_spots.append({
                        "title": f"Data quality issue in {col}",
                        "description": f"{null_pct:.1f}% of {col} readings are missing, reducing diagnostic capability",
                        "recommended_sensor": None,
                        "diagnostic_coverage_improvement": min(20, null_pct / 2),
                    })

        return blind_spots[:5]  # Top 5 blind spots

    def _calculate_health_score(self, anomalies: List[Anomaly], margins: List[Dict]) -> float:
        """Calculate overall system health score."""
        score = 100.0

        # Deduct for anomalies
        severity_deductions = {
            Severity.CRITICAL: 25,
            Severity.HIGH: 15,
            Severity.MEDIUM: 8,
            Severity.LOW: 3,
            Severity.INFO: 1,
        }

        for anomaly in anomalies:
            score -= severity_deductions.get(anomaly.severity, 5)

        # Deduct for low margins
        for margin in margins:
            if margin["margin_percentage"] < 10:
                score -= 10
            elif margin["margin_percentage"] < 20:
                score -= 5
            elif margin["margin_percentage"] < 30:
                score -= 2

        return max(0, min(100, score))

    def _generate_insights(self, anomalies: List[Anomaly], trends: Dict, correlations: Dict, domain: Dict) -> List[str]:
        """Generate high-level insights from the analysis."""
        insights = []

        # Anomaly summary
        critical_count = sum(1 for a in anomalies if a.severity == Severity.CRITICAL)
        high_count = sum(1 for a in anomalies if a.severity == Severity.HIGH)

        if critical_count > 0:
            insights.append(f"URGENT: {critical_count} critical anomalies require immediate attention")
        if high_count > 0:
            insights.append(f"{high_count} high-severity issues detected that should be addressed soon")

        # Trend insights
        degrading_trends = [k for k, v in trends.items() if v.get("direction") == "increasing" and
                          any(p in k.lower() for p in domain.get("critical_parameters", []))]
        if degrading_trends:
            insights.append(f"Concerning upward trends detected in: {', '.join(degrading_trends[:3])}")

        # Overall status
        if not anomalies:
            insights.append("No significant anomalies detected - system appears to be operating normally")
        elif len(anomalies) < 3 and critical_count == 0:
            insights.append("Minor issues detected but overall system health is acceptable")

        return insights

    def _generate_summary(self, system_type: str, anomalies: List[Anomaly], health_score: float,
                         record_count: int, domain: Dict) -> str:
        """Generate an executive summary of the analysis."""
        system_name = {
            "industrial": "Industrial Equipment",
            "vehicle": "Vehicle",
            "robot": "Robotic System",
            "medical_device": "Medical Device",
            "aerospace": "Aerospace System",
        }.get(system_type, "System")

        critical = sum(1 for a in anomalies if a.severity == Severity.CRITICAL)
        high = sum(1 for a in anomalies if a.severity == Severity.HIGH)
        medium = sum(1 for a in anomalies if a.severity == Severity.MEDIUM)

        status = "CRITICAL" if critical > 0 else ("WARNING" if high > 0 else ("ATTENTION" if medium > 0 else "HEALTHY"))

        summary = f"{system_name} Analysis Summary\n\n"
        summary += f"Health Score: {health_score:.0f}/100 - Status: {status}\n\n"
        summary += f"Analyzed {record_count:,} data points across multiple parameters.\n\n"

        if critical > 0:
            summary += f"CRITICAL: Found {critical} critical issue(s) requiring immediate action.\n"
        if high > 0:
            summary += f"HIGH: Found {high} high-severity issue(s) that need attention soon.\n"
        if medium > 0:
            summary += f"MEDIUM: Found {medium} medium-severity issue(s) to monitor.\n"

        if not anomalies:
            summary += "No significant anomalies detected. System is operating within normal parameters."

        return summary

    def _generate_recommendations(self, anomalies: List[Anomaly], margins: List[Dict], domain: Dict) -> List[Dict]:
        """Generate prioritized list of recommendations."""
        recommendations = []

        # Collect all recommendations from anomalies
        for anomaly in sorted(anomalies, key=lambda a: a.severity.value):
            for rec in anomaly.recommendations:
                rec["source_anomaly"] = anomaly.title
                recommendations.append(rec)

        # Add margin-based recommendations
        for margin in margins:
            if margin["margin_percentage"] < 20:
                recommendations.append({
                    "type": "preventive",
                    "priority": "high" if margin["margin_percentage"] < 10 else "medium",
                    "action": f"Review {margin['component']} operating conditions - only {margin['margin_percentage']:.0f}% margin to limit",
                    "source_anomaly": "Engineering margin analysis",
                })

        return recommendations[:10]  # Top 10 recommendations

    def _get_severity_from_percentage(self, pct: float) -> Severity:
        """Convert percentage deviation to severity level."""
        if pct > 50:
            return Severity.CRITICAL
        elif pct > 30:
            return Severity.HIGH
        elif pct > 15:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _empty_result(self, system_id: str, system_type: str) -> AnalysisResult:
        """Return empty result when no data available."""
        return AnalysisResult(
            system_id=system_id,
            system_type=system_type,
            health_score=100.0,
            anomalies=[],
            insights=["No data available for analysis"],
            engineering_margins=[],
            blind_spots=[],
            correlation_matrix={},
            trend_analysis={},
            summary="No data available for analysis. Please upload telemetry data to enable analysis.",
            recommendations=[],
            analyzed_at=datetime.utcnow().isoformat(),
        )


# Global instance
analysis_engine = AnalysisEngine()
