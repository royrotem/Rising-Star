"""
Physics-Aware Anomaly Detection Service

Detects behavioral deviations and calculates engineering margins
with understanding of physical context.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DetectedAnomaly:
    """Represents a detected anomaly."""
    anomaly_id: str
    anomaly_type: str
    severity: str
    title: str
    description: str
    affected_fields: List[str]
    expected_range: Dict[str, float]
    actual_values: Dict[str, float]
    deviation_percentage: float
    engineering_margin: float
    timestamp: str
    confidence: float
    context: Dict[str, Any]


@dataclass
class EngineeringMarginResult:
    """Result of engineering margin analysis."""
    component: str
    parameter: str
    current_value: float
    design_limit: float
    margin_percentage: float
    trend: str  # "stable", "degrading", "improving"
    projected_breach_date: Optional[str]
    safety_buffer: float
    optimization_potential: float


class AnomalyDetectionService:
    """
    Physics-aware anomaly detection that goes beyond simple thresholds.
    Understands behavioral context and calculates engineering margins.
    """

    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.baseline_profiles: Dict[str, Dict] = {}

    async def analyze_system(
        self,
        system_id: str,
        data: pd.DataFrame,
        design_specs: Dict[str, Dict],
        historical_baseline: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive anomaly analysis on system data.
        
        Args:
            system_id: System identifier
            data: Current telemetry data
            design_specs: Engineering design specifications with limits
            historical_baseline: Historical data for baseline comparison
        
        Returns:
            Analysis results with anomalies and margins
        """
        results = {
            "system_id": system_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "anomalies": [],
            "engineering_margins": [],
            "behavioral_deviations": [],
            "health_score": 100.0,
        }

        # 1. Detect statistical anomalies
        statistical_anomalies = await self._detect_statistical_anomalies(data)
        results["anomalies"].extend(statistical_anomalies)

        # 2. Detect behavioral deviations
        if historical_baseline is not None:
            behavioral = await self._detect_behavioral_deviations(
                data, historical_baseline
            )
            results["behavioral_deviations"] = behavioral
            results["anomalies"].extend(behavioral)

        # 3. Calculate engineering margins
        margins = await self._calculate_engineering_margins(data, design_specs)
        results["engineering_margins"] = margins

        # 4. Detect contextual anomalies (physics-aware)
        contextual = await self._detect_contextual_anomalies(data, design_specs)
        results["anomalies"].extend(contextual)

        # 5. Calculate overall health score
        results["health_score"] = self._calculate_health_score(
            results["anomalies"],
            results["engineering_margins"]
        )

        # 6. Prioritize by 80/20 impact
        results["anomalies"] = self._prioritize_by_impact(results["anomalies"])

        return results

    async def _detect_statistical_anomalies(
        self,
        data: pd.DataFrame
    ) -> List[Dict]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return anomalies

        # Prepare data for isolation forest
        clean_data = data[numeric_cols].dropna()
        if len(clean_data) < 10:
            return anomalies

        # Fit isolation forest
        scaled_data = self.scaler.fit_transform(clean_data)
        predictions = self.isolation_forest.fit_predict(scaled_data)
        anomaly_scores = self.isolation_forest.decision_function(scaled_data)

        # Find anomalous points
        anomaly_indices = np.where(predictions == -1)[0]

        for idx in anomaly_indices[:10]:  # Limit to top 10
            row = clean_data.iloc[idx]
            
            # Find which features contributed most to anomaly
            contributions = self._calculate_feature_contributions(
                row, clean_data, numeric_cols
            )
            top_contributors = sorted(
                contributions.items(), key=lambda x: x[1], reverse=True
            )[:3]

            anomalies.append({
                "anomaly_type": "statistical_outlier",
                "severity": self._score_to_severity(anomaly_scores[idx]),
                "title": f"Statistical anomaly detected in {top_contributors[0][0]}",
                "description": f"Unusual values detected: {', '.join([f'{k}: {row[k]:.2f}' for k, _ in top_contributors])}",
                "affected_fields": [k for k, _ in top_contributors],
                "confidence": min(abs(anomaly_scores[idx]) / 0.5, 1.0),
                "timestamp": datetime.utcnow().isoformat(),
                "index": int(idx),
            })

        return anomalies

    async def _detect_behavioral_deviations(
        self,
        current_data: pd.DataFrame,
        baseline_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Detect when system behavior changes from historical baseline.
        This catches issues even when no absolute thresholds are crossed.
        """
        deviations = []
        numeric_cols = set(current_data.select_dtypes(include=[np.number]).columns) & \
                       set(baseline_data.select_dtypes(include=[np.number]).columns)

        for col in numeric_cols:
            baseline_stats = {
                "mean": baseline_data[col].mean(),
                "std": baseline_data[col].std(),
                "median": baseline_data[col].median(),
            }
            current_stats = {
                "mean": current_data[col].mean(),
                "std": current_data[col].std(),
                "median": current_data[col].median(),
            }

            # Detect mean shift
            if baseline_stats["std"] > 0:
                z_score = abs(current_stats["mean"] - baseline_stats["mean"]) / baseline_stats["std"]
                
                if z_score > 2:  # More than 2 std deviations
                    deviation_pct = abs(current_stats["mean"] - baseline_stats["mean"]) / abs(baseline_stats["mean"]) * 100
                    
                    deviations.append({
                        "anomaly_type": "behavioral_deviation",
                        "severity": "medium" if z_score < 3 else "high",
                        "title": f"Behavioral change detected in {col}",
                        "description": f"{col} is now averaging {current_stats['mean']:.2f} vs historical {baseline_stats['mean']:.2f} ({deviation_pct:.1f}% change)",
                        "affected_fields": [col],
                        "expected_range": {
                            "min": baseline_stats["mean"] - 2 * baseline_stats["std"],
                            "max": baseline_stats["mean"] + 2 * baseline_stats["std"],
                        },
                        "actual_values": {"mean": current_stats["mean"]},
                        "deviation_percentage": deviation_pct,
                        "confidence": min(z_score / 5, 1.0),
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            # Detect variance change (system becoming unstable)
            if baseline_stats["std"] > 0:
                variance_ratio = current_stats["std"] / baseline_stats["std"]
                
                if variance_ratio > 1.5:
                    deviations.append({
                        "anomaly_type": "variance_increase",
                        "severity": "medium",
                        "title": f"Increased variability in {col}",
                        "description": f"{col} showing {(variance_ratio - 1) * 100:.0f}% more variance than baseline - possible instability",
                        "affected_fields": [col],
                        "confidence": min((variance_ratio - 1) / 2, 1.0),
                        "timestamp": datetime.utcnow().isoformat(),
                    })

        return deviations

    async def _calculate_engineering_margins(
        self,
        data: pd.DataFrame,
        design_specs: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Calculate how close each parameter is to its design limits.
        Reveals hidden safety buffers and degradation trends.
        """
        margins = []

        for param, specs in design_specs.items():
            if param not in data.columns:
                continue

            current_values = data[param].dropna()
            if len(current_values) == 0:
                continue

            current_max = current_values.max()
            current_min = current_values.min()
            current_mean = current_values.mean()

            design_max = specs.get("max_limit")
            design_min = specs.get("min_limit")

            if design_max is not None:
                margin_to_max = ((design_max - current_max) / design_max) * 100
                
                # Calculate trend
                if len(current_values) > 10:
                    trend_slope = np.polyfit(range(len(current_values)), current_values, 1)[0]
                    trend = "degrading" if trend_slope > 0 else "improving" if trend_slope < 0 else "stable"
                    
                    # Project when limit will be breached
                    if trend == "degrading" and trend_slope > 0:
                        points_to_breach = (design_max - current_max) / trend_slope
                        if points_to_breach > 0:
                            breach_date = datetime.utcnow() + timedelta(hours=points_to_breach)
                        else:
                            breach_date = None
                    else:
                        breach_date = None
                else:
                    trend = "stable"
                    breach_date = None

                margins.append({
                    "component": specs.get("component", "unknown"),
                    "parameter": param,
                    "current_value": float(current_max),
                    "design_limit": float(design_max),
                    "margin_percentage": float(margin_to_max),
                    "trend": trend,
                    "projected_breach_date": breach_date.isoformat() if breach_date else None,
                    "safety_buffer": float(margin_to_max) if margin_to_max > 20 else 0,
                    "optimization_potential": float(margin_to_max - 20) if margin_to_max > 40 else 0,
                    "safety_critical": specs.get("safety_critical", False),
                })

        return margins

    async def _detect_contextual_anomalies(
        self,
        data: pd.DataFrame,
        design_specs: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Detect anomalies based on physical relationships and context.
        E.g., motor drawing more current at specific humidity.
        """
        anomalies = []

        # Define physical relationships to check
        relationships = [
            {
                "name": "power_temperature",
                "fields": ["power", "temperature"],
                "check": self._check_power_temperature_relationship,
            },
            {
                "name": "current_load",
                "fields": ["current", "load"],
                "check": self._check_current_load_relationship,
            },
        ]

        for rel in relationships:
            # Check if required fields exist (fuzzy match)
            matched_fields = {}
            for required_field in rel["fields"]:
                for col in data.columns:
                    if required_field.lower() in col.lower():
                        matched_fields[required_field] = col
                        break

            if len(matched_fields) == len(rel["fields"]):
                anomaly = await rel["check"](data, matched_fields)
                if anomaly:
                    anomalies.append(anomaly)

        return anomalies

    async def _check_power_temperature_relationship(
        self,
        data: pd.DataFrame,
        fields: Dict[str, str]
    ) -> Optional[Dict]:
        """Check if power consumption correlates abnormally with temperature."""
        power_col = fields.get("power")
        temp_col = fields.get("temperature")

        if not power_col or not temp_col:
            return None

        correlation = data[power_col].corr(data[temp_col])

        if correlation > 0.8:
            return {
                "anomaly_type": "contextual",
                "severity": "medium",
                "title": "Abnormal power-temperature coupling",
                "description": f"Power consumption is highly correlated with temperature ({correlation:.2f}). This may indicate cooling system inefficiency.",
                "affected_fields": [power_col, temp_col],
                "confidence": correlation,
                "timestamp": datetime.utcnow().isoformat(),
            }

        return None

    async def _check_current_load_relationship(
        self,
        data: pd.DataFrame,
        fields: Dict[str, str]
    ) -> Optional[Dict]:
        """Check if current draw is proportional to load."""
        current_col = fields.get("current")
        load_col = fields.get("load")

        if not current_col or not load_col:
            return None

        # Calculate expected vs actual current ratio
        # In ideal systems, current should scale linearly with load
        if data[load_col].std() > 0:
            ratio = data[current_col] / (data[load_col] + 0.001)
            ratio_std = ratio.std() / ratio.mean() if ratio.mean() > 0 else 0

            if ratio_std > 0.3:  # High variance in the ratio
                return {
                    "anomaly_type": "contextual",
                    "severity": "low",
                    "title": "Inconsistent current-load relationship",
                    "description": f"Current draw is not proportional to load (variance ratio: {ratio_std:.2f}). Possible efficiency degradation.",
                    "affected_fields": [current_col, load_col],
                    "confidence": min(ratio_std, 1.0),
                    "timestamp": datetime.utcnow().isoformat(),
                }

        return None

    def _calculate_feature_contributions(
        self,
        row: pd.Series,
        data: pd.DataFrame,
        columns: pd.Index
    ) -> Dict[str, float]:
        """Calculate how much each feature contributed to an anomaly."""
        contributions = {}
        for col in columns:
            mean = data[col].mean()
            std = data[col].std()
            if std > 0:
                z_score = abs(row[col] - mean) / std
                contributions[col] = z_score
            else:
                contributions[col] = 0
        return contributions

    def _score_to_severity(self, anomaly_score: float) -> str:
        """Convert anomaly score to severity level."""
        if anomaly_score < -0.5:
            return "critical"
        elif anomaly_score < -0.3:
            return "high"
        elif anomaly_score < -0.1:
            return "medium"
        else:
            return "low"

    def _calculate_health_score(
        self,
        anomalies: List[Dict],
        margins: List[Dict]
    ) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0

        # Deduct points for anomalies
        severity_deductions = {
            "critical": 25,
            "high": 15,
            "medium": 8,
            "low": 3,
        }

        for anomaly in anomalies:
            severity = anomaly.get("severity", "low")
            score -= severity_deductions.get(severity, 0)

        # Deduct points for low margins
        for margin in margins:
            if margin.get("safety_critical"):
                if margin["margin_percentage"] < 10:
                    score -= 20
                elif margin["margin_percentage"] < 20:
                    score -= 10

        return max(0, min(100, score))

    def _prioritize_by_impact(self, anomalies: List[Dict]) -> List[Dict]:
        """
        Sort anomalies by 80/20 impact score.
        Focus on the 20% of issues causing 80% of problems.
        """
        def impact_score(anomaly: Dict) -> float:
            severity_weights = {
                "critical": 100,
                "high": 70,
                "medium": 40,
                "low": 10,
            }
            base_score = severity_weights.get(anomaly.get("severity", "low"), 10)
            confidence = anomaly.get("confidence", 0.5)
            return base_score * confidence

        for anomaly in anomalies:
            anomaly["impact_score"] = impact_score(anomaly)

        return sorted(anomalies, key=lambda x: x["impact_score"], reverse=True)
