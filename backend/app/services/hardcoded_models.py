"""
Hard-Coded Anomaly Detection Models

Five lightweight, deterministic algorithms that require no pre-trained model
files.  They operate purely on the numeric columns of the ingested data
using classical statistical / rule-based techniques.

Algorithms:
  1. Z-Score Detector          — flags values > k standard deviations from mean
  2. IQR Outlier Detector      — inter-quartile-range fence method
  3. Moving Average Deviation  — sliding-window mean vs actual value
  4. Min-Max Boundary Checker  — values outside [global_min, global_max] bands
  5. Isolation Score Detector   — simplified isolation-forest-style random splits
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("uaie.hardcoded_models")

# ─── Severity helpers (mirrors ml_models.py) ─────────────────────

_SEVERITY_THRESHOLDS = [
    (0.9, "critical"),
    (0.7, "high"),
    (0.4, "medium"),
]


def _score_to_severity(score: float) -> str:
    for threshold, label in _SEVERITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "low"


def _severity_weight(severity: str) -> float:
    return {"critical": 0.95, "high": 0.80, "medium": 0.60, "low": 0.35}.get(severity, 0.5)


def _build_anomaly(
    *,
    anomaly_type: str,
    title: str,
    description: str,
    severity: str,
    confidence: float,
    affected_fields: List[str],
    index: int,
    detected_by: str,
    model_metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build a standardised anomaly dict compatible with the pipeline."""
    impact = round(_severity_weight(severity) * confidence * 100, 1)
    return {
        "anomaly_type": anomaly_type,
        "severity": severity,
        "title": title,
        "description": description,
        "affected_fields": affected_fields,
        "confidence": round(confidence, 4),
        "impact_score": impact,
        "timestamp": datetime.now().isoformat(),
        "index": index,
        "detected_by": detected_by,
        "detection_type": "hardcoded_model",
        "model_metadata": model_metadata or {},
        "contributing_agents": [detected_by],
        "web_references": [],
        "agent_perspectives": [],
    }


# ═══════════════════════════════════════════════════════════════════
# 1. Z-Score Detector
# ═══════════════════════════════════════════════════════════════════

class ZScoreDetector:
    """Flags data-points that are more than *threshold* standard deviations
    away from the column mean."""

    name = "Z-Score Detector"

    def __init__(self, threshold: float = 3.0, max_anomalies_per_col: int = 20) -> None:
        self.threshold = threshold
        self.max_per_col = max_anomalies_per_col

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            return []

        anomalies: List[Dict] = []
        for col in numeric.columns:
            series = numeric[col].dropna()
            if len(series) < 5:
                continue
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue

            z_scores = ((series - mean) / std).abs()
            outliers = z_scores[z_scores > self.threshold].sort_values(ascending=False)

            for count, (idx, z) in enumerate(outliers.items()):
                if count >= self.max_per_col:
                    break
                score = min(float(z) / (self.threshold * 2), 1.0)
                severity = _score_to_severity(score)
                anomalies.append(_build_anomaly(
                    anomaly_type="hardcoded_zscore",
                    title=f"Z-Score outlier in '{col}' at row {idx}",
                    description=(
                        f"Value {series[idx]:.4g} is {z:.2f} standard deviations from "
                        f"the mean ({mean:.4g}). Threshold: {self.threshold} σ."
                    ),
                    severity=severity,
                    confidence=round(score, 4),
                    affected_fields=[col],
                    index=int(idx),
                    detected_by=self.name,
                    model_metadata={"z_score": round(float(z), 4), "mean": round(float(mean), 4),
                                    "std": round(float(std), 4), "threshold": self.threshold},
                ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# 2. IQR Outlier Detector
# ═══════════════════════════════════════════════════════════════════

class IQROutlierDetector:
    """Uses the inter-quartile range method: outliers are beyond
    Q1 - k*IQR  or  Q3 + k*IQR."""

    name = "IQR Outlier Detector"

    def __init__(self, k: float = 1.5, max_anomalies_per_col: int = 20) -> None:
        self.k = k
        self.max_per_col = max_anomalies_per_col

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            return []

        anomalies: List[Dict] = []
        for col in numeric.columns:
            series = numeric[col].dropna()
            if len(series) < 5:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower = q1 - self.k * iqr
            upper = q3 + self.k * iqr
            mask = (series < lower) | (series > upper)
            outlier_idx = series[mask].index

            for count, idx in enumerate(outlier_idx):
                if count >= self.max_per_col:
                    break
                val = float(series[idx])
                distance = max(abs(val - lower), abs(val - upper)) / iqr
                score = min(distance / (self.k * 3), 1.0)
                severity = _score_to_severity(score)
                anomalies.append(_build_anomaly(
                    anomaly_type="hardcoded_iqr",
                    title=f"IQR outlier in '{col}' at row {idx}",
                    description=(
                        f"Value {val:.4g} is outside the IQR fence "
                        f"[{lower:.4g}, {upper:.4g}] (k={self.k})."
                    ),
                    severity=severity,
                    confidence=round(score, 4),
                    affected_fields=[col],
                    index=int(idx),
                    detected_by=self.name,
                    model_metadata={"value": val, "lower_fence": round(float(lower), 4),
                                    "upper_fence": round(float(upper), 4), "iqr": round(float(iqr), 4)},
                ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# 3. Moving Average Deviation Detector
# ═══════════════════════════════════════════════════════════════════

class MovingAverageDeviationDetector:
    """Compares each value to a rolling mean. Large deviations (relative
    to the rolling std) are flagged."""

    name = "Moving Average Deviation"

    def __init__(self, window: int = 10, std_factor: float = 2.5,
                 max_anomalies_per_col: int = 20) -> None:
        self.window = window
        self.std_factor = std_factor
        self.max_per_col = max_anomalies_per_col

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty or len(numeric) < self.window + 2:
            return []

        anomalies: List[Dict] = []
        for col in numeric.columns:
            series = numeric[col].dropna()
            if len(series) < self.window + 2:
                continue

            rolling_mean = series.rolling(self.window, min_periods=self.window).mean()
            rolling_std = series.rolling(self.window, min_periods=self.window).std()

            for idx in series.index[self.window:]:
                rm = rolling_mean.get(idx)
                rs = rolling_std.get(idx)
                if rm is None or rs is None or rs == 0:
                    continue
                deviation = abs(series[idx] - rm) / rs
                if deviation > self.std_factor:
                    score = min(float(deviation) / (self.std_factor * 2.5), 1.0)
                    severity = _score_to_severity(score)
                    anomalies.append(_build_anomaly(
                        anomaly_type="hardcoded_moving_avg",
                        title=f"Moving-avg deviation in '{col}' at row {idx}",
                        description=(
                            f"Value {series[idx]:.4g} deviates {deviation:.2f}σ from the "
                            f"{self.window}-period rolling mean ({rm:.4g})."
                        ),
                        severity=severity,
                        confidence=round(score, 4),
                        affected_fields=[col],
                        index=int(idx),
                        detected_by=self.name,
                        model_metadata={"deviation_sigma": round(float(deviation), 4),
                                        "rolling_mean": round(float(rm), 4),
                                        "rolling_std": round(float(rs), 4),
                                        "window": self.window},
                    ))
                    if len([a for a in anomalies if col in a["affected_fields"]]) >= self.max_per_col:
                        break
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# 4. Min-Max Boundary Checker
# ═══════════════════════════════════════════════════════════════════

class MinMaxBoundaryChecker:
    """Splits data into a 'training' segment (first 70 %) and a 'test'
    segment (last 30 %). Values in the test segment that fall outside the
    min/max range observed in training are flagged."""

    name = "Min-Max Boundary Checker"

    def __init__(self, margin_pct: float = 0.05, max_anomalies_per_col: int = 20) -> None:
        self.margin_pct = margin_pct
        self.max_per_col = max_anomalies_per_col

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty or len(numeric) < 10:
            return []

        split = int(len(numeric) * 0.7)
        train = numeric.iloc[:split]
        test = numeric.iloc[split:]

        anomalies: List[Dict] = []
        for col in numeric.columns:
            t_min = train[col].min()
            t_max = train[col].max()
            span = t_max - t_min
            if pd.isna(span) or span == 0:
                continue

            margin = span * self.margin_pct
            lower = t_min - margin
            upper = t_max + margin

            test_series = test[col].dropna()
            mask = (test_series < lower) | (test_series > upper)
            outlier_idx = test_series[mask].index

            for count, idx in enumerate(outlier_idx):
                if count >= self.max_per_col:
                    break
                val = float(test_series[idx])
                breach = max(lower - val, val - upper, 0) / span
                score = min(float(breach) / 0.5, 1.0)
                severity = _score_to_severity(score)
                anomalies.append(_build_anomaly(
                    anomaly_type="hardcoded_minmax",
                    title=f"Min-Max boundary breach in '{col}' at row {idx}",
                    description=(
                        f"Value {val:.4g} is outside the historical range "
                        f"[{lower:.4g}, {upper:.4g}] (margin {self.margin_pct*100:.0f}%)."
                    ),
                    severity=severity,
                    confidence=round(score, 4),
                    affected_fields=[col],
                    index=int(idx),
                    detected_by=self.name,
                    model_metadata={"value": val, "train_min": round(float(t_min), 4),
                                    "train_max": round(float(t_max), 4),
                                    "lower_bound": round(float(lower), 4),
                                    "upper_bound": round(float(upper), 4)},
                ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# 5. Isolation Score Detector (simplified)
# ═══════════════════════════════════════════════════════════════════

class IsolationScoreDetector:
    """A simplified isolation-forest-style detector: for each data point,
    performs *n_splits* random single-feature splits and measures how few
    splits are needed to isolate the point.  Points that are isolated
    quickly are likely anomalies."""

    name = "Isolation Score Detector"

    def __init__(self, n_splits: int = 100, contamination: float = 0.05,
                 max_anomalies: int = 50, seed: int = 42) -> None:
        self.n_splits = n_splits
        self.contamination = contamination
        self.max_anomalies = max_anomalies
        self.seed = seed

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all").fillna(0)
        if numeric.empty or len(numeric) < 10 or numeric.shape[1] < 2:
            return []

        rng = np.random.RandomState(self.seed)
        n_rows, n_cols = numeric.shape
        values = numeric.values

        # For each random split, count how many points end up on the
        # minority side.  Accumulate a "isolation depth" per row.
        isolation_depth = np.zeros(n_rows, dtype=float)

        for _ in range(self.n_splits):
            col_idx = rng.randint(0, n_cols)
            col_vals = values[:, col_idx]
            lo, hi = col_vals.min(), col_vals.max()
            if lo == hi:
                continue
            threshold = rng.uniform(lo, hi)
            left_mask = col_vals <= threshold
            left_count = left_mask.sum()
            right_count = n_rows - left_count
            # The side with fewer points is more "isolating"
            minority_size = np.where(left_mask, left_count, right_count)
            # Depth contribution: deeper (higher) means harder to isolate → normal
            isolation_depth += np.log2(minority_size.astype(float) + 1)

        # Normalise to 0-1 anomaly score (lower depth = more anomalous)
        max_depth = isolation_depth.max()
        if max_depth == 0:
            return []
        anomaly_scores = 1.0 - (isolation_depth / max_depth)

        # Take the top-k most anomalous
        n_anomalies = max(1, int(n_rows * self.contamination))
        n_anomalies = min(n_anomalies, self.max_anomalies)
        top_indices = np.argsort(anomaly_scores)[-n_anomalies:][::-1]

        # Only keep those above a minimal score threshold
        anomalies: List[Dict] = []
        for idx in top_indices:
            score = float(anomaly_scores[idx])
            if score < 0.3:
                continue
            # Find the columns contributing most to isolation
            row = values[idx]
            col_means = values.mean(axis=0)
            col_stds = values.std(axis=0)
            col_stds[col_stds == 0] = 1
            abs_z = np.abs((row - col_means) / col_stds)
            top_feat_idx = np.argsort(abs_z)[-3:][::-1]
            top_features = [numeric.columns[j] for j in top_feat_idx]

            severity = _score_to_severity(score)
            anomalies.append(_build_anomaly(
                anomaly_type="hardcoded_isolation",
                title=f"Isolation anomaly at row {idx}",
                description=(
                    f"Row {idx} scored {score:.3f} on isolation analysis "
                    f"({self.n_splits} random splits). "
                    f"Most deviant features: {', '.join(top_features)}."
                ),
                severity=severity,
                confidence=round(score, 4),
                affected_fields=top_features,
                index=int(idx),
                detected_by=self.name,
                model_metadata={"isolation_score": round(score, 4),
                                "n_splits": self.n_splits,
                                "top_features": top_features},
            ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# Hard-Coded Model Orchestrator
# ═══════════════════════════════════════════════════════════════════

class HardcodedModelOrchestrator:
    """Runs all hard-coded anomaly detectors and collects results."""

    def __init__(self) -> None:
        self.zscore = ZScoreDetector()
        self.iqr = IQROutlierDetector()
        self.moving_avg = MovingAverageDeviationDetector()
        self.minmax = MinMaxBoundaryChecker()
        self.isolation = IsolationScoreDetector()
        self.detectors = [self.zscore, self.iqr, self.moving_avg,
                          self.minmax, self.isolation]

    @property
    def models_available(self) -> Dict[str, bool]:
        return {d.name: True for d in self.detectors}

    async def run_all(self, records: List[Dict]) -> Dict[str, Any]:
        """Run all hard-coded detectors concurrently."""
        df = pd.DataFrame(records)
        if df.empty:
            return {
                "anomalies": [],
                "model_statuses": [],
                "total_findings": 0,
                "models_available": self.models_available,
            }

        tasks = []
        for detector in self.detectors:
            tasks.append(asyncio.to_thread(detector.detect, df))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_anomalies: List[Dict] = []
        model_statuses: List[Dict] = []

        for detector, result in zip(self.detectors, results):
            if isinstance(result, Exception):
                logger.error("Hardcoded model '%s' failed: %s", detector.name, result)
                model_statuses.append({
                    "model": detector.name,
                    "status": "error",
                    "findings": 0,
                    "elapsed_seconds": 0,
                    "error": str(result),
                })
            else:
                findings = result if isinstance(result, list) else []
                all_anomalies.extend(findings)
                model_statuses.append({
                    "model": detector.name,
                    "status": "success",
                    "findings": len(findings),
                    "elapsed_seconds": 0,
                })

        return {
            "anomalies": all_anomalies,
            "model_statuses": model_statuses,
            "total_findings": len(all_anomalies),
            "models_available": self.models_available,
        }


# ─── Global instance ──────────────────────────────────────────────

hardcoded_orchestrator = HardcodedModelOrchestrator()
