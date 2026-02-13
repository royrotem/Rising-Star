"""
Hard-Coded Anomaly Detection Models

Nine lightweight, deterministic algorithms that require no pre-trained model
files.  They operate purely on the numeric columns of the ingested data
using classical statistical / rule-based techniques.

Algorithms:
  1. Z-Score Detector                — flags values > k standard deviations from mean
  2. IQR Outlier Detector            — inter-quartile-range fence method
  3. Moving Average Deviation        — sliding-window mean vs actual value
  4. Min-Max Boundary Checker        — values outside [global_min, global_max] bands
  5. Isolation Score Detector        — simplified isolation-forest-style random splits
  6. DBSCAN Outlier Detector         — density-based clustering, noise points = anomalies
  7. Local Outlier Factor (LOF)      — local density ratio anomaly scoring
  8. Elliptic Envelope Detector      — robust covariance (Mahalanobis distance)
  9. EWMA Control Chart Detector     — exponentially weighted moving average deviation
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
# 6. DBSCAN Outlier Detector
# ═══════════════════════════════════════════════════════════════════

class DBSCANOutlierDetector:
    """Uses DBSCAN clustering to identify noise points (label = -1) as
    anomalies.  DBSCAN is effective at finding points in low-density
    regions without assuming any particular distribution shape.

    Parameters are auto-tuned: eps is derived from the k-nearest-neighbour
    distance distribution, and min_samples scales with dataset size.
    """

    name = "DBSCAN Outlier Detector"

    def __init__(self, eps: Optional[float] = None, min_samples: Optional[int] = None,
                 max_anomalies: int = 50) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.max_anomalies = max_anomalies

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not HAS_SKLEARN:
            return []

        numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all").fillna(0)
        if numeric.empty or len(numeric) < 20 or numeric.shape[1] < 2:
            return []

        scaler = StandardScaler()
        X = scaler.fit_transform(numeric.values)

        # Auto-tune eps using k-distance heuristic (k = min_samples)
        min_samples = self.min_samples or max(5, len(X) // 50)
        if self.eps is None:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min_samples)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            k_distances = np.sort(distances[:, -1])
            # Use the "knee" — 90th percentile of k-distances
            eps = float(np.percentile(k_distances, 90))
            eps = max(eps, 0.5)  # floor
        else:
            eps = self.eps

        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X)

        noise_indices = np.where(labels == -1)[0]
        if len(noise_indices) == 0:
            return []

        # Score each noise point by distance to nearest cluster centroid
        cluster_labels = set(labels) - {-1}
        if cluster_labels:
            centroids = np.array([X[labels == c].mean(axis=0) for c in cluster_labels])
        else:
            centroids = np.array([X.mean(axis=0)])

        anomalies: List[Dict] = []
        for idx in noise_indices[:self.max_anomalies]:
            point = X[idx]
            min_dist = float(np.min(np.linalg.norm(centroids - point, axis=1)))
            score = min(min_dist / 5.0, 1.0)
            severity = _score_to_severity(score)

            # Top deviating features
            col_means = X.mean(axis=0)
            col_stds = X.std(axis=0)
            col_stds[col_stds == 0] = 1
            abs_z = np.abs((point - col_means) / col_stds)
            top_feat_idx = np.argsort(abs_z)[-3:][::-1]
            top_features = [numeric.columns[j] for j in top_feat_idx]

            anomalies.append(_build_anomaly(
                anomaly_type="hardcoded_dbscan",
                title=f"DBSCAN noise point at row {idx}",
                description=(
                    f"Row {idx} was classified as noise by DBSCAN clustering "
                    f"(eps={eps:.3f}, min_samples={min_samples}). "
                    f"Distance to nearest cluster: {min_dist:.3f}. "
                    f"Top deviating features: {', '.join(top_features)}."
                ),
                severity=severity,
                confidence=round(score, 4),
                affected_fields=top_features,
                index=int(idx),
                detected_by=self.name,
                model_metadata={
                    "distance_to_cluster": round(min_dist, 4),
                    "eps": round(eps, 4),
                    "min_samples": min_samples,
                    "n_clusters": len(cluster_labels),
                    "n_noise": len(noise_indices),
                },
            ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# 7. Local Outlier Factor (LOF) Detector
# ═══════════════════════════════════════════════════════════════════

class LOFDetector:
    """Local Outlier Factor measures the local deviation of density of a
    given sample with respect to its neighbours.  Points with substantially
    lower local density than their neighbours are considered outliers.

    sklearn's LOF returns negative_outlier_factor_ where more negative = more
    anomalous.  We normalise this to a 0-1 score.
    """

    name = "Local Outlier Factor"

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05,
                 max_anomalies: int = 50) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.max_anomalies = max_anomalies

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not HAS_SKLEARN:
            return []

        numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all").fillna(0)
        if numeric.empty or len(numeric) < 30 or numeric.shape[1] < 2:
            return []

        scaler = StandardScaler()
        X = scaler.fit_transform(numeric.values)

        n_neighbors = min(self.n_neighbors, len(X) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1,
        )
        labels = lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_  # higher = more anomalous (>1 is normal boundary)

        outlier_indices = np.where(labels == -1)[0]
        if len(outlier_indices) == 0:
            return []

        # Sort by LOF score descending
        sorted_indices = outlier_indices[np.argsort(scores[outlier_indices])[::-1]]

        anomalies: List[Dict] = []
        for idx in sorted_indices[:self.max_anomalies]:
            lof_score = float(scores[idx])
            # Normalise: LOF > 1 is anomalous; map to 0-1 range
            norm_score = min((lof_score - 1.0) / 2.0, 1.0)
            norm_score = max(norm_score, 0.1)
            severity = _score_to_severity(norm_score)

            point = X[idx]
            col_means = X.mean(axis=0)
            col_stds = X.std(axis=0)
            col_stds[col_stds == 0] = 1
            abs_z = np.abs((point - col_means) / col_stds)
            top_feat_idx = np.argsort(abs_z)[-3:][::-1]
            top_features = [numeric.columns[j] for j in top_feat_idx]

            anomalies.append(_build_anomaly(
                anomaly_type="hardcoded_lof",
                title=f"LOF outlier at row {idx}",
                description=(
                    f"Row {idx} has a Local Outlier Factor of {lof_score:.3f} "
                    f"(>1.0 = anomalous). Its local density is significantly "
                    f"lower than its {n_neighbors} nearest neighbours. "
                    f"Top features: {', '.join(top_features)}."
                ),
                severity=severity,
                confidence=round(norm_score, 4),
                affected_fields=top_features,
                index=int(idx),
                detected_by=self.name,
                model_metadata={
                    "lof_score": round(lof_score, 4),
                    "n_neighbors": n_neighbors,
                    "contamination": self.contamination,
                },
            ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# 8. Elliptic Envelope (Robust Covariance) Detector
# ═══════════════════════════════════════════════════════════════════

class EllipticEnvelopeDetector:
    """Fits a robust covariance estimate (Minimum Covariance Determinant)
    and uses the Mahalanobis distance to flag multivariate outliers.

    Unlike univariate methods, this catches points that are only anomalous
    when considering the joint distribution of multiple features.
    """

    name = "Elliptic Envelope"

    def __init__(self, contamination: float = 0.05, max_anomalies: int = 50) -> None:
        self.contamination = contamination
        self.max_anomalies = max_anomalies

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not HAS_SKLEARN:
            return []

        numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all").fillna(0)
        if numeric.empty or len(numeric) < 30 or numeric.shape[1] < 2:
            return []

        # Limit features to avoid singular covariance matrix
        # Use at most 15 columns with highest variance
        if numeric.shape[1] > 15:
            variances = numeric.var().sort_values(ascending=False)
            numeric = numeric[variances.head(15).index]

        scaler = StandardScaler()
        X = scaler.fit_transform(numeric.values)

        try:
            ee = EllipticEnvelope(
                contamination=self.contamination,
                support_fraction=max(0.5, 1.0 - self.contamination * 2),
                random_state=42,
            )
            labels = ee.fit_predict(X)
            mahal_distances = ee.mahalanobis(X)
        except Exception as exc:
            logger.warning("EllipticEnvelope failed: %s", exc)
            return []

        outlier_indices = np.where(labels == -1)[0]
        if len(outlier_indices) == 0:
            return []

        # Normalise Mahalanobis distance to 0-1 score
        max_dist = mahal_distances.max()
        if max_dist == 0:
            return []

        sorted_indices = outlier_indices[np.argsort(mahal_distances[outlier_indices])[::-1]]

        anomalies: List[Dict] = []
        for idx in sorted_indices[:self.max_anomalies]:
            dist = float(mahal_distances[idx])
            score = min(dist / max_dist, 1.0)
            severity = _score_to_severity(score)

            point = X[idx]
            col_means = X.mean(axis=0)
            col_stds = X.std(axis=0)
            col_stds[col_stds == 0] = 1
            abs_z = np.abs((point - col_means) / col_stds)
            top_feat_idx = np.argsort(abs_z)[-3:][::-1]
            top_features = [numeric.columns[j] for j in top_feat_idx]

            anomalies.append(_build_anomaly(
                anomaly_type="hardcoded_elliptic_envelope",
                title=f"Multivariate outlier at row {idx}",
                description=(
                    f"Row {idx} has Mahalanobis distance {dist:.2f} from the "
                    f"robust covariance centre — flagged by Elliptic Envelope. "
                    f"This point is anomalous in the joint distribution of features. "
                    f"Top contributors: {', '.join(top_features)}."
                ),
                severity=severity,
                confidence=round(score, 4),
                affected_fields=top_features,
                index=int(idx),
                detected_by=self.name,
                model_metadata={
                    "mahalanobis_distance": round(dist, 4),
                    "contamination": self.contamination,
                },
            ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# 9. EWMA Control Chart Detector
# ═══════════════════════════════════════════════════════════════════

class EWMAControlChartDetector:
    """Exponentially Weighted Moving Average (EWMA) control chart.

    EWMA gives more weight to recent observations, making it sensitive to
    small, sustained shifts while being robust to isolated spikes.
    Control limits are calculated analytically from the smoothing parameter
    λ and the number of observations.
    """

    name = "EWMA Control Chart"

    def __init__(self, lam: float = 0.2, sigma_limit: float = 3.0,
                 max_anomalies_per_col: int = 20) -> None:
        self.lam = lam       # smoothing parameter (0 < λ ≤ 1)
        self.sigma_limit = sigma_limit  # control limit multiplier
        self.max_per_col = max_anomalies_per_col

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty or len(numeric) < 15:
            return []

        anomalies: List[Dict] = []
        lam = self.lam

        for col in numeric.columns:
            series = numeric[col].dropna()
            if len(series) < 15:
                continue

            mu = series.mean()
            sigma = series.std()
            if sigma == 0:
                continue

            values = series.values
            ewma = np.zeros(len(values))
            ewma[0] = mu  # start at process mean

            for i in range(1, len(values)):
                ewma[i] = lam * values[i] + (1 - lam) * ewma[i - 1]

            # Time-varying control limits
            # UCL/LCL = mu ± L * sigma * sqrt(λ/(2-λ) * (1-(1-λ)^(2i)))
            indices = np.arange(1, len(values) + 1, dtype=float)
            factor = np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * indices)))
            ucl = mu + self.sigma_limit * sigma * factor
            lcl = mu - self.sigma_limit * sigma * factor

            breach_mask = (ewma > ucl) | (ewma < lcl)
            breach_indices = np.where(breach_mask)[0]

            col_count = 0
            for idx in breach_indices:
                if col_count >= self.max_per_col:
                    break

                deviation = abs(ewma[idx] - mu) / sigma
                score = min(float(deviation) / (self.sigma_limit * 2), 1.0)
                severity = _score_to_severity(score)

                direction = "above UCL" if ewma[idx] > ucl[idx] else "below LCL"

                anomalies.append(_build_anomaly(
                    anomaly_type="hardcoded_ewma",
                    title=f"EWMA control breach in '{col}' at row {int(series.index[idx])}",
                    description=(
                        f"EWMA statistic ({ewma[idx]:.4g}) is {direction} "
                        f"(λ={lam}, L={self.sigma_limit}). "
                        f"This indicates a sustained shift from the process mean ({mu:.4g})."
                    ),
                    severity=severity,
                    confidence=round(score, 4),
                    affected_fields=[col],
                    index=int(series.index[idx]),
                    detected_by=self.name,
                    model_metadata={
                        "ewma_value": round(float(ewma[idx]), 4),
                        "ucl": round(float(ucl[idx]), 4),
                        "lcl": round(float(lcl[idx]), 4),
                        "process_mean": round(float(mu), 4),
                        "lambda": lam,
                        "sigma_limit": self.sigma_limit,
                    },
                ))
                col_count += 1

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
        self.dbscan = DBSCANOutlierDetector()
        self.lof = LOFDetector()
        self.elliptic = EllipticEnvelopeDetector()
        self.ewma = EWMAControlChartDetector()
        self.detectors = [
            self.zscore, self.iqr, self.moving_avg,
            self.minmax, self.isolation,
            self.dbscan, self.lof, self.elliptic, self.ewma,
        ]

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
