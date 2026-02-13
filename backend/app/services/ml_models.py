"""
ML Model Inference Service

Pre-trained models (XGBoost, 1D-CNN Autoencoder, Logistic Regression) are
loaded from backend/models/ and run inference only.

Self-training models (Isolation Forest, One-Class SVM, Gaussian Mixture Model,
KDE) fit on the incoming data at analysis time — no pre-trained files needed.

Each detector gracefully degrades: if its dependencies or model files are
missing the detector reports itself as unavailable and the pipeline continues.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("uaie.ml_models")

# ─── Conditional imports ───────────────────────────────────────────

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("xgboost not installed — XGBoost detector will be unavailable")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("torch not installed — CNN autoencoder detector will be unavailable")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import StandardScaler as SkScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.info("scikit-learn not installed — self-training ML detectors will be unavailable")

from ..core.config import settings


def _get_models_dir() -> Path:
    if settings.MODELS_DIR:
        return Path(settings.MODELS_DIR).resolve()
    return (Path(__file__).parent.parent.parent / "models").resolve()


MODELS_DIR = _get_models_dir()

# ─── Severity mapping ─────────────────────────────────────────────

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


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


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
    """Build a standardized anomaly dict compatible with the pipeline format."""
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
        "detection_type": "ml_model",
        "model_metadata": model_metadata or {},
        "contributing_agents": [detected_by],
        "web_references": [],
        "agent_perspectives": [],
    }


# ═══════════════════════════════════════════════════════════════════
# XGBoost Anomaly Detector
# ═══════════════════════════════════════════════════════════════════

class XGBoostAnomalyDetector:
    name = "XGBoost Anomaly Detector"

    def __init__(self) -> None:
        self.available = False
        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}

        if not HAS_XGBOOST:
            logger.warning("XGBoost library not installed — detector unavailable")
            return

        model_path = MODELS_DIR / "xgboost_anomaly.json"
        scaler_path = MODELS_DIR / "xgboost_scaler.joblib"
        meta_path = MODELS_DIR / "xgboost_metadata.json"

        if not model_path.exists():
            logger.warning("XGBoost model not found at %s — skipping", model_path)
            return

        try:
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            self.metadata = _load_json(meta_path) or {}
            self.available = True
            logger.info("XGBoost detector loaded (features=%s, threshold=%s)",
                        len(self.metadata.get("feature_names", [])),
                        self.metadata.get("anomaly_threshold"))
        except Exception as e:
            logger.warning("Failed to load XGBoost model: %s", e)

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not self.available or self.model is None:
            return []

        feature_names = self.metadata.get("feature_names", [])
        threshold = self.metadata.get("anomaly_threshold", 0.5)

        # Select numeric columns and match to training features
        numeric_df = df.select_dtypes(include=["number"])
        available_features = [f for f in feature_names if f in numeric_df.columns]
        missing_features = [f for f in feature_names if f not in numeric_df.columns]

        if missing_features:
            logger.warning("XGBoost: missing features from data: %s", missing_features)

        if not available_features:
            logger.warning("XGBoost: no matching features found in data — skipping")
            return []

        X = numeric_df[available_features].copy()
        X = X.fillna(0)

        # Scale if scaler available
        if self.scaler is not None:
            try:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=available_features,
                    index=X.index,
                )
            except Exception as e:
                logger.warning("XGBoost scaler failed: %s — using raw data", e)
                X_scaled = X
        else:
            X_scaled = X

        # Run prediction
        try:
            dmatrix = xgb.DMatrix(X_scaled, feature_names=available_features)
            scores = self.model.predict(dmatrix)
        except Exception as e:
            logger.error("XGBoost prediction failed: %s", e)
            return []

        anomalies: List[Dict] = []
        for i, score in enumerate(scores):
            if score >= threshold:
                # Top contributing features: absolute deviation from column means
                row = X_scaled.iloc[i]
                col_means = X_scaled.mean()
                deviations = (row - col_means).abs().sort_values(ascending=False)
                top_features = deviations.head(3).index.tolist()

                severity = _score_to_severity(float(score))
                anomalies.append(_build_anomaly(
                    anomaly_type="ml_xgboost",
                    title=f"XGBoost anomaly at row {i}",
                    description=(
                        f"XGBoost classifier flagged row {i} with score {score:.3f} "
                        f"(threshold {threshold}). Top features: {', '.join(top_features)}"
                    ),
                    severity=severity,
                    confidence=float(min(score, 1.0)),
                    affected_fields=top_features,
                    index=i,
                    detected_by=self.name,
                    model_metadata={
                        "score": float(score),
                        "threshold": threshold,
                        "top_features": top_features,
                    },
                ))

        return anomalies


# ═══════════════════════════════════════════════════════════════════
# 1D-CNN Autoencoder Detector
# ═══════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class _Autoencoder(nn.Module):
        """Feedforward autoencoder matching the saved state_dict layout."""

        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, n_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))
else:
    _Autoencoder = None  # type: ignore[assignment,misc]


class CNNAutoencoderDetector:
    name = "1D-CNN Autoencoder"

    def __init__(self) -> None:
        self.available = False
        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}

        if not HAS_TORCH:
            logger.warning("torch not installed — CNN autoencoder detector unavailable")
            return

        model_path = MODELS_DIR / "cnn_autoencoder.pt"
        scaler_path = MODELS_DIR / "cnn_scaler.joblib"
        meta_path = MODELS_DIR / "cnn_metadata.json"

        if not model_path.exists():
            logger.warning("CNN autoencoder model not found at %s — skipping", model_path)
            return

        try:
            self.metadata = _load_json(meta_path) or {}
            n_features = self.metadata.get("training_info", {}).get("n_features", 17)
            self.model = _Autoencoder(n_features)
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            self.available = True
            logger.info("CNN autoencoder loaded (features=%s, threshold=%s)",
                        n_features,
                        self.metadata.get("anomaly_threshold"))
        except Exception as e:
            logger.warning("Failed to load CNN autoencoder: %s", e)

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not self.available or self.model is None:
            return []

        feature_names = self.metadata.get("feature_names", [])
        threshold = self.metadata.get("anomaly_threshold", 0.042)

        numeric_df = df.select_dtypes(include=["number"])
        available_features = [f for f in feature_names if f in numeric_df.columns]

        if not available_features:
            logger.warning("CNN: no matching features found in data — skipping")
            return []

        X = numeric_df[available_features].copy().fillna(0)

        # Scale
        if self.scaler is not None:
            try:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=available_features,
                    index=X.index,
                )
            except Exception as e:
                logger.warning("CNN scaler failed: %s — using raw data", e)
                X_scaled = X
        else:
            X_scaled = X

        # Run reconstruction (per-row autoencoder)
        try:
            input_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
            with torch.no_grad():
                reconstructed = self.model(input_tensor).numpy()
            mse_per_row = np.mean((X_scaled.values - reconstructed) ** 2, axis=1)
        except Exception as e:
            logger.error("CNN prediction failed: %s", e)
            return []

        anomalies: List[Dict] = []
        for i, mse in enumerate(mse_per_row):
            if mse >= threshold:
                # Per-feature reconstruction error
                feature_errors = (X_scaled.values[i] - reconstructed[i]) ** 2
                top_idx = np.argsort(feature_errors)[-3:][::-1]
                top_features = [available_features[j] for j in top_idx]

                # Normalize score to 0-1 range
                score = min(float(mse) / (threshold * 5), 1.0)
                severity = _score_to_severity(score)

                anomalies.append(_build_anomaly(
                    anomaly_type="ml_cnn_autoencoder",
                    title=f"CNN autoencoder anomaly at row {i}",
                    description=(
                        f"Autoencoder reconstruction error {mse:.4f} exceeds "
                        f"threshold {threshold} at row {i}. "
                        f"Top contributing features: {', '.join(top_features)}"
                    ),
                    severity=severity,
                    confidence=round(score, 4),
                    affected_fields=top_features,
                    index=i,
                    detected_by=self.name,
                    model_metadata={
                        "reconstruction_error": float(mse),
                        "threshold": threshold,
                        "top_features": top_features,
                        "feature_errors": {
                            available_features[j]: float(feature_errors[j])
                            for j in top_idx
                        },
                    },
                ))

        return anomalies


# ═══════════════════════════════════════════════════════════════════
# Logistic Regression Meta-Learner
# ═══════════════════════════════════════════════════════════════════

class LogisticRegressionMetaLearner:
    name = "Logistic Regression Baseline"

    def __init__(self) -> None:
        self.available = False
        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}

        model_path = MODELS_DIR / "logreg_model.joblib"
        scaler_path = MODELS_DIR / "logreg_scaler.joblib"
        meta_path = MODELS_DIR / "logreg_metadata.json"

        if not model_path.exists():
            logger.warning("Logistic Regression model not found at %s — skipping", model_path)
            return

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            self.metadata = _load_json(meta_path) or {}
            self.available = True
            logger.info("Logistic Regression detector loaded (features=%s, threshold=%s)",
                        len(self.metadata.get("feature_names", [])),
                        self.metadata.get("anomaly_threshold", 0.7))
        except Exception as e:
            logger.warning("Failed to load Logistic Regression model: %s", e)

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not self.available or self.model is None:
            return []

        feature_names = self.metadata.get("feature_names", [])
        threshold = self.metadata.get("anomaly_threshold", 0.7)

        numeric_df = df.select_dtypes(include=["number"])
        available_features = [f for f in feature_names if f in numeric_df.columns]

        if not available_features:
            logger.warning("LogReg: no matching features found in data — skipping")
            return []

        X = numeric_df[available_features].copy().fillna(0)

        # Scale
        if self.scaler is not None:
            try:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=available_features,
                    index=X.index,
                )
            except Exception as e:
                logger.warning("LogReg scaler failed: %s — using raw data", e)
                X_scaled = X
        else:
            X_scaled = X

        # Predict probabilities
        try:
            probas = self.model.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            logger.error("LogReg prediction failed: %s", e)
            return []

        # Feature importance from model coefficients
        try:
            coefs = np.abs(self.model.coef_[0])
            coef_order = np.argsort(coefs)[::-1]
        except Exception:
            coef_order = np.arange(len(available_features))

        anomalies: List[Dict] = []
        for i, prob in enumerate(probas):
            if prob >= threshold:
                top_idx = coef_order[:3]
                top_features = [available_features[j] for j in top_idx if j < len(available_features)]

                severity = _score_to_severity(float(prob))
                anomalies.append(_build_anomaly(
                    anomaly_type="ml_logistic_regression",
                    title=f"Logistic Regression anomaly at row {i}",
                    description=(
                        f"Logistic Regression flagged row {i} with probability {prob:.3f} "
                        f"(threshold {threshold}). Key features: {', '.join(top_features)}"
                    ),
                    severity=severity,
                    confidence=float(prob),
                    affected_fields=top_features,
                    index=i,
                    detected_by=self.name,
                    model_metadata={
                        "probability": float(prob),
                        "threshold": threshold,
                        "top_features": top_features,
                    },
                ))

        return anomalies


# ═══════════════════════════════════════════════════════════════════
# Isolation Forest (sklearn — self-training)
# ═══════════════════════════════════════════════════════════════════

class IsolationForestDetector:
    """Full sklearn Isolation Forest.  Fits on the incoming data, so no
    pre-trained model file is required.

    Key parameters auto-tuned via the data:
      - n_estimators: 200 trees (higher than default for better resolution)
      - contamination: auto (uses offset-based heuristic)
      - max_features: min(1.0, 8 / n_features) — limits tree depth for speed
    """

    name = "Isolation Forest (sklearn)"

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 max_anomalies: int = 50) -> None:
        self.available = HAS_SKLEARN
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_anomalies = max_anomalies

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not self.available:
            return []

        numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all").fillna(0)
        if numeric.empty or len(numeric) < 30 or numeric.shape[1] < 2:
            return []

        scaler = SkScaler()
        X = scaler.fit_transform(numeric.values)

        max_features = min(1.0, 8.0 / X.shape[1]) if X.shape[1] > 8 else 1.0

        iso = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )
        labels = iso.fit_predict(X)
        scores = -iso.score_samples(X)  # higher = more anomalous

        outlier_indices = np.where(labels == -1)[0]
        if len(outlier_indices) == 0:
            return []

        # Sort by anomaly score
        sorted_indices = outlier_indices[np.argsort(scores[outlier_indices])[::-1]]

        anomalies: List[Dict] = []
        for idx in sorted_indices[:self.max_anomalies]:
            raw_score = float(scores[idx])
            # Normalise to 0-1: scores typically range from 0.4 to 0.7+
            norm_score = min(max((raw_score - 0.5) * 4.0, 0.1), 1.0)
            severity = _score_to_severity(norm_score)

            # Top features
            point = X[idx]
            col_means = X.mean(axis=0)
            col_stds = X.std(axis=0)
            col_stds[col_stds == 0] = 1
            abs_z = np.abs((point - col_means) / col_stds)
            top_feat_idx = np.argsort(abs_z)[-3:][::-1]
            top_features = [numeric.columns[j] for j in top_feat_idx]

            anomalies.append(_build_anomaly(
                anomaly_type="ml_isolation_forest",
                title=f"Isolation Forest anomaly at row {idx}",
                description=(
                    f"sklearn IsolationForest ({self.n_estimators} trees) flagged "
                    f"row {idx} with anomaly score {raw_score:.4f}. "
                    f"Top features: {', '.join(top_features)}."
                ),
                severity=severity,
                confidence=round(norm_score, 4),
                affected_fields=top_features,
                index=int(idx),
                detected_by=self.name,
                model_metadata={
                    "anomaly_score": round(raw_score, 4),
                    "n_estimators": self.n_estimators,
                    "contamination": self.contamination,
                    "max_features": round(max_features, 4),
                    "top_features": top_features,
                },
            ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# One-Class SVM (self-training)
# ═══════════════════════════════════════════════════════════════════

class OneClassSVMDetector:
    """One-Class SVM for novelty detection.  Learns a decision boundary
    around the normal data using an RBF kernel.

    Parameters:
      - kernel: RBF (default) — captures non-linear boundaries
      - nu: upper bound on fraction of training errors (~contamination)
      - gamma: 'scale' (auto-tuned from data variance)
    """

    name = "One-Class SVM"

    def __init__(self, nu: float = 0.05, kernel: str = "rbf",
                 max_anomalies: int = 50) -> None:
        self.available = HAS_SKLEARN
        self.nu = nu
        self.kernel = kernel
        self.max_anomalies = max_anomalies

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not self.available:
            return []

        numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all").fillna(0)
        if numeric.empty or len(numeric) < 30 or numeric.shape[1] < 2:
            return []

        # Subsample if too large (SVM is O(n²))
        max_rows = 5000
        if len(numeric) > max_rows:
            sample_idx = np.random.RandomState(42).choice(len(numeric), max_rows, replace=False)
            sample_idx = np.sort(sample_idx)
            X_raw = numeric.iloc[sample_idx].values
            index_map = sample_idx
        else:
            X_raw = numeric.values
            index_map = np.arange(len(numeric))

        scaler = SkScaler()
        X = scaler.fit_transform(X_raw)

        try:
            svm = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma="scale")
            labels = svm.fit_predict(X)
            decision = svm.decision_function(X)  # negative = anomalous
        except Exception as exc:
            logger.warning("One-Class SVM failed: %s", exc)
            return []

        outlier_mask = labels == -1
        outlier_indices = np.where(outlier_mask)[0]
        if len(outlier_indices) == 0:
            return []

        # Score: more negative decision = more anomalous
        min_dec = decision.min()
        max_dec = decision.max()
        score_range = max_dec - min_dec if max_dec != min_dec else 1.0

        sorted_indices = outlier_indices[np.argsort(decision[outlier_indices])]

        anomalies: List[Dict] = []
        for idx in sorted_indices[:self.max_anomalies]:
            original_idx = int(index_map[idx])
            dec_val = float(decision[idx])
            norm_score = min(max((max_dec - dec_val) / score_range, 0.1), 1.0)
            severity = _score_to_severity(norm_score)

            point = X[idx]
            col_means = X.mean(axis=0)
            col_stds = X.std(axis=0)
            col_stds[col_stds == 0] = 1
            abs_z = np.abs((point - col_means) / col_stds)
            top_feat_idx = np.argsort(abs_z)[-3:][::-1]
            top_features = [numeric.columns[j] for j in top_feat_idx]

            anomalies.append(_build_anomaly(
                anomaly_type="ml_one_class_svm",
                title=f"One-Class SVM anomaly at row {original_idx}",
                description=(
                    f"One-Class SVM (kernel={self.kernel}, nu={self.nu}) flagged "
                    f"row {original_idx} with decision value {dec_val:.4f}. "
                    f"Top features: {', '.join(top_features)}."
                ),
                severity=severity,
                confidence=round(norm_score, 4),
                affected_fields=top_features,
                index=original_idx,
                detected_by=self.name,
                model_metadata={
                    "decision_value": round(dec_val, 4),
                    "kernel": self.kernel,
                    "nu": self.nu,
                    "top_features": top_features,
                },
            ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# Gaussian Mixture Model (self-training)
# ═══════════════════════════════════════════════════════════════════

class GMMDetector:
    """Fits a Gaussian Mixture Model and flags low-probability points.

    The number of components is auto-selected via BIC from candidates
    [2, 3, 5, 8] to balance complexity and fit.

    Points whose log-likelihood falls below a percentile threshold are
    flagged as anomalies.
    """

    name = "Gaussian Mixture Model"

    def __init__(self, contamination: float = 0.05, max_components: int = 8,
                 max_anomalies: int = 50) -> None:
        self.available = HAS_SKLEARN
        self.contamination = contamination
        self.max_components = max_components
        self.max_anomalies = max_anomalies

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not self.available:
            return []

        numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all").fillna(0)
        if numeric.empty or len(numeric) < 50 or numeric.shape[1] < 2:
            return []

        # Limit features
        if numeric.shape[1] > 12:
            variances = numeric.var().sort_values(ascending=False)
            numeric = numeric[variances.head(12).index]

        scaler = SkScaler()
        X = scaler.fit_transform(numeric.values)

        # Auto-select n_components via BIC
        best_gmm = None
        best_bic = np.inf
        candidates = [c for c in [2, 3, 5, self.max_components] if c <= len(X) // 5]
        if not candidates:
            candidates = [2]

        for n_comp in candidates:
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    random_state=42,
                    max_iter=200,
                    n_init=2,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception:
                continue

        if best_gmm is None:
            return []

        log_probs = best_gmm.score_samples(X)
        # Threshold: bottom contamination-percentile of log-likelihoods
        threshold = np.percentile(log_probs, self.contamination * 100)

        outlier_indices = np.where(log_probs < threshold)[0]
        if len(outlier_indices) == 0:
            return []

        sorted_indices = outlier_indices[np.argsort(log_probs[outlier_indices])]

        max_lp = log_probs.max()
        min_lp = log_probs.min()
        lp_range = max_lp - min_lp if max_lp != min_lp else 1.0

        anomalies: List[Dict] = []
        for idx in sorted_indices[:self.max_anomalies]:
            lp = float(log_probs[idx])
            norm_score = min(max((max_lp - lp) / lp_range, 0.1), 1.0)
            severity = _score_to_severity(norm_score)

            point = X[idx]
            col_means = X.mean(axis=0)
            col_stds = X.std(axis=0)
            col_stds[col_stds == 0] = 1
            abs_z = np.abs((point - col_means) / col_stds)
            top_feat_idx = np.argsort(abs_z)[-3:][::-1]
            top_features = [numeric.columns[j] for j in top_feat_idx]

            anomalies.append(_build_anomaly(
                anomaly_type="ml_gmm",
                title=f"GMM low-probability anomaly at row {idx}",
                description=(
                    f"Gaussian Mixture Model ({best_gmm.n_components} components, "
                    f"BIC={best_bic:.0f}) assigned log-likelihood {lp:.3f} to "
                    f"row {idx} (threshold {threshold:.3f}). "
                    f"Top features: {', '.join(top_features)}."
                ),
                severity=severity,
                confidence=round(norm_score, 4),
                affected_fields=top_features,
                index=int(idx),
                detected_by=self.name,
                model_metadata={
                    "log_likelihood": round(lp, 4),
                    "threshold": round(float(threshold), 4),
                    "n_components": best_gmm.n_components,
                    "bic": round(float(best_bic), 2),
                    "top_features": top_features,
                },
            ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# Kernel Density Estimation (KDE) Detector
# ═══════════════════════════════════════════════════════════════════

class KDEDetector:
    """Non-parametric density estimation using Gaussian KDE from sklearn.

    Unlike GMM, KDE makes no assumption about the number of components.
    Bandwidth is auto-tuned via Silverman's rule scaled by a grid search
    over a small set of multipliers.

    Points in low-density regions (bottom percentile) are flagged.
    """

    name = "KDE Anomaly Detector"

    def __init__(self, contamination: float = 0.05, max_anomalies: int = 50) -> None:
        self.available = HAS_SKLEARN
        self.contamination = contamination
        self.max_anomalies = max_anomalies

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        if not self.available:
            return []

        numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all").fillna(0)
        if numeric.empty or len(numeric) < 50 or numeric.shape[1] < 2:
            return []

        # Limit features for KDE (curse of dimensionality)
        if numeric.shape[1] > 8:
            variances = numeric.var().sort_values(ascending=False)
            numeric = numeric[variances.head(8).index]

        scaler = SkScaler()
        X = scaler.fit_transform(numeric.values)

        # Silverman's rule for bandwidth
        n, d = X.shape
        silverman_bw = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))

        # Quick grid search over bandwidth multipliers
        best_kde = None
        best_score = -np.inf
        for mult in [0.5, 1.0, 1.5, 2.0]:
            bw = silverman_bw * mult
            kde = KernelDensity(kernel="gaussian", bandwidth=bw)
            kde.fit(X)
            cv_score = kde.score(X)  # log-likelihood on training data
            if cv_score > best_score:
                best_score = cv_score
                best_kde = kde

        if best_kde is None:
            return []

        log_densities = best_kde.score_samples(X)
        threshold = np.percentile(log_densities, self.contamination * 100)

        outlier_indices = np.where(log_densities < threshold)[0]
        if len(outlier_indices) == 0:
            return []

        sorted_indices = outlier_indices[np.argsort(log_densities[outlier_indices])]

        max_ld = log_densities.max()
        min_ld = log_densities.min()
        ld_range = max_ld - min_ld if max_ld != min_ld else 1.0

        anomalies: List[Dict] = []
        for idx in sorted_indices[:self.max_anomalies]:
            ld = float(log_densities[idx])
            norm_score = min(max((max_ld - ld) / ld_range, 0.1), 1.0)
            severity = _score_to_severity(norm_score)

            point = X[idx]
            col_means = X.mean(axis=0)
            col_stds = X.std(axis=0)
            col_stds[col_stds == 0] = 1
            abs_z = np.abs((point - col_means) / col_stds)
            top_feat_idx = np.argsort(abs_z)[-3:][::-1]
            top_features = [numeric.columns[j] for j in top_feat_idx]

            anomalies.append(_build_anomaly(
                anomaly_type="ml_kde",
                title=f"KDE low-density anomaly at row {idx}",
                description=(
                    f"Kernel Density Estimation (bandwidth={best_kde.bandwidth:.4f}) "
                    f"assigned log-density {ld:.3f} to row {idx} "
                    f"(threshold {threshold:.3f}). "
                    f"Top features: {', '.join(top_features)}."
                ),
                severity=severity,
                confidence=round(norm_score, 4),
                affected_fields=top_features,
                index=int(idx),
                detected_by=self.name,
                model_metadata={
                    "log_density": round(ld, 4),
                    "threshold": round(float(threshold), 4),
                    "bandwidth": round(float(best_kde.bandwidth), 4),
                    "top_features": top_features,
                },
            ))
        return anomalies


# ═══════════════════════════════════════════════════════════════════
# ML Model Orchestrator
# ═══════════════════════════════════════════════════════════════════

class MLModelOrchestrator:
    """Runs all available ML detectors and collects results."""

    def __init__(self) -> None:
        # Pre-trained models (loaded from files)
        self.xgboost = XGBoostAnomalyDetector()
        self.cnn = CNNAutoencoderDetector()
        self.logreg = LogisticRegressionMetaLearner()
        # Self-training models (fit on incoming data)
        self.isolation_forest = IsolationForestDetector()
        self.one_class_svm = OneClassSVMDetector()
        self.gmm = GMMDetector()
        self.kde = KDEDetector()
        self.detectors = [
            self.xgboost, self.cnn, self.logreg,
            self.isolation_forest, self.one_class_svm,
            self.gmm, self.kde,
        ]

    @property
    def models_available(self) -> Dict[str, bool]:
        return {d.name: d.available for d in self.detectors}

    @property
    def any_available(self) -> bool:
        return any(d.available for d in self.detectors)

    async def run_all(self, records: List[Dict]) -> Dict[str, Any]:
        """Run all available ML models concurrently and collect results."""
        df = pd.DataFrame(records)
        if df.empty:
            return {
                "anomalies": [],
                "model_statuses": [],
                "total_findings": 0,
                "models_available": self.models_available,
            }

        # Run available models concurrently in thread pool (CPU-bound)
        tasks = []
        task_names = []
        for detector in self.detectors:
            if detector.available:
                tasks.append(asyncio.to_thread(detector.detect, df))
                task_names.append(detector.name)

        if not tasks:
            return {
                "anomalies": [],
                "model_statuses": [
                    {"model": d.name, "status": "unavailable", "findings": 0, "elapsed_seconds": 0}
                    for d in self.detectors
                ],
                "total_findings": 0,
                "models_available": self.models_available,
            }

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_anomalies: List[Dict] = []
        model_statuses: List[Dict] = []

        # Map results back to detector names
        result_map: Dict[str, Any] = {}
        for name, result in zip(task_names, results):
            result_map[name] = result

        for detector in self.detectors:
            if not detector.available:
                model_statuses.append({
                    "model": detector.name,
                    "status": "unavailable",
                    "findings": 0,
                    "elapsed_seconds": 0,
                })
                continue

            result = result_map.get(detector.name)
            if isinstance(result, Exception):
                logger.error("ML model '%s' failed: %s", detector.name, result)
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

ml_orchestrator = MLModelOrchestrator()
