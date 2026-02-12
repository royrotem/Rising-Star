"""
ML Model Inference Service

Loads pre-trained XGBoost, 1D-CNN Autoencoder, and Logistic Regression models
from backend/models/ and runs inference only — no on-the-fly training.

Models are trained in a separate project and transferred as serialized files.
Each detector gracefully degrades: if its model files are missing the detector
reports itself as unavailable and the pipeline continues without it.
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

from ..core.config import settings


def _get_models_dir() -> Path:
    if settings.MODELS_DIR:
        return Path(settings.MODELS_DIR)
    return Path(__file__).parent.parent.parent / "models"


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
# ML Model Orchestrator
# ═══════════════════════════════════════════════════════════════════

class MLModelOrchestrator:
    """Runs all available ML detectors and collects results."""

    def __init__(self) -> None:
        self.xgboost = XGBoostAnomalyDetector()
        self.cnn = CNNAutoencoderDetector()
        self.logreg = LogisticRegressionMetaLearner()
        self.detectors = [self.xgboost, self.cnn, self.logreg]

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
