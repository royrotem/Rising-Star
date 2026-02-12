#!/usr/bin/env python3
"""
Evaluate ML anomaly-detection models against labelled demo data.

Dataset
-------
This is a UAV anomaly detection dataset based on a software-in-the-loop
simulation environment, and the dataset contains some of the anomaly logs
of the UAV.

    Label 0 — UAV is in a normal state
    Label 1 — GPS anomaly
    Label 2 — Accelerometer anomaly
    Label 3 — Engine anomaly
    Label 4 — RC (remote control) anomaly

All three models are *binary* anomaly detectors (normal vs anomaly), so
labels 1-4 are collapsed into a single "anomaly" class for evaluation.

Usage
-----
    python evaluate_models.py                       # defaults
    python evaluate_models.py --data path/to.csv    # custom CSV
    python evaluate_models.py --models-dir ./models # custom models path
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "demo_data.csv"
DEFAULT_MODELS = ROOT / "backend" / "models"

LABEL_NAMES = {
    0: "Normal",
    1: "GPS anomaly",
    2: "Accelerometer anomaly",
    3: "Engine anomaly",
    4: "RC anomaly",
}

# ── helpers ────────────────────────────────────────────────────────


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print accuracy, precision, recall, F1, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {acc:.4f}  ({int(acc * len(y_true))}/{len(y_true)})")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print()
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                Pred Normal  Pred Anomaly")
    print(f"  Actual Normal   {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"  Actual Anomaly  {cm[1][0]:>6}        {cm[1][1]:>6}")
    print()
    print(f"  Detailed report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0
        )
    )


# ── model runners ─────────────────────────────────────────────────


def run_xgboost(models_dir: Path, X: pd.DataFrame) -> np.ndarray | None:
    """Return binary predictions (0/1) or None if unavailable."""
    try:
        import xgboost as xgb
    except ImportError:
        print("[SKIP] xgboost not installed")
        return None

    model_path = models_dir / "xgboost_anomaly.json"
    scaler_path = models_dir / "xgboost_scaler.joblib"
    meta_path = models_dir / "xgboost_metadata.json"

    if not model_path.exists():
        print(f"[SKIP] XGBoost model not found at {model_path}")
        return None

    meta = load_metadata(meta_path)
    threshold = meta.get("anomaly_threshold", 0.5)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    X_in = X.copy()
    if scaler is not None:
        X_in = pd.DataFrame(scaler.transform(X_in), columns=X.columns, index=X.index)

    dmatrix = xgb.DMatrix(X_in, feature_names=list(X.columns))
    scores = booster.predict(dmatrix)
    preds = (scores >= threshold).astype(int)

    print(f"  Threshold: {threshold:.4f}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Flagged as anomaly: {preds.sum()}/{len(preds)}")
    return preds


def run_cnn_autoencoder(models_dir: Path, X: pd.DataFrame) -> np.ndarray | None:
    """Return binary predictions (0/1) or None if unavailable."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[SKIP] torch not installed")
        return None

    model_path = models_dir / "cnn_autoencoder.pt"
    scaler_path = models_dir / "cnn_scaler.joblib"
    meta_path = models_dir / "cnn_metadata.json"

    if not model_path.exists():
        print(f"[SKIP] CNN model not found at {model_path}")
        return None

    meta = load_metadata(meta_path)
    threshold = meta.get("anomaly_threshold", 0.042)
    n_features = meta.get("training_info", {}).get("n_features", X.shape[1])

    # Reconstruct the autoencoder architecture
    class Autoencoder(nn.Module):
        def __init__(self, n_feat: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32)
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, n_feat)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

    model = Autoencoder(n_features)
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    X_in = X.copy()
    if scaler is not None:
        X_in = pd.DataFrame(scaler.transform(X_in), columns=X.columns, index=X.index)

    input_tensor = torch.tensor(X_in.values, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(input_tensor).numpy()

    mse_per_row = np.mean((X_in.values - reconstructed) ** 2, axis=1)
    preds = (mse_per_row >= threshold).astype(int)

    print(f"  Threshold: {threshold:.4f}")
    print(f"  MSE range: [{mse_per_row.min():.4f}, {mse_per_row.max():.4f}]")
    print(f"  Flagged as anomaly: {preds.sum()}/{len(preds)}")
    return preds


def run_logreg(models_dir: Path, X: pd.DataFrame) -> np.ndarray | None:
    """Return binary predictions (0/1) or None if unavailable."""
    model_path = models_dir / "logreg_model.joblib"
    scaler_path = models_dir / "logreg_scaler.joblib"
    meta_path = models_dir / "logreg_metadata.json"

    if not model_path.exists():
        print(f"[SKIP] LogReg model not found at {model_path}")
        return None

    meta = load_metadata(meta_path)
    threshold = meta.get("anomaly_threshold", 0.7)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    X_in = X.copy()
    if scaler is not None:
        X_in = pd.DataFrame(scaler.transform(X_in), columns=X.columns, index=X.index)

    probas = model.predict_proba(X_in)[:, 1]
    preds = (probas >= threshold).astype(int)

    print(f"  Threshold: {threshold:.4f}")
    print(f"  Probability range: [{probas.min():.4f}, {probas.max():.4f}]")
    print(f"  Flagged as anomaly: {preds.sum()}/{len(preds)}")
    return preds


# ── main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UAV anomaly detection models")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to labelled CSV")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS, help="Path to models/")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")
    print(f"Models directory: {args.models_dir}\n")

    # Show label distribution
    label_col = "labels" if "labels" in df.columns else "Label"
    print("Label distribution:")
    for label_val, count in df[label_col].value_counts().sort_index().items():
        desc = LABEL_NAMES.get(label_val, f"Unknown ({label_val})")
        print(f"  {label_val} — {desc}: {count}")
    print()

    # Binary ground truth: 0 = normal, 1 = any anomaly
    y_true = (df[label_col] != 0).astype(int).values

    print(f"Binary split: {(y_true == 0).sum()} normal, {(y_true == 1).sum()} anomaly\n")

    # ── Select features ───────────────────────────────────────────
    meta = load_metadata(args.models_dir / "xgboost_metadata.json")
    feature_names = meta.get("feature_names", [])
    if not feature_names:
        feature_names = [c for c in df.columns if c not in ("timestamp", label_col)]

    X = df[feature_names].copy().fillna(0)
    print(f"Features ({len(feature_names)}): {', '.join(feature_names)}\n")

    # ── Run each model ────────────────────────────────────────────
    runners = [
        ("XGBoost Anomaly Detector", run_xgboost),
        ("1D-CNN Autoencoder", run_cnn_autoencoder),
        ("Logistic Regression", run_logreg),
    ]

    all_preds: dict[str, np.ndarray] = {}

    for name, runner_fn in runners:
        print(f"Running {name}...")
        preds = runner_fn(args.models_dir, X)
        if preds is not None:
            all_preds[name] = preds
            print_metrics(name, y_true, preds)

    # ── Ensemble (majority vote) ──────────────────────────────────
    if len(all_preds) >= 2:
        stacked = np.column_stack(list(all_preds.values()))
        ensemble_preds = (stacked.mean(axis=1) >= 0.5).astype(int)
        print_metrics(f"Ensemble — majority vote ({len(all_preds)} models)", y_true, ensemble_preds)

    # ── Per-anomaly-type breakdown ────────────────────────────────
    if len(all_preds) > 0:
        print(f"\n{'=' * 60}")
        print("  Per-anomaly-type recall (detection rate)")
        print(f"{'=' * 60}")
        anomaly_types = sorted(df[label_col].unique())
        header = f"  {'Type':<25}" + "".join(f"{n:>15}" for n in all_preds) + f"{'Samples':>10}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for label_val in anomaly_types:
            mask = df[label_col] == label_val
            n = mask.sum()
            desc = LABEL_NAMES.get(label_val, f"Label {label_val}")
            row = f"  {desc:<25}"
            for name, preds in all_preds.items():
                if label_val == 0:
                    # For normal: show how many were correctly predicted as normal
                    rate = (preds[mask] == 0).mean()
                else:
                    # For anomaly: show how many were correctly flagged
                    rate = (preds[mask] == 1).mean()
                row += f"{rate:>14.1%} "
            row += f"{n:>9}"
            print(row)
        print()


if __name__ == "__main__":
    main()
