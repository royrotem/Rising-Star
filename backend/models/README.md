# ML Model Storage & Configuration

Pre-trained ML models used by the UAIE anomaly detection pipeline. Models are trained externally and loaded at startup for inference only.

---

## Directory Layout

```
backend/models/
  .gitkeep                    # Keeps directory in git
  xgboost_anomaly.json        # XGBoost model (native JSON format)
  xgboost_scaler.joblib       # StandardScaler for XGBoost
  xgboost_metadata.json       # Feature names, threshold, training info
  cnn_autoencoder.pt           # PyTorch autoencoder state_dict
  cnn_scaler.joblib            # StandardScaler for CNN
  cnn_metadata.json            # Window size, feature names, threshold
  logreg_model.joblib          # LogReg sklearn model
  logreg_scaler.joblib         # StandardScaler for LogReg
  logreg_metadata.json         # Feature names, threshold
```

Each model consists of three files:
- **Model file** — the serialized trained model
- **Scaler file** (`.joblib`) — a `StandardScaler` used to normalize input features
- **Metadata file** (`*_metadata.json`) — feature names, anomaly threshold, and training info

---

## Metadata Format

Every `*_metadata.json` file follows this schema:

```json
{
  "feature_names": ["temperature", "vibration", "pressure"],
  "anomaly_threshold": 0.042,
  "training_info": {
    "n_samples": 50000,
    "n_features": 12,
    "trained_at": "2026-02-10T..."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `feature_names` | `string[]` | Ordered list of feature columns the model expects |
| `anomaly_threshold` | `float` | Score above which a data point is flagged as anomalous |
| `training_info.n_samples` | `int` | Number of samples used during training |
| `training_info.n_features` | `int` | Number of features used during training |
| `training_info.trained_at` | `string` | ISO 8601 timestamp of when the model was trained |

---

## Configuration

The models directory path is configurable via the `MODELS_DIR` environment variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `backend/models/` | Absolute path to the directory containing model files |

### Local Development

No configuration needed. The pipeline defaults to `backend/models/` relative to the project root.

### Docker / Production

Set `MODELS_DIR` in your `.env` file or pass it as an environment variable:

```bash
# .env
MODELS_DIR=/app/models
```

Or in `docker-compose.yml`:

```yaml
services:
  backend:
    environment:
      - MODELS_DIR=/app/models
```

The default Docker setup (`COPY backend/ .` in Dockerfile and `./backend:/app` bind mount in docker-compose) already makes `backend/models/` available at `/app/models/` inside the container, so no extra volume mounts are needed for the default model set.

To use an external model volume:

```yaml
services:
  backend:
    environment:
      - MODELS_DIR=/models
    volumes:
      - ./my-trained-models:/models:ro
```

---

## Graceful Degradation

Each detector independently checks for its model files at startup:

- If a model file is **missing**, the detector logs a warning and marks itself as unavailable.
- If a scaler file is missing, the detector runs without scaling (uses raw data).
- If a metadata file is missing, the detector uses built-in defaults.
- The pipeline **continues** with whichever detectors are available.

Check model availability at startup via the backend logs:

```
INFO  XGBoost detector loaded (features=12, threshold=0.042)
WARN  CNN autoencoder model not found at /app/models/cnn_autoencoder.pt — skipping
INFO  Logistic Regression detector loaded (features=12, threshold=0.7)
```

---

## Adding a New Model

1. Train the model externally and export:
   - Model file in the appropriate format (`.json`, `.pt`, `.joblib`)
   - A fitted `StandardScaler` saved with `joblib.dump()`
   - A `*_metadata.json` with feature names and threshold

2. Place all three files in the models directory.

3. Restart the backend — the orchestrator picks up new files automatically.

---

## Git & Ignored Files

Trained model files **are tracked** in git (they are small, <50 MB total). The following temporary files produced by training tools are excluded via `.gitignore`:

```
backend/models/*.tmp
backend/models/*.bak
backend/models/checkpoint
```
