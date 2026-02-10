"""
Baseline & Historical Analysis Store for UAIE

Captures periodic snapshots of system statistics and analysis results,
enabling trend-over-time views, historical comparisons, and baseline
deviation alerts.  Persists to JSON files — additive feature, safe to
remove.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import sanitize_for_json


class BaselineStore:
    """
    File-based store for historical snapshots of system statistics.

    Each system gets a JSON file containing an array of snapshot entries:
    [
      {
        "id": "snap-20240115120000",
        "timestamp": "2024-01-15T12:00:00",
        "health_score": 85.5,
        "record_count": 1500,
        "field_stats": { "temperature": { "mean": 72.3, ... }, ... },
        "anomaly_count": 3,
        "anomaly_summary": { "critical": 0, "high": 1, "medium": 2, ... },
      },
      ...
    ]
    """

    MAX_SNAPSHOTS = 100

    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR", "/app/data")
            if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
                data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self.baseline_dir = Path(data_dir) / "baselines"
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, system_id: str) -> Path:
        return self.baseline_dir / f"{system_id}.json"

    def _load(self, system_id: str) -> List[Dict]:
        with self._lock:
            p = self._path(system_id)
            if not p.exists():
                return []
            try:
                return json.loads(p.read_text())
            except Exception:
                return []

    def _save(self, system_id: str, snapshots: List[Dict]):
        with self._lock:
            p = self._path(system_id)
            # Keep only the most recent snapshots
            if len(snapshots) > self.MAX_SNAPSHOTS:
                snapshots = snapshots[-self.MAX_SNAPSHOTS:]
            p.write_text(json.dumps(snapshots, indent=2, default=str))

    # ── Public API ────────────────────────────────────────────────

    def capture_snapshot(
        self,
        system_id: str,
        system: Dict,
        records: List[Dict],
        analysis: Optional[Dict] = None,
    ) -> Dict:
        """
        Capture a point-in-time snapshot of the system's data statistics.
        Optionally includes analysis summary if an analysis was just run.
        """
        now = datetime.utcnow()
        snap_id = f"snap-{now.strftime('%Y%m%d%H%M%S')}"

        snapshot: Dict[str, Any] = {
            "id": snap_id,
            "timestamp": now.isoformat(),
            "health_score": system.get("health_score"),
            "record_count": len(records),
            "field_count": 0,
            "field_stats": {},
            "anomaly_count": 0,
            "anomaly_summary": {},
        }

        if records:
            df = pd.DataFrame(records)
            snapshot["field_count"] = len(df.columns)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            field_stats = {}
            for col in numeric_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                field_stats[col] = sanitize_for_json({
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "median": float(s.median()),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                    "null_pct": round(float(df[col].isna().sum() / len(df) * 100), 1),
                })
            snapshot["field_stats"] = field_stats

        if analysis:
            anomalies = analysis.get("anomalies", [])
            snapshot["anomaly_count"] = len(anomalies)
            severity_counts: Dict[str, int] = {}
            for a in anomalies:
                sev = a.get("severity", "info")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            snapshot["anomaly_summary"] = severity_counts

        snapshots = self._load(system_id)
        snapshots.append(snapshot)
        self._save(system_id, snapshots)

        return snapshot

    def get_history(
        self,
        system_id: str,
        limit: int = 50,
    ) -> List[Dict]:
        """Get historical snapshots, most recent first."""
        snapshots = self._load(system_id)
        return list(reversed(snapshots[-limit:]))

    def get_latest(self, system_id: str) -> Optional[Dict]:
        """Get the most recent snapshot."""
        snapshots = self._load(system_id)
        return snapshots[-1] if snapshots else None

    def get_baseline(self, system_id: str) -> Optional[Dict]:
        """
        Compute an aggregate baseline from all historical snapshots.
        Returns average values and ranges for each tracked field.
        """
        snapshots = self._load(system_id)
        if not snapshots:
            return None

        # Aggregate health scores
        health_scores = [s["health_score"] for s in snapshots if s.get("health_score") is not None]
        anomaly_counts = [s.get("anomaly_count", 0) for s in snapshots]

        baseline: Dict[str, Any] = {
            "system_id": system_id,
            "snapshot_count": len(snapshots),
            "first_snapshot": snapshots[0]["timestamp"],
            "last_snapshot": snapshots[-1]["timestamp"],
            "health_score": {
                "mean": round(float(np.mean(health_scores)), 1) if health_scores else None,
                "min": round(float(np.min(health_scores)), 1) if health_scores else None,
                "max": round(float(np.max(health_scores)), 1) if health_scores else None,
                "std": round(float(np.std(health_scores)), 2) if len(health_scores) > 1 else 0,
            },
            "anomaly_count": {
                "mean": round(float(np.mean(anomaly_counts)), 1),
                "min": int(np.min(anomaly_counts)),
                "max": int(np.max(anomaly_counts)),
            },
            "field_baselines": {},
        }

        # Aggregate per-field baselines
        all_fields: Dict[str, List[Dict]] = {}
        for snap in snapshots:
            for field, stats in snap.get("field_stats", {}).items():
                if field not in all_fields:
                    all_fields[field] = []
                all_fields[field].append(stats)

        for field, history in all_fields.items():
            means = [h["mean"] for h in history if h.get("mean") is not None]
            stds = [h["std"] for h in history if h.get("std") is not None]
            if not means:
                continue
            baseline["field_baselines"][field] = sanitize_for_json({
                "mean_of_means": round(float(np.mean(means)), 4),
                "std_of_means": round(float(np.std(means)), 4) if len(means) > 1 else 0,
                "avg_std": round(float(np.mean(stds)), 4) if stds else 0,
                "range_of_means": [round(float(np.min(means)), 4), round(float(np.max(means)), 4)],
                "snapshots_with_data": len(means),
            })

        return baseline

    def compare_to_baseline(
        self,
        system_id: str,
        current_records: List[Dict],
    ) -> Dict:
        """
        Compare current data against the historical baseline.
        Returns per-field deviation scores.
        """
        baseline = self.get_baseline(system_id)
        if not baseline:
            return {"status": "no_baseline", "message": "No historical data available for comparison."}

        if not current_records:
            return {"status": "no_data", "message": "No current data to compare."}

        df = pd.DataFrame(current_records)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        deviations = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue

            field_bl = baseline.get("field_baselines", {}).get(col)
            if not field_bl:
                continue

            current_mean = float(s.mean())
            baseline_mean = field_bl.get("mean_of_means", 0)
            baseline_std = field_bl.get("std_of_means", 0)

            if baseline_std > 0:
                z_deviation = abs(current_mean - baseline_mean) / baseline_std
            elif baseline_mean != 0:
                z_deviation = abs(current_mean - baseline_mean) / abs(baseline_mean) * 3
            else:
                z_deviation = 0

            pct_change = 0
            if baseline_mean != 0:
                pct_change = (current_mean - baseline_mean) / abs(baseline_mean) * 100

            status = "normal"
            if z_deviation > 3:
                status = "critical_deviation"
            elif z_deviation > 2:
                status = "significant_deviation"
            elif z_deviation > 1:
                status = "minor_deviation"

            deviations.append(sanitize_for_json({
                "field": col,
                "current_mean": round(current_mean, 4),
                "baseline_mean": round(baseline_mean, 4),
                "baseline_std": round(baseline_std, 4),
                "z_deviation": round(z_deviation, 2),
                "pct_change": round(pct_change, 1),
                "status": status,
            }))

        deviations.sort(key=lambda x: x.get("z_deviation", 0), reverse=True)

        critical = sum(1 for d in deviations if d["status"] == "critical_deviation")
        significant = sum(1 for d in deviations if d["status"] == "significant_deviation")

        return {
            "status": "compared",
            "snapshot_count": baseline.get("snapshot_count", 0),
            "baseline_period": f"{baseline.get('first_snapshot', '?')} — {baseline.get('last_snapshot', '?')}",
            "critical_deviations": critical,
            "significant_deviations": significant,
            "fields_compared": len(deviations),
            "deviations": deviations,
        }

    def clear(self, system_id: str):
        """Delete all snapshots for a system."""
        with self._lock:
            p = self._path(system_id)
            if p.exists():
                p.unlink()


baseline_store = BaselineStore()
