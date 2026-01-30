"""
Anomaly Feedback Storage Service

Persists user feedback on anomalies (relevant, false positive, already known).
Provides adaptive threshold suggestions based on accumulated feedback.
This is an additive feature module - removing it does not affect core analysis.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class FeedbackStore:
    """
    File-based persistent storage for anomaly feedback.
    Thread-safe with in-memory cache backed by JSON files.
    """

    def __init__(self, data_dir: str = None):
        import os

        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR", "/app/data")
            if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
                data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

        self.feedback_dir = Path(data_dir) / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all feedback files into memory cache."""
        with self._lock:
            for feedback_file in self.feedback_dir.glob("*.json"):
                try:
                    system_id = feedback_file.stem
                    with open(feedback_file) as f:
                        self._cache[system_id] = json.load(f)
                except Exception as e:
                    print(f"Error loading feedback {feedback_file}: {e}")

    def _persist(self, system_id: str) -> None:
        """Write cached feedback for a system to disk."""
        feedback_file = self.feedback_dir / f"{system_id}.json"
        entries = self._cache.get(system_id, [])
        with open(feedback_file, "w") as f:
            json.dump(entries, f, indent=2, default=str)

    def add_feedback(
        self,
        system_id: str,
        anomaly_id: str,
        anomaly_title: str,
        anomaly_type: str,
        severity: str,
        feedback_type: str,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record user feedback for a specific anomaly.

        Args:
            system_id: The system this anomaly belongs to.
            anomaly_id: Unique anomaly identifier.
            anomaly_title: Human-readable anomaly title.
            anomaly_type: The anomaly detection type.
            severity: Anomaly severity level.
            feedback_type: One of 'relevant', 'false_positive', 'already_known'.
            comment: Optional user comment.

        Returns:
            The created feedback entry.
        """
        entry = {
            "id": f"fb-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{anomaly_id[:8]}",
            "system_id": system_id,
            "anomaly_id": anomaly_id,
            "anomaly_title": anomaly_title,
            "anomaly_type": anomaly_type,
            "severity": severity,
            "feedback_type": feedback_type,
            "comment": comment,
            "created_at": datetime.utcnow().isoformat(),
        }

        with self._lock:
            if system_id not in self._cache:
                self._cache[system_id] = []
            self._cache[system_id].append(entry)
            self._persist(system_id)

        return entry

    def get_feedback(
        self,
        system_id: str,
        anomaly_id: Optional[str] = None,
        feedback_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve feedback entries for a system, optionally filtered.

        Args:
            system_id: System to query.
            anomaly_id: Filter by specific anomaly.
            feedback_type: Filter by feedback type.

        Returns:
            List of matching feedback entries (newest first).
        """
        with self._lock:
            entries = list(self._cache.get(system_id, []))

        if anomaly_id:
            entries = [e for e in entries if e["anomaly_id"] == anomaly_id]
        if feedback_type:
            entries = [e for e in entries if e["feedback_type"] == feedback_type]

        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return entries

    def get_feedback_summary(self, system_id: str) -> Dict[str, Any]:
        """
        Compute summary statistics from all feedback for a system.

        Returns:
            Dict with counts by type, false-positive rate, and top false-positive patterns.
        """
        entries = self.get_feedback(system_id)

        total = len(entries)
        by_type = {"relevant": 0, "false_positive": 0, "already_known": 0}
        by_anomaly_type: Dict[str, Dict[str, int]] = {}

        for entry in entries:
            ft = entry.get("feedback_type", "")
            if ft in by_type:
                by_type[ft] += 1

            at = entry.get("anomaly_type", "unknown")
            if at not in by_anomaly_type:
                by_anomaly_type[at] = {"relevant": 0, "false_positive": 0, "already_known": 0}
            if ft in by_anomaly_type[at]:
                by_anomaly_type[at][ft] += 1

        false_positive_rate = by_type["false_positive"] / total if total > 0 else 0.0

        # Top false-positive anomaly types
        fp_patterns = []
        for atype, counts in by_anomaly_type.items():
            type_total = sum(counts.values())
            if type_total > 0:
                fp_patterns.append({
                    "anomaly_type": atype,
                    "total": type_total,
                    "false_positive_count": counts["false_positive"],
                    "false_positive_rate": counts["false_positive"] / type_total,
                })
        fp_patterns.sort(key=lambda p: p["false_positive_rate"], reverse=True)

        return {
            "system_id": system_id,
            "total_feedback": total,
            "by_type": by_type,
            "false_positive_rate": round(false_positive_rate, 3),
            "false_positive_patterns": fp_patterns[:10],
            "confidence_score": round(1.0 - false_positive_rate, 3) if total >= 5 else None,
        }

    def get_adaptive_thresholds(self, system_id: str) -> Dict[str, Any]:
        """
        Suggest threshold adjustments based on feedback patterns.

        If a particular anomaly type has a high false-positive rate,
        recommend raising the detection threshold for it.

        Returns:
            Dict with per-anomaly-type threshold adjustment suggestions.
        """
        summary = self.get_feedback_summary(system_id)
        adjustments: List[Dict[str, Any]] = []

        for pattern in summary.get("false_positive_patterns", []):
            fp_rate = pattern["false_positive_rate"]
            total = pattern["total"]

            # Only suggest adjustments with sufficient data (>=3 feedback entries)
            if total < 3:
                continue

            if fp_rate >= 0.7:
                adjustments.append({
                    "anomaly_type": pattern["anomaly_type"],
                    "recommendation": "significantly_raise_threshold",
                    "reason": f"{fp_rate:.0%} false positive rate over {total} reviews",
                    "suggested_sensitivity_change": -0.3,
                })
            elif fp_rate >= 0.4:
                adjustments.append({
                    "anomaly_type": pattern["anomaly_type"],
                    "recommendation": "raise_threshold",
                    "reason": f"{fp_rate:.0%} false positive rate over {total} reviews",
                    "suggested_sensitivity_change": -0.15,
                })
            elif fp_rate <= 0.1 and pattern["false_positive_count"] == 0:
                adjustments.append({
                    "anomaly_type": pattern["anomaly_type"],
                    "recommendation": "threshold_optimal",
                    "reason": f"No false positives over {total} reviews",
                    "suggested_sensitivity_change": 0.0,
                })

        return {
            "system_id": system_id,
            "total_feedback": summary["total_feedback"],
            "adjustments": adjustments,
            "overall_confidence": summary.get("confidence_score"),
        }

    def delete_feedback(self, system_id: str, feedback_id: str) -> bool:
        """Delete a specific feedback entry."""
        with self._lock:
            entries = self._cache.get(system_id, [])
            original_len = len(entries)
            self._cache[system_id] = [e for e in entries if e.get("id") != feedback_id]

            if len(self._cache[system_id]) < original_len:
                self._persist(system_id)
                return True
            return False


# Global instance
feedback_store = FeedbackStore()
