"""
Scheduled Auto-Analysis Service (Watchdog Mode)

Runs analysis pipelines on configurable intervals per system.
Uses asyncio background tasks — no external dependencies required.

Persistence: schedule configs stored as JSON in {data_dir}/schedules/.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import get_data_dir, sanitize_for_json, save_analysis

logger = logging.getLogger("uaie.scheduler")


# ─── Schedule model ───────────────────────────────────────────────────

VALID_INTERVALS = ("1h", "6h", "12h", "24h", "7d")

INTERVAL_SECONDS = {
    "1h": 3600,
    "6h": 6 * 3600,
    "12h": 12 * 3600,
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
}


def _schedules_dir() -> Path:
    d = get_data_dir() / "schedules"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_schedule(system_id: str) -> Optional[Dict[str, Any]]:
    path = _schedules_dir() / f"{system_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_schedule(system_id: str, schedule: Dict[str, Any]) -> None:
    path = _schedules_dir() / f"{system_id}.json"
    path.write_text(json.dumps(schedule, indent=2, default=str))


def _delete_schedule(system_id: str) -> bool:
    path = _schedules_dir() / f"{system_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def _list_schedules() -> List[Dict[str, Any]]:
    schedules = []
    for path in _schedules_dir().glob("*.json"):
        try:
            schedules.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return schedules


# ─── Scheduler engine ─────────────────────────────────────────────────

class AnalysisScheduler:
    """
    Background scheduler that periodically runs the analysis pipeline
    for systems with watchdog mode enabled.
    """

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        # Track per-system state so we know when the next run is due
        self._last_run: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background scheduler loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Watchdog scheduler started")

    async def stop(self) -> None:
        """Gracefully stop the scheduler."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Watchdog scheduler stopped")

    # ── Schedule CRUD (called from API layer) ─────────────────────

    def get_schedule(self, system_id: str) -> Optional[Dict[str, Any]]:
        return _load_schedule(system_id)

    def list_schedules(self) -> List[Dict[str, Any]]:
        return _list_schedules()

    def set_schedule(
        self,
        system_id: str,
        enabled: bool,
        interval: str = "24h",
    ) -> Dict[str, Any]:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {VALID_INTERVALS}")

        existing = _load_schedule(system_id) or {}
        schedule = {
            "system_id": system_id,
            "enabled": enabled,
            "interval": interval,
            "created_at": existing.get("created_at", datetime.utcnow().isoformat()),
            "updated_at": datetime.utcnow().isoformat(),
            "last_run_at": existing.get("last_run_at"),
            "last_run_status": existing.get("last_run_status"),
            "run_count": existing.get("run_count", 0),
        }
        _save_schedule(system_id, schedule)

        # Reset last run tracking so the next cycle picks it up immediately
        # if newly enabled, or clears it if disabled
        if not enabled:
            self._last_run.pop(system_id, None)

        return schedule

    def delete_schedule(self, system_id: str) -> bool:
        self._last_run.pop(system_id, None)
        return _delete_schedule(system_id)

    # ── Background loop ───────────────────────────────────────────

    async def _loop(self) -> None:
        """Main scheduler loop — checks every 30 seconds for due schedules."""
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                logger.error("Scheduler tick error: %s", exc)
            await asyncio.sleep(30)

    async def _tick(self) -> None:
        """Single scheduler tick — run any due analyses."""
        schedules = _list_schedules()
        now = time.time()

        for sched in schedules:
            if not sched.get("enabled"):
                continue

            system_id = sched["system_id"]
            interval_key = sched.get("interval", "24h")
            interval_secs = INTERVAL_SECONDS.get(interval_key, 86400)

            last = self._last_run.get(system_id, 0)
            if now - last < interval_secs:
                continue

            # Time to run
            logger.info("Watchdog: running scheduled analysis for system %s", system_id)
            self._last_run[system_id] = now
            await self._run_analysis(system_id, sched)

    async def _run_analysis(self, system_id: str, sched: Dict) -> None:
        """Run the full analysis pipeline for a system (reuses existing engine)."""
        # Late imports to avoid circular dependency
        from ..services.data_store import data_store
        from ..services.analysis_engine import analysis_engine
        from ..services.ai_agents import orchestrator as ai_orchestrator
        from ..services.recommendation import build_data_profile
        from ..api.app_settings import get_ai_settings
        from ..utils import anomaly_to_dict, merge_ai_anomalies

        system = data_store.get_system(system_id)
        if not system:
            logger.warning("Watchdog: system %s not found, skipping", system_id)
            self._update_schedule_status(system_id, sched, "error", "System not found")
            return

        records = data_store.get_ingested_records(system_id, limit=50000)
        if not records:
            logger.info("Watchdog: system %s has no data, skipping", system_id)
            self._update_schedule_status(system_id, sched, "skipped", "No data ingested")
            return

        try:
            sources = data_store.get_data_sources(system_id)
            discovered_schema = system.get("discovered_schema", [])
            system_type = system.get("system_type", "industrial")
            system_name = system.get("name", "Unknown System")

            # ── Rule-based analysis ──
            result = await analysis_engine.analyze(
                system_id=system_id,
                system_type=system_type,
                records=records,
                discovered_schema=discovered_schema,
                metadata=system.get("metadata", {}),
            )
            anomalies = [anomaly_to_dict(a) for a in result.anomalies]

            # ── AI multi-agent analysis ──
            ai_cfg = get_ai_settings()
            ai_result = None
            agent_statuses = []

            if ai_cfg.get("enable_ai_agents", True):
                try:
                    data_profile = build_data_profile(records, discovered_schema)
                    metadata_context = ""
                    meta = system.get("metadata", {})
                    if meta.get("description"):
                        metadata_context = meta["description"]

                    ai_result = await ai_orchestrator.run_analysis(
                        system_id=system_id,
                        system_type=system_type,
                        system_name=system_name,
                        data_profile=data_profile,
                        metadata_context=metadata_context,
                        enable_web_grounding=ai_cfg.get("enable_web_grounding", True),
                    )
                    agent_statuses = ai_result.get("agent_statuses", []) if ai_result else []
                    merge_ai_anomalies(anomalies, ai_result)
                except Exception as e:
                    logger.warning("Watchdog: AI agents failed for %s: %s", system_id, e)
                    agent_statuses = [{"agent": "AI Orchestrator", "status": "error", "error": str(e)}]

            anomalies.sort(key=lambda a: a.get("impact_score", 0), reverse=True)

            analysis_result = {
                "system_id": system_id,
                "timestamp": result.analyzed_at,
                "health_score": result.health_score,
                "triggered_by": "watchdog",
                "data_analyzed": {
                    "record_count": len(records),
                    "source_count": len(sources),
                    "field_count": (
                        len(set(f.get("name", "") for f in discovered_schema))
                        if discovered_schema else 0
                    ),
                },
                "anomalies": anomalies,
                "engineering_margins": result.engineering_margins,
                "blind_spots": result.blind_spots,
                "correlation_analysis": result.correlation_matrix,
                "trend_analysis": result.trend_analysis,
                "insights": result.insights,
                "insights_summary": result.summary,
                "recommendations": result.recommendations,
                "ai_analysis": {
                    "ai_powered": ai_result.get("ai_powered", False) if ai_result else False,
                    "agents_used": ai_result.get("agents_used", []) if ai_result else [],
                    "agent_statuses": agent_statuses,
                    "total_findings_raw": ai_result.get("total_findings_raw", 0) if ai_result else 0,
                    "total_anomalies_unified": ai_result.get("total_anomalies_unified", 0) if ai_result else 0,
                },
            }

            # Update system health & status
            updates: Dict[str, Any] = {}
            if result.health_score is not None:
                updates["health_score"] = result.health_score
            if anomalies:
                updates["status"] = "anomaly_detected"
                updates["anomaly_count"] = len(anomalies)
            else:
                updates["status"] = "healthy"
                updates["anomaly_count"] = 0
            updates["last_analysis_at"] = datetime.utcnow().isoformat()
            data_store.update_system(system_id, updates)

            save_analysis(system_id, analysis_result)

            logger.info(
                "Watchdog: analysis complete for %s — health=%.1f, anomalies=%d",
                system_id,
                result.health_score if result.health_score is not None else 0,
                len(anomalies),
            )
            self._update_schedule_status(system_id, sched, "success")

        except Exception as exc:
            logger.error("Watchdog: analysis failed for %s: %s", system_id, exc)
            self._update_schedule_status(system_id, sched, "error", str(exc))

    def _update_schedule_status(
        self,
        system_id: str,
        sched: Dict,
        status: str,
        error_msg: str = "",
    ) -> None:
        sched["last_run_at"] = datetime.utcnow().isoformat()
        sched["last_run_status"] = status
        sched["run_count"] = sched.get("run_count", 0) + 1
        if error_msg:
            sched["last_error"] = error_msg
        else:
            sched.pop("last_error", None)
        _save_schedule(system_id, sched)


# Global singleton
scheduler = AnalysisScheduler()
