"""
SSE Streaming Analysis Endpoint

Streams real-time progress events while running the analysis pipeline.
This is an additive feature module — removing it does not affect the
existing POST /systems/{id}/analyze endpoint.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..services.data_store import data_store
from ..services.analysis_engine import analysis_engine
from ..services.ai_agents import orchestrator as ai_orchestrator, ALL_AGENT_NAMES
from ..services.recommendation import build_data_profile
from ..utils import (
    sanitize_for_json,
    anomaly_to_dict,
    merge_ai_anomalies,
    save_analysis,
)
from .app_settings import get_ai_settings

router = APIRouter(prefix="/systems", tags=["Streaming"])


def _sse_event(event: str, data: Any) -> str:
    """Format an SSE event string."""
    payload = json.dumps(sanitize_for_json(data), default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get("/{system_id}/analyze-stream")
async def analyze_system_stream(
    system_id: str,
    agents: Optional[str] = Query(None, description="Comma-separated list of agent names to run. If omitted, all agents run."),
):
    """
    Stream analysis progress via Server-Sent Events.

    Query params:
      - agents: comma-separated agent names (e.g. "Statistical Analyst,Domain Expert")

    Events emitted:
      - stage: { stage, message, progress }  — progress updates
      - layer_complete: { layer, anomaly_count }  — a detection layer finished
      - agent_complete: { agent, status, findings }  — an AI agent finished
      - result: { ...full analysis result }  — final result
      - error: { message }  — on failure
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    # Parse agent selection
    selected_agents: Optional[List[str]] = None
    if agents:
        selected_agents = [a.strip() for a in agents.split(",") if a.strip()]

    async def event_stream():
        try:
            t0 = time.time()

            # ── Stage 1: Loading data ──
            yield _sse_event("stage", {
                "stage": "loading_data",
                "message": "Loading system data...",
                "progress": 5,
            })

            records = data_store.get_ingested_records(system_id, limit=50000)
            sources = data_store.get_data_sources(system_id)
            discovered_schema = system.get("discovered_schema", [])
            system_type = system.get("system_type", "industrial")
            system_name = system.get("name", "Unknown System")

            if not records:
                yield _sse_event("error", {
                    "message": "No data ingested. Upload data before running analysis.",
                })
                return

            yield _sse_event("stage", {
                "stage": "data_loaded",
                "message": f"Loaded {len(records)} records from {len(sources)} sources",
                "progress": 10,
            })

            # Small delay so the client can render
            await asyncio.sleep(0.05)

            # ── Stage 2: Rule-based analysis engine (6 layers) ──
            yield _sse_event("stage", {
                "stage": "rule_engine",
                "message": "Running 6-layer anomaly detection engine...",
                "progress": 15,
            })

            result = await analysis_engine.analyze(
                system_id=system_id,
                system_type=system_type,
                records=records,
                discovered_schema=discovered_schema,
                metadata=system.get("metadata", {}),
            )

            anomalies = [anomaly_to_dict(a) for a in result.anomalies]

            # Report layer results
            layer_names = [
                "Statistical Outlier Detection",
                "Threshold-Based Detection",
                "Trend Analysis",
                "Correlation Analysis",
                "Pattern Detection",
                "Rate of Change Analysis",
            ]
            for i, layer_name in enumerate(layer_names):
                yield _sse_event("layer_complete", {
                    "layer": layer_name,
                    "layer_index": i + 1,
                    "total_layers": 6,
                    "anomaly_count": len(anomalies),
                })
                # Small stagger for visual effect
                await asyncio.sleep(0.05)

            yield _sse_event("stage", {
                "stage": "rule_engine_complete",
                "message": f"Rule engine found {len(anomalies)} anomalies",
                "progress": 50,
            })

            # ── Stage 3: AI multi-agent analysis ──
            ai_cfg = get_ai_settings()
            ai_result = None
            agent_statuses: List[Dict] = []

            if ai_cfg.get("enable_ai_agents", True):
                agent_count = len(selected_agents) if selected_agents else 13
                yield _sse_event("stage", {
                    "stage": "ai_agents",
                    "message": f"Launching AI agent swarm ({agent_count} specialized agents)...",
                    "progress": 55,
                })

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
                        selected_agents=selected_agents,
                    )

                    # Emit per-agent statuses
                    ai_agent_statuses = ai_result.get("agent_statuses", []) if ai_result else []
                    for idx, st in enumerate(ai_agent_statuses):
                        yield _sse_event("agent_complete", {
                            "agent": st.get("agent", f"Agent {idx+1}"),
                            "status": st.get("status", "unknown"),
                            "findings": st.get("findings", 0),
                            "perspective": st.get("perspective", ""),
                        })
                        await asyncio.sleep(0.05)

                    agent_statuses = ai_agent_statuses

                    merge_ai_anomalies(anomalies, ai_result)

                    yield _sse_event("stage", {
                        "stage": "ai_agents_complete",
                        "message": f"AI agents contributed {ai_result.get('total_findings_raw', 0)} raw findings",
                        "progress": 85,
                    })

                except Exception as e:
                    yield _sse_event("stage", {
                        "stage": "ai_agents_error",
                        "message": f"AI agents failed (using rule-based only): {e}",
                        "progress": 85,
                    })
                    agent_statuses = [{"agent": "AI Orchestrator", "status": "error", "error": str(e)}]
            else:
                yield _sse_event("stage", {
                    "stage": "ai_agents_skipped",
                    "message": "AI agents disabled — using rule-based analysis only",
                    "progress": 85,
                })
                agent_statuses = [{"agent": "AI Orchestrator", "status": "disabled", "findings": 0}]

            # ── Stage 4: Finalizing ──
            yield _sse_event("stage", {
                "stage": "finalizing",
                "message": "Building final report...",
                "progress": 90,
            })

            anomalies.sort(key=lambda a: a.get("impact_score", 0), reverse=True)

            analysis_result = {
                "system_id": system_id,
                "timestamp": result.analyzed_at,
                "health_score": result.health_score,
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

            # Update system with analysis results
            updates: Dict[str, Any] = {}
            if result.health_score is not None:
                updates["health_score"] = result.health_score
            if anomalies:
                updates["status"] = "anomaly_detected"
                updates["anomaly_count"] = len(anomalies)
            else:
                updates["status"] = "healthy"
                updates["anomaly_count"] = 0
            updates["last_analysis_at"] = datetime.now().isoformat()
            if updates:
                data_store.update_system(system_id, updates)

            save_analysis(system_id, analysis_result)

            elapsed = round(time.time() - t0, 2)

            yield _sse_event("stage", {
                "stage": "complete",
                "message": f"Analysis complete in {elapsed}s",
                "progress": 100,
            })

            yield _sse_event("result", analysis_result)

        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/available-agents")
async def list_available_agents():
    """Return the list of available AI agents with their descriptions."""
    agent_info = []
    for agent in ai_orchestrator.agents:
        agent_info.append({
            "name": agent.name,
            "perspective": agent.perspective,
        })
    return {"agents": agent_info}
