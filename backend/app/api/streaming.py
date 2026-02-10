"""
SSE Streaming Analysis Endpoint

Streams real-time progress events while running the analysis pipeline.
This is an additive feature module — removing it does not affect the
existing POST /systems/{id}/analyze endpoint.
"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..services.data_store import data_store
from ..services.analysis_engine import analysis_engine
from ..services.ai_agents import orchestrator as ai_orchestrator, ALL_AGENT_NAMES
from ..services.agentic_analyzers import agentic_orchestrator
from ..services.recommendation import build_data_profile
from ..utils import (
    sanitize_for_json,
    anomaly_to_dict,
    merge_ai_anomalies,
    save_analysis,
)
from .app_settings import get_ai_settings, get_anthropic_api_key

logger = logging.getLogger("uaie.streaming")

router = APIRouter(prefix="/systems", tags=["Streaming"])


def _sse_event(event: str, data: Any) -> str:
    """Format an SSE event string."""
    payload = json.dumps(sanitize_for_json(data), default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get("/{system_id}/analyze-stream")
async def analyze_system_stream(
    system_id: str,
    agents: Optional[str] = Query(None, description="Comma-separated list of agent names to run. If omitted, all agents run."),
    mode: Optional[str] = Query("standard", description="Analysis mode: 'standard' (25 LLM agents) or 'agentic' (5 tool-using AI agents)"),
):
    """
    Stream analysis progress via Server-Sent Events.

    Query params:
      - agents: comma-separated agent names (e.g. "Statistical Analyst,Domain Expert")
      - mode: 'standard' (25 specialized LLM prompts) or 'agentic' (5 AI agents with tool-use)

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

    # Determine analysis mode
    use_agentic = mode == "agentic"

    async def event_stream():
        try:
            t0 = time.time()
            logger.info("=" * 60)
            logger.info("ANALYZE-STREAM START | system_id=%s | mode=%s", system_id, mode)
            logger.info("  system: %s", system.get("name", "?"))
            logger.info("  system_type: %s", system.get("system_type", "?"))
            logger.info("  selected_agents: %s", selected_agents or "ALL")
            logger.info("  analysis_mode: %s", "AGENTIC (tool-using AI)" if use_agentic else "STANDARD (25 LLM agents)")

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

            logger.info("[Stage 1] Loaded %d records, %d sources, %d schema fields",
                        len(records) if records else 0,
                        len(sources) if sources else 0,
                        len(discovered_schema) if discovered_schema else 0)

            if not records:
                logger.warning("[Stage 1] No records — aborting analysis")
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
            logger.info("[Stage 2] Running 6-layer rule-based analysis engine...")
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
            logger.info("[Stage 2] Rule engine complete: %d anomalies, health_score=%.1f",
                        len(anomalies), result.health_score or 0)

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
            agentic_result = None
            agent_statuses: List[Dict] = []

            api_key = get_anthropic_api_key()
            ai_enabled = ai_cfg.get("enable_ai_agents", True)
            logger.info("[Stage 3] AI config: ai_enabled=%s, api_key=%s (len=%d), web_grounding=%s, agentic_mode=%s",
                        ai_enabled,
                        "YES" if api_key else "NO",
                        len(api_key) if api_key else 0,
                        ai_cfg.get("enable_web_grounding", True),
                        use_agentic)

            if ai_enabled:
                # Build schema context for agents
                schema_context = ""
                if discovered_schema:
                    field_names = [f.get("name", "") for f in discovered_schema if f.get("name")]
                    schema_context = f"Fields: {', '.join(field_names)}"
                meta = system.get("metadata", {})
                if meta.get("description"):
                    schema_context += f"\nDescription: {meta['description']}"

                if use_agentic:
                    # ═══════════════════════════════════════════════════════════
                    # AGENTIC MODE: Tool-using AI agents that explore data
                    # ═══════════════════════════════════════════════════════════
                    agentic_agent_names = ["Data Explorer", "Pattern Miner", "Anomaly Hunter",
                                          "Time Series Investigator", "Root Cause Investigator"]
                    agent_count = len(agentic_agent_names)
                    logger.info("[Stage 3] AGENTIC MODE: Launching %d tool-using AI agents...", agent_count)

                    yield _sse_event("stage", {
                        "stage": "agentic_agents",
                        "message": f"Launching {agent_count} agentic AI investigators (tool-use enabled)...",
                        "progress": 55,
                    })

                    try:
                        t_ai_start = time.time()
                        agentic_result = await agentic_orchestrator.run_analysis(
                            records=records,
                            system_type=system_type,
                            system_name=system_name,
                            schema_context=schema_context,
                            selected_agents=selected_agents,
                        )
                        t_ai_elapsed = round(time.time() - t_ai_start, 2)

                        logger.info("[Stage 3] Agentic orchestrator finished in %.2fs", t_ai_elapsed)
                        if agentic_result:
                            logger.info("[Stage 3] Agentic result: agents_used=%s, total_findings=%d",
                                        agentic_result.get("agents_used", []),
                                        agentic_result.get("total_findings", 0))

                        # Emit per-agent statuses
                        agentic_statuses = agentic_result.get("agent_statuses", []) if agentic_result else []
                        for idx, st in enumerate(agentic_statuses):
                            agent_name = st.get("agent", f"Agent {idx+1}")
                            agent_status = st.get("status", "unknown")
                            agent_findings = st.get("findings", 0)
                            logger.info("[Stage 3]   agentic '%s': status=%s, findings=%d",
                                        agent_name, agent_status, agent_findings)
                            yield _sse_event("agent_complete", {
                                "agent": f"🤖 {agent_name}",
                                "status": agent_status,
                                "findings": agent_findings,
                                "perspective": "Tool-using AI agent",
                                "agentic": True,
                            })
                            await asyncio.sleep(0.1)

                        agent_statuses = agentic_statuses

                        # Convert agentic findings to anomalies format
                        if agentic_result and agentic_result.get("findings"):
                            for finding in agentic_result["findings"]:
                                anomalies.append({
                                    "id": f"agentic_{hash(finding['title'])}",
                                    "title": finding.get("title", "Untitled"),
                                    "description": finding.get("description", ""),
                                    "severity": finding.get("severity", "medium"),
                                    "impact_score": {"critical": 95, "high": 80, "medium": 60, "low": 40, "info": 20}.get(finding.get("severity", "medium"), 60),
                                    "affected_fields": finding.get("affected_fields", []),
                                    "evidence": finding.get("evidence", {}),
                                    "recommendation": finding.get("recommendation", ""),
                                    "detected_by": f"Agentic: {finding.get('agent', 'Unknown')}",
                                    "detection_type": "agentic_ai",
                                    "iterations_used": finding.get("iterations_used", 0),
                                })

                        yield _sse_event("stage", {
                            "stage": "agentic_complete",
                            "message": f"Agentic AI found {agentic_result.get('total_findings', 0) if agentic_result else 0} findings",
                            "progress": 85,
                        })

                    except Exception as e:
                        logger.error("[Stage 3] Agentic agents EXCEPTION: %s: %s", type(e).__name__, e)
                        logger.error(traceback.format_exc())
                        yield _sse_event("stage", {
                            "stage": "agentic_error",
                            "message": f"Agentic agents failed: {e}",
                            "progress": 85,
                        })
                        agent_statuses = [{"agent": "Agentic Orchestrator", "status": "error", "error": str(e)}]

                else:
                    # ═══════════════════════════════════════════════════════════
                    # STANDARD MODE: 25 specialized LLM prompt agents
                    # ═══════════════════════════════════════════════════════════
                    agent_count = len(selected_agents) if selected_agents else len(ALL_AGENT_NAMES)
                    logger.info("[Stage 3] STANDARD MODE: Launching %d AI agents...", agent_count)
                    yield _sse_event("stage", {
                        "stage": "ai_agents",
                        "message": f"Launching AI agent swarm ({agent_count} specialized agents)...",
                        "progress": 55,
                    })

                    try:
                        data_profile = build_data_profile(records, discovered_schema)
                        logger.info("[Stage 3] Data profile built: %d fields, %d records, %d sample rows",
                                    data_profile.get("field_count", 0),
                                    data_profile.get("record_count", 0),
                                    len(data_profile.get("sample_rows", [])))

                        metadata_context = ""
                        meta = system.get("metadata", {})
                        if meta.get("description"):
                            metadata_context = meta["description"]

                        t_ai_start = time.time()
                        ai_result = await ai_orchestrator.run_analysis(
                            system_id=system_id,
                            system_type=system_type,
                            system_name=system_name,
                            data_profile=data_profile,
                            metadata_context=metadata_context,
                            enable_web_grounding=ai_cfg.get("enable_web_grounding", True),
                            selected_agents=selected_agents,
                        )
                        t_ai_elapsed = round(time.time() - t_ai_start, 2)

                        logger.info("[Stage 3] AI orchestrator finished in %.2fs", t_ai_elapsed)
                        if ai_result:
                            logger.info("[Stage 3] AI result: ai_powered=%s, agents_used=%s, raw_findings=%d, unified=%d",
                                        ai_result.get("ai_powered"),
                                        ai_result.get("agents_used", []),
                                        ai_result.get("total_findings_raw", 0),
                                        ai_result.get("total_anomalies_unified", 0))

                        # Emit per-agent statuses
                        ai_agent_statuses = ai_result.get("agent_statuses", []) if ai_result else []
                        for idx, st in enumerate(ai_agent_statuses):
                            agent_name = st.get("agent", f"Agent {idx+1}")
                            agent_status = st.get("status", "unknown")
                            agent_findings = st.get("findings", 0)
                            logger.info("[Stage 3]   agent '%s': status=%s, findings=%d%s",
                                        agent_name, agent_status, agent_findings,
                                        f", error={st.get('error')}" if st.get("error") else "")
                            yield _sse_event("agent_complete", {
                                "agent": agent_name,
                                "status": agent_status,
                                "findings": agent_findings,
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
                        logger.error("[Stage 3] AI agents EXCEPTION: %s: %s", type(e).__name__, e)
                        logger.error(traceback.format_exc())
                        yield _sse_event("stage", {
                            "stage": "ai_agents_error",
                            "message": f"AI agents failed (using rule-based only): {e}",
                            "progress": 85,
                        })
                        agent_statuses = [{"agent": "AI Orchestrator", "status": "error", "error": str(e)}]
            else:
                logger.info("[Stage 3] AI agents DISABLED — skipping")
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
                    "mode": "agentic" if use_agentic else "standard",
                    "ai_powered": (agentic_result is not None) if use_agentic else (ai_result.get("ai_powered", False) if ai_result else False),
                    "agents_used": (agentic_result.get("agents_used", []) if agentic_result else []) if use_agentic else (ai_result.get("agents_used", []) if ai_result else []),
                    "agent_statuses": agent_statuses,
                    "total_findings_raw": (agentic_result.get("total_findings", 0) if agentic_result else 0) if use_agentic else (ai_result.get("total_findings_raw", 0) if ai_result else 0),
                    "total_anomalies_unified": (agentic_result.get("total_findings", 0) if agentic_result else 0) if use_agentic else (ai_result.get("total_anomalies_unified", 0) if ai_result else 0),
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

            logger.info("=" * 60)
            logger.info("ANALYZE-STREAM COMPLETE | system_id=%s | elapsed=%.2fs", system_id, elapsed)
            logger.info("  health_score: %.1f", analysis_result.get("health_score", 0) or 0)
            logger.info("  total_anomalies: %d", len(anomalies))
            logger.info("  ai_powered: %s", analysis_result.get("ai_analysis", {}).get("ai_powered", False))
            logger.info("  agents_used: %s", analysis_result.get("ai_analysis", {}).get("agents_used", []))
            logger.info("  raw_findings: %d, unified: %d",
                        analysis_result.get("ai_analysis", {}).get("total_findings_raw", 0),
                        analysis_result.get("ai_analysis", {}).get("total_anomalies_unified", 0))
            logger.info("=" * 60)

            yield _sse_event("stage", {
                "stage": "complete",
                "message": f"Analysis complete in {elapsed}s",
                "progress": 100,
            })

            yield _sse_event("result", analysis_result)

        except Exception as exc:
            logger.error("ANALYZE-STREAM EXCEPTION: %s: %s", type(exc).__name__, exc)
            logger.error(traceback.format_exc())
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
