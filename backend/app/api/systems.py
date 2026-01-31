"""
Systems API Endpoints

Handles system management, data ingestion, and analysis.
"""

import json
import math
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ..services.data_store import data_store
from ..services.ingestion import IngestionService
from ..services.analysis_engine import analysis_engine
from ..services.ai_agents import orchestrator as ai_orchestrator
from ..services.recommendation import (
    build_data_profile,
    enrich_fields_with_context,
    generate_system_recommendation,
    titles_overlap,
)
from .app_settings import get_ai_settings
from .schemas import (
    AnalysisRequest,
    ConversationQuery,
    FieldConfirmation,
    SystemCreate,
    SystemResponse,
)

router = APIRouter(prefix="/systems", tags=["Systems"])


# ─── JSON serialization helpers ───────────────────────────────────────

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ─── Service instances ────────────────────────────────────────────────

ingestion_service = IngestionService()

# ─── Demo mode ────────────────────────────────────────────────────────

_DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"


def _init_demo_systems() -> None:
    """Initialize demo systems for demonstration purposes."""
    demo_systems = [
        {
            "id": "demo-1",
            "name": "Fleet Vehicle Alpha",
            "system_type": "vehicle",
            "serial_number": "VH-2024-001",
            "model": "EV-X1",
            "metadata": {"manufacturer": "UAIE Demo", "year": 2024},
            "status": "anomaly_detected",
            "health_score": 87.5,
            "discovered_schema": {},
            "confirmed_fields": {},
            "is_demo": True,
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "demo-2",
            "name": "Robot Arm Unit 7",
            "system_type": "robot",
            "serial_number": "RA-2024-007",
            "model": "ARM-6DOF",
            "metadata": {"manufacturer": "UAIE Demo", "year": 2024},
            "status": "active",
            "health_score": 94.2,
            "discovered_schema": {},
            "confirmed_fields": {},
            "is_demo": True,
            "created_at": "2024-01-02T00:00:00Z",
        },
        {
            "id": "demo-3",
            "name": "Medical Scanner MRI-3",
            "system_type": "medical_device",
            "serial_number": "MRI-2024-003",
            "model": "MRI-X500",
            "metadata": {"manufacturer": "UAIE Demo", "year": 2024},
            "status": "active",
            "health_score": 99.1,
            "discovered_schema": {},
            "confirmed_fields": {},
            "is_demo": True,
            "created_at": "2024-01-03T00:00:00Z",
        },
    ]

    for system in demo_systems:
        if not data_store.get_system(system["id"]):
            data_store.create_system(system)


if _DEMO_MODE:
    _init_demo_systems()


# ═══════════════════════════════════════════════════════════════════════
# File Analysis & System Creation
# ═══════════════════════════════════════════════════════════════════════

@router.post("/analyze-files")
async def analyze_files(files: List[UploadFile] = File(...)):
    """
    Analyze uploaded files to discover schema and suggest system configuration.

    Processes all uploaded files, discovers relationships between them,
    and provides AI recommendations for system name, type, and description.
    Records are stored temporarily and can be associated with a system
    using the returned ``analysis_id``.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    analysis_id = str(uuid.uuid4())

    all_discovered_fields: List[Dict] = []
    all_confirmation_requests: List[Dict] = []
    all_metadata: List[Dict] = []
    all_records: List[Dict] = []
    total_records = 0
    file_summaries: List[Dict] = []
    file_records_map: Dict[str, List[Dict]] = {}

    for file in files:
        try:
            result = await ingestion_service.ingest_file(
                file_content=file.file,
                filename=file.filename,
                system_id="temp_analysis",
                source_name=file.filename,
            )

            records = result.get("sample_records", [])
            file_records_map[file.filename] = records
            all_records.extend(records)

            for field in result.get("discovered_fields", []):
                field["source_file"] = file.filename
                all_discovered_fields.append(field)

            all_confirmation_requests.extend(result.get("confirmation_requests", []))
            total_records += result.get("record_count", 0)

            metadata_info = result.get("metadata_info", {})
            if metadata_info.get("dataset_description"):
                all_metadata.append(metadata_info)

            file_summaries.append({
                "filename": file.filename,
                "record_count": result.get("record_count", 0),
                "fields": [f.get("name", "") for f in result.get("discovered_fields", [])],
                "field_types": {
                    f.get("name", ""): f.get("inferred_type", "")
                    for f in result.get("discovered_fields", [])
                },
                "metadata": metadata_info,
                "relationships": result.get("relationships", []),
            })

            file.file.seek(0)

        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            continue

    data_store.store_temp_analysis(
        analysis_id=analysis_id,
        records=all_records,
        file_summaries=file_summaries,
        discovered_fields=all_discovered_fields,
        file_records_map=file_records_map,
    )

    # Second pass: enrich fields with combined context from all files
    combined_field_descriptions: Dict[str, str] = {}
    combined_context_texts: List[str] = []

    for meta in all_metadata:
        if meta.get("field_descriptions"):
            combined_field_descriptions.update(meta["field_descriptions"])
        if meta.get("context_texts"):
            combined_context_texts.extend(meta["context_texts"])
        elif meta.get("dataset_description"):
            combined_context_texts.append(meta["dataset_description"])

    all_discovered_fields = enrich_fields_with_context(
        all_discovered_fields,
        combined_field_descriptions,
        combined_context_texts,
    )

    recommendation = generate_system_recommendation(
        file_summaries, all_discovered_fields, all_metadata,
    )

    return {
        "status": "success",
        "analysis_id": analysis_id,
        "files_analyzed": len(files),
        "total_records": total_records,
        "discovered_fields": all_discovered_fields,
        "confirmation_requests": all_confirmation_requests,
        "recommendation": recommendation,
        "context_extracted": len(combined_context_texts) > 0,
        "fields_enriched": len(combined_field_descriptions),
    }


@router.post("/", response_model=SystemResponse)
async def create_system(system: SystemCreate):
    """Create a new monitored system.

    If ``analysis_id`` is provided, pre-analyzed data will be associated
    with this system automatically.
    """
    system_id = str(uuid.uuid4())

    system_data = {
        "id": system_id,
        "name": system.name,
        "system_type": system.system_type,
        "serial_number": system.serial_number,
        "model": system.model,
        "metadata": system.metadata or {},
        "status": "active",
        "health_score": 100.0,
        "discovered_schema": {},
        "confirmed_fields": {},
        "is_demo": False,
        "created_at": datetime.utcnow().isoformat(),
    }

    created_system = data_store.create_system(system_data)

    if system.analysis_id:
        moved = data_store.move_temp_to_system(system.analysis_id, system_id)
        if moved:
            created_system = data_store.get_system(system_id)

    return SystemResponse(**created_system)


# ═══════════════════════════════════════════════════════════════════════
# System CRUD
# ═══════════════════════════════════════════════════════════════════════

@router.get("/", response_model=List[SystemResponse])
async def list_systems(
    status: Optional[str] = None,
    system_type: Optional[str] = None,
    include_demo: bool = Query(default=True, description="Include demo systems in results"),
):
    """List all monitored systems."""
    systems = data_store.list_systems(include_demo=include_demo)

    if status:
        systems = [s for s in systems if s.get("status") == status]
    if system_type:
        systems = [s for s in systems if s.get("system_type") == system_type]

    return [SystemResponse(**s) for s in systems]


@router.get("/{system_id}", response_model=Dict[str, Any])
async def get_system(system_id: str):
    """Get detailed information about a system."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")
    return system


@router.delete("/{system_id}")
async def delete_system(system_id: str):
    """Delete a system and all its data."""
    success = data_store.delete_system(system_id)
    if not success:
        raise HTTPException(status_code=404, detail="System not found")
    return {"status": "deleted", "system_id": system_id}


# ═══════════════════════════════════════════════════════════════════════
# Data Ingestion & Schema Confirmation
# ═══════════════════════════════════════════════════════════════════════

@router.post("/{system_id}/ingest")
async def ingest_data(
    system_id: str,
    file: UploadFile = File(...),
    source_name: str = Query(default="uploaded_file"),
):
    """
    Ingest a data file and perform autonomous schema discovery.

    Implements the Zero-Knowledge Ingestion approach -- the system
    analyses the uploaded data "blind" and learns its structure.
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    try:
        result = await ingestion_service.ingest_file(
            file_content=file.file,
            filename=file.filename,
            system_id=system_id,
            source_name=source_name,
        )

        data_store.update_system(system_id, {
            "discovered_schema": result.get("discovered_fields", {}),
            "status": "data_ingested",
        })

        source_id = str(uuid.uuid4())
        data_store.store_ingested_data(
            system_id=system_id,
            source_id=source_id,
            source_name=source_name,
            records=result.get("sample_records", []),
            discovered_schema={
                "fields": result.get("discovered_fields", []),
                "relationships": result.get("relationships", []),
            },
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
            },
        )

        return {
            "status": "success",
            "source_id": source_id,
            "record_count": result.get("record_count"),
            "discovered_fields": result.get("discovered_fields"),
            "relationships": result.get("relationships"),
            "confirmation_requests": result.get("confirmation_requests"),
            "sample_records": result.get("sample_records", [])[:5],
            "message": "Data ingested. Please review the discovered schema and confirm field mappings.",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{system_id}/confirm-fields")
async def confirm_fields(
    system_id: str,
    confirmations: List[FieldConfirmation],
):
    """
    Human-in-the-Loop field confirmation.

    Engineers can confirm or correct the AI's schema inference.
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    confirmed = system.get("confirmed_fields", {})

    for conf in confirmations:
        entry = {
            "confirmed": True,
            "type": conf.confirmed_type,
            "unit": conf.confirmed_unit,
            "meaning": conf.confirmed_meaning,
            "confirmed_at": datetime.utcnow().isoformat(),
        }
        if not conf.is_correct:
            entry["corrected"] = True
        confirmed[conf.field_name] = entry

    data_store.update_system(system_id, {
        "confirmed_fields": confirmed,
        "status": "configured",
    })

    return {
        "status": "success",
        "confirmed_count": len(confirmations),
        "message": "Field mappings updated. The system will use these confirmations for future analysis.",
    }


# ═══════════════════════════════════════════════════════════════════════
# Data Retrieval
# ═══════════════════════════════════════════════════════════════════════

@router.get("/{system_id}/data")
async def get_system_data(
    system_id: str,
    source_id: Optional[str] = None,
    limit: int = Query(default=100, le=10000),
    offset: int = Query(default=0, ge=0),
):
    """Get ingested data records for a system."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id, source_id, limit, offset)

    return {
        "system_id": system_id,
        "source_id": source_id,
        "records": records,
        "count": len(records),
        "limit": limit,
        "offset": offset,
    }


@router.get("/{system_id}/statistics")
async def get_system_statistics(system_id: str):
    """Get statistics about a system's ingested data."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    stats = data_store.get_system_statistics(system_id)
    return {"system_id": system_id, "system_name": system.get("name"), **stats}


@router.get("/{system_id}/sources")
async def get_data_sources(system_id: str):
    """Get all data sources for a system."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    sources = data_store.get_data_sources(system_id)
    return {"system_id": system_id, "sources": sources, "count": len(sources)}


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

@router.get("/{system_id}/analysis")
async def get_saved_analysis(system_id: str):
    """Return the most recent saved analysis result, or 404 if none exists."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    from pathlib import Path
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    analysis_path = Path(data_dir) / "analyses" / f"{system_id}.json"

    if not analysis_path.exists():
        raise HTTPException(status_code=404, detail="No analysis found. Run analysis first.")

    try:
        return json.loads(analysis_path.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load analysis: {e}")


@router.post("/{system_id}/analyze")
async def analyze_system(system_id: str, request: AnalysisRequest):
    """
    Run comprehensive AI-powered analysis on a system.

    Triggers multi-layered analysis combining rule-based detection
    (statistical, threshold, trend, correlation, pattern, rate-of-change)
    with LLM-powered multi-agent AI analysis when enabled.
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id, limit=50000)
    sources = data_store.get_data_sources(system_id)
    discovered_schema = system.get("discovered_schema", [])
    system_type = system.get("system_type", "industrial")
    system_name = system.get("name", "Unknown System")

    # ── Rule-based analysis engine ──
    result = await analysis_engine.analyze(
        system_id=system_id,
        system_type=system_type,
        records=records,
        discovered_schema=discovered_schema,
        metadata=system.get("metadata", {}),
    )

    anomalies = [_anomaly_to_dict(a) for a in result.anomalies]

    # ── AI multi-agent analysis ──
    ai_result, agent_statuses = await _run_ai_analysis(
        system, records, discovered_schema, system_type, system_name, anomalies,
    )

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
    updates["last_analysis_at"] = datetime.utcnow().isoformat()
    if updates:
        data_store.update_system(system_id, updates)

    # Persist analysis result so the PDF report endpoint can find it
    try:
        from pathlib import Path
        data_dir = os.environ.get("DATA_DIR", "/app/data")
        if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        analyses_dir = Path(data_dir) / "analyses"
        analyses_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = analyses_dir / f"{system_id}.json"
        analysis_path.write_text(json.dumps(
            _sanitize_for_json(analysis_result), indent=2, default=str
        ))
    except Exception:
        pass

    return _sanitize_for_json(analysis_result)


def _anomaly_to_dict(a) -> Dict[str, Any]:
    """Convert a rule-based Anomaly dataclass to API dict format."""
    return {
        "id": a.id,
        "type": a.anomaly_type.value,
        "severity": a.severity.value,
        "title": a.title,
        "description": a.description,
        "affected_fields": [a.field_name] + a.related_fields,
        "natural_language_explanation": a.natural_language_explanation,
        "possible_causes": a.possible_causes,
        "recommendations": a.recommendations,
        "impact_score": a.impact_score,
        "confidence": a.confidence,
        "value": a.value,
        "expected_range": a.expected_range,
        "contributing_agents": ["Rule Engine"],
        "web_references": [],
        "agent_perspectives": [],
    }


async def _run_ai_analysis(
    system: Dict,
    records: List[Dict],
    discovered_schema: List[Dict],
    system_type: str,
    system_name: str,
    anomalies: List[Dict],
) -> tuple:
    """Run AI multi-agent analysis and merge findings into *anomalies* in-place.

    Returns ``(ai_result_dict | None, agent_statuses_list)``.
    """
    ai_cfg = get_ai_settings()

    if not ai_cfg.get("enable_ai_agents", True):
        return None, [{"agent": "AI Orchestrator", "status": "disabled", "findings": 0}]

    try:
        data_profile = build_data_profile(records, discovered_schema)

        metadata_context = ""
        meta = system.get("metadata", {})
        if meta.get("description"):
            metadata_context = meta["description"]

        ai_result = await ai_orchestrator.run_analysis(
            system_id=system["id"],
            system_type=system_type,
            system_name=system_name,
            data_profile=data_profile,
            metadata_context=metadata_context,
            enable_web_grounding=ai_cfg.get("enable_web_grounding", True),
        )

        _merge_ai_anomalies(anomalies, ai_result)

        agent_statuses = ai_result.get("agent_statuses", []) if ai_result else []
        return ai_result, agent_statuses

    except Exception as e:
        print(f"[Analysis] AI agent analysis failed (using rule-based only): {e}")
        return None, [{"agent": "AI Orchestrator", "status": "error", "error": str(e)}]


def _merge_ai_anomalies(anomalies: List[Dict], ai_result: Optional[Dict]) -> None:
    """Merge AI-found anomalies into the existing list, deduplicating by title overlap."""
    if not ai_result or not ai_result.get("anomalies"):
        return

    for ai_anomaly in ai_result["anomalies"]:
        is_duplicate = False
        for existing in anomalies:
            if titles_overlap(existing.get("title", ""), ai_anomaly.get("title", "")):
                existing.setdefault("contributing_agents", []).extend(
                    ai_anomaly.get("contributing_agents", [])
                )
                existing.setdefault("agent_perspectives", []).extend(
                    ai_anomaly.get("agent_perspectives", [])
                )
                existing.setdefault("web_references", []).extend(
                    ai_anomaly.get("web_references", [])
                )
                if ai_anomaly.get("confidence", 0) > existing.get("confidence", 0):
                    existing["confidence"] = ai_anomaly["confidence"]
                is_duplicate = True
                break

        if not is_duplicate:
            anomalies.append(ai_anomaly)


# ═══════════════════════════════════════════════════════════════════════
# Conversational Query
# ═══════════════════════════════════════════════════════════════════════

@router.post("/{system_id}/query")
async def query_system(system_id: str, request: ConversationQuery):
    """
    Conversational query interface.

    Engineers can ask questions in natural language about their data.
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    query = request.query.lower()
    records = data_store.get_ingested_records(system_id, limit=1000)

    if not records:
        return {
            "type": "no_data",
            "query": request.query,
            "response": "No data has been ingested for this system yet. Please upload telemetry data first.",
        }

    df = pd.DataFrame(records)

    if "show" in query or "find" in query or "get" in query:
        return {
            "type": "data_query",
            "query": request.query,
            "response": f"Found {len(df)} records in the system.",
            "summary": {
                "total_records": len(df),
                "fields": list(df.columns),
                "time_range": "All available data",
            },
            "sample_results": df.head(5).to_dict("records"),
        }

    if "average" in query or "mean" in query:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        return {
            "type": "statistics",
            "query": request.query,
            "response": "Here are the average values for numeric fields:",
            "data": {col: float(df[col].mean()) for col in numeric_cols},
        }

    if "max" in query or "maximum" in query:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        return {
            "type": "statistics",
            "query": request.query,
            "response": "Here are the maximum values for numeric fields:",
            "data": {col: float(df[col].max()) for col in numeric_cols},
        }

    if "min" in query or "minimum" in query:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        return {
            "type": "statistics",
            "query": request.query,
            "response": "Here are the minimum values for numeric fields:",
            "data": {col: float(df[col].min()) for col in numeric_cols},
        }

    stats = data_store.get_system_statistics(system_id)
    return {
        "type": "general",
        "query": request.query,
        "response": (
            f"System '{system['name']}' has {stats['total_records']} records "
            f"with {stats['field_count']} fields. "
            "You can ask specific questions like 'Show me the data', "
            "'What is the average temperature?', or 'Find maximum values'."
        ),
        "system_info": {
            "name": system["name"],
            "type": system["system_type"],
            "status": system.get("status", "active"),
            "health_score": system.get("health_score"),
        },
        "data_summary": stats,
    }


# ═══════════════════════════════════════════════════════════════════════
# Impact Radar & Next-Gen Specs
# ═══════════════════════════════════════════════════════════════════════

@router.get("/{system_id}/impact-radar")
async def get_impact_radar(system_id: str):
    """
    Get the 80/20 Impact Radar view.

    Returns the prioritized list of issues, focusing on the 20%
    of anomalies causing 80% of problems.
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id, limit=10000)

    if not records:
        return {
            "system_id": system_id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_anomalies": 0,
            "high_impact_anomalies": 0,
            "impact_distribution": None,
            "prioritized_issues": [],
            "message": "No data ingested. Upload data to see impact analysis.",
        }

    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    issues: List[Dict] = []
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            outlier_pct = len(df[abs(df[col] - mean) > 2 * std]) / len(df) * 100
            if outlier_pct > 1:
                issues.append({
                    "title": f"{col} outliers",
                    "impact_score": min(100, outlier_pct * 10),
                    "affected_percentage": outlier_pct,
                    "recommended_action": f"Investigate {col} anomalies",
                })

    issues.sort(key=lambda x: x["impact_score"], reverse=True)

    total_impact = sum(i["impact_score"] for i in issues)
    cumulative = 0
    high_impact_count = 0
    for issue in issues:
        cumulative += issue["impact_score"]
        high_impact_count += 1
        if cumulative >= total_impact * 0.8:
            break

    return {
        "system_id": system_id,
        "timestamp": datetime.utcnow().isoformat(),
        "total_anomalies": len(issues),
        "high_impact_anomalies": high_impact_count,
        "impact_distribution": {
            "top_20_percent": {"anomaly_count": high_impact_count, "impact_percentage": 80},
            "remaining_80_percent": {
                "anomaly_count": len(issues) - high_impact_count,
                "impact_percentage": 20,
            },
        } if issues else None,
        "prioritized_issues": [
            {"rank": i + 1, **issue} for i, issue in enumerate(issues[:10])
        ],
    }


@router.get("/{system_id}/next-gen-specs")
async def get_next_gen_specs(system_id: str):
    """
    Get AI-generated specifications for the next product generation.

    Based on blind spot analysis and operational data, generates
    recommendations for sensors, data architecture, and capabilities.
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id, limit=1000)
    stats = data_store.get_system_statistics(system_id)

    new_sensors: List[Dict] = []
    data_arch_recommendations: Dict[str, str] = {}

    if records:
        df = pd.DataFrame(records)

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"] and len(df) < 1000:
                data_arch_recommendations[col] = "Increase sampling rate"

        system_type = system.get("system_type", "")
        if system_type == "vehicle" and not any("vibration" in c.lower() for c in df.columns):
            new_sensors.append({
                "type": "3-axis Accelerometer",
                "location": "Suspension/Motor mount",
                "sampling_rate": "1kHz",
                "rationale": "Enable vibration analysis for predictive maintenance",
                "estimated_cost": 150,
                "diagnostic_value": "High",
            })
        elif system_type == "robot" and not any("torque" in c.lower() for c in df.columns):
            new_sensors.append({
                "type": "Torque Sensor",
                "location": "Joint actuators",
                "sampling_rate": "100Hz",
                "rationale": "Monitor joint loads for wear prediction",
                "estimated_cost": 200,
                "diagnostic_value": "High",
            })

    return {
        "system_id": system_id,
        "generated_at": datetime.utcnow().isoformat(),
        "current_generation": system.get("model", "Current"),
        "data_analyzed": stats,
        "recommended_improvements": {
            "new_sensors": new_sensors or [{
                "type": "Additional sensors recommended after data analysis",
                "location": "TBD",
                "sampling_rate": "TBD",
                "rationale": "Upload more data for specific recommendations",
                "estimated_cost": 0,
                "diagnostic_value": "TBD",
            }],
            "data_architecture": data_arch_recommendations or {
                "recommendation": "Upload telemetry data for architecture recommendations",
            },
            "connectivity": {
                "recommendation": (
                    "Add real-time streaming for critical parameters" if records else "TBD"
                ),
            },
        },
        "expected_benefits": {
            "diagnostic_coverage": "+35%" if records else "TBD",
            "early_warning_capability": "+50%" if records else "TBD",
            "false_positive_reduction": "-25%" if records else "TBD",
        },
    }
