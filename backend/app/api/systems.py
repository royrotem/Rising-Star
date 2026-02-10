"""
Systems API Endpoints

Handles system management, data ingestion, and analysis.
"""

import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

logger = logging.getLogger("uaie.systems")

from ..services.data_store import data_store
from ..services.ingestion import IngestionService
from ..services.analysis_engine import analysis_engine
from ..services.ai_agents import orchestrator as ai_orchestrator
from ..services.recommendation import (
    build_data_profile,
    enrich_fields_with_context,
    generate_system_recommendation,
)
from ..services.statistical_profiler import build_field_profiles, build_dataset_summary
from ..services.llm_discovery import (
    discover_with_llm,
    cross_validate,
    fallback_rule_based,
    SYSTEM_TYPES,
)
from ..utils import (
    sanitize_for_json,
    anomaly_to_dict,
    merge_ai_anomalies,
    load_saved_analysis,
    save_analysis,
    get_data_dir,
)
from .app_settings import get_ai_settings, get_anthropic_api_key
from .schemas import (
    AnalysisRequest,
    ConversationQuery,
    FieldConfirmation,
    SystemCreate,
    SystemResponse,
)

router = APIRouter(prefix="/systems", tags=["Systems"])


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

    NEW ARCHITECTURE (v2):
      1. Parse all files (local)
      2. Classify files as data vs description (local)
      3. Extract context from description files (local)
      4. Build rich statistical profiles for every field (local)
      5. ONE LLM call to Claude for holistic interpretation (fields + system + relationships)
      6. Cross-validate LLM output against statistical profiles (local)
      7. Return enriched structured result

    Falls back to rule-based discovery when no API key is configured.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    analysis_id = str(uuid.uuid4())
    logger.info("=" * 60)
    logger.info("ANALYZE-FILES START | analysis_id=%s | files=%d", analysis_id, len(files))
    for f in files:
        logger.info("  file: %s (content_type=%s)", f.filename, f.content_type)

    # ── Pass 1: parse all files ──────────────────────────────────────
    parsed_files: List[Dict] = []

    for file in files:
        try:
            logger.info("[Pass 1] Parsing file: %s", file.filename)
            result = await ingestion_service.ingest_file(
                file_content=file.file,
                filename=file.filename,
                system_id="temp_analysis",
                source_name=file.filename,
            )
            fields_found = len(result.get("discovered_fields", []))
            records_found = result.get("record_count", 0)
            logger.info("[Pass 1] OK: %s → %d fields, %d records", file.filename, fields_found, records_found)
            parsed_files.append({
                "filename": file.filename,
                "result": result,
            })
            file.file.seek(0)
        except Exception as e:
            logger.error("[Pass 1] FAILED: %s → %s", file.filename, e)
            logger.error(traceback.format_exc())
            parsed_files.append({
                "filename": file.filename,
                "result": None,
                "error": str(e),
            })

    # ── Pass 2: classify files as data vs description ────────────────
    logger.info("[Pass 2] Classifying %d parsed files...", len(parsed_files))
    all_parsed_field_names: List[str] = []
    for pf in parsed_files:
        result = pf.get("result")
        if result:
            for field in result.get("discovered_fields", []):
                name = field.get("name", "")
                if name not in ("line_number", "content"):
                    all_parsed_field_names.append(name)

    from ..services.ingestion import DiscoveredField as _DF
    for pf in parsed_files:
        result = pf.get("result")
        if not result:
            pf["role"] = "error"
            continue

        records = result.get("sample_records", [])
        fields_raw = result.get("discovered_fields", [])
        fields_as_df = [
            _DF(
                name=f.get("name", ""),
                inferred_type=f.get("inferred_type", "string"),
                sample_values=f.get("sample_values"),
            )
            for f in fields_raw
        ]

        this_file_fields = {f.get("name", "") for f in fields_raw}
        other_fields = [f for f in all_parsed_field_names if f not in this_file_fields]

        pf["role"] = ingestion_service.classify_file_role(
            pf["filename"], records, fields_as_df, other_fields,
        )

    for pf in parsed_files:
        logger.info("[Pass 2]   %s → role=%s", pf["filename"], pf.get("role", "?"))

    # ── Pass 3: extract descriptions from description files ──────────
    description_field_map: Dict[str, str] = {}
    description_context_texts: List[str] = []
    data_field_names = []

    for pf in parsed_files:
        if pf["role"] == "data" and pf.get("result"):
            for field in pf["result"].get("discovered_fields", []):
                name = field.get("name", "")
                if name not in ("line_number", "content"):
                    data_field_names.append(name)

    for pf in parsed_files:
        if pf["role"] == "description" and pf.get("result"):
            records = pf["result"].get("sample_records", [])
            extracted = ingestion_service.extract_descriptions_from_file(
                records, data_field_names,
            )
            description_field_map.update(extracted)
            ctx = ingestion_service.get_description_file_context(records)
            if ctx:
                description_context_texts.append(ctx)

    # ── Pass 4: build records + file summaries from data files ───────
    all_records: List[Dict] = []
    total_records = 0
    file_summaries: List[Dict] = []
    file_records_map: Dict[str, List[Dict]] = {}
    all_metadata: List[Dict] = []
    # Keep legacy discovered_fields for backward compatibility
    all_discovered_fields_legacy: List[Dict] = []

    for pf in parsed_files:
        result = pf.get("result")
        if not result:
            file_summaries.append({
                "filename": pf["filename"],
                "status": "error",
                "error": pf.get("error", "Unknown error"),
                "record_count": 0,
                "fields": [],
                "field_types": {},
                "role": "error",
            })
            continue

        records = result.get("sample_records", [])
        role = pf["role"]

        if role == "data":
            file_records_map[pf["filename"]] = records
            all_records.extend(records)

            for field in result.get("discovered_fields", []):
                field["source_file"] = pf["filename"]
                field["field_category"] = ingestion_service.classify_field_relevance(field)
                all_discovered_fields_legacy.append(field)

            total_records += result.get("record_count", 0)

            metadata_info = result.get("metadata_info", {})
            if metadata_info.get("dataset_description"):
                all_metadata.append(metadata_info)

        file_summaries.append({
            "filename": pf["filename"],
            "record_count": result.get("record_count", 0),
            "fields": [f.get("name", "") for f in result.get("discovered_fields", [])],
            "field_types": {
                f.get("name", ""): f.get("inferred_type", "")
                for f in result.get("discovered_fields", [])
            },
            "metadata": result.get("metadata_info", {}),
            "relationships": result.get("relationships", []),
            "role": role,
        })

    data_store.store_temp_analysis(
        analysis_id=analysis_id,
        records=all_records,
        file_summaries=file_summaries,
        discovered_fields=all_discovered_fields_legacy,
        file_records_map=file_records_map,
    )

    # ── Pass 5: statistical profiling ────────────────────────────────
    logger.info("[Pass 5] Building statistical profiles from %d records...", len(all_records))
    field_profiles = build_field_profiles(all_records)
    dataset_summary = build_dataset_summary(all_records, field_profiles)
    logger.info("[Pass 5] Profiled %d fields. Dataset summary: %s", len(field_profiles), json.dumps(dataset_summary, default=str)[:500])

    # Combine description context
    combined_context_texts: List[str] = list(description_context_texts)
    for meta in all_metadata:
        if meta.get("context_texts"):
            combined_context_texts.extend(meta["context_texts"])
        elif meta.get("dataset_description"):
            combined_context_texts.append(meta["dataset_description"])
    description_context = "\n\n".join(combined_context_texts)[:4000]

    file_classification = {
        "data_files": [s["filename"] for s in file_summaries if s.get("role") == "data"],
        "description_files": [s["filename"] for s in file_summaries if s.get("role") == "description"],
        "error_files": [s["filename"] for s in file_summaries if s.get("role") == "error"],
    }

    # ── Pass 6: LLM-powered discovery (or fallback) ─────────────────
    api_key = get_anthropic_api_key()
    ai_cfg = get_ai_settings()
    has_key = bool(api_key)
    ai_enabled = ai_cfg.get("enable_ai_agents", True)
    has_profiles = len(field_profiles) > 0
    use_llm = has_key and ai_enabled and has_profiles

    logger.info("[Pass 6] LLM decision: api_key=%s (len=%d), ai_enabled=%s, field_profiles=%d → use_llm=%s",
                "YES" if has_key else "NO",
                len(api_key) if api_key else 0,
                ai_enabled,
                len(field_profiles),
                use_llm)

    llm_result = None
    if use_llm:
        logger.info("[Pass 6] Calling LLM (discover_with_llm)...")
        try:
            llm_result = await discover_with_llm(
                field_profiles=field_profiles,
                dataset_summary=dataset_summary,
                description_context=description_context,
                file_classification=file_classification,
                api_key=api_key,
            )
            if llm_result:
                logger.info("[Pass 6] LLM SUCCESS: system_type=%s, fields=%d, relationships=%d",
                            llm_result.get("system_identification", {}).get("system_type", "?"),
                            len(llm_result.get("fields", [])),
                            len(llm_result.get("field_relationships", [])))
            else:
                logger.warning("[Pass 6] LLM returned None (call succeeded but no result)")
        except Exception as e:
            logger.error("[Pass 6] LLM call EXCEPTION: %s", e)
            logger.error(traceback.format_exc())
    else:
        logger.info("[Pass 6] Skipping LLM — using rule-based fallback")

    if llm_result:
        # ── Pass 7: cross-validate ──────────────────────────────────
        logger.info("[Pass 7] Cross-validating LLM output...")
        llm_result = cross_validate(llm_result, field_profiles)
        ai_powered = True
        logger.info("[Pass 7] Cross-validation complete")
    else:
        # Fallback to rule-based
        logger.info("[Pass 6-fallback] Running fallback_rule_based...")
        llm_result = fallback_rule_based(field_profiles, dataset_summary)
        ai_powered = False
        logger.info("[Pass 6-fallback] Rule-based result: system_type=%s, fields=%d",
                    llm_result.get("system_identification", {}).get("system_type", "?"),
                    len(llm_result.get("fields", [])))

    # ── Build recommendation from LLM result ─────────────────────────
    sys_id = llm_result.get("system_identification", {})
    recommendation = {
        "suggested_name": _suggest_name_from_files(file_summaries, sys_id.get("system_type", "generic_iot")),
        "suggested_type": sys_id.get("system_type", "generic_iot"),
        "suggested_description": sys_id.get("system_description", ""),
        "confidence": sys_id.get("system_type_confidence", 0.5),
        "system_subtype": sys_id.get("system_subtype"),
        "domain": sys_id.get("domain"),
        "detected_components": sys_id.get("detected_components", []),
        "probable_use_case": sys_id.get("probable_use_case"),
        "data_characteristics": sys_id.get("data_characteristics", {}),
        "reasoning": f"{'AI-powered' if ai_powered else 'Rule-based'} analysis of {len(field_profiles)} fields across {len(file_classification['data_files'])} data file(s).",
        "analysis_summary": {
            "files_analyzed": len(file_summaries),
            "total_records": total_records,
            "unique_fields": len(field_profiles),
            "ai_powered": ai_powered,
        },
    }

    # ── Build discovered_fields from LLM output (enriched format) ────
    discovered_fields_enriched = llm_result.get("fields", [])

    # Merge source_file info from legacy fields
    legacy_map = {}
    for lf in all_discovered_fields_legacy:
        legacy_map[lf.get("name", "")] = lf
    for ef in discovered_fields_enriched:
        lf = legacy_map.get(ef.get("name", ""))
        if lf:
            ef["source_file"] = lf.get("source_file")
            # Preserve sample_values/statistics from parsing if LLM didn't provide
            if not ef.get("sample_values") and lf.get("sample_values"):
                ef["sample_values"] = lf["sample_values"]
            if not ef.get("statistics") and lf.get("statistics"):
                ef["statistics"] = lf["statistics"]

    # Also store the LLM discovery result alongside the temp analysis
    data_store.store_temp_analysis(
        analysis_id=analysis_id,
        records=all_records,
        file_summaries=file_summaries,
        discovered_fields=discovered_fields_enriched,
        file_records_map=file_records_map,
    )

    file_errors = [s for s in file_summaries if s.get("status") == "error"]

    logger.info("=" * 60)
    logger.info("ANALYZE-FILES COMPLETE | analysis_id=%s", analysis_id)
    logger.info("  ai_powered: %s", ai_powered)
    logger.info("  total_records: %d", total_records)
    logger.info("  discovered_fields: %d", len(discovered_fields_enriched))
    logger.info("  recommendation: type=%s, confidence=%.2f",
                recommendation.get("suggested_type", "?"),
                recommendation.get("confidence", 0))
    logger.info("  field_relationships: %d", len(llm_result.get("field_relationships", [])))
    logger.info("  blind_spots: %d", len(llm_result.get("blind_spots", [])))
    logger.info("  confirmation_requests: %d", len(llm_result.get("recommended_confirmation_fields", [])))
    logger.info("=" * 60)

    return {
        "status": "success",
        "analysis_id": analysis_id,
        "files_analyzed": len(files),
        "total_records": total_records,
        "ai_powered": ai_powered,
        # New enriched fields (LLM or fallback)
        "discovered_fields": discovered_fields_enriched,
        # System identification
        "recommendation": recommendation,
        # Relationships discovered by LLM
        "field_relationships": llm_result.get("field_relationships", []),
        # Blind spots
        "blind_spots": llm_result.get("blind_spots", []),
        # Smart confirmation requests (only truly uncertain fields)
        "confirmation_requests": llm_result.get("recommended_confirmation_fields", []),
        # File info
        "file_classification": file_classification,
        "file_errors": file_errors,
        # Context extraction stats
        "context_extracted": len(combined_context_texts) > 0,
        "fields_enriched": len(description_field_map) if not ai_powered else len(discovered_fields_enriched),
        # Available system types (for frontend dropdown)
        "available_system_types": SYSTEM_TYPES,
    }


def _suggest_name_from_files(file_summaries: List[Dict], system_type: str) -> str:
    """Generate a suggested system name from filenames or type."""
    from ..services.llm_discovery import SYSTEM_TYPES
    file_names = [s.get("filename", "") for s in file_summaries if s.get("role") != "error"]
    for fn in file_names:
        clean_name = fn.replace("_", " ").replace("-", " ").split(".")[0]
        if len(clean_name) > 3:
            return f"{clean_name.title()} System"
    return SYSTEM_TYPES.get(system_type, "Data System")


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

    result = load_saved_analysis(system_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No analysis found. Run analysis first.")
    return result


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

    anomalies = [anomaly_to_dict(a) for a in result.anomalies]

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

    save_analysis(system_id, analysis_result)

    return sanitize_for_json(analysis_result)


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

        merge_ai_anomalies(anomalies, ai_result)

        agent_statuses = ai_result.get("agent_statuses", []) if ai_result else []
        return ai_result, agent_statuses

    except Exception as e:
        print(f"[Analysis] AI agent analysis failed (using rule-based only): {e}")
        return None, [{"agent": "AI Orchestrator", "status": "error", "error": str(e)}]


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


# ═══════════════════════════════════════════════════════════════════════
# Demo Mode
# ═══════════════════════════════════════════════════════════════════════

@router.post("/demo/create")
async def create_demo_system():
    """
    Create a fully-populated demo system with realistic HVAC data.

    This endpoint creates a complete demo system with:
    - 1000 records of realistic HVAC sensor data
    - Embedded anomalies that trigger all 25 AI agents
    - Pre-discovered schema with engineering context
    - Ready for immediate analysis demonstration

    Returns the created system ID so the frontend can navigate to it.
    """
    from ..services.demo_generator import generate_full_demo_package

    logger.info("=" * 60)
    logger.info("DEMO MODE: Creating demo system...")

    # Generate the demo package
    demo = generate_full_demo_package()

    # Create the system
    system_id = f"demo-hvac-{str(uuid.uuid4())[:8]}"

    system_data = {
        "id": system_id,
        "name": demo["metadata"]["system_name"],
        "system_type": demo["metadata"]["system_type"],
        "serial_number": "DEMO-001",
        "model": "HVAC-Demo",
        "metadata": {
            "manufacturer": "UAIE Demo",
            "description": demo["metadata"]["description"],
            "is_demo": True,
        },
        "status": "data_ingested",
        "health_score": 100.0,
        "discovered_schema": demo["discovered_fields"],
        "confirmed_fields": {},
        "is_demo": True,
        "created_at": datetime.utcnow().isoformat(),
    }

    created_system = data_store.create_system(system_data)
    logger.info("DEMO: Created system %s", system_id)

    # Store the demo data
    source_id = str(uuid.uuid4())
    data_store.store_ingested_data(
        system_id=system_id,
        source_id=source_id,
        source_name="hvac_telemetry.csv",
        records=demo["records"],
        discovered_schema={
            "fields": demo["discovered_fields"],
            "relationships": demo["relationships"],
        },
        metadata={
            "filename": "hvac_telemetry.csv",
            "demo_mode": True,
            "anomalies_injected": demo["metadata"]["demo_anomalies_injected"],
        },
    )
    logger.info("DEMO: Stored %d records", len(demo["records"]))

    # Update system with data source info
    data_store.update_system(system_id, {
        "status": "data_ingested",
    })

    logger.info("DEMO: System ready for analysis")
    logger.info("=" * 60)

    return {
        "status": "success",
        "message": "Demo system created successfully. Ready for analysis!",
        "system_id": system_id,
        "system_name": demo["metadata"]["system_name"],
        "record_count": len(demo["records"]),
        "field_count": len(demo["discovered_fields"]),
        "anomaly_types_injected": demo["metadata"]["anomaly_types"],
        "recommendation": demo["recommendation"],
    }


@router.delete("/demo/cleanup")
async def cleanup_demo_systems():
    """
    Delete all demo systems.

    Useful for cleaning up after demonstrations.
    """
    systems = data_store.list_systems(include_demo=True)
    demo_systems = [s for s in systems if s.get("is_demo", False)]

    deleted_count = 0
    for system in demo_systems:
        if data_store.delete_system(system["id"]):
            deleted_count += 1

    return {
        "status": "success",
        "deleted_count": deleted_count,
        "message": f"Deleted {deleted_count} demo system(s)",
    }
