"""
PDF Report API for UAIE

Provides endpoints to generate and download system analysis reports
in PDF format.  Additive feature — removing this file and its router
registration in main.py cleanly disables the feature.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..services.data_store import data_store
from ..services.report_generator import generate_report
from ..utils import (
    build_field_statistics,
    load_saved_analysis,
    save_analysis,
    sanitize_for_json,
)

router = APIRouter(prefix="/reports", tags=["reports"])


def _pdf_response(pdf_bytes: bytes, system: dict, system_id: str) -> Response:
    """Build a PDF download Response."""
    filename = f"UAIE_Report_{system.get('name', 'system').replace(' ', '_')}_{system_id[:8]}.pdf"
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/systems/{system_id}/pdf")
async def download_report(system_id: str):
    """Generate and download a PDF report for a system."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id)
    statistics = build_field_statistics(records, system.get("source_count", 1))
    analysis = load_saved_analysis(system_id)

    try:
        pdf_bytes = generate_report(system, analysis, statistics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

    return _pdf_response(pdf_bytes, system, system_id)


@router.post("/systems/{system_id}/analyze-and-report")
async def analyze_and_report(system_id: str):
    """Run a fresh analysis and immediately generate a PDF report."""
    from ..services.analysis_engine import AnalysisEngine

    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = data_store.get_ingested_records(system_id)
    if not records:
        raise HTTPException(status_code=400, detail="No data available for analysis")

    engine = AnalysisEngine()
    df = pd.DataFrame(records)
    schema = data_store.get_schema(system_id)
    schema_fields = list(schema.values()) if isinstance(schema, dict) else (schema or [])
    analysis_result = engine.analyze(
        system_id=system_id,
        system_type=system.get("system_type", "generic"),
        system_name=system.get("name", ""),
        data=df,
        schema_fields=schema_fields,
    )

    save_analysis(system_id, sanitize_for_json(analysis_result))
    statistics = build_field_statistics(records, system.get("source_count", 1))

    try:
        pdf_bytes = generate_report(system, analysis_result, statistics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

    return _pdf_response(pdf_bytes, system, system_id)
