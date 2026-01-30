"""
PDF Report API for UAIE

Provides endpoints to generate and download system analysis reports
in PDF format.  Additive feature — removing this file and its router
registration in main.py cleanly disables the feature.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..services.report_generator import generate_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/systems/{system_id}/pdf")
async def download_report(system_id: str):
    """Generate and download a PDF report for a system."""
    # Load system
    from ..api.systems import _load_systems, _load_records

    systems = _load_systems()
    system = next((s for s in systems if s.get("id") == system_id), None)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    # Load records
    records = _load_records(system_id)

    # Load statistics
    statistics = None
    try:
        import pandas as pd
        if records:
            df = pd.DataFrame(records)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            fields = []
            for col in df.columns:
                info = {
                    "name": col,
                    "type": str(df[col].dtype),
                    "null_count": int(df[col].isna().sum()),
                    "unique_count": int(df[col].nunique()),
                }
                if col in numeric_cols:
                    info["min"] = float(df[col].min()) if len(df[col].dropna()) > 0 else None
                    info["max"] = float(df[col].max()) if len(df[col].dropna()) > 0 else None
                    info["mean"] = float(df[col].mean()) if len(df[col].dropna()) > 0 else None
                    info["std"] = float(df[col].std()) if len(df[col].dropna()) > 0 else None
                fields.append(info)

            statistics = {
                "total_records": len(df),
                "total_sources": system.get("source_count", 1),
                "field_count": len(df.columns),
                "fields": fields,
            }
    except Exception:
        pass

    # Load latest analysis result if saved
    analysis = None
    try:
        import json
        from pathlib import Path
        import os

        data_dir = os.environ.get("DATA_DIR", "/app/data")
        if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        analysis_path = Path(data_dir) / "analyses" / f"{system_id}.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text())
    except Exception:
        pass

    # Generate PDF
    try:
        pdf_bytes = generate_report(system, analysis, statistics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

    filename = f"UAIE_Report_{system.get('name', 'system').replace(' ', '_')}_{system_id[:8]}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/systems/{system_id}/analyze-and-report")
async def analyze_and_report(system_id: str):
    """Run a fresh analysis and immediately generate a PDF report."""
    from ..api.systems import _load_systems, _load_records, _load_schema

    systems = _load_systems()
    system = next((s for s in systems if s.get("id") == system_id), None)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    records = _load_records(system_id)
    if not records:
        raise HTTPException(status_code=400, detail="No data available for analysis")

    # Run analysis
    from ..services.analysis_engine import AnalysisEngine
    import pandas as pd

    engine = AnalysisEngine()
    df = pd.DataFrame(records)
    schema_fields = _load_schema(system_id)
    analysis_result = engine.analyze(
        system_id=system_id,
        system_type=system.get("system_type", "generic"),
        system_name=system.get("name", ""),
        data=df,
        schema_fields=schema_fields,
    )

    # Save analysis for future use
    try:
        import json
        from pathlib import Path
        import os

        data_dir = os.environ.get("DATA_DIR", "/app/data")
        if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        analyses_dir = Path(data_dir) / "analyses"
        analyses_dir.mkdir(parents=True, exist_ok=True)

        from ..services.chat_service import _sanitize
        analysis_path = analyses_dir / f"{system_id}.json"
        analysis_path.write_text(json.dumps(_sanitize(analysis_result), indent=2, default=str))
    except Exception:
        pass

    # Build statistics
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    fields = []
    for col in df.columns:
        info = {
            "name": col,
            "type": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique()),
        }
        if col in numeric_cols:
            info["min"] = float(df[col].min()) if len(df[col].dropna()) > 0 else None
            info["max"] = float(df[col].max()) if len(df[col].dropna()) > 0 else None
            info["mean"] = float(df[col].mean()) if len(df[col].dropna()) > 0 else None
            info["std"] = float(df[col].std()) if len(df[col].dropna()) > 0 else None
        fields.append(info)

    statistics = {
        "total_records": len(df),
        "total_sources": system.get("source_count", 1),
        "field_count": len(df.columns),
        "fields": fields,
    }

    # Generate PDF
    try:
        pdf_bytes = generate_report(system, analysis_result, statistics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

    filename = f"UAIE_Report_{system.get('name', 'system').replace(' ', '_')}_{system_id[:8]}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
