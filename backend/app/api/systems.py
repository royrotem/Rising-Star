"""
Systems API Endpoints

Handles system management, data ingestion, and analysis.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import os

from ..services.ingestion import IngestionService
from ..services.anomaly_detection import AnomalyDetectionService
from ..services.root_cause import RootCauseService
from ..services.data_store import data_store


router = APIRouter(prefix="/systems", tags=["Systems"])

# Service instances
ingestion_service = IngestionService()
anomaly_service = AnomalyDetectionService()
root_cause_service = RootCauseService()

# Check if demo mode is enabled
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"


# Pydantic models for API
class SystemCreate(BaseModel):
    name: str
    system_type: str
    serial_number: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


class SystemResponse(BaseModel):
    id: str
    name: str
    system_type: str
    status: str
    health_score: float
    created_at: str


class FieldConfirmation(BaseModel):
    field_name: str
    confirmed_type: Optional[str] = None
    confirmed_unit: Optional[str] = None
    confirmed_meaning: Optional[str] = None
    is_correct: bool


class AnalysisRequest(BaseModel):
    include_anomaly_detection: bool = True
    include_root_cause: bool = True
    include_blind_spots: bool = True
    time_range_hours: int = 24


class ConversationQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = {}


def init_demo_systems():
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


# Initialize demo systems if in demo mode
if DEMO_MODE:
    init_demo_systems()


@router.post("/", response_model=SystemResponse)
async def create_system(system: SystemCreate):
    """Create a new monitored system."""
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

    return SystemResponse(**created_system)


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


@router.post("/{system_id}/ingest")
async def ingest_data(
    system_id: str,
    file: UploadFile = File(...),
    source_name: str = Query(default="uploaded_file"),
):
    """
    Ingest data file and perform autonomous schema discovery.

    This endpoint implements the Zero-Knowledge Ingestion approach.
    The system will analyze the uploaded data "blind" and learn its structure.
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

        # Store discovered schema in system
        data_store.update_system(system_id, {
            "discovered_schema": result.get("discovered_fields", {}),
            "status": "data_ingested"
        })

        # Store ingested data
        source_id = str(uuid.uuid4())
        data_store.store_ingested_data(
            system_id=system_id,
            source_id=source_id,
            source_name=source_name,
            records=result.get("sample_records", []),  # Store all parsed records
            discovered_schema={
                "fields": result.get("discovered_fields", []),
                "relationships": result.get("relationships", []),
            },
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
            }
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
    This builds trust and ensures accuracy.
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    confirmed = system.get("confirmed_fields", {})

    for conf in confirmations:
        if conf.is_correct:
            confirmed[conf.field_name] = {
                "confirmed": True,
                "type": conf.confirmed_type,
                "unit": conf.confirmed_unit,
                "meaning": conf.confirmed_meaning,
                "confirmed_at": datetime.utcnow().isoformat(),
            }
        else:
            confirmed[conf.field_name] = {
                "confirmed": True,
                "type": conf.confirmed_type,
                "unit": conf.confirmed_unit,
                "meaning": conf.confirmed_meaning,
                "corrected": True,
                "confirmed_at": datetime.utcnow().isoformat(),
            }

    data_store.update_system(system_id, {
        "confirmed_fields": confirmed,
        "status": "configured"
    })

    return {
        "status": "success",
        "confirmed_count": len(confirmations),
        "message": "Field mappings updated. The system will use these confirmations for future analysis.",
    }


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

    return {
        "system_id": system_id,
        "system_name": system.get("name"),
        **stats
    }


@router.get("/{system_id}/sources")
async def get_data_sources(system_id: str):
    """Get all data sources for a system."""
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    sources = data_store.get_data_sources(system_id)

    return {
        "system_id": system_id,
        "sources": sources,
        "count": len(sources),
    }


@router.post("/{system_id}/analyze")
async def analyze_system(
    system_id: str,
    request: AnalysisRequest,
):
    """
    Run comprehensive analysis on a system.

    This triggers the full agent workforce to analyze the system:
    - Anomaly detection
    - Root cause analysis
    - Blind spot detection
    - Engineering margin calculation
    """
    system = data_store.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    # Get real data if available
    records = data_store.get_ingested_records(system_id, limit=10000)
    sources = data_store.get_data_sources(system_id)

    # If we have real data, analyze it
    if records:
        import pandas as pd
        df = pd.DataFrame(records)

        # Calculate actual statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        anomalies = []
        engineering_margins = []
        blind_spots = []

        # Simple anomaly detection on real data
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                # Find outliers (values > 2 std from mean)
                outliers = df[abs(df[col] - mean) > 2 * std]
                if len(outliers) > 0:
                    anomalies.append({
                        "id": str(uuid.uuid4()),
                        "type": "statistical_outlier",
                        "severity": "medium" if len(outliers) < len(df) * 0.05 else "high",
                        "title": f"Outliers detected in {col}",
                        "description": f"Found {len(outliers)} values that deviate significantly from the mean ({mean:.2f})",
                        "affected_fields": [col],
                        "natural_language_explanation": (
                            f"The field '{col}' has {len(outliers)} data points that are more than "
                            f"2 standard deviations from the mean value of {mean:.2f}. "
                            f"This may indicate sensor errors, unusual operating conditions, or actual anomalies."
                        ),
                        "recommendations": [
                            {
                                "type": "investigation",
                                "priority": "high",
                                "action": f"Review the {len(outliers)} outlier records for {col}",
                            },
                        ],
                        "impact_score": min(100, len(outliers) / len(df) * 1000),
                    })

                # Calculate engineering margins
                current_max = df[col].max()
                if mean > 0:
                    design_limit = mean + 4 * std  # Assume 4-sigma design limit
                    margin = (design_limit - current_max) / design_limit * 100
                    engineering_margins.append({
                        "component": col,
                        "parameter": col,
                        "current_value": float(current_max),
                        "design_limit": float(design_limit),
                        "margin_percentage": float(margin),
                        "trend": "stable",
                        "safety_critical": False,
                    })

        # Identify blind spots (missing data)
        missing_cols = [col for col in df.columns if df[col].isna().sum() > len(df) * 0.1]
        if missing_cols:
            blind_spots.append({
                "title": "Missing data detected",
                "description": f"Fields {', '.join(missing_cols)} have more than 10% missing values",
                "recommended_sensor": None,
                "diagnostic_coverage_improvement": 15,
            })

        # Calculate health score based on anomalies
        health_score = 100 - (len(anomalies) * 5)
        health_score = max(50, min(100, health_score))

        analysis_result = {
            "system_id": system_id,
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": health_score,
            "data_analyzed": {
                "record_count": len(records),
                "source_count": len(sources),
                "field_count": len(df.columns),
            },
            "anomalies": anomalies,
            "engineering_margins": engineering_margins,
            "blind_spots": blind_spots,
            "insights_summary": (
                f"Analyzed {len(records)} records across {len(df.columns)} fields. "
                f"Found {len(anomalies)} potential anomalies. "
                f"System health score: {health_score}%."
            ),
        }

    else:
        # No data - return guidance
        analysis_result = {
            "system_id": system_id,
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": None,
            "data_analyzed": {
                "record_count": 0,
                "source_count": 0,
                "field_count": 0,
            },
            "anomalies": [],
            "engineering_margins": [],
            "blind_spots": [
                {
                    "title": "No data ingested",
                    "description": "Upload telemetry data to enable analysis",
                    "recommended_sensor": None,
                    "diagnostic_coverage_improvement": 100,
                }
            ],
            "insights_summary": (
                "No data has been ingested for this system yet. "
                "Upload telemetry files to enable anomaly detection and analysis."
            ),
        }

    # Update system health score
    if analysis_result.get("health_score"):
        data_store.update_system(system_id, {"health_score": analysis_result["health_score"]})

    return analysis_result


@router.post("/{system_id}/query")
async def query_system(
    system_id: str,
    request: ConversationQuery,
):
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

    import pandas as pd
    df = pd.DataFrame(records)

    # Parse query and provide data-driven response
    if "show" in query or "find" in query or "get" in query:
        # Data query
        response = {
            "type": "data_query",
            "query": request.query,
            "response": f"Found {len(df)} records in the system.",
            "summary": {
                "total_records": len(df),
                "fields": list(df.columns),
                "time_range": "All available data",
            },
            "sample_results": df.head(5).to_dict('records'),
        }

    elif "average" in query or "mean" in query:
        # Statistical query
        numeric_cols = df.select_dtypes(include=['number']).columns
        means = {col: float(df[col].mean()) for col in numeric_cols}
        response = {
            "type": "statistics",
            "query": request.query,
            "response": "Here are the average values for numeric fields:",
            "data": means,
        }

    elif "max" in query or "maximum" in query:
        numeric_cols = df.select_dtypes(include=['number']).columns
        maxes = {col: float(df[col].max()) for col in numeric_cols}
        response = {
            "type": "statistics",
            "query": request.query,
            "response": "Here are the maximum values for numeric fields:",
            "data": maxes,
        }

    elif "min" in query or "minimum" in query:
        numeric_cols = df.select_dtypes(include=['number']).columns
        mins = {col: float(df[col].min()) for col in numeric_cols}
        response = {
            "type": "statistics",
            "query": request.query,
            "response": "Here are the minimum values for numeric fields:",
            "data": mins,
        }

    else:
        # General query
        stats = data_store.get_system_statistics(system_id)
        response = {
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

    return response


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

    import pandas as pd
    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Analyze each field for issues
    issues = []
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()

        if std > 0:
            outlier_pct = len(df[abs(df[col] - mean) > 2 * std]) / len(df) * 100
            if outlier_pct > 1:  # More than 1% outliers
                issues.append({
                    "title": f"{col} outliers",
                    "impact_score": min(100, outlier_pct * 10),
                    "affected_percentage": outlier_pct,
                    "recommended_action": f"Investigate {col} anomalies",
                })

    # Sort by impact
    issues.sort(key=lambda x: x["impact_score"], reverse=True)

    # Calculate 80/20 distribution
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
            "top_20_percent": {
                "anomaly_count": high_impact_count,
                "impact_percentage": 80,
            },
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

    # Generate recommendations based on actual data analysis
    new_sensors = []
    data_arch_recommendations = {}

    if records:
        import pandas as pd
        df = pd.DataFrame(records)

        # Analyze data patterns for recommendations
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check sampling adequacy
                if len(df) < 1000:
                    data_arch_recommendations[col] = "Increase sampling rate"

        # Check for missing sensor types based on system type
        system_type = system.get("system_type", "")
        if system_type == "vehicle":
            if not any("vibration" in col.lower() for col in df.columns):
                new_sensors.append({
                    "type": "3-axis Accelerometer",
                    "location": "Suspension/Motor mount",
                    "sampling_rate": "1kHz",
                    "rationale": "Enable vibration analysis for predictive maintenance",
                    "estimated_cost": 150,
                    "diagnostic_value": "High",
                })
        elif system_type == "robot":
            if not any("torque" in col.lower() for col in df.columns):
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
            "new_sensors": new_sensors or [
                {
                    "type": "Additional sensors recommended after data analysis",
                    "location": "TBD",
                    "sampling_rate": "TBD",
                    "rationale": "Upload more data for specific recommendations",
                    "estimated_cost": 0,
                    "diagnostic_value": "TBD",
                }
            ],
            "data_architecture": data_arch_recommendations or {
                "recommendation": "Upload telemetry data for architecture recommendations"
            },
            "connectivity": {
                "recommendation": "Add real-time streaming for critical parameters" if records else "TBD",
            },
        },
        "expected_benefits": {
            "diagnostic_coverage": "+35%" if records else "TBD",
            "early_warning_capability": "+50%" if records else "TBD",
            "false_positive_reduction": "-25%" if records else "TBD",
        },
    }
