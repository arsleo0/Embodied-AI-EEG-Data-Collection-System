"""REST API endpoints for the consciousness research workbench.

Provides endpoints for session management, analysis,
reports, and consciousness state queries.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

import numpy as np


router = APIRouter()


# Pydantic models for request/response
class SessionCreate(BaseModel):
    """Session creation request."""
    name: str = Field(..., description="Session name")
    device: str = Field(default="muse_2", description="EEG device type")
    scenario: str = Field(default="free", description="Scenario name")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Session response."""
    session_id: str
    name: str
    device: str
    scenario: str
    status: str
    created_at: str
    duration: float = 0.0


class MarkerCreate(BaseModel):
    """Event marker creation request."""
    label: str
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisRequest(BaseModel):
    """Analysis request."""
    session_id: str
    analysis_type: str = "full"
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Analysis response."""
    analysis_id: str
    session_id: str
    status: str
    results: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ConsciousnessQuery(BaseModel):
    """Consciousness state query."""
    features: list[float]
    include_confidence: bool = True


class ConsciousnessResponse(BaseModel):
    """Consciousness state response."""
    state: str
    confidence: float
    probabilities: dict[str, float]


# In-memory storage (replace with proper DB in production)
_sessions: dict[str, dict] = {}
_analyses: dict[str, dict] = {}


# Session endpoints
@router.post("/sessions", response_model=SessionResponse)
async def create_session(session: SessionCreate) -> SessionResponse:
    """Create a new recording session."""
    import uuid

    session_id = str(uuid.uuid4())[:8]

    session_data = {
        "session_id": session_id,
        "name": session.name,
        "device": session.device,
        "scenario": session.scenario,
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "duration": 0.0,
        "metadata": session.metadata,
        "markers": [],
        "data": [],
    }

    _sessions[session_id] = session_data

    return SessionResponse(**session_data)


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    limit: int = Query(default=10, le=100),
    offset: int = Query(default=0),
) -> list[SessionResponse]:
    """List all sessions."""
    sessions = list(_sessions.values())
    sessions = sessions[offset:offset + limit]
    return [SessionResponse(**s) for s in sessions]


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """Get session details."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(**_sessions[session_id])


@router.post("/sessions/{session_id}/start")
async def start_session(session_id: str) -> dict:
    """Start recording for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    _sessions[session_id]["status"] = "recording"
    _sessions[session_id]["start_time"] = datetime.now().isoformat()

    return {"status": "recording", "session_id": session_id}


@router.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str) -> dict:
    """Stop recording for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    session["status"] = "completed"
    session["end_time"] = datetime.now().isoformat()

    # Calculate duration
    if "start_time" in session:
        start = datetime.fromisoformat(session["start_time"])
        end = datetime.fromisoformat(session["end_time"])
        session["duration"] = (end - start).total_seconds()

    return {"status": "completed", "duration": session["duration"]}


@router.post("/sessions/{session_id}/markers")
async def add_marker(session_id: str, marker: MarkerCreate) -> dict:
    """Add an event marker to a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    marker_data = {
        "label": marker.label,
        "timestamp": marker.timestamp or datetime.now().timestamp(),
        "metadata": marker.metadata,
    }

    _sessions[session_id]["markers"].append(marker_data)

    return {"status": "added", "marker": marker_data}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del _sessions[session_id]

    return {"status": "deleted", "session_id": session_id}


# Analysis endpoints
@router.post("/analysis", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """Run analysis on a session."""
    if request.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    import uuid
    analysis_id = str(uuid.uuid4())[:8]

    # Placeholder results
    results = {
        "band_powers": {
            "delta": 0.15,
            "theta": 0.20,
            "alpha": 0.30,
            "beta": 0.25,
            "gamma": 0.10,
        },
        "quality_score": 0.85,
        "dominant_state": "focus",
    }

    analysis_data = {
        "analysis_id": analysis_id,
        "session_id": request.session_id,
        "status": "completed",
        "results": results,
        "created_at": datetime.now().isoformat(),
    }

    _analyses[analysis_id] = analysis_data

    return AnalysisResponse(**analysis_data)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str) -> AnalysisResponse:
    """Get analysis results."""
    if analysis_id not in _analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisResponse(**_analyses[analysis_id])


# Consciousness endpoints
@router.post("/consciousness/classify", response_model=ConsciousnessResponse)
async def classify_state(query: ConsciousnessQuery) -> ConsciousnessResponse:
    """Classify consciousness state from features."""
    # Placeholder classification
    states = ["focus", "relaxed", "drowsy", "anxious"]

    # Simple mock classification
    if len(query.features) > 0:
        feature_sum = sum(query.features)
        state_idx = int(abs(feature_sum * 100)) % len(states)
    else:
        state_idx = 0

    state = states[state_idx]
    confidence = 0.75 + (hash(str(query.features)) % 25) / 100

    probabilities = {s: 0.1 for s in states}
    probabilities[state] = confidence

    return ConsciousnessResponse(
        state=state,
        confidence=confidence,
        probabilities=probabilities,
    )


@router.get("/consciousness/states")
async def list_states() -> dict:
    """List available consciousness states."""
    return {
        "states": [
            {"name": "focus", "description": "Concentrated attention"},
            {"name": "relaxed", "description": "Calm, restful state"},
            {"name": "drowsy", "description": "Reduced alertness"},
            {"name": "anxious", "description": "Heightened arousal"},
            {"name": "creative", "description": "Divergent thinking"},
            {"name": "flow", "description": "Optimal engagement"},
        ]
    }


# Report endpoints
@router.post("/reports/session/{session_id}")
async def generate_session_report(
    session_id: str,
    format: str = Query(default="html", regex="^(html|json|markdown)$"),
) -> dict:
    """Generate a session report."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "status": "generated",
        "session_id": session_id,
        "format": format,
        "download_url": f"/api/v1/reports/download/{session_id}.{format}",
    }


# Data endpoints
@router.post("/data/upload/{session_id}")
async def upload_data(
    session_id: str,
    file: UploadFile = File(...),
) -> dict:
    """Upload EEG data for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    content = await file.read()

    return {
        "status": "uploaded",
        "session_id": session_id,
        "filename": file.filename,
        "size_bytes": len(content),
    }


@router.get("/data/{session_id}")
async def get_data(
    session_id: str,
    start: float = Query(default=0, description="Start time in seconds"),
    end: float = Query(default=-1, description="End time in seconds"),
) -> dict:
    """Get EEG data for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Placeholder data
    return {
        "session_id": session_id,
        "channels": ["TP9", "AF7", "AF8", "TP10"],
        "sample_rate": 256,
        "samples": 0,
        "time_range": [start, end if end > 0 else 0],
    }


# Device endpoints
@router.get("/devices")
async def list_devices() -> dict:
    """List supported EEG devices."""
    return {
        "devices": [
            {"id": "muse_2", "name": "Muse 2", "channels": 4},
            {"id": "muse_s", "name": "Muse S", "channels": 4},
            {"id": "synthetic", "name": "Synthetic", "channels": 4},
        ]
    }


# System endpoints
@router.get("/system/info")
async def system_info() -> dict:
    """Get system information."""
    return {
        "name": "Consciousness Research Workbench",
        "version": "1.0.0",
        "active_sessions": len([s for s in _sessions.values() if s["status"] == "recording"]),
        "total_sessions": len(_sessions),
        "uptime": "N/A",
    }
