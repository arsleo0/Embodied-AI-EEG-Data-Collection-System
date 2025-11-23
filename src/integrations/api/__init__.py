"""REST API for consciousness research workbench.

Provides a FastAPI-based REST API for session management,
real-time data streaming, and analysis endpoints.
"""

from .server import create_app, APIConfig
from .endpoints import router as api_router
from .websocket import WebSocketManager, get_ws_manager
from .client import WorkbenchClient

__version__ = "1.0.0"

__all__ = [
    "create_app",
    "APIConfig",
    "api_router",
    "WebSocketManager",
    "get_ws_manager",
    "WorkbenchClient",
]
