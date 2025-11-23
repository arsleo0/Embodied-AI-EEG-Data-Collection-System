"""FastAPI server for the consciousness research workbench.

Provides REST API with authentication, rate limiting,
and CORS configuration.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API server configuration.

    Attributes:
        title: API title.
        version: API version.
        description: API description.
        host: Server host.
        port: Server port.
        cors_origins: Allowed CORS origins.
        rate_limit: Requests per minute.
        api_key: Optional API key for auth.
        debug: Debug mode.
    """

    title: str = "Consciousness Research Workbench API"
    version: str = "1.0.0"
    description: str = "REST API for EEG data collection and consciousness research"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] | None = None
    rate_limit: int = 60
    api_key: str | None = None
    debug: bool = False


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute.
        """
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[datetime]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed.

        Args:
            client_id: Client identifier.

        Returns:
            True if allowed.
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        if client_id not in self._requests:
            self._requests[client_id] = []

        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > minute_ago
        ]

        if len(self._requests[client_id]) >= self.requests_per_minute:
            return False

        self._requests[client_id].append(now)
        return True


def create_app(config: APIConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: API configuration.

    Returns:
        Configured FastAPI app.
    """
    config = config or APIConfig()

    app = FastAPI(
        title=config.title,
        version=config.version,
        description=config.description,
        debug=config.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    origins = config.cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiter
    rate_limiter = RateLimiter(config.rate_limit)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """Rate limiting middleware."""
        client_ip = request.client.host if request.client else "unknown"

        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
            )

        return await call_next(request)

    # API key authentication
    if config.api_key:
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            """API key authentication middleware."""
            # Skip auth for docs
            if request.url.path in ["/docs", "/redoc", "/openapi.json", "/"]:
                return await call_next(request)

            api_key = request.headers.get("X-API-Key")
            if api_key != config.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid API key"},
                )

            return await call_next(request)

    # Store config in app state
    app.state.config = config

    # Include routers
    from .endpoints import router as api_router
    from .websocket import router as ws_router

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/ws")

    # Root endpoint
    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "name": config.title,
            "version": config.version,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    # Health check
    @app.get("/api/v1/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": config.version,
        }

    logger.info(f"API server configured: {config.title} v{config.version}")

    return app


def run_server(config: APIConfig | None = None) -> None:
    """Run the API server.

    Args:
        config: API configuration.
    """
    import uvicorn

    config = config or APIConfig()
    app = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug" if config.debug else "info",
    )
