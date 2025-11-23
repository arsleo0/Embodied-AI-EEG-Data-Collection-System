#!/usr/bin/env python3
"""Launch the REST API server.

Starts the FastAPI-based REST API server with configurable
host, port, and authentication settings.

Usage:
    python scripts/start_api_server.py
    python scripts/start_api_server.py --port 8080
    python scripts/start_api_server.py --debug --reload
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(
        description="Launch consciousness research workbench API server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication",
    )
    parser.add_argument(
        "--cors-origins",
        type=str,
        nargs="+",
        default=["*"],
        help="Allowed CORS origins",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Rate limit (requests per minute)",
    )

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required.")
        print("Install with: pip install uvicorn")
        sys.exit(1)

    print("=" * 60)
    print("  Consciousness Research Workbench - API Server")
    print("=" * 60)
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    print(f"Workers: {args.workers}")
    print(f"\nAPI Documentation: http://localhost:{args.port}/docs")
    print(f"Health Check: http://localhost:{args.port}/api/v1/health")
    print("\nPress Ctrl+C to stop.\n")

    # Create config
    from src.integrations.api import APIConfig, create_app

    config = APIConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        api_key=args.api_key,
        cors_origins=args.cors_origins,
        rate_limit=args.rate_limit,
    )

    # Run server
    if args.reload:
        # Development mode with reload
        uvicorn.run(
            "src.integrations.api.server:create_app",
            host=args.host,
            port=args.port,
            reload=True,
            factory=True,
            log_level="debug" if args.debug else "info",
        )
    else:
        # Production mode
        app = create_app(config)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="debug" if args.debug else "info",
        )


if __name__ == "__main__":
    main()
