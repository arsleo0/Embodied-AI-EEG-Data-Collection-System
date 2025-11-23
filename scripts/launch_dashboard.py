#!/usr/bin/env python3
"""Launch the consciousness research dashboard.

A real-time dashboard for monitoring EEG sessions,
consciousness states, and signal quality.

Usage:
    python scripts/launch_dashboard.py
    python scripts/launch_dashboard.py --port 8502
    python scripts/launch_dashboard.py --demo
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Launch the dashboard."""
    parser = argparse.ArgumentParser(
        description="Launch consciousness research dashboard"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run dashboard on (default: 8501)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with simulated data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="muse_2",
        help="EEG device type (default: muse_2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: Streamlit is required.")
        print("Install with: pip install streamlit")
        sys.exit(1)

    # Get path to dashboard app
    app_path = Path(__file__).parent.parent / "src" / "integrations" / "dashboard" / "app.py"

    if not app_path.exists():
        print(f"Error: Dashboard app not found at {app_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Consciousness Research Workbench - Dashboard")
    print("=" * 60)
    print(f"\nStarting dashboard on port {args.port}...")
    print(f"Mode: {'Demo' if args.demo else 'Live'}")
    print(f"Device: {args.device}")
    print("\nOpen your browser to: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop.\n")

    # Build streamlit arguments
    st_args = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--",  # Separator for app arguments
    ]

    if args.demo:
        st_args.append("--demo")
    if args.device:
        st_args.extend(["--device", args.device])
    if args.config:
        st_args.extend(["--config", args.config])

    # Run streamlit
    sys.argv = st_args
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
