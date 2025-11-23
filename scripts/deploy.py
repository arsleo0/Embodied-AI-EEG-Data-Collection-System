#!/usr/bin/env python3
"""Deployment helper for the consciousness research workbench.

Provides utilities for Docker deployment, configuration
validation, and system setup.

Usage:
    python scripts/deploy.py build
    python scripts/deploy.py up
    python scripts/deploy.py down
    python scripts/deploy.py logs
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a shell command.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.

    Returns:
        Return code.
    """
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def docker_build(args) -> int:
    """Build Docker images."""
    project_root = Path(__file__).parent.parent
    deployment_dir = project_root / "deployment"

    cmd = ["docker-compose", "-f", str(deployment_dir / "docker-compose.yml")]

    if args.profile:
        cmd.extend(["--profile", args.profile])

    cmd.extend(["build"])

    if args.no_cache:
        cmd.append("--no-cache")

    return run_command(cmd, cwd=project_root)


def docker_up(args) -> int:
    """Start Docker services."""
    project_root = Path(__file__).parent.parent
    deployment_dir = project_root / "deployment"

    cmd = ["docker-compose", "-f", str(deployment_dir / "docker-compose.yml")]

    if args.profile:
        cmd.extend(["--profile", args.profile])

    cmd.extend(["up"])

    if args.detach:
        cmd.append("-d")

    return run_command(cmd, cwd=project_root)


def docker_down(args) -> int:
    """Stop Docker services."""
    project_root = Path(__file__).parent.parent
    deployment_dir = project_root / "deployment"

    cmd = ["docker-compose", "-f", str(deployment_dir / "docker-compose.yml")]
    cmd.extend(["down"])

    if args.volumes:
        cmd.append("-v")

    return run_command(cmd, cwd=project_root)


def docker_logs(args) -> int:
    """Show Docker logs."""
    project_root = Path(__file__).parent.parent
    deployment_dir = project_root / "deployment"

    cmd = ["docker-compose", "-f", str(deployment_dir / "docker-compose.yml")]
    cmd.extend(["logs"])

    if args.follow:
        cmd.append("-f")

    if args.service:
        cmd.append(args.service)

    return run_command(cmd, cwd=project_root)


def validate_config(args) -> int:
    """Validate configuration files."""
    import yaml

    project_root = Path(__file__).parent.parent
    config_files = [
        project_root / "config.yaml",
        project_root / "deployment" / "config" / "production.yaml",
    ]

    errors = []

    for config_file in config_files:
        if not config_file.exists():
            print(f"Warning: {config_file} not found")
            continue

        try:
            with open(config_file) as f:
                yaml.safe_load(f)
            print(f"✓ {config_file.name} is valid")
        except yaml.YAMLError as e:
            errors.append(f"{config_file.name}: {e}")
            print(f"✗ {config_file.name} has errors")

    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nAll configuration files are valid!")
    return 0


def check_requirements(args) -> int:
    """Check system requirements."""
    requirements = []

    # Check Python version
    import sys
    if sys.version_info < (3, 10):
        requirements.append("Python 3.10+ required")
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check Docker
    result = subprocess.run(
        ["docker", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"✓ Docker installed")
    else:
        requirements.append("Docker not installed")

    # Check docker-compose
    result = subprocess.run(
        ["docker-compose", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"✓ Docker Compose installed")
    else:
        requirements.append("Docker Compose not installed")

    if requirements:
        print("\nMissing requirements:")
        for req in requirements:
            print(f"  - {req}")
        return 1

    print("\nAll requirements satisfied!")
    return 0


def main():
    """Main deployment helper."""
    parser = argparse.ArgumentParser(
        description="Deployment helper for consciousness research workbench"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build Docker images")
    build_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without cache",
    )
    build_parser.add_argument(
        "--profile",
        type=str,
        help="Docker Compose profile",
    )

    # Up command
    up_parser = subparsers.add_parser("up", help="Start services")
    up_parser.add_argument(
        "-d", "--detach",
        action="store_true",
        help="Run in background",
    )
    up_parser.add_argument(
        "--profile",
        type=str,
        help="Docker Compose profile",
    )

    # Down command
    down_parser = subparsers.add_parser("down", help="Stop services")
    down_parser.add_argument(
        "-v", "--volumes",
        action="store_true",
        help="Remove volumes",
    )

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show logs")
    logs_parser.add_argument(
        "-f", "--follow",
        action="store_true",
        help="Follow logs",
    )
    logs_parser.add_argument(
        "service",
        nargs="?",
        help="Service name",
    )

    # Validate command
    subparsers.add_parser("validate", help="Validate configuration")

    # Check command
    subparsers.add_parser("check", help="Check requirements")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "build": docker_build,
        "up": docker_up,
        "down": docker_down,
        "logs": docker_logs,
        "validate": validate_config,
        "check": check_requirements,
    }

    sys.exit(commands[args.command](args))


if __name__ == "__main__":
    main()
