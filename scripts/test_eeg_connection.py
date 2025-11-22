#!/usr/bin/env python3
"""Test EEG device connection.

Usage:
    python scripts/test_eeg_connection.py
    python scripts/test_eeg_connection.py --device muse_2
    python scripts/test_eeg_connection.py --device synthetic --duration 3
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.eeg.connection import test_connection, list_available_devices


@click.command()
@click.option(
    "--device",
    "-d",
    type=click.Choice(list_available_devices()),
    default=None,
    help="Device type to test. Uses config default if not specified.",
)
@click.option(
    "--duration",
    "-t",
    type=float,
    default=5.0,
    help="Duration to stream data for testing (seconds).",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file.",
)
def main(device: str | None, duration: float, config: str | None):
    """Test connection to EEG device.

    This script connects to your EEG device, streams data for a few seconds,
    and reports the results. Use it to verify your setup before running scenarios.
    """
    click.echo("\n" + "=" * 50)
    click.echo("  EEG Connection Test")
    click.echo("=" * 50 + "\n")

    if device:
        click.echo(f"Device: {device}")
    else:
        click.echo("Device: (from config)")

    click.echo(f"Test duration: {duration}s\n")

    # Run test
    results = test_connection(
        device_type=device,
        duration=duration,
        config_path=config,
    )

    # Display results
    click.echo("\n" + "-" * 50)
    click.echo("  Results")
    click.echo("-" * 50 + "\n")

    if results["success"]:
        click.secho("  Status: SUCCESS", fg="green", bold=True)
    else:
        click.secho("  Status: FAILED", fg="red", bold=True)

    click.echo(f"\n  Device Info:")
    for key, value in results["device_info"].items():
        click.echo(f"    {key}: {value}")

    click.echo(f"\n  Samples received: {results['samples_received']:,}")
    click.echo(f"  Actual sample rate: {results['actual_rate']:.1f} Hz")

    if results["errors"]:
        click.echo("\n  Errors:")
        for error in results["errors"]:
            click.secho(f"    - {error}", fg="red")

    click.echo("\n" + "=" * 50 + "\n")

    # Exit code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
