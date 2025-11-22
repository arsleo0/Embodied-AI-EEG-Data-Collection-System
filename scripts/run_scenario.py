#!/usr/bin/env python3
"""Run an EEG data collection scenario.

Usage:
    python scripts/run_scenario.py urban_walk
    python scripts/run_scenario.py meditation --duration 300
    python scripts/run_scenario.py --list
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.config.settings import load_config
from src.scenarios.runner import ScenarioRunner


@click.command()
@click.argument("scenario", required=False)
@click.option(
    "--duration",
    "-d",
    type=int,
    default=None,
    help="Override scenario duration (seconds).",
)
@click.option(
    "--list",
    "-l",
    "list_scenarios",
    is_flag=True,
    help="List available scenarios.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file.",
)
@click.option(
    "--chunk-interval",
    type=float,
    default=1.0,
    help="Interval for collecting data chunks (seconds).",
)
def main(
    scenario: str | None,
    duration: int | None,
    list_scenarios: bool,
    config: str | None,
    chunk_interval: float,
):
    """Run an EEG data collection scenario.

    SCENARIO is the name of the scenario to run (e.g., urban_walk, meditation).

    During recording, use keyboard shortcuts to add event markers:
    - q: Stop recording
    - n: Mark notable event
    - p: Mark positive emotion
    - g: Mark negative emotion
    - f: Mark focus/flow state

    Data is saved to data/raw/ as HDF5 files.
    """
    # Load configuration
    cfg = load_config(config)
    runner = ScenarioRunner(cfg)

    # List scenarios
    if list_scenarios:
        scenarios = runner.list_scenarios()
        click.echo("\nAvailable scenarios:")
        for name in scenarios:
            click.echo(f"  - {name}")
        click.echo()
        return

    # Check scenario argument
    if not scenario:
        click.echo("Error: Please specify a scenario name or use --list")
        click.echo("Usage: python scripts/run_scenario.py <scenario_name>")
        sys.exit(1)

    # Run scenario
    try:
        results = runner.run(
            scenario_name=scenario,
            duration=duration,
            chunk_interval=chunk_interval,
        )

        # Exit code based on success
        has_error = results.get("error") is not None
        sys.exit(1 if has_error else 0)

    except KeyboardInterrupt:
        click.echo("\nScenario interrupted.")
        sys.exit(0)

    except Exception as e:
        click.secho(f"\nError: {e}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
