"""Command-line scenario runner."""

import sys
import time
from pathlib import Path
from typing import Any

from ..config.settings import Config, load_config
from ..data.logger import DataLogger
from .base import BaseScenario, ScenarioConfig


class ScenarioRunner:
    """Command-line based scenario runner."""

    # Default markers available in all scenarios
    DEFAULT_MARKERS = [
        {"name": "notable", "key": "n", "description": "Mark notable event"},
        {"name": "positive", "key": "p", "description": "Positive emotion/event"},
        {"name": "negative", "key": "g", "description": "Negative emotion/event"},
        {"name": "confused", "key": "c", "description": "Confusion/uncertainty"},
        {"name": "focused", "key": "f", "description": "High focus/flow state"},
    ]

    def __init__(self, config: Config | None = None):
        """Initialize scenario runner.

        Args:
            config: Configuration instance.
        """
        self.config = config or load_config()
        self.logger = DataLogger(self.config)
        self._scenario: BaseScenario | None = None
        self._is_running = False

    def list_scenarios(self) -> list[str]:
        """List available scenarios.

        Returns:
            List of scenario names.
        """
        scenarios_path = Path(self.config.get("scenarios.definitions_path", "scenarios"))

        if not scenarios_path.exists():
            return []

        # Find all YAML files
        scenarios = []
        for path in scenarios_path.glob("*.yaml"):
            scenarios.append(path.stem)

        # Add built-in types from config
        built_in = self.config.get("scenarios.types", [])
        for scenario in built_in:
            if scenario not in scenarios:
                scenarios.append(scenario)

        return sorted(scenarios)

    def load_scenario(self, name: str) -> BaseScenario:
        """Load a scenario by name.

        Args:
            name: Scenario name.

        Returns:
            Loaded scenario.

        Raises:
            FileNotFoundError: If scenario not found.
        """
        scenarios_path = Path(self.config.get("scenarios.definitions_path", "scenarios"))
        yaml_path = scenarios_path / f"{name}.yaml"

        if yaml_path.exists():
            config = ScenarioConfig.from_yaml(yaml_path)
        else:
            # Create default config for built-in scenario
            config = self._get_default_scenario_config(name)

        return BaseScenario(config)

    def _get_default_scenario_config(self, name: str) -> ScenarioConfig:
        """Get default configuration for built-in scenario.

        Args:
            name: Scenario name.

        Returns:
            ScenarioConfig instance.
        """
        defaults = {
            "urban_walk": {
                "description": "City navigation with varying environmental stimuli",
                "instructions": [
                    "Walk through different urban environments",
                    "Note transitions between quiet and busy areas",
                    "Mark emotional responses to stimuli",
                ],
            },
            "creative_flow": {
                "description": "Creative work session (design, modeling, art)",
                "instructions": [
                    "Begin your creative task",
                    "Mark moments of insight or flow",
                    "Note frustration or blocks",
                ],
            },
            "social_interaction": {
                "description": "Social contexts and emotional responses",
                "instructions": [
                    "Engage in social environments",
                    "Mark different interaction types",
                    "Note comfort levels",
                ],
            },
            "problem_solving": {
                "description": "Cognitive tasks like coding or design decisions",
                "instructions": [
                    "Work on problem-solving task",
                    "Mark decision points",
                    "Note cognitive load levels",
                ],
            },
            "meditation": {
                "description": "Calm baseline and meditation practice",
                "duration": 600,  # 10 minutes default
                "instructions": [
                    "Find a comfortable position",
                    "Focus on breath",
                    "Mark mind wandering",
                ],
            },
        }

        scenario_defaults = defaults.get(name, {})

        return ScenarioConfig(
            name=name,
            description=scenario_defaults.get("description", f"Scenario: {name}"),
            duration=scenario_defaults.get("duration"),
            instructions=scenario_defaults.get("instructions", []),
        )

    def run(
        self,
        scenario_name: str,
        duration: int | None = None,
        chunk_interval: float = 1.0,
    ) -> dict[str, Any]:
        """Run a scenario.

        Args:
            scenario_name: Name of scenario to run.
            duration: Override duration in seconds. None for manual stop.
            chunk_interval: Interval for collecting data chunks.

        Returns:
            Results dictionary with session info.
        """
        # Load scenario
        self._scenario = self.load_scenario(scenario_name)

        if duration is None:
            duration = self._scenario.config.duration

        # Start session
        session = self.logger.start_session(scenario_name)

        self._is_running = True
        self._scenario.start()

        # Print instructions
        self._print_header(scenario_name, duration)
        self._print_instructions()
        self._print_controls()

        results = {
            "session_id": session.session_id,
            "scenario": scenario_name,
            "samples_collected": 0,
            "markers": [],
            "duration": 0,
            "error": None,
        }

        try:
            start_time = time.time()

            while self._is_running:
                # Check duration limit
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    print("\n[Duration reached]")
                    break

                # Collect data chunk
                samples = self.logger.collect_chunk(chunk_interval)
                results["samples_collected"] += samples

                # Check for keyboard input (non-blocking)
                if self._check_input():
                    break

                # Update status
                self._print_status(elapsed, results["samples_collected"])

        except KeyboardInterrupt:
            print("\n\n[Stopped by user]")

        except Exception as e:
            results["error"] = str(e)
            print(f"\n[Error: {e}]")

        finally:
            # Stop scenario and session
            self._scenario.stop()
            session = self.logger.stop_session()

            results["duration"] = session.duration if session else 0
            results["markers"] = [
                {"name": m.name, "timestamp": m.timestamp}
                for m in (session.markers if session else [])
            ]

            self._is_running = False

        # Print summary
        self._print_summary(results)

        return results

    def _check_input(self) -> bool:
        """Check for keyboard input (non-blocking).

        Returns:
            True if should stop.
        """
        # Simple implementation - check for 'q' to quit
        # In a real implementation, you'd use msvcrt (Windows) or select (Unix)
        # for non-blocking input
        try:
            import select

            # Unix/Linux
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1)
                return self._handle_key(key)
        except (ImportError, OSError):
            pass

        return False

    def _handle_key(self, key: str) -> bool:
        """Handle keyboard input.

        Args:
            key: Pressed key.

        Returns:
            True if should stop.
        """
        if key.lower() == "q":
            return True

        # Check for marker keys
        if self._scenario:
            marker_config = self._scenario.get_marker_for_key(key.lower())
            if marker_config:
                self.logger.add_marker(marker_config.name)
                print(f"\n  [Marker: {marker_config.name}]")
                return False

        # Check default markers
        for marker in self.DEFAULT_MARKERS:
            if marker["key"] == key.lower():
                self.logger.add_marker(marker["name"])
                print(f"\n  [Marker: {marker['name']}]")
                return False

        return False

    def _print_header(self, scenario_name: str, duration: int | None) -> None:
        """Print scenario header."""
        print("\n" + "=" * 60)
        print(f"  SCENARIO: {scenario_name}")
        if duration:
            print(f"  Duration: {duration} seconds")
        else:
            print("  Duration: Manual (press 'q' to stop)")
        print("=" * 60 + "\n")

    def _print_instructions(self) -> None:
        """Print scenario instructions."""
        if not self._scenario:
            return

        instructions = self._scenario.get_instructions()
        if instructions:
            print("Instructions:")
            for i, instruction in enumerate(instructions, 1):
                print(f"  {i}. {instruction}")
            print()

    def _print_controls(self) -> None:
        """Print control instructions."""
        print("Controls:")
        print("  q - Stop recording")
        print("  n - Mark notable event")
        print("  p - Mark positive emotion")
        print("  g - Mark negative emotion")
        print("  f - Mark focus/flow state")

        if self._scenario and self._scenario.config.markers:
            print("\nScenario markers:")
            for marker in self._scenario.config.markers:
                print(f"  {marker.key} - {marker.name}: {marker.description}")

        print("\n" + "-" * 60)
        print("Recording started... Press 'q' to stop.\n")

    def _print_status(self, elapsed: float, samples: int) -> None:
        """Print status line."""
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"\r  Time: {mins:02d}:{secs:02d} | Samples: {samples:,}", end="", flush=True)

    def _print_summary(self, results: dict[str, Any]) -> None:
        """Print session summary."""
        print("\n\n" + "=" * 60)
        print("  SESSION COMPLETE")
        print("=" * 60)
        print(f"  Session ID: {results['session_id'][:8]}...")
        print(f"  Duration: {results['duration']:.1f} seconds")
        print(f"  Samples: {results['samples_collected']:,}")
        print(f"  Markers: {len(results['markers'])}")

        if results["error"]:
            print(f"  Error: {results['error']}")

        print("=" * 60 + "\n")

    def stop(self) -> None:
        """Stop the current scenario."""
        self._is_running = False


def run_scenario(
    scenario_name: str,
    duration: int | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Convenience function to run a scenario.

    Args:
        scenario_name: Name of scenario.
        duration: Duration in seconds.
        config_path: Path to config file.

    Returns:
        Results dictionary.
    """
    config = load_config(config_path)
    runner = ScenarioRunner(config)
    return runner.run(scenario_name, duration)
