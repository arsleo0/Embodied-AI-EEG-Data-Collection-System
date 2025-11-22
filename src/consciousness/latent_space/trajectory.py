"""Consciousness trajectory tracking and analysis.

Track consciousness state transitions over time and
compute trajectory metrics for temporal analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ConsciousnessTrajectory:
    """Container for consciousness trajectory data.

    Attributes:
        embeddings: Sequence of embedding vectors.
        timestamps: Time values for each point.
        labels: Optional state labels.
        metrics: Computed trajectory metrics.
    """

    embeddings: np.ndarray
    timestamps: np.ndarray
    labels: list[str] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trajectory data."""
        if len(self.embeddings) != len(self.timestamps):
            raise ValueError("Embeddings and timestamps must have same length")

        if self.labels is not None and len(self.labels) != len(self.embeddings):
            raise ValueError("Labels must match embeddings length")

    @property
    def duration(self) -> float:
        """Total trajectory duration."""
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def n_points(self) -> int:
        """Number of trajectory points."""
        return len(self.embeddings)

    def get_segment(
        self,
        start_time: float,
        end_time: float,
    ) -> "ConsciousnessTrajectory":
        """Extract trajectory segment.

        Args:
            start_time: Segment start time.
            end_time: Segment end time.

        Returns:
            New trajectory for the segment.
        """
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        labels = [self.labels[i] for i in np.where(mask)[0]] if self.labels else None

        return ConsciousnessTrajectory(
            embeddings=self.embeddings[mask],
            timestamps=self.timestamps[mask],
            labels=labels,
        )


class TrajectoryTracker:
    """Track consciousness state trajectories over time.

    Records embedding sequences and computes temporal
    metrics like velocity, smoothness, and state transitions.

    Example:
        >>> tracker = TrajectoryTracker()
        >>> for t, embedding in zip(times, embeddings):
        ...     tracker.add_point(embedding, t)
        >>> trajectory = tracker.get_trajectory()
        >>> metrics = tracker.compute_metrics()
    """

    def __init__(self):
        """Initialize trajectory tracker."""
        self._embeddings: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._labels: list[str] = []
        self._is_recording = False

    def start_recording(self) -> None:
        """Start recording trajectory."""
        self._is_recording = True

    def stop_recording(self) -> None:
        """Stop recording trajectory."""
        self._is_recording = False

    def add_point(
        self,
        embedding: np.ndarray,
        timestamp: float,
        label: str | None = None,
    ) -> None:
        """Add point to trajectory.

        Args:
            embedding: Embedding vector.
            timestamp: Time value.
            label: Optional state label.
        """
        self._embeddings.append(embedding.copy())
        self._timestamps.append(timestamp)
        if label is not None:
            self._labels.append(label)

    def clear(self) -> None:
        """Clear recorded trajectory."""
        self._embeddings = []
        self._timestamps = []
        self._labels = []

    def get_trajectory(self) -> ConsciousnessTrajectory:
        """Get recorded trajectory.

        Returns:
            ConsciousnessTrajectory object.
        """
        if not self._embeddings:
            raise ValueError("No trajectory recorded")

        return ConsciousnessTrajectory(
            embeddings=np.array(self._embeddings),
            timestamps=np.array(self._timestamps),
            labels=self._labels if self._labels else None,
        )

    def compute_metrics(self) -> dict[str, Any]:
        """Compute trajectory metrics.

        Returns:
            Dictionary of trajectory metrics.
        """
        trajectory = self.get_trajectory()
        return compute_trajectory_metrics(trajectory)

    @property
    def n_points(self) -> int:
        """Number of recorded points."""
        return len(self._embeddings)

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording


def compute_trajectory_metrics(
    trajectory: ConsciousnessTrajectory,
) -> dict[str, Any]:
    """Compute comprehensive trajectory metrics.

    Args:
        trajectory: Consciousness trajectory.

    Returns:
        Dictionary of metrics including:
        - velocity: Instantaneous velocities
        - mean_velocity: Average velocity
        - smoothness: Trajectory smoothness (jerk)
        - total_distance: Cumulative path length
        - displacement: Direct distance from start to end
        - efficiency: displacement / total_distance
        - state_transitions: Number of state changes
        - dwell_times: Time spent in each state
    """
    embeddings = trajectory.embeddings
    timestamps = trajectory.timestamps

    if len(embeddings) < 2:
        return {"error": "Need at least 2 points"}

    # Compute displacements
    displacements = np.diff(embeddings, axis=0)
    time_diffs = np.diff(timestamps)

    # Avoid division by zero
    time_diffs = np.maximum(time_diffs, 1e-10)

    # Velocities (distance / time)
    distances = np.linalg.norm(displacements, axis=1)
    velocities = distances / time_diffs

    # Accelerations
    if len(velocities) > 1:
        velocity_diffs = np.diff(velocities)
        accelerations = velocity_diffs / time_diffs[:-1]
    else:
        accelerations = np.array([0])

    # Jerk (rate of change of acceleration) - smoothness metric
    if len(accelerations) > 1:
        jerk = np.diff(accelerations) / time_diffs[:-2] if len(time_diffs) > 2 else np.array([0])
        smoothness = -np.mean(np.abs(jerk))  # Higher is smoother
    else:
        smoothness = 0.0

    # Path metrics
    total_distance = np.sum(distances)
    displacement = np.linalg.norm(embeddings[-1] - embeddings[0])
    efficiency = displacement / total_distance if total_distance > 0 else 1.0

    # Curvature
    curvatures = compute_curvature(embeddings)

    metrics = {
        "velocity": {
            "values": velocities.tolist(),
            "mean": float(np.mean(velocities)),
            "std": float(np.std(velocities)),
            "max": float(np.max(velocities)),
            "min": float(np.min(velocities)),
        },
        "acceleration": {
            "values": accelerations.tolist(),
            "mean": float(np.mean(np.abs(accelerations))),
            "max": float(np.max(np.abs(accelerations))),
        },
        "smoothness": float(smoothness),
        "total_distance": float(total_distance),
        "displacement": float(displacement),
        "efficiency": float(efficiency),
        "duration": float(trajectory.duration),
        "n_points": trajectory.n_points,
        "curvature": {
            "mean": float(np.mean(curvatures)),
            "max": float(np.max(curvatures)),
        },
    }

    # State transition analysis
    if trajectory.labels:
        transitions = compute_state_transitions(trajectory.labels)
        dwell_times = compute_dwell_times(trajectory.labels, timestamps)
        metrics["state_transitions"] = transitions
        metrics["dwell_times"] = dwell_times

    return metrics


def compute_curvature(points: np.ndarray) -> np.ndarray:
    """Compute curvature along trajectory.

    Uses discrete curvature approximation based on
    angle between consecutive segments.

    Args:
        points: Trajectory points (n_points, n_dims).

    Returns:
        Curvature values for each interior point.
    """
    if len(points) < 3:
        return np.array([0])

    curvatures = []

    for i in range(1, len(points) - 1):
        # Vectors from point i to neighbors
        v1 = points[i - 1] - points[i]
        v2 = points[i + 1] - points[i]

        # Lengths
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1e-10 or len2 < 1e-10:
            curvatures.append(0)
            continue

        # Cosine of angle
        cos_angle = np.dot(v1, v2) / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1, 1)

        # Curvature approximation (1/R where R is radius of curvature)
        angle = np.arccos(cos_angle)
        chord = np.linalg.norm(points[i + 1] - points[i - 1])

        if chord > 1e-10:
            curvature = 2 * np.sin(angle) / chord
        else:
            curvature = 0

        curvatures.append(curvature)

    return np.array(curvatures)


def compute_state_transitions(labels: list[str]) -> dict[str, Any]:
    """Compute state transition statistics.

    Args:
        labels: Sequence of state labels.

    Returns:
        Transition statistics including:
        - n_transitions: Total number of transitions
        - transition_matrix: State-to-state counts
        - transition_list: List of (from, to, index) tuples
    """
    if not labels:
        return {}

    transitions = []
    transition_counts: dict[tuple[str, str], int] = {}

    prev_label = labels[0]
    for i, label in enumerate(labels[1:], 1):
        if label != prev_label:
            transitions.append((prev_label, label, i))

            key = (prev_label, label)
            transition_counts[key] = transition_counts.get(key, 0) + 1

        prev_label = label

    # Build transition matrix
    unique_states = sorted(set(labels))
    n_states = len(unique_states)
    state_to_idx = {s: i for i, s in enumerate(unique_states)}

    matrix = np.zeros((n_states, n_states), dtype=int)
    for (from_state, to_state), count in transition_counts.items():
        i = state_to_idx[from_state]
        j = state_to_idx[to_state]
        matrix[i, j] = count

    return {
        "n_transitions": len(transitions),
        "transition_list": transitions,
        "transition_matrix": matrix.tolist(),
        "states": unique_states,
    }


def compute_dwell_times(
    labels: list[str],
    timestamps: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute time spent in each state.

    Args:
        labels: Sequence of state labels.
        timestamps: Time values.

    Returns:
        Dictionary with dwell time statistics per state.
    """
    if not labels or len(labels) != len(timestamps):
        return {}

    # Group consecutive labels
    state_durations: dict[str, list[float]] = {}

    current_state = labels[0]
    state_start = timestamps[0]

    for i in range(1, len(labels)):
        if labels[i] != current_state:
            # State ended
            duration = timestamps[i] - state_start
            if current_state not in state_durations:
                state_durations[current_state] = []
            state_durations[current_state].append(duration)

            # New state starts
            current_state = labels[i]
            state_start = timestamps[i]

    # Handle last state
    duration = timestamps[-1] - state_start
    if current_state not in state_durations:
        state_durations[current_state] = []
    state_durations[current_state].append(duration)

    # Compute statistics
    results = {}
    for state, durations in state_durations.items():
        results[state] = {
            "total": float(np.sum(durations)),
            "mean": float(np.mean(durations)),
            "std": float(np.std(durations)) if len(durations) > 1 else 0.0,
            "n_visits": len(durations),
        }

    return results


def detect_state_changes(
    trajectory: ConsciousnessTrajectory,
    threshold: float = 0.5,
    min_duration: float = 1.0,
) -> list[dict[str, Any]]:
    """Detect significant state changes in trajectory.

    Uses velocity and direction changes to identify
    transitions between consciousness states.

    Args:
        trajectory: Consciousness trajectory.
        threshold: Velocity threshold for state change.
        min_duration: Minimum duration between changes.

    Returns:
        List of detected state changes with timestamps.
    """
    embeddings = trajectory.embeddings
    timestamps = trajectory.timestamps

    if len(embeddings) < 3:
        return []

    # Compute velocities
    displacements = np.diff(embeddings, axis=0)
    time_diffs = np.diff(timestamps)
    time_diffs = np.maximum(time_diffs, 1e-10)

    velocities = np.linalg.norm(displacements, axis=1) / time_diffs

    # Compute direction changes (cosine similarity between consecutive displacements)
    direction_changes = []
    for i in range(len(displacements) - 1):
        v1 = displacements[i]
        v2 = displacements[i + 1]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 > 1e-10 and len2 > 1e-10:
            cos_sim = np.dot(v1, v2) / (len1 * len2)
            direction_changes.append(1 - cos_sim)  # 0 = same direction, 2 = opposite
        else:
            direction_changes.append(0)

    direction_changes = np.array(direction_changes)

    # Detect changes
    state_changes = []
    last_change_time = timestamps[0]

    for i in range(len(direction_changes)):
        # Check velocity spike and direction change
        is_velocity_spike = velocities[i] > threshold * np.mean(velocities)
        is_direction_change = direction_changes[i] > 0.5

        time_since_last = timestamps[i + 1] - last_change_time

        if (is_velocity_spike or is_direction_change) and time_since_last >= min_duration:
            state_changes.append({
                "index": i + 1,
                "timestamp": float(timestamps[i + 1]),
                "velocity": float(velocities[i]),
                "direction_change": float(direction_changes[i]),
            })
            last_change_time = timestamps[i + 1]

    return state_changes


def interpolate_trajectory(
    trajectory: ConsciousnessTrajectory,
    target_timestamps: np.ndarray,
) -> ConsciousnessTrajectory:
    """Interpolate trajectory to new timestamps.

    Args:
        trajectory: Original trajectory.
        target_timestamps: New timestamps for interpolation.

    Returns:
        Interpolated trajectory.
    """
    from scipy.interpolate import interp1d

    # Interpolate each dimension
    n_dims = trajectory.embeddings.shape[1]
    interpolated = np.zeros((len(target_timestamps), n_dims))

    for d in range(n_dims):
        f = interp1d(
            trajectory.timestamps,
            trajectory.embeddings[:, d],
            kind="cubic",
            fill_value="extrapolate",
        )
        interpolated[:, d] = f(target_timestamps)

    # Interpolate labels if present (use nearest)
    labels = None
    if trajectory.labels:
        label_times = np.searchsorted(trajectory.timestamps, target_timestamps)
        label_times = np.clip(label_times, 0, len(trajectory.labels) - 1)
        labels = [trajectory.labels[i] for i in label_times]

    return ConsciousnessTrajectory(
        embeddings=interpolated,
        timestamps=target_timestamps,
        labels=labels,
    )


def save_trajectory(
    trajectory: ConsciousnessTrajectory,
    path: str | Path,
) -> None:
    """Save trajectory to file.

    Args:
        trajectory: Trajectory to save.
        path: Output file path (.npz).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        embeddings=trajectory.embeddings,
        timestamps=trajectory.timestamps,
        labels=np.array(trajectory.labels) if trajectory.labels else None,
        metrics=trajectory.metrics,
    )


def load_trajectory(path: str | Path) -> ConsciousnessTrajectory:
    """Load trajectory from file.

    Args:
        path: Input file path.

    Returns:
        Loaded trajectory.
    """
    data = np.load(path, allow_pickle=True)

    labels = data["labels"]
    if labels is not None:
        labels = labels.tolist()

    trajectory = ConsciousnessTrajectory(
        embeddings=data["embeddings"],
        timestamps=data["timestamps"],
        labels=labels,
    )

    metrics = data["metrics"]
    if metrics is not None:
        trajectory.metrics = dict(metrics.item())

    return trajectory
