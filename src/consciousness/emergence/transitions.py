"""State transition analysis for consciousness research.

Detects and characterizes transitions between consciousness states,
including phase transitions and hysteresis effects.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StateTransition:
    """Represents a state transition event.

    Attributes:
        from_state: Source state.
        to_state: Target state.
        timestamp: Time of transition.
        duration: Transition duration.
        smoothness: Transition smoothness (0-1).
        features: Transition features.
    """

    from_state: str
    to_state: str
    timestamp: float
    duration: float
    smoothness: float
    features: dict[str, float]


@dataclass
class TransitionAnalysis:
    """Complete transition analysis results.

    Attributes:
        transitions: List of detected transitions.
        transition_matrix: State transition probability matrix.
        mean_smoothness: Mean transition smoothness.
        hysteresis: Hysteresis score.
        dominant_path: Most common transition sequence.
    """

    transitions: list[StateTransition]
    transition_matrix: np.ndarray
    mean_smoothness: float
    hysteresis: float
    dominant_path: list[str]


class TransitionDetector:
    """Detect and characterize state transitions.

    Identifies transitions between consciousness states and
    analyzes their characteristics including smoothness,
    duration, and hysteresis effects.

    Example:
        >>> detector = TransitionDetector()
        >>> analysis = detector.analyze(embeddings, timestamps, labels)
        >>> print(f"Detected {len(analysis.transitions)} transitions")
    """

    def __init__(
        self,
        threshold: float = 0.3,
        min_duration: float = 0.5,
    ):
        """Initialize transition detector.

        Args:
            threshold: Distance threshold for transition detection.
            min_duration: Minimum time between transitions (seconds).
        """
        self.threshold = threshold
        self.min_duration = min_duration

    def analyze(
        self,
        embeddings: np.ndarray,
        timestamps: np.ndarray,
        labels: list[str] | None = None,
    ) -> TransitionAnalysis:
        """Analyze state transitions.

        Args:
            embeddings: Embedding sequence (samples, features).
            timestamps: Time values for each sample.
            labels: Optional state labels.

        Returns:
            TransitionAnalysis results.
        """
        # Detect transitions
        transitions = self._detect_transitions(embeddings, timestamps, labels)

        # Build transition matrix
        if labels:
            transition_matrix, states = self._build_transition_matrix(labels)
        else:
            transition_matrix = np.array([[1.0]])
            states = ["unknown"]

        # Compute metrics
        if transitions:
            mean_smoothness = np.mean([t.smoothness for t in transitions])
        else:
            mean_smoothness = 1.0

        # Analyze hysteresis
        hysteresis = self._analyze_hysteresis(embeddings, labels)

        # Find dominant path
        dominant_path = self._find_dominant_path(labels) if labels else []

        return TransitionAnalysis(
            transitions=transitions,
            transition_matrix=transition_matrix,
            mean_smoothness=mean_smoothness,
            hysteresis=hysteresis,
            dominant_path=dominant_path,
        )

    def _detect_transitions(
        self,
        embeddings: np.ndarray,
        timestamps: np.ndarray,
        labels: list[str] | None,
    ) -> list[StateTransition]:
        """Detect state transitions from embeddings.

        Args:
            embeddings: Embedding sequence.
            timestamps: Time values.
            labels: State labels.

        Returns:
            List of detected transitions.
        """
        transitions = []
        n_samples = len(embeddings)

        if n_samples < 2:
            return transitions

        # Compute velocities (embedding changes)
        velocities = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
        time_diffs = np.diff(timestamps)
        time_diffs[time_diffs == 0] = 1e-10

        speeds = velocities / time_diffs

        # Detect transitions by velocity spikes
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        threshold_speed = mean_speed + self.threshold * std_speed

        last_transition_time = timestamps[0]

        for i in range(len(speeds)):
            if speeds[i] > threshold_speed:
                if timestamps[i + 1] - last_transition_time >= self.min_duration:
                    # Transition detected
                    from_state = labels[i] if labels else f"state_{i}"
                    to_state = labels[i + 1] if labels else f"state_{i+1}"

                    # Estimate duration (time for speed to return to normal)
                    duration = self._estimate_duration(speeds, i, time_diffs)

                    # Compute smoothness
                    smoothness = self._compute_smoothness(
                        embeddings, i, min(i + 5, n_samples - 1)
                    )

                    # Extract features
                    features = {
                        "velocity": float(speeds[i]),
                        "acceleration": float(
                            (speeds[i] - speeds[i-1]) / time_diffs[i-1]
                            if i > 0 else 0
                        ),
                        "distance": float(velocities[i]),
                    }

                    transitions.append(StateTransition(
                        from_state=from_state,
                        to_state=to_state,
                        timestamp=float(timestamps[i]),
                        duration=duration,
                        smoothness=smoothness,
                        features=features,
                    ))

                    last_transition_time = timestamps[i + 1]

        return transitions

    def _estimate_duration(
        self,
        speeds: np.ndarray,
        start_idx: int,
        time_diffs: np.ndarray,
    ) -> float:
        """Estimate transition duration.

        Args:
            speeds: Speed sequence.
            start_idx: Transition start index.
            time_diffs: Time differences.

        Returns:
            Duration in seconds.
        """
        mean_speed = np.mean(speeds)
        duration = 0.0

        for i in range(start_idx, len(speeds)):
            duration += time_diffs[i] if i < len(time_diffs) else 0
            if speeds[i] < mean_speed:
                break

        return max(duration, time_diffs[start_idx] if start_idx < len(time_diffs) else 0.1)

    def _compute_smoothness(
        self,
        embeddings: np.ndarray,
        start: int,
        end: int,
    ) -> float:
        """Compute transition smoothness.

        Smooth transitions have low jerk (rate of acceleration change).

        Args:
            embeddings: Embedding sequence.
            start: Start index.
            end: End index.

        Returns:
            Smoothness score (0-1).
        """
        if end - start < 3:
            return 0.5

        segment = embeddings[start:end + 1]

        # Compute displacements, velocities, accelerations
        displacements = np.diff(segment, axis=0)
        velocities = np.linalg.norm(displacements, axis=1)

        if len(velocities) < 2:
            return 0.5

        accelerations = np.diff(velocities)

        if len(accelerations) < 1:
            return 0.5

        # Jerk (rate of acceleration change)
        jerk = np.diff(accelerations) if len(accelerations) > 1 else np.array([0])

        # Smoothness = inverse of jerk magnitude
        jerk_magnitude = np.mean(np.abs(jerk))
        smoothness = 1 / (1 + jerk_magnitude)

        return float(np.clip(smoothness, 0, 1))

    def _build_transition_matrix(
        self,
        labels: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Build state transition probability matrix.

        Args:
            labels: Sequence of state labels.

        Returns:
            Tuple of (transition_matrix, state_list).
        """
        unique_states = sorted(set(labels))
        n_states = len(unique_states)
        state_to_idx = {s: i for i, s in enumerate(unique_states)}

        # Count transitions
        counts = np.zeros((n_states, n_states))

        for i in range(len(labels) - 1):
            from_idx = state_to_idx[labels[i]]
            to_idx = state_to_idx[labels[i + 1]]
            counts[from_idx, to_idx] += 1

        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = counts / row_sums

        return matrix, unique_states

    def _analyze_hysteresis(
        self,
        embeddings: np.ndarray,
        labels: list[str] | None,
    ) -> float:
        """Analyze hysteresis in state transitions.

        Hysteresis means the path from A→B differs from B→A,
        indicating state-dependent thresholds.

        Args:
            embeddings: Embedding sequence.
            labels: State labels.

        Returns:
            Hysteresis score (0-1).
        """
        if labels is None or len(set(labels)) < 2:
            return 0.0

        # Find transitions A→B and B→A
        forward_transitions = []
        backward_transitions = []

        unique_pairs = set()
        for i in range(len(labels) - 1):
            if labels[i] != labels[i + 1]:
                pair = (labels[i], labels[i + 1])
                reverse = (labels[i + 1], labels[i])

                if reverse in unique_pairs:
                    # Found a pair of forward/backward transitions
                    # Compare their characteristics
                    forward_dist = np.linalg.norm(
                        embeddings[i + 1] - embeddings[i]
                    )
                    forward_transitions.append(forward_dist)
                else:
                    unique_pairs.add(pair)
                    backward_transitions.append(
                        np.linalg.norm(embeddings[i + 1] - embeddings[i])
                    )

        if not forward_transitions or not backward_transitions:
            return 0.0

        # Hysteresis = asymmetry between forward and backward
        mean_forward = np.mean(forward_transitions)
        mean_backward = np.mean(backward_transitions)

        asymmetry = abs(mean_forward - mean_backward) / (
            mean_forward + mean_backward + 1e-10
        )

        return float(np.clip(asymmetry, 0, 1))

    def _find_dominant_path(self, labels: list[str]) -> list[str]:
        """Find most common transition sequence.

        Args:
            labels: Sequence of state labels.

        Returns:
            Most common path (sequence of states).
        """
        if len(labels) < 3:
            return labels

        # Find most common trigrams
        trigrams: dict[tuple[str, str, str], int] = {}

        for i in range(len(labels) - 2):
            trigram = (labels[i], labels[i + 1], labels[i + 2])
            trigrams[trigram] = trigrams.get(trigram, 0) + 1

        if not trigrams:
            return labels[:3] if len(labels) >= 3 else labels

        # Most common trigram
        dominant = max(trigrams, key=trigrams.get)

        return list(dominant)


def compute_transition_smoothness(
    embeddings: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> float:
    """Compute overall transition smoothness.

    Args:
        embeddings: Embedding sequence.
        timestamps: Optional timestamps.

    Returns:
        Smoothness score (0-1).
    """
    if timestamps is None:
        timestamps = np.arange(len(embeddings))

    detector = TransitionDetector()
    analysis = detector.analyze(embeddings, timestamps)

    return analysis.mean_smoothness


def detect_phase_transition_points(
    embeddings: np.ndarray,
    threshold_percentile: float = 95,
) -> list[int]:
    """Detect points where phase transitions occur.

    Args:
        embeddings: Embedding sequence.
        threshold_percentile: Percentile for threshold.

    Returns:
        List of transition point indices.
    """
    if len(embeddings) < 2:
        return []

    # Compute velocities
    velocities = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)

    # Threshold
    threshold = np.percentile(velocities, threshold_percentile)

    # Find peaks above threshold
    points = np.where(velocities > threshold)[0].tolist()

    return points


def compute_transition_entropy(labels: list[str]) -> float:
    """Compute entropy of state transitions.

    Higher entropy = more unpredictable transitions.

    Args:
        labels: Sequence of state labels.

    Returns:
        Transition entropy.
    """
    if len(labels) < 2:
        return 0.0

    # Count transitions
    transitions: dict[tuple[str, str], int] = {}
    total = 0

    for i in range(len(labels) - 1):
        key = (labels[i], labels[i + 1])
        transitions[key] = transitions.get(key, 0) + 1
        total += 1

    if total == 0:
        return 0.0

    # Compute entropy
    entropy = 0.0
    for count in transitions.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


class HysteresisAnalyzer:
    """Analyze hysteresis effects in consciousness transitions.

    Hysteresis indicates that the path to a state depends
    on history, not just current conditions.

    Example:
        >>> analyzer = HysteresisAnalyzer()
        >>> score = analyzer.analyze(embeddings, labels)
    """

    def __init__(self, n_neighbors: int = 5):
        """Initialize hysteresis analyzer.

        Args:
            n_neighbors: Neighbors for local analysis.
        """
        self.n_neighbors = n_neighbors

    def analyze(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> dict[str, Any]:
        """Analyze hysteresis in state transitions.

        Args:
            embeddings: Embedding sequence.
            labels: State labels.

        Returns:
            Hysteresis analysis results.
        """
        # Find state boundaries
        boundaries = self._find_boundaries(embeddings, labels)

        # Compute asymmetry at boundaries
        asymmetries = []
        for state_pair, indices in boundaries.items():
            if len(indices) >= 2:
                # Compare entry and exit points
                asymmetry = self._compute_boundary_asymmetry(
                    embeddings, indices, labels
                )
                asymmetries.append(asymmetry)

        global_hysteresis = np.mean(asymmetries) if asymmetries else 0.0

        return {
            "global_hysteresis": float(global_hysteresis),
            "boundary_count": len(boundaries),
            "state_pairs": list(boundaries.keys()),
        }

    def _find_boundaries(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> dict[tuple[str, str], list[int]]:
        """Find state transition boundaries.

        Args:
            embeddings: Embedding sequence.
            labels: State labels.

        Returns:
            Dictionary mapping state pairs to boundary indices.
        """
        boundaries: dict[tuple[str, str], list[int]] = {}

        for i in range(len(labels) - 1):
            if labels[i] != labels[i + 1]:
                pair = (labels[i], labels[i + 1])
                if pair not in boundaries:
                    boundaries[pair] = []
                boundaries[pair].append(i)

        return boundaries

    def _compute_boundary_asymmetry(
        self,
        embeddings: np.ndarray,
        indices: list[int],
        labels: list[str],
    ) -> float:
        """Compute asymmetry at state boundaries.

        Args:
            embeddings: Embedding sequence.
            indices: Boundary indices.
            labels: State labels.

        Returns:
            Asymmetry score.
        """
        if len(indices) < 2:
            return 0.0

        # Compare boundary characteristics
        distances = []
        for idx in indices:
            if idx > 0 and idx < len(embeddings) - 1:
                dist = np.linalg.norm(embeddings[idx + 1] - embeddings[idx])
                distances.append(dist)

        if len(distances) < 2:
            return 0.0

        # Asymmetry = coefficient of variation
        return float(np.std(distances) / (np.mean(distances) + 1e-10))
