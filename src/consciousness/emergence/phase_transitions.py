"""Phase transition detection for consciousness research.

Detects critical points, order parameters, and phase
transitions in neural dynamics.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PhaseTransitionMetrics:
    """Metrics for phase transition analysis.

    Attributes:
        order_parameter: System order parameter.
        susceptibility: Response to perturbations.
        correlation_length: Spatial correlation length.
        critical_point: Distance to critical point.
        phase: Identified phase ("ordered", "critical", "disordered").
    """

    order_parameter: float
    susceptibility: float
    correlation_length: float
    critical_point: float
    phase: str


class PhaseTransitionDetector:
    """Detect phase transitions in neural dynamics.

    Identifies critical points where the system undergoes
    qualitative changes, relevant for consciousness transitions.

    Example:
        >>> detector = PhaseTransitionDetector()
        >>> metrics = detector.analyze(eeg_data)
        >>> print(f"Phase: {metrics.phase}")
    """

    def __init__(self, fs: float = 256.0):
        """Initialize phase transition detector.

        Args:
            fs: Sampling frequency in Hz.
        """
        self.fs = fs

    def analyze(self, data: np.ndarray) -> PhaseTransitionMetrics:
        """Analyze phase transition indicators.

        Args:
            data: Neural data (channels, samples).

        Returns:
            PhaseTransitionMetrics object.
        """
        # Compute order parameter
        order_parameter = self._compute_order_parameter(data)

        # Compute susceptibility
        susceptibility = self._compute_susceptibility(data)

        # Compute correlation length
        correlation_length = self._compute_correlation_length(data)

        # Estimate distance to critical point
        critical_point = self._estimate_criticality(
            order_parameter, susceptibility, correlation_length
        )

        # Identify phase
        phase = self._identify_phase(
            order_parameter, susceptibility, critical_point
        )

        return PhaseTransitionMetrics(
            order_parameter=order_parameter,
            susceptibility=susceptibility,
            correlation_length=correlation_length,
            critical_point=critical_point,
            phase=phase,
        )

    def _compute_order_parameter(self, data: np.ndarray) -> float:
        """Compute order parameter.

        The order parameter measures the degree of
        organization in the system.

        Args:
            data: Neural data (channels, samples).

        Returns:
            Order parameter (0-1).
        """
        n_channels = data.shape[0]

        # Order parameter: synchronization across channels
        # Use Kuramoto-like order parameter

        # Compute analytic signal for each channel
        from scipy.signal import hilbert

        phases = []
        for i in range(n_channels):
            analytic = hilbert(data[i])
            phase = np.angle(analytic)
            phases.append(phase)

        phases = np.array(phases)

        # Order parameter = magnitude of mean phase vector
        mean_phase = np.mean(np.exp(1j * phases), axis=0)
        r = np.abs(mean_phase)

        return float(np.mean(r))

    def _compute_susceptibility(self, data: np.ndarray) -> float:
        """Compute susceptibility.

        Susceptibility measures how the system responds
        to small perturbations. High near critical points.

        Args:
            data: Neural data (channels, samples).

        Returns:
            Susceptibility value.
        """
        n_samples = data.shape[1]

        # Compute variance of order parameter over time
        window_size = int(0.5 * self.fs)  # 500ms windows

        if n_samples < 2 * window_size:
            return 0.0

        order_params = []

        for i in range(0, n_samples - window_size, window_size // 2):
            window = data[:, i:i + window_size]
            op = self._compute_order_parameter(window)
            order_params.append(op)

        if len(order_params) < 2:
            return 0.0

        # Susceptibility = variance of order parameter
        # (normalized by mean to get relative fluctuations)
        mean_op = np.mean(order_params)
        var_op = np.var(order_params)

        if mean_op > 0:
            susceptibility = var_op / mean_op
        else:
            susceptibility = 0.0

        return float(susceptibility)

    def _compute_correlation_length(self, data: np.ndarray) -> float:
        """Compute spatial correlation length.

        Correlation length diverges at critical points.

        Args:
            data: Neural data (channels, samples).

        Returns:
            Normalized correlation length (0-1).
        """
        n_channels = data.shape[0]

        if n_channels < 2:
            return 0.0

        # Compute correlation matrix
        corr_matrix = np.corrcoef(data)
        np.fill_diagonal(corr_matrix, 0)

        # Correlation length ~ how far correlations extend
        # Use spectral radius as proxy
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        # Normalize
        correlation_length = max_eigenvalue / n_channels

        return float(np.clip(correlation_length, 0, 1))

    def _estimate_criticality(
        self,
        order_parameter: float,
        susceptibility: float,
        correlation_length: float,
    ) -> float:
        """Estimate distance to critical point.

        Args:
            order_parameter: Order parameter value.
            susceptibility: Susceptibility value.
            correlation_length: Correlation length.

        Returns:
            Criticality score (0-1, 1 = at critical point).
        """
        # At critical point:
        # - Order parameter is intermediate (~0.5)
        # - Susceptibility is high
        # - Correlation length is high

        # Score how close to critical
        op_score = 1 - 2 * abs(order_parameter - 0.5)  # Peaks at 0.5
        susc_score = np.tanh(susceptibility)  # Saturates at high values
        corr_score = correlation_length

        criticality = (op_score + susc_score + corr_score) / 3

        return float(np.clip(criticality, 0, 1))

    def _identify_phase(
        self,
        order_parameter: float,
        susceptibility: float,
        critical_point: float,
    ) -> str:
        """Identify the current phase.

        Args:
            order_parameter: Order parameter value.
            susceptibility: Susceptibility value.
            critical_point: Criticality score.

        Returns:
            Phase name.
        """
        if critical_point > 0.7:
            return "critical"
        elif order_parameter > 0.7:
            return "ordered"
        elif order_parameter < 0.3:
            return "disordered"
        else:
            return "transitional"

    def detect_critical_transitions(
        self,
        data: np.ndarray,
        window_size: float = 2.0,
        step_size: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Detect critical transitions over time.

        Args:
            data: Neural data (channels, samples).
            window_size: Window size in seconds.
            step_size: Step size in seconds.

        Returns:
            List of detected critical points.
        """
        window_samples = int(window_size * self.fs)
        step_samples = int(step_size * self.fs)
        n_samples = data.shape[1]

        critical_points = []

        prev_phase = None
        prev_criticality = 0

        for i, start in enumerate(range(0, n_samples - window_samples, step_samples)):
            window = data[:, start:start + window_samples]
            metrics = self.analyze(window)

            # Detect phase change
            if prev_phase is not None and metrics.phase != prev_phase:
                # Check if it's a significant transition
                if abs(metrics.critical_point - prev_criticality) > 0.2:
                    critical_points.append({
                        "index": i,
                        "time": start / self.fs,
                        "from_phase": prev_phase,
                        "to_phase": metrics.phase,
                        "criticality": metrics.critical_point,
                    })

            prev_phase = metrics.phase
            prev_criticality = metrics.critical_point

        return critical_points


def compute_avalanche_statistics(
    data: np.ndarray,
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Compute neuronal avalanche statistics.

    Avalanche statistics can indicate criticality
    (power-law distributions).

    Args:
        data: Neural data (channels, samples).
        threshold: Z-score threshold for activity.

    Returns:
        Avalanche statistics.
    """
    # Binarize activity
    z_scores = (data - np.mean(data, axis=1, keepdims=True)) / (
        np.std(data, axis=1, keepdims=True) + 1e-10
    )
    active = np.abs(z_scores) > threshold

    # Detect avalanches (connected active regions)
    total_active = np.sum(active, axis=0)

    avalanche_sizes = []
    avalanche_durations = []

    in_avalanche = False
    current_size = 0
    current_duration = 0

    for i in range(len(total_active)):
        if total_active[i] > 0:
            if not in_avalanche:
                in_avalanche = True
                current_size = 0
                current_duration = 0

            current_size += total_active[i]
            current_duration += 1
        else:
            if in_avalanche:
                avalanche_sizes.append(current_size)
                avalanche_durations.append(current_duration)
                in_avalanche = False

    # Handle ongoing avalanche
    if in_avalanche:
        avalanche_sizes.append(current_size)
        avalanche_durations.append(current_duration)

    if not avalanche_sizes:
        return {
            "n_avalanches": 0,
            "mean_size": 0,
            "mean_duration": 0,
            "power_law_exponent": 0,
        }

    # Estimate power law exponent (simple method)
    sizes = np.array(avalanche_sizes)
    if len(sizes) > 10 and np.min(sizes) > 0:
        # Maximum likelihood estimator for power law
        alpha = 1 + len(sizes) / np.sum(np.log(sizes / np.min(sizes)))
    else:
        alpha = 0

    return {
        "n_avalanches": len(avalanche_sizes),
        "mean_size": float(np.mean(avalanche_sizes)),
        "mean_duration": float(np.mean(avalanche_durations)),
        "power_law_exponent": float(alpha),
        "sizes": avalanche_sizes,
        "durations": avalanche_durations,
    }


def compute_edge_of_chaos(data: np.ndarray) -> float:
    """Estimate proximity to edge of chaos.

    The edge of chaos is thought to be optimal for
    information processing in neural systems.

    Args:
        data: Neural data (channels, samples).

    Returns:
        Edge of chaos score (0-1, 1 = at edge).
    """
    detector = PhaseTransitionDetector()
    metrics = detector.analyze(data)

    # Edge of chaos = critical point
    return metrics.critical_point


def construct_phase_diagram(
    data: np.ndarray,
    parameter_range: np.ndarray,
    fs: float = 256.0,
) -> dict[str, Any]:
    """Construct a phase diagram by varying parameters.

    Constructs diagram by varying analysis parameters
    to map out phase space.

    Args:
        data: Neural data.
        parameter_range: Range of parameter values.
        fs: Sampling frequency.

    Returns:
        Phase diagram data.
    """
    detector = PhaseTransitionDetector(fs=fs)

    order_params = []
    susceptibilities = []
    phases = []

    # Vary window size as control parameter
    for param in parameter_range:
        window_size = param
        window_samples = int(window_size * fs)

        if window_samples > data.shape[1]:
            continue

        # Use first window
        window = data[:, :window_samples]
        metrics = detector.analyze(window)

        order_params.append(metrics.order_parameter)
        susceptibilities.append(metrics.susceptibility)
        phases.append(metrics.phase)

    return {
        "parameters": parameter_range[:len(order_params)].tolist(),
        "order_parameters": order_params,
        "susceptibilities": susceptibilities,
        "phases": phases,
    }
