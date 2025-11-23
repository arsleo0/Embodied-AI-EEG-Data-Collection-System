"""Report generation engine.

Generates comprehensive reports from EEG sessions and analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import numpy as np


@dataclass
class SessionReport:
    """Session report data container.

    Attributes:
        title: Report title.
        session_id: Unique session identifier.
        start_time: Session start time.
        duration: Session duration in seconds.
        device: EEG device used.
        scenario: Scenario name.
        channels: Channel names.
        sample_rate: Sampling frequency.
        n_samples: Total samples recorded.
        quality_summary: Signal quality summary.
        markers: Event markers.
        figures: Generated figures.
        metrics: Session metrics.
    """

    title: str
    session_id: str
    start_time: datetime
    duration: float
    device: str
    scenario: str
    channels: list[str]
    sample_rate: float
    n_samples: int
    quality_summary: dict[str, Any] = field(default_factory=dict)
    markers: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Analysis report data container.

    Attributes:
        title: Report title.
        analysis_type: Type of analysis performed.
        timestamp: Analysis timestamp.
        input_data: Input data description.
        methods: Methods used.
        results: Analysis results.
        statistics: Statistical summary.
        figures: Generated figures.
        conclusions: Analysis conclusions.
    """

    title: str
    analysis_type: str
    timestamp: datetime
    input_data: dict[str, Any]
    methods: list[str]
    results: dict[str, Any]
    statistics: dict[str, Any]
    figures: list[dict[str, Any]] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)


class ReportGenerator:
    """Generate reports from session and analysis data.

    Supports multiple output formats and customizable templates.

    Example:
        >>> generator = ReportGenerator()
        >>> report = generator.create_session_report(session_data)
        >>> generator.export(report, "session_report.html", format="html")
    """

    def __init__(
        self,
        template_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        """Initialize report generator.

        Args:
            template_dir: Directory for report templates.
            output_dir: Default output directory.
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.output_dir = Path(output_dir) if output_dir else Path(".")

    def create_session_report(
        self,
        session_data: dict[str, Any],
        include_figures: bool = True,
    ) -> SessionReport:
        """Create a session report.

        Args:
            session_data: Session data dictionary.
            include_figures: Whether to generate figures.

        Returns:
            SessionReport object.
        """
        # Extract basic info
        report = SessionReport(
            title=session_data.get("title", "EEG Session Report"),
            session_id=session_data.get("session_id", "unknown"),
            start_time=session_data.get("start_time", datetime.now()),
            duration=session_data.get("duration", 0),
            device=session_data.get("device", "Unknown"),
            scenario=session_data.get("scenario", "Free Recording"),
            channels=session_data.get("channels", ["TP9", "AF7", "AF8", "TP10"]),
            sample_rate=session_data.get("sample_rate", 256),
            n_samples=session_data.get("n_samples", 0),
        )

        # Quality summary
        if "quality" in session_data:
            report.quality_summary = session_data["quality"]

        # Markers
        if "markers" in session_data:
            report.markers = session_data["markers"]

        # Compute metrics
        report.metrics = self._compute_session_metrics(session_data)

        # Generate figures
        if include_figures and "data" in session_data:
            report.figures = self._generate_session_figures(session_data)

        return report

    def create_analysis_report(
        self,
        analysis_data: dict[str, Any],
        include_figures: bool = True,
    ) -> AnalysisReport:
        """Create an analysis report.

        Args:
            analysis_data: Analysis results dictionary.
            include_figures: Whether to generate figures.

        Returns:
            AnalysisReport object.
        """
        report = AnalysisReport(
            title=analysis_data.get("title", "Analysis Report"),
            analysis_type=analysis_data.get("type", "General"),
            timestamp=analysis_data.get("timestamp", datetime.now()),
            input_data=analysis_data.get("input", {}),
            methods=analysis_data.get("methods", []),
            results=analysis_data.get("results", {}),
            statistics=analysis_data.get("statistics", {}),
        )

        # Extract conclusions
        if "conclusions" in analysis_data:
            report.conclusions = analysis_data["conclusions"]

        # Generate figures
        if include_figures:
            report.figures = self._generate_analysis_figures(analysis_data)

        return report

    def _compute_session_metrics(
        self,
        session_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute session metrics.

        Args:
            session_data: Session data.

        Returns:
            Metrics dictionary.
        """
        metrics = {
            "duration_minutes": session_data.get("duration", 0) / 60,
            "samples_per_channel": session_data.get("n_samples", 0),
        }

        # Quality metrics
        if "quality" in session_data:
            quality = session_data["quality"]
            if isinstance(quality, dict):
                scores = list(quality.values())
                metrics["mean_quality"] = np.mean(scores) if scores else 0
                metrics["min_quality"] = np.min(scores) if scores else 0

        # Marker count
        if "markers" in session_data:
            metrics["n_markers"] = len(session_data["markers"])

        return metrics

    def _generate_session_figures(
        self,
        session_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate figures for session report.

        Args:
            session_data: Session data.

        Returns:
            List of figure specifications.
        """
        figures = []

        # Signal overview figure
        figures.append({
            "type": "eeg_overview",
            "title": "EEG Signal Overview",
            "description": "Multi-channel EEG recording overview",
        })

        # Quality figure
        figures.append({
            "type": "quality_summary",
            "title": "Signal Quality",
            "description": "Channel quality scores",
        })

        # Band power figure
        figures.append({
            "type": "band_power",
            "title": "Band Power Distribution",
            "description": "Power across frequency bands",
        })

        return figures

    def _generate_analysis_figures(
        self,
        analysis_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate figures for analysis report.

        Args:
            analysis_data: Analysis results.

        Returns:
            List of figure specifications.
        """
        figures = []

        # Result visualization
        figures.append({
            "type": "results_summary",
            "title": "Analysis Results",
            "description": "Summary of analysis results",
        })

        # Statistics figure
        if "statistics" in analysis_data:
            figures.append({
                "type": "statistics",
                "title": "Statistical Summary",
                "description": "Statistical analysis results",
            })

        return figures

    def export_html(
        self,
        report: SessionReport | AnalysisReport,
        output_path: str | Path,
    ) -> Path:
        """Export report to HTML.

        Args:
            report: Report object.
            output_path: Output file path.

        Returns:
            Path to exported file.
        """
        output_path = Path(output_path)

        # Generate HTML content
        if isinstance(report, SessionReport):
            html = self._session_to_html(report)
        else:
            html = self._analysis_to_html(report)

        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

        return output_path

    def _session_to_html(self, report: SessionReport) -> str:
        """Convert session report to HTML.

        Args:
            report: Session report.

        Returns:
            HTML string.
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #1f77b4; }}
        h2 {{ color: #2c3e50; border-bottom: 1px solid #eee; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px;
                   background: #f8f9fa; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>

    <h2>Session Information</h2>
    <table>
        <tr><th>Session ID</th><td>{report.session_id}</td></tr>
        <tr><th>Start Time</th><td>{report.start_time}</td></tr>
        <tr><th>Duration</th><td>{report.duration:.1f} seconds</td></tr>
        <tr><th>Device</th><td>{report.device}</td></tr>
        <tr><th>Scenario</th><td>{report.scenario}</td></tr>
        <tr><th>Sample Rate</th><td>{report.sample_rate} Hz</td></tr>
        <tr><th>Total Samples</th><td>{report.n_samples:,}</td></tr>
    </table>

    <h2>Session Metrics</h2>
    <div class="metrics">
"""

        for key, value in report.metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            html += f"""
        <div class="metric">
            <div class="metric-value">{value_str}</div>
            <div class="metric-label">{key.replace('_', ' ').title()}</div>
        </div>
"""

        html += """
    </div>

    <h2>Event Markers</h2>
    <table>
        <tr><th>Time</th><th>Event</th></tr>
"""

        for marker in report.markers:
            html += f"""
        <tr><td>{marker.get('time', 'N/A')}</td><td>{marker.get('label', 'Unknown')}</td></tr>
"""

        html += """
    </table>

    <footer>
        <p>Generated by Consciousness Research Workbench</p>
    </footer>
</body>
</html>
"""
        return html

    def _analysis_to_html(self, report: AnalysisReport) -> str:
        """Convert analysis report to HTML.

        Args:
            report: Analysis report.

        Returns:
            HTML string.
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #1f77b4; }}
        h2 {{ color: #2c3e50; border-bottom: 1px solid #eee; }}
        .result {{ margin: 10px 0; padding: 10px; background: #f8f9fa; }}
        .conclusion {{ padding: 10px; border-left: 3px solid #1f77b4; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>

    <h2>Analysis Details</h2>
    <p><strong>Type:</strong> {report.analysis_type}</p>
    <p><strong>Timestamp:</strong> {report.timestamp}</p>

    <h2>Methods</h2>
    <ul>
"""

        for method in report.methods:
            html += f"        <li>{method}</li>\n"

        html += """
    </ul>

    <h2>Results</h2>
"""

        for key, value in report.results.items():
            html += f"""
    <div class="result">
        <strong>{key}:</strong> {value}
    </div>
"""

        html += """
    <h2>Conclusions</h2>
"""

        for conclusion in report.conclusions:
            html += f"""
    <div class="conclusion">{conclusion}</div>
"""

        html += """
    <footer>
        <p>Generated by Consciousness Research Workbench</p>
    </footer>
</body>
</html>
"""
        return html

    def export_json(
        self,
        report: SessionReport | AnalysisReport,
        output_path: str | Path,
    ) -> Path:
        """Export report to JSON.

        Args:
            report: Report object.
            output_path: Output file path.

        Returns:
            Path to exported file.
        """
        output_path = Path(output_path)

        # Convert to dict
        if isinstance(report, SessionReport):
            data = {
                "type": "session",
                "title": report.title,
                "session_id": report.session_id,
                "start_time": report.start_time.isoformat(),
                "duration": report.duration,
                "device": report.device,
                "scenario": report.scenario,
                "channels": report.channels,
                "sample_rate": report.sample_rate,
                "n_samples": report.n_samples,
                "quality_summary": report.quality_summary,
                "markers": report.markers,
                "metrics": report.metrics,
            }
        else:
            data = {
                "type": "analysis",
                "title": report.title,
                "analysis_type": report.analysis_type,
                "timestamp": report.timestamp.isoformat(),
                "input_data": report.input_data,
                "methods": report.methods,
                "results": report.results,
                "statistics": report.statistics,
                "conclusions": report.conclusions,
            }

        # Write JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return output_path


def generate_session_report(
    session_data: dict[str, Any],
    output_path: str | Path,
    format: str = "html",
) -> Path:
    """Generate and export a session report.

    Args:
        session_data: Session data dictionary.
        output_path: Output file path.
        format: Output format ("html", "json").

    Returns:
        Path to generated report.
    """
    generator = ReportGenerator()
    report = generator.create_session_report(session_data)

    if format == "html":
        return generator.export_html(report, output_path)
    elif format == "json":
        return generator.export_json(report, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def generate_analysis_report(
    analysis_data: dict[str, Any],
    output_path: str | Path,
    format: str = "html",
) -> Path:
    """Generate and export an analysis report.

    Args:
        analysis_data: Analysis results dictionary.
        output_path: Output file path.
        format: Output format ("html", "json").

    Returns:
        Path to generated report.
    """
    generator = ReportGenerator()
    report = generator.create_analysis_report(analysis_data)

    if format == "html":
        return generator.export_html(report, output_path)
    elif format == "json":
        return generator.export_json(report, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
