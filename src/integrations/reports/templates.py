"""Report templates for consciousness research.

Provides customizable templates for different report types
including session summaries, analysis results, and comparisons.
"""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


@dataclass
class ReportTemplate:
    """Base report template.

    Attributes:
        name: Template name.
        description: Template description.
        sections: List of section names.
        css_style: CSS styling.
        header_html: Header HTML template.
        footer_html: Footer HTML template.
    """

    name: str
    description: str
    sections: list[str] = field(default_factory=list)
    css_style: str = ""
    header_html: str = ""
    footer_html: str = ""

    def render(self, data: dict[str, Any]) -> str:
        """Render template with data.

        Args:
            data: Data to render.

        Returns:
            Rendered HTML string.
        """
        raise NotImplementedError("Subclasses must implement render()")


class SessionTemplate(ReportTemplate):
    """Template for session reports.

    Generates comprehensive session summaries with
    signal quality, metrics, and event markers.
    """

    def __init__(self):
        """Initialize session template."""
        super().__init__(
            name="Session Report",
            description="Comprehensive EEG session summary",
            sections=[
                "header",
                "session_info",
                "signal_quality",
                "metrics",
                "markers",
                "figures",
                "footer",
            ],
        )

        self.css_style = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1 {
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
        }
        h2 {
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .info-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #1f77b4;
        }
        .info-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        .info-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            min-width: 150px;
            text-align: center;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
        }
        .metric-label {
            font-size: 11px;
            opacity: 0.9;
            text-transform: uppercase;
            margin-top: 5px;
        }
        .quality-bar {
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        .quality-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .quality-good { background: #28a745; }
        .quality-medium { background: #ffc107; }
        .quality-poor { background: #dc3545; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .marker-tag {
            display: inline-block;
            padding: 3px 8px;
            background: #e3f2fd;
            border-radius: 3px;
            font-size: 12px;
            color: #1565c0;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #666;
            font-size: 12px;
        }
        """

    def render(self, data: dict[str, Any]) -> str:
        """Render session report.

        Args:
            data: Session data dictionary.

        Returns:
            Rendered HTML string.
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data.get('title', 'Session Report')}</title>
    <style>{self.css_style}</style>
</head>
<body>
    <div class="report-container">
        <h1>{data.get('title', 'EEG Session Report')}</h1>

        <h2>Session Information</h2>
        <div class="info-grid">
            <div class="info-card">
                <div class="info-label">Session ID</div>
                <div class="info-value">{data.get('session_id', 'N/A')}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Start Time</div>
                <div class="info-value">{data.get('start_time', 'N/A')}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Duration</div>
                <div class="info-value">{data.get('duration', 0):.1f}s</div>
            </div>
            <div class="info-card">
                <div class="info-label">Device</div>
                <div class="info-value">{data.get('device', 'Unknown')}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Scenario</div>
                <div class="info-value">{data.get('scenario', 'Free Recording')}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Sample Rate</div>
                <div class="info-value">{data.get('sample_rate', 256)} Hz</div>
            </div>
        </div>

        <h2>Signal Quality</h2>
"""

        # Quality section
        quality = data.get('quality_summary', {})
        if quality:
            html += '<div class="info-grid">'
            for channel, score in quality.items():
                if isinstance(score, (int, float)):
                    score_pct = score * 100 if score <= 1 else score
                    quality_class = (
                        'quality-good' if score_pct >= 70 else
                        'quality-medium' if score_pct >= 40 else
                        'quality-poor'
                    )
                    html += f"""
            <div class="info-card">
                <div class="info-label">{channel}</div>
                <div class="quality-bar">
                    <div class="quality-fill {quality_class}" style="width: {score_pct}%"></div>
                </div>
                <div class="info-value">{score_pct:.0f}%</div>
            </div>
"""
            html += '</div>'
        else:
            html += '<p>No quality data available.</p>'

        # Metrics section
        html += """
        <h2>Session Metrics</h2>
        <div class="metric-container">
"""
        metrics = data.get('metrics', {})
        for key, value in metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            label = key.replace('_', ' ').title()
            html += f"""
            <div class="metric-box">
                <div class="metric-value">{value_str}</div>
                <div class="metric-label">{label}</div>
            </div>
"""
        html += '</div>'

        # Markers section
        html += """
        <h2>Event Markers</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Event</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
"""
        markers = data.get('markers', [])
        if markers:
            for marker in markers:
                time_val = marker.get('time', 'N/A')
                label = marker.get('label', 'Unknown')
                details = marker.get('details', '')
                html += f"""
                <tr>
                    <td>{time_val}</td>
                    <td><span class="marker-tag">{label}</span></td>
                    <td>{details}</td>
                </tr>
"""
        else:
            html += '<tr><td colspan="3">No markers recorded.</td></tr>'

        html += """
            </tbody>
        </table>

        <div class="footer">
            <p>Generated by Consciousness Research Workbench</p>
            <p>Report generated on {timestamp}</p>
        </div>
    </div>
</body>
</html>
""".format(timestamp=data.get('generated_at', 'N/A'))

        return html


class AnalysisTemplate(ReportTemplate):
    """Template for analysis reports.

    Generates detailed analysis reports with methods,
    results, statistics, and conclusions.
    """

    def __init__(self):
        """Initialize analysis template."""
        super().__init__(
            name="Analysis Report",
            description="Detailed analysis results",
            sections=[
                "header",
                "analysis_info",
                "methods",
                "results",
                "statistics",
                "figures",
                "conclusions",
                "footer",
            ],
        )

        self.css_style = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f4f8;
        }
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-radius: 10px;
        }
        h1 {
            color: #2d3748;
            margin-bottom: 5px;
        }
        .subtitle {
            color: #718096;
            font-size: 14px;
            margin-bottom: 30px;
        }
        h2 {
            color: #4a5568;
            font-size: 20px;
            margin-top: 35px;
            display: flex;
            align-items: center;
        }
        h2::before {
            content: '';
            width: 4px;
            height: 20px;
            background: #4299e1;
            margin-right: 10px;
            border-radius: 2px;
        }
        .method-list {
            list-style: none;
            padding: 0;
        }
        .method-list li {
            padding: 10px 15px;
            background: #edf2f7;
            margin: 8px 0;
            border-radius: 5px;
            border-left: 3px solid #4299e1;
        }
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .result-card {
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        .result-key {
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 8px;
        }
        .result-value {
            color: #4a5568;
            font-size: 15px;
        }
        .stats-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 15px 0;
        }
        .stats-table th {
            background: #4299e1;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .stats-table th:first-child {
            border-radius: 8px 0 0 0;
        }
        .stats-table th:last-child {
            border-radius: 0 8px 0 0;
        }
        .stats-table td {
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }
        .stats-table tr:last-child td:first-child {
            border-radius: 0 0 0 8px;
        }
        .stats-table tr:last-child td:last-child {
            border-radius: 0 0 8px 0;
        }
        .conclusion-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .conclusion-box p {
            margin: 0;
            line-height: 1.6;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e2e8f0;
            text-align: center;
            color: #718096;
            font-size: 12px;
        }
        """

    def render(self, data: dict[str, Any]) -> str:
        """Render analysis report.

        Args:
            data: Analysis data dictionary.

        Returns:
            Rendered HTML string.
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data.get('title', 'Analysis Report')}</title>
    <style>{self.css_style}</style>
</head>
<body>
    <div class="report-container">
        <h1>{data.get('title', 'Analysis Report')}</h1>
        <div class="subtitle">
            {data.get('analysis_type', 'General Analysis')} |
            {data.get('timestamp', 'N/A')}
        </div>

        <h2>Methods</h2>
        <ul class="method-list">
"""

        methods = data.get('methods', [])
        if methods:
            for method in methods:
                html += f'            <li>{method}</li>\n'
        else:
            html += '            <li>No methods specified</li>\n'

        html += """
        </ul>

        <h2>Results</h2>
        <div class="result-grid">
"""

        results = data.get('results', {})
        for key, value in results.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            elif isinstance(value, dict):
                value_str = ', '.join(f"{k}: {v}" for k, v in value.items())
            elif isinstance(value, list):
                value_str = ', '.join(str(v) for v in value[:5])
                if len(value) > 5:
                    value_str += f" ... (+{len(value) - 5} more)"
            else:
                value_str = str(value)

            html += f"""
            <div class="result-card">
                <div class="result-key">{key.replace('_', ' ').title()}</div>
                <div class="result-value">{value_str}</div>
            </div>
"""

        html += """
        </div>

        <h2>Statistics</h2>
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
"""

        statistics = data.get('statistics', {})
        if statistics:
            for key, value in statistics.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value_str}</td>
                </tr>
"""
        else:
            html += '<tr><td colspan="2">No statistics available</td></tr>'

        html += """
            </tbody>
        </table>

        <h2>Conclusions</h2>
"""

        conclusions = data.get('conclusions', [])
        if conclusions:
            for conclusion in conclusions:
                html += f"""
        <div class="conclusion-box">
            <p>{conclusion}</p>
        </div>
"""
        else:
            html += '<p>No conclusions drawn.</p>'

        html += f"""

        <div class="footer">
            <p>Generated by Consciousness Research Workbench</p>
            <p>Analysis completed: {data.get('timestamp', 'N/A')}</p>
        </div>
    </div>
</body>
</html>
"""

        return html


class ComparisonTemplate(ReportTemplate):
    """Template for comparison reports.

    Generates side-by-side comparisons of multiple
    sessions or analyses.
    """

    def __init__(self):
        """Initialize comparison template."""
        super().__init__(
            name="Comparison Report",
            description="Multi-session/analysis comparison",
            sections=[
                "header",
                "comparison_summary",
                "detailed_comparison",
                "charts",
                "conclusions",
                "footer",
            ],
        )

        self.css_style = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .report-container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1 {
            color: #1f77b4;
            text-align: center;
        }
        h2 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .comparison-table th {
            background: #1f77b4;
            color: white;
            padding: 15px;
            text-align: left;
        }
        .comparison-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }
        .comparison-table tr:nth-child(even) {
            background: #f8f9fa;
        }
        .comparison-table tr:hover {
            background: #e3f2fd;
        }
        .metric-label {
            font-weight: 600;
            color: #333;
        }
        .better {
            color: #28a745;
            font-weight: bold;
        }
        .worse {
            color: #dc3545;
        }
        .neutral {
            color: #666;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .summary-title {
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }
        .summary-value {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
            margin: 10px 0;
        }
        .conclusion-item {
            padding: 15px;
            background: #e3f2fd;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #1f77b4;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #666;
            font-size: 12px;
        }
        """

    def render(self, data: dict[str, Any]) -> str:
        """Render comparison report.

        Args:
            data: Comparison data dictionary with 'items' list.

        Returns:
            Rendered HTML string.
        """
        items = data.get('items', [])
        metrics = data.get('metrics', [])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data.get('title', 'Comparison Report')}</title>
    <style>{self.css_style}</style>
</head>
<body>
    <div class="report-container">
        <h1>{data.get('title', 'Comparison Report')}</h1>

        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Items Compared</div>
                <div class="summary-value">{len(items)}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Metrics Analyzed</div>
                <div class="summary-value">{len(metrics)}</div>
            </div>
        </div>

        <h2>Detailed Comparison</h2>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
"""

        # Add column headers for each item
        for item in items:
            name = item.get('name', 'Unknown')
            html += f'                    <th>{name}</th>\n'

        html += """
                </tr>
            </thead>
            <tbody>
"""

        # Add rows for each metric
        for metric in metrics:
            html += f"""
                <tr>
                    <td class="metric-label">{metric.replace('_', ' ').title()}</td>
"""
            # Find best value for this metric
            values = []
            for item in items:
                val = item.get('values', {}).get(metric, 'N/A')
                if isinstance(val, (int, float)):
                    values.append(val)
                else:
                    values.append(None)

            # Determine best (assuming higher is better for now)
            best_val = max([v for v in values if v is not None], default=None)

            for i, item in enumerate(items):
                val = item.get('values', {}).get(metric, 'N/A')
                if isinstance(val, float):
                    val_str = f"{val:.4f}"
                else:
                    val_str = str(val)

                # Highlight best value
                if values[i] == best_val and best_val is not None:
                    css_class = 'better'
                else:
                    css_class = 'neutral'

                html += f'                    <td class="{css_class}">{val_str}</td>\n'

            html += '                </tr>\n'

        html += """
            </tbody>
        </table>

        <h2>Conclusions</h2>
"""

        conclusions = data.get('conclusions', [])
        if conclusions:
            for conclusion in conclusions:
                html += f"""
        <div class="conclusion-item">{conclusion}</div>
"""
        else:
            html += '<p>No conclusions provided.</p>'

        html += f"""

        <div class="footer">
            <p>Generated by Consciousness Research Workbench</p>
            <p>Comparison generated: {data.get('timestamp', 'N/A')}</p>
        </div>
    </div>
</body>
</html>
"""

        return html


def get_template(template_type: str) -> ReportTemplate:
    """Get a report template by type.

    Args:
        template_type: Type of template ("session", "analysis", "comparison").

    Returns:
        ReportTemplate instance.

    Raises:
        ValueError: If template type is unknown.
    """
    templates = {
        "session": SessionTemplate,
        "analysis": AnalysisTemplate,
        "comparison": ComparisonTemplate,
    }

    if template_type not in templates:
        raise ValueError(f"Unknown template type: {template_type}")

    return templates[template_type]()


def list_templates() -> list[dict[str, str]]:
    """List available templates.

    Returns:
        List of template info dictionaries.
    """
    return [
        {
            "type": "session",
            "name": "Session Report",
            "description": "Comprehensive EEG session summary",
        },
        {
            "type": "analysis",
            "name": "Analysis Report",
            "description": "Detailed analysis results",
        },
        {
            "type": "comparison",
            "name": "Comparison Report",
            "description": "Multi-session/analysis comparison",
        },
    ]
