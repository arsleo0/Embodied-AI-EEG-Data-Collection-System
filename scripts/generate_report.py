#!/usr/bin/env python3
"""Generate reports from EEG session data.

Creates comprehensive reports in HTML, JSON, Markdown,
or PDF formats from recorded sessions.

Usage:
    python scripts/generate_report.py session_data.json
    python scripts/generate_report.py session_data.json -o report.html
    python scripts/generate_report.py session_data.json --format pdf
    python scripts/generate_report.py --demo
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.integrations.reports import (
    ReportGenerator,
    generate_session_report,
    generate_analysis_report,
)
from src.integrations.reports.templates import (
    SessionTemplate,
    AnalysisTemplate,
    ComparisonTemplate,
)
from src.integrations.reports.exporters import (
    PDFExporter,
    HTMLExporter,
    MarkdownExporter,
    DataExporter,
)


def generate_demo_session_data() -> dict:
    """Generate demo session data.

    Returns:
        Session data dictionary.
    """
    return {
        "title": "Demo EEG Session Report",
        "session_id": "demo_20231115_001",
        "start_time": datetime.now().isoformat(),
        "duration": 300.0,
        "device": "Muse 2",
        "scenario": "Meditation Practice",
        "channels": ["TP9", "AF7", "AF8", "TP10"],
        "sample_rate": 256,
        "n_samples": 76800,
        "quality_summary": {
            "TP9": 0.85,
            "AF7": 0.92,
            "AF8": 0.88,
            "TP10": 0.79,
        },
        "markers": [
            {"time": 0.0, "label": "session_start", "details": "Recording started"},
            {"time": 60.0, "label": "eyes_closed", "details": "Begin meditation"},
            {"time": 180.0, "label": "deep_focus", "details": "Deep focus achieved"},
            {"time": 300.0, "label": "session_end", "details": "Recording ended"},
        ],
        "metrics": {
            "duration_minutes": 5.0,
            "samples_per_channel": 76800,
            "mean_quality": 0.86,
            "min_quality": 0.79,
            "n_markers": 4,
            "alpha_power_ratio": 0.35,
            "theta_power_ratio": 0.22,
        },
        "generated_at": datetime.now().isoformat(),
    }


def generate_demo_analysis_data() -> dict:
    """Generate demo analysis data.

    Returns:
        Analysis data dictionary.
    """
    return {
        "title": "Consciousness State Analysis",
        "analysis_type": "State Classification",
        "timestamp": datetime.now().isoformat(),
        "input_data": {
            "source": "demo_session",
            "duration": "5 minutes",
            "channels": 4,
        },
        "methods": [
            "Band power extraction (Delta, Theta, Alpha, Beta, Gamma)",
            "Spectral entropy calculation",
            "Cross-channel coherence analysis",
            "Random Forest classification",
            "UMAP dimensionality reduction",
        ],
        "results": {
            "dominant_state": "focus",
            "state_probabilities": {
                "focus": 0.65,
                "relaxed": 0.25,
                "drowsy": 0.10,
            },
            "alpha_peak_frequency": 10.2,
            "theta_alpha_ratio": 0.63,
        },
        "statistics": {
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89,
            "f1_score": 0.87,
            "classification_confidence": 0.82,
        },
        "conclusions": [
            "The session showed predominantly focused attention states with high alpha activity.",
            "Signal quality was good across all channels (mean: 86%).",
            "State transitions were smooth with minimal noise contamination.",
            "Recommend continued monitoring to establish baseline patterns.",
        ],
    }


def main():
    """Generate report from session data."""
    parser = argparse.ArgumentParser(
        description="Generate reports from EEG session data"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="Input data file (JSON)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "json", "markdown", "pdf"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["session", "analysis"],
        default="session",
        help="Report type (default: session)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate demo report with sample data",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["default", "minimal", "detailed"],
        default="default",
        help="Report template style",
    )

    args = parser.parse_args()

    # Load or generate data
    if args.demo:
        if args.type == "session":
            data = generate_demo_session_data()
        else:
            data = generate_demo_analysis_data()
        print("Using demo data...")
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        with open(input_path) as f:
            data = json.load(f)
        print(f"Loaded data from {input_path}")
    else:
        print("Error: Provide input file or use --demo flag")
        parser.print_help()
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "md" if args.format == "markdown" else args.format
        output_path = Path(f"{args.type}_report_{timestamp}.{ext}")

    # Generate report
    print(f"\nGenerating {args.type} report...")
    print(f"Format: {args.format}")
    print(f"Output: {output_path}")

    generator = ReportGenerator()

    if args.type == "session":
        report = generator.create_session_report(data)
    else:
        report = generator.create_analysis_report(data)

    # Export
    if args.format == "html":
        result = generator.export_html(report, output_path)
    elif args.format == "json":
        result = generator.export_json(report, output_path)
    elif args.format == "markdown":
        exporter = MarkdownExporter()
        result = exporter.export(data, output_path, report_type=args.type)
        if not result.success:
            print(f"Error: {result.message}")
            sys.exit(1)
    elif args.format == "pdf":
        # First generate HTML, then convert to PDF
        template = SessionTemplate() if args.type == "session" else AnalysisTemplate()
        html_content = template.render(data)

        exporter = PDFExporter()
        result = exporter.export(html_content, output_path)
        if not result.success:
            print(f"Note: {result.message}")

    print(f"\nReport generated successfully!")
    print(f"Output: {output_path}")

    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
