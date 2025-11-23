"""Report exporters for multiple formats.

Provides export functionality for PDF, HTML, Markdown,
DOCX, and various data formats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import csv

import numpy as np


@dataclass
class ExportResult:
    """Result of export operation.

    Attributes:
        success: Whether export succeeded.
        path: Path to exported file.
        format: Export format used.
        size_bytes: File size in bytes.
        message: Status message.
    """

    success: bool
    path: Path | None
    format: str
    size_bytes: int = 0
    message: str = ""


class PDFExporter:
    """Export reports to PDF format.

    Uses HTML to PDF conversion for high-quality output.

    Example:
        >>> exporter = PDFExporter()
        >>> result = exporter.export(html_content, "report.pdf")
    """

    def __init__(self, page_size: str = "A4"):
        """Initialize PDF exporter.

        Args:
            page_size: Page size ("A4", "Letter").
        """
        self.page_size = page_size

    def export(
        self,
        content: str,
        output_path: str | Path,
        title: str = "Report",
    ) -> ExportResult:
        """Export HTML content to PDF.

        Args:
            content: HTML content string.
            output_path: Output file path.
            title: Document title.

        Returns:
            ExportResult object.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Try weasyprint first
            try:
                from weasyprint import HTML
                HTML(string=content).write_pdf(str(output_path))
            except ImportError:
                # Fall back to saving as HTML with PDF extension note
                html_path = output_path.with_suffix('.html')
                html_path.write_text(content)
                return ExportResult(
                    success=True,
                    path=html_path,
                    format="html",
                    size_bytes=html_path.stat().st_size,
                    message="PDF export requires weasyprint. Saved as HTML.",
                )

            return ExportResult(
                success=True,
                path=output_path,
                format="pdf",
                size_bytes=output_path.stat().st_size,
                message="PDF exported successfully",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                path=None,
                format="pdf",
                message=f"PDF export failed: {str(e)}",
            )


class HTMLExporter:
    """Export reports to HTML format.

    Generates standalone HTML files with embedded styles.

    Example:
        >>> exporter = HTMLExporter()
        >>> result = exporter.export(content, "report.html")
    """

    def __init__(self, minify: bool = False):
        """Initialize HTML exporter.

        Args:
            minify: Whether to minify output.
        """
        self.minify = minify

    def export(
        self,
        content: str,
        output_path: str | Path,
    ) -> ExportResult:
        """Export to HTML file.

        Args:
            content: HTML content string.
            output_path: Output file path.

        Returns:
            ExportResult object.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.minify:
                # Simple minification
                content = ' '.join(content.split())

            output_path.write_text(content)

            return ExportResult(
                success=True,
                path=output_path,
                format="html",
                size_bytes=output_path.stat().st_size,
                message="HTML exported successfully",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                path=None,
                format="html",
                message=f"HTML export failed: {str(e)}",
            )


class MarkdownExporter:
    """Export reports to Markdown format.

    Converts HTML or structured data to Markdown.

    Example:
        >>> exporter = MarkdownExporter()
        >>> result = exporter.export(data, "report.md")
    """

    def export(
        self,
        data: dict[str, Any],
        output_path: str | Path,
        report_type: str = "session",
    ) -> ExportResult:
        """Export to Markdown file.

        Args:
            data: Report data dictionary.
            output_path: Output file path.
            report_type: Type of report.

        Returns:
            ExportResult object.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if report_type == "session":
                md = self._session_to_markdown(data)
            elif report_type == "analysis":
                md = self._analysis_to_markdown(data)
            else:
                md = self._generic_to_markdown(data)

            output_path.write_text(md)

            return ExportResult(
                success=True,
                path=output_path,
                format="markdown",
                size_bytes=output_path.stat().st_size,
                message="Markdown exported successfully",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                path=None,
                format="markdown",
                message=f"Markdown export failed: {str(e)}",
            )

    def _session_to_markdown(self, data: dict[str, Any]) -> str:
        """Convert session data to Markdown.

        Args:
            data: Session data.

        Returns:
            Markdown string.
        """
        md = f"# {data.get('title', 'EEG Session Report')}\n\n"

        md += "## Session Information\n\n"
        md += f"- **Session ID**: {data.get('session_id', 'N/A')}\n"
        md += f"- **Start Time**: {data.get('start_time', 'N/A')}\n"
        md += f"- **Duration**: {data.get('duration', 0):.1f} seconds\n"
        md += f"- **Device**: {data.get('device', 'Unknown')}\n"
        md += f"- **Scenario**: {data.get('scenario', 'Free Recording')}\n"
        md += f"- **Sample Rate**: {data.get('sample_rate', 256)} Hz\n\n"

        # Quality
        md += "## Signal Quality\n\n"
        quality = data.get('quality_summary', {})
        if quality:
            md += "| Channel | Quality |\n"
            md += "|---------|--------|\n"
            for channel, score in quality.items():
                if isinstance(score, (int, float)):
                    pct = score * 100 if score <= 1 else score
                    md += f"| {channel} | {pct:.0f}% |\n"
        else:
            md += "No quality data available.\n"
        md += "\n"

        # Metrics
        md += "## Session Metrics\n\n"
        metrics = data.get('metrics', {})
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                md += f"- **{key.replace('_', ' ').title()}**: {value_str}\n"
        else:
            md += "No metrics computed.\n"
        md += "\n"

        # Markers
        md += "## Event Markers\n\n"
        markers = data.get('markers', [])
        if markers:
            md += "| Time | Event | Details |\n"
            md += "|------|-------|--------|\n"
            for marker in markers:
                time_val = marker.get('time', 'N/A')
                label = marker.get('label', 'Unknown')
                details = marker.get('details', '')
                md += f"| {time_val} | {label} | {details} |\n"
        else:
            md += "No markers recorded.\n"
        md += "\n"

        md += "---\n"
        md += "*Generated by Consciousness Research Workbench*\n"

        return md

    def _analysis_to_markdown(self, data: dict[str, Any]) -> str:
        """Convert analysis data to Markdown.

        Args:
            data: Analysis data.

        Returns:
            Markdown string.
        """
        md = f"# {data.get('title', 'Analysis Report')}\n\n"

        md += f"**Type**: {data.get('analysis_type', 'General')}\n"
        md += f"**Timestamp**: {data.get('timestamp', 'N/A')}\n\n"

        # Methods
        md += "## Methods\n\n"
        methods = data.get('methods', [])
        if methods:
            for method in methods:
                md += f"- {method}\n"
        else:
            md += "No methods specified.\n"
        md += "\n"

        # Results
        md += "## Results\n\n"
        results = data.get('results', {})
        for key, value in results.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            md += f"### {key.replace('_', ' ').title()}\n\n"
            md += f"{value_str}\n\n"

        # Statistics
        md += "## Statistics\n\n"
        statistics = data.get('statistics', {})
        if statistics:
            md += "| Metric | Value |\n"
            md += "|--------|-------|\n"
            for key, value in statistics.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                md += f"| {key.replace('_', ' ').title()} | {value_str} |\n"
        md += "\n"

        # Conclusions
        md += "## Conclusions\n\n"
        conclusions = data.get('conclusions', [])
        if conclusions:
            for i, conclusion in enumerate(conclusions, 1):
                md += f"{i}. {conclusion}\n"
        else:
            md += "No conclusions drawn.\n"
        md += "\n"

        md += "---\n"
        md += "*Generated by Consciousness Research Workbench*\n"

        return md

    def _generic_to_markdown(self, data: dict[str, Any]) -> str:
        """Convert generic data to Markdown.

        Args:
            data: Data dictionary.

        Returns:
            Markdown string.
        """
        md = f"# {data.get('title', 'Report')}\n\n"

        for key, value in data.items():
            if key == 'title':
                continue

            md += f"## {key.replace('_', ' ').title()}\n\n"

            if isinstance(value, dict):
                for k, v in value.items():
                    md += f"- **{k}**: {v}\n"
            elif isinstance(value, list):
                for item in value:
                    md += f"- {item}\n"
            else:
                md += f"{value}\n"

            md += "\n"

        return md


class DataExporter:
    """Export data to various formats.

    Supports CSV, Excel, JSON, and HDF5.

    Example:
        >>> exporter = DataExporter()
        >>> exporter.to_csv(data, "output.csv")
        >>> exporter.to_excel(data, "output.xlsx")
    """

    def to_csv(
        self,
        data: dict[str, Any] | np.ndarray,
        output_path: str | Path,
        headers: list[str] | None = None,
    ) -> ExportResult:
        """Export data to CSV.

        Args:
            data: Data to export (dict or array).
            output_path: Output file path.
            headers: Column headers.

        Returns:
            ExportResult object.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(data, np.ndarray):
                # Array data
                if headers is None:
                    headers = [f"col_{i}" for i in range(data.shape[1] if data.ndim > 1 else 1)]

                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    if data.ndim == 1:
                        for val in data:
                            writer.writerow([val])
                    else:
                        writer.writerows(data)

            elif isinstance(data, dict):
                # Dictionary data
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)

                    # Check if values are lists/arrays
                    if all(isinstance(v, (list, np.ndarray)) for v in data.values()):
                        # Column-oriented data
                        writer.writerow(data.keys())
                        rows = zip(*data.values())
                        writer.writerows(rows)
                    else:
                        # Key-value pairs
                        writer.writerow(['Key', 'Value'])
                        for key, value in data.items():
                            writer.writerow([key, value])

            return ExportResult(
                success=True,
                path=output_path,
                format="csv",
                size_bytes=output_path.stat().st_size,
                message="CSV exported successfully",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                path=None,
                format="csv",
                message=f"CSV export failed: {str(e)}",
            )

    def to_excel(
        self,
        data: dict[str, Any],
        output_path: str | Path,
        sheet_name: str = "Data",
    ) -> ExportResult:
        """Export data to Excel.

        Args:
            data: Data to export.
            output_path: Output file path.
            sheet_name: Worksheet name.

        Returns:
            ExportResult object.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            try:
                import openpyxl
                from openpyxl import Workbook
            except ImportError:
                return ExportResult(
                    success=False,
                    path=None,
                    format="excel",
                    message="Excel export requires openpyxl",
                )

            wb = Workbook()
            ws = wb.active
            ws.title = sheet_name

            # Write data
            if isinstance(data, dict):
                # Check if values are lists/arrays
                if all(isinstance(v, (list, np.ndarray)) for v in data.values()):
                    # Column-oriented data
                    for col, (key, values) in enumerate(data.items(), 1):
                        ws.cell(row=1, column=col, value=key)
                        for row, val in enumerate(values, 2):
                            if isinstance(val, (np.integer, np.floating)):
                                val = float(val)
                            ws.cell(row=row, column=col, value=val)
                else:
                    # Key-value pairs
                    ws.cell(row=1, column=1, value="Key")
                    ws.cell(row=1, column=2, value="Value")
                    for row, (key, value) in enumerate(data.items(), 2):
                        ws.cell(row=row, column=1, value=key)
                        if isinstance(value, (np.integer, np.floating)):
                            value = float(value)
                        elif isinstance(value, (list, np.ndarray)):
                            value = str(value)
                        ws.cell(row=row, column=2, value=value)

            wb.save(output_path)

            return ExportResult(
                success=True,
                path=output_path,
                format="excel",
                size_bytes=output_path.stat().st_size,
                message="Excel exported successfully",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                path=None,
                format="excel",
                message=f"Excel export failed: {str(e)}",
            )

    def to_json(
        self,
        data: dict[str, Any],
        output_path: str | Path,
        indent: int = 2,
    ) -> ExportResult:
        """Export data to JSON.

        Args:
            data: Data to export.
            output_path: Output file path.
            indent: JSON indentation.

        Returns:
            ExportResult object.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=indent, default=str)

            return ExportResult(
                success=True,
                path=output_path,
                format="json",
                size_bytes=output_path.stat().st_size,
                message="JSON exported successfully",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                path=None,
                format="json",
                message=f"JSON export failed: {str(e)}",
            )

    def to_hdf5(
        self,
        data: dict[str, np.ndarray],
        output_path: str | Path,
    ) -> ExportResult:
        """Export data to HDF5.

        Args:
            data: Dictionary of arrays.
            output_path: Output file path.

        Returns:
            ExportResult object.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            try:
                import h5py
            except ImportError:
                return ExportResult(
                    success=False,
                    path=None,
                    format="hdf5",
                    message="HDF5 export requires h5py",
                )

            with h5py.File(output_path, 'w') as f:
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value)
                    else:
                        # Try to convert to array
                        f.create_dataset(key, data=np.array(value))

            return ExportResult(
                success=True,
                path=output_path,
                format="hdf5",
                size_bytes=output_path.stat().st_size,
                message="HDF5 exported successfully",
            )

        except Exception as e:
            return ExportResult(
                success=False,
                path=None,
                format="hdf5",
                message=f"HDF5 export failed: {str(e)}",
            )


def export_figures(
    figures: list[Any],
    output_dir: str | Path,
    format: str = "png",
    dpi: int = 150,
) -> list[ExportResult]:
    """Export matplotlib figures.

    Args:
        figures: List of matplotlib figures.
        output_dir: Output directory.
        format: Image format ("png", "svg", "pdf").
        dpi: Resolution for raster formats.

    Returns:
        List of ExportResult objects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, fig in enumerate(figures):
        output_path = output_dir / f"figure_{i+1:02d}.{format}"

        try:
            # Get title from figure if available
            title = ""
            if hasattr(fig, '_suptitle') and fig._suptitle:
                title = fig._suptitle.get_text()

            fig.savefig(
                output_path,
                format=format,
                dpi=dpi,
                bbox_inches='tight',
            )

            results.append(ExportResult(
                success=True,
                path=output_path,
                format=format,
                size_bytes=output_path.stat().st_size,
                message=f"Figure exported: {title or f'figure_{i+1}'}",
            ))

        except Exception as e:
            results.append(ExportResult(
                success=False,
                path=None,
                format=format,
                message=f"Figure export failed: {str(e)}",
            ))

    return results


def export_interactive_html(
    figure: Any,
    output_path: str | Path,
) -> ExportResult:
    """Export plotly figure to interactive HTML.

    Args:
        figure: Plotly figure object.
        output_path: Output file path.

    Returns:
        ExportResult object.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        try:
            import plotly
        except ImportError:
            return ExportResult(
                success=False,
                path=None,
                format="html",
                message="Interactive export requires plotly",
            )

        figure.write_html(str(output_path))

        return ExportResult(
            success=True,
            path=output_path,
            format="html",
            size_bytes=output_path.stat().st_size,
            message="Interactive HTML exported successfully",
        )

    except Exception as e:
        return ExportResult(
            success=False,
            path=None,
            format="html",
            message=f"Interactive export failed: {str(e)}",
        )
