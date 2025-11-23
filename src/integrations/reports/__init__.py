"""Report generation for consciousness research.

Provides comprehensive report generation in multiple formats
including PDF, HTML, and DOCX.
"""

from .generator import (
    ReportGenerator,
    SessionReport,
    AnalysisReport,
    generate_session_report,
    generate_analysis_report,
)
from .templates import (
    ReportTemplate,
    SessionTemplate,
    AnalysisTemplate,
    ComparisonTemplate,
)
from .exporters import (
    PDFExporter,
    HTMLExporter,
    MarkdownExporter,
    DataExporter,
    export_figures,
)

__all__ = [
    # Generator
    "ReportGenerator",
    "SessionReport",
    "AnalysisReport",
    "generate_session_report",
    "generate_analysis_report",
    # Templates
    "ReportTemplate",
    "SessionTemplate",
    "AnalysisTemplate",
    "ComparisonTemplate",
    # Exporters
    "PDFExporter",
    "HTMLExporter",
    "MarkdownExporter",
    "DataExporter",
    "export_figures",
]
