"""Tests for TIER 4 user experience and visualization modules."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


class TestDashboardComponents:
    """Tests for dashboard components."""

    def test_streaming_handler(self):
        """Test streaming data handler."""
        from src.integrations.dashboard.live_monitor import StreamingHandler

        handler = StreamingHandler(buffer_size=1000, n_channels=4)

        # Add data
        data = np.random.randn(4, 100)
        handler.add_data(data)

        assert handler.get_sample_count() == 100

        # Get data
        retrieved = handler.get_data(50)
        assert retrieved.shape == (4, 50)

        # Clear
        handler.clear()
        assert handler.get_sample_count() == 0

    def test_streaming_handler_overflow(self):
        """Test buffer overflow handling."""
        from src.integrations.dashboard.live_monitor import StreamingHandler

        handler = StreamingHandler(buffer_size=100, n_channels=2)

        # Add more than buffer
        for _ in range(5):
            data = np.random.randn(2, 50)
            handler.add_data(data)

        # Should not exceed buffer
        assert handler.get_sample_count() <= 100

    def test_live_monitor(self):
        """Test live monitor."""
        from src.integrations.dashboard.live_monitor import LiveMonitor

        monitor = LiveMonitor(n_channels=4)

        # Add data
        data = np.random.randn(4, 256)
        monitor.update(data)

        # Get quality
        quality = monitor.get_quality_scores()
        assert len(quality) == 4
        assert all(0 <= q <= 100 for q in quality.values())

        # Get alerts
        alerts = monitor.get_alerts()
        assert isinstance(alerts, list)


class TestReportGenerator:
    """Tests for report generation."""

    def test_session_report_creation(self):
        """Test session report creation."""
        from src.integrations.reports import ReportGenerator, SessionReport

        generator = ReportGenerator()

        session_data = {
            "title": "Test Session",
            "session_id": "test_001",
            "start_time": datetime.now(),
            "duration": 300.0,
            "device": "Muse 2",
            "scenario": "Test",
            "channels": ["TP9", "AF7", "AF8", "TP10"],
            "sample_rate": 256,
            "n_samples": 76800,
            "quality": {"TP9": 0.9, "AF7": 0.85},
            "markers": [{"time": 0, "label": "start"}],
        }

        report = generator.create_session_report(session_data)

        assert isinstance(report, SessionReport)
        assert report.session_id == "test_001"
        assert report.duration == 300.0

    def test_analysis_report_creation(self):
        """Test analysis report creation."""
        from src.integrations.reports import ReportGenerator, AnalysisReport

        generator = ReportGenerator()

        analysis_data = {
            "title": "Test Analysis",
            "type": "Classification",
            "timestamp": datetime.now(),
            "methods": ["Method 1", "Method 2"],
            "results": {"accuracy": 0.85},
            "statistics": {"mean": 0.5},
            "conclusions": ["Conclusion 1"],
        }

        report = generator.create_analysis_report(analysis_data)

        assert isinstance(report, AnalysisReport)
        assert report.analysis_type == "Classification"
        assert "accuracy" in report.results

    def test_html_export(self):
        """Test HTML export."""
        from src.integrations.reports import ReportGenerator

        generator = ReportGenerator()

        session_data = {
            "title": "Export Test",
            "session_id": "export_001",
            "start_time": datetime.now(),
            "duration": 60.0,
            "device": "Test",
            "scenario": "Test",
            "channels": ["Ch1"],
            "sample_rate": 256,
            "n_samples": 15360,
        }

        report = generator.create_session_report(session_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"
            result = generator.export_html(report, output_path)

            assert result.exists()
            content = result.read_text()
            assert "Export Test" in content
            assert "export_001" in content

    def test_json_export(self):
        """Test JSON export."""
        from src.integrations.reports import ReportGenerator
        import json

        generator = ReportGenerator()

        session_data = {
            "title": "JSON Test",
            "session_id": "json_001",
            "start_time": datetime.now(),
            "duration": 60.0,
            "device": "Test",
            "scenario": "Test",
            "channels": ["Ch1"],
            "sample_rate": 256,
            "n_samples": 15360,
        }

        report = generator.create_session_report(session_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.json"
            result = generator.export_json(report, output_path)

            assert result.exists()

            with open(result) as f:
                data = json.load(f)

            assert data["session_id"] == "json_001"
            assert data["type"] == "session"


class TestReportTemplates:
    """Tests for report templates."""

    def test_session_template(self):
        """Test session template rendering."""
        from src.integrations.reports.templates import SessionTemplate

        template = SessionTemplate()

        data = {
            "title": "Template Test",
            "session_id": "template_001",
            "start_time": "2023-11-15T10:00:00",
            "duration": 300.0,
            "device": "Muse 2",
            "scenario": "Meditation",
            "sample_rate": 256,
            "quality_summary": {"TP9": 0.9, "AF7": 0.85},
            "metrics": {"mean_quality": 0.87},
            "markers": [],
            "generated_at": "2023-11-15T10:05:00",
        }

        html = template.render(data)

        assert "<!DOCTYPE html>" in html
        assert "Template Test" in html
        assert "template_001" in html
        assert "Muse 2" in html

    def test_analysis_template(self):
        """Test analysis template rendering."""
        from src.integrations.reports.templates import AnalysisTemplate

        template = AnalysisTemplate()

        data = {
            "title": "Analysis Test",
            "analysis_type": "Classification",
            "timestamp": "2023-11-15T10:00:00",
            "methods": ["Method A", "Method B"],
            "results": {"accuracy": 0.85},
            "statistics": {"f1": 0.82},
            "conclusions": ["Test conclusion"],
        }

        html = template.render(data)

        assert "<!DOCTYPE html>" in html
        assert "Analysis Test" in html
        assert "Method A" in html
        assert "Test conclusion" in html

    def test_comparison_template(self):
        """Test comparison template rendering."""
        from src.integrations.reports.templates import ComparisonTemplate

        template = ComparisonTemplate()

        data = {
            "title": "Comparison Test",
            "items": [
                {"name": "Session A", "values": {"metric": 0.8}},
                {"name": "Session B", "values": {"metric": 0.9}},
            ],
            "metrics": ["metric"],
            "conclusions": ["B is better"],
            "timestamp": "2023-11-15",
        }

        html = template.render(data)

        assert "<!DOCTYPE html>" in html
        assert "Session A" in html
        assert "Session B" in html

    def test_get_template(self):
        """Test template retrieval."""
        from src.integrations.reports.templates import (
            get_template,
            SessionTemplate,
            AnalysisTemplate,
        )

        session = get_template("session")
        assert isinstance(session, SessionTemplate)

        analysis = get_template("analysis")
        assert isinstance(analysis, AnalysisTemplate)

        with pytest.raises(ValueError):
            get_template("invalid")


class TestExporters:
    """Tests for report exporters."""

    def test_html_exporter(self):
        """Test HTML exporter."""
        from src.integrations.reports.exporters import HTMLExporter

        exporter = HTMLExporter()
        content = "<html><body>Test</body></html>"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.html"
            result = exporter.export(content, output_path)

            assert result.success
            assert result.path.exists()
            assert result.format == "html"

    def test_markdown_exporter_session(self):
        """Test Markdown exporter for session."""
        from src.integrations.reports.exporters import MarkdownExporter

        exporter = MarkdownExporter()

        data = {
            "title": "MD Test",
            "session_id": "md_001",
            "start_time": "2023-11-15",
            "duration": 60.0,
            "device": "Test",
            "scenario": "Test",
            "sample_rate": 256,
            "quality_summary": {"Ch1": 0.9},
            "metrics": {"test": 1.0},
            "markers": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"
            result = exporter.export(data, output_path, report_type="session")

            assert result.success
            content = output_path.read_text()
            assert "# MD Test" in content
            assert "md_001" in content

    def test_data_exporter_csv(self):
        """Test CSV export."""
        from src.integrations.reports.exporters import DataExporter

        exporter = DataExporter()

        data = {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            result = exporter.to_csv(data, output_path)

            assert result.success
            content = output_path.read_text()
            assert "col1" in content
            assert "col2" in content

    def test_data_exporter_json(self):
        """Test JSON export."""
        from src.integrations.reports.exporters import DataExporter
        import json

        exporter = DataExporter()

        data = {"key": "value", "number": 42}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.json"
            result = exporter.to_json(data, output_path)

            assert result.success

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded["key"] == "value"
            assert loaded["number"] == 42

    def test_data_exporter_array(self):
        """Test CSV export with numpy array."""
        from src.integrations.reports.exporters import DataExporter

        exporter = DataExporter()
        data = np.random.randn(10, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "array.csv"
            result = exporter.to_csv(
                data, output_path,
                headers=["x", "y", "z"]
            )

            assert result.success


class TestLatentSpaceViewer:
    """Tests for latent space visualization."""

    def test_viewer_initialization(self):
        """Test viewer initialization."""
        from src.integrations.visualization import LatentSpaceViewer

        viewer = LatentSpaceViewer()
        assert viewer.embeddings is None

    def test_set_data(self):
        """Test setting data."""
        from src.integrations.visualization import LatentSpaceViewer

        viewer = LatentSpaceViewer()

        embeddings = np.random.randn(100, 3)
        labels = ["state_a"] * 50 + ["state_b"] * 50
        timestamps = np.linspace(0, 100, 100)

        viewer.set_data(embeddings, labels, timestamps)

        assert viewer.embeddings is not None
        assert len(viewer.labels) == 100
        assert len(viewer.timestamps) == 100

    def test_set_data_invalid_dimensions(self):
        """Test setting data with wrong dimensions."""
        from src.integrations.visualization import LatentSpaceViewer

        viewer = LatentSpaceViewer()
        embeddings = np.random.randn(100, 5)  # Should be 3D

        with pytest.raises(ValueError):
            viewer.set_data(embeddings)

    def test_create_3d_scatter(self):
        """Test 3D scatter plot creation."""
        from src.integrations.visualization import create_3d_scatter

        embeddings = np.random.randn(50, 3)
        labels = ["a"] * 25 + ["b"] * 25

        fig = create_3d_scatter(embeddings, labels)

        assert fig is not None
        assert hasattr(fig, 'data')

    def test_create_trajectory_plot(self):
        """Test trajectory plot creation."""
        from src.integrations.visualization import create_trajectory_plot

        embeddings = np.random.randn(50, 3)
        timestamps = np.linspace(0, 100, 50)

        fig = create_trajectory_plot(embeddings, timestamps)

        assert fig is not None

    def test_export_html(self):
        """Test HTML export."""
        from src.integrations.visualization import LatentSpaceViewer

        viewer = LatentSpaceViewer()
        embeddings = np.random.randn(50, 3)
        viewer.set_data(embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "latent.html"
            result = viewer.export_html(output_path)

            assert result.exists()


class TestInteractivePlots:
    """Tests for interactive plotting."""

    def test_time_series_plot(self):
        """Test time series plot."""
        from src.integrations.visualization import TimeSeriesPlot

        plotter = TimeSeriesPlot()
        data = np.random.randn(4, 1024)

        fig = plotter.plot(data, fs=256, channels=['TP9', 'AF7', 'AF8', 'TP10'])

        assert fig is not None
        assert len(fig.data) == 4  # 4 channels

    def test_spectrogram_plot(self):
        """Test spectrogram plot."""
        from src.integrations.visualization import SpectrogramPlot

        plotter = SpectrogramPlot()
        data = np.random.randn(1024)

        fig = plotter.plot(data, fs=256)

        assert fig is not None

    def test_band_power_bars(self):
        """Test band power bar chart."""
        from src.integrations.visualization import BandPowerPlot

        plotter = BandPowerPlot()

        powers = {
            'Delta': 0.1,
            'Theta': 0.15,
            'Alpha': 0.35,
            'Beta': 0.25,
            'Gamma': 0.15,
        }

        fig = plotter.plot_bars(powers)

        assert fig is not None
        assert len(fig.data) == 1

    def test_band_power_radar(self):
        """Test band power radar chart."""
        from src.integrations.visualization import BandPowerPlot

        plotter = BandPowerPlot()
        powers = np.array([0.1, 0.15, 0.35, 0.25, 0.15])

        fig = plotter.plot_radar(powers)

        assert fig is not None

    def test_band_power_comparison(self):
        """Test band power comparison."""
        from src.integrations.visualization import BandPowerPlot

        plotter = BandPowerPlot()

        powers1 = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        powers2 = np.array([0.15, 0.25, 0.25, 0.2, 0.15])

        fig = plotter.plot_comparison(
            [powers1, powers2],
            labels=["Session 1", "Session 2"]
        )

        assert fig is not None
        assert len(fig.data) == 2

    def test_topo_plot(self):
        """Test topographic plot."""
        from src.integrations.visualization import TopoPlot

        plotter = TopoPlot()

        values = {'TP9': 0.8, 'AF7': 0.9, 'AF8': 0.85, 'TP10': 0.75}
        fig = plotter.plot(values)

        assert fig is not None

    def test_dashboard_layout(self):
        """Test dashboard layout creation."""
        from src.integrations.visualization import create_dashboard_layout

        data = np.random.randn(4, 2560)
        fig = create_dashboard_layout(
            data,
            fs=256,
            channels=['TP9', 'AF7', 'AF8', 'TP10']
        )

        assert fig is not None


class TestIntegration:
    """Integration tests for TIER 4 modules."""

    def test_full_report_pipeline(self):
        """Test complete report generation pipeline."""
        from src.integrations.reports import (
            ReportGenerator,
            generate_session_report,
        )
        from src.integrations.reports.templates import SessionTemplate
        from src.integrations.reports.exporters import (
            HTMLExporter,
            MarkdownExporter,
        )

        # Create session data
        session_data = {
            "title": "Integration Test Session",
            "session_id": "int_001",
            "start_time": datetime.now(),
            "duration": 300.0,
            "device": "Muse 2",
            "scenario": "Test",
            "channels": ["TP9", "AF7", "AF8", "TP10"],
            "sample_rate": 256,
            "n_samples": 76800,
            "quality": {"TP9": 0.9, "AF7": 0.85, "AF8": 0.88, "TP10": 0.82},
            "markers": [
                {"time": 0, "label": "start"},
                {"time": 300, "label": "end"},
            ],
        }

        # Generate using convenience function
        with tempfile.TemporaryDirectory() as tmpdir:
            # HTML
            html_path = Path(tmpdir) / "report.html"
            result = generate_session_report(session_data, html_path, format="html")
            assert result.exists()

            # JSON
            json_path = Path(tmpdir) / "report.json"
            result = generate_session_report(session_data, json_path, format="json")
            assert result.exists()

            # Markdown via exporter
            md_exporter = MarkdownExporter()
            md_path = Path(tmpdir) / "report.md"
            md_result = md_exporter.export(
                session_data, md_path, report_type="session"
            )
            assert md_result.success

    def test_visualization_to_export(self):
        """Test visualization creation and export."""
        from src.integrations.visualization import (
            LatentSpaceViewer,
            TimeSeriesPlot,
            BandPowerPlot,
        )

        # Generate test data
        embeddings = np.random.randn(100, 3)
        labels = ["focus"] * 50 + ["relaxed"] * 50
        eeg_data = np.random.randn(4, 2560)

        # Create visualizations
        viewer = LatentSpaceViewer()
        viewer.set_data(embeddings, labels)
        latent_fig = viewer.create_figure()

        ts_plotter = TimeSeriesPlot()
        ts_fig = ts_plotter.plot(eeg_data, fs=256)

        bp_plotter = BandPowerPlot()
        bp_fig = bp_plotter.plot_bars(
            {'Alpha': 0.3, 'Beta': 0.25, 'Theta': 0.2}
        )

        # Export all
        with tempfile.TemporaryDirectory() as tmpdir:
            latent_fig.write_html(str(Path(tmpdir) / "latent.html"))
            ts_fig.write_html(str(Path(tmpdir) / "timeseries.html"))
            bp_fig.write_html(str(Path(tmpdir) / "bandpower.html"))

            assert (Path(tmpdir) / "latent.html").exists()
            assert (Path(tmpdir) / "timeseries.html").exists()
            assert (Path(tmpdir) / "bandpower.html").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
