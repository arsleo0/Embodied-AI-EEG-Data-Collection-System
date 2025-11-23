"""Main dashboard application.

Streamlit-based dashboard for real-time EEG monitoring
and consciousness state visualization.
"""

from pathlib import Path
from typing import Any
import numpy as np

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_dashboard():
    """Create and configure the Streamlit dashboard.

    Returns:
        Dashboard configuration dict.
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError(
            "streamlit required. Install with: pip install streamlit"
        )

    return {
        "title": "Consciousness Research Workbench",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }


def run_dashboard():
    """Run the main dashboard application."""
    if not STREAMLIT_AVAILABLE:
        raise ImportError("streamlit required")

    # Page config
    st.set_page_config(
        page_title="Consciousness Research Workbench",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-bad { color: #dc3545; }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">🧠 Consciousness Research Workbench</p>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Control Panel")

        # Device selection
        device_type = st.selectbox(
            "EEG Device",
            ["Muse 2", "Muse S", "Synthetic"],
            index=2
        )

        # Connection status
        if st.button("Connect Device"):
            with st.spinner("Connecting..."):
                # Simulated connection
                import time
                time.sleep(1)
                st.session_state['connected'] = True
                st.success("Connected!")

        # Recording controls
        st.subheader("Recording")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state['recording'] = True
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state['recording'] = False

        # Scenario selection
        st.subheader("Scenario")
        scenario = st.selectbox(
            "Select Scenario",
            ["Free Recording", "Meditation", "Focus Task",
             "Urban Walk", "Creative Flow"]
        )

        # Analysis settings
        st.subheader("Analysis")

        window_size = st.slider(
            "Window Size (s)",
            min_value=1.0,
            max_value=10.0,
            value=2.0
        )

        update_rate = st.slider(
            "Update Rate (Hz)",
            min_value=1,
            max_value=30,
            value=10
        )

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Monitor",
        "🧠 Consciousness",
        "📈 Analysis",
        "📋 Session"
    ])

    # Tab 1: Live Monitor
    with tab1:
        _render_live_monitor()

    # Tab 2: Consciousness State
    with tab2:
        _render_consciousness_tab()

    # Tab 3: Analysis
    with tab3:
        _render_analysis_tab()

    # Tab 4: Session Info
    with tab4:
        _render_session_tab()


def _render_live_monitor():
    """Render the live monitoring tab."""
    st.header("Live EEG Monitor")

    # Signal quality indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "TP9 (L Ear)",
            "Good",
            delta="98%",
            delta_color="normal"
        )
    with col2:
        st.metric(
            "AF7 (L Front)",
            "Good",
            delta="95%",
            delta_color="normal"
        )
    with col3:
        st.metric(
            "AF8 (R Front)",
            "Fair",
            delta="78%",
            delta_color="off"
        )
    with col4:
        st.metric(
            "TP10 (R Ear)",
            "Good",
            delta="96%",
            delta_color="normal"
        )

    # EEG signal plot
    if PLOTLY_AVAILABLE:
        # Generate synthetic data for demo
        t = np.linspace(0, 5, 1280)
        channels = ["TP9", "AF7", "AF8", "TP10"]

        fig = go.Figure()

        for i, ch in enumerate(channels):
            # Synthetic EEG-like signal
            signal = (
                20 * np.sin(2 * np.pi * 10 * t) +  # Alpha
                10 * np.sin(2 * np.pi * 20 * t) +  # Beta
                np.random.randn(len(t)) * 5
            )
            signal = signal - i * 100  # Offset for display

            fig.add_trace(go.Scatter(
                x=t,
                y=signal,
                name=ch,
                line=dict(width=1)
            ))

        fig.update_layout(
            title="Real-time EEG Signals",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (μV)",
            height=400,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Install plotly for interactive charts")

    # Band power display
    st.subheader("Band Power")

    col1, col2, col3, col4, col5 = st.columns(5)

    bands = [
        ("Delta", "1-4 Hz", 15),
        ("Theta", "4-8 Hz", 25),
        ("Alpha", "8-13 Hz", 35),
        ("Beta", "13-30 Hz", 20),
        ("Gamma", "30-50 Hz", 5)
    ]

    for col, (name, freq, power) in zip(
        [col1, col2, col3, col4, col5], bands
    ):
        with col:
            st.metric(name, f"{power}%", freq)


def _render_consciousness_tab():
    """Render the consciousness state tab."""
    st.header("Consciousness State Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Latent space visualization
        if PLOTLY_AVAILABLE:
            # Generate synthetic latent space
            np.random.seed(42)
            n_points = 100

            # Create clustered points
            states = ["Meditation", "Focus", "Neutral", "Flow"]
            colors = ["blue", "green", "gray", "purple"]

            fig = go.Figure()

            for i, (state, color) in enumerate(zip(states, colors)):
                x = np.random.randn(n_points // 4) + i * 2
                y = np.random.randn(n_points // 4) + (i % 2) * 2

                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name=state,
                    marker=dict(size=8, color=color, opacity=0.7)
                ))

            # Current state marker
            fig.add_trace(go.Scatter(
                x=[4], y=[2],
                mode='markers',
                name='Current',
                marker=dict(size=15, color='red', symbol='star')
            ))

            fig.update_layout(
                title="Consciousness Latent Space",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                height=400,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # State metrics
        st.subheader("Current State")

        st.markdown("""
            <div class="metric-card">
                <h3>🧘 Flow State</h3>
                <p>Confidence: 85%</p>
            </div>
        """, unsafe_allow_html=True)

        st.metric("Meta-awareness", "0.72", delta="0.05")
        st.metric("Integration", "0.68", delta="-0.02")
        st.metric("Attention Focus", "0.81", delta="0.08")

        # Emotional state
        st.subheader("Emotional State")
        st.metric("Valence", "+0.4")
        st.metric("Arousal", "0.6")


def _render_analysis_tab():
    """Render the analysis results tab."""
    st.header("Analysis Results")

    # Feature importance
    if PLOTLY_AVAILABLE:
        features = [
            "Alpha Power", "Beta/Theta Ratio", "Frontal Asymmetry",
            "Gamma Coherence", "Theta FM", "Alpha Asymmetry"
        ]
        importance = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]

        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='steelblue'
        ))

        fig.update_layout(
            title="Feature Importance for State Classification",
            xaxis_title="Importance",
            height=300,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Complexity metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Complexity")
        st.metric("Lempel-Ziv", "0.72")
        st.metric("Fractal Dim", "1.65")
        st.metric("Permutation Entropy", "0.84")

    with col2:
        st.subheader("Integration")
        st.metric("Phi (Φ)", "0.42")
        st.metric("Integration", "0.68")
        st.metric("Differentiation", "0.75")

    with col3:
        st.subheader("Criticality")
        st.metric("Edge of Chaos", "0.71")
        st.metric("Order Parameter", "0.55")
        st.metric("Phase", "Critical")


def _render_session_tab():
    """Render the session information tab."""
    st.header("Session Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recording Details")

        st.write("**Start Time:** 2024-01-15 10:30:00")
        st.write("**Duration:** 00:15:32")
        st.write("**Samples:** 238,592")
        st.write("**Device:** Muse 2")
        st.write("**Scenario:** Focus Task")

        # Markers
        st.subheader("Event Markers")
        markers = [
            {"time": "00:00:00", "event": "Session Start"},
            {"time": "00:02:15", "event": "Task Begin"},
            {"time": "00:05:30", "event": "User Marker"},
            {"time": "00:10:00", "event": "Break"},
            {"time": "00:15:00", "event": "Session End"},
        ]

        for m in markers:
            st.write(f"**{m['time']}** - {m['event']}")

    with col2:
        st.subheader("Data Quality")

        quality_data = {
            "Channel": ["TP9", "AF7", "AF8", "TP10"],
            "Good (%)": [92, 88, 75, 94],
            "Artifacts": [3, 5, 12, 2]
        }

        st.dataframe(quality_data, use_container_width=True)

        # Export options
        st.subheader("Export")

        export_format = st.selectbox(
            "Format",
            ["HDF5", "CSV", "Parquet", "EDF"]
        )

        if st.button("Export Data", use_container_width=True):
            st.success(f"Data exported as {export_format}")

        if st.button("Generate Report", use_container_width=True):
            st.success("Report generated: session_report.pdf")


def main():
    """Main entry point for dashboard."""
    run_dashboard()


if __name__ == "__main__":
    main()
