#!/usr/bin/env python3
"""Visualize consciousness states in 3D latent space.

Creates interactive 3D visualizations of consciousness
state embeddings for exploration and analysis.

Usage:
    python scripts/visualize_latent_space.py embeddings.npz
    python scripts/visualize_latent_space.py --demo
    python scripts/visualize_latent_space.py --app
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def generate_demo_embeddings(
    n_points: int = 500,
    n_states: int = 4,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Generate demo embedding data.

    Args:
        n_points: Total number of points.
        n_states: Number of consciousness states.

    Returns:
        Tuple of (embeddings, labels, timestamps).
    """
    # State names
    state_names = ["focus", "relaxed", "anxious", "creative", "flow", "drowsy"]
    states = state_names[:n_states]

    # Generate clustered embeddings
    points_per_state = n_points // n_states
    embeddings = []
    labels = []

    for i, state in enumerate(states):
        # Each state has a different center
        center = np.array([
            np.cos(2 * np.pi * i / n_states) * 2,
            np.sin(2 * np.pi * i / n_states) * 2,
            np.random.randn() * 0.5,
        ])

        # Generate points around center
        state_points = center + np.random.randn(points_per_state, 3) * 0.5
        embeddings.append(state_points)
        labels.extend([state] * points_per_state)

    embeddings = np.vstack(embeddings)

    # Generate timestamps (simulate recording over time)
    timestamps = np.linspace(0, 300, len(labels))  # 5 minutes

    # Add some temporal structure (transitions)
    shuffle_idx = np.argsort(timestamps + np.random.randn(len(timestamps)) * 10)
    embeddings = embeddings[shuffle_idx]
    labels = [labels[i] for i in shuffle_idx]

    return embeddings, labels, timestamps


def main():
    """Visualize latent space embeddings."""
    parser = argparse.ArgumentParser(
        description="Visualize consciousness states in 3D latent space"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="Input embeddings file (NPZ format)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output HTML file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo data",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Launch interactive Streamlit app",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Show temporal trajectory",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animated trajectory",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=500,
        help="Number of points for demo (default: 500)",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=4,
        help="Number of states for demo (default: 4)",
    )

    args = parser.parse_args()

    # Launch Streamlit app
    if args.app:
        from src.integrations.visualization import LatentSpaceApp
        app = LatentSpaceApp()
        app.run()
        return

    # Load or generate data
    if args.demo:
        embeddings, labels, timestamps = generate_demo_embeddings(
            args.n_points, args.n_states
        )
        print(f"Generated demo data: {len(embeddings)} points, {args.n_states} states")
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        data = np.load(input_path, allow_pickle=True)
        embeddings = data.get('embeddings', data.get('arr_0'))
        labels = data.get('labels', None)
        if labels is not None:
            labels = labels.tolist()
        timestamps = data.get('timestamps', None)

        print(f"Loaded embeddings: {embeddings.shape}")
    else:
        print("Error: Provide input file, use --demo, or use --app")
        parser.print_help()
        sys.exit(1)

    # Create viewer
    from src.integrations.visualization import (
        LatentSpaceViewer,
        create_3d_scatter,
        create_trajectory_plot,
    )

    viewer = LatentSpaceViewer()
    viewer.set_data(embeddings, labels, timestamps)

    # Generate visualization
    print("\nCreating visualization...")

    if args.animate and timestamps is not None:
        print("Creating animated trajectory...")
        fig = viewer.animate_trajectory(duration=10.0)
    elif args.trajectory and timestamps is not None:
        print("Creating trajectory plot...")
        fig = viewer.create_figure(show_trajectory=True)
    else:
        print("Creating 3D scatter plot...")
        fig = viewer.create_figure()

    # Display or save
    if args.output:
        output_path = Path(args.output)
        fig.write_html(str(output_path))
        print(f"\nVisualization saved to: {output_path}")
        print(f"Open in browser to interact with the 3D plot.")
    else:
        # Show in browser
        print("\nOpening in browser...")
        fig.show()

    # Print statistics
    print("\n" + "=" * 50)
    print("Embedding Statistics:")
    print("=" * 50)
    print(f"Total points: {len(embeddings)}")

    if labels:
        unique_labels = list(set(labels))
        print(f"States: {', '.join(unique_labels)}")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  - {label}: {count} points ({count/len(labels)*100:.1f}%)")

    print(f"\nDimensions:")
    for i in range(3):
        values = embeddings[:, i]
        print(f"  Dim {i+1}: mean={np.mean(values):.3f}, "
              f"std={np.std(values):.3f}, "
              f"range=[{np.min(values):.3f}, {np.max(values):.3f}]")

    if timestamps is not None:
        print(f"\nTime range: {timestamps.min():.1f}s - {timestamps.max():.1f}s")
        print(f"Duration: {timestamps.max() - timestamps.min():.1f}s")


if __name__ == "__main__":
    main()
