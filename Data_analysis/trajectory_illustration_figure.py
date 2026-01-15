"""
Trajectory Illustration Figure: What β > 0 and β < 0 Mean

This script creates a figure showing example trial trajectories from each
quadrant of the heading_deviation vs cumulative_turn plot, illustrating
the intuitive meaning of the regression slope.

Author: Analysis for Peng et al. 2025
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DATA_PATH = '/workspace/Peng'
RESULTS_PATH = f'{PROJECT_DATA_PATH}/results'
FIGURES_PATH = f'{PROJECT_DATA_PATH}/figures'

os.makedirs(FIGURES_PATH, exist_ok=True)

# Colorblind-friendly palette
COLORS = {
    'light': '#0072B2',      # Blue
    'dark': '#D55E00',       # Orange/Vermillion
    'positive': '#009E73',   # Bluish green
    'negative': '#CC79A7',   # Reddish purple
    'neutral': '#999999',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_cumulative_turn(hd):
    """
    Compute cumulative turn from heading changes throughout a trial.

    Args:
        hd: Array of heading values in radians

    Returns:
        Total cumulative turn in radians (signed)
    """
    if len(hd) < 2:
        return 0

    # Compute wrapped heading differences
    hd_diffs = np.diff(hd)
    hd_diffs = np.arctan2(np.sin(hd_diffs), np.cos(hd_diffs))

    # Sum to get total turn
    return np.sum(hd_diffs)


def get_trial_trajectory(reconstruction_df, session, trial):
    """
    Extract full trajectory for a specific trial.
    """
    mask = (reconstruction_df['session'] == session) & \
           (reconstruction_df['trial'] == trial)
    trial_data = reconstruction_df[mask].copy()

    if len(trial_data) == 0:
        return None

    # Sort by time
    trial_data = trial_data.sort_values('recTime')

    return trial_data


def plot_trajectory_with_gradient(ax, x, y, color='blue', linewidth=2, alpha=0.8):
    """
    Plot trajectory with color gradient showing time progression.
    """
    # Remove any remaining NaN by interpolation
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return None

    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create color array based on time (index)
    norm = plt.Normalize(0, len(x) - 1)

    # Create gradient from light gray to the specified color
    cmap = LinearSegmentedColormap.from_list('traj', ['#CCCCCC', color])

    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
    lc.set_array(np.arange(len(x)))
    lc.set_linewidth(linewidth)

    line = ax.add_collection(lc)

    # Need to set axis limits for LineCollection to show
    ax.autoscale_view()

    return line, x, y


def find_quadrant_examples(endpoints_df, reconstruction_df, condition='all_dark',
                           speed_threshold=5.0, n_candidates=20):
    """
    Find example trials from each quadrant that complete approximately 1 loop.

    Returns dict with keys 'Q1', 'Q2', 'Q3', 'Q4', each containing trial info.
    """
    # Filter endpoints
    mask = (endpoints_df['condition'] == condition) & \
           (endpoints_df['speed_threshold'] == speed_threshold)
    filtered = endpoints_df[mask].copy()

    if len(filtered) == 0:
        print(f"No data for condition={condition}, speed={speed_threshold}")
        return None

    X = filtered['integrated_ang_vel'].values
    Y = filtered['mvtDirError'].values

    # Define quadrants
    quadrants = {
        'Q1': (X > 0) & (Y > 0),  # top-right: + turn, + error
        'Q2': (X < 0) & (Y > 0),  # top-left: - turn, + error
        'Q3': (X < 0) & (Y < 0),  # bottom-left: - turn, - error
        'Q4': (X > 0) & (Y < 0),  # bottom-right: + turn, - error
    }

    results = {}

    for q_name, q_mask in quadrants.items():
        q_data = filtered[q_mask].copy()

        if len(q_data) == 0:
            print(f"No data in {q_name}")
            continue

        # Sort by "extremeness" (distance from origin)
        q_data['distance'] = np.sqrt(q_data['integrated_ang_vel']**2 +
                                     q_data['mvtDirError']**2)
        q_data = q_data.sort_values('distance', ascending=False)

        # Check top candidates for single-loop trials (based on cumulative turn from heading)
        found = False
        TWO_PI = 2 * np.pi

        for idx, row in q_data.head(n_candidates).iterrows():
            session = row['session']
            trial = int(row['trial'])

            # Get full trial trajectory
            trial_traj = get_trial_trajectory(reconstruction_df, session, trial)

            if trial_traj is None or len(trial_traj) < 10:
                continue

            # Get actual positions and heading
            x = trial_traj['xPose'].values
            y = trial_traj['yPose'].values
            hd = trial_traj['hdPose'].values

            # Remove NaN values
            valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(hd))
            x = x[valid]
            y = y[valid]
            hd = hd[valid]

            if len(x) < 10:
                continue

            # Compute cumulative turn from heading changes
            cum_turn = compute_cumulative_turn(hd)
            n_loops = np.abs(cum_turn) / TWO_PI

            # Accept trials with approximately 1 loop (30% tolerance: 0.7 to 1.3)
            if 0.7 <= n_loops <= 1.3:
                results[q_name] = {
                    'session': session,
                    'trial': trial,
                    'integrated_ang_vel': row['integrated_ang_vel'],
                    'mvtDirError': row['mvtDirError'],
                    'cum_turn': cum_turn,
                    'n_loops': n_loops,
                    'x': x,
                    'y': y
                }
                found = True
                print(f"{q_name}: Found trial {session}_T{trial} with {n_loops:.2f} loops ({cum_turn:.2f} rad)")
                break

        if not found:
            # Relax criteria if no single-loop trial found
            print(f"{q_name}: No single-loop trial found, using most extreme")
            row = q_data.iloc[0]
            session = row['session']
            trial = int(row['trial'])

            trial_traj = get_trial_trajectory(reconstruction_df, session, trial)
            if trial_traj is not None and len(trial_traj) >= 10:
                x = trial_traj['xPose'].values
                y = trial_traj['yPose'].values
                hd = trial_traj['hdPose'].values
                valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(hd))
                x = x[valid]
                y = y[valid]
                hd = hd[valid]

                cum_turn = compute_cumulative_turn(hd)
                n_loops = np.abs(cum_turn) / TWO_PI

                results[q_name] = {
                    'session': session,
                    'trial': trial,
                    'integrated_ang_vel': row['integrated_ang_vel'],
                    'mvtDirError': row['mvtDirError'],
                    'cum_turn': cum_turn,
                    'n_loops': n_loops,
                    'x': x,
                    'y': y
                }

    return results


def create_trajectory_illustration_figure():
    """
    Create the main figure showing trajectories from each quadrant.
    """
    print("Loading data...")

    # Load data
    endpoints = pd.read_csv(f'{RESULTS_PATH}/pure_turn_section_endpoints_relaxed.csv')

    print(f"Loading reconstruction data (this may take a moment)...")
    reconstruction = pd.read_csv(f'{RESULTS_PATH}/reconstuctionDFAutoPI.csv',
                                  usecols=['session', 'trial', 'xPose', 'yPose', 'hdPose', 'recTime', 'condition'])

    print(f"Loaded {len(endpoints)} endpoints, {len(reconstruction)} reconstruction points")

    # Find example trials
    print("\nFinding example trials for each quadrant...")
    examples = find_quadrant_examples(endpoints, reconstruction,
                                       condition='all_dark',
                                       speed_threshold=5.0,
                                       n_candidates=50)

    if examples is None or len(examples) == 0:
        print("ERROR: Could not find example trials")
        return None

    # ==========================================================================
    # CREATE FIGURE
    # ==========================================================================

    print("\nCreating figure...")

    fig = plt.figure(figsize=(14, 12))

    # Main grid: scatter plot in center, trajectories in corners
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.5, 1],
                  width_ratios=[1, 1.5, 1], hspace=0.3, wspace=0.3)

    # Trajectory axes (corners)
    ax_q2 = fig.add_subplot(gs[0, 0])  # top-left
    ax_q1 = fig.add_subplot(gs[0, 2])  # top-right
    ax_scatter = fig.add_subplot(gs[1, 1])  # center
    ax_q3 = fig.add_subplot(gs[2, 0])  # bottom-left
    ax_q4 = fig.add_subplot(gs[2, 2])  # bottom-right

    # Map quadrants to axes and positions
    quadrant_axes = {
        'Q1': ax_q1,  # top-right
        'Q2': ax_q2,  # top-left
        'Q3': ax_q3,  # bottom-left
        'Q4': ax_q4,  # bottom-right
    }

    # Quadrant interpretations
    quadrant_labels = {
        'Q1': ('+ Turn, + Error', 'β > 0', 'Overestimation'),
        'Q2': ('- Turn, + Error', 'β < 0', 'Underestimation'),
        'Q3': ('- Turn, - Error', 'β > 0', 'Overestimation'),
        'Q4': ('+ Turn, - Error', 'β < 0', 'Underestimation'),
    }

    # Color for β > 0 vs β < 0
    quadrant_colors = {
        'Q1': COLORS['dark'],    # β > 0
        'Q2': COLORS['light'],   # β < 0
        'Q3': COLORS['dark'],    # β > 0
        'Q4': COLORS['light'],   # β < 0
    }

    # ==========================================================================
    # SCATTER PLOT (CENTER)
    # ==========================================================================

    # Filter for dark condition
    dark_endpoints = endpoints[(endpoints['condition'] == 'all_dark') &
                               (endpoints['speed_threshold'] == 5.0)]

    X = dark_endpoints['integrated_ang_vel'].values
    Y = dark_endpoints['mvtDirError'].values

    # Color by quadrant
    colors = np.where((X > 0) & (Y > 0), COLORS['dark'],
              np.where((X < 0) & (Y > 0), COLORS['light'],
              np.where((X < 0) & (Y < 0), COLORS['dark'],
              COLORS['light'])))

    ax_scatter.scatter(X, Y, c=colors, alpha=0.4, s=15, edgecolor='none')

    # Reference lines
    ax_scatter.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_scatter.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Quadrant labels
    ax_scatter.text(0.75, 0.85, 'β > 0', transform=ax_scatter.transAxes,
                    fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax_scatter.text(0.15, 0.85, 'β < 0', transform=ax_scatter.transAxes,
                    fontsize=14, fontweight='bold', color=COLORS['light'])
    ax_scatter.text(0.15, 0.1, 'β > 0', transform=ax_scatter.transAxes,
                    fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax_scatter.text(0.75, 0.1, 'β < 0', transform=ax_scatter.transAxes,
                    fontsize=14, fontweight='bold', color=COLORS['light'])

    # Mark example points
    for q_name, data in examples.items():
        ax_scatter.scatter(data['integrated_ang_vel'], data['mvtDirError'],
                          c='black', s=100, marker='*', zorder=10,
                          edgecolor='white', linewidth=1)

    ax_scatter.set_xlabel('Cumulative Turn (radians)', fontsize=12, fontweight='bold')
    ax_scatter.set_ylabel('Heading Deviation (radians)', fontsize=12, fontweight='bold')
    ax_scatter.set_title('Dark Condition (speed ≥ 5.0 cm/s)\n★ = example trajectories shown in corners',
                        fontsize=11, fontweight='bold')
    ax_scatter.grid(True, alpha=0.3)

    # ==========================================================================
    # TRAJECTORY PLOTS (CORNERS)
    # ==========================================================================

    for q_name, ax in quadrant_axes.items():
        if q_name not in examples:
            ax.text(0.5, 0.5, 'No example\nfound', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_aspect('equal')
            continue

        data = examples[q_name]
        x = data['x'].copy()
        y = data['y'].copy()
        color = quadrant_colors[q_name]
        labels = quadrant_labels[q_name]

        # Remove NaN values
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]

        if len(x) < 10:
            ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            continue

        # Plot trajectory with gradient
        result = plot_trajectory_with_gradient(ax, x, y, color=color, linewidth=2.5, alpha=0.9)

        if result is not None:
            _, x_clean, y_clean = result
            x, y = x_clean, y_clean

        # Mark start (green) and end (red)
        ax.scatter(x[0], y[0], c='green', s=100, marker='o', zorder=10,
                  edgecolor='white', linewidth=2, label='Start')
        ax.scatter(x[-1], y[-1], c='red', s=100, marker='s', zorder=10,
                  edgecolor='white', linewidth=2, label='End')

        # Mark lever position (approximately at arena center)
        lever_x, lever_y = 0, -7  # approximate lever position in actual coords
        ax.scatter(lever_x, lever_y, c='black', s=60, marker='x', zorder=10, linewidth=2)
        ax.annotate('Lever', (lever_x, lever_y), xytext=(lever_x + 3, lever_y + 3), fontsize=8)

        # Add arena boundary (approximate) - arena is ~80cm diameter = 40cm radius
        theta = np.linspace(0, 2*np.pi, 100)
        arena_r = 40  # cm
        ax.plot(lever_x + arena_r * np.cos(theta), lever_y + arena_r * np.sin(theta),
               'k--', alpha=0.3, linewidth=1)

        ax.set_aspect('equal')
        # Actual coordinates in cm
        ax.set_xlim(-50, 50)
        ax.set_ylim(-80, 50)

        # Title with quadrant info
        cum_turn = data['cum_turn']
        err_val = data['mvtDirError']
        n_loops = data['n_loops']

        title = f"{labels[0]}\n{labels[1]}: {labels[2]}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)

        # Add stats annotation
        stats_text = f"Cum. turn: {cum_turn:.2f} rad\nError: {err_val:.2f} rad\nLoops: {n_loops:.1f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('X (cm)', fontsize=9)
        ax.set_ylabel('Y (cm)', fontsize=9)
        ax.grid(True, alpha=0.3)

    # ==========================================================================
    # LEGEND AND TITLE
    # ==========================================================================

    fig.suptitle('Trajectory Examples Illustrating β Sign Interpretation\n'
                 'Dark condition: heading deviation vs cumulative turn',
                 fontsize=14, fontweight='bold', y=0.98)

    # Add explanation text
    explanation = (
        "β > 0 (orange): Turns in same direction as heading error → OVERESTIMATION\n"
        "β < 0 (blue): Turns opposite to heading error → UNDERESTIMATION/CORRECTION"
    )
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=10,
             style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor='orange', alpha=0.9))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save
    save_path = f'{FIGURES_PATH}/trajectory_interpretation_figure.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {save_path}")

    # Also save PDF
    save_path_pdf = f'{FIGURES_PATH}/trajectory_interpretation_figure.pdf'
    fig.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path_pdf}")

    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Creating Trajectory Illustration Figure")
    print("=" * 60)

    fig = create_trajectory_illustration_figure()

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)

    plt.show()
