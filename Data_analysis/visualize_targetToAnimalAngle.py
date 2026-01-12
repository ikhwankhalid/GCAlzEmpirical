"""
Visualize targetToAnimalAngle from navPathInstan.csv

This script creates a 3x3 grid of subplots showing trial trajectories
colored by targetToAnimalAngle to verify the angle calculation.

targetToAnimalAngle is the angle FROM the lever TO the animal's position,
in radians [-pi, pi], where:
  - 0 = East
  - pi/2 = North
  - +/-pi = West
  - -pi/2 = South
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# Configuration
DATA_PATH = r"E:\GitHub\Peng_et.al_2025_noInt\Peng"
SESSION = "jp486-18032023-0108"
ANIMAL = "jp486"
RANDOM_SEED = 42  # For reproducibility

def load_nav_instan_data(data_path, animal, session):
    """Load navPathInstan.csv for the specified session."""
    file_path = f"{data_path}/{animal}/{session}/navPathInstan.csv"
    df = pd.read_csv(file_path)
    return df

def load_trial_conditions(data_path, animal, session):
    """Load navPathSummary.csv to get light/dark condition per trial."""
    file_path = f"{data_path}/{animal}/{session}/navPathSummary.csv"
    df = pd.read_csv(file_path)
    # Get unique trial-condition mapping (use first occurrence per trial)
    trial_conditions = df.groupby('trialNo')['light'].first().to_dict()
    return trial_conditions

def get_lever_position(trial_data):
    """
    Calculate lever position from animal position and target-to-animal vector.
    leverPos = animalPos - targetToAnimalVector
    """
    lever_x = trial_data['x'] - trial_data['targetToAnimalX']
    lever_y = trial_data['y'] - trial_data['targetToAnimalY']
    # Use median to get stable estimate (lever doesn't move within trial)
    return np.nanmedian(lever_x), np.nanmedian(lever_y)

def plot_trial(ax, trial_data, trial_no, cmap, norm, condition=''):
    """Plot a single trial trajectory with angle coloring using line plot."""
    # Filter to only use the '_all' segment to avoid duplicated/overlapping data
    # The navPathInstan.csv contains multiple segments (search, homing, atLever, etc.)
    # that overlap - using only '_all' gives the complete continuous trajectory
    all_segment_mask = trial_data['name'].str.endswith('_all')
    if all_segment_mask.any():
        trial_data = trial_data[all_segment_mask].copy()

    # Sort by time to ensure correct trajectory order
    trial_data = trial_data.sort_values('timeRes')

    # Get valid data points (non-NaN angles)
    valid_mask = ~np.isnan(trial_data['targetToAnimalAngle'])
    x = trial_data.loc[valid_mask, 'x'].values
    y = trial_data.loc[valid_mask, 'y'].values
    angles = trial_data.loc[valid_mask, 'targetToAnimalAngle'].values

    if len(x) == 0:
        ax.set_title(f"Trial {trial_no}\n(No valid data)")
        return

    # Get lever position
    lever_x, lever_y = get_lever_position(trial_data)

    # Create line segments for LineCollection
    # Each segment connects consecutive points
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create LineCollection with colors based on angle (use midpoint angle for each segment)
    segment_angles = (angles[:-1] + angles[1:]) / 2  # Average angle for each segment
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5, alpha=0.8)
    lc.set_array(segment_angles)
    ax.add_collection(lc)

    # Mark start of trajectory (beginning of trial)
    ax.scatter(x[0], y[0], marker='o', s=80, c='lime', edgecolors='black',
               linewidths=1.5, zorder=11, label='Start')

    # Add arrow showing initial movement direction (normalized)
    if len(x) >= 2:
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0:
            # Normalize and scale arrow length
            arrow_length = 8  # cm
            dx_norm = (dx / mag) * arrow_length
            dy_norm = (dy / mag) * arrow_length
            ax.annotate('', xy=(x[0] + dx_norm, y[0] + dy_norm), xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle='->', color='lime', lw=2,
                                       mutation_scale=15), zorder=12)
            # Dummy artist for legend
            ax.plot([], [], color='lime', lw=2, marker='>', markersize=6,
                    label='Initial direction')

    # Mark lever position
    ax.scatter(lever_x, lever_y, marker='*', s=200, c='red',
               edgecolors='black', linewidths=1, zorder=10, label='Lever')


    # Draw arena boundary (approximate 80cm radius circle centered at origin)
    theta = np.linspace(0, 2*np.pi, 100)
    arena_r = 40  # radius in cm
    ax.plot(arena_r * np.cos(theta), arena_r * np.sin(theta),
            'k--', alpha=0.3, lw=1)

    # Configure subplot
    ax.set_aspect('equal')
    title = f"Trial {trial_no}"
    if condition:
        title += f" ({condition})"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x (cm)', fontsize=8)
    ax.set_ylabel('y (cm)', fontsize=8)
    ax.tick_params(labelsize=7)

    # Set axis limits based on data extent with padding
    # (Required for LineCollection which doesn't auto-scale)
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    padding = max(x_range, y_range) * 0.15 + 5  # Extra padding for labels
    ax.set_xlim(x.min() - padding, x.max() + padding)
    ax.set_ylim(y.min() - padding, y.max() + padding)
    ax.autoscale_view()

def main():
    print(f"Loading data for session: {SESSION}")

    # Load data
    df = load_nav_instan_data(DATA_PATH, ANIMAL, SESSION)
    print(f"Loaded {len(df)} data points")

    # Load trial conditions (light/dark)
    trial_conditions = load_trial_conditions(DATA_PATH, ANIMAL, SESSION)
    print(f"Loaded conditions for {len(trial_conditions)} trials")

    # Get unique trials
    trials = df['trialNo'].unique()
    print(f"Found {len(trials)} unique trials")

    # Randomly select 9 trials
    np.random.seed(RANDOM_SEED)
    if len(trials) >= 9:
        selected_trials = np.random.choice(trials, size=9, replace=False)
    else:
        selected_trials = trials
        print(f"Warning: Only {len(trials)} trials available")

    selected_trials = sorted(selected_trials)
    print(f"Selected trials: {selected_trials}")

    # Set up colormap and normalization for circular angle data
    cmap = cm.twilight  # Circular colormap
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(f'targetToAnimalAngle Verification\nSession: {SESSION}\n'
                 f'Angle = direction FROM lever TO animal (radians)',
                 fontsize=12, fontweight='bold')

    # Plot each trial
    for idx, (ax, trial_no) in enumerate(zip(axes.flat, selected_trials)):
        trial_data = df[df['trialNo'] == trial_no].copy()
        condition = trial_conditions.get(trial_no, '')
        plot_trial(ax, trial_data, trial_no, cmap, norm, condition)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('targetToAnimalAngle (radians)', fontsize=10)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    # Add shared legend (get handles from first subplot)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.90, 0.02),
               fontsize=9, framealpha=0.9, edgecolor='black')

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.88, top=0.90, bottom=0.05, wspace=0.3, hspace=0.3)

    # Save figure
    output_path = r"E:\GitHub\Peng_et.al_2025_noInt\Data_analysis\visualize_targetToAnimalAngle.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    plt.show()

    # Print verification info
    print("\n" + "="*60)
    print("VERIFICATION GUIDE:")
    print("="*60)
    print("If the calculation is correct:")
    print("  - Points EAST of the lever (red star) should be colored ~0 (middle of colorbar)")
    print("  - Points NORTH of the lever should be colored ~pi/2 (upper-middle)")
    print("  - Points WEST of the lever should be colored ~+/-pi (top/bottom of colorbar)")
    print("  - Points SOUTH of the lever should be colored ~-pi/2 (lower-middle)")
    print("="*60)

if __name__ == "__main__":
    main()
