"""
Integrated Angular Velocity Analysis: Visualization and Comparison

This script creates comparison visualizations showing the difference between:
1. OLD APPROACH: Split by turn direction (tests asymmetry)
2. NEW APPROACH: Signed regression (tests signed relationship)

The visualizations demonstrate why the signed regression directly tests the
hypothesis: "Does signed cumulative turning predict signed heading deviation?"

Author: Analysis generated for Peng et al. 2025
Date: 2025-11-24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_DATA_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng'
RESULTS_PATH = os.path.join(PROJECT_DATA_PATH, "results")
FIGURES_PATH = os.path.join(PROJECT_DATA_PATH, "figures")

# Make figures directory if it doesn't exist
os.makedirs(FIGURES_PATH, exist_ok=True)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_signed_vs_split_comparison(data, condition_name, speed_threshold, save_path=None):
    """
    Create side-by-side comparison of signed regression vs split approaches.

    Parameters
    ----------
    data : DataFrame
        Bout data with integrated_ang_vel and mvtDirError
    condition_name : str
        Condition name for title
    speed_threshold : float
        Speed threshold for title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get data
    X = data['integrated_ang_vel'].values
    Y = data['mvtDirError'].values

    # Remove NaN
    valid = ~(np.isnan(X) | np.isnan(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) < 10:
        print(f"Warning: Not enough data points for {condition_name}, speed {speed_threshold}")
        return

    # Compute regressions
    slope_signed, intercept_signed, r_signed, p_signed, se_signed = stats.linregress(X, Y)

    # Split by direction
    left_mask = X > 0
    right_mask = X < 0

    X_left = X[left_mask]
    Y_left = Y[left_mask]
    X_right = X[right_mask]
    Y_right = Y[right_mask]

    slope_left, intercept_left = np.nan, np.nan
    slope_right, intercept_right = np.nan, np.nan

    if len(X_left) > 10:
        slope_left, intercept_left, _, _, _ = stats.linregress(X_left, Y_left)
    if len(X_right) > 10:
        slope_right, intercept_right, _, _, _ = stats.linregress(X_right, Y_right)

    # =========================================================================
    # LEFT PANEL: Signed Regression (NEW APPROACH)
    # =========================================================================
    ax = axes[0]

    # Scatter plot
    ax.scatter(X, Y, alpha=0.3, s=5, c='gray', rasterized=True)

    # Regression line
    x_line = np.array([X.min(), X.max()])
    y_line = slope_signed * x_line + intercept_signed
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'β={slope_signed:.4f}')

    # Zero lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Labels and title
    ax.set_xlabel('Cumulative Turn In Bout (rad)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Heading Deviation (rad)', fontsize=12, fontweight='bold')
    ax.set_title('Signed Regression\n' +
                 f'Direct test of signed relationship\n' +
                 f'β={slope_signed:.4f}, R²={r_signed**2:.3f}, p={p_signed:.4e}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # RIGHT PANEL: Split by Direction (OLD APPROACH)
    # =========================================================================
    ax = axes[1]

    # Scatter plots with colors
    if len(X_left) > 0:
        ax.scatter(X_left, Y_left, alpha=0.3, s=5, c='blue', label='Left turns', rasterized=True)
    if len(X_right) > 0:
        ax.scatter(X_right, Y_right, alpha=0.3, s=5, c='orange', label='Right turns', rasterized=True)

    # Regression lines for each direction
    if not np.isnan(slope_left):
        x_left_line = np.array([X_left.min(), X_left.max()])
        y_left_line = slope_left * x_left_line + intercept_left
        ax.plot(x_left_line, y_left_line, 'b-', linewidth=2, label=f'β_left={slope_left:.4f}')

    if not np.isnan(slope_right):
        x_right_line = np.array([X_right.min(), X_right.max()])
        y_right_line = slope_right * x_right_line + intercept_right
        ax.plot(x_right_line, y_right_line, 'orange', linewidth=2, linestyle='-', label=f'β_right={slope_right:.4f}')

    # Zero lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Labels and title
    ax.set_xlabel('Cumulative Turn In Bout (rad)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Heading Deviation (rad)', fontsize=12, fontweight='bold')
    ax.set_title('Split by Direction\n' +
                 f'Tests asymmetry, not signed relationship\n' +
                 f'β_left={slope_left:.4f}, β_right={slope_right:.4f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'{condition_name} | Speed ≥ {speed_threshold} cm/s\n' +
                 f'Note: Different slopes in right panel reflect ONE signed relationship, not asymmetry!',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_all_conditions_summary(regression_results, save_path=None):
    """
    Create summary plot comparing signed regression vs asymmetry test across conditions.

    Parameters
    ----------
    regression_results : DataFrame
        Results from analysis with both approaches
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter valid results
    valid_data = regression_results.dropna(subset=['beta_signed', 'p_signed'])

    if len(valid_data) == 0:
        print("No valid results to plot")
        return

    # Create condition labels
    valid_data['label'] = valid_data['condition'] + f"\n{valid_data['speed_threshold']}cm/s"

    # =========================================================================
    # Panel 1: Beta coefficients comparison
    # =========================================================================
    ax = axes[0, 0]
    x = np.arange(len(valid_data))
    width = 0.25

    ax.bar(x - width, valid_data['beta_left'], width, label='β_left (old)', alpha=0.7, color='blue')
    ax.bar(x, valid_data['beta_right'], width, label='β_right (old)', alpha=0.7, color='orange')
    ax.bar(x + width, valid_data['beta_signed'], width, label='β_signed (NEW)', alpha=0.7, color='red')

    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Beta Coefficient', fontsize=11, fontweight='bold')
    ax.set_title('Regression Slopes: Comparison of Approaches', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 2: P-values comparison
    # =========================================================================
    ax = axes[0, 1]

    ax.scatter(valid_data['p_signed'], valid_data['p_asymmetry'], s=100, alpha=0.6, c='purple')

    # Add significance threshold lines
    ax.axhline(0.05, color='r', linestyle='--', linewidth=1, label='p=0.05')
    ax.axvline(0.05, color='r', linestyle='--', linewidth=1)

    # Diagonal line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('P-value: Signed Regression (NEW)', fontsize=11, fontweight='bold')
    ax.set_ylabel('P-value: Asymmetry Test (OLD)', fontsize=11, fontweight='bold')
    ax.set_title('Statistical Significance Comparison', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: R-squared values
    # =========================================================================
    ax = axes[1, 0]

    ax.bar(x, valid_data['r_squared_signed'], color='darkred', alpha=0.7)
    ax.set_ylabel('R² (Signed Regression)', fontsize=11, fontweight='bold')
    ax.set_title('Explained Variance by Signed Regression', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, max(0.1, valid_data['r_squared_signed'].max() * 1.2)])
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 4: Effect direction summary
    # =========================================================================
    ax = axes[1, 1]

    # Count effect directions
    negative_sig = ((valid_data['beta_signed'] < 0) & (valid_data['p_signed'] < 0.05)).sum()
    positive_sig = ((valid_data['beta_signed'] > 0) & (valid_data['p_signed'] < 0.05)).sum()
    negative_ns = ((valid_data['beta_signed'] < 0) & (valid_data['p_signed'] >= 0.05)).sum()
    positive_ns = ((valid_data['beta_signed'] > 0) & (valid_data['p_signed'] >= 0.05)).sum()

    categories = ['Negative\n(sig)', 'Negative\n(n.s.)', 'Positive\n(n.s.)', 'Positive\n(sig)']
    counts = [negative_sig, negative_ns, positive_ns, positive_sig]
    colors = ['darkgreen', 'lightgreen', 'lightcoral', 'darkred']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Effect Direction Summary\n(Expected: Negative β)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add counts on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("INTEGRATED ANGULAR VELOCITY VISUALIZATION")
    print("="*80)

    # Load bout data
    bout_data_fn = os.path.join(RESULTS_PATH, "integrated_ang_vel_bout_data.csv")
    print(f"\nLoading bout data: {bout_data_fn}")

    if not os.path.exists(bout_data_fn):
        print(f"ERROR: Bout data file not found!")
        print(f"Please run integrated_angular_velocity_bouts_analysis.py first")
        exit(1)

    bout_data = pd.read_csv(bout_data_fn)
    print(f"Loaded {len(bout_data):,} rows")

    # Load regression results
    reg_results_fn = os.path.join(RESULTS_PATH, "integrated_ang_vel_regression_results.csv")
    print(f"Loading regression results: {reg_results_fn}")

    if not os.path.exists(reg_results_fn):
        print(f"ERROR: Regression results file not found!")
        print(f"Please run integrated_angular_velocity_bouts_analysis.py first")
        exit(1)

    regression_results = pd.read_csv(reg_results_fn)
    print(f"Loaded {len(regression_results)} conditions")

    # Create output directory for this analysis
    output_dir = os.path.join(FIGURES_PATH, "integrated_angular_velocity_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving figures to: {output_dir}")

    # =========================================================================
    # Generate comparison plots for each condition
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING CONDITION-SPECIFIC COMPARISON PLOTS")
    print("="*80)

    for _, row in regression_results.iterrows():
        condition = row['condition']
        speed_threshold = row['speed_threshold']

        print(f"\nPlotting: {condition}, speed >= {speed_threshold} cm/s")

        # Filter bout data for this condition
        condition_data = bout_data[
            (bout_data['condition'] == condition) &
            (bout_data['speed_threshold'] == speed_threshold)
        ]

        if len(condition_data) < 10:
            print(f"  Skipping - insufficient data")
            continue

        # Create safe filename
        safe_condition = condition.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(output_dir,
                                f"comparison_{safe_condition}_speed{speed_threshold}.png")

        # Generate plot
        fig = plot_signed_vs_split_comparison(condition_data, condition, speed_threshold, save_path)
        if fig:
            plt.close(fig)

    # =========================================================================
    # Generate summary comparison plot
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING SUMMARY COMPARISON PLOT")
    print("="*80)

    summary_path = os.path.join(output_dir, "summary_all_conditions.png")
    fig = plot_all_conditions_summary(regression_results, summary_path)
    if fig:
        plt.close(fig)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll figures saved to: {output_dir}")
    print(f"\nKey figures:")
    print(f"  - summary_all_conditions.png: Overall comparison across all conditions")
    print(f"  - comparison_*.png: Individual condition comparisons")
