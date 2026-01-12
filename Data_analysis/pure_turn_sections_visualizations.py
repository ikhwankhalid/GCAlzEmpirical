"""
Pure Turn Sections Analysis: Visualization and Comparison

This script creates visualizations for the pure turn sections analysis,
showing the relationship between integrated angular velocity and heading
deviation within continuous same-direction turning periods.

Key difference from bout-based analysis:
- Data is from pure turn sections (no sign changes in angular velocity)
- Integration resets at each section start
- All data points within a section have same turn direction

Author: Analysis generated for Peng et al. 2025
Date: 2025-11-26
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

def plot_signed_vs_split_comparison(data, condition_name, speed_threshold,
                                     save_path=None):
    """
    Create side-by-side comparison of signed regression vs split approaches.

    Parameters
    ----------
    data : DataFrame
        Section data with integrated_ang_vel and mvtDirError
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
        print(f"Warning: Not enough data for {condition_name}, "
              f"speed {speed_threshold}")
        return

    # Compute regressions
    slope_signed, intercept_signed, r_signed, p_signed, _ = stats.linregress(
        X, Y)

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
        slope_right, intercept_right, _, _, _ = stats.linregress(
            X_right, Y_right)

    # =========================================================================
    # LEFT PANEL: Signed Regression
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
    ax.set_xlabel('Cumulative Turn In Section (rad)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Heading Deviation (rad)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Signed Regression\n'
        f'Direct test of signed relationship\n'
        f'β={slope_signed:.4f}, R²={r_signed**2:.3f}, p={p_signed:.4e}',
        fontsize=11, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # RIGHT PANEL: Split by Direction
    # =========================================================================
    ax = axes[1]

    # Scatter plots with colors
    if len(X_left) > 0:
        ax.scatter(X_left, Y_left, alpha=0.3, s=5, c='blue',
                   label='Left sections', rasterized=True)
    if len(X_right) > 0:
        ax.scatter(X_right, Y_right, alpha=0.3, s=5, c='orange',
                   label='Right sections', rasterized=True)

    # Regression lines for each direction
    if not np.isnan(slope_left):
        x_left_line = np.array([X_left.min(), X_left.max()])
        y_left_line = slope_left * x_left_line + intercept_left
        ax.plot(x_left_line, y_left_line, 'b-', linewidth=2,
                label=f'β_left={slope_left:.4f}')

    if not np.isnan(slope_right):
        x_right_line = np.array([X_right.min(), X_right.max()])
        y_right_line = slope_right * x_right_line + intercept_right
        ax.plot(x_right_line, y_right_line, 'orange', linewidth=2,
                linestyle='-', label=f'β_right={slope_right:.4f}')

    # Zero lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Labels and title
    ax.set_xlabel('Cumulative Turn In Section (rad)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Heading Deviation (rad)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Split by Direction\n'
        f'Tests asymmetry, not signed relationship\n'
        f'β_left={slope_left:.4f}, β_right={slope_right:.4f}',
        fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f'{condition_name} | Speed ≥ {speed_threshold} cm/s\n'
        'Pure Turn Sections (no angular velocity sign changes)',
        fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_all_conditions_summary(regression_results, save_path=None):
    """
    Create summary plot comparing results across conditions.

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
    valid_data = valid_data.copy()
    valid_data['label'] = (valid_data['condition'] + "\n" +
                           valid_data['speed_threshold'].astype(str) + "cm/s")

    # =========================================================================
    # Panel 1: Beta coefficients comparison
    # =========================================================================
    ax = axes[0, 0]
    x = np.arange(len(valid_data))
    width = 0.25

    ax.bar(x - width, valid_data['beta_left'], width,
           label='β_left', alpha=0.7, color='blue')
    ax.bar(x, valid_data['beta_right'], width,
           label='β_right', alpha=0.7, color='orange')
    ax.bar(x + width, valid_data['beta_signed'], width,
           label='β_signed', alpha=0.7, color='red')

    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Beta Coefficient', fontsize=11, fontweight='bold')
    ax.set_title('Regression Slopes: Comparison of Approaches',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right',
                       fontsize=8)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 2: P-values comparison
    # =========================================================================
    ax = axes[0, 1]

    ax.scatter(valid_data['p_signed'], valid_data['p_asymmetry'],
               s=100, alpha=0.6, c='purple')

    # Add significance threshold lines
    ax.axhline(0.05, color='r', linestyle='--', linewidth=1, label='p=0.05')
    ax.axvline(0.05, color='r', linestyle='--', linewidth=1)

    # Diagonal line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('P-value: Signed Regression', fontsize=11, fontweight='bold')
    ax.set_ylabel('P-value: Asymmetry Test', fontsize=11, fontweight='bold')
    ax.set_title('Statistical Significance Comparison',
                 fontsize=12, fontweight='bold')
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
    ax.set_title('Explained Variance by Signed Regression',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right',
                       fontsize=8)
    ax.set_ylim([0, max(0.1, valid_data['r_squared_signed'].max() * 1.2)])
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 4: Effect direction summary
    # =========================================================================
    ax = axes[1, 1]

    # Count effect directions
    negative_sig = ((valid_data['beta_signed'] < 0) &
                    (valid_data['p_signed'] < 0.05)).sum()
    positive_sig = ((valid_data['beta_signed'] > 0) &
                    (valid_data['p_signed'] < 0.05)).sum()
    negative_ns = ((valid_data['beta_signed'] < 0) &
                   (valid_data['p_signed'] >= 0.05)).sum()
    positive_ns = ((valid_data['beta_signed'] > 0) &
                   (valid_data['p_signed'] >= 0.05)).sum()

    categories = ['Negative\n(sig)', 'Negative\n(n.s.)',
                  'Positive\n(n.s.)', 'Positive\n(sig)']
    counts = [negative_sig, negative_ns, positive_ns, positive_sig]
    colors = ['darkgreen', 'lightgreen', 'lightcoral', 'darkred']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Effect Direction Summary\n(Expected: Negative β)',
                 fontsize=12, fontweight='bold')
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


def plot_section_distribution(data, condition_name, speed_threshold,
                               save_path=None):
    """
    Plot distribution of section properties.

    Parameters
    ----------
    data : DataFrame
        Section data
    condition_name : str
        Condition name for title
    speed_threshold : float
        Speed threshold for title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # =========================================================================
    # Panel 1: Distribution of integrated angular velocity
    # =========================================================================
    ax = axes[0, 0]

    left_data = data[data['turn_direction'] == 'left']['integrated_ang_vel']
    right_data = data[data['turn_direction'] == 'right']['integrated_ang_vel']

    ax.hist(left_data, bins=50, alpha=0.6, color='blue', label='Left sections')
    ax.hist(right_data, bins=50, alpha=0.6, color='orange',
            label='Right sections')

    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Integrated Angular Velocity (rad)', fontsize=11,
                  fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Cumulative Turn in Sections',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: Section length distribution
    # =========================================================================
    ax = axes[0, 1]

    section_lengths = data.groupby('section_id')['section_length'].first()
    ax.hist(section_lengths, bins=30, alpha=0.7, color='gray')

    ax.set_xlabel('Section Length (timepoints)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Section Lengths',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    mean_len = section_lengths.mean()
    median_len = section_lengths.median()
    ax.axvline(mean_len, color='r', linestyle='-', linewidth=2,
               label=f'Mean: {mean_len:.1f}')
    ax.axvline(median_len, color='b', linestyle='--', linewidth=2,
               label=f'Median: {median_len:.1f}')
    ax.legend()

    # =========================================================================
    # Panel 3: Heading deviation by turn direction
    # =========================================================================
    ax = axes[1, 0]

    left_err = data[data['turn_direction'] == 'left']['mvtDirError']
    right_err = data[data['turn_direction'] == 'right']['mvtDirError']

    parts = ax.violinplot([left_err.dropna(), right_err.dropna()],
                          positions=[0, 1], showmeans=True, showmedians=True)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Left Sections', 'Right Sections'])
    ax.set_ylabel('Heading Deviation (rad)', fontsize=11, fontweight='bold')
    ax.set_title('Heading Deviation by Section Direction',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Sections per trial
    # =========================================================================
    ax = axes[1, 1]

    sections_per_trial = data.groupby('trial_id')['section_id'].nunique()
    ax.hist(sections_per_trial, bins=30, alpha=0.7, color='purple')

    ax.set_xlabel('Sections per Trial', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Sections per Trial',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    mean_spt = sections_per_trial.mean()
    ax.axvline(mean_spt, color='r', linestyle='-', linewidth=2,
               label=f'Mean: {mean_spt:.1f}')
    ax.legend()

    # Overall title
    fig.suptitle(f'{condition_name} | Speed ≥ {speed_threshold} cm/s',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# ENDPOINT ANALYSIS VISUALIZATION (fixes zero-clustering)
# =============================================================================

def plot_endpoint_comparison(endpoints, condition_name, speed_threshold, save_path=None):
    """Create scatter plot using only section endpoints."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    X = endpoints['integrated_ang_vel'].values
    Y = endpoints['mvtDirError'].values

    valid = ~(np.isnan(X) | np.isnan(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) < 10:
        print(f"Warning: Not enough endpoint data for {condition_name}, speed {speed_threshold}")
        plt.close(fig)
        return None

    slope_signed, intercept_signed, r_signed, p_signed, _ = stats.linregress(X, Y)

    turn_dirs = endpoints['turn_direction'].values[valid]
    left_mask = turn_dirs == 'left'
    right_mask = turn_dirs == 'right'

    X_left, Y_left = X[left_mask], Y[left_mask]
    X_right, Y_right = X[right_mask], Y[right_mask]

    slope_left = np.nan
    slope_right = np.nan

    if len(X_left) > 10:
        slope_left, intercept_left, _, _, _ = stats.linregress(X_left, Y_left)
    if len(X_right) > 10:
        slope_right, intercept_right, _, _, _ = stats.linregress(X_right, Y_right)

    # Left panel: Signed
    ax = axes[0]
    ax.scatter(X, Y, alpha=0.5, s=20, c='gray', rasterized=True)
    x_line = np.array([X.min(), X.max()])
    ax.plot(x_line, slope_signed * x_line + intercept_signed, 'r-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Total Turn In Section (rad)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Heading Deviation at Section End (rad)', fontsize=12, fontweight='bold')
    ax.set_title(f'Signed Regression (ENDPOINTS)\nβ={slope_signed:.4f}, R²={r_signed**2:.3f}', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right panel: Split
    ax = axes[1]
    if len(X_left) > 0:
        ax.scatter(X_left, Y_left, alpha=0.5, s=20, c='blue', label=f'Left (n={len(X_left)})')
    if len(X_right) > 0:
        ax.scatter(X_right, Y_right, alpha=0.5, s=20, c='orange', label=f'Right (n={len(X_right)})')
    if not np.isnan(slope_left):
        ax.plot([X_left.min(), X_left.max()], [slope_left * X_left.min() + intercept_left, slope_left * X_left.max() + intercept_left], 'b-', linewidth=2)
    if not np.isnan(slope_right):
        ax.plot([X_right.min(), X_right.max()], [slope_right * X_right.min() + intercept_right, slope_right * X_right.max() + intercept_right], color='orange', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Total Turn In Section (rad)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Heading Deviation at Section End (rad)', fontsize=12, fontweight='bold')
    ax.set_title(f'Split by Direction (ENDPOINTS)\nβ_left={slope_left:.4f}, β_right={slope_right:.4f}', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{condition_name} | Speed ≥ {speed_threshold} cm/s\nSECTION ENDPOINTS ONLY (n={len(X)} sections)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_endpoint_summary(endpoint_regression_results, save_path=None):
    """Create summary plot for endpoint analysis across all conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    valid_data = endpoint_regression_results.dropna(subset=['beta_signed', 'p_signed']).copy()
    if len(valid_data) == 0:
        print("No valid endpoint results to plot")
        return None

    valid_data['label'] = valid_data['condition'] + "\n" + valid_data['speed_threshold'].astype(str) + "cm/s"
    x = np.arange(len(valid_data))

    # Beta coefficients
    ax = axes[0, 0]
    colors = ['green' if b < 0 else 'red' for b in valid_data['beta_signed']]
    ax.bar(x, valid_data['beta_signed'], color=colors, alpha=0.7)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylabel('Beta Coefficient (Endpoints)')
    ax.set_title('Regression Slopes: Endpoint Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # R-squared
    ax = axes[0, 1]
    ax.bar(x, valid_data['r_squared_signed'], color='darkred', alpha=0.7)
    ax.set_ylabel('R² (Endpoint Regression)')
    ax.set_title('Explained Variance')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # P-values
    ax = axes[1, 0]
    ax.bar(x, -np.log10(valid_data['p_signed']), color='purple', alpha=0.7)
    ax.axhline(-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Sample size
    ax = axes[1, 1]
    ax.bar(x, valid_data['n_sections'], color='steelblue', alpha=0.7)
    ax.set_ylabel('Number of Sections')
    ax.set_title('Sample Size per Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('ENDPOINT ANALYSIS SUMMARY\n(One point per section - no zero-clustering)', fontsize=14, fontweight='bold', y=1.02)
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
    print("PURE TURN SECTIONS VISUALIZATION")
    print("="*80)

    # Load section data
    section_data_fn = os.path.join(RESULTS_PATH, "pure_turn_section_data.csv")
    print(f"\nLoading section data: {section_data_fn}")

    if not os.path.exists(section_data_fn):
        print("ERROR: Section data file not found!")
        print("Please run pure_turn_sections_analysis.py first")
        exit(1)

    section_data = pd.read_csv(section_data_fn)
    print(f"Loaded {len(section_data):,} rows")

    # Load regression results
    reg_results_fn = os.path.join(
        RESULTS_PATH, "pure_turn_section_regression_results.csv")
    print(f"Loading regression results: {reg_results_fn}")

    if not os.path.exists(reg_results_fn):
        print("ERROR: Regression results file not found!")
        print("Please run pure_turn_sections_analysis.py first")
        exit(1)

    regression_results = pd.read_csv(reg_results_fn)
    print(f"Loaded {len(regression_results)} conditions")

    # Load endpoint data
    endpoint_data_fn = os.path.join(RESULTS_PATH, "pure_turn_section_endpoints.csv")
    endpoint_data = None
    if os.path.exists(endpoint_data_fn):
        print(f"Loading endpoint data: {endpoint_data_fn}")
        endpoint_data = pd.read_csv(endpoint_data_fn)
        print(f"Loaded {len(endpoint_data):,} section endpoints")
    else:
        print(f"Note: Endpoint data file not found: {endpoint_data_fn}")
        print("  Run pure_turn_sections_analysis.py to generate endpoint data")

    # Load endpoint regression results
    endpoint_reg_fn = os.path.join(RESULTS_PATH, "pure_turn_section_endpoints_regression.csv")
    endpoint_regression_results = None
    if os.path.exists(endpoint_reg_fn):
        print(f"Loading endpoint regression results: {endpoint_reg_fn}")
        endpoint_regression_results = pd.read_csv(endpoint_reg_fn)
        print(f"Loaded {len(endpoint_regression_results)} endpoint regression results")

    # Create output directory for this analysis
    output_dir = os.path.join(FIGURES_PATH, "pure_turn_sections_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving figures to: {output_dir}")

    # Create subdirectory for endpoint plots
    endpoint_dir = os.path.join(output_dir, "endpoints")
    os.makedirs(endpoint_dir, exist_ok=True)

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

        # Filter section data for this condition
        condition_data = section_data[
            (section_data['condition'] == condition) &
            (section_data['speed_threshold'] == speed_threshold)
        ]

        if len(condition_data) < 10:
            print("  Skipping - insufficient data")
            continue

        # Create safe filename
        safe_condition = condition.replace('/', '_').replace(' ', '_')

        # Comparison plot
        save_path = os.path.join(
            output_dir,
            f"comparison_{safe_condition}_speed{speed_threshold}.png")

        fig = plot_signed_vs_split_comparison(
            condition_data, condition, speed_threshold, save_path)
        if fig:
            plt.close(fig)

        # Distribution plot
        dist_save_path = os.path.join(
            output_dir,
            f"distribution_{safe_condition}_speed{speed_threshold}.png")

        fig = plot_section_distribution(
            condition_data, condition, speed_threshold, dist_save_path)
        if fig:
            plt.close(fig)

        # Endpoint comparison plot (fixes zero-clustering)
        if endpoint_data is not None:
            endpoint_condition_data = endpoint_data[
                (endpoint_data['condition'] == condition) &
                (endpoint_data['speed_threshold'] == speed_threshold)
            ]

            if len(endpoint_condition_data) >= 10:
                endpoint_save_path = os.path.join(
                    endpoint_dir,
                    f"comparison_endpoints_{safe_condition}_speed{speed_threshold}.png")

                fig = plot_endpoint_comparison(
                    endpoint_condition_data, condition, speed_threshold,
                    endpoint_save_path)
                if fig:
                    plt.close(fig)
            else:
                print(f"  Skipping endpoint plot - insufficient data ({len(endpoint_condition_data)} sections)")

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

    # =========================================================================
    # Generate endpoint summary plot
    # =========================================================================
    if endpoint_regression_results is not None and len(endpoint_regression_results) > 0:
        print("\n" + "="*80)
        print("GENERATING ENDPOINT SUMMARY PLOT")
        print("="*80)

        endpoint_summary_path = os.path.join(endpoint_dir, "summary_endpoints_all_conditions.png")
        fig = plot_endpoint_summary(endpoint_regression_results, endpoint_summary_path)
        if fig:
            plt.close(fig)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nKey figures:")
    print("  - summary_all_conditions.png: Overall comparison across conditions (all timepoints)")
    print("  - comparison_*.png: Individual condition comparisons (all timepoints)")
    print("  - distribution_*.png: Section property distributions")
    print(f"\nEndpoint figures (fixes zero-clustering) saved to: {endpoint_dir}")
    print("  - summary_endpoints_all_conditions.png: Endpoint analysis summary")
    print("  - comparison_endpoints_*.png: Individual endpoint comparisons")
