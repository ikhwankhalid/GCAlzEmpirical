"""
Pure Turn Sections Analysis: Visualization and Comparison (RELAXED FILTERING)

This script creates visualizations for the pure turn sections analysis
with RELAXED filtering parameters, showing the relationship between
integrated angular velocity and heading deviation within continuous
same-direction turning periods.

Key difference from original:
- Uses relaxed filtering parameters for more data retention
- Loads data from *_relaxed.csv files
- Saves figures to separate relaxed output directory

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
        'Pure Turn Sections - RELAXED FILTERING',
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
    fig.suptitle(f'{condition_name} | Speed ≥ {speed_threshold} cm/s\n'
                 'RELAXED FILTERING',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# ENDPOINT ANALYSIS VISUALIZATION (fixes zero-clustering)
# =============================================================================

def plot_endpoint_comparison(endpoints, condition_name, speed_threshold,
                              save_path=None):
    """
    Create scatter plot using only section endpoints.

    This eliminates zero-clustering by showing one point per section,
    where each point represents:
    - X: total cumulative turn in the section
    - Y: heading error at section end

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data with one row per section
    condition_name : str
        Condition name for title
    speed_threshold : float
        Speed threshold for title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get data
    X = endpoints['integrated_ang_vel'].values
    Y = endpoints['mvtDirError'].values

    # Remove NaN
    valid = ~(np.isnan(X) | np.isnan(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) < 10:
        print(f"Warning: Not enough endpoint data for {condition_name}, "
              f"speed {speed_threshold}")
        plt.close(fig)
        return None

    # Compute regressions
    slope_signed, intercept_signed, r_signed, p_signed, _ = stats.linregress(
        X, Y)

    # Split by direction (using turn_direction column)
    turn_dirs = endpoints['turn_direction'].values[valid]
    left_mask = turn_dirs == 'left'
    right_mask = turn_dirs == 'right'

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
    # LEFT PANEL: Signed Regression (Endpoints)
    # =========================================================================
    ax = axes[0]

    # Scatter plot - larger points since we have fewer data points
    ax.scatter(X, Y, alpha=0.5, s=20, c='gray', rasterized=True)

    # Regression line
    x_line = np.array([X.min(), X.max()])
    y_line = slope_signed * x_line + intercept_signed
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'β={slope_signed:.4f}')

    # Zero lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Labels and title
    ax.set_xlabel('Total Turn In Section (rad)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Heading Deviation at Section End (rad)', fontsize=12,
                  fontweight='bold')
    ax.set_title(
        'Signed Regression (ENDPOINTS)\n'
        f'Direct test: total turn → final error\n'
        f'β={slope_signed:.4f}, R²={r_signed**2:.3f}, p={p_signed:.4e}',
        fontsize=11, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # RIGHT PANEL: Split by Direction (Endpoints)
    # =========================================================================
    ax = axes[1]

    # Scatter plots with colors
    if len(X_left) > 0:
        ax.scatter(X_left, Y_left, alpha=0.5, s=20, c='blue',
                   label=f'Left sections (n={len(X_left)})', rasterized=True)
    if len(X_right) > 0:
        ax.scatter(X_right, Y_right, alpha=0.5, s=20, c='orange',
                   label=f'Right sections (n={len(X_right)})', rasterized=True)

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
    ax.set_xlabel('Total Turn In Section (rad)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Heading Deviation at Section End (rad)', fontsize=12,
                  fontweight='bold')
    ax.set_title(
        'Split by Direction (ENDPOINTS)\n'
        f'Tests asymmetry\n'
        f'β_left={slope_left:.4f}, β_right={slope_right:.4f}',
        fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f'{condition_name} | Speed ≥ {speed_threshold} cm/s\n'
        f'SECTION ENDPOINTS ONLY (n={len(X)} sections) - NO ZERO CLUSTERING',
        fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_endpoint_summary(endpoint_regression_results, save_path=None):
    """
    Create summary plot for endpoint analysis across all conditions.

    Parameters
    ----------
    endpoint_regression_results : DataFrame
        Results from endpoint regression analysis
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter valid results
    valid_data = endpoint_regression_results.dropna(
        subset=['beta_signed', 'p_signed']).copy()

    if len(valid_data) == 0:
        print("No valid endpoint results to plot")
        return None

    # Create condition labels
    valid_data['label'] = (valid_data['condition'] + "\n" +
                           valid_data['speed_threshold'].astype(str) + "cm/s")

    # =========================================================================
    # Panel 1: Beta coefficients
    # =========================================================================
    ax = axes[0, 0]
    x = np.arange(len(valid_data))

    colors = ['green' if b < 0 else 'red' for b in valid_data['beta_signed']]
    ax.bar(x, valid_data['beta_signed'], color=colors, alpha=0.7)

    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Beta Coefficient (Endpoints)', fontsize=11, fontweight='bold')
    ax.set_title('Regression Slopes: Endpoint Analysis\n(Expected: β < 0)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right',
                       fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 2: R-squared values
    # =========================================================================
    ax = axes[0, 1]

    ax.bar(x, valid_data['r_squared_signed'], color='darkred', alpha=0.7)
    ax.set_ylabel('R² (Endpoint Regression)', fontsize=11, fontweight='bold')
    ax.set_title('Explained Variance by Endpoint Regression',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right',
                       fontsize=8)
    ax.set_ylim([0, max(0.1, valid_data['r_squared_signed'].max() * 1.2)])
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 3: P-values
    # =========================================================================
    ax = axes[1, 0]

    ax.bar(x, -np.log10(valid_data['p_signed']), color='purple', alpha=0.7)
    ax.axhline(-np.log10(0.05), color='r', linestyle='--', linewidth=2,
               label='p=0.05')
    ax.axhline(-np.log10(0.001), color='orange', linestyle='--', linewidth=2,
               label='p=0.001')

    ax.set_ylabel('-log10(p-value)', fontsize=11, fontweight='bold')
    ax.set_title('Statistical Significance (Endpoint Analysis)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right',
                       fontsize=8)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 4: Number of sections per condition
    # =========================================================================
    ax = axes[1, 1]

    ax.bar(x, valid_data['n_sections'], color='steelblue', alpha=0.7)
    ax.set_ylabel('Number of Sections', fontsize=11, fontweight='bold')
    ax.set_title('Sample Size per Condition',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['label'], rotation=45, ha='right',
                       fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('ENDPOINT ANALYSIS SUMMARY\n'
                 '(One data point per section - eliminates zero-clustering)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# POOLED VS PER-ANIMAL COMPARISON VISUALIZATIONS
# =============================================================================

# Colorblind-friendly palette for 7 animals
ANIMAL_COLORS = {
    'jp486': '#1f77b4',   # Blue (largest dataset)
    'jp3269': '#ff7f0e',  # Orange
    'jp3120': '#2ca02c',  # Green
    'jp451': '#d62728',   # Red
    'jp452': '#9467bd',   # Purple
    'jp1686': '#8c564b',  # Brown
    'mn8578': '#e377c2'   # Pink
}


def extract_animal_id(session_or_trial_id):
    """Extract animal ID from session or trial_id string."""
    import re
    # Handle trial_id format: "session_Tnumber"
    if '_T' in str(session_or_trial_id):
        session_part = str(session_or_trial_id).split('_T')[0]
    else:
        session_part = str(session_or_trial_id)
    # Extract animal ID (everything before first hyphen)
    match = re.match(r'^([^-]+)', session_part)
    return match.group(1) if match else session_part


def plot_pooled_vs_per_animal_comparison(endpoints, per_animal_results, save_path=None,
                                          condition_filter=None, speed_filter=None):
    """
    Create main 2-panel figure comparing pooled vs per-animal analysis.

    Panel A: All data pooled with single regression line (gray points, red line)
    Panel B: Same data colored by animal with per-animal regression lines

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data with integrated_ang_vel, mvtDirError columns
    per_animal_results : DataFrame
        Per-animal regression statistics (used only if no filters applied)
    save_path : str, optional
        Path to save figure
    condition_filter : str, optional
        Filter to specific condition (e.g., 'homingFromLeavingLever_light')
    speed_filter : float, optional
        Filter to specific speed threshold (e.g., 5.0)
    """
    # Filter data if specified
    data = endpoints.copy()
    if condition_filter is not None:
        data = data[data['condition'] == condition_filter]
    if speed_filter is not None:
        data = data[data['speed_threshold'] == speed_filter]

    if len(data) < 20:
        print(f"  Skipping - insufficient data after filtering ({len(data)} points)")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Ensure animal_id column exists
    if 'animal_id' not in data.columns:
        if 'trial_id' in data.columns:
            data['animal_id'] = data['trial_id'].apply(extract_animal_id)
        elif 'session' in data.columns:
            data['animal_id'] = data['session'].apply(extract_animal_id)

    # Get valid data
    valid_mask = ~(data['integrated_ang_vel'].isna() | data['mvtDirError'].isna())
    X_all = data.loc[valid_mask, 'integrated_ang_vel'].values
    Y_all = data.loc[valid_mask, 'mvtDirError'].values

    # Compute pooled regression
    slope_pooled, intercept_pooled, r_pooled, p_pooled, _ = stats.linregress(X_all, Y_all)
    r2_pooled = r_pooled ** 2

    # Compute per-animal regression on the filtered data
    per_animal_stats = []
    for animal_id in data['animal_id'].unique():
        animal_mask = (data['animal_id'] == animal_id) & valid_mask
        X_animal = data.loc[animal_mask, 'integrated_ang_vel'].values
        Y_animal = data.loc[animal_mask, 'mvtDirError'].values
        if len(X_animal) >= 10:
            slope, intercept, r, p, _ = stats.linregress(X_animal, Y_animal)
            per_animal_stats.append({
                'animal_id': animal_id,
                'n_sections': len(X_animal),
                'beta': slope,
                'intercept': intercept,
                'r_squared': r ** 2,
                'p_value': p
            })
    per_animal_df = pd.DataFrame(per_animal_stats)

    # =========================================================================
    # Panel A: Pooled Analysis
    # =========================================================================
    ax = axes[0]
    ax.scatter(X_all, Y_all, alpha=0.15, s=3, c='gray', rasterized=True, label='_nolegend_')

    # Regression line
    x_line = np.array([X_all.min(), X_all.max()])
    y_line = slope_pooled * x_line + intercept_pooled
    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'Pooled: beta={slope_pooled:.3f}', zorder=10)

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Build title with filter info
    title_suffix = ""
    if condition_filter or speed_filter:
        parts = []
        if condition_filter:
            parts.append(condition_filter)
        if speed_filter:
            parts.append(f"speed >= {speed_filter} cm/s")
        title_suffix = f"\n({', '.join(parts)})"

    ax.set_xlabel('Cumulative Turn in Section (rad)', fontsize=12)
    ax.set_ylabel('Heading Deviation (rad)', fontsize=12)
    ax.set_title(f'A. Pooled Analysis (All Animals Combined){title_suffix}\n'
                 f'beta = {slope_pooled:.4f}, R^2 = {r2_pooled:.4f}, n = {len(X_all):,}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim([X_all.min() - 0.2, X_all.max() + 0.2])
    ax.set_ylim([Y_all.min() - 0.5, Y_all.max() + 0.5])

    # =========================================================================
    # Panel B: Per-Animal Analysis
    # =========================================================================
    ax = axes[1]

    if len(per_animal_df) == 0:
        ax.text(0.5, 0.5, 'No animals with >= 10 data points',
                ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")
        return fig

    mean_r2 = per_animal_df['r_squared'].mean()
    max_r2 = per_animal_df['r_squared'].max()

    # Plot each animal's data and regression line
    for _, animal_row in per_animal_df.iterrows():
        animal_id = animal_row['animal_id']
        animal_mask = (data['animal_id'] == animal_id) & valid_mask
        X_animal = data.loc[animal_mask, 'integrated_ang_vel'].values
        Y_animal = data.loc[animal_mask, 'mvtDirError'].values

        if len(X_animal) < 10:
            continue

        color = ANIMAL_COLORS.get(animal_id, '#333333')

        # Scatter points
        ax.scatter(X_animal, Y_animal, alpha=0.25, s=5, c=color, rasterized=True, label='_nolegend_')

        # Get regression parameters
        slope = animal_row['beta']
        intercept = animal_row['intercept']
        r2 = animal_row['r_squared']

        # Regression line
        x_line = np.array([X_animal.min(), X_animal.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linewidth=2.5, zorder=10,
                label=f'{animal_id}: R^2={r2:.3f}')

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Cumulative Turn in Section (rad)', fontsize=12)
    ax.set_ylabel('Heading Deviation (rad)', fontsize=12)
    ax.set_title(f'B. Per-Animal Analysis (Individual Regression Lines){title_suffix}\n'
                 f'Mean R^2 = {mean_r2:.4f}, Max R^2 = {max_r2:.4f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=1, framealpha=0.9)
    ax.set_xlim([X_all.min() - 0.2, X_all.max() + 0.2])
    ax.set_ylim([Y_all.min() - 0.5, Y_all.max() + 0.5])

    # Add improvement annotation
    improvement = mean_r2 / r2_pooled if r2_pooled > 0 else 0
    fig.text(0.5, 0.01, f'Per-animal analysis shows {improvement:.1f}x improvement in mean R^2',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_per_animal_facets(endpoints, per_animal_results, save_path=None,
                           condition_filter=None, speed_filter=None):
    """
    Create 2x4 faceted scatter plot with one panel per animal plus summary.

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data
    per_animal_results : DataFrame
        Per-animal regression statistics (used only if no filters applied)
    save_path : str, optional
        Path to save figure
    condition_filter : str, optional
        Filter to specific condition
    speed_filter : float, optional
        Filter to specific speed threshold
    """
    # Filter data if specified
    data = endpoints.copy()
    if condition_filter is not None:
        data = data[data['condition'] == condition_filter]
    if speed_filter is not None:
        data = data[data['speed_threshold'] == speed_filter]

    if len(data) < 20:
        print(f"  Skipping facets - insufficient data after filtering ({len(data)} points)")
        return None

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Ensure animal_id column exists
    if 'animal_id' not in data.columns:
        if 'trial_id' in data.columns:
            data['animal_id'] = data['trial_id'].apply(extract_animal_id)
        elif 'session' in data.columns:
            data['animal_id'] = data['session'].apply(extract_animal_id)

    # Get valid data
    valid_mask = ~(data['integrated_ang_vel'].isna() | data['mvtDirError'].isna())

    # Compute per-animal regression on the filtered data
    per_animal_stats = []
    for animal_id in data['animal_id'].unique():
        animal_mask = (data['animal_id'] == animal_id) & valid_mask
        X_animal = data.loc[animal_mask, 'integrated_ang_vel'].values
        Y_animal = data.loc[animal_mask, 'mvtDirError'].values
        if len(X_animal) >= 10:
            slope, intercept, r, p, _ = stats.linregress(X_animal, Y_animal)
            per_animal_stats.append({
                'animal_id': animal_id,
                'n_sections': len(X_animal),
                'beta': slope,
                'intercept': intercept,
                'r_squared': r ** 2,
                'p_value': p
            })
    per_animal_df = pd.DataFrame(per_animal_stats)

    if len(per_animal_df) == 0:
        print(f"  Skipping facets - no animals with >= 10 data points")
        plt.close(fig)
        return None

    # Sort animals by sample size (descending)
    per_animal_sorted = per_animal_df.sort_values('n_sections', ascending=False)

    # Compute pooled R2 for reference
    X_all = data.loc[valid_mask, 'integrated_ang_vel'].values
    Y_all = data.loc[valid_mask, 'mvtDirError'].values
    _, _, r_pooled, _, _ = stats.linregress(X_all, Y_all)
    r2_pooled = r_pooled ** 2

    # Get consistent axis limits
    x_min, x_max = X_all.min() - 0.2, X_all.max() + 0.2
    y_min, y_max = Y_all.min() - 0.5, Y_all.max() + 0.5

    # Plot each animal (first 7 panels)
    n_animals = len(per_animal_sorted)
    for i, (_, animal_row) in enumerate(per_animal_sorted.iterrows()):
        if i >= 7:
            break

        ax = axes[i]
        animal_id = animal_row['animal_id']

        animal_mask = (data['animal_id'] == animal_id) & valid_mask
        X_animal = data.loc[animal_mask, 'integrated_ang_vel'].values
        Y_animal = data.loc[animal_mask, 'mvtDirError'].values

        color = ANIMAL_COLORS.get(animal_id, '#333333')

        # Scatter points
        ax.scatter(X_animal, Y_animal, alpha=0.4, s=8, c=color, rasterized=True)

        # Regression line
        slope = animal_row['beta']
        intercept = animal_row['intercept']
        r2 = animal_row['r_squared']
        p_val = animal_row['p_value']

        x_line = np.array([X_animal.min(), X_animal.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='black', linewidth=2.5, zorder=10)

        # Reference lines
        ax.axhline(0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
        ax.axvline(0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)

        # Title with stats
        sig_marker = '*' if p_val < 0.05 else ''
        ax.set_title(f'{animal_id} (n={animal_row["n_sections"]:,}){sig_marker}\n'
                     f'beta={slope:.3f}, R^2={r2:.4f}',
                     fontsize=10, fontweight='bold', color=color)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        if i >= 4:  # Bottom row
            ax.set_xlabel('Cumulative Turn (rad)', fontsize=9)
        if i % 4 == 0:  # Left column
            ax.set_ylabel('Heading Deviation (rad)', fontsize=9)

        ax.tick_params(labelsize=8)

    # Clear unused panels (if fewer than 7 animals)
    for i in range(n_animals, 7):
        axes[i].axis('off')

    # Panel 8: Summary bar chart
    ax = axes[7]

    # Sort by R^2 for display
    per_animal_by_r2 = per_animal_sorted.sort_values('r_squared', ascending=True)

    animal_ids = per_animal_by_r2['animal_id'].values
    r2_values = per_animal_by_r2['r_squared'].values
    colors = [ANIMAL_COLORS.get(aid, '#333333') for aid in animal_ids]

    bars = ax.barh(range(len(animal_ids)), r2_values, color=colors, alpha=0.8)
    ax.axvline(r2_pooled, color='red', linestyle='--', linewidth=2, label=f'Pooled R^2={r2_pooled:.4f}')
    ax.set_yticks(range(len(animal_ids)))
    ax.set_yticklabels(animal_ids, fontsize=9)
    ax.set_xlabel('R^2', fontsize=10)
    ax.set_title('Per-Animal R^2 Comparison\n(vs Pooled)', fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim([0, max(r2_values) * 1.2])

    # Build title with filter info
    title_parts = ['Per-Animal Heading Deviation vs Cumulative Turn Analysis']
    if condition_filter or speed_filter:
        filter_desc = []
        if condition_filter:
            filter_desc.append(condition_filter)
        if speed_filter:
            filter_desc.append(f"speed >= {speed_filter} cm/s")
        title_parts.append(f"({', '.join(filter_desc)})")
    title_parts.append('(* indicates p < 0.05)')

    fig.suptitle('\n'.join(title_parts),
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_r_squared_improvement_bars(comparison_df, save_path=None):
    """
    Create grouped bar chart comparing pooled vs per-animal R^2 by condition.

    Parameters
    ----------
    comparison_df : DataFrame
        Model comparison data with pooled_r_squared and per_animal_mean_r_squared
    save_path : str, optional
        Path to save figure
    """
    # Sort by improvement factor
    comparison_df = comparison_df.copy()
    comparison_df['improvement'] = comparison_df['per_animal_mean_r_squared'] / comparison_df['pooled_r_squared']
    comparison_df['improvement'] = comparison_df['improvement'].replace([np.inf, -np.inf], np.nan)

    # Filter to valid rows and sort by improvement
    valid_df = comparison_df.dropna(subset=['pooled_r_squared', 'per_animal_mean_r_squared'])
    valid_df = valid_df.sort_values('improvement', ascending=False)

    # Create labels
    valid_df['label'] = valid_df['condition'].str.replace('FromLeavingLever', '').str.replace('ToLeverPath', '')
    valid_df['label'] = valid_df['label'] + '\n' + valid_df['speed_threshold'].astype(str) + ' cm/s'

    # Select top conditions by data volume or improvement
    if len(valid_df) > 16:
        valid_df = valid_df.head(16)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(valid_df))
    width = 0.35

    # Bars
    bars1 = ax.bar(x - width/2, valid_df['pooled_r_squared'], width,
                   label='Pooled R^2', color='gray', alpha=0.7)
    bars2 = ax.bar(x + width/2, valid_df['per_animal_mean_r_squared'], width,
                   label='Per-Animal Mean R^2', color='steelblue', alpha=0.8)

    # Add improvement labels on top of per-animal bars
    for i, (_, row) in enumerate(valid_df.iterrows()):
        improvement = row['improvement']
        if not np.isnan(improvement) and improvement > 1:
            ax.annotate(f'{improvement:.1f}x',
                       xy=(x[i] + width/2, row['per_animal_mean_r_squared']),
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       color='darkblue')

    ax.set_xlabel('Condition / Speed Threshold', fontsize=12)
    ax.set_ylabel('R^2', fontsize=12)
    ax.set_title('R^2 Improvement: Pooled vs Per-Animal Analysis\n'
                 '(Numbers show improvement factor)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_df['label'], rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Log scale if values span multiple orders of magnitude
    max_val = valid_df['per_animal_mean_r_squared'].max()
    min_val = valid_df['pooled_r_squared'].min()
    if max_val > 0 and min_val > 0 and max_val / min_val > 100:
        ax.set_yscale('log')
        ax.set_ylabel('R^2 (log scale)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_best_conditions_comparison(endpoints, model_comparison, save_path=None):
    """
    Create 2x2 panel figure showing pooled vs per-animal for top 4 conditions.

    Each panel shows a scatter plot with both pooled and per-animal regression lines
    for one of the best conditions (highest per-animal R^2).

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data
    model_comparison : DataFrame
        Model comparison data with condition, speed_threshold, R^2 values
    save_path : str, optional
        Path to save figure
    """
    # Select top 4 conditions by per-animal mean R^2
    best_conditions = [
        ('searchToLeverPath_dark', 5.0),        # R^2 = 0.470 (HIGHEST!)
        ('atLever_dark', 5.0),                   # R^2 = 0.239
        ('homingFromLeavingLever_light', 5.0),  # R^2 = 0.188
        ('all_dark', 5.0),                       # R^2 = 0.110
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Ensure animal_id column exists
    data = endpoints.copy()
    if 'animal_id' not in data.columns:
        if 'trial_id' in data.columns:
            data['animal_id'] = data['trial_id'].apply(extract_animal_id)
        elif 'session' in data.columns:
            data['animal_id'] = data['session'].apply(extract_animal_id)

    for i, (condition, speed) in enumerate(best_conditions):
        ax = axes[i]

        # Filter data
        cond_data = data[(data['condition'] == condition) &
                         (data['speed_threshold'] == speed)]

        if len(cond_data) < 10:
            ax.text(0.5, 0.5, f'Insufficient data\n{condition}\nspeed >= {speed}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        valid_mask = ~(cond_data['integrated_ang_vel'].isna() | cond_data['mvtDirError'].isna())
        X_all = cond_data.loc[valid_mask, 'integrated_ang_vel'].values
        Y_all = cond_data.loc[valid_mask, 'mvtDirError'].values

        if len(X_all) < 10:
            ax.text(0.5, 0.5, f'Insufficient data\n{condition}\nspeed >= {speed}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Compute pooled regression
        slope_pooled, intercept_pooled, r_pooled, _, _ = stats.linregress(X_all, Y_all)
        r2_pooled = r_pooled ** 2

        # Compute per-animal regressions
        per_animal_stats = []
        for animal_id in cond_data['animal_id'].unique():
            animal_mask = (cond_data['animal_id'] == animal_id) & valid_mask
            X_animal = cond_data.loc[animal_mask, 'integrated_ang_vel'].values
            Y_animal = cond_data.loc[animal_mask, 'mvtDirError'].values
            if len(X_animal) >= 5:  # Lower threshold for sparse conditions
                slope, intercept, r, p, _ = stats.linregress(X_animal, Y_animal)
                per_animal_stats.append({
                    'animal_id': animal_id,
                    'n': len(X_animal),
                    'beta': slope,
                    'intercept': intercept,
                    'r_squared': r ** 2,
                    'X': X_animal,
                    'Y': Y_animal
                })

        # Plot data points colored by animal
        for stats_row in per_animal_stats:
            animal_id = stats_row['animal_id']
            color = ANIMAL_COLORS.get(animal_id, '#333333')
            ax.scatter(stats_row['X'], stats_row['Y'], alpha=0.5, s=15,
                       c=color, rasterized=True, label='_nolegend_')

            # Per-animal regression line
            x_line = np.array([stats_row['X'].min(), stats_row['X'].max()])
            y_line = stats_row['beta'] * x_line + stats_row['intercept']
            ax.plot(x_line, y_line, color=color, linewidth=1.5, alpha=0.7,
                    label=f'{animal_id}: R^2={stats_row["r_squared"]:.3f}')

        # Pooled regression line (thick red dashed)
        x_line_pooled = np.array([X_all.min(), X_all.max()])
        y_line_pooled = slope_pooled * x_line_pooled + intercept_pooled
        ax.plot(x_line_pooled, y_line_pooled, 'r--', linewidth=3, zorder=15,
                label=f'Pooled: R^2={r2_pooled:.4f}')

        # Reference lines
        ax.axhline(0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
        ax.axvline(0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)

        # Calculate mean per-animal R^2
        mean_r2 = np.mean([s['r_squared'] for s in per_animal_stats])

        # Title
        short_condition = condition.replace('FromLeavingLever', '').replace('ToLeverPath', '')
        ax.set_title(f'{short_condition} (speed >= {speed} cm/s)\n'
                     f'n = {len(X_all)}, Pooled R^2 = {r2_pooled:.4f}, '
                     f'Mean per-animal R^2 = {mean_r2:.4f}',
                     fontsize=11, fontweight='bold')

        ax.set_xlabel('Cumulative Turn in Section (rad)', fontsize=10)
        ax.set_ylabel('Heading Deviation (rad)', fontsize=10)
        ax.legend(loc='best', fontsize=8, ncol=2, framealpha=0.9)

    fig.suptitle('Best Conditions: Pooled vs Per-Animal Regression Comparison\n'
                 '(Dashed red = pooled; solid = per-animal)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_validation_forest(validation_df, save_path=None):
    """
    Create forest plot showing beta estimates with bootstrap confidence intervals.

    Parameters
    ----------
    validation_df : DataFrame
        Output from validate_sparse_conditions()
    save_path : str, optional
        Path to save figure
    """
    if validation_df is None or len(validation_df) == 0:
        print("No validation data to plot")
        return None

    # Remove rows with NaN beta
    df = validation_df.dropna(subset=['beta']).copy()

    if len(df) == 0:
        print("No valid validation data to plot")
        return None

    # Create label combining condition and speed
    df['label'] = df['condition'].str.replace('FromLeavingLever', '').str.replace('ToLeverPath', '')
    df['label'] = df['label'] + '\n(' + df['speed_threshold'].astype(str) + ' cm/s, n=' + df['n'].astype(str) + ')'

    # Sort by beta value for visual clarity
    df = df.sort_values('beta', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    y_positions = np.arange(len(df))

    # Plot error bars (CIs)
    for i, (_, row) in enumerate(df.iterrows()):
        ci_low = row['beta_ci_low']
        ci_high = row['beta_ci_high']
        beta = row['beta']

        # Color based on significance
        if row['perm_p_value'] < 0.05:
            color = 'darkblue' if beta < 0 else 'darkred'
            marker = 'o'
        else:
            color = 'gray'
            marker = 's'

        # Error bar
        ax.errorbar(beta, i, xerr=[[beta - ci_low], [ci_high - beta]],
                    fmt=marker, color=color, capsize=5, capthick=2, markersize=8,
                    linewidth=2, alpha=0.8)

        # Add significance annotation
        if row['perm_p_value'] < 0.001:
            ax.annotate('***', (ci_high + 0.02, i), fontsize=12, fontweight='bold', color=color)
        elif row['perm_p_value'] < 0.01:
            ax.annotate('**', (ci_high + 0.02, i), fontsize=12, fontweight='bold', color=color)
        elif row['perm_p_value'] < 0.05:
            ax.annotate('*', (ci_high + 0.02, i), fontsize=12, fontweight='bold', color=color)

    # Reference line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df['label'].values, fontsize=10)

    # Labels and title
    ax.set_xlabel('Beta (Slope: Cumulative Turn -> Heading Deviation)', fontsize=12)
    ax.set_title('Statistical Validation: Bootstrap 95% CI for Regression Slopes\n'
                 '(Blue = negative slope, Red = positive slope, Gray = not significant)\n'
                 '* p<0.05, ** p<0.01, *** p<0.001 (permutation test)',
                 fontsize=12, fontweight='bold')

    # Add grid
    ax.grid(True, axis='x', alpha=0.3)

    # Expand x-axis slightly to fit annotations
    xlim = ax.get_xlim()
    ax.set_xlim([xlim[0], xlim[1] + 0.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_trial_beta_distribution(trial_results, aggregate_stats, condition,
                                  speed_threshold, save_path=None):
    """
    Create histogram of per-trial beta values with statistical annotations.

    Shows:
    - Distribution of trial-level beta values
    - Vertical line at beta=0 (null hypothesis)
    - Mean beta with bootstrap CI
    - One-sample t-test result
    - Proportion of positive vs negative betas

    Parameters
    ----------
    trial_results : DataFrame
        Per-trial regression results from per_trial_regression()
    aggregate_stats : dict
        Aggregate statistics from per_trial_regression()
    condition : str
        Condition name for title
    speed_threshold : float
        Speed threshold for title
    save_path : str, optional
        Path to save figure
    """
    if len(trial_results) == 0:
        print(f"No trial results to plot for {condition}")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    betas = trial_results['beta'].values

    # =========================================================================
    # Panel A: Histogram of trial-level betas
    # =========================================================================
    ax = axes[0]

    # Histogram
    n_bins = min(30, len(betas) // 3)
    n_bins = max(10, n_bins)

    counts, bins, patches = ax.hist(betas, bins=n_bins, alpha=0.7, color='steelblue',
                                     edgecolor='white', linewidth=0.5)

    # Color bars based on sign
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('cornflowerblue')
        else:
            patch.set_facecolor('indianred')

    # Vertical line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='β = 0')

    # Mean line
    mean_beta = aggregate_stats['mean_beta']
    ax.axvline(mean_beta, color='red', linestyle='-', linewidth=2.5,
               label=f'Mean β = {mean_beta:.4f}')

    # CI shading
    ci_low = aggregate_stats['ci_low']
    ci_high = aggregate_stats['ci_high']
    ax.axvspan(ci_low, ci_high, alpha=0.2, color='red', label='95% CI')

    # Labels
    ax.set_xlabel('Trial-Level Beta (Slope)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Trials', fontsize=12, fontweight='bold')

    # Format condition name for title
    short_condition = condition.replace('FromLeavingLever', '').replace('ToLeverPath', '')

    ax.set_title(f'A. Distribution of Per-Trial Beta Values\n'
                 f'{short_condition} | speed >= {speed_threshold} cm/s\n'
                 f'(n = {aggregate_stats["n_trials"]} trials, {aggregate_stats["n_total_sections"]} total sections)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel B: Statistical summary
    # =========================================================================
    ax = axes[1]
    ax.axis('off')

    # Build summary text
    summary_lines = [
        "STATISTICAL SUMMARY",
        "=" * 40,
        "",
        f"Total trials analyzed: {aggregate_stats['n_trials']}",
        f"Total sections: {aggregate_stats['n_total_sections']}",
        f"Minimum sections per trial: {aggregate_stats['min_sections']}",
        "",
        "BETA DISTRIBUTION",
        "-" * 40,
        f"Mean beta: {aggregate_stats['mean_beta']:.4f}",
        f"Median beta: {aggregate_stats['median_beta']:.4f}",
        f"Std beta: {aggregate_stats['std_beta']:.4f}",
        f"SE of mean: {aggregate_stats['se_beta']:.4f}",
        "",
        f"95% Bootstrap CI: [{aggregate_stats['ci_low']:.4f}, {aggregate_stats['ci_high']:.4f}]",
        "",
        "HYPOTHESIS TEST",
        "-" * 40,
        "H0: Mean beta = 0 (no relationship)",
        f"One-sample t = {aggregate_stats['one_sample_t']:.3f}",
        f"p-value = {aggregate_stats['one_sample_p']:.4f}",
        "",
    ]

    # Add significance interpretation
    p = aggregate_stats['one_sample_p']
    if p < 0.001:
        summary_lines.append("*** HIGHLY SIGNIFICANT (p < 0.001)")
    elif p < 0.01:
        summary_lines.append("** SIGNIFICANT (p < 0.01)")
    elif p < 0.05:
        summary_lines.append("* SIGNIFICANT (p < 0.05)")
    else:
        summary_lines.append("NOT SIGNIFICANT (p >= 0.05)")

    summary_lines.extend([
        "",
        "DIRECTION COUNTS",
        "-" * 40,
        f"Positive beta trials: {aggregate_stats['n_positive_beta']} ({aggregate_stats['prop_positive']*100:.1f}%)",
        f"Negative beta trials: {aggregate_stats['n_negative_beta']} ({(1-aggregate_stats['prop_positive'])*100:.1f}%)",
        "",
        "INTERPRETATION",
        "-" * 40,
    ])

    # Add interpretation
    if mean_beta > 0 and p < 0.05:
        summary_lines.append("Positive slope: cumulative turn predicts")
        summary_lines.append("heading error in the SAME direction.")
        summary_lines.append("-> Evidence for integration DRIFT in dark")
    elif mean_beta < 0 and p < 0.05:
        summary_lines.append("Negative slope: cumulative turn predicts")
        summary_lines.append("heading error in OPPOSITE direction.")
        summary_lines.append("-> Evidence for ERROR CORRECTION (light)")
    else:
        summary_lines.append("No significant relationship between")
        summary_lines.append("cumulative turn and heading error.")

    # Display text
    summary_text = "\n".join(summary_lines)
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_light_vs_dark_trial_comparison(light_results, light_stats,
                                         dark_results, dark_stats,
                                         speed_threshold, save_path=None):
    """
    Create side-by-side comparison of trial-level betas for light vs dark.

    Parameters
    ----------
    light_results : DataFrame
        Per-trial results for light condition
    light_stats : dict
        Aggregate stats for light condition
    dark_results : DataFrame
        Per-trial results for dark condition
    dark_stats : dict
        Aggregate stats for dark condition
    speed_threshold : float
        Speed threshold used
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # =========================================================================
    # Panel A: Light condition histogram
    # =========================================================================
    ax = axes[0]

    if len(light_results) > 0:
        betas_light = light_results['beta'].values
        ax.hist(betas_light, bins=min(30, len(betas_light)//3), alpha=0.7,
                color='gold', edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.axvline(light_stats['mean_beta'], color='darkgoldenrod', linestyle='-',
                   linewidth=2.5, label=f'Mean={light_stats["mean_beta"]:.3f}')
        ax.axvspan(light_stats['ci_low'], light_stats['ci_high'], alpha=0.2, color='gold')

    ax.set_xlabel('Trial-Level Beta', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'A. LIGHT Condition\n(n={light_stats.get("n_trials", 0)} trials)\n'
                 f'p={light_stats.get("one_sample_p", 1.0):.4f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel B: Dark condition histogram
    # =========================================================================
    ax = axes[1]

    if len(dark_results) > 0:
        betas_dark = dark_results['beta'].values
        ax.hist(betas_dark, bins=min(30, len(betas_dark)//3), alpha=0.7,
                color='darkslategray', edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.axvline(dark_stats['mean_beta'], color='red', linestyle='-',
                   linewidth=2.5, label=f'Mean={dark_stats["mean_beta"]:.3f}')
        ax.axvspan(dark_stats['ci_low'], dark_stats['ci_high'], alpha=0.2, color='red')

    ax.set_xlabel('Trial-Level Beta', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'B. DARK Condition\n(n={dark_stats.get("n_trials", 0)} trials)\n'
                 f'p={dark_stats.get("one_sample_p", 1.0):.4f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel C: Direct comparison
    # =========================================================================
    ax = axes[2]

    # Bar plot comparing means
    conditions = ['Light', 'Dark']
    means = [light_stats.get('mean_beta', 0), dark_stats.get('mean_beta', 0)]
    errors = [
        (light_stats.get('mean_beta', 0) - light_stats.get('ci_low', 0),
         light_stats.get('ci_high', 0) - light_stats.get('mean_beta', 0)),
        (dark_stats.get('mean_beta', 0) - dark_stats.get('ci_low', 0),
         dark_stats.get('ci_high', 0) - dark_stats.get('mean_beta', 0))
    ]

    colors = ['gold', 'darkslategray']
    x = np.arange(len(conditions))

    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.6)
    ax.errorbar(x, means, yerr=np.array(errors).T, fmt='none', color='black',
                capsize=10, capthick=2, linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylabel('Mean Trial-Level Beta', fontsize=11)
    ax.set_title('C. Light vs Dark Comparison\n(Error bars = 95% CI)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add significance annotations
    for i, (mean, stats) in enumerate(zip(means, [light_stats, dark_stats])):
        p = stats.get('one_sample_p', 1.0)
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        y_offset = 0.02 if mean >= 0 else -0.02
        va = 'bottom' if mean >= 0 else 'top'
        ax.text(i, mean + y_offset, sig, ha='center', va=va, fontsize=14, fontweight='bold')

    fig.suptitle(f'Per-Trial Analysis: Light vs Dark at {speed_threshold} cm/s\n'
                 f'(Hypothesis: β < 0 in light [correction], β > 0 in dark [drift])',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PURE TURN SECTIONS VISUALIZATION (RELAXED FILTERING)")
    print("="*80)

    # Load section data - RELAXED VERSION
    section_data_fn = os.path.join(RESULTS_PATH, "pure_turn_section_data_relaxed.csv")
    print(f"\nLoading section data: {section_data_fn}")

    if not os.path.exists(section_data_fn):
        print("ERROR: Section data file not found!")
        print("Please run pure_turn_sections_analysis_relaxed.py first")
        exit(1)

    section_data = pd.read_csv(section_data_fn)
    print(f"Loaded {len(section_data):,} rows")

    # Load regression results - RELAXED VERSION
    reg_results_fn = os.path.join(
        RESULTS_PATH, "pure_turn_section_regression_results_relaxed.csv")
    print(f"Loading regression results: {reg_results_fn}")

    if not os.path.exists(reg_results_fn):
        print("ERROR: Regression results file not found!")
        print("Please run pure_turn_sections_analysis_relaxed.py first")
        exit(1)

    regression_results = pd.read_csv(reg_results_fn)
    print(f"Loaded {len(regression_results)} conditions")

    # Load endpoint data (one row per section - fixes zero-clustering)
    endpoint_data_fn = os.path.join(RESULTS_PATH, "pure_turn_section_endpoints_relaxed.csv")
    endpoint_data = None
    if os.path.exists(endpoint_data_fn):
        print(f"Loading endpoint data: {endpoint_data_fn}")
        endpoint_data = pd.read_csv(endpoint_data_fn)
        print(f"Loaded {len(endpoint_data):,} section endpoints")
    else:
        print(f"Note: Endpoint data file not found: {endpoint_data_fn}")
        print("  Run pure_turn_sections_analysis_relaxed.py to generate endpoint data")

    # Load endpoint regression results
    endpoint_reg_fn = os.path.join(RESULTS_PATH, "pure_turn_section_endpoints_regression_relaxed.csv")
    endpoint_regression_results = None
    if os.path.exists(endpoint_reg_fn):
        print(f"Loading endpoint regression results: {endpoint_reg_fn}")
        endpoint_regression_results = pd.read_csv(endpoint_reg_fn)
        print(f"Loaded {len(endpoint_regression_results)} endpoint regression results")

    # Load per-animal regression results
    per_animal_fn = os.path.join(RESULTS_PATH, "pure_turn_section_per_animal_relaxed.csv")
    per_animal_results = None
    if os.path.exists(per_animal_fn):
        print(f"Loading per-animal results: {per_animal_fn}")
        per_animal_results = pd.read_csv(per_animal_fn)
        print(f"Loaded results for {len(per_animal_results)} animals")

    # Load model comparison results
    comparison_fn = os.path.join(RESULTS_PATH, "pure_turn_section_model_comparison_relaxed.csv")
    model_comparison = None
    if os.path.exists(comparison_fn):
        print(f"Loading model comparison: {comparison_fn}")
        model_comparison = pd.read_csv(comparison_fn)
        print(f"Loaded {len(model_comparison)} condition comparisons")

    # Create output directory for RELAXED analysis
    output_dir = os.path.join(FIGURES_PATH, "pure_turn_sections_analysis_relaxed")
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
    # Generate summary comparison plot (all timepoints)
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

    # =========================================================================
    # Generate per-animal comparison plots
    # =========================================================================
    if endpoint_data is not None and per_animal_results is not None:
        print("\n" + "="*80)
        print("GENERATING PER-ANIMAL COMPARISON PLOTS")
        print("="*80)

        # Create per_animal subdirectory
        per_animal_dir = os.path.join(output_dir, "per_animal")
        os.makedirs(per_animal_dir, exist_ok=True)

        # Figure 1: Pooled vs Per-Animal main comparison
        print("\nGenerating pooled vs per-animal comparison plot...")
        comparison_path = os.path.join(per_animal_dir, "pooled_vs_per_animal_comparison.png")
        fig = plot_pooled_vs_per_animal_comparison(
            endpoint_data, per_animal_results, comparison_path)
        if fig:
            plt.close(fig)
            print(f"  Saved: {comparison_path}")

        # Figure 2: Per-animal faceted scatter
        print("\nGenerating per-animal faceted scatter plot...")
        facets_path = os.path.join(per_animal_dir, "per_animal_facets.png")
        fig = plot_per_animal_facets(
            endpoint_data, per_animal_results, facets_path)
        if fig:
            plt.close(fig)
            print(f"  Saved: {facets_path}")

        # Figure 3: R^2 improvement bar chart
        if model_comparison is not None:
            print("\nGenerating R^2 improvement bar chart...")
            bars_path = os.path.join(per_animal_dir, "r_squared_improvement_bars.png")
            fig = plot_r_squared_improvement_bars(model_comparison, bars_path)
            if fig:
                plt.close(fig)
                print(f"  Saved: {bars_path}")

        # =====================================================================
        # CONDITION-SPECIFIC FIGURES (High R^2 conditions)
        # =====================================================================
        print("\n" + "-"*60)
        print("GENERATING CONDITION-SPECIFIC HIGH R^2 FIGURES")
        print("-"*60)

        # Best conditions with highest per-animal R^2
        best_conditions = [
            ('homingFromLeavingLever_light', 5.0),  # R^2 = 0.188
            ('all_dark', 5.0),                       # R^2 = 0.110
            ('homingFromLeavingLever_dark', 3.0),   # R^2 = 0.101
            ('atLever_dark', 5.0),                   # R^2 = 0.239
            ('searchToLeverPath_dark', 5.0),        # R^2 = 0.470 (HIGHEST!)
            ('searchToLeverPath_dark', 3.0),        # R^2 = 0.035, more data (n=285)
        ]

        for condition, speed in best_conditions:
            safe_name = f"{condition}_speed{speed}"
            print(f"\n  Condition: {condition}, speed >= {speed} cm/s")

            # Pooled vs per-animal comparison for this condition
            path = os.path.join(per_animal_dir, f"pooled_vs_per_animal_{safe_name}.png")
            fig = plot_pooled_vs_per_animal_comparison(
                endpoint_data, per_animal_results, path,
                condition_filter=condition, speed_filter=speed)
            if fig:
                plt.close(fig)

            # Per-animal facets for this condition
            path = os.path.join(per_animal_dir, f"per_animal_facets_{safe_name}.png")
            fig = plot_per_animal_facets(
                endpoint_data, per_animal_results, path,
                condition_filter=condition, speed_filter=speed)
            if fig:
                plt.close(fig)

        # Figure 4: Best conditions comparison (2x2 panel)
        if model_comparison is not None:
            print("\n  Generating best conditions comparison (2x2 panel)...")
            best_path = os.path.join(per_animal_dir, "best_conditions_comparison.png")
            fig = plot_best_conditions_comparison(endpoint_data, model_comparison, best_path)
            if fig:
                plt.close(fig)

    else:
        if endpoint_data is None:
            print("\nSkipping per-animal plots: endpoint data not available")
        if per_animal_results is None:
            print("\nSkipping per-animal plots: per-animal results not available")

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
    if endpoint_data is not None and per_animal_results is not None:
        print(f"\nPer-animal comparison figures saved to: {per_animal_dir}")
        print("  Overall (all data combined):")
        print("    - pooled_vs_per_animal_comparison.png: Side-by-side pooled vs per-animal")
        print("    - per_animal_facets.png: Individual animal scatter plots")
        if model_comparison is not None:
            print("    - r_squared_improvement_bars.png: R^2 improvement by condition")
        print("  Condition-specific (HIGH R^2 conditions):")
        print("    - pooled_vs_per_animal_searchToLeverPath_dark_speed5.0.png (R^2~0.47 HIGHEST!)")
        print("    - pooled_vs_per_animal_searchToLeverPath_dark_speed3.0.png (R^2~0.04, n=285)")
        print("    - pooled_vs_per_animal_homingFromLeavingLever_light_speed5.0.png (R^2~0.19)")
        print("    - pooled_vs_per_animal_all_dark_speed5.0.png (R^2~0.11)")
        print("    - pooled_vs_per_animal_homingFromLeavingLever_dark_speed3.0.png (R^2~0.10)")
        print("    - pooled_vs_per_animal_atLever_dark_speed5.0.png (R^2~0.24)")
        print("    - best_conditions_comparison.png: 2x2 panel summary of best conditions")
