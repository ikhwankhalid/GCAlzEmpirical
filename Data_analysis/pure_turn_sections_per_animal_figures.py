"""
Pure Turn Sections Analysis: Per-Animal Figures and Forest Plots

This script creates:
1. Individual animal scatter/endpoint plots for all 8 conditions
2. Cross-animal forest plots showing beta estimates with 95% CIs

Requires running pure_turn_sections_analysis_relaxed.py first to generate:
- pure_turn_section_endpoints_relaxed.csv
- pure_turn_section_per_animal_by_condition_relaxed.csv

Author: Analysis generated for Peng et al. 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t as t_dist
import os
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Plotting configuration
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

# Paths - update PROJECT_DATA_PATH as needed
PROJECT_DATA_PATH = '/workspace/Peng'
RESULTS_PATH = os.path.join(PROJECT_DATA_PATH, "results")
FIGURES_PATH = os.path.join(PROJECT_DATA_PATH, "figures")

# 7 animals in dataset (from useAble sessions in analysis script)
ANIMALS = ['jp486', 'jp3269', 'jp452', 'jp3120', 'jp451', 'mn8578', 'jp1686']

# Colorblind-friendly palette for animals
ANIMAL_COLORS = {
    'jp486': '#1f77b4',   # Blue (largest dataset)
    'jp3269': '#ff7f0e',  # Orange
    'jp3120': '#2ca02c',  # Green
    'jp451': '#d62728',   # Red
    'jp452': '#9467bd',   # Purple
    'jp1686': '#8c564b',  # Brown
    'mn8578': '#e377c2'   # Pink
}

# Conditions from analysis script
CONDITIONS = [
    'all_light', 'all_dark',
    'searchToLeverPath_light', 'searchToLeverPath_dark',
    'homingFromLeavingLever_light', 'homingFromLeavingLever_dark',
    'atLever_light', 'atLever_dark'
]

SPEED_THRESHOLDS = [1.0, 2.0, 3.0, 5.0]

# Minimum sections threshold (lowered from 10 to allow more animal/condition combinations)
MIN_SECTIONS = 5


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_animal_id(session_or_trial_id):
    """Extract animal ID from session or trial_id string."""
    # Handle trial_id format: "session_Tnumber"
    if '_T' in str(session_or_trial_id):
        session_part = str(session_or_trial_id).split('_T')[0]
    else:
        session_part = str(session_or_trial_id)
    # Extract animal ID (everything before first hyphen)
    match = re.match(r'^([^-]+)', session_part)
    return match.group(1) if match else session_part


def format_condition_name(condition):
    """Shorten condition name for display."""
    replacements = {
        'FromLeavingLever': '',
        'ToLeverPath': '',
        'homingFromLeavingLever': 'homing',
        'searchToLeverPath': 'search',
        'atLever': 'atLever'
    }
    result = condition
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.2f}"
    else:
        return f"p = {p:.2f}"


# =============================================================================
# INDIVIDUAL ANIMAL SCATTER PLOT (ALL TIMEPOINTS)
# =============================================================================

def plot_individual_animal_scatter(all_timepoints, animal_id, condition, speed_threshold,
                                    save_path=None, min_timepoints=50, max_plot_points=50000):
    """
    Create scatter plot with regression line for single animal using ALL TIMEPOINTS.

    Shows heading_deviation vs cumulative_turn with regression line and 95% CI band.
    Uses all timepoints within each pure turn section (not just endpoints).
    This reveals the "zero-clustering" artifact where early timepoints cluster near origin.

    Parameters
    ----------
    all_timepoints : DataFrame
        All timepoints data with integrated_ang_vel, mvtDirError, mouse columns
    animal_id : str
        Animal identifier (e.g., 'jp486')
    condition : str
        Behavioral condition
    speed_threshold : float
        Speed threshold in cm/s
    save_path : str, optional
        Path to save figure
    min_timepoints : int
        Minimum number of timepoints required
    max_plot_points : int
        Maximum points to plot (downsample if exceeded for performance)

    Returns
    -------
    fig : Figure or None
        Matplotlib figure, or None if insufficient data
    """
    # Filter data by condition and speed
    mask = ((all_timepoints['condition'] == condition) &
            (all_timepoints['speed_threshold'] == speed_threshold))

    data = all_timepoints[mask]

    # Filter by animal using 'mouse' column
    if 'mouse' in data.columns:
        animal_data = data[data['mouse'] == animal_id]
    elif 'animal_id' in data.columns:
        animal_data = data[data['animal_id'] == animal_id]
    else:
        return None

    X = animal_data['integrated_ang_vel'].values
    Y = animal_data['mvtDirError'].values
    valid = ~(np.isnan(X) | np.isnan(Y))
    X, Y = X[valid], Y[valid]

    n = len(X)
    if n < min_timepoints:
        return None

    # Compute regression on all data
    slope, intercept, r, p, se = stats.linregress(X, Y)
    r_squared = r ** 2

    # Compute 95% CI
    df = n - 2
    if df > 0:
        t_crit = t_dist.ppf(0.975, df)
        ci_lower = slope - t_crit * se
        ci_upper = slope + t_crit * se
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    # Downsample for plotting if too many points
    if n > max_plot_points:
        idx = np.random.choice(n, max_plot_points, replace=False)
        X_plot, Y_plot = X[idx], Y[idx]
        downsampled = True
    else:
        X_plot, Y_plot = X, Y
        downsampled = False

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    color = ANIMAL_COLORS.get(animal_id, '#333333')

    # Scatter points - smaller for dense data
    ax.scatter(X_plot, Y_plot, alpha=0.3, s=8, c=color, edgecolor='none',
               rasterized=True)

    # Regression line
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='black', linewidth=2.5, label=f'beta = {slope:.4f}')

    # 95% CI band for regression line
    x_mean = np.mean(X)
    se_fit = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((X - x_mean)**2))
    y_upper = y_line + t_crit * se_fit
    y_lower = y_line - t_crit * se_fit
    ax.fill_between(x_line, y_lower, y_upper, alpha=0.2, color=color)

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Title and labels
    ds_note = f" (showing {max_plot_points:,}/{n:,})" if downsampled else ""
    short_cond = format_condition_name(condition)
    ax.set_title(f'{animal_id} - {short_cond} [ALL TIMEPOINTS]{ds_note}\n'
                 f'speed >= {speed_threshold} cm/s | n = {n:,} timepoints\n'
                 f'beta = {slope:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], '
                 f'R^2 = {r_squared:.4f}, {format_pvalue(p)}',
                 fontsize=11, fontweight='bold', color=color)

    ax.set_xlabel('Cumulative Turn in Section (rad)', fontsize=12)
    ax.set_ylabel('Heading Deviation (rad)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# INDIVIDUAL ANIMAL ENDPOINT PLOT
# =============================================================================

def plot_individual_animal_endpoints(endpoints, animal_id, condition, speed_threshold,
                                      save_path=None, min_sections=10):
    """
    Create endpoint-only scatter plot for single animal (one point per section).

    This version uses the endpoint data directly, with one point per pure turn section.

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data with integrated_ang_vel, mvtDirError columns
    animal_id : str
        Animal identifier
    condition : str
        Behavioral condition
    speed_threshold : float
        Speed threshold in cm/s
    save_path : str, optional
        Path to save figure
    min_sections : int
        Minimum number of sections required

    Returns
    -------
    fig : Figure or None
        Matplotlib figure, or None if insufficient data
    """
    # Filter data
    mask = ((endpoints['condition'] == condition) &
            (endpoints['speed_threshold'] == speed_threshold))

    data = endpoints[mask].copy()

    # Add animal_id if needed
    if 'animal_id' not in data.columns:
        if 'trial_id' in data.columns:
            data['animal_id'] = data['trial_id'].apply(
                lambda x: extract_animal_id(x.split('_')[0]) if '_' in str(x) else extract_animal_id(x))
        elif 'session' in data.columns:
            data['animal_id'] = data['session'].apply(extract_animal_id)
        else:
            return None

    animal_data = data[data['animal_id'] == animal_id]

    X = animal_data['integrated_ang_vel'].values
    Y = animal_data['mvtDirError'].values
    valid = ~(np.isnan(X) | np.isnan(Y))
    X, Y = X[valid], Y[valid]

    n = len(X)
    if n < min_sections:
        return None

    # Compute regression
    slope, intercept, r, p, se = stats.linregress(X, Y)
    r_squared = r ** 2

    # Compute 95% CI
    df = n - 2
    if df > 0:
        t_crit = t_dist.ppf(0.975, df)
        ci_lower = slope - t_crit * se
        ci_upper = slope + t_crit * se
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    color = ANIMAL_COLORS.get(animal_id, '#333333')

    # Scatter points - larger for endpoints
    ax.scatter(X, Y, alpha=0.6, s=40, c=color, edgecolor='white', linewidth=0.5,
               rasterized=True, marker='o')

    # Regression line
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='black', linewidth=2.5, label=f'beta = {slope:.4f}')

    # 95% CI band for regression line
    x_mean = np.mean(X)
    se_fit = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((X - x_mean)**2))
    y_upper = y_line + t_crit * se_fit
    y_lower = y_line - t_crit * se_fit
    ax.fill_between(x_line, y_lower, y_upper, alpha=0.2, color=color)

    # Reference lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Title and labels
    sparse_note = " (sparse)" if n < 30 else ""
    short_cond = format_condition_name(condition)
    ax.set_title(f'{animal_id} - {short_cond} [ENDPOINT]{sparse_note}\n'
                 f'speed >= {speed_threshold} cm/s | n = {n} sections\n'
                 f'beta = {slope:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], '
                 f'R^2 = {r_squared:.4f}, {format_pvalue(p)}',
                 fontsize=11, fontweight='bold', color=color)

    ax.set_xlabel('Total Cumulative Turn in Section (rad)', fontsize=12)
    ax.set_ylabel('Heading Deviation at Section End (rad)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# FOREST PLOT: CROSS-ANIMAL BETA COMPARISON
# =============================================================================

def plot_forest_beta_by_animal(per_animal_df, condition, speed_threshold,
                                include_pooled=True, save_path=None):
    """
    Create forest plot showing beta estimates with 95% CIs for each animal.

    Parameters
    ----------
    per_animal_df : DataFrame
        Per-animal-condition regression results with columns:
        animal_id, condition, speed_threshold, beta, se, ci_lower, ci_upper, n_sections
    condition : str
        Condition to plot
    speed_threshold : float
        Speed threshold to plot
    include_pooled : bool
        Whether to show pooled (inverse-variance weighted) estimate
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : Figure or None
        Matplotlib figure, or None if no data
    """
    # Filter to condition/speed
    mask = ((per_animal_df['condition'] == condition) &
            (per_animal_df['speed_threshold'] == speed_threshold))
    df = per_animal_df[mask].copy()

    if len(df) == 0:
        return None

    # Sort by beta value for visual clarity
    df = df.sort_values('beta', ascending=True).reset_index(drop=True)

    # Compute pooled estimate (inverse-variance weighted)
    if include_pooled and len(df) > 1:
        # Avoid division by zero
        valid_se = df['se'].values
        valid_se = np.where(valid_se > 0, valid_se, 1e-10)
        weights = 1 / (valid_se ** 2)
        pooled_beta = np.sum(weights * df['beta'].values) / np.sum(weights)
        pooled_se = 1 / np.sqrt(np.sum(weights))
        pooled_ci_lower = pooled_beta - 1.96 * pooled_se
        pooled_ci_upper = pooled_beta + 1.96 * pooled_se
    else:
        include_pooled = False

    # Create figure
    n_animals = len(df)
    fig_height = max(4, 0.7 * n_animals + (1.5 if include_pooled else 0.5))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n_animals)

    # Plot each animal
    for i, (_, row) in enumerate(df.iterrows()):
        animal_id = row['animal_id']
        beta = row['beta']
        ci_low = row['ci_lower']
        ci_high = row['ci_upper']
        n = row['n_sections']
        p = row['p_value']

        color = ANIMAL_COLORS.get(animal_id, '#333333')

        # Determine marker based on significance
        if p < 0.05:
            marker = 'o'
            alpha = 1.0
            linewidth = 2.5
        else:
            marker = 's'
            alpha = 0.6
            linewidth = 1.5

        # Point estimate
        ax.scatter(beta, i, s=100, c=color, marker=marker, zorder=3,
                   edgecolor='black', linewidth=0.5, alpha=alpha)

        # CI line
        ax.hlines(i, ci_low, ci_high, colors=color, linewidth=linewidth, alpha=alpha)

        # Add sample size annotation on the right
        ci_max = max(df['ci_upper'].max(), pooled_ci_upper if include_pooled else 0)
        ax.annotate(f'n={int(n)}', (ci_max + 0.02, i), fontsize=9, va='center',
                    color='gray', ha='left')

        # Significance stars
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = ''

        if stars:
            ax.annotate(stars, (ci_max + 0.08, i), fontsize=12, fontweight='bold',
                        va='center', color=color, ha='left')

    # Add pooled estimate
    if include_pooled:
        y_pooled = n_animals + 0.7
        ax.scatter(pooled_beta, y_pooled, s=150, c='black', marker='D', zorder=3,
                   edgecolor='white', linewidth=1)
        ax.hlines(y_pooled, pooled_ci_lower, pooled_ci_upper, colors='black',
                  linewidth=3)

        # Add horizontal separator
        ax.axhline(n_animals + 0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Reference line at zero
    ax.axvline(0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)

    # Y-axis labels
    y_ticks = list(y_positions)
    y_labels = list(df['animal_id'].values)
    if include_pooled:
        y_ticks.append(n_animals + 0.7)
        y_labels.append('Pooled')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)

    # Color y-tick labels by animal
    for i, label in enumerate(ax.get_yticklabels()):
        if i < len(df):
            animal_id = df.iloc[i]['animal_id']
            label.set_color(ANIMAL_COLORS.get(animal_id, 'black'))
            label.set_fontweight('bold')
        else:
            # Pooled label
            label.set_fontweight('bold')

    # Labels and title
    short_cond = format_condition_name(condition)
    ax.set_xlabel('Beta (Cumulative Turn -> Heading Deviation)', fontsize=12)
    ax.set_title(f'Cross-Animal Comparison: {short_cond}\n'
                 f'speed >= {speed_threshold} cm/s | '
                 f'* p<0.05, ** p<0.01, *** p<0.001',
                 fontsize=12, fontweight='bold')

    # Expand y-limits
    y_max = n_animals + (2.0 if include_pooled else 0.5)
    ax.set_ylim(-0.5, y_max)

    # Expand x-limits for annotations
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + 0.15 * (x_max - x_min))

    # Grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# SUMMARY FOREST PLOT: ALL CONDITIONS
# =============================================================================

def plot_forest_summary_all_conditions(per_animal_df, speed_threshold, save_path=None):
    """
    Create summary forest plot showing mean beta per animal across all conditions.

    Parameters
    ----------
    per_animal_df : DataFrame
        Per-animal-condition regression results
    speed_threshold : float
        Speed threshold to plot
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : Figure or None
    """
    # Filter to speed
    df = per_animal_df[per_animal_df['speed_threshold'] == speed_threshold].copy()

    if len(df) == 0:
        return None

    # Compute mean beta per animal across conditions
    summary = df.groupby('animal_id').agg({
        'beta': ['mean', 'std', 'count'],
        'n_sections': 'sum'
    }).reset_index()
    summary.columns = ['animal_id', 'mean_beta', 'std_beta', 'n_conditions', 'total_sections']

    # Compute SE of mean
    summary['se_mean'] = summary['std_beta'] / np.sqrt(summary['n_conditions'])
    summary['ci_lower'] = summary['mean_beta'] - 1.96 * summary['se_mean']
    summary['ci_upper'] = summary['mean_beta'] + 1.96 * summary['se_mean']

    # Sort by mean beta
    summary = summary.sort_values('mean_beta', ascending=True).reset_index(drop=True)

    # Create figure
    n_animals = len(summary)
    fig_height = max(4, 0.7 * n_animals + 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n_animals)

    for i, (_, row) in enumerate(summary.iterrows()):
        animal_id = row['animal_id']
        color = ANIMAL_COLORS.get(animal_id, '#333333')

        # Point estimate
        ax.scatter(row['mean_beta'], i, s=100, c=color, marker='o', zorder=3,
                   edgecolor='black', linewidth=0.5)

        # CI line
        ax.hlines(i, row['ci_lower'], row['ci_upper'], colors=color, linewidth=2.5)

        # Annotation
        ax.annotate(f"n={int(row['n_conditions'])} cond, {int(row['total_sections'])} sect",
                    (row['ci_upper'] + 0.02, i), fontsize=9, va='center', color='gray')

    # Reference line at zero
    ax.axvline(0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary['animal_id'].values, fontsize=11)

    for i, label in enumerate(ax.get_yticklabels()):
        animal_id = summary.iloc[i]['animal_id']
        label.set_color(ANIMAL_COLORS.get(animal_id, 'black'))
        label.set_fontweight('bold')

    ax.set_xlabel('Mean Beta Across Conditions (95% CI)', fontsize=12)
    ax.set_title(f'Cross-Animal Summary: Mean Beta Across All Conditions\n'
                 f'speed >= {speed_threshold} cm/s',
                 fontsize=12, fontweight='bold')

    ax.set_ylim(-0.5, n_animals + 0.5)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PER-ANIMAL FIGURES AND FOREST PLOTS")
    print("="*80)

    # Load data files
    all_timepoints_fn = os.path.join(RESULTS_PATH, "pure_turn_section_data_relaxed.csv")
    endpoint_fn = os.path.join(RESULTS_PATH, "pure_turn_section_endpoints_relaxed.csv")
    per_animal_cond_fn = os.path.join(RESULTS_PATH, "pure_turn_section_per_animal_by_condition_relaxed.csv")

    print(f"\nLoading data from:")
    print(f"  All timepoints: {all_timepoints_fn}")
    print(f"  Endpoints: {endpoint_fn}")
    print(f"  Per-animal by condition: {per_animal_cond_fn}")

    # Check if files exist
    if not os.path.exists(all_timepoints_fn):
        print(f"\nERROR: All timepoints file not found: {all_timepoints_fn}")
        print("Please run pure_turn_sections_analysis_relaxed.py first.")
        exit(1)

    if not os.path.exists(endpoint_fn):
        print(f"\nERROR: Endpoint file not found: {endpoint_fn}")
        print("Please run pure_turn_sections_analysis_relaxed.py first.")
        exit(1)

    if not os.path.exists(per_animal_cond_fn):
        print(f"\nERROR: Per-animal-by-condition file not found: {per_animal_cond_fn}")
        print("Please run pure_turn_sections_analysis_relaxed.py first.")
        exit(1)

    print("\nLoading all timepoints data (this may take a moment)...")
    all_timepoints = pd.read_csv(all_timepoints_fn)
    print(f"  All timepoints: {len(all_timepoints):,} rows")

    endpoints = pd.read_csv(endpoint_fn)
    print(f"  Endpoints: {len(endpoints):,} rows")

    per_animal_by_condition = pd.read_csv(per_animal_cond_fn)
    print(f"  Per-animal by condition: {len(per_animal_by_condition)} rows")

    # Create output directories
    base_dir = os.path.join(FIGURES_PATH, "pure_turn_sections_analysis_relaxed", "per_animal")
    individual_dir = os.path.join(base_dir, "individual_animals")
    forest_dir = os.path.join(base_dir, "forest_plots")
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(forest_dir, exist_ok=True)

    print(f"\nOutput directories:")
    print(f"  Individual animals: {individual_dir}")
    print(f"  Forest plots: {forest_dir}")

    # Get list of animals from the data
    animals_in_data = per_animal_by_condition['animal_id'].unique()
    print(f"\nAnimals in data: {list(animals_in_data)}")

    # ==========================================================================
    # GENERATE INDIVIDUAL ANIMAL FIGURES
    # ==========================================================================
    print("\n" + "="*60)
    print("GENERATING INDIVIDUAL ANIMAL FIGURES")
    print("="*60)

    n_scatter_created = 0
    n_endpoint_created = 0

    for animal_id in animals_in_data:
        animal_dir = os.path.join(individual_dir, animal_id)
        os.makedirs(animal_dir, exist_ok=True)

        print(f"\n  Processing {animal_id}...")

        for condition in CONDITIONS:
            for speed in SPEED_THRESHOLDS:
                # Scatter plot (uses ALL TIMEPOINTS data)
                save_path = os.path.join(animal_dir,
                    f"scatter_{condition}_speed{speed}.png")
                fig = plot_individual_animal_scatter(
                    all_timepoints, animal_id, condition, speed, save_path)
                if fig:
                    plt.close(fig)
                    n_scatter_created += 1

                # Endpoint plot (uses ENDPOINT data - one per section)
                save_path = os.path.join(animal_dir,
                    f"endpoint_{condition}_speed{speed}.png")
                fig = plot_individual_animal_endpoints(
                    endpoints, animal_id, condition, speed, save_path, MIN_SECTIONS)
                if fig:
                    plt.close(fig)
                    n_endpoint_created += 1

    print(f"\n  Created {n_scatter_created} scatter plots")
    print(f"  Created {n_endpoint_created} endpoint plots")

    # ==========================================================================
    # GENERATE FOREST PLOTS
    # ==========================================================================
    print("\n" + "="*60)
    print("GENERATING FOREST PLOTS")
    print("="*60)

    n_forest_created = 0

    for condition in CONDITIONS:
        for speed in SPEED_THRESHOLDS:
            save_path = os.path.join(forest_dir,
                f"forest_beta_{condition}_speed{speed}.png")
            fig = plot_forest_beta_by_animal(
                per_animal_by_condition, condition, speed, save_path=save_path)
            if fig:
                plt.close(fig)
                n_forest_created += 1
                print(f"  Created: forest_beta_{condition}_speed{speed}.png")

    print(f"\n  Created {n_forest_created} condition-specific forest plots")

    # Summary forest plots (one per speed threshold)
    print("\n  Creating summary forest plots...")
    for speed in SPEED_THRESHOLDS:
        save_path = os.path.join(forest_dir,
            f"forest_summary_all_conditions_speed{speed}.png")
        fig = plot_forest_summary_all_conditions(
            per_animal_by_condition, speed, save_path=save_path)
        if fig:
            plt.close(fig)
            print(f"  Created: forest_summary_all_conditions_speed{speed}.png")

    # ==========================================================================
    # COMPLETE
    # ==========================================================================
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nTotal figures created:")
    print(f"  - {n_scatter_created} individual animal scatter plots (ALL TIMEPOINTS data)")
    print(f"  - {n_endpoint_created} individual animal endpoint plots (ENDPOINT data - one per section)")
    print(f"  - {n_forest_created} condition-specific forest plots")
    print(f"  - {len(SPEED_THRESHOLDS)} summary forest plots")
    print(f"\nFigures saved to: {base_dir}")
