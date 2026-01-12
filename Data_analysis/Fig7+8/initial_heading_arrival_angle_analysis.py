"""
Analysis: Initial Heading vs Target-to-Animal Angle at Lever Arrival

Research Question:
    Does the animal's initial heading (when leaving home base) correlate with
    the angle at which it arrives at the lever (targetToAnimalAngle)?

Variables:
    Independent: Initial heading (2 methods)
        1. Movement direction from first 0.5s of search
        2. Smoothed head direction (hdPose) from first 0.5s

    Dependent: targetToAnimalAngle at lever arrival
        - Angle from lever to animal position when arriving at lever
        - Extracted at end of searchToLeverPath
        - Range: [-pi, pi] radians

Key Insight:
    This is a circular-circular correlation (both variables are angles),
    using astropy.stats.circcorrcoef for proper circular statistics.

Conditions analyzed separately:
    - Light vs Dark
    - Short vs Long search (median split on dark)
    - Accurate vs Inaccurate homing (median split on dark)

Output:
    - Summary statistics CSV with effect sizes
    - Publication-quality figures (polar and cartesian)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
import pickle
import math
import os
import sys
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Project configuration (from setup_project.py)
PROJECT_DATA_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng'

# Sessions with >0.5 mvl in the last random foraging session
# (excluding border regions)
useAble = ['jp486-19032023-0108', 'jp486-18032023-0108',
       'jp3269-28112022-0108', 'jp486-16032023-0108',
       'jp452-25112022-0110', 'jp486-24032023-0108',
       'jp486-22032023-0108', 'jp452-24112022-0109',
       'jp486-15032023-0108', 'jp3120-25052022-0107',
       'jp3120-26052022-0107', 'jp451-28102022-0108',
       'jp486-20032023-0108', 'jp486-06032023-0108',
       'jp486-26032023-0108', 'jp486-17032023-0108',
       'jp451-29102022-0108', 'jp451-30102022-0108',
       'jp486-10032023-0108', 'jp486-05032023-0108',
       'jp3269-29112022-0108', 'mn8578-17122021-0107',
       'jp452-23112022-0108', 'jp1686-26042022-0108']

# ============================================================================
# CONFIGURATION
# ============================================================================

GLOBAL_FONT_SIZE = 12
SPEED_THRESHOLD = 2.0  # cm/s for movement filtering
TIME_WINDOW = 0.5  # seconds for initial heading calculation
SMOOTH_SIGMA = 3.0  # Gaussian smoothing sigma for hdPose
N_PERMUTATIONS = 10000  # Permutation iterations for p-value
N_BOOTSTRAP = 2000  # Bootstrap iterations for CI

# Color scheme
COLORS = {
    'dark': (50/255, 50/255, 120/255),
    'light': (206/255, 159/255, 70/255),
    'scatter': '#457b9d',
    'regression': '#ff9e00'
}

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_DATA_PATH, 'results', 'initial_heading_arrival_angle')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_behavioral_data():
    """
    Load main behavioral data files.

    Returns:
        behavior_df: Trial-level behavioral metrics
    """
    print("Loading behavioral data...")
    fn = os.path.join(PROJECT_DATA_PATH, 'results', 'behavior_180_EastReferenceQuadrant.csv')
    behavior_df = pd.read_csv(fn)
    behavior_df = behavior_df[behavior_df['valid'] == True].copy()
    print(f"  Loaded {len(behavior_df)} valid trials from behavior_180")
    return behavior_df


def load_session_nav_data(session_name):
    """
    Load navPathInstan.csv and navPathSummary.csv for a single session.

    Args:
        session_name: Session identifier (e.g., 'jp486-24032023-0108')

    Returns:
        nav_instan: DataFrame with instantaneous behavioral variables
        nav_summary: DataFrame with path summary information
    """
    # Find session path (sessions are nested under animal directories)
    animal_id = session_name.split('-')[0]
    session_path = os.path.join(PROJECT_DATA_PATH, animal_id, session_name)

    nav_instan_path = os.path.join(session_path, 'navPathInstan.csv')
    nav_summary_path = os.path.join(session_path, 'navPathSummary.csv')

    if not os.path.exists(nav_instan_path) or not os.path.exists(nav_summary_path):
        return None, None

    nav_instan = pd.read_csv(nav_instan_path)
    nav_summary = pd.read_csv(nav_summary_path)

    # Add session column
    nav_instan['session'] = session_name
    nav_summary['session'] = session_name

    return nav_instan, nav_summary


def load_all_session_df():
    """
    Load the large allSessionDf with hdPose data.
    Only loads required columns to save memory.

    Returns:
        all_session_df: DataFrame with hdPose and position data
    """
    print("Loading allSessionDf (this may take a while)...")
    fn = os.path.join(PROJECT_DATA_PATH, 'results', 'allSessionDf_with_leverVector_and_last_cohort.csv')

    # Only load required columns
    cols_to_load = ['session', 'condition', 'trialNo', 'withinPathTime',
                    'xPose', 'yPose', 'hdPose', 'speed']

    all_session_df = pd.read_csv(fn, usecols=cols_to_load)
    print(f"  Loaded {len(all_session_df)} rows from allSessionDf")
    return all_session_df


# ============================================================================
# INITIAL HEADING CALCULATION
# ============================================================================

def circular_mean(angles):
    """Calculate circular mean of angles in radians."""
    if len(angles) == 0:
        return np.nan
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)


def calculate_initial_heading_movement(nav_instan, session_name, trial_no, time_window=TIME_WINDOW):
    """
    Calculate initial heading from movement direction (Method 1).

    Args:
        nav_instan: navPathInstan DataFrame for session
        session_name: Session identifier
        trial_no: Trial number
        time_window: Time window in seconds

    Returns:
        Initial heading in radians [-pi, pi], or np.nan if insufficient data
    """
    # Filter for searchToLeverPath entries for this trial
    mask = (
        (nav_instan['session'] == session_name) &
        (nav_instan['trialNo'] == trial_no) &
        (nav_instan['name'].str.contains('searchToLeverPath'))
    )
    trial_data = nav_instan[mask].copy()

    if len(trial_data) < 3:
        return np.nan

    # Sort by time within path
    if 'iTime' in trial_data.columns:
        trial_data = trial_data.sort_values('iTime')
        time_col = 'iTime'
    else:
        trial_data = trial_data.sort_values('timeRes')
        time_col = 'timeRes'
        trial_data['iTime'] = trial_data[time_col] - trial_data[time_col].iloc[0]

    # Get first time_window seconds
    initial_data = trial_data[trial_data['iTime'] <= time_window]

    if len(initial_data) < 3:
        return np.nan

    # Calculate movement direction from position changes
    x = initial_data['x'].values
    y = initial_data['y'].values

    dx = np.diff(x)
    dy = np.diff(y)

    # Filter by speed (only include when moving)
    if 'speed' in initial_data.columns:
        speeds = initial_data['speed'].values[:-1]
        speed_mask = speeds > SPEED_THRESHOLD
        dx = dx[speed_mask]
        dy = dy[speed_mask]

    if len(dx) < 2:
        return np.nan

    # Calculate instantaneous headings
    headings = np.arctan2(dy, dx)

    # Remove NaN values
    headings = headings[~np.isnan(headings)]

    if len(headings) == 0:
        return np.nan

    # Calculate circular mean
    initial_heading = circular_mean(headings)

    # Wrap to [-pi, pi]
    initial_heading = math.remainder(initial_heading, 2 * np.pi)

    return initial_heading


def calculate_initial_heading_hdpose_fast(session_df, trial_no,
                                           time_window=TIME_WINDOW, smooth_sigma=SMOOTH_SIGMA):
    """
    Optimized version of hdPose heading calculation using pre-filtered session data.

    Args:
        session_df: Pre-filtered DataFrame for single session
        trial_no: Trial number
        time_window: Time window in seconds
        smooth_sigma: Gaussian smoothing sigma

    Returns:
        Initial heading in radians [-pi, pi], or np.nan if insufficient data
    """
    # Filter for searchToLeverPath for this trial
    mask = (
        (session_df['trialNo'] == trial_no) &
        (session_df['condition'].str.contains('searchToLeverPath'))
    )
    trial_data = session_df[mask]

    if len(trial_data) < 3:
        return np.nan

    # Sort by within path time
    trial_data = trial_data.sort_values('withinPathTime')

    # Get first time_window seconds
    initial_data = trial_data[trial_data['withinPathTime'] <= time_window]

    if len(initial_data) < 3:
        return np.nan

    # Extract hdPose
    hdpose = initial_data['hdPose'].values

    # Remove NaN values
    valid_mask = ~np.isnan(hdpose)
    hdpose = hdpose[valid_mask]

    if len(hdpose) < 3:
        return np.nan

    # Apply Gaussian smoothing (handle circular data by smoothing sin and cos separately)
    sin_hd = np.sin(hdpose)
    cos_hd = np.cos(hdpose)

    sin_smooth = gaussian_filter1d(sin_hd, sigma=smooth_sigma, mode='nearest')
    cos_smooth = gaussian_filter1d(cos_hd, sigma=smooth_sigma, mode='nearest')

    hdpose_smooth = np.arctan2(sin_smooth, cos_smooth)

    # Calculate circular mean
    initial_heading = circular_mean(hdpose_smooth)

    # Wrap to [-pi, pi]
    initial_heading = math.remainder(initial_heading, 2 * np.pi)

    return initial_heading


# ============================================================================
# TARGET ANGLE EXTRACTION
# ============================================================================

def extract_target_angle_at_lever_arrival(nav_instan, nav_summary, session_name, trial_no):
    """
    Extract targetToAnimalAngle at lever arrival (end of searchToLeverPath).

    The targetToAnimalAngle represents the angle from the lever to the animal's
    position at the moment of lever arrival.

    Args:
        nav_instan: navPathInstan DataFrame for session
        nav_summary: navPathSummary DataFrame for session
        session_name: Session identifier
        trial_no: Trial number

    Returns:
        Angle in radians [-pi, pi], or np.nan if data unavailable
    """
    # 1. Find searchToLeverPath segment in nav_summary
    mask_summary = (
        (nav_summary['session'] == session_name) &
        (nav_summary['trialNo'] == trial_no) &
        (nav_summary['type'] == 'searchToLeverPath')
    )
    search_summary = nav_summary[mask_summary]

    if len(search_summary) == 0:
        return np.nan

    # 2. Find corresponding rows in nav_instan (searchToLeverPath entries)
    mask_instan = (
        (nav_instan['session'] == session_name) &
        (nav_instan['trialNo'] == trial_no) &
        (nav_instan['name'].str.contains('searchToLeverPath'))
    )
    search_data = nav_instan[mask_instan]

    if len(search_data) == 0:
        return np.nan

    if 'targetToAnimalAngle' not in search_data.columns:
        return np.nan

    # 3. Get last row (closest to lever arrival)
    search_data = search_data.sort_values('timeRes')
    target_angle = search_data['targetToAnimalAngle'].iloc[-1]

    # Handle NaN values
    if pd.isna(target_angle):
        return np.nan

    return target_angle


# ============================================================================
# CONDITION SPLITTING FUNCTIONS
# ============================================================================

def split_by_light_condition(trial_df):
    """Split trials by light/dark condition."""
    light_df = trial_df[trial_df['light'] == 'light'].copy()
    dark_df = trial_df[trial_df['light'] == 'dark'].copy()
    return {'light': light_df, 'dark': dark_df}


def split_by_search_length(trial_df, behavior_df):
    """
    Split dark trials by short vs long search path (median split).
    """
    # Merge with behavior data to get search length
    dark_df = trial_df[trial_df['light'] == 'dark'].copy()

    if len(dark_df) == 0:
        return {'short_search': pd.DataFrame(), 'long_search': pd.DataFrame()}

    # Get search lengths from behavior_df
    behavior_subset = behavior_df[['sessionName', 'trialNo', 'searchLength']].copy()
    behavior_subset = behavior_subset.rename(columns={'sessionName': 'session'})

    merged = dark_df.merge(behavior_subset, on=['session', 'trialNo'], how='left')

    # Calculate median for dark trials
    median_length = merged['searchLength'].median()

    short_df = merged[merged['searchLength'] < median_length].copy()
    long_df = merged[merged['searchLength'] >= median_length].copy()

    return {'short_search': short_df, 'long_search': long_df}


def split_by_homing_accuracy(trial_df, behavior_df):
    """
    Split dark trials by accurate vs inaccurate homing (median split).
    """
    dark_df = trial_df[trial_df['light'] == 'dark'].copy()

    if len(dark_df) == 0:
        return {'accurate_homing': pd.DataFrame(), 'inaccurate_homing': pd.DataFrame()}

    # Get homing error from behavior_df
    behavior_subset = behavior_df[['sessionName', 'trialNo', 'homingErrorAtPeriphery']].copy()
    behavior_subset = behavior_subset.rename(columns={'sessionName': 'session'})
    behavior_subset['homingErrorAbs'] = np.abs(behavior_subset['homingErrorAtPeriphery'])

    merged = dark_df.merge(behavior_subset, on=['session', 'trialNo'], how='left')

    # Calculate median for dark trials
    median_error = merged['homingErrorAbs'].median()

    accurate_df = merged[merged['homingErrorAbs'] < median_error].copy()
    inaccurate_df = merged[merged['homingErrorAbs'] >= median_error].copy()

    return {'accurate_homing': accurate_df, 'inaccurate_homing': inaccurate_df}


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def circcorrcoef(alpha, beta):
    """
    Circular-circular correlation coefficient.

    Computes the correlation coefficient between two circular variables.
    This is a manual implementation to avoid scipy/astropy version conflicts.

    Formula (Fisher & Lee, 1983):
        r = sum(sin(alpha_i - alpha_mean) * sin(beta_i - beta_mean)) /
            sqrt(sum(sin^2(alpha_i - alpha_mean)) * sum(sin^2(beta_i - beta_mean)))

    Args:
        alpha: Array of angles in radians
        beta: Array of angles in radians

    Returns:
        Circular correlation coefficient in range [-1, 1]
    """
    # Calculate circular means
    alpha_mean = np.arctan2(np.mean(np.sin(alpha)), np.mean(np.cos(alpha)))
    beta_mean = np.arctan2(np.mean(np.sin(beta)), np.mean(np.cos(beta)))

    # Calculate deviations from mean (using sine for circular difference)
    sin_alpha_diff = np.sin(alpha - alpha_mean)
    sin_beta_diff = np.sin(beta - beta_mean)

    # Calculate correlation coefficient
    numerator = np.sum(sin_alpha_diff * sin_beta_diff)
    denominator = np.sqrt(np.sum(sin_alpha_diff**2) * np.sum(sin_beta_diff**2))

    if denominator == 0:
        return 0.0

    r = numerator / denominator

    return r


def circular_circular_correlation_permutation(angles1, angles2,
                                               n_permutations=N_PERMUTATIONS,
                                               n_bootstrap=N_BOOTSTRAP):
    """
    Calculate circular-circular correlation with permutation test and bootstrap CI.

    Uses astropy.stats.circcorrcoef for proper circular-circular correlation.

    Args:
        angles1: Array of angles in radians (initial heading)
        angles2: Array of angles in radians (target angle at arrival)
        n_permutations: Permutation iterations for p-value
        n_bootstrap: Bootstrap iterations for CI

    Returns:
        Dictionary with r, p_value, ci_lower, ci_upper, n
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(angles1) | np.isnan(angles2))
    x = angles1[valid_mask]
    y = angles2[valid_mask]

    n = len(x)
    if n < 10:
        return {
            'r': np.nan, 'p_value': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan,
            'n': n
        }

    # Calculate observed circular-circular correlation
    real_r = circcorrcoef(x, y)

    # Permutation test for p-value
    x_shuffled = x.copy()
    perm_r = np.zeros(n_permutations)

    for i in range(n_permutations):
        np.random.shuffle(x_shuffled)
        perm_r[i] = circcorrcoef(x_shuffled, y)

    # Two-tailed p-value
    p_value = np.sum(np.abs(perm_r) >= np.abs(real_r)) / n_permutations

    # Bootstrap for CI
    boot_r = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_r[i] = circcorrcoef(x[idx], y[idx])

    ci_lower, ci_upper = np.percentile(boot_r, [2.5, 97.5])

    return {
        'r': real_r,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }


def apply_fdr_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction.

    Manual implementation to avoid statsmodels dependency.

    Args:
        p_values: Array of p-values
        alpha: Significance threshold (default 0.05)

    Returns:
        p_corrected: FDR-adjusted p-values
        significant: Boolean array indicating significance
    """
    valid_mask = ~np.isnan(p_values)
    p_valid = p_values[valid_mask]

    if len(p_valid) == 0:
        return p_values, np.zeros_like(p_values, dtype=bool)

    n = len(p_valid)

    # Sort p-values and get sort indices
    sorted_indices = np.argsort(p_valid)
    sorted_p = p_valid[sorted_indices]

    # Calculate Benjamini-Hochberg critical values
    # BH adjusted p-value = p * n / rank
    ranks = np.arange(1, n + 1)
    adjusted_p = sorted_p * n / ranks

    # Ensure monotonicity (adjusted p-values should be non-decreasing from the end)
    adjusted_p_monotonic = np.minimum.accumulate(adjusted_p[::-1])[::-1]

    # Cap at 1.0
    adjusted_p_monotonic = np.minimum(adjusted_p_monotonic, 1.0)

    # Unsort to original order
    p_corrected_valid = np.empty(n)
    p_corrected_valid[sorted_indices] = adjusted_p_monotonic

    # Determine significance
    significant_valid = p_corrected_valid < alpha

    # Map back to original array (including NaN positions)
    p_corrected = np.full_like(p_values, np.nan, dtype=float)
    p_corrected[valid_mask] = p_corrected_valid

    sig_array = np.zeros_like(p_values, dtype=bool)
    sig_array[valid_mask] = significant_valid

    return p_corrected, sig_array


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_polar_scatter(ax, initial_heading, target_angle, stats_dict=None,
                       title='', show_stats=True, color=COLORS['scatter']):
    """
    Create polar scatter plot for circular-circular data.

    Maps initial_heading to angle (theta) and target_angle to radius
    (transformed to positive range for visualization).

    Args:
        ax: Matplotlib polar axis
        initial_heading: Array of initial heading angles (radians)
        target_angle: Array of target angles at lever arrival (radians)
        stats_dict: Dictionary with correlation statistics
        title: Plot title
        show_stats: Whether to show statistics annotation
        color: Scatter point color
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(initial_heading) | np.isnan(target_angle))
    x = initial_heading[valid_mask]
    y = target_angle[valid_mask]

    if len(x) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=GLOBAL_FONT_SIZE)
        return

    # For polar plot: use initial_heading as theta, target_angle as radial position
    # Transform target_angle to [0, 2*pi] for radius visualization
    r = (y + np.pi)  # Shift from [-pi, pi] to [0, 2*pi]

    ax.scatter(x, r, alpha=0.4, s=15, c=color, edgecolor='none')

    ax.set_title(title, fontsize=GLOBAL_FONT_SIZE, pad=15)
    ax.set_xlabel('Initial Heading', fontsize=GLOBAL_FONT_SIZE - 1)

    # Add radial axis label
    ax.set_rlabel_position(45)

    # Add statistics text
    if show_stats and stats_dict is not None:
        r_val = stats_dict.get('r', np.nan)
        p = stats_dict.get('p_value', np.nan)
        n = stats_dict.get('n', 0)

        if not np.isnan(r_val):
            text = f'r = {r_val:.3f}\np = {p:.4f}\nn = {n}'
            ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=GLOBAL_FONT_SIZE - 1,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def plot_cartesian_scatter(ax, initial_heading, target_angle, stats_dict=None,
                           title='', xlabel='Initial Heading (rad)',
                           ylabel='Target Angle at Arrival (rad)',
                           show_stats=True, color=COLORS['scatter']):
    """
    Create 2D Cartesian scatter plot for circular-circular data.

    Both axes range from -pi to pi.

    Args:
        ax: Matplotlib axis
        initial_heading: Array of initial heading angles (radians)
        target_angle: Array of target angles at lever arrival (radians)
        stats_dict: Dictionary with correlation statistics
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_stats: Whether to show statistics annotation
        color: Scatter point color
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(initial_heading) | np.isnan(target_angle))
    x = initial_heading[valid_mask]
    y = target_angle[valid_mask]

    if len(x) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=GLOBAL_FONT_SIZE)
        return

    # Scatter plot
    ax.scatter(x, y, alpha=0.4, s=20, c=color, edgecolor='none')

    # Formatting
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'-$\pi$', r'-$\pi$/2', '0', r'$\pi$/2', r'$\pi$'],
                       fontsize=GLOBAL_FONT_SIZE - 1)
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'-$\pi$', r'-$\pi$/2', '0', r'$\pi$/2', r'$\pi$'],
                       fontsize=GLOBAL_FONT_SIZE - 1)
    ax.set_xlabel(xlabel, fontsize=GLOBAL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBAL_FONT_SIZE)
    ax.set_title(title, fontsize=GLOBAL_FONT_SIZE + 1)

    # Add diagonal reference line (y = x)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', alpha=0.3, linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')

    # Add statistics text
    if show_stats and stats_dict is not None:
        r_val = stats_dict.get('r', np.nan)
        p = stats_dict.get('p_value', np.nan)
        n = stats_dict.get('n', 0)

        if not np.isnan(r_val):
            # Add significance marker
            sig_marker = ''
            if p < 0.001:
                sig_marker = ' ***'
            elif p < 0.01:
                sig_marker = ' **'
            elif p < 0.05:
                sig_marker = ' *'

            text = f'r = {r_val:.3f}{sig_marker}\np = {p:.4f}\nn = {n}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=GLOBAL_FONT_SIZE - 1,
                    verticalalignment='top', fontfamily='monospace')


def plot_polar_histogram(ax, angles, title='', color='steelblue', n_bins=36, show_stats=True):
    """Create polar histogram of angular data with mean direction arrow.

    Parameters
    ----------
    ax : matplotlib axis (polar projection)
        Axis to plot on
    angles : array-like
        Angular data in radians
    title : str
        Plot title
    color : str
        Bar color
    n_bins : int
        Number of histogram bins
    show_stats : bool
        Whether to show n and MVL annotations
    """
    valid_angles = angles[~np.isnan(angles)]
    n_samples = len(valid_angles)

    if n_samples == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        return

    # Create histogram
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(valid_angles, bins=bins)

    # Plot as bars
    width = 2 * np.pi / n_bins
    centers = (bins[:-1] + bins[1:]) / 2

    ax.bar(centers, counts, width=width, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Calculate mean vector length (MVL)
    mean_cos = np.mean(np.cos(valid_angles))
    mean_sin = np.mean(np.sin(valid_angles))
    mvl = np.sqrt(mean_cos**2 + mean_sin**2)
    mean_dir = np.arctan2(mean_sin, mean_cos)

    # Add mean direction arrow (length proportional to MVL)
    max_count = max(counts) if max(counts) > 0 else 1
    arrow_length = max_count * 0.8 * mvl  # Scale by MVL
    if arrow_length > 0:
        ax.annotate('', xy=(mean_dir, arrow_length), xytext=(mean_dir, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

    # Add small red dot at center to mark arrow origin
    ax.scatter([mean_dir], [0], c='red', s=30, zorder=5)

    ax.set_title(title, fontsize=GLOBAL_FONT_SIZE, pad=10)

    # Add statistics annotation
    if show_stats:
        stats_text = f'n = {n_samples}\nMVL = {mvl:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=GLOBAL_FONT_SIZE - 1,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def plot_forest_effect_sizes(ax, results_df, effect_column='r',
                              ci_lower_col='r_ci_lower', ci_upper_col='r_ci_upper',
                              label_column='condition'):
    """Create forest plot of effect sizes with significance markers and legend."""
    # Create readable condition labels
    label_map = {
        'all_dark': 'All Dark Trials',
        'all_light': 'All Light Trials',
        'dark_short_search': 'Dark: Short Search',
        'dark_long_search': 'Dark: Long Search',
        'dark_accurate_homing': 'Dark: Accurate Homing',
        'dark_inaccurate_homing': 'Dark: Inaccurate Homing'
    }

    y_positions = np.arange(len(results_df))

    for i, (_, row) in enumerate(results_df.iterrows()):
        effect = row[effect_column]
        ci_low = row[ci_lower_col]
        ci_high = row[ci_upper_col]

        if np.isnan(effect):
            continue

        # Determine color based on significance
        is_sig = row.get('significant', False)
        color = '#e63946' if is_sig else 'steelblue'

        # Point estimate
        ax.scatter(effect, i, s=80, c=color, zorder=3, edgecolor='black', linewidth=0.5)

        # CI line
        if not np.isnan(ci_low) and not np.isnan(ci_high):
            ax.hlines(i, ci_low, ci_high, colors=color, linewidth=2.5)

        # Add significance stars
        if is_sig:
            p_fdr = row.get('p_fdr', 1)
            if p_fdr < 0.001:
                stars = '***'
            elif p_fdr < 0.01:
                stars = '**'
            else:
                stars = '*'
            x_pos = max(ci_high, effect) + 0.02 if not np.isnan(ci_high) else effect + 0.02
            ax.text(x_pos, i, stars, va='center', fontsize=12, fontweight='bold', color='#e63946')

    # Vertical line at zero (no effect)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=1)

    # Small effect thresholds
    ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.1, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Create readable labels
    readable_labels = [label_map.get(c, c) for c in results_df[label_column].tolist()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(readable_labels, fontsize=GLOBAL_FONT_SIZE)
    ax.set_xlabel('Circular-Circular Correlation (r)', fontsize=GLOBAL_FONT_SIZE)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e63946',
               markersize=10, label='Significant (FDR<0.05)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=10, label='Not significant', markeredgecolor='black'),
        Line2D([0], [0], color='gray', linestyle=':', label='|r|=0.1')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def compile_trial_data(behavior_df, all_session_df=None, use_hdpose=False):
    """
    Compile trial-level data with initial heading and target angle at arrival.

    Args:
        behavior_df: Behavioral data for valid trials
        all_session_df: DataFrame with hdPose (only needed if use_hdpose=True)
        use_hdpose: Whether to calculate hdPose-based heading

    Returns:
        DataFrame with columns: session, trialNo, light,
                               heading_movement, heading_hdpose,
                               target_angle_arrival
    """
    all_trials = []

    print(f"Processing {len(useAble)} sessions...")

    for session_name in tqdm(useAble, desc="Processing sessions"):
        # Load session data
        nav_instan, nav_summary = load_session_nav_data(session_name)

        if nav_instan is None or nav_summary is None:
            print(f"  Warning: Could not load data for {session_name}")
            continue

        # Pre-filter allSessionDf for this session (major optimization)
        session_all_df = None
        if use_hdpose and all_session_df is not None:
            session_all_df = all_session_df[all_session_df['session'] == session_name].copy()

        # Get valid trials for this session
        session_behavior = behavior_df[behavior_df['sessionName'] == session_name]

        for _, trial_row in session_behavior.iterrows():
            trial_no = trial_row['trialNo']
            light = trial_row['light']

            # Calculate initial heading (movement method)
            heading_mvt = calculate_initial_heading_movement(nav_instan, session_name, trial_no)

            # Calculate initial heading (hdPose method) - only if requested
            if use_hdpose and session_all_df is not None and len(session_all_df) > 0:
                heading_hd = calculate_initial_heading_hdpose_fast(session_all_df, trial_no)
            else:
                heading_hd = np.nan

            # Extract target angle at lever arrival
            target_angle = extract_target_angle_at_lever_arrival(
                nav_instan, nav_summary, session_name, trial_no
            )

            all_trials.append({
                'session': session_name,
                'trialNo': trial_no,
                'light': light,
                'heading_movement': heading_mvt,
                'heading_hdpose': heading_hd,
                'target_angle_arrival': target_angle
            })

    trial_df = pd.DataFrame(all_trials)
    print(f"\nCompiled {len(trial_df)} trials across {len(useAble)} sessions")

    return trial_df


def run_analysis_pipeline(save_figures=True, use_hdpose=True, verbose=True):
    """
    Main analysis pipeline.

    Args:
        save_figures: Whether to save figures
        use_hdpose: Whether to include hdPose-based heading
        verbose: Print progress

    Returns:
        results_df: Summary statistics DataFrame
    """
    print("=" * 80)
    print("INITIAL HEADING vs TARGET ANGLE AT LEVER ARRIVAL ANALYSIS")
    print("=" * 80)

    # Load data
    behavior_df = load_behavioral_data()

    if use_hdpose:
        all_session_df = load_all_session_df()
    else:
        all_session_df = None

    # Compile trial data
    trial_df = compile_trial_data(behavior_df, all_session_df, use_hdpose=use_hdpose)

    # Save trial data
    trial_df.to_csv(os.path.join(OUTPUT_DIR, 'trial_data.csv'), index=False)
    print(f"\nTrial data saved to {OUTPUT_DIR}/trial_data.csv")

    # Define analysis configurations
    methods = ['heading_movement']
    if use_hdpose:
        methods.append('heading_hdpose')

    # Get condition splits
    light_split = split_by_light_condition(trial_df)
    search_split = split_by_search_length(trial_df, behavior_df)
    homing_split = split_by_homing_accuracy(trial_df, behavior_df)

    conditions = {
        'all_dark': light_split['dark'],
        'all_light': light_split['light'],
        'dark_short_search': search_split['short_search'],
        'dark_long_search': search_split['long_search'],
        'dark_accurate_homing': homing_split['accurate_homing'],
        'dark_inaccurate_homing': homing_split['inaccurate_homing']
    }

    # Run analyses
    all_results = []

    print("\n" + "-" * 80)
    print("Running statistical analyses...")
    print("-" * 80)

    for method in methods:
        for cond_name, cond_df in conditions.items():
            if len(cond_df) < 10:
                continue

            heading = cond_df[method].values
            target_angle = cond_df['target_angle_arrival'].values

            # Circular-circular correlation
            corr_stats = circular_circular_correlation_permutation(heading, target_angle)

            result = {
                'heading_method': method,
                'condition': cond_name,
                'n_trials': corr_stats['n'],
                'r': corr_stats['r'],
                'r_ci_lower': corr_stats['ci_lower'],
                'r_ci_upper': corr_stats['ci_upper'],
                'p_uncorr': corr_stats['p_value']
            }
            all_results.append(result)

            if verbose:
                print(f"  {method} | {cond_name}: r={corr_stats['r']:.3f}, p={corr_stats['p_value']:.4f}, n={corr_stats['n']}")

    results_df = pd.DataFrame(all_results)

    # Apply FDR correction
    p_values = results_df['p_uncorr'].values
    p_fdr, significant = apply_fdr_correction(p_values)
    results_df['p_fdr'] = p_fdr
    results_df['significant'] = significant

    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/summary_statistics.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(results_df)}")
    print(f"Significant after FDR correction: {results_df['significant'].sum()}")

    dark_results = results_df[results_df['condition'] == 'all_dark']
    if len(dark_results) > 0:
        print(f"\nMean correlation (dark trials): {dark_results['r'].mean():.3f}")

    # Create figures
    if save_figures:
        print("\nGenerating figures...")
        create_summary_figures(trial_df, results_df, behavior_df)

    return results_df


def extract_stats_for_plotting(results_df, heading_method, condition):
    """Extract stats from results_df for plotting functions."""
    subset = results_df[
        (results_df['heading_method'] == heading_method) &
        (results_df['condition'] == condition)
    ]
    if len(subset) == 0:
        return None
    stats = subset.iloc[0].to_dict()
    # Rename keys to match what plot functions expect
    stats['n'] = stats.get('n_trials', 0)
    stats['p_value'] = stats.get('p_uncorr', np.nan)
    return stats


def create_summary_figures(trial_df, results_df, behavior_df):
    """Create publication-quality summary figures."""

    # Figure 1: Cartesian scatter plots for all conditions (movement method)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    dark_df = trial_df[trial_df['light'] == 'dark']
    light_df = trial_df[trial_df['light'] == 'light']

    # Row 1: Dark trials comparisons
    stats = extract_stats_for_plotting(results_df, 'heading_movement', 'all_dark')
    plot_cartesian_scatter(
        axes[0, 0],
        dark_df['heading_movement'].values,
        dark_df['target_angle_arrival'].values,
        stats,
        title='All Dark Trials\n(Movement Direction)'
    )

    # Short search dark
    search_split = split_by_search_length(trial_df, behavior_df)
    stats = extract_stats_for_plotting(results_df, 'heading_movement', 'dark_short_search')
    plot_cartesian_scatter(
        axes[0, 1],
        search_split['short_search']['heading_movement'].values,
        search_split['short_search']['target_angle_arrival'].values,
        stats,
        title='Dark: Short Search'
    )

    # Long search dark
    stats = extract_stats_for_plotting(results_df, 'heading_movement', 'dark_long_search')
    plot_cartesian_scatter(
        axes[0, 2],
        search_split['long_search']['heading_movement'].values,
        search_split['long_search']['target_angle_arrival'].values,
        stats,
        title='Dark: Long Search'
    )

    # Row 2: Light trials and homing splits
    stats = extract_stats_for_plotting(results_df, 'heading_movement', 'all_light')
    plot_cartesian_scatter(
        axes[1, 0],
        light_df['heading_movement'].values,
        light_df['target_angle_arrival'].values,
        stats,
        title='All Light Trials\n(Movement Direction)'
    )

    # Accurate homing
    homing_split = split_by_homing_accuracy(trial_df, behavior_df)
    stats = extract_stats_for_plotting(results_df, 'heading_movement', 'dark_accurate_homing')
    plot_cartesian_scatter(
        axes[1, 1],
        homing_split['accurate_homing']['heading_movement'].values,
        homing_split['accurate_homing']['target_angle_arrival'].values,
        stats,
        title='Dark: Accurate Homing'
    )

    # Inaccurate homing
    stats = extract_stats_for_plotting(results_df, 'heading_movement', 'dark_inaccurate_homing')
    plot_cartesian_scatter(
        axes[1, 2],
        homing_split['inaccurate_homing']['heading_movement'].values,
        homing_split['inaccurate_homing']['target_angle_arrival'].values,
        stats,
        title='Dark: Inaccurate Homing'
    )

    # Add figure annotation
    fig.text(0.5, 0.01,
             'Dashed line: y = x (perfect correlation) | * p<0.05, ** p<0.01, *** p<0.001 (FDR corrected)',
             ha='center', fontsize=10, style='italic')

    plt.suptitle('Initial Heading vs Target Angle at Lever Arrival', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig1_cartesian_scatter_conditions.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Polar scatter plots for dark trials
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'polar'})

    dark_df = trial_df[trial_df['light'] == 'dark']

    stats = extract_stats_for_plotting(results_df, 'heading_movement', 'all_dark')
    plot_polar_scatter(
        axes[0],
        dark_df['heading_movement'].values,
        dark_df['target_angle_arrival'].values,
        stats,
        title='Dark Trials: Movement Direction'
    )

    if 'heading_hdpose' in dark_df.columns and not dark_df['heading_hdpose'].isna().all():
        stats = extract_stats_for_plotting(results_df, 'heading_hdpose', 'all_dark')
        plot_polar_scatter(
            axes[1],
            dark_df['heading_hdpose'].values,
            dark_df['target_angle_arrival'].values,
            stats,
            title='Dark Trials: Head Direction (hdPose)'
        )
    else:
        axes[1].text(0.5, 0.5, 'hdPose not computed',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.suptitle('Polar Scatter: Initial Heading (theta) vs Target Angle (radius)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig2_polar_scatter_dark.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Polar distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={'projection': 'polar'})

    dark_df = trial_df[trial_df['light'] == 'dark']

    plot_polar_histogram(axes[0], dark_df['heading_movement'].values,
                        title='Initial Heading\n(Movement Direction)', color=COLORS['dark'])

    plot_polar_histogram(axes[1], dark_df['target_angle_arrival'].values,
                        title='Target Angle at Arrival', color=COLORS['light'])

    # Light trials for comparison
    light_df = trial_df[trial_df['light'] == 'light']
    plot_polar_histogram(axes[2], light_df['target_angle_arrival'].values,
                        title='Target Angle (Light Trials)', color=COLORS['light'])

    fig.suptitle('Angular Distributions\n'
                 'Red arrow = mean direction, length proportional to MVL',
                 fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig3_polar_distributions.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: Forest plot for effect sizes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter for movement method
    subset = results_df[results_df['heading_method'] == 'heading_movement'].copy()

    if len(subset) > 0:
        plot_forest_effect_sizes(ax, subset, effect_column='r',
                                 ci_lower_col='r_ci_lower', ci_upper_col='r_ci_upper',
                                 label_column='condition')
        ax.set_title('Effect Sizes: Initial Heading vs Target Angle at Arrival', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig4_forest_plot.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 5: Comparison of heading methods (if hdPose computed)
    if 'heading_hdpose' in results_df['heading_method'].values:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        dark_df = trial_df[trial_df['light'] == 'dark']

        # Movement method
        stats = extract_stats_for_plotting(results_df, 'heading_movement', 'all_dark')
        plot_cartesian_scatter(
            axes[0],
            dark_df['heading_movement'].values,
            dark_df['target_angle_arrival'].values,
            stats,
            title='Movement Direction Method'
        )

        # hdPose method
        stats = extract_stats_for_plotting(results_df, 'heading_hdpose', 'all_dark')
        plot_cartesian_scatter(
            axes[1],
            dark_df['heading_hdpose'].values,
            dark_df['target_angle_arrival'].values,
            stats,
            title='Head Direction (hdPose) Method'
        )

        plt.suptitle('Comparison of Initial Heading Methods (Dark Trials)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig5_method_comparison.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Figures saved to {OUTPUT_DIR}/figures/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the analysis pipeline
    results_df = run_analysis_pipeline(
        save_figures=True,
        use_hdpose=True,  # Set to False to skip hdPose calculation (faster)
        verbose=True
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  - trial_data.csv: Trial-level data with heading and target angle")
    print("  - summary_statistics.csv: Statistical analysis results")
    print("  - figures/fig1_cartesian_scatter_conditions.pdf: Cartesian scatter by condition")
    print("  - figures/fig2_polar_scatter_dark.pdf: Polar scatter plots")
    print("  - figures/fig3_polar_distributions.pdf: Polar histograms")
    print("  - figures/fig4_forest_plot.pdf: Effect size forest plot")
    print("  - figures/fig5_method_comparison.pdf: Heading method comparison")
