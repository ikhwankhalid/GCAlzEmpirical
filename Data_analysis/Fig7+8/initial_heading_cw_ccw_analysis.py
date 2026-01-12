"""
Analysis: Initial Heading vs Clockwise/Counterclockwise Movement Around Lever

Research Question:
    Does the animal's initial heading (when leaving home base) determine whether
    it runs clockwise or counterclockwise around the lever during a trial?

Variables:
    Independent: Initial heading (2 methods)
        1. Movement direction from first 0.5s of search
        2. Smoothed head direction (hdPose) from first 0.5s

    Dependent: CW/CCW metric from cumSumDiffAngleAroundTarget (3 metrics)
        1. Final value at trial end
        2. Value at lever arrival (end of search)
        3. Value at end of at-lever period
        Sign convention: Negative = CCW, Positive = CW

Conditions analyzed separately:
    - Light vs Dark
    - Short vs Long search (median split on dark)
    - Accurate vs Inaccurate homing (median split on dark)

Output:
    - Summary statistics CSV with effect sizes
    - Publication-quality figures
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
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from astropy.stats import circcorrcoef
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
    'cw': '#e63946',
    'ccw': '#457b9d',
    'regression': '#ff9e00'
}

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_DATA_PATH, 'results', 'initial_heading_cwccw')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

# Output directory for wrapped analysis
OUTPUT_DIR_WRAPPED = os.path.join(PROJECT_DATA_PATH, 'results', 'initial_heading_cwccw_wrapped')
os.makedirs(OUTPUT_DIR_WRAPPED, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR_WRAPPED, 'figures'), exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def wrap_to_pi(angle):
    """
    Wrap angle to [-pi, pi] range.

    Args:
        angle: Angle in radians (can be any value)

    Returns:
        Angle wrapped to [-pi, pi] range, or np.nan if input is nan
    """
    if np.isnan(angle):
        return np.nan
    return math.remainder(angle, 2 * np.pi)

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


def calculate_initial_heading_hdpose(all_session_df, session_name, trial_no,
                                      time_window=TIME_WINDOW, smooth_sigma=SMOOTH_SIGMA):
    """
    Calculate initial heading from smoothed head direction (Method 2).

    Args:
        all_session_df: DataFrame with hdPose data
        session_name: Session identifier
        trial_no: Trial number
        time_window: Time window in seconds
        smooth_sigma: Gaussian smoothing sigma

    Returns:
        Initial heading in radians [-pi, pi], or np.nan if insufficient data
    """
    # Filter for searchToLeverPath_dark for this trial
    mask = (
        (all_session_df['session'] == session_name) &
        (all_session_df['trialNo'] == trial_no) &
        (all_session_df['condition'].str.contains('searchToLeverPath'))
    )
    trial_data = all_session_df[mask].copy()

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
# CW/CCW METRIC EXTRACTION
# ============================================================================

def extract_cwccw_trial_end(nav_instan, session_name, trial_no):
    """
    Extract CW/CCW metric: Final cumSumDiffAngleAroundTarget at trial end.

    Returns:
        Cumulative angular displacement (radians). Negative=CCW, Positive=CW.
    """
    # Filter for 'all' path type (complete trial)
    mask = (
        (nav_instan['session'] == session_name) &
        (nav_instan['trialNo'] == trial_no) &
        (nav_instan['name'].str.endswith('_all'))
    )
    trial_data = nav_instan[mask]

    if len(trial_data) == 0:
        return np.nan

    # Get last row (end of trial)
    if 'cumSumDiffAngleAroundTarget' not in trial_data.columns:
        return np.nan

    # Sort by time and get final value
    if 'iTime' in trial_data.columns:
        trial_data = trial_data.sort_values('iTime')
    else:
        trial_data = trial_data.sort_values('timeRes')

    final_value = trial_data['cumSumDiffAngleAroundTarget'].iloc[-1]

    return final_value


def extract_cwccw_lever_arrival(nav_instan, nav_summary, session_name, trial_no):
    """
    Extract CW/CCW metric: cumSumDiffAngleAroundTarget at lever arrival.

    Returns:
        Cumulative angular displacement at end of search phase.
    """
    # Get end time of searchToLeverPath from nav_summary
    mask_summary = (
        (nav_summary['session'] == session_name) &
        (nav_summary['trialNo'] == trial_no) &
        (nav_summary['type'] == 'searchToLeverPath')
    )
    search_summary = nav_summary[mask_summary]

    if len(search_summary) == 0:
        return np.nan

    end_time = search_summary['endTimeRes'].iloc[0]

    # Get cumSumDiffAngleAroundTarget from 'all' path at that time
    mask_instan = (
        (nav_instan['session'] == session_name) &
        (nav_instan['trialNo'] == trial_no) &
        (nav_instan['name'].str.endswith('_all'))
    )
    trial_data = nav_instan[mask_instan]

    if len(trial_data) == 0 or 'cumSumDiffAngleAroundTarget' not in trial_data.columns:
        return np.nan

    # Find row closest to end_time
    trial_data = trial_data.copy()
    trial_data['time_diff'] = np.abs(trial_data['timeRes'] - end_time)
    closest_row = trial_data.loc[trial_data['time_diff'].idxmin()]

    return closest_row['cumSumDiffAngleAroundTarget']


def extract_cwccw_at_lever(nav_instan, nav_summary, session_name, trial_no):
    """
    Extract CW/CCW metric: cumSumDiffAngleAroundTarget at end of at-lever period.

    Returns:
        Cumulative angular displacement at end of at-lever phase.
    """
    # Get end time of atLever from nav_summary
    mask_summary = (
        (nav_summary['session'] == session_name) &
        (nav_summary['trialNo'] == trial_no) &
        (nav_summary['type'] == 'atLever')
    )
    lever_summary = nav_summary[mask_summary]

    if len(lever_summary) == 0:
        return np.nan

    end_time = lever_summary['endTimeRes'].iloc[0]

    # Get cumSumDiffAngleAroundTarget from 'all' path at that time
    mask_instan = (
        (nav_instan['session'] == session_name) &
        (nav_instan['trialNo'] == trial_no) &
        (nav_instan['name'].str.endswith('_all'))
    )
    trial_data = nav_instan[mask_instan]

    if len(trial_data) == 0 or 'cumSumDiffAngleAroundTarget' not in trial_data.columns:
        return np.nan

    # Find row closest to end_time
    trial_data = trial_data.copy()
    trial_data['time_diff'] = np.abs(trial_data['timeRes'] - end_time)
    closest_row = trial_data.loc[trial_data['time_diff'].idxmin()]

    return closest_row['cumSumDiffAngleAroundTarget']


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

def circular_linear_correlation_permutation(initial_heading, cwccw_metric,
                                             n_permutations=N_PERMUTATIONS,
                                             n_bootstrap=N_BOOTSTRAP):
    """
    Calculate circular-linear correlation with permutation test and bootstrap CI.

    Args:
        initial_heading: Array of initial heading angles (radians)
        cwccw_metric: Array of CW/CCW metric values
        n_permutations: Permutation iterations for p-value
        n_bootstrap: Bootstrap iterations for CI

    Returns:
        Dictionary with r, p_value, ci_lower, ci_upper, slope, n
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(initial_heading) | np.isnan(cwccw_metric))
    x = initial_heading[valid_mask]
    y = cwccw_metric[valid_mask]

    n = len(x)
    if n < 10:
        return {
            'r': np.nan, 'p_value': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan,
            'slope': np.nan, 'n': n
        }

    # Calculate observed correlation
    real_r = circcorrcoef(x, y)

    # Linear regression for slope (using sin/cos representation)
    X = np.column_stack([np.sin(x), np.cos(x)])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    slope = np.sqrt(model.params[1]**2 + model.params[2]**2)  # Combined effect

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
        'slope': slope,
        'n': n
    }


def logistic_regression_cwccw(initial_heading, cwccw_metric, threshold=0.0):
    """
    Logistic regression for binary CW/CCW prediction from initial heading.

    Args:
        initial_heading: Array of initial heading angles
        cwccw_metric: Array of CW/CCW metric values
        threshold: Value to binarize CW/CCW (default 0)

    Returns:
        Dictionary with odds_ratio, CI, p_value, accuracy, auc, n_cw, n_ccw
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(initial_heading) | np.isnan(cwccw_metric))
    x = initial_heading[valid_mask]
    y_continuous = cwccw_metric[valid_mask]

    # Binarize: CW (>threshold) = 1, CCW (<=threshold) = 0
    y = (y_continuous > threshold).astype(int)

    n_cw = np.sum(y == 1)
    n_ccw = np.sum(y == 0)

    if n_cw < 5 or n_ccw < 5:
        return {
            'odds_ratio': np.nan, 'or_ci_lower': np.nan, 'or_ci_upper': np.nan,
            'p_value': np.nan, 'accuracy': np.nan, 'auc': np.nan,
            'n_cw': n_cw, 'n_ccw': n_ccw
        }

    # Create features from circular variable
    X = np.column_stack([np.sin(x), np.cos(x)])

    try:
        # Fit with statsmodels for CI
        X_sm = sm.add_constant(X)
        logit_model = sm.Logit(y, X_sm)
        result = logit_model.fit(disp=0)

        # Combined effect (magnitude of circular coefficient)
        coef_sin = result.params[1]
        coef_cos = result.params[2]
        combined_coef = np.sqrt(coef_sin**2 + coef_cos**2)
        odds_ratio = np.exp(combined_coef)

        # Get p-value (use LR test or coefficient p-values)
        p_value = min(result.pvalues[1], result.pvalues[2])

        # CI via bootstrap
        n = len(x)
        boot_or = []
        for _ in range(500):
            idx = np.random.choice(n, size=n, replace=True)
            try:
                boot_result = sm.Logit(y[idx], X_sm[idx]).fit(disp=0)
                boot_coef = np.sqrt(boot_result.params[1]**2 + boot_result.params[2]**2)
                boot_or.append(np.exp(boot_coef))
            except:
                continue

        if len(boot_or) > 100:
            or_ci_lower, or_ci_upper = np.percentile(boot_or, [2.5, 97.5])
        else:
            or_ci_lower, or_ci_upper = np.nan, np.nan

        # Calculate accuracy and AUC using sklearn
        clf = LogisticRegression(penalty=None, max_iter=1000)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]

        accuracy = np.mean(y_pred == y)
        auc = roc_auc_score(y, y_prob)

    except Exception as e:
        return {
            'odds_ratio': np.nan, 'or_ci_lower': np.nan, 'or_ci_upper': np.nan,
            'p_value': np.nan, 'accuracy': np.nan, 'auc': np.nan,
            'n_cw': n_cw, 'n_ccw': n_ccw
        }

    return {
        'odds_ratio': odds_ratio,
        'or_ci_lower': or_ci_lower,
        'or_ci_upper': or_ci_upper,
        'p_value': p_value,
        'accuracy': accuracy,
        'auc': auc,
        'n_cw': n_cw,
        'n_ccw': n_ccw
    }


def apply_fdr_correction(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    valid_mask = ~np.isnan(p_values)
    p_valid = p_values[valid_mask]

    if len(p_valid) == 0:
        return p_values, np.zeros_like(p_values, dtype=bool)

    significant, p_corrected_valid, _, _ = multipletests(p_valid, alpha=alpha, method='fdr_bh')

    p_corrected = np.full_like(p_values, np.nan)
    p_corrected[valid_mask] = p_corrected_valid

    sig_array = np.zeros_like(p_values, dtype=bool)
    sig_array[valid_mask] = significant

    return p_corrected, sig_array


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_scatter_with_regression(ax, initial_heading, cwccw_metric, stats_dict,
                                  title='', xlabel='Initial Heading (rad)',
                                  ylabel='Angular Displacement (rad)',
                                  show_stats=True):
    """Create scatter plot with regression line and statistics annotation."""
    # Remove NaN values
    valid_mask = ~(np.isnan(initial_heading) | np.isnan(cwccw_metric))
    x = initial_heading[valid_mask]
    y = cwccw_metric[valid_mask]

    if len(x) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=GLOBAL_FONT_SIZE)
        return

    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=20, c='steelblue', edgecolor='none')

    # Regression line (sin/cos model)
    X = np.column_stack([np.sin(x), np.cos(x)])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    x_line = np.linspace(-np.pi, np.pi, 100)
    X_line = np.column_stack([np.ones(100), np.sin(x_line), np.cos(x_line)])
    y_line = model.predict(X_line)
    ax.plot(x_line, y_line, color=COLORS['regression'], lw=2, linestyle='-')

    # Formatting
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'], fontsize=GLOBAL_FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=GLOBAL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBAL_FONT_SIZE)
    ax.set_title(title, fontsize=GLOBAL_FONT_SIZE + 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add statistics text
    if show_stats and stats_dict is not None:
        r = stats_dict.get('r', np.nan)
        p = stats_dict.get('p_value', np.nan)
        n = stats_dict.get('n', 0)

        if not np.isnan(r):
            text = f'r = {r:.3f}\np = {p:.4f}\nn = {n}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=GLOBAL_FONT_SIZE - 1,
                    verticalalignment='top', fontfamily='monospace')


def plot_polar_histogram(ax, angles, title='', color='steelblue', n_bins=36, show_stats=True):
    """Create polar histogram of angular data with mean direction arrow and statistics.

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

    # Calculate mean vector length (MVL) - correct formula
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
        # Position in upper left of polar plot
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
    ax.set_xlabel('Circular-Linear Correlation (r)', fontsize=GLOBAL_FONT_SIZE)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e63946',
               markersize=10, label='Significant (FDR<0.05)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=10, label='Not significant', markeredgecolor='black'),
        Line2D([0], [0], color='gray', linestyle=':', label='Small effect (|r|=0.1)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def compile_trial_data(behavior_df, all_session_df=None, use_hdpose=False):
    """
    Compile trial-level data with initial heading and CW/CCW metrics.

    Args:
        behavior_df: Behavioral data for valid trials
        all_session_df: DataFrame with hdPose (only needed if use_hdpose=True)
        use_hdpose: Whether to calculate hdPose-based heading

    Returns:
        DataFrame with columns: session, trialNo, light,
                               heading_movement, heading_hdpose,
                               cwccw_trial_end, cwccw_lever_arrival, cwccw_at_lever
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

            # Extract CW/CCW metrics
            cwccw_end = extract_cwccw_trial_end(nav_instan, session_name, trial_no)
            cwccw_lever = extract_cwccw_lever_arrival(nav_instan, nav_summary, session_name, trial_no)
            cwccw_atlever = extract_cwccw_at_lever(nav_instan, nav_summary, session_name, trial_no)

            all_trials.append({
                'session': session_name,
                'trialNo': trial_no,
                'light': light,
                'heading_movement': heading_mvt,
                'heading_hdpose': heading_hd,
                'cwccw_trial_end': cwccw_end,
                'cwccw_lever_arrival': cwccw_lever,
                'cwccw_at_lever': cwccw_atlever,
                # Wrapped versions (to [-pi, pi] range)
                'cwccw_trial_end_wrapped': wrap_to_pi(cwccw_end),
                'cwccw_lever_arrival_wrapped': wrap_to_pi(cwccw_lever),
                'cwccw_at_lever_wrapped': wrap_to_pi(cwccw_atlever)
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
    print("INITIAL HEADING vs CW/CCW MOVEMENT ANALYSIS")
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
    metrics = ['cwccw_trial_end', 'cwccw_lever_arrival', 'cwccw_at_lever']
    metrics_wrapped = ['cwccw_trial_end_wrapped', 'cwccw_lever_arrival_wrapped', 'cwccw_at_lever_wrapped']
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

    for metric in metrics:
        for method in methods:
            for cond_name, cond_df in conditions.items():
                if len(cond_df) < 10:
                    continue

                heading = cond_df[method].values
                cwccw = cond_df[metric].values

                # Circular-linear correlation
                corr_stats = circular_linear_correlation_permutation(heading, cwccw)

                # Logistic regression
                logit_stats = logistic_regression_cwccw(heading, cwccw)

                result = {
                    'metric': metric,
                    'heading_method': method,
                    'condition': cond_name,
                    'n_trials': corr_stats['n'],
                    'r': corr_stats['r'],
                    'r_ci_lower': corr_stats['ci_lower'],
                    'r_ci_upper': corr_stats['ci_upper'],
                    'p_uncorr': corr_stats['p_value'],
                    'odds_ratio': logit_stats['odds_ratio'],
                    'or_ci_lower': logit_stats['or_ci_lower'],
                    'or_ci_upper': logit_stats['or_ci_upper'],
                    'or_p': logit_stats['p_value'],
                    'auc': logit_stats['auc'],
                    'n_cw': logit_stats['n_cw'],
                    'n_ccw': logit_stats['n_ccw']
                }
                all_results.append(result)

                if verbose:
                    print(f"  {metric} | {method} | {cond_name}: r={corr_stats['r']:.3f}, p={corr_stats['p_value']:.4f}")

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
    print("SUMMARY (UNWRAPPED)")
    print("=" * 80)
    print(f"Total tests: {len(results_df)}")
    print(f"Significant after FDR correction: {results_df['significant'].sum()}")
    print(f"\nMean correlation (dark trials): {results_df[results_df['condition'] == 'all_dark']['r'].mean():.3f}")

    # Create figures
    if save_figures:
        print("\nGenerating figures (unwrapped)...")
        create_summary_figures(trial_df, results_df, behavior_df)

    # =========================================================================
    # WRAPPED ANALYSIS (angles wrapped to [-pi, pi])
    # =========================================================================
    print("\n" + "=" * 80)
    print("WRAPPED ANALYSIS (angles wrapped to [-pi, +pi])")
    print("=" * 80)

    # Save wrapped trial data
    trial_df.to_csv(os.path.join(OUTPUT_DIR_WRAPPED, 'trial_data_wrapped.csv'), index=False)
    print(f"\nWrapped trial data saved to {OUTPUT_DIR_WRAPPED}/trial_data_wrapped.csv")

    # Run wrapped analyses
    all_results_wrapped = []

    print("\n" + "-" * 80)
    print("Running statistical analyses (wrapped)...")
    print("-" * 80)

    for metric in metrics_wrapped:
        for method in methods:
            for cond_name, cond_df in conditions.items():
                if len(cond_df) < 10:
                    continue

                heading = cond_df[method].values
                cwccw = cond_df[metric].values

                # Circular-linear correlation
                corr_stats = circular_linear_correlation_permutation(heading, cwccw)

                # Logistic regression
                logit_stats = logistic_regression_cwccw(heading, cwccw)

                result = {
                    'metric': metric,
                    'heading_method': method,
                    'condition': cond_name,
                    'n_trials': corr_stats['n'],
                    'r': corr_stats['r'],
                    'r_ci_lower': corr_stats['ci_lower'],
                    'r_ci_upper': corr_stats['ci_upper'],
                    'p_uncorr': corr_stats['p_value'],
                    'odds_ratio': logit_stats['odds_ratio'],
                    'or_ci_lower': logit_stats['or_ci_lower'],
                    'or_ci_upper': logit_stats['or_ci_upper'],
                    'or_p': logit_stats['p_value'],
                    'auc': logit_stats['auc'],
                    'n_cw': logit_stats['n_cw'],
                    'n_ccw': logit_stats['n_ccw']
                }
                all_results_wrapped.append(result)

                if verbose:
                    print(f"  {metric} | {method} | {cond_name}: r={corr_stats['r']:.3f}, p={corr_stats['p_value']:.4f}")

    results_df_wrapped = pd.DataFrame(all_results_wrapped)

    # Apply FDR correction
    p_values_wrapped = results_df_wrapped['p_uncorr'].values
    p_fdr_wrapped, significant_wrapped = apply_fdr_correction(p_values_wrapped)
    results_df_wrapped['p_fdr'] = p_fdr_wrapped
    results_df_wrapped['significant'] = significant_wrapped

    # Save wrapped results
    results_df_wrapped.to_csv(os.path.join(OUTPUT_DIR_WRAPPED, 'summary_statistics_wrapped.csv'), index=False)
    print(f"\nWrapped results saved to {OUTPUT_DIR_WRAPPED}/summary_statistics_wrapped.csv")

    # Print wrapped summary
    print("\n" + "=" * 80)
    print("SUMMARY (WRAPPED)")
    print("=" * 80)
    print(f"Total tests: {len(results_df_wrapped)}")
    print(f"Significant after FDR correction: {results_df_wrapped['significant'].sum()}")
    print(f"\nMean correlation (dark trials): {results_df_wrapped[results_df_wrapped['condition'] == 'all_dark']['r'].mean():.3f}")

    # Create wrapped figures
    if save_figures:
        print("\nGenerating figures (wrapped)...")
        create_summary_figures_wrapped(trial_df, results_df_wrapped, behavior_df)

    # =========================================================================
    # COMPARISON: UNWRAPPED vs WRAPPED
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: UNWRAPPED vs WRAPPED")
    print("=" * 80)

    # Compare key metrics for dark trials with movement method
    for metric, metric_wrapped in zip(metrics, metrics_wrapped):
        unwrapped_row = results_df[
            (results_df['metric'] == metric) &
            (results_df['heading_method'] == 'heading_movement') &
            (results_df['condition'] == 'all_dark')
        ]
        wrapped_row = results_df_wrapped[
            (results_df_wrapped['metric'] == metric_wrapped) &
            (results_df_wrapped['heading_method'] == 'heading_movement') &
            (results_df_wrapped['condition'] == 'all_dark')
        ]

        if len(unwrapped_row) > 0 and len(wrapped_row) > 0:
            r_unwrapped = unwrapped_row['r'].values[0]
            r_wrapped = wrapped_row['r'].values[0]
            p_unwrapped = unwrapped_row['p_uncorr'].values[0]
            p_wrapped = wrapped_row['p_uncorr'].values[0]

            # Determine if relationship is stronger/weaker
            if abs(r_wrapped) > abs(r_unwrapped):
                change = "STRONGER"
            elif abs(r_wrapped) < abs(r_unwrapped):
                change = "WEAKER"
            else:
                change = "SAME"

            metric_short = metric.replace('cwccw_', '')
            print(f"\n{metric_short}:")
            print(f"  Unwrapped: r = {r_unwrapped:.4f}, p = {p_unwrapped:.4f}")
            print(f"  Wrapped:   r = {r_wrapped:.4f}, p = {p_wrapped:.4f}")
            print(f"  Change:    {change} (delta_r = {r_wrapped - r_unwrapped:+.4f})")

    return results_df, results_df_wrapped


def extract_stats_for_plotting(results_df, metric, heading_method, condition):
    """Extract stats from results_df and rename keys for plotting functions."""
    subset = results_df[
        (results_df['metric'] == metric) &
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

    # Figure 1: Main scatter plots for dark trials
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    dark_df = trial_df[trial_df['light'] == 'dark']

    metrics = ['cwccw_trial_end', 'cwccw_lever_arrival', 'cwccw_at_lever']
    metric_labels = ['Trial End', 'Lever Arrival', 'At Lever']

    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Movement method
        stats = extract_stats_for_plotting(results_df, metric, 'heading_movement', 'all_dark')

        # Add significance marker to title
        sig_marker = ''
        if stats and stats.get('significant', False):
            p_fdr = stats.get('p_fdr', 1)
            if p_fdr < 0.001:
                sig_marker = ' ***'
            elif p_fdr < 0.01:
                sig_marker = ' **'
            elif p_fdr < 0.05:
                sig_marker = ' *'

        plot_scatter_with_regression(
            axes[0, col],
            dark_df['heading_movement'].values,
            dark_df[metric].values,
            stats,
            title=f'{label}{sig_marker}\n(Movement Direction)',
            ylabel='CW (+) / CCW (-) rad' if col == 0 else ''
        )

        # hdPose method (if available)
        if 'heading_hdpose' in dark_df.columns and not dark_df['heading_hdpose'].isna().all():
            stats = extract_stats_for_plotting(results_df, metric, 'heading_hdpose', 'all_dark')

            sig_marker = ''
            if stats and stats.get('significant', False):
                p_fdr = stats.get('p_fdr', 1)
                if p_fdr < 0.001:
                    sig_marker = ' ***'
                elif p_fdr < 0.01:
                    sig_marker = ' **'
                elif p_fdr < 0.05:
                    sig_marker = ' *'

            plot_scatter_with_regression(
                axes[1, col],
                dark_df['heading_hdpose'].values,
                dark_df[metric].values,
                stats,
                title=f'{label}{sig_marker}\n(Head Direction)',
                ylabel='CW (+) / CCW (-) rad' if col == 0 else ''
            )
        else:
            axes[1, col].text(0.5, 0.5, 'hdPose not computed',
                             ha='center', va='center', transform=axes[1, col].transAxes)
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])

    # Add figure legend/annotation
    fig.text(0.5, 0.01,
             'Orange line: circular-linear regression | CW/CCW: cumulative angular displacement around lever\n'
             '* p<0.05, ** p<0.01, *** p<0.001 (FDR corrected)',
             ha='center', fontsize=10, style='italic')

    plt.suptitle('Initial Heading vs CW/CCW Movement Around Lever (Dark Trials)', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig1_scatter_dark_trials.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Condition comparisons (forest plot)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter for movement method, trial_end metric
    subset = results_df[
        (results_df['heading_method'] == 'heading_movement') &
        (results_df['metric'] == 'cwccw_trial_end')
    ].copy()

    if len(subset) > 0:
        plot_forest_effect_sizes(ax, subset, effect_column='r',
                                 ci_lower_col='r_ci_lower', ci_upper_col='r_ci_upper',
                                 label_column='condition')
        ax.set_title('Effect Sizes: Initial Heading vs CW/CCW (Trial End)', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig2_forest_plot.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Polar distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={'projection': 'polar'})

    dark_df = trial_df[trial_df['light'] == 'dark']

    plot_polar_histogram(axes[0], dark_df['heading_movement'].values,
                        title='All Dark Trials\n(Combined)', color='steelblue')

    cw_mask = dark_df['cwccw_trial_end'] > 0
    plot_polar_histogram(axes[1], dark_df.loc[cw_mask, 'heading_movement'].values,
                        title='Clockwise (CW) Trials\n(+cumSumDiffAngle)', color=COLORS['cw'])

    ccw_mask = dark_df['cwccw_trial_end'] <= 0
    plot_polar_histogram(axes[2], dark_df.loc[ccw_mask, 'heading_movement'].values,
                        title='Counter-Clockwise (CCW) Trials\n(-cumSumDiffAngle)', color=COLORS['ccw'])

    # Main title with explanation
    fig.suptitle('Initial Heading Direction When Leaving Home\n'
                 '(Movement direction in first 0.5s of search, dark trials only)\n'
                 'Red arrow = mean direction, length ∝ MVL',
                 fontsize=12, y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig3_polar_distributions.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {OUTPUT_DIR}/figures/")


def create_summary_figures_wrapped(trial_df, results_df_wrapped, behavior_df):
    """Create publication-quality summary figures for WRAPPED analysis."""

    # Figure 1: Main scatter plots for dark trials (WRAPPED)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    dark_df = trial_df[trial_df['light'] == 'dark']

    metrics_wrapped = ['cwccw_trial_end_wrapped', 'cwccw_lever_arrival_wrapped', 'cwccw_at_lever_wrapped']
    metric_labels = ['Trial End', 'Lever Arrival', 'At Lever']

    for col, (metric, label) in enumerate(zip(metrics_wrapped, metric_labels)):
        # Movement method
        stats = extract_stats_for_plotting(results_df_wrapped, metric, 'heading_movement', 'all_dark')

        # Add significance marker to title
        sig_marker = ''
        if stats and stats.get('significant', False):
            p_fdr = stats.get('p_fdr', 1)
            if p_fdr < 0.001:
                sig_marker = ' ***'
            elif p_fdr < 0.01:
                sig_marker = ' **'
            elif p_fdr < 0.05:
                sig_marker = ' *'

        plot_scatter_with_regression(
            axes[0, col],
            dark_df['heading_movement'].values,
            dark_df[metric].values,
            stats,
            title=f'{label}{sig_marker}\n(Movement Direction)',
            ylabel='CW (+) / CCW (-) rad [wrapped ±π]' if col == 0 else ''
        )

        # hdPose method (if available)
        if 'heading_hdpose' in dark_df.columns and not dark_df['heading_hdpose'].isna().all():
            stats = extract_stats_for_plotting(results_df_wrapped, metric, 'heading_hdpose', 'all_dark')

            sig_marker = ''
            if stats and stats.get('significant', False):
                p_fdr = stats.get('p_fdr', 1)
                if p_fdr < 0.001:
                    sig_marker = ' ***'
                elif p_fdr < 0.01:
                    sig_marker = ' **'
                elif p_fdr < 0.05:
                    sig_marker = ' *'

            plot_scatter_with_regression(
                axes[1, col],
                dark_df['heading_hdpose'].values,
                dark_df[metric].values,
                stats,
                title=f'{label}{sig_marker}\n(Head Direction)',
                ylabel='CW (+) / CCW (-) rad [wrapped ±π]' if col == 0 else ''
            )
        else:
            axes[1, col].text(0.5, 0.5, 'hdPose not computed',
                             ha='center', va='center', transform=axes[1, col].transAxes)
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])

    # Add figure legend/annotation
    fig.text(0.5, 0.01,
             'Orange line: circular-linear regression | CW/CCW: angular displacement WRAPPED to [-π, +π]\n'
             '* p<0.05, ** p<0.01, *** p<0.001 (FDR corrected)',
             ha='center', fontsize=10, style='italic')

    plt.suptitle('Initial Heading vs CW/CCW Movement Around Lever (Dark Trials)\n[Wrapped to ±π]', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR_WRAPPED, 'figures', 'fig1_scatter_dark_trials_wrapped.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Condition comparisons (forest plot) - WRAPPED
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter for movement method, trial_end_wrapped metric
    subset = results_df_wrapped[
        (results_df_wrapped['heading_method'] == 'heading_movement') &
        (results_df_wrapped['metric'] == 'cwccw_trial_end_wrapped')
    ].copy()

    if len(subset) > 0:
        plot_forest_effect_sizes(ax, subset, effect_column='r',
                                 ci_lower_col='r_ci_lower', ci_upper_col='r_ci_upper',
                                 label_column='condition')
        ax.set_title('Effect Sizes: Initial Heading vs CW/CCW (Trial End)\n[Wrapped to ±π]', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_WRAPPED, 'figures', 'fig2_forest_plot_wrapped.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Polar distributions (WRAPPED)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={'projection': 'polar'})

    dark_df = trial_df[trial_df['light'] == 'dark']

    plot_polar_histogram(axes[0], dark_df['heading_movement'].values,
                        title='All Dark Trials\n(Combined)', color='steelblue')

    # Use wrapped metric for CW/CCW split
    cw_mask = dark_df['cwccw_trial_end_wrapped'] > 0
    plot_polar_histogram(axes[1], dark_df.loc[cw_mask, 'heading_movement'].values,
                        title='Clockwise (CW) Trials\n(wrapped angle > 0)', color=COLORS['cw'])

    ccw_mask = dark_df['cwccw_trial_end_wrapped'] <= 0
    plot_polar_histogram(axes[2], dark_df.loc[ccw_mask, 'heading_movement'].values,
                        title='Counter-Clockwise (CCW) Trials\n(wrapped angle ≤ 0)', color=COLORS['ccw'])

    # Main title with explanation
    fig.suptitle('Initial Heading Direction When Leaving Home\n'
                 '(Movement direction in first 0.5s of search, dark trials only)\n'
                 'CW/CCW based on WRAPPED angle | Red arrow = mean direction, length ∝ MVL',
                 fontsize=12, y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_WRAPPED, 'figures', 'fig3_polar_distributions_wrapped.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Wrapped figures saved to {OUTPUT_DIR_WRAPPED}/figures/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the analysis pipeline
    results_df, results_df_wrapped = run_analysis_pipeline(
        save_figures=True,
        use_hdpose=True,  # Set to False to skip hdPose calculation (faster)
        verbose=True
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"\n--- UNWRAPPED ANALYSIS ---")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  - trial_data.csv: Trial-level data with heading and CW/CCW metrics")
    print("  - summary_statistics.csv: Statistical analysis results")
    print("  - figures/fig1_scatter_dark_trials.pdf: Main scatter plots")
    print("  - figures/fig2_forest_plot.pdf: Effect size forest plot")
    print("  - figures/fig3_polar_distributions.pdf: Polar histograms")

    print(f"\n--- WRAPPED ANALYSIS (angles wrapped to [-pi, +pi]) ---")
    print(f"Output directory: {OUTPUT_DIR_WRAPPED}")
    print("\nFiles generated:")
    print("  - trial_data_wrapped.csv: Trial-level data with wrapped CW/CCW metrics")
    print("  - summary_statistics_wrapped.csv: Statistical analysis results (wrapped)")
    print("  - figures/fig1_scatter_dark_trials_wrapped.pdf: Main scatter plots (wrapped)")
    print("  - figures/fig2_forest_plot_wrapped.pdf: Effect size forest plot (wrapped)")
    print("  - figures/fig3_polar_distributions_wrapped.pdf: Polar histograms (wrapped)")
