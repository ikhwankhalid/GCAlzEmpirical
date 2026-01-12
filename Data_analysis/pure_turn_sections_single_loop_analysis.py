"""
Pure Turn Sections Analysis: Single-Loop Trials Only

Research Question:
    Does heading deviation vary based on erroneous integration of self-motion,
    when analyzing only trials where the animal completes at most one loop
    around the lever?

Key Modification from pure_turn_sections_analysis_relaxed.py:
    - Pre-filters trials based on cumSumDiffAngleAroundTarget from navPathInstan.csv
    - Excludes trials where max |cumSumDiffAngleAroundTarget| > threshold at ANY point
    - This removes multi-loop trials where the animal circled the lever more than once

Rationale:
    Trials where animals circle the lever multiple times represent complex spatial
    behavior that may confound the relationship between self-motion integration
    and heading error. By filtering to single-loop trials, we isolate cleaner
    path-integration scenarios.

Author: Analysis generated for Peng et al. 2025
Date: 2025-01-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_DATA_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng'

# Sessions to use (from setup_project.py)
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

# Trial filtering parameters
CUMULATIVE_ANGLE_THRESHOLD = 7.5  # radians (~2.4*pi, allows for noise)

# Analysis parameters (same as relaxed version)
SPEED_THRESHOLDS = [1.0, 2.0, 3.0, 5.0]  # cm/s
MIN_BOUT_LENGTH = 3  # timepoints
MIN_SECTION_LENGTH = 3  # timepoints
ANGULAR_VELOCITY_THRESHOLD = 0.005  # rad/s
SMOOTH_WINDOW = 0.5  # for angular velocity smoothing
MAX_INTEGRATED_THRESHOLD = np.pi  # Maximum integrated angular velocity (radians)

# Conditions to analyze
CONDITIONS = [
    'all_light',
    'all_dark',
    'searchToLeverPath_light',
    'searchToLeverPath_dark',
    'homingFromLeavingLever_light',
    'homingFromLeavingLever_dark',
    'atLever_light',
    'atLever_dark'
]

# =============================================================================
# TRIAL FILTERING FUNCTIONS
# =============================================================================

def load_session_nav_instan(session_name):
    """
    Load navPathInstan.csv for a single session.

    Args:
        session_name: Session identifier (e.g., 'jp486-24032023-0108')

    Returns:
        DataFrame with instantaneous behavioral variables, or None if not found
    """
    animal_id = session_name.split('-')[0]
    session_path = os.path.join(PROJECT_DATA_PATH, animal_id, session_name)
    nav_instan_path = os.path.join(session_path, 'navPathInstan.csv')

    if not os.path.exists(nav_instan_path):
        return None

    nav_instan = pd.read_csv(nav_instan_path)
    nav_instan['session'] = session_name
    return nav_instan


def get_trial_max_cumulative_angle(nav_instan, session_name):
    """
    Compute max |cumSumDiffAngleAroundTarget| per trial.

    IMPORTANT: Uses MAXIMUM value at ANY point during trial (most conservative).
    If animal ever exceeded threshold, even momentarily, trial is excluded.

    Args:
        nav_instan: DataFrame from navPathInstan.csv
        session_name: Session identifier

    Returns:
        dict: {trialNo: max_abs_cum_angle}
    """
    if nav_instan is None or 'cumSumDiffAngleAroundTarget' not in nav_instan.columns:
        return {}

    # Filter for complete trial trajectory (name ends with '_all')
    all_path_mask = nav_instan['name'].str.endswith('_all')
    trial_data = nav_instan[all_path_mask].copy()

    if len(trial_data) == 0:
        return {}

    # Compute max |cumSumDiffAngleAroundTarget| per trial
    trial_max = trial_data.groupby('trialNo')['cumSumDiffAngleAroundTarget'].apply(
        lambda x: np.nanmax(np.abs(x))
    ).to_dict()

    return trial_max


def load_all_session_max_angles(sessions):
    """
    Load navPathInstan for all sessions and compute max cumulative angles.

    Args:
        sessions: List of session names

    Returns:
        dict: {(session, trialNo): max_abs_cum_angle}
    """
    print("Loading navPathInstan data for trial filtering...")
    all_max_angles = {}

    for session_name in tqdm(sessions, desc="Loading session data"):
        nav_instan = load_session_nav_instan(session_name)
        if nav_instan is not None:
            trial_max = get_trial_max_cumulative_angle(nav_instan, session_name)
            for trial_no, max_angle in trial_max.items():
                all_max_angles[(session_name, trial_no)] = max_angle

    print(f"  Loaded max angles for {len(all_max_angles)} trials across {len(sessions)} sessions")
    return all_max_angles


def filter_single_loop_trials(df, session_trial_max_angles, threshold=7.5):
    """
    Remove trials where animal circled lever more than once.

    Args:
        df: Reconstruction DataFrame with 'session' and 'trial' columns
        session_trial_max_angles: dict of {(session, trial): max_angle}
        threshold: Maximum allowed cumulative angle (radians)

    Returns:
        tuple: (filtered_df, filter_stats_dict)
    """
    n_before = len(df)
    n_trials_before = df.groupby(['session', 'trial']).ngroups

    # Create mask for valid trials
    def is_single_loop(row):
        key = (row['session'], row['trial'])
        if key not in session_trial_max_angles:
            return True  # Keep trial if no data available (conservative)
        return session_trial_max_angles[key] <= threshold

    # Apply filter (vectorized approach for speed)
    df = df.copy()
    df['_session_trial'] = list(zip(df['session'], df['trial']))
    valid_trials = {k for k, v in session_trial_max_angles.items() if v <= threshold}
    # Also include trials not in the dict (no navPathInstan data)
    all_trials = set(df['_session_trial'].unique())
    trials_in_dict = set(session_trial_max_angles.keys())
    trials_not_in_dict = all_trials - trials_in_dict

    keep_trials = valid_trials | trials_not_in_dict
    df_filtered = df[df['_session_trial'].isin(keep_trials)].copy()
    df_filtered = df_filtered.drop(columns=['_session_trial'])

    n_after = len(df_filtered)
    n_trials_after = df_filtered.groupby(['session', 'trial']).ngroups

    # Compute statistics
    excluded_trials = trials_in_dict - valid_trials
    n_excluded = len(excluded_trials)
    n_no_data = len(trials_not_in_dict)

    filter_stats = {
        'n_rows_before': n_before,
        'n_rows_after': n_after,
        'n_rows_removed': n_before - n_after,
        'pct_rows_removed': 100 * (n_before - n_after) / n_before if n_before > 0 else 0,
        'n_trials_before': n_trials_before,
        'n_trials_after': n_trials_after,
        'n_trials_excluded': n_excluded,
        'n_trials_no_data': n_no_data,
        'pct_trials_excluded': 100 * n_excluded / n_trials_before if n_trials_before > 0 else 0,
        'threshold': threshold
    }

    return df_filtered, filter_stats


# =============================================================================
# UTILITY FUNCTIONS (from pure_turn_sections_analysis_relaxed.py)
# =============================================================================

def calculate_heading_from_position(x, y, smooth_window=1.0):
    """Calculate instantaneous heading from position trajectory."""
    if smooth_window > 1:
        x_smooth = scipy.ndimage.gaussian_filter1d(x, sigma=smooth_window/3)
        y_smooth = scipy.ndimage.gaussian_filter1d(y, sigma=smooth_window/3)
    else:
        x_smooth, y_smooth = x, y

    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)
    heading = np.full(len(x), np.nan)
    heading[1:] = np.arctan2(dy, dx)
    return heading


def calculate_angular_velocity(heading, time, smooth_window=1.0):
    """Calculate instantaneous angular velocity (turning rate) in rad/s."""
    dheading = np.diff(heading)
    dheading = np.arctan2(np.sin(dheading), np.cos(dheading))

    dt = np.diff(time)
    dt[dt == 0] = np.nan

    angular_vel = dheading / dt
    angular_velocity = np.full(len(heading), np.nan)
    angular_velocity[1:] = angular_vel

    if smooth_window > 0:
        valid_mask = ~np.isnan(angular_velocity)
        if np.sum(valid_mask) > 0:
            angular_velocity[valid_mask] = scipy.ndimage.gaussian_filter1d(
                angular_velocity[valid_mask], sigma=smooth_window)

    return angular_velocity


def calculate_speed(x, y, time):
    """Calculate instantaneous speed in cm/s."""
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(time)
    dt[dt == 0] = np.nan

    distance = np.sqrt(dx**2 + dy**2)
    speed_vals = distance / dt

    speed = np.full(len(x), np.nan)
    speed[1:] = speed_vals
    return speed


# =============================================================================
# BOUT AND SECTION IDENTIFICATION
# =============================================================================

def identify_continuous_bouts(speed, threshold, min_length=3):
    """Identify continuous movement bouts where speed >= threshold."""
    moving = speed >= threshold
    moving = moving & ~np.isnan(speed)

    moving_padded = np.concatenate(([False], moving, [False]))
    diff = np.diff(moving_padded.astype(int))

    bout_starts = np.where(diff == 1)[0]
    bout_ends = np.where(diff == -1)[0]

    bouts = [(start, end) for start, end in zip(bout_starts, bout_ends)
             if (end - start) >= min_length]

    return bouts


def identify_pure_turn_sections(angular_velocity, min_length, zero_threshold):
    """
    Identify continuous sections where angular velocity maintains the same sign.
    """
    n = len(angular_velocity)

    signs = np.zeros(n)
    for i in range(n):
        if np.isnan(angular_velocity[i]):
            signs[i] = 0
        elif np.abs(angular_velocity[i]) < zero_threshold:
            signs[i] = 0
        elif angular_velocity[i] > 0:
            signs[i] = 1
        else:
            signs[i] = -1

    sections = []
    i = 0
    while i < n:
        if signs[i] == 0:
            i += 1
            continue

        current_sign = signs[i]
        start_idx = i

        while i < n and signs[i] == current_sign:
            i += 1
        end_idx = i

        if (end_idx - start_idx) >= min_length:
            direction = 'left' if current_sign > 0 else 'right'
            sections.append((start_idx, end_idx, direction))

    return sections


def integrate_within_section(angular_velocity, time, section_start, section_end, max_integrated=None):
    """Compute integrated angular velocity within a single pure turn section."""
    omega = angular_velocity[section_start:section_end]
    t = time[section_start:section_end]

    dt = np.diff(t)

    integrated = np.zeros(len(omega))
    integrated[1:] = np.nancumsum(omega[:-1] * dt)

    effective_end = section_end
    if max_integrated is not None:
        exceeds_threshold = np.where(np.abs(integrated) > max_integrated)[0]
        if len(exceeds_threshold) > 0:
            trunc_idx = exceeds_threshold[0]
            integrated = integrated[:trunc_idx]
            effective_end = section_start + trunc_idx

    return integrated, effective_end


def process_trial_with_pure_sections(trial_data, speed_threshold, min_bout_length=3,
                                      min_section_length=3, angular_velocity_threshold=0.005,
                                      max_integrated=None):
    """Process a single trial: identify bouts, then pure turn sections within bouts."""
    if len(trial_data) < min_bout_length:
        return None

    heading = calculate_heading_from_position(trial_data['x'].values, trial_data['y'].values)
    angular_velocity = calculate_angular_velocity(heading, trial_data['recTime'].values)
    speed = calculate_speed(trial_data['x'].values, trial_data['y'].values, trial_data['recTime'].values)

    bouts = identify_continuous_bouts(speed, speed_threshold, min_bout_length)

    if len(bouts) == 0:
        return None

    section_results = []
    section_id_global = 0

    for bout_idx, (bout_start, bout_end) in enumerate(bouts):
        bout_ang_vel = angular_velocity[bout_start:bout_end]

        sections_in_bout = identify_pure_turn_sections(
            bout_ang_vel, min_section_length, angular_velocity_threshold
        )

        for section_start_rel, section_end_rel, direction in sections_in_bout:
            section_start = bout_start + section_start_rel
            section_end = bout_start + section_end_rel

            integrated_ang_vel, effective_end = integrate_within_section(
                angular_velocity, trial_data['recTime'].values,
                section_start, section_end, max_integrated
            )

            effective_length = effective_end - section_start
            if effective_length < min_section_length:
                continue

            section_slice = trial_data.iloc[section_start:effective_end].copy()
            section_slice['bout_id'] = bout_idx
            section_slice['section_id'] = section_id_global
            section_slice['section_start'] = section_start
            section_slice['section_end'] = effective_end
            section_slice['section_length'] = effective_length
            section_slice['time_in_section'] = (section_slice['recTime'].values -
                                                 section_slice['recTime'].values[0])
            section_slice['angular_velocity'] = angular_velocity[section_start:effective_end]
            section_slice['integrated_ang_vel'] = integrated_ang_vel
            section_slice['speed'] = speed[section_start:effective_end]
            section_slice['heading'] = heading[section_start:effective_end]
            section_slice['turn_direction'] = direction

            section_results.append(section_slice)
            section_id_global += 1

    if len(section_results) > 0:
        return pd.concat(section_results, ignore_index=True)
    else:
        return None


# =============================================================================
# REGRESSION ANALYSIS FUNCTIONS
# =============================================================================

def regression_by_turn_direction(data, condition_name, speed_threshold):
    """Perform regression analysis separately for left and right turn sections."""
    left_data = data[data['turn_direction'] == 'left'].copy()
    right_data = data[data['turn_direction'] == 'right'].copy()

    results = {
        'condition': condition_name,
        'speed_threshold': speed_threshold,
        'n_total': len(data),
        'n_left': len(left_data),
        'n_right': len(right_data),
        'n_sections': data['section_id'].nunique() if 'section_id' in data.columns else 0,
        'n_left_sections': left_data['section_id'].nunique() if 'section_id' in left_data.columns and len(left_data) > 0 else 0,
        'n_right_sections': right_data['section_id'].nunique() if 'section_id' in right_data.columns and len(right_data) > 0 else 0
    }

    # Regression for left turns
    if len(left_data) > 10:
        X_left = left_data['integrated_ang_vel'].values
        Y_left = left_data['mvtDirError'].values

        valid = ~(np.isnan(X_left) | np.isnan(Y_left))
        X_left = X_left[valid]
        Y_left = Y_left[valid]

        if len(X_left) > 10:
            slope_left, intercept_left, r_left, p_left, se_left = stats.linregress(X_left, Y_left)
            results['beta_left'] = slope_left
            results['intercept_left'] = intercept_left
            results['r_left'] = r_left
            results['p_left'] = p_left
            results['se_left'] = se_left
        else:
            results['beta_left'] = np.nan
            results['intercept_left'] = np.nan
            results['r_left'] = np.nan
            results['p_left'] = np.nan
            results['se_left'] = np.nan
    else:
        results['beta_left'] = np.nan
        results['intercept_left'] = np.nan
        results['r_left'] = np.nan
        results['p_left'] = np.nan
        results['se_left'] = np.nan

    # Regression for right turns
    if len(right_data) > 10:
        X_right = right_data['integrated_ang_vel'].values
        Y_right = right_data['mvtDirError'].values

        valid = ~(np.isnan(X_right) | np.isnan(Y_right))
        X_right = X_right[valid]
        Y_right = Y_right[valid]

        if len(X_right) > 10:
            slope_right, intercept_right, r_right, p_right, se_right = stats.linregress(X_right, Y_right)
            results['beta_right'] = slope_right
            results['intercept_right'] = intercept_right
            results['r_right'] = r_right
            results['p_right'] = p_right
            results['se_right'] = se_right
        else:
            results['beta_right'] = np.nan
            results['intercept_right'] = np.nan
            results['r_right'] = np.nan
            results['p_right'] = np.nan
            results['se_right'] = np.nan
    else:
        results['beta_right'] = np.nan
        results['intercept_right'] = np.nan
        results['r_right'] = np.nan
        results['p_right'] = np.nan
        results['se_right'] = np.nan

    # Test for asymmetry
    if not np.isnan(results['beta_left']) and not np.isnan(results['beta_right']):
        results['beta_diff'] = results['beta_left'] - results['beta_right']

        se_diff = np.sqrt(results['se_left']**2 + results['se_right']**2)
        z_stat = results['beta_diff'] / se_diff if se_diff > 0 else 0
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        results['z_asymmetry'] = z_stat
        results['p_asymmetry'] = p_diff
    else:
        results['beta_diff'] = np.nan
        results['z_asymmetry'] = np.nan
        results['p_asymmetry'] = np.nan

    return results


def signed_regression_analysis(data, condition_name, speed_threshold):
    """Perform signed regression analysis between integrated angular velocity and heading deviation."""
    results = {
        'condition': condition_name,
        'speed_threshold': speed_threshold,
        'n_total': len(data),
        'n_sections': data['section_id'].nunique() if 'section_id' in data.columns else 0
    }

    X = data['integrated_ang_vel'].values
    Y = data['mvtDirError'].values

    valid = ~(np.isnan(X) | np.isnan(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) > 10:
        slope, intercept, r, p, se = stats.linregress(X, Y)

        results['beta_signed'] = slope
        results['intercept_signed'] = intercept
        results['r_signed'] = r
        results['r_squared_signed'] = r**2
        results['p_signed'] = p
        results['se_signed'] = se

        if np.std(X) > 0 and np.std(Y) > 0:
            beta_standardized = slope * (np.std(X) / np.std(Y))
            results['beta_standardized'] = beta_standardized
        else:
            results['beta_standardized'] = np.nan

        from scipy.stats import t as t_dist
        df = len(X) - 2
        t_crit = t_dist.ppf(0.975, df)
        ci_lower = slope - t_crit * se
        ci_upper = slope + t_crit * se
        results['ci_lower_95'] = ci_lower
        results['ci_upper_95'] = ci_upper

        results['integrated_ang_vel_mean'] = np.mean(X)
        results['integrated_ang_vel_std'] = np.std(X)
        results['integrated_ang_vel_range'] = (np.min(X), np.max(X))
        results['mvtDirError_mean'] = np.mean(Y)
        results['mvtDirError_std'] = np.std(Y)
    else:
        results['beta_signed'] = np.nan
        results['intercept_signed'] = np.nan
        results['r_signed'] = np.nan
        results['r_squared_signed'] = np.nan
        results['p_signed'] = np.nan
        results['se_signed'] = np.nan
        results['beta_standardized'] = np.nan
        results['ci_lower_95'] = np.nan
        results['ci_upper_95'] = np.nan
        results['integrated_ang_vel_mean'] = np.nan
        results['integrated_ang_vel_std'] = np.nan
        results['integrated_ang_vel_range'] = (np.nan, np.nan)
        results['mvtDirError_mean'] = np.nan
        results['mvtDirError_std'] = np.nan

    return results


def test_section_accumulation(data):
    """Test if heading deviation magnitude grows over time within sections."""
    data = data.copy()
    data['abs_heading_dev'] = np.abs(data['mvtDirError'])

    X = data['time_in_section'].values
    Y = data['abs_heading_dev'].values

    valid = ~(np.isnan(X) | np.isnan(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) > 10:
        slope, intercept, r, p, se = stats.linregress(X, Y)
        return {
            'accumulation_slope': slope,
            'accumulation_r': r,
            'accumulation_p': p,
            'accumulation_se': se
        }
    else:
        return {
            'accumulation_slope': np.nan,
            'accumulation_r': np.nan,
            'accumulation_p': np.nan,
            'accumulation_se': np.nan
        }


# =============================================================================
# SECTION ENDPOINT ANALYSIS
# =============================================================================

def extract_section_endpoints(section_data):
    """Extract the final timepoint of each pure turn section."""
    if len(section_data) == 0:
        return pd.DataFrame()

    section_data = section_data.copy()
    section_data['unique_section_key'] = (
        section_data['condition'].astype(str) + '_' +
        section_data['speed_threshold'].astype(str) + '_' +
        section_data['trial_id'].astype(str) + '_' +
        section_data['section_id'].astype(str)
    )

    endpoints = section_data.groupby('unique_section_key').apply(
        lambda x: x.iloc[-1]
    ).reset_index(drop=True)

    endpoints = endpoints.drop(columns=['unique_section_key'], errors='ignore')
    return endpoints


def regression_on_endpoints(endpoints, condition_name, speed_threshold):
    """Perform regression analysis on section endpoints only."""
    results = {
        'condition': condition_name,
        'speed_threshold': speed_threshold,
        'analysis_type': 'endpoints',
        'n_sections': len(endpoints)
    }

    left_endpoints = endpoints[endpoints['turn_direction'] == 'left']
    right_endpoints = endpoints[endpoints['turn_direction'] == 'right']
    results['n_left_sections'] = len(left_endpoints)
    results['n_right_sections'] = len(right_endpoints)

    X = endpoints['integrated_ang_vel'].values
    Y = endpoints['mvtDirError'].values
    valid = ~(np.isnan(X) | np.isnan(Y))
    X_valid, Y_valid = X[valid], Y[valid]

    if len(X_valid) > 10:
        slope, intercept, r, p, se = stats.linregress(X_valid, Y_valid)
        results['beta_signed'] = slope
        results['r_squared_signed'] = r**2
        results['p_signed'] = p
        results['se_signed'] = se
        from scipy.stats import t as t_dist
        df = len(X_valid) - 2
        t_crit = t_dist.ppf(0.975, df)
        results['ci_lower_95'] = slope - t_crit * se
        results['ci_upper_95'] = slope + t_crit * se
    else:
        results['beta_signed'] = np.nan
        results['r_squared_signed'] = np.nan
        results['p_signed'] = np.nan
        results['se_signed'] = np.nan
        results['ci_lower_95'] = np.nan
        results['ci_upper_95'] = np.nan

    # Left/right regressions
    for name, subset in [('left', left_endpoints), ('right', right_endpoints)]:
        if len(subset) > 10:
            X_sub = subset['integrated_ang_vel'].values
            Y_sub = subset['mvtDirError'].values
            valid_sub = ~(np.isnan(X_sub) | np.isnan(Y_sub))
            if np.sum(valid_sub) > 10:
                slope_sub, _, _, p_sub, se_sub = stats.linregress(X_sub[valid_sub], Y_sub[valid_sub])
                results[f'beta_{name}'] = slope_sub
                results[f'p_{name}'] = p_sub
                results[f'se_{name}'] = se_sub
            else:
                results[f'beta_{name}'] = np.nan
                results[f'p_{name}'] = np.nan
                results[f'se_{name}'] = np.nan
        else:
            results[f'beta_{name}'] = np.nan
            results[f'p_{name}'] = np.nan
            results[f'se_{name}'] = np.nan

    # Asymmetry test
    if not np.isnan(results.get('beta_left', np.nan)) and not np.isnan(results.get('beta_right', np.nan)):
        results['beta_diff'] = results['beta_left'] - results['beta_right']
        se_diff = np.sqrt(results['se_left']**2 + results['se_right']**2)
        z_stat = results['beta_diff'] / se_diff if se_diff > 0 else 0
        results['p_asymmetry'] = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        results['beta_diff'] = np.nan
        results['p_asymmetry'] = np.nan

    return results


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_all_conditions(df, conditions, speed_thresholds, min_bout_length=3,
                           min_section_length=3, angular_velocity_threshold=0.005,
                           max_integrated=None):
    """Process all conditions and speed thresholds."""
    all_section_data = []
    regression_results = []
    all_endpoint_data = []
    endpoint_regression_results = []

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Processing condition: {condition}")
        print(f"{'='*60}")

        condition_df = df[df.condition == condition].copy()
        condition_df['session_trial'] = condition_df['session'] + '_T' + condition_df['trial'].astype(str)
        unique_trials = condition_df['session_trial'].unique()
        print(f"Found {len(unique_trials)} trials (after single-loop filtering)")

        for speed_threshold in speed_thresholds:
            print(f"\n  Speed threshold: {speed_threshold} cm/s")

            section_data_list = []

            for trial_id in tqdm(unique_trials, desc=f"  Processing trials"):
                trial_data = condition_df[condition_df['session_trial'] == trial_id].copy()
                trial_data = trial_data.sort_values('recTime')

                section_data = process_trial_with_pure_sections(
                    trial_data, speed_threshold, min_bout_length,
                    min_section_length, angular_velocity_threshold, max_integrated
                )

                if section_data is not None:
                    section_data['condition'] = condition
                    section_data['speed_threshold'] = speed_threshold
                    section_data['trial_id'] = trial_id
                    section_data_list.append(section_data)

            if len(section_data_list) > 0:
                combined = pd.concat(section_data_list, ignore_index=True)
                all_section_data.append(combined)

                n_left_sections = combined[combined['turn_direction'] == 'left']['section_id'].nunique()
                n_right_sections = combined[combined['turn_direction'] == 'right']['section_id'].nunique()

                print(f"    Total sections: {combined['section_id'].nunique()}")
                print(f"    Left sections: {n_left_sections}")
                print(f"    Right sections: {n_right_sections}")
                print(f"    Total timepoints: {len(combined)}")

                # Regression by turn direction
                reg_results_old = regression_by_turn_direction(combined, condition, speed_threshold)

                # Signed regression
                reg_results_signed = signed_regression_analysis(combined, condition, speed_threshold)

                # Merge results
                reg_results = {**reg_results_old, **reg_results_signed}

                # Accumulation test
                accum_results = test_section_accumulation(combined)
                reg_results.update(accum_results)

                regression_results.append(reg_results)

                # Endpoint analysis
                endpoints = extract_section_endpoints(combined)
                if len(endpoints) > 0:
                    all_endpoint_data.append(endpoints)
                    endpoint_reg = regression_on_endpoints(endpoints, condition, speed_threshold)
                    endpoint_regression_results.append(endpoint_reg)
                    print(f"    ENDPOINT ANALYSIS (n={len(endpoints)} sections):")
                    if not np.isnan(endpoint_reg.get('beta_signed', np.nan)):
                        print(f"      beta_signed: {endpoint_reg['beta_signed']:.6f}, R²={endpoint_reg['r_squared_signed']:.4f}")

                print(f"    ALL TIMEPOINTS:")
                print(f"      beta_signed: {reg_results['beta_signed']:.6f}, p={reg_results['p_signed']:.4f}")
                print(f"      R²={reg_results['r_squared_signed']:.4f}, 95% CI=[{reg_results['ci_lower_95']:.6f}, {reg_results['ci_upper_95']:.6f}]")

    # Combine results
    if len(all_section_data) > 0:
        all_section_data = pd.concat(all_section_data, ignore_index=True)
    else:
        all_section_data = pd.DataFrame()

    regression_results = pd.DataFrame(regression_results)

    if len(all_endpoint_data) > 0:
        all_endpoint_data = pd.concat(all_endpoint_data, ignore_index=True)
    else:
        all_endpoint_data = pd.DataFrame()

    endpoint_regression_results = pd.DataFrame(endpoint_regression_results)

    return all_section_data, regression_results, all_endpoint_data, endpoint_regression_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PURE TURN SECTIONS ANALYSIS (SINGLE-LOOP TRIALS ONLY)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data path: {PROJECT_DATA_PATH}")
    print(f"  Cumulative angle threshold: {CUMULATIVE_ANGLE_THRESHOLD} radians (~{CUMULATIVE_ANGLE_THRESHOLD/np.pi:.2f}*pi)")
    print(f"  Speed thresholds: {SPEED_THRESHOLDS} cm/s")
    print(f"  Minimum bout length: {MIN_BOUT_LENGTH} timepoints")
    print(f"  Minimum section length: {MIN_SECTION_LENGTH} timepoints")
    print(f"  Angular velocity threshold: {ANGULAR_VELOCITY_THRESHOLD} rad/s")
    print(f"  Max integrated threshold: {MAX_INTEGRATED_THRESHOLD} radians")
    print(f"  Conditions: {len(CONDITIONS)}")
    print(f"  Sessions: {len(useAble)}")

    # Step 1: Load trial filtering data
    print("\n" + "="*80)
    print("STEP 1: LOADING TRIAL FILTERING DATA")
    print("="*80)

    session_trial_max_angles = load_all_session_max_angles(useAble)

    # Step 2: Load reconstruction data
    print("\n" + "="*80)
    print("STEP 2: LOADING RECONSTRUCTION DATA")
    print("="*80)
    fn = os.path.join(PROJECT_DATA_PATH, "results", "reconstuctionDFAutoPI.csv")
    print(f"Loading: {fn}")
    dfAutoPI = pd.read_csv(fn)
    print(f"Loaded {len(dfAutoPI):,} rows")

    # Filter for useable sessions
    dfAutoPI = dfAutoPI[dfAutoPI.session.isin(useAble)]
    print(f"After filtering for useable sessions: {len(dfAutoPI):,} rows")

    # Step 3: Apply single-loop trial filter
    print("\n" + "="*80)
    print("STEP 3: FILTERING TO SINGLE-LOOP TRIALS")
    print("="*80)

    dfAutoPI_filtered, filter_stats = filter_single_loop_trials(
        dfAutoPI, session_trial_max_angles, CUMULATIVE_ANGLE_THRESHOLD
    )

    print(f"\nFiltering Statistics:")
    print(f"  Threshold: {filter_stats['threshold']:.2f} radians (~{filter_stats['threshold']/np.pi:.2f}*pi)")
    print(f"  Trials before: {filter_stats['n_trials_before']}")
    print(f"  Trials after: {filter_stats['n_trials_after']}")
    print(f"  Trials excluded (multi-loop): {filter_stats['n_trials_excluded']}")
    print(f"  Trials with no navPathInstan data: {filter_stats['n_trials_no_data']}")
    print(f"  Percentage excluded: {filter_stats['pct_trials_excluded']:.1f}%")
    print(f"  Rows before: {filter_stats['n_rows_before']:,}")
    print(f"  Rows after: {filter_stats['n_rows_after']:,}")
    print(f"  Rows removed: {filter_stats['n_rows_removed']:,} ({filter_stats['pct_rows_removed']:.1f}%)")

    # Save filter stats
    results_dir = os.path.join(PROJECT_DATA_PATH, "results")
    filter_stats_fn = os.path.join(results_dir, "single_loop_trial_filter_stats.csv")
    pd.DataFrame([filter_stats]).to_csv(filter_stats_fn, index=False)
    print(f"\nSaved filter stats: {filter_stats_fn}")

    # Step 4: Process all conditions
    print("\n" + "="*80)
    print("STEP 4: PROCESSING ALL CONDITIONS")
    print("="*80)

    all_section_data, regression_results, all_endpoint_data, endpoint_regression_results = process_all_conditions(
        dfAutoPI_filtered,
        CONDITIONS,
        SPEED_THRESHOLDS,
        MIN_BOUT_LENGTH,
        MIN_SECTION_LENGTH,
        ANGULAR_VELOCITY_THRESHOLD,
        MAX_INTEGRATED_THRESHOLD
    )

    # Step 5: Save results
    print("\n" + "="*80)
    print("STEP 5: SAVING RESULTS")
    print("="*80)

    if len(all_section_data) > 0:
        section_data_fn = os.path.join(results_dir, "pure_turn_section_data_single_loop.csv")
        all_section_data.to_csv(section_data_fn, index=False)
        print(f"Saved section data: {section_data_fn}")
        print(f"  Total rows: {len(all_section_data):,}")

    if len(regression_results) > 0:
        reg_results_fn = os.path.join(results_dir, "pure_turn_section_regression_single_loop.csv")
        regression_results.to_csv(reg_results_fn, index=False)
        print(f"Saved regression results: {reg_results_fn}")

    # Save endpoint data
    if len(all_endpoint_data) > 0:
        endpoint_data_fn = os.path.join(results_dir, "pure_turn_section_endpoints_single_loop.csv")
        all_endpoint_data.to_csv(endpoint_data_fn, index=False)
        print(f"Saved endpoint data: {endpoint_data_fn}")
        print(f"  Total sections: {len(all_endpoint_data):,}")

    if len(endpoint_regression_results) > 0:
        endpoint_reg_fn = os.path.join(results_dir, "pure_turn_section_endpoints_regression_single_loop.csv")
        endpoint_regression_results.to_csv(endpoint_reg_fn, index=False)
        print(f"Saved endpoint regression results: {endpoint_reg_fn}")
        print(f"  Total rows: {len(regression_results)}")

    # Step 6: Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    if len(regression_results) > 0:
        print("\nSIGNED REGRESSION RESULTS (Single-Loop Trials Only)")
        print("="*60)
        print("\nDirect test: Does signed cumulative turning predict signed heading deviation?")
        print("Expected: beta < 0 (accumulated left turns -> rightward heading error)\n")

        signed_cols = ['condition', 'speed_threshold', 'n_sections', 'beta_signed',
                      'r_squared_signed', 'p_signed', 'ci_lower_95', 'ci_upper_95']
        available_cols = [col for col in signed_cols if col in regression_results.columns]
        print(regression_results[available_cols].to_string(index=False))

        sig_signed = (regression_results['p_signed'] < 0.05).sum()
        total_signed = len(regression_results[~regression_results['p_signed'].isna()])
        print(f"\nSignificant signed relationships (p < 0.05): {sig_signed}/{total_signed}")

        negative_betas = (regression_results['beta_signed'] < 0).sum()
        print(f"Negative beta (expected direction): {negative_betas}/{total_signed}")

        # Light vs Dark comparison
        print("\n" + "="*60)
        print("LIGHT vs DARK COMPARISON")
        print("="*60)

        for speed_thresh in SPEED_THRESHOLDS:
            light_row = regression_results[
                (regression_results['condition'] == 'all_light') &
                (regression_results['speed_threshold'] == speed_thresh)
            ]
            dark_row = regression_results[
                (regression_results['condition'] == 'all_dark') &
                (regression_results['speed_threshold'] == speed_thresh)
            ]

            if len(light_row) > 0 and len(dark_row) > 0:
                beta_light = light_row['beta_signed'].values[0]
                beta_dark = dark_row['beta_signed'].values[0]
                p_light = light_row['p_signed'].values[0]
                p_dark = dark_row['p_signed'].values[0]

                print(f"\nSpeed threshold: {speed_thresh} cm/s")
                print(f"  Light: beta={beta_light:.6f}, p={p_light:.4f}")
                print(f"  Dark:  beta={beta_dark:.6f}, p={p_dark:.4f}")
                print(f"  Difference: {beta_dark - beta_light:.6f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE (SINGLE-LOOP TRIALS ONLY)")
    print("="*80)
