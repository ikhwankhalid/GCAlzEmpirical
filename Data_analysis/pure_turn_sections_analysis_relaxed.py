"""
Pure Turn Sections Analysis: Testing for Systematic Integration Errors
(RELAXED FILTERING VERSION)

This script tests whether heading deviation results from systematic gain errors when
animals integrate angular velocity during PURE turning sections (continuous periods
where instantaneous angular velocity maintains the same sign).

Key Modification from integrated_angular_velocity_bouts_analysis.py:
- After identifying movement bouts, further split into "pure turn sections"
- Pure turn sections are continuous periods where angular velocity doesn't change sign
- Integration resets at each section start
- Near-zero angular velocities are excluded to avoid noise-induced section breaks

RELAXED PARAMETERS (compared to original):
- Speed thresholds: [1.0, 2.0, 3.0, 5.0] (removed 10.0, added 3.0)
- Min bout length: 3 (was 5)
- Min section length: 3 (was 5)
- Angular velocity threshold: 0.005 rad/s (was 0.01)

Author: Analysis generated for Peng et al. 2025
Date: 2025-11-26
Modified: 2025-12-09 (relaxed filtering)
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
# CONFIGURATION - RELAXED PARAMETERS
# =============================================================================

# Paths
PROJECT_DATA_PATH = '/workspace/Peng'

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

# Analysis parameters - RELAXED
SPEED_THRESHOLDS = [1.0, 2.0, 3.0, 5.0]  # cm/s - removed 10.0, added 3.0 for finer granularity
MIN_BOUT_LENGTH = 3  # timepoints - relaxed from 5
MIN_SECTION_LENGTH = 3  # timepoints - relaxed from 5
ANGULAR_VELOCITY_THRESHOLD = 0.005  # rad/s - relaxed from 0.01 to capture gentler turns
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
# UTILITY FUNCTIONS FROM EXISTING SCRIPTS
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
    """
    Calculate instantaneous angular velocity (turning rate) in rad/s.

    Parameters
    ----------
    heading : array
        Heading angles in radians
    time : array
        Time points in seconds
    smooth_window : float
        Gaussian smoothing sigma (default 1.0)

    Returns
    -------
    angular_velocity : array
        Angular velocity in rad/s
    """
    dheading = np.diff(heading)
    # Wrap to [-pi, pi]
    dheading = np.arctan2(np.sin(dheading), np.cos(dheading))

    dt = np.diff(time)
    dt[dt == 0] = np.nan

    angular_vel = dheading / dt
    angular_velocity = np.full(len(heading), np.nan)
    angular_velocity[1:] = angular_vel

    # Smooth if requested
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
# BOUT IDENTIFICATION FUNCTIONS
# =============================================================================

def identify_continuous_bouts(speed, threshold, min_length=3):
    """
    Identify continuous movement bouts where speed >= threshold.

    Parameters
    ----------
    speed : array
        Instantaneous speed values
    threshold : float
        Speed threshold for movement (cm/s)
    min_length : int
        Minimum bout length in timepoints

    Returns
    -------
    bouts : list of tuples
        List of (start_idx, end_idx) for each valid bout
    """
    # Create boolean array for movement
    moving = speed >= threshold
    moving = moving & ~np.isnan(speed)  # Also exclude NaN

    # Find transitions
    moving_padded = np.concatenate(([False], moving, [False]))
    diff = np.diff(moving_padded.astype(int))

    bout_starts = np.where(diff == 1)[0]
    bout_ends = np.where(diff == -1)[0]

    # Filter by minimum length
    bouts = [(start, end) for start, end in zip(bout_starts, bout_ends)
             if (end - start) >= min_length]

    return bouts

# =============================================================================
# PURE TURN SECTION IDENTIFICATION
# =============================================================================

def identify_pure_turn_sections(angular_velocity, min_length, zero_threshold):
    """
    Identify continuous sections where angular velocity maintains the same sign.

    This function finds "pure turn sections" - continuous periods where the animal
    is turning consistently in one direction (left or right) without reversals.

    Parameters
    ----------
    angular_velocity : array
        Instantaneous angular velocity in rad/s
    min_length : int
        Minimum section length in timepoints
    zero_threshold : float
        Angular velocity magnitude below which is considered "zero" (excluded)

    Returns
    -------
    sections : list of tuples
        List of (start_idx, end_idx, direction) where direction is 'left' or 'right'
    """
    n = len(angular_velocity)

    # Classify each timepoint: +1 for left, -1 for right, 0 for zero/NaN
    signs = np.zeros(n)
    for i in range(n):
        if np.isnan(angular_velocity[i]):
            signs[i] = 0
        elif np.abs(angular_velocity[i]) < zero_threshold:
            signs[i] = 0  # Near-zero, excluded
        elif angular_velocity[i] > 0:
            signs[i] = 1  # Left turn
        else:
            signs[i] = -1  # Right turn

    # Find runs of consecutive same-sign points (excluding zeros)
    sections = []
    i = 0
    while i < n:
        # Skip zeros
        if signs[i] == 0:
            i += 1
            continue

        # Start of a potential section
        current_sign = signs[i]
        start_idx = i

        # Find end of this section
        while i < n and signs[i] == current_sign:
            i += 1
        end_idx = i

        # Check minimum length
        if (end_idx - start_idx) >= min_length:
            direction = 'left' if current_sign > 0 else 'right'
            sections.append((start_idx, end_idx, direction))

    return sections

def get_section_statistics(sections, angular_velocity, time):
    """
    Calculate statistics for identified pure turn sections.

    Parameters
    ----------
    sections : list of tuples
        List of (start_idx, end_idx, direction)
    angular_velocity : array
        Angular velocity values
    time : array
        Time values

    Returns
    -------
    stats : dict
        Statistics about sections
    """
    if len(sections) == 0:
        return {
            'n_sections': 0,
            'n_left_sections': 0,
            'n_right_sections': 0,
            'total_timepoints': 0,
            'mean_section_length': 0,
            'median_section_length': 0,
            'mean_section_duration': 0,
            'mean_angular_velocity': 0
        }

    section_lengths = [end - start for start, end, _ in sections]
    section_durations = [time[end-1] - time[start] if end > start else 0
                         for start, end, _ in sections]
    section_ang_vels = [np.nanmean(np.abs(angular_velocity[start:end]))
                        for start, end, _ in sections]

    n_left = sum(1 for _, _, d in sections if d == 'left')
    n_right = sum(1 for _, _, d in sections if d == 'right')

    return {
        'n_sections': len(sections),
        'n_left_sections': n_left,
        'n_right_sections': n_right,
        'total_timepoints': sum(section_lengths),
        'mean_section_length': np.mean(section_lengths),
        'median_section_length': np.median(section_lengths),
        'mean_section_duration': np.mean(section_durations),
        'mean_angular_velocity': np.mean(section_ang_vels)
    }

# =============================================================================
# SECTION-BASED INTEGRATION FUNCTIONS
# =============================================================================

def integrate_within_section(angular_velocity, time, section_start, section_end, max_integrated=None):
    """
    Compute integrated angular velocity within a single pure turn section.
    Integration resets to 0 at section start.

    Parameters
    ----------
    angular_velocity : array
        Angular velocity in rad/s
    time : array
        Time points in seconds
    section_start : int
        Start index of section
    section_end : int
        End index of section
    max_integrated : float, optional
        Maximum absolute integrated angular velocity (radians).
        If specified, section will be truncated at first point where
        |integrated_ang_vel| > max_integrated.

    Returns
    -------
    integrated_ang_vel : array
        Cumulative integrated angular velocity (radians) from section start.
    effective_end : int
        Effective section end index after truncation.
    """
    omega = angular_velocity[section_start:section_end]
    t = time[section_start:section_end]

    # Compute dt
    dt = np.diff(t)

    # Cumulative integration using trapezoidal rule
    integrated = np.zeros(len(omega))
    integrated[1:] = np.nancumsum(omega[:-1] * dt)

    # Apply threshold if specified
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
    """
    Process a single trial: identify bouts, then pure turn sections within bouts.

    Parameters
    ----------
    trial_data : DataFrame
        Trial data with columns: x, y, recTime, mvtDirError, etc.
    speed_threshold : float
        Speed threshold for defining bouts (cm/s)
    min_bout_length : int
        Minimum bout length in timepoints
    min_section_length : int
        Minimum pure turn section length in timepoints
    angular_velocity_threshold : float
        Angular velocity below this is considered zero (rad/s)
    max_integrated : float, optional
        Maximum absolute integrated angular velocity for truncation

    Returns
    -------
    section_data : DataFrame
        Processed data with section information, or None if no valid sections
    """
    if len(trial_data) < min_bout_length:
        return None

    # Calculate kinematics
    heading = calculate_heading_from_position(trial_data['x'].values, trial_data['y'].values)
    angular_velocity = calculate_angular_velocity(heading, trial_data['recTime'].values)
    speed = calculate_speed(trial_data['x'].values, trial_data['y'].values, trial_data['recTime'].values)

    # Identify movement bouts first
    bouts = identify_continuous_bouts(speed, speed_threshold, min_bout_length)

    if len(bouts) == 0:
        return None

    # Process each bout to find pure turn sections
    section_results = []
    section_id_global = 0

    for bout_idx, (bout_start, bout_end) in enumerate(bouts):
        # Get angular velocity within this bout
        bout_ang_vel = angular_velocity[bout_start:bout_end]

        # Identify pure turn sections within this bout
        sections_in_bout = identify_pure_turn_sections(
            bout_ang_vel, min_section_length, angular_velocity_threshold
        )

        for section_start_rel, section_end_rel, direction in sections_in_bout:
            # Convert to absolute indices
            section_start = bout_start + section_start_rel
            section_end = bout_start + section_end_rel

            # Integrate within this section (reset at section start)
            integrated_ang_vel, effective_end = integrate_within_section(
                angular_velocity, trial_data['recTime'].values,
                section_start, section_end, max_integrated
            )

            # Check if section still meets minimum length after truncation
            effective_length = effective_end - section_start
            if effective_length < min_section_length:
                continue

            # Extract data for this section
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
            section_slice['turn_direction'] = direction  # All same within section

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
    """
    Perform regression analysis separately for left and right turn sections.

    Parameters
    ----------
    data : DataFrame
        Processed section data with integrated_ang_vel and mvtDirError columns
    condition_name : str
        Name of condition (for results table)
    speed_threshold : float
        Speed threshold used (for results table)

    Returns
    -------
    results : dict
        Regression statistics including slopes and p-values
    """
    # Separate by turn direction
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

    # Test for asymmetry (difference in slopes)
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
    """
    Perform signed regression analysis between integrated angular velocity
    and heading deviation without splitting by direction.

    Parameters
    ----------
    data : DataFrame
        Processed section data with integrated_ang_vel and mvtDirError columns
    condition_name : str
        Name of condition (for results table)
    speed_threshold : float
        Speed threshold used (for results table)

    Returns
    -------
    results : dict
        Regression statistics for signed relationship
    """
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
    """
    Test if heading deviation magnitude grows over time within sections.

    Parameters
    ----------
    data : DataFrame
        Section data with time_in_section and mvtDirError

    Returns
    -------
    stats : dict
        Accumulation statistics
    """
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
    """
    Extract the final timepoint of each pure turn section.

    This provides one data point per section where:
    - integrated_ang_vel = total cumulative turn in section
    - mvtDirError = heading error at section end

    This eliminates the zero-clustering issue where early timepoints in each
    section have near-zero cumulative turn values, confounding the regression.

    Parameters
    ----------
    section_data : DataFrame
        Full section data with all timepoints

    Returns
    -------
    endpoints : DataFrame
        One row per section with endpoint values
    """
    if len(section_data) == 0:
        return pd.DataFrame()

    # Group by unique section identifier and take last row
    # Need to create a unique section key that includes condition and speed_threshold
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

    # Drop the temporary key column
    endpoints = endpoints.drop(columns=['unique_section_key'], errors='ignore')

    return endpoints


def regression_on_endpoints(endpoints, condition_name, speed_threshold):
    """
    Perform regression analysis on section endpoints only.

    Tests: Does total accumulated turn predict heading error at section end?

    This approach:
    - Uses one independent data point per section (no autocorrelation)
    - Eliminates zero-clustering (each point represents total turn in section)
    - Directly tests the integration error hypothesis

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data with one row per section
    condition_name : str
        Name of condition (for results table)
    speed_threshold : float
        Speed threshold used (for results table)

    Returns
    -------
    results : dict
        Regression statistics for endpoint analysis
    """
    results = {
        'condition': condition_name,
        'speed_threshold': speed_threshold,
        'analysis_type': 'endpoints',
        'n_sections': len(endpoints)
    }

    # Separate by turn direction
    left_endpoints = endpoints[endpoints['turn_direction'] == 'left']
    right_endpoints = endpoints[endpoints['turn_direction'] == 'right']

    results['n_left_sections'] = len(left_endpoints)
    results['n_right_sections'] = len(right_endpoints)

    # Signed regression (all data together)
    X = endpoints['integrated_ang_vel'].values
    Y = endpoints['mvtDirError'].values

    valid = ~(np.isnan(X) | np.isnan(Y))
    X_valid = X[valid]
    Y_valid = Y[valid]

    if len(X_valid) > 10:
        slope, intercept, r, p, se = stats.linregress(X_valid, Y_valid)

        results['beta_signed'] = slope
        results['intercept_signed'] = intercept
        results['r_signed'] = r
        results['r_squared_signed'] = r**2
        results['p_signed'] = p
        results['se_signed'] = se

        # 95% CI
        from scipy.stats import t as t_dist
        df = len(X_valid) - 2
        t_crit = t_dist.ppf(0.975, df)
        results['ci_lower_95'] = slope - t_crit * se
        results['ci_upper_95'] = slope + t_crit * se

        # Distribution stats
        results['integrated_ang_vel_mean'] = np.mean(X_valid)
        results['integrated_ang_vel_std'] = np.std(X_valid)
        results['mvtDirError_mean'] = np.mean(Y_valid)
        results['mvtDirError_std'] = np.std(Y_valid)
    else:
        results['beta_signed'] = np.nan
        results['intercept_signed'] = np.nan
        results['r_signed'] = np.nan
        results['r_squared_signed'] = np.nan
        results['p_signed'] = np.nan
        results['se_signed'] = np.nan
        results['ci_lower_95'] = np.nan
        results['ci_upper_95'] = np.nan
        results['integrated_ang_vel_mean'] = np.nan
        results['integrated_ang_vel_std'] = np.nan
        results['mvtDirError_mean'] = np.nan
        results['mvtDirError_std'] = np.nan

    # Left turn regression
    if len(left_endpoints) > 10:
        X_left = left_endpoints['integrated_ang_vel'].values
        Y_left = left_endpoints['mvtDirError'].values
        valid_left = ~(np.isnan(X_left) | np.isnan(Y_left))
        X_left = X_left[valid_left]
        Y_left = Y_left[valid_left]

        if len(X_left) > 10:
            slope_left, intercept_left, r_left, p_left, se_left = stats.linregress(X_left, Y_left)
            results['beta_left'] = slope_left
            results['p_left'] = p_left
            results['se_left'] = se_left
        else:
            results['beta_left'] = np.nan
            results['p_left'] = np.nan
            results['se_left'] = np.nan
    else:
        results['beta_left'] = np.nan
        results['p_left'] = np.nan
        results['se_left'] = np.nan

    # Right turn regression
    if len(right_endpoints) > 10:
        X_right = right_endpoints['integrated_ang_vel'].values
        Y_right = right_endpoints['mvtDirError'].values
        valid_right = ~(np.isnan(X_right) | np.isnan(Y_right))
        X_right = X_right[valid_right]
        Y_right = Y_right[valid_right]

        if len(X_right) > 10:
            slope_right, intercept_right, r_right, p_right, se_right = stats.linregress(X_right, Y_right)
            results['beta_right'] = slope_right
            results['p_right'] = p_right
            results['se_right'] = se_right
        else:
            results['beta_right'] = np.nan
            results['p_right'] = np.nan
            results['se_right'] = np.nan
    else:
        results['beta_right'] = np.nan
        results['p_right'] = np.nan
        results['se_right'] = np.nan

    # Asymmetry test
    if not np.isnan(results.get('beta_left', np.nan)) and not np.isnan(results.get('beta_right', np.nan)):
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


# =============================================================================
# ADVANCED R² IMPROVEMENT STRATEGIES
# =============================================================================

def extract_animal_id(session_name):
    """
    Extract animal ID from session name.

    Session names follow format: ANIMAL_ID-DATE-SESSIONNUM
    e.g., 'jp486-19032023-0108' -> 'jp486'

    Parameters
    ----------
    session_name : str
        Full session identifier

    Returns
    -------
    animal_id : str
        Extracted animal identifier
    """
    import re
    match = re.match(r'^([^-]+)', str(session_name))
    return match.group(1) if match else str(session_name)


def mixed_effects_regression(endpoints, condition_filter=None, speed_filter=None):
    """
    Run mixed effects model with animal-level random effects.

    This accounts for between-animal heterogeneity in integration gain,
    which can dramatically inflate residual variance in pooled regression.

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data with one row per section
    condition_filter : str, optional
        If provided, filter to this condition
    speed_filter : float, optional
        If provided, filter to this speed threshold

    Returns
    -------
    results : dict
        Mixed effects model results including:
        - fixed_effect_beta: Population-level gain error
        - random_intercept_var: Between-animal variance
        - random_slope_var: Between-animal slope variance (if fit)
        - pseudo_r_squared: Variance explained
    """
    from statsmodels.formula.api import mixedlm
    import warnings

    # Make a copy and add animal_id
    data = endpoints.copy()
    if 'animal_id' not in data.columns:
        # Extract from session or trial_id
        if 'session' in data.columns:
            data['animal_id'] = data['session'].apply(extract_animal_id)
        elif 'trial_id' in data.columns:
            data['animal_id'] = data['trial_id'].apply(lambda x: extract_animal_id(x.split('_')[0]))
        else:
            return {'error': 'No session identifier found'}

    # Apply filters
    if condition_filter is not None:
        data = data[data['condition'] == condition_filter]
    if speed_filter is not None:
        data = data[data['speed_threshold'] == speed_filter]

    # Remove NaN values
    data = data.dropna(subset=['integrated_ang_vel', 'mvtDirError'])

    if len(data) < 20:
        return {'error': f'Insufficient data: n={len(data)}'}

    n_animals = data['animal_id'].nunique()
    if n_animals < 2:
        return {'error': f'Need at least 2 animals, found {n_animals}'}

    results = {
        'n_sections': len(data),
        'n_animals': n_animals,
        'condition': condition_filter,
        'speed_threshold': speed_filter
    }

    # Compute null model variance for pseudo-R²
    null_var = data['mvtDirError'].var()
    results['total_variance'] = null_var

    # Simple OLS for comparison
    slope, intercept, r, p, se = stats.linregress(
        data['integrated_ang_vel'].values,
        data['mvtDirError'].values
    )
    results['ols_beta'] = slope
    results['ols_r_squared'] = r**2
    results['ols_p'] = p

    # Random intercept model
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_ri = mixedlm(
                "mvtDirError ~ integrated_ang_vel",
                data,
                groups=data["animal_id"]
            ).fit(method='powell')

        results['ri_fixed_beta'] = model_ri.fe_params['integrated_ang_vel']
        results['ri_fixed_intercept'] = model_ri.fe_params['Intercept']
        results['ri_fixed_p'] = model_ri.pvalues['integrated_ang_vel']
        results['ri_random_intercept_var'] = float(model_ri.cov_re.iloc[0, 0])
        results['ri_residual_var'] = model_ri.scale

        # Pseudo-R² (variance reduction)
        results['ri_pseudo_r_squared'] = 1 - (model_ri.scale / null_var)

        # Intraclass correlation (ICC): proportion of variance due to animals
        results['ri_icc'] = results['ri_random_intercept_var'] / (
            results['ri_random_intercept_var'] + results['ri_residual_var']
        )

    except Exception as e:
        results['ri_error'] = str(e)

    # Random slope model (animal-specific gains)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_rs = mixedlm(
                "mvtDirError ~ integrated_ang_vel",
                data,
                groups=data["animal_id"],
                re_formula="~integrated_ang_vel"
            ).fit(method='powell')

        results['rs_fixed_beta'] = model_rs.fe_params['integrated_ang_vel']
        results['rs_fixed_intercept'] = model_rs.fe_params['Intercept']
        results['rs_fixed_p'] = model_rs.pvalues['integrated_ang_vel']
        results['rs_residual_var'] = model_rs.scale

        # Extract random effects covariance matrix
        cov_re = model_rs.cov_re
        results['rs_random_intercept_var'] = float(cov_re.iloc[0, 0])
        if cov_re.shape[0] > 1:
            results['rs_random_slope_var'] = float(cov_re.iloc[1, 1])
            results['rs_random_cov'] = float(cov_re.iloc[0, 1])

        # Pseudo-R²
        results['rs_pseudo_r_squared'] = 1 - (model_rs.scale / null_var)

        # Log-likelihood for model comparison
        results['ri_llf'] = model_ri.llf if 'ri_fixed_beta' in results else np.nan
        results['rs_llf'] = model_rs.llf

    except Exception as e:
        results['rs_error'] = str(e)

    return results


def per_animal_regression(endpoints, min_sections=10):
    """
    Run regression separately for each animal.

    If animals have heterogeneous integration gains, per-animal R² should
    be substantially higher than pooled R².

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data with one row per section
    min_sections : int
        Minimum number of sections required per animal

    Returns
    -------
    results_df : DataFrame
        Per-animal regression results
    """
    # Make a copy and add animal_id
    data = endpoints.copy()
    if 'animal_id' not in data.columns:
        if 'session' in data.columns:
            data['animal_id'] = data['session'].apply(extract_animal_id)
        elif 'trial_id' in data.columns:
            data['animal_id'] = data['trial_id'].apply(lambda x: extract_animal_id(x.split('_')[0]))
        else:
            return pd.DataFrame()

    results = []

    for animal_id in data['animal_id'].unique():
        animal_data = data[data['animal_id'] == animal_id]

        X = animal_data['integrated_ang_vel'].values
        Y = animal_data['mvtDirError'].values
        valid = ~(np.isnan(X) | np.isnan(Y))
        X, Y = X[valid], Y[valid]

        if len(X) >= min_sections:
            slope, intercept, r, p, se = stats.linregress(X, Y)
            results.append({
                'animal_id': animal_id,
                'n_sections': len(X),
                'beta': slope,
                'intercept': intercept,
                'r_squared': r**2,
                'r': r,
                'p_value': p,
                'se': se,
                'x_mean': np.mean(X),
                'x_std': np.std(X),
                'y_mean': np.mean(Y),
                'y_std': np.std(Y)
            })

    return pd.DataFrame(results)


def per_animal_regression_by_condition(endpoints, conditions, speed_thresholds, min_sections=10):
    """
    Run per-animal regression separately for each condition and speed threshold.

    This enables creation of per-animal figures for each condition and
    cross-animal forest plots comparing beta estimates.

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data with integrated_ang_vel, mvtDirError, condition, speed_threshold
    conditions : list
        List of condition names
    speed_thresholds : list
        List of speed thresholds
    min_sections : int
        Minimum sections required per animal/condition/speed combination

    Returns
    -------
    results_df : DataFrame
        Per-animal-condition regression results with columns:
        animal_id, condition, speed_threshold, n_sections, beta, se,
        ci_lower, ci_upper, r_squared, p_value
    """
    from scipy.stats import t as t_dist

    # Add animal_id if not present
    data = endpoints.copy()
    if 'animal_id' not in data.columns:
        if 'session' in data.columns:
            data['animal_id'] = data['session'].apply(extract_animal_id)
        elif 'trial_id' in data.columns:
            data['animal_id'] = data['trial_id'].apply(
                lambda x: extract_animal_id(x.split('_')[0]))
        else:
            print("Warning: Cannot extract animal_id from data")
            return pd.DataFrame()

    results = []

    for condition in conditions:
        for speed in speed_thresholds:
            # Filter to this condition/speed
            mask = (data['condition'] == condition) & (data['speed_threshold'] == speed)
            cond_data = data[mask]

            if len(cond_data) == 0:
                continue

            for animal_id in cond_data['animal_id'].unique():
                animal_data = cond_data[cond_data['animal_id'] == animal_id]

                X = animal_data['integrated_ang_vel'].values
                Y = animal_data['mvtDirError'].values
                valid = ~(np.isnan(X) | np.isnan(Y))
                X, Y = X[valid], Y[valid]

                if len(X) >= min_sections:
                    slope, intercept, r, p, se = stats.linregress(X, Y)

                    # Compute analytical 95% CI
                    df = len(X) - 2
                    if df > 0:
                        t_crit = t_dist.ppf(0.975, df)
                        ci_lower = slope - t_crit * se
                        ci_upper = slope + t_crit * se
                    else:
                        ci_lower = np.nan
                        ci_upper = np.nan

                    results.append({
                        'animal_id': animal_id,
                        'condition': condition,
                        'speed_threshold': speed,
                        'n_sections': len(X),
                        'beta': slope,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'r_squared': r**2,
                        'r': r,
                        'p_value': p,
                        'intercept': intercept
                    })

    return pd.DataFrame(results)


def compute_model_comparison(endpoints, condition_filter=None, speed_filter=None):
    """
    Compare R² across different modeling approaches.

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data
    condition_filter : str, optional
        Filter to specific condition
    speed_filter : float, optional
        Filter to specific speed threshold

    Returns
    -------
    comparison : dict
        R² values from different approaches
    """
    data = endpoints.copy()

    # Apply filters
    if condition_filter is not None:
        data = data[data['condition'] == condition_filter]
    if speed_filter is not None:
        data = data[data['speed_threshold'] == speed_filter]

    # Add animal_id
    if 'animal_id' not in data.columns:
        if 'session' in data.columns:
            data['animal_id'] = data['session'].apply(extract_animal_id)
        elif 'trial_id' in data.columns:
            data['animal_id'] = data['trial_id'].apply(lambda x: extract_animal_id(x.split('_')[0]))

    comparison = {
        'condition': condition_filter,
        'speed_threshold': speed_filter,
        'n_sections': len(data),
        'n_animals': data['animal_id'].nunique() if 'animal_id' in data.columns else np.nan
    }

    # Remove NaN
    data = data.dropna(subset=['integrated_ang_vel', 'mvtDirError'])
    comparison['n_valid'] = len(data)

    if len(data) < 10:
        return comparison

    X = data['integrated_ang_vel'].values
    Y = data['mvtDirError'].values

    # 1. Simple OLS (pooled)
    slope, intercept, r, p, se = stats.linregress(X, Y)
    comparison['pooled_r_squared'] = r**2
    comparison['pooled_beta'] = slope
    comparison['pooled_p'] = p

    # 2. Per-animal mean R²
    per_animal = per_animal_regression(data, min_sections=5)
    if len(per_animal) > 0:
        comparison['per_animal_mean_r_squared'] = per_animal['r_squared'].mean()
        comparison['per_animal_median_r_squared'] = per_animal['r_squared'].median()
        comparison['per_animal_max_r_squared'] = per_animal['r_squared'].max()
        comparison['per_animal_n_animals'] = len(per_animal)

    # 3. Mixed effects
    mixed_results = mixed_effects_regression(data)
    if 'ri_pseudo_r_squared' in mixed_results:
        comparison['mixed_ri_pseudo_r_squared'] = mixed_results['ri_pseudo_r_squared']
        comparison['mixed_ri_icc'] = mixed_results.get('ri_icc', np.nan)
    if 'rs_pseudo_r_squared' in mixed_results:
        comparison['mixed_rs_pseudo_r_squared'] = mixed_results['rs_pseudo_r_squared']

    return comparison


# =============================================================================
# STATISTICAL VALIDATION FUNCTIONS
# =============================================================================

def bootstrap_regression_ci(X, Y, n_bootstrap=1000, ci=95, random_state=42):
    """
    Compute bootstrap confidence intervals for regression parameters.

    Parameters
    ----------
    X : array-like
        Predictor variable (cumulative turn)
    Y : array-like
        Response variable (heading deviation)
    n_bootstrap : int
        Number of bootstrap samples
    ci : int
        Confidence interval percentage (e.g., 95)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        'beta_mean', 'beta_ci_low', 'beta_ci_high',
        'r_squared_mean', 'r_squared_ci_low', 'r_squared_ci_high',
        'intercept_mean', 'intercept_ci_low', 'intercept_ci_high'
    """
    np.random.seed(random_state)
    n = len(X)

    if n < 5:
        return {
            'beta_mean': np.nan, 'beta_ci_low': np.nan, 'beta_ci_high': np.nan,
            'r_squared_mean': np.nan, 'r_squared_ci_low': np.nan, 'r_squared_ci_high': np.nan,
            'intercept_mean': np.nan, 'intercept_ci_low': np.nan, 'intercept_ci_high': np.nan
        }

    betas = []
    r_squareds = []
    intercepts = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        Y_boot = Y[idx]

        # Fit regression
        try:
            slope, intercept, r, p, se = stats.linregress(X_boot, Y_boot)
            betas.append(slope)
            r_squareds.append(r ** 2)
            intercepts.append(intercept)
        except:
            continue

    if len(betas) < n_bootstrap * 0.5:
        return {
            'beta_mean': np.nan, 'beta_ci_low': np.nan, 'beta_ci_high': np.nan,
            'r_squared_mean': np.nan, 'r_squared_ci_low': np.nan, 'r_squared_ci_high': np.nan,
            'intercept_mean': np.nan, 'intercept_ci_low': np.nan, 'intercept_ci_high': np.nan
        }

    betas = np.array(betas)
    r_squareds = np.array(r_squareds)
    intercepts = np.array(intercepts)

    alpha = (100 - ci) / 2
    return {
        'beta_mean': np.mean(betas),
        'beta_ci_low': np.percentile(betas, alpha),
        'beta_ci_high': np.percentile(betas, 100 - alpha),
        'r_squared_mean': np.mean(r_squareds),
        'r_squared_ci_low': np.percentile(r_squareds, alpha),
        'r_squared_ci_high': np.percentile(r_squareds, 100 - alpha),
        'intercept_mean': np.mean(intercepts),
        'intercept_ci_low': np.percentile(intercepts, alpha),
        'intercept_ci_high': np.percentile(intercepts, 100 - alpha)
    }


def permutation_test_correlation(X, Y, n_permutations=10000, random_state=42):
    """
    Permutation test for significance of correlation.

    Shuffles Y values to create null distribution and computes the proportion
    of permuted correlations with absolute value >= observed.

    Parameters
    ----------
    X : array-like
        Predictor variable
    Y : array-like
        Response variable
    n_permutations : int
        Number of permutations
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        'observed_r': observed correlation
        'perm_p_value': permutation p-value (two-tailed)
        'perm_p_value_one_tail': one-tailed p-value
    """
    np.random.seed(random_state)

    if len(X) < 5:
        return {'observed_r': np.nan, 'perm_p_value': np.nan, 'perm_p_value_one_tail': np.nan}

    # Observed correlation
    observed_r, _ = pearsonr(X, Y)

    # Generate null distribution
    null_rs = []
    for _ in range(n_permutations):
        Y_shuffled = np.random.permutation(Y)
        r_perm, _ = pearsonr(X, Y_shuffled)
        null_rs.append(r_perm)

    null_rs = np.array(null_rs)

    # Two-tailed p-value: proportion of |null_r| >= |observed_r|
    perm_p_value = np.mean(np.abs(null_rs) >= np.abs(observed_r))

    # One-tailed p-value: proportion of null_r in same direction and >= observed
    if observed_r >= 0:
        perm_p_one_tail = np.mean(null_rs >= observed_r)
    else:
        perm_p_one_tail = np.mean(null_rs <= observed_r)

    return {
        'observed_r': observed_r,
        'perm_p_value': perm_p_value,
        'perm_p_value_one_tail': perm_p_one_tail
    }


def per_trial_regression(endpoints, condition='all_dark', speed_threshold=2.0, min_sections=5):
    """
    Run regression for each trial separately and aggregate results.

    This approach addresses trial-level clustering - each trial is an independent
    "experiment" and treating sections as independent ignores this structure.

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data with one row per section
    condition : str
        Condition to analyze (e.g., 'all_dark')
    speed_threshold : float
        Speed threshold (e.g., 2.0 cm/s)
    min_sections : int
        Minimum sections required per trial to include in analysis

    Returns
    -------
    trial_results : DataFrame
        Per-trial regression results (beta, R², n_sections for each trial)
    aggregate_stats : dict
        Aggregate statistics including:
        - mean_beta, se_beta, ci_low, ci_high
        - one_sample_t, one_sample_p (test if mean β differs from 0)
        - n_trials, n_positive_beta, n_negative_beta
    """
    # Filter to condition and speed
    mask = (endpoints['condition'] == condition) & (endpoints['speed_threshold'] == speed_threshold)
    data = endpoints[mask].copy()

    if len(data) == 0:
        return pd.DataFrame(), {'error': 'No data for this condition/speed'}

    # Count sections per trial
    sections_per_trial = data.groupby('trial_id').size()
    valid_trials = sections_per_trial[sections_per_trial >= min_sections].index

    if len(valid_trials) == 0:
        return pd.DataFrame(), {'error': f'No trials with >= {min_sections} sections'}

    # Run regression for each trial
    trial_results = []

    for trial_id in valid_trials:
        trial_data = data[data['trial_id'] == trial_id]

        X = trial_data['integrated_ang_vel'].values
        Y = trial_data['mvtDirError'].values

        valid = ~(np.isnan(X) | np.isnan(Y))
        X_valid, Y_valid = X[valid], Y[valid]

        if len(X_valid) >= min_sections:
            slope, intercept, r, p, se = stats.linregress(X_valid, Y_valid)

            trial_results.append({
                'trial_id': trial_id,
                'n_sections': len(X_valid),
                'beta': slope,
                'intercept': intercept,
                'r_squared': r**2,
                'r': r,
                'p_value': p,
                'se': se,
                'x_mean': np.mean(X_valid),
                'x_std': np.std(X_valid),
                'y_mean': np.mean(Y_valid),
                'y_std': np.std(Y_valid)
            })

    trial_results_df = pd.DataFrame(trial_results)

    if len(trial_results_df) == 0:
        return trial_results_df, {'error': 'No valid trials after filtering'}

    # Aggregate statistics
    betas = trial_results_df['beta'].values

    # One-sample t-test: H₀: mean(β) = 0
    t_stat, t_pvalue = stats.ttest_1samp(betas, 0)

    # Bootstrap CI for mean beta
    np.random.seed(42)
    n_bootstrap = 1000
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(betas, size=len(betas), replace=True)
        boot_means.append(np.mean(boot_sample))
    boot_means = np.array(boot_means)

    aggregate_stats = {
        'condition': condition,
        'speed_threshold': speed_threshold,
        'min_sections': min_sections,
        'n_trials': len(trial_results_df),
        'n_total_sections': trial_results_df['n_sections'].sum(),
        'mean_beta': np.mean(betas),
        'median_beta': np.median(betas),
        'se_beta': np.std(betas) / np.sqrt(len(betas)),
        'std_beta': np.std(betas),
        'ci_low': np.percentile(boot_means, 2.5),
        'ci_high': np.percentile(boot_means, 97.5),
        'one_sample_t': t_stat,
        'one_sample_p': t_pvalue,
        'n_positive_beta': np.sum(betas > 0),
        'n_negative_beta': np.sum(betas < 0),
        'prop_positive': np.mean(betas > 0),
        'mean_r_squared': trial_results_df['r_squared'].mean(),
        'median_r_squared': trial_results_df['r_squared'].median()
    }

    return trial_results_df, aggregate_stats


def validate_sparse_conditions(endpoints, n_bootstrap=1000, n_permutations=10000):
    """
    Run validation on sparse high-R^2 conditions.

    Parameters
    ----------
    endpoints : DataFrame
        Endpoint data
    n_bootstrap : int
        Number of bootstrap samples
    n_permutations : int
        Number of permutations for permutation test

    Returns
    -------
    validation_df : DataFrame
        Validation results for each condition
    """
    # Conditions to validate (high R^2 or scientifically important)
    conditions_to_validate = [
        ('searchToLeverPath_dark', 5.0),   # Highest R^2 (n=41)
        ('searchToLeverPath_dark', 3.0),   # More data (n=285)
        ('atLever_dark', 5.0),             # Second highest R^2 (n=90)
        ('homingFromLeavingLever_light', 5.0),  # Light condition (n=230)
        ('homingFromLeavingLever_dark', 3.0),   # Dark condition (n=258)
        ('all_dark', 5.0),                 # All dark (n=211)
        ('all_light', 5.0),                # All light for comparison
    ]

    results = []

    for condition, speed in conditions_to_validate:
        print(f"\nValidating: {condition} @ {speed} cm/s")

        # Filter data
        mask = (endpoints['condition'] == condition) & (endpoints['speed_threshold'] == speed)
        data = endpoints[mask].dropna(subset=['integrated_ang_vel', 'mvtDirError'])

        n = len(data)
        print(f"  n = {n}")

        if n < 5:
            print("  Skipping - insufficient data")
            results.append({
                'condition': condition,
                'speed_threshold': speed,
                'n': n,
                'beta': np.nan,
                'beta_ci_low': np.nan,
                'beta_ci_high': np.nan,
                'r_squared': np.nan,
                'r_squared_ci_low': np.nan,
                'r_squared_ci_high': np.nan,
                'perm_p_value': np.nan,
                'ci_excludes_zero': np.nan
            })
            continue

        X = data['integrated_ang_vel'].values
        Y = data['mvtDirError'].values

        # Point estimates
        slope, intercept, r, p, se = stats.linregress(X, Y)
        r_squared = r ** 2

        # Bootstrap CIs
        print(f"  Running bootstrap ({n_bootstrap} samples)...")
        boot_results = bootstrap_regression_ci(X, Y, n_bootstrap=n_bootstrap)

        # Permutation test
        print(f"  Running permutation test ({n_permutations} permutations)...")
        perm_results = permutation_test_correlation(X, Y, n_permutations=n_permutations)

        # Check if CI excludes zero
        ci_excludes_zero = (boot_results['beta_ci_low'] > 0) or (boot_results['beta_ci_high'] < 0)

        results.append({
            'condition': condition,
            'speed_threshold': speed,
            'n': n,
            'beta': slope,
            'beta_se': se,
            'beta_ci_low': boot_results['beta_ci_low'],
            'beta_ci_high': boot_results['beta_ci_high'],
            'r_squared': r_squared,
            'r_squared_ci_low': boot_results['r_squared_ci_low'],
            'r_squared_ci_high': boot_results['r_squared_ci_high'],
            'parametric_p': p,
            'perm_p_value': perm_results['perm_p_value'],
            'ci_excludes_zero': ci_excludes_zero
        })

        print(f"  beta = {slope:.4f} [{boot_results['beta_ci_low']:.4f}, {boot_results['beta_ci_high']:.4f}]")
        print(f"  R^2 = {r_squared:.4f} [{boot_results['r_squared_ci_low']:.4f}, {boot_results['r_squared_ci_high']:.4f}]")
        print(f"  Permutation p = {perm_results['perm_p_value']:.4f}")
        print(f"  CI excludes zero: {ci_excludes_zero}")

    return pd.DataFrame(results)


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_all_conditions(df, conditions, speed_thresholds, min_bout_length=3,
                           min_section_length=3, angular_velocity_threshold=0.005,
                           max_integrated=None):
    """
    Process all conditions and speed thresholds.

    Parameters
    ----------
    df : DataFrame
        Full reconstruction dataset
    conditions : list
        List of condition names to process
    speed_thresholds : list
        List of speed thresholds to test
    min_bout_length : int
        Minimum bout length
    min_section_length : int
        Minimum pure turn section length
    angular_velocity_threshold : float
        Near-zero threshold for angular velocity
    max_integrated : float, optional
        Maximum absolute integrated angular velocity for truncation

    Returns
    -------
    all_section_data : DataFrame
        Combined section data across all conditions (all timepoints)
    regression_results : DataFrame
        Regression statistics for each condition/threshold (all timepoints)
    all_endpoint_data : DataFrame
        Combined endpoint data across all conditions (one row per section)
    endpoint_regression_results : DataFrame
        Regression statistics for endpoints (one row per section)
    """
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
        print(f"Found {len(unique_trials)} trials")

        for speed_threshold in speed_thresholds:
            print(f"\n  Speed threshold: {speed_threshold} cm/s")

            section_data_list = []

            for trial_id in tqdm(unique_trials, desc=f"  Processing trials"):
                trial_data = condition_df[condition_df['session_trial'] == trial_id].copy()
                trial_data = trial_data.sort_values('recTime')

                # Process trial with pure turn sections
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
                print(f"    Left timepoints: {(combined['turn_direction'] == 'left').sum()}")
                print(f"    Right timepoints: {(combined['turn_direction'] == 'right').sum()}")

                # Regression by turn direction (old approach - all timepoints)
                reg_results_old = regression_by_turn_direction(combined, condition, speed_threshold)

                # Signed regression (new approach - all timepoints)
                reg_results_signed = signed_regression_analysis(combined, condition, speed_threshold)

                # Merge results
                reg_results = {**reg_results_old, **reg_results_signed}

                # Accumulation test
                accum_results = test_section_accumulation(combined)
                reg_results.update(accum_results)

                regression_results.append(reg_results)

                # ============================================================
                # ENDPOINT ANALYSIS (one data point per section)
                # ============================================================
                endpoints = extract_section_endpoints(combined)
                if len(endpoints) > 0:
                    all_endpoint_data.append(endpoints)

                    # Endpoint regression
                    endpoint_reg_results = regression_on_endpoints(
                        endpoints, condition, speed_threshold
                    )
                    endpoint_regression_results.append(endpoint_reg_results)

                    print(f"    ENDPOINT ANALYSIS (n={len(endpoints)} sections):")
                    if not np.isnan(endpoint_reg_results.get('beta_signed', np.nan)):
                        print(f"      beta_signed: {endpoint_reg_results['beta_signed']:.6f}, p={endpoint_reg_results['p_signed']:.4f}")
                        print(f"      R²={endpoint_reg_results['r_squared_signed']:.4f}, 95% CI=[{endpoint_reg_results['ci_lower_95']:.6f}, {endpoint_reg_results['ci_upper_95']:.6f}]")
                    else:
                        print(f"      Insufficient data for endpoint regression")

                print(f"    ALL TIMEPOINTS ANALYSIS:")
                print(f"      SPLIT BY DIRECTION:")
                print(f"        beta_left: {reg_results['beta_left']:.6f}, p={reg_results['p_left']:.4f}")
                print(f"        beta_right: {reg_results['beta_right']:.6f}, p={reg_results['p_right']:.4f}")
                print(f"        Asymmetry: beta_diff={reg_results['beta_diff']:.6f}, p={reg_results['p_asymmetry']:.4f}")
                print(f"      SIGNED REGRESSION:")
                print(f"        beta_signed: {reg_results['beta_signed']:.6f}, p={reg_results['p_signed']:.4f}")
                print(f"        R²={reg_results['r_squared_signed']:.4f}, 95% CI=[{reg_results['ci_lower_95']:.6f}, {reg_results['ci_upper_95']:.6f}]")

    # Combine results
    if len(all_section_data) > 0:
        all_section_data = pd.concat(all_section_data, ignore_index=True)
    else:
        all_section_data = pd.DataFrame()

    regression_results = pd.DataFrame(regression_results)

    # Combine endpoint results
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
    print("PURE TURN SECTIONS ANALYSIS (RELAXED FILTERING)")
    print("="*80)
    print(f"\nConfiguration (RELAXED PARAMETERS):")
    print(f"  Data path: {PROJECT_DATA_PATH}")
    print(f"  Speed thresholds: {SPEED_THRESHOLDS} cm/s")
    print(f"  Minimum bout length: {MIN_BOUT_LENGTH} timepoints (relaxed from 5)")
    print(f"  Minimum section length: {MIN_SECTION_LENGTH} timepoints (relaxed from 5)")
    print(f"  Angular velocity threshold: {ANGULAR_VELOCITY_THRESHOLD} rad/s (relaxed from 0.01)")
    print(f"  Max integrated threshold: {MAX_INTEGRATED_THRESHOLD} radians" if MAX_INTEGRATED_THRESHOLD is not None else "  Max integrated threshold: None")
    print(f"  Conditions: {len(CONDITIONS)}")
    print(f"  Sessions: {len(useAble)}")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    fn = os.path.join(PROJECT_DATA_PATH, "results", "reconstuctionDFAutoPI.csv")
    print(f"Loading: {fn}")
    dfAutoPI = pd.read_csv(fn)
    print(f"Loaded {len(dfAutoPI):,} rows")

    # Filter for useable sessions
    dfAutoPI = dfAutoPI[dfAutoPI.session.isin(useAble)]
    print(f"After filtering: {len(dfAutoPI):,} rows")
    print(f"Sessions: {dfAutoPI.session.nunique()}")

    # Process all conditions
    print("\n" + "="*80)
    print("PROCESSING ALL CONDITIONS")
    print("="*80)

    all_section_data, regression_results, all_endpoint_data, endpoint_regression_results = process_all_conditions(
        dfAutoPI,
        CONDITIONS,
        SPEED_THRESHOLDS,
        MIN_BOUT_LENGTH,
        MIN_SECTION_LENGTH,
        ANGULAR_VELOCITY_THRESHOLD,
        MAX_INTEGRATED_THRESHOLD
    )

    # Save results with _relaxed suffix
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_dir = os.path.join(PROJECT_DATA_PATH, "results")

    # Save all-timepoints data
    if len(all_section_data) > 0:
        section_data_fn = os.path.join(results_dir, "pure_turn_section_data_relaxed.csv")
        all_section_data.to_csv(section_data_fn, index=False)
        print(f"Saved section data (all timepoints): {section_data_fn}")
        print(f"  Total rows: {len(all_section_data):,}")

    if len(regression_results) > 0:
        reg_results_fn = os.path.join(results_dir, "pure_turn_section_regression_results_relaxed.csv")
        regression_results.to_csv(reg_results_fn, index=False)
        print(f"Saved regression results (all timepoints): {reg_results_fn}")
        print(f"  Total rows: {len(regression_results)}")

    # Save endpoint data (one row per section - fixes zero-clustering)
    if len(all_endpoint_data) > 0:
        endpoint_data_fn = os.path.join(results_dir, "pure_turn_section_endpoints_relaxed.csv")
        all_endpoint_data.to_csv(endpoint_data_fn, index=False)
        print(f"Saved endpoint data (one per section): {endpoint_data_fn}")
        print(f"  Total sections: {len(all_endpoint_data):,}")

    if len(endpoint_regression_results) > 0:
        endpoint_reg_fn = os.path.join(results_dir, "pure_turn_section_endpoints_regression_relaxed.csv")
        endpoint_regression_results.to_csv(endpoint_reg_fn, index=False)
        print(f"Saved endpoint regression results: {endpoint_reg_fn}")
        print(f"  Total rows: {len(endpoint_regression_results)}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    if len(regression_results) > 0:
        print("\n" + "="*80)
        print("SIGNED REGRESSION RESULTS")
        print("="*80)
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

        print("\n" + "="*80)
        print("SPLIT BY DIRECTION RESULTS")
        print("="*80)

        old_cols = ['condition', 'speed_threshold', 'n_left_sections', 'n_right_sections',
                   'beta_left', 'beta_right', 'beta_diff', 'p_asymmetry']
        available_old_cols = [col for col in old_cols if col in regression_results.columns]
        print(regression_results[available_old_cols].to_string(index=False))

        sig_asymmetries = (regression_results['p_asymmetry'] < 0.05).sum()
        total_asymmetry = len(regression_results[~regression_results['p_asymmetry'].isna()])
        print(f"\nSignificant asymmetries (p < 0.05): {sig_asymmetries}/{total_asymmetry}")

    # =========================================================================
    # ENDPOINT ANALYSIS SUMMARY (fixes zero-clustering issue)
    # =========================================================================
    if len(endpoint_regression_results) > 0:
        print("\n" + "="*80)
        print("ENDPOINT ANALYSIS RESULTS (ONE DATA POINT PER SECTION)")
        print("="*80)
        print("\nThis analysis uses only the FINAL timepoint of each section,")
        print("eliminating the zero-clustering issue where early timepoints")
        print("in each section have near-zero cumulative turn values.\n")

        endpoint_cols = ['condition', 'speed_threshold', 'n_sections', 'beta_signed',
                        'r_squared_signed', 'p_signed', 'ci_lower_95', 'ci_upper_95']
        available_endpoint_cols = [col for col in endpoint_cols if col in endpoint_regression_results.columns]
        print(endpoint_regression_results[available_endpoint_cols].to_string(index=False))

        sig_endpoint = (endpoint_regression_results['p_signed'] < 0.05).sum()
        total_endpoint = len(endpoint_regression_results[~endpoint_regression_results['p_signed'].isna()])
        print(f"\nSignificant signed relationships (p < 0.05): {sig_endpoint}/{total_endpoint}")

        negative_endpoint_betas = (endpoint_regression_results['beta_signed'] < 0).sum()
        print(f"Negative beta (expected direction): {negative_endpoint_betas}/{total_endpoint}")

        # Compare with all-timepoints analysis
        print("\n" + "-"*60)
        print("COMPARISON: Endpoints vs All Timepoints")
        print("-"*60)
        if len(regression_results) > 0:
            for _, ep_row in endpoint_regression_results.iterrows():
                condition = ep_row['condition']
                speed = ep_row['speed_threshold']
                tp_row = regression_results[
                    (regression_results['condition'] == condition) &
                    (regression_results['speed_threshold'] == speed)
                ]
                if len(tp_row) > 0:
                    tp_row = tp_row.iloc[0]
                    print(f"\n{condition}, speed >= {speed} cm/s:")
                    print(f"  Endpoints:       beta={ep_row['beta_signed']:.4f}, R^2={ep_row['r_squared_signed']:.4f}, n={ep_row['n_sections']}")
                    print(f"  All timepoints:  beta={tp_row['beta_signed']:.4f}, R^2={tp_row['r_squared_signed']:.4f}, n={tp_row['n_total']}")

    # =========================================================================
    # ADVANCED R² IMPROVEMENT ANALYSIS
    # =========================================================================
    if len(all_endpoint_data) > 0:
        print("\n" + "="*80)
        print("ADVANCED R^2 IMPROVEMENT ANALYSIS")
        print("="*80)
        print("\nTesting strategies to legitimately improve R^2 values:")
        print("1. Per-animal stratified regression")
        print("2. Mixed effects model (random intercept)")
        print("3. Mixed effects model (random slope)")

        # Add animal_id to endpoint data
        all_endpoint_data['animal_id'] = all_endpoint_data['trial_id'].apply(
            lambda x: extract_animal_id(x.split('_')[0])
        )

        # Per-animal analysis across all data
        print("\n" + "-"*60)
        print("PER-ANIMAL REGRESSION RESULTS")
        print("-"*60)

        per_animal_results = per_animal_regression(all_endpoint_data, min_sections=20)
        if len(per_animal_results) > 0:
            print("\n  Per-animal R^2 values:")
            for _, row in per_animal_results.iterrows():
                sig = "*" if row['p_value'] < 0.05 else " "
                print(f"    {row['animal_id']}: R^2={row['r_squared']:.4f}, beta={row['beta']:.4f}, n={row['n_sections']}{sig}")

            print(f"\n  Summary:")
            print(f"    Mean per-animal R^2:   {per_animal_results['r_squared'].mean():.4f}")
            print(f"    Median per-animal R^2: {per_animal_results['r_squared'].median():.4f}")
            print(f"    Max per-animal R^2:    {per_animal_results['r_squared'].max():.4f}")

            # Save per-animal results
            per_animal_fn = os.path.join(results_dir, "pure_turn_section_per_animal_relaxed.csv")
            per_animal_results.to_csv(per_animal_fn, index=False)
            print(f"\n  Saved to: {per_animal_fn}")

        # Per-animal regression by condition (for per-animal figures and forest plots)
        print("\n" + "-"*60)
        print("PER-ANIMAL REGRESSION BY CONDITION")
        print("-"*60)

        per_animal_by_condition = per_animal_regression_by_condition(
            all_endpoint_data, CONDITIONS, SPEED_THRESHOLDS, min_sections=5
        )

        if len(per_animal_by_condition) > 0:
            per_animal_cond_fn = os.path.join(results_dir, "pure_turn_section_per_animal_by_condition_relaxed.csv")
            per_animal_by_condition.to_csv(per_animal_cond_fn, index=False)
            print(f"\n  Saved to: {per_animal_cond_fn}")
            print(f"  Total rows: {len(per_animal_by_condition)}")
            print(f"  Animals: {per_animal_by_condition['animal_id'].nunique()}")
            print(f"  Conditions: {per_animal_by_condition['condition'].nunique()}")

            # Summary by animal
            print("\n  Rows per animal:")
            for animal_id in sorted(per_animal_by_condition['animal_id'].unique()):
                n_rows = len(per_animal_by_condition[per_animal_by_condition['animal_id'] == animal_id])
                print(f"    {animal_id}: {n_rows} condition-speed combinations")
        else:
            print("\n  Warning: No per-animal-by-condition results generated")

        # Mixed effects analysis across all data
        print("\n" + "-"*60)
        print("MIXED EFFECTS MODEL RESULTS (ALL DATA POOLED)")
        print("-"*60)

        mixed_results_all = mixed_effects_regression(all_endpoint_data)
        if 'error' not in mixed_results_all:
            print(f"\n  Data: n={mixed_results_all['n_sections']} sections, {mixed_results_all['n_animals']} animals")
            print(f"\n  OLS (pooled) baseline:")
            print(f"    beta = {mixed_results_all['ols_beta']:.6f}")
            print(f"    R^2 = {mixed_results_all['ols_r_squared']:.6f}")

            if 'ri_fixed_beta' in mixed_results_all:
                print(f"\n  Random Intercept Model:")
                print(f"    Fixed effect beta = {mixed_results_all['ri_fixed_beta']:.6f} (p={mixed_results_all['ri_fixed_p']:.4f})")
                print(f"    Random intercept var = {mixed_results_all['ri_random_intercept_var']:.6f}")
                print(f"    Residual var = {mixed_results_all['ri_residual_var']:.6f}")
                print(f"    ICC = {mixed_results_all['ri_icc']:.4f} ({mixed_results_all['ri_icc']*100:.1f}% of variance from animals)")
                print(f"    Pseudo-R^2 = {mixed_results_all['ri_pseudo_r_squared']:.6f}")

            if 'rs_fixed_beta' in mixed_results_all:
                print(f"\n  Random Slope Model (animal-specific gains):")
                print(f"    Fixed effect beta = {mixed_results_all['rs_fixed_beta']:.6f} (p={mixed_results_all['rs_fixed_p']:.4f})")
                print(f"    Random intercept var = {mixed_results_all.get('rs_random_intercept_var', 'N/A')}")
                print(f"    Random slope var = {mixed_results_all.get('rs_random_slope_var', 'N/A')}")
                print(f"    Pseudo-R^2 = {mixed_results_all['rs_pseudo_r_squared']:.6f}")
        else:
            print(f"\n  Error: {mixed_results_all['error']}")

        # Model comparison by condition
        print("\n" + "-"*60)
        print("MODEL COMPARISON BY CONDITION")
        print("-"*60)

        model_comparisons = []
        for condition in CONDITIONS:
            for speed in SPEED_THRESHOLDS:
                comparison = compute_model_comparison(
                    all_endpoint_data,
                    condition_filter=condition,
                    speed_filter=speed
                )
                if comparison.get('n_valid', 0) > 10:
                    model_comparisons.append(comparison)

        if len(model_comparisons) > 0:
            comparison_df = pd.DataFrame(model_comparisons)

            print("\n  R^2 Comparison (pooled vs per-animal mean vs mixed effects):")
            print(f"  {'Condition':<30} {'Speed':>6} {'Pooled':>8} {'PerAnim':>8} {'Mixed':>8} {'ICC':>6}")
            print("  " + "-"*76)

            for _, row in comparison_df.iterrows():
                cond = row['condition'][:28] if len(str(row['condition'])) > 28 else row['condition']
                pooled = f"{row.get('pooled_r_squared', np.nan):.4f}" if not np.isnan(row.get('pooled_r_squared', np.nan)) else "N/A"
                per_anim = f"{row.get('per_animal_mean_r_squared', np.nan):.4f}" if not np.isnan(row.get('per_animal_mean_r_squared', np.nan)) else "N/A"
                mixed = f"{row.get('mixed_ri_pseudo_r_squared', np.nan):.4f}" if not np.isnan(row.get('mixed_ri_pseudo_r_squared', np.nan)) else "N/A"
                icc = f"{row.get('mixed_ri_icc', np.nan):.2f}" if not np.isnan(row.get('mixed_ri_icc', np.nan)) else "N/A"
                print(f"  {cond:<30} {row['speed_threshold']:>6.1f} {pooled:>8} {per_anim:>8} {mixed:>8} {icc:>6}")

            # Save model comparison
            comparison_fn = os.path.join(results_dir, "pure_turn_section_model_comparison_relaxed.csv")
            comparison_df.to_csv(comparison_fn, index=False)
            print(f"\n  Saved to: {comparison_fn}")

            # Summary statistics
            print("\n" + "-"*60)
            print("R^2 IMPROVEMENT SUMMARY")
            print("-"*60)

            pooled_mean = comparison_df['pooled_r_squared'].mean()
            per_animal_mean = comparison_df['per_animal_mean_r_squared'].mean()
            mixed_mean = comparison_df['mixed_ri_pseudo_r_squared'].mean()

            print(f"\n  Mean R^2 across conditions/speeds:")
            print(f"    Pooled OLS:      {pooled_mean:.6f}")
            print(f"    Per-animal mean: {per_animal_mean:.6f} ({per_animal_mean/pooled_mean:.1f}x improvement)")
            print(f"    Mixed effects:   {mixed_mean:.6f} ({mixed_mean/pooled_mean:.1f}x improvement)")

            # Save mixed effects results for all data
            mixed_results_fn = os.path.join(results_dir, "pure_turn_section_mixed_effects_relaxed.csv")
            pd.DataFrame([mixed_results_all]).to_csv(mixed_results_fn, index=False)
            print(f"\n  Mixed effects results saved to: {mixed_results_fn}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE (RELAXED FILTERING)")
    print("="*80)
    print(f"\nRelaxed parameters compared to original:")
    print(f"  - Speed thresholds: [1.0, 2.0, 3.0, 5.0] (was [1.0, 2.0, 5.0, 10.0])")
    print(f"  - Min bout length: 3 (was 5)")
    print(f"  - Min section length: 3 (was 5)")
    print(f"  - Angular velocity threshold: 0.005 rad/s (was 0.01)")
    print(f"\nThis should capture significantly more data points at each speed threshold.")
    print(f"\nOutput files:")
    print(f"  - pure_turn_section_endpoints_relaxed.csv (one row per section)")
    print(f"  - pure_turn_section_endpoints_regression_relaxed.csv (regression stats)")
    print(f"  - pure_turn_section_per_animal_relaxed.csv (per-animal regression)")
    print(f"  - pure_turn_section_per_animal_by_condition_relaxed.csv (per-animal by condition)")
    print(f"  - pure_turn_section_mixed_effects_relaxed.csv (mixed effects model)")
    print(f"  - pure_turn_section_model_comparison_relaxed.csv (R^2 comparison)")
