"""
Pure Turn Sections Analysis: Testing for Systematic Integration Errors

This script tests whether heading deviation results from systematic gain errors when
animals integrate angular velocity during PURE turning sections (continuous periods
where instantaneous angular velocity maintains the same sign).

Key Modification from integrated_angular_velocity_bouts_analysis.py:
- After identifying movement bouts, further split into "pure turn sections"
- Pure turn sections are continuous periods where angular velocity doesn't change sign
- Integration resets at each section start
- Near-zero angular velocities are excluded to avoid noise-induced section breaks

This approach prevents confounds where:
- Cumulative turn is near zero due to back-and-forth turning
- Many data points cluster near zero integrated angular velocity

Author: Analysis generated for Peng et al. 2025
Date: 2025-11-26
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

# Analysis parameters
SPEED_THRESHOLDS = [1.0, 2.0, 5.0, 10.0]  # cm/s for bout identification
MIN_BOUT_LENGTH = 5  # timepoints for initial bout identification
MIN_SECTION_LENGTH = 5  # timepoints for pure turn sections (configurable)
ANGULAR_VELOCITY_THRESHOLD = 0.01  # rad/s - exclude near-zero angular velocities
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

def process_trial_with_pure_sections(trial_data, speed_threshold, min_bout_length=5, 
                                      min_section_length=5, angular_velocity_threshold=0.01,
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
    """
    Perform regression analysis on section endpoints only.

    Tests: Does total accumulated turn predict heading error at section end?
    """
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

        from scipy.stats import t as t_dist
        df = len(X_valid) - 2
        t_crit = t_dist.ppf(0.975, df)
        results['ci_lower_95'] = slope - t_crit * se
        results['ci_upper_95'] = slope + t_crit * se

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
            slope_left, _, _, p_left, se_left = stats.linregress(X_left, Y_left)
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
            slope_right, _, _, p_right, se_right = stats.linregress(X_right, Y_right)
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
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_all_conditions(df, conditions, speed_thresholds, min_bout_length=5,
                           min_section_length=5, angular_velocity_threshold=0.01,
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
    print("PURE TURN SECTIONS ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data path: {PROJECT_DATA_PATH}")
    print(f"  Speed thresholds: {SPEED_THRESHOLDS} cm/s")
    print(f"  Minimum bout length: {MIN_BOUT_LENGTH} timepoints")
    print(f"  Minimum section length: {MIN_SECTION_LENGTH} timepoints")
    print(f"  Angular velocity threshold: {ANGULAR_VELOCITY_THRESHOLD} rad/s")
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

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_dir = os.path.join(PROJECT_DATA_PATH, "results")

    # Save all-timepoints data
    if len(all_section_data) > 0:
        section_data_fn = os.path.join(results_dir, "pure_turn_section_data.csv")
        all_section_data.to_csv(section_data_fn, index=False)
        print(f"Saved section data (all timepoints): {section_data_fn}")
        print(f"  Total rows: {len(all_section_data):,}")

    if len(regression_results) > 0:
        reg_results_fn = os.path.join(results_dir, "pure_turn_section_regression_results.csv")
        regression_results.to_csv(reg_results_fn, index=False)
        print(f"Saved regression results (all timepoints): {reg_results_fn}")
        print(f"  Total rows: {len(regression_results)}")

    # Save endpoint data (one row per section - fixes zero-clustering)
    if len(all_endpoint_data) > 0:
        endpoint_data_fn = os.path.join(results_dir, "pure_turn_section_endpoints.csv")
        all_endpoint_data.to_csv(endpoint_data_fn, index=False)
        print(f"Saved endpoint data (one per section): {endpoint_data_fn}")
        print(f"  Total sections: {len(all_endpoint_data):,}")

    if len(endpoint_regression_results) > 0:
        endpoint_reg_fn = os.path.join(results_dir, "pure_turn_section_endpoints_regression.csv")
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
        print("Expected: β < 0 (accumulated left turns → rightward heading error)\n")
        
        signed_cols = ['condition', 'speed_threshold', 'n_sections', 'beta_signed', 
                      'r_squared_signed', 'p_signed', 'ci_lower_95', 'ci_upper_95']
        available_cols = [col for col in signed_cols if col in regression_results.columns]
        print(regression_results[available_cols].to_string(index=False))
        
        sig_signed = (regression_results['p_signed'] < 0.05).sum()
        total_signed = len(regression_results[~regression_results['p_signed'].isna()])
        print(f"\nSignificant signed relationships (p < 0.05): {sig_signed}/{total_signed}")
        
        negative_betas = (regression_results['beta_signed'] < 0).sum()
        print(f"Negative β (expected direction): {negative_betas}/{total_signed}")
        
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
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey differences from bout-based analysis:")
    print(f"  1. Data split into pure turn sections (no sign changes in angular velocity)")
    print(f"  2. Integration resets at each section start")
    print(f"  3. Near-zero angular velocities excluded (threshold: {ANGULAR_VELOCITY_THRESHOLD} rad/s)")
    print(f"  4. Minimum section length: {MIN_SECTION_LENGTH} timepoints")
    print(f"\nNext steps:")
    print(f"  1. Run visualization script: pure_turn_sections_visualizations.py")
    print(f"  2. Compare results to bout-based analysis")
