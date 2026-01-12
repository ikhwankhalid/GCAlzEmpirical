"""
Integrated Angular Velocity Analysis: Testing for Systematic Integration Errors

This script tests whether heading deviation results from systematic gain errors when
animals integrate angular velocity during continuous movement bouts.

Key Hypothesis:
If animals have asymmetric gain errors (e.g., over-integrate left turns vs
under-integrate right turns), then heading deviation should show opposite biases
for accumulated left vs right turns.

Analysis Approach:
1. Identify continuous movement bouts (speed > threshold, length >= min_length)
2. Within each bout, compute integrated angular velocity from bout start
3. Classify timepoints by accumulated turn direction (left if Ω(t) > 0, right if Ω(t) < 0)
4. Test for asymmetry: regression slopes β_left vs β_right
5. Compare across speed thresholds, behavioral phases, and light conditions

Author: Analysis generated for Peng et al. 2025
Date: 2025-11-24
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
SPEED_THRESHOLDS = [0.5, 1.0, 2.0, 10.0]  # cm/s
MIN_BOUT_LENGTH = 5  # timepoints (~1 second at 20 Hz)
SMOOTH_WINDOW = 0.5  # for angular velocity smoothing
MAX_INTEGRATED_THRESHOLD = np.pi  # Set to π, 2π, etc. to enable bout truncation at threshold (radians)

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
        Minimum bout length in timepoints (default 20 = ~1 second)

    Returns
    -------
    bouts : list of tuples
        List of (start_idx, end_idx) for each valid bout
    """
    # Create boolean array for movement
    moving = speed >= threshold
    moving = moving & ~np.isnan(speed)  # Also exclude NaN

    # Find transitions
    # Add False at boundaries to detect bouts at edges
    moving_padded = np.concatenate(([False], moving, [False]))
    diff = np.diff(moving_padded.astype(int))

    bout_starts = np.where(diff == 1)[0]
    bout_ends = np.where(diff == -1)[0]

    # Filter by minimum length
    bouts = [(start, end) for start, end in zip(bout_starts, bout_ends)
             if (end - start) >= min_length]

    return bouts


def get_bout_statistics(bouts, speed, time):
    """
    Calculate statistics for identified bouts.

    Parameters
    ----------
    bouts : list of tuples
        List of (start_idx, end_idx) for each bout
    speed : array
        Speed values
    time : array
        Time values

    Returns
    -------
    stats : dict
        Statistics about bouts
    """
    if len(bouts) == 0:
        return {
            'n_bouts': 0,
            'total_timepoints': 0,
            'mean_bout_length': 0,
            'median_bout_length': 0,
            'mean_bout_duration': 0,
            'mean_bout_speed': 0
        }

    bout_lengths = [end - start for start, end in bouts]
    bout_durations = [time[end-1] - time[start] if end > start else 0 for start, end in bouts]
    bout_speeds = [np.nanmean(speed[start:end]) for start, end in bouts]

    return {
        'n_bouts': len(bouts),
        'total_timepoints': sum(bout_lengths),
        'mean_bout_length': np.mean(bout_lengths),
        'median_bout_length': np.median(bout_lengths),
        'mean_bout_duration': np.mean(bout_durations),
        'mean_bout_speed': np.mean(bout_speeds)
    }


# =============================================================================
# BOUT-BASED INTEGRATION FUNCTIONS
# =============================================================================

def integrate_within_bout(angular_velocity, time, bout_start, bout_end, max_integrated=None):
    """
    Compute integrated angular velocity within a single bout.
    Integration resets to 0 at bout start.

    Parameters
    ----------
    angular_velocity : array
        Angular velocity in rad/s
    time : array
        Time points in seconds
    bout_start : int
        Start index of bout
    bout_end : int
        End index of bout
    max_integrated : float, optional
        Maximum absolute integrated angular velocity (radians).
        If specified, bout will be truncated at first point where
        |integrated_ang_vel| > max_integrated. This removes outliers
        with extreme accumulated turns. Default is None (no truncation).

    Returns
    -------
    integrated_ang_vel : array
        Cumulative integrated angular velocity (radians) from bout start.
        May be shorter than (bout_end - bout_start) if truncated.
    effective_end : int
        Effective bout end index after truncation. If no truncation occurs,
        this equals bout_end.
    """
    omega = angular_velocity[bout_start:bout_end]
    t = time[bout_start:bout_end]

    # Compute dt
    dt = np.diff(t)

    # Cumulative integration using trapezoidal rule
    # Omega(t) = integral from 0 to t of omega(tau) d_tau
    integrated = np.zeros(len(omega))
    integrated[1:] = np.nancumsum(omega[:-1] * dt)

    # Apply threshold if specified (symmetric: truncate at |integrated| > threshold)
    effective_end = bout_end
    if max_integrated is not None:
        exceeds_threshold = np.where(np.abs(integrated) > max_integrated)[0]
        if len(exceeds_threshold) > 0:
            # Truncate at first exceedance
            trunc_idx = exceeds_threshold[0]
            integrated = integrated[:trunc_idx]
            effective_end = bout_start + trunc_idx

    return integrated, effective_end


def process_trial_with_bouts(trial_data, speed_threshold, min_bout_length=20, max_integrated=None):
    """
    Process a single trial: identify bouts and compute integrated angular velocity.

    Parameters
    ----------
    trial_data : DataFrame
        Trial data with columns: x, y, recTime, mvtDirError, etc.
    speed_threshold : float
        Speed threshold for defining bouts (cm/s)
    min_bout_length : int
        Minimum bout length in timepoints
    max_integrated : float, optional
        Maximum absolute integrated angular velocity (radians) for truncation.
        Bouts exceeding this threshold will be truncated. If truncated bout
        is shorter than min_bout_length, it will be excluded.

    Returns
    -------
    bout_data : DataFrame
        Processed data with bout information, or None if no valid bouts
    """
    if len(trial_data) < min_bout_length:
        return None

    # Calculate kinematics
    heading = calculate_heading_from_position(trial_data['x'].values, trial_data['y'].values)
    angular_velocity = calculate_angular_velocity(heading, trial_data['recTime'].values)
    speed = calculate_speed(trial_data['x'].values, trial_data['y'].values, trial_data['recTime'].values)

    # Identify bouts
    bouts = identify_continuous_bouts(speed, speed_threshold, min_bout_length)

    if len(bouts) == 0:
        return None

    # Process each bout
    bout_results = []

    for bout_idx, (start, end) in enumerate(bouts):
        # Integrate within this bout (may be truncated at threshold)
        integrated_ang_vel, effective_end = integrate_within_bout(
            angular_velocity, trial_data['recTime'].values, start, end, max_integrated)

        # Check if truncated bout still meets minimum length requirement
        effective_length = effective_end - start
        if effective_length < min_bout_length:
            continue  # Skip this bout if too short after truncation

        # Extract data for this bout using effective end
        bout_slice = trial_data.iloc[start:effective_end].copy()
        bout_slice['bout_id'] = bout_idx
        bout_slice['bout_start'] = start
        bout_slice['bout_end'] = effective_end
        bout_slice['bout_length'] = effective_length
        bout_slice['time_in_bout'] = bout_slice['recTime'].values - bout_slice['recTime'].values[0]
        bout_slice['angular_velocity'] = angular_velocity[start:effective_end]
        bout_slice['integrated_ang_vel'] = integrated_ang_vel
        bout_slice['speed'] = speed[start:effective_end]
        bout_slice['heading'] = heading[start:effective_end]

        # Classify by accumulated turn direction
        bout_slice['turn_direction'] = np.where(integrated_ang_vel > 0, 'left',
                                                 np.where(integrated_ang_vel < 0, 'right', 'zero'))

        bout_results.append(bout_slice)

    if len(bout_results) > 0:
        return pd.concat(bout_results, ignore_index=True)
    else:
        return None


# =============================================================================
# REGRESSION ANALYSIS FUNCTIONS
# =============================================================================

def regression_by_turn_direction(data, condition_name, speed_threshold):
    """
    Perform regression analysis separately for left and right accumulated turns.

    Parameters
    ----------
    data : DataFrame
        Processed bout data with integrated_ang_vel and mvtDirError columns
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
        'n_bouts': data['bout_id'].nunique() if 'bout_id' in data.columns else 0
    }

    # Regression for left turns
    if len(left_data) > 10:  # Need minimum data
        X_left = left_data['integrated_ang_vel'].values
        Y_left = left_data['mvtDirError'].values

        # Remove NaN
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

        # Remove NaN
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

        # Z-test for difference in slopes
        se_diff = np.sqrt(results['se_left']**2 + results['se_right']**2)
        z_stat = results['beta_diff'] / se_diff if se_diff > 0 else 0
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed

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

    This directly tests the hypothesis: Does signed cumulative turning
    predict signed heading deviation? (Expected: negative correlation if
    animals overestimate turns and compensate in opposite direction)

    Parameters
    ----------
    data : DataFrame
        Processed bout data with integrated_ang_vel and mvtDirError columns
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
        'n_bouts': data['bout_id'].nunique() if 'bout_id' in data.columns else 0
    }

    # Get signed values (no splitting by direction)
    X = data['integrated_ang_vel'].values
    Y = data['mvtDirError'].values

    # Remove NaN
    valid = ~(np.isnan(X) | np.isnan(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) > 10:
        # Simple linear regression
        slope, intercept, r, p, se = stats.linregress(X, Y)

        results['beta_signed'] = slope
        results['intercept_signed'] = intercept
        results['r_signed'] = r
        results['r_squared_signed'] = r**2
        results['p_signed'] = p
        results['se_signed'] = se

        # Compute standardized beta for effect size
        if np.std(X) > 0 and np.std(Y) > 0:
            beta_standardized = slope * (np.std(X) / np.std(Y))
            results['beta_standardized'] = beta_standardized
        else:
            results['beta_standardized'] = np.nan

        # 95% confidence interval for slope
        from scipy.stats import t as t_dist
        df = len(X) - 2
        t_crit = t_dist.ppf(0.975, df)
        ci_lower = slope - t_crit * se
        ci_upper = slope + t_crit * se
        results['ci_lower_95'] = ci_lower
        results['ci_upper_95'] = ci_upper

        # Mean and range of predictor and outcome
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


# =============================================================================
# ACCUMULATION ANALYSIS
# =============================================================================

def test_bout_accumulation(data):
    """
    Test if heading deviation magnitude grows over time within bouts.

    Parameters
    ----------
    data : DataFrame
        Bout data with time_in_bout and mvtDirError

    Returns
    -------
    stats : dict
        Accumulation statistics
    """
    # Absolute heading deviation
    data['abs_heading_dev'] = np.abs(data['mvtDirError'])

    # Regression: |heading_dev| ~ time_in_bout
    X = data['time_in_bout'].values
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
# SHUFFLE CONTROLS
# =============================================================================

def shuffle_integrated_ang_vel_within_bouts(data, n_shuffles=100, random_seed=42):
    """
    Shuffle integrated angular velocity within bouts to test if structure matters.

    This tests whether the relationship between integrated_ang_vel and heading_deviation
    depends on the temporal structure within bouts, or if it's just an artifact.

    Parameters
    ----------
    data : DataFrame
        Bout data with bout_id, integrated_ang_vel, mvtDirError
    n_shuffles : int
        Number of shuffle iterations
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    shuffle_results : DataFrame
        Regression results for each shuffle iteration
    """
    np.random.seed(random_seed)

    shuffle_results = []

    for shuffle_iter in tqdm(range(n_shuffles), desc="  Shuffle control"):
        # Shuffle integrated_ang_vel within each bout
        shuffled_data = data.copy()

        for bout_id in shuffled_data['bout_id'].unique():
            bout_mask = shuffled_data['bout_id'] == bout_id
            bout_values = shuffled_data.loc[bout_mask, 'integrated_ang_vel'].values
            shuffled_values = np.random.permutation(bout_values)
            shuffled_data.loc[bout_mask, 'integrated_ang_vel'] = shuffled_values

        # Re-classify by turn direction after shuffling
        shuffled_data['turn_direction'] = np.where(shuffled_data['integrated_ang_vel'] > 0, 'left',
                                                     np.where(shuffled_data['integrated_ang_vel'] < 0, 'right', 'zero'))

        # Regression analysis on shuffled data
        results = regression_by_turn_direction(shuffled_data, 'shuffled', 0)
        results['shuffle_iter'] = shuffle_iter
        shuffle_results.append(results)

    return pd.DataFrame(shuffle_results)


def compare_to_shuffle(observed_results, shuffle_results):
    """
    Compare observed regression results to shuffle distribution.

    Parameters
    ----------
    observed_results : dict or Series
        Observed regression statistics
    shuffle_results : DataFrame
        Shuffle control results

    Returns
    -------
    comparison : dict
        P-values for observed vs shuffle
    """
    comparison = {}

    # Compare beta_left
    if not np.isnan(observed_results['beta_left']):
        shuffle_beta_left = shuffle_results['beta_left'].dropna()
        if len(shuffle_beta_left) > 0:
            p_left = np.mean(np.abs(shuffle_beta_left) >= np.abs(observed_results['beta_left']))
            comparison['p_shuffle_beta_left'] = p_left

    # Compare beta_right
    if not np.isnan(observed_results['beta_right']):
        shuffle_beta_right = shuffle_results['beta_right'].dropna()
        if len(shuffle_beta_right) > 0:
            p_right = np.mean(np.abs(shuffle_beta_right) >= np.abs(observed_results['beta_right']))
            comparison['p_shuffle_beta_right'] = p_right

    # Compare asymmetry (beta_diff)
    if not np.isnan(observed_results['beta_diff']):
        shuffle_beta_diff = shuffle_results['beta_diff'].dropna()
        if len(shuffle_beta_diff) > 0:
            p_diff = np.mean(np.abs(shuffle_beta_diff) >= np.abs(observed_results['beta_diff']))
            comparison['p_shuffle_asymmetry'] = p_diff

    return comparison


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_all_conditions(df, conditions, speed_thresholds, min_bout_length=20, max_integrated=None):
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
    max_integrated : float, optional
        Maximum absolute integrated angular velocity (radians) for truncation

    Returns
    -------
    all_bout_data : DataFrame
        Combined bout data across all conditions
    regression_results : DataFrame
        Regression statistics for each condition/threshold
    """
    all_bout_data = []
    regression_results = []

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

            bout_data_list = []

            for trial_id in tqdm(unique_trials, desc=f"  Processing trials"):
                trial_data = condition_df[condition_df['session_trial'] == trial_id].copy()
                trial_data = trial_data.sort_values('recTime')

                # Process trial
                bout_data = process_trial_with_bouts(trial_data, speed_threshold, min_bout_length, max_integrated)

                if bout_data is not None:
                    bout_data['condition'] = condition
                    bout_data['speed_threshold'] = speed_threshold
                    bout_data['trial_id'] = trial_id
                    bout_data_list.append(bout_data)

            if len(bout_data_list) > 0:
                # Combine all bout data for this condition/threshold
                combined = pd.concat(bout_data_list, ignore_index=True)
                all_bout_data.append(combined)

                print(f"    Total bouts: {combined['bout_id'].nunique()}")
                print(f"    Total timepoints: {len(combined)}")
                print(f"    Left timepoints: {(combined['turn_direction'] == 'left').sum()}")
                print(f"    Right timepoints: {(combined['turn_direction'] == 'right').sum()}")

                # OLD APPROACH: Regression by turn direction (tests asymmetry)
                reg_results_old = regression_by_turn_direction(combined, condition, speed_threshold)

                # NEW APPROACH: Signed regression (tests signed relationship)
                reg_results_signed = signed_regression_analysis(combined, condition, speed_threshold)

                # Merge both results for comparison
                reg_results = {**reg_results_old, **reg_results_signed}

                # Accumulation test
                accum_results = test_bout_accumulation(combined)
                reg_results.update(accum_results)

                regression_results.append(reg_results)

                # Print comparison
                print(f"    OLD APPROACH (split by direction):")
                print(f"      beta_left: {reg_results['beta_left']:.6f}, p={reg_results['p_left']:.4f}")
                print(f"      beta_right: {reg_results['beta_right']:.6f}, p={reg_results['p_right']:.4f}")
                print(f"      Asymmetry: beta_diff={reg_results['beta_diff']:.6f}, p={reg_results['p_asymmetry']:.4f}")
                print(f"    NEW APPROACH (signed regression):")
                print(f"      beta_signed: {reg_results['beta_signed']:.6f}, p={reg_results['p_signed']:.4f}")
                print(f"      R²={reg_results['r_squared_signed']:.4f}, 95% CI=[{reg_results['ci_lower_95']:.6f}, {reg_results['ci_upper_95']:.6f}]")

    # Combine results
    if len(all_bout_data) > 0:
        all_bout_data = pd.concat(all_bout_data, ignore_index=True)
    else:
        all_bout_data = pd.DataFrame()

    regression_results = pd.DataFrame(regression_results)

    return all_bout_data, regression_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("INTEGRATED ANGULAR VELOCITY BOUT-BASED ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data path: {PROJECT_DATA_PATH}")
    print(f"  Speed thresholds: {SPEED_THRESHOLDS} cm/s")
    print(f"  Minimum bout length: {MIN_BOUT_LENGTH} timepoints (~{MIN_BOUT_LENGTH/20:.1f} seconds)")
    print(f"  Max integrated threshold: {MAX_INTEGRATED_THRESHOLD} radians" if MAX_INTEGRATED_THRESHOLD is not None else "  Max integrated threshold: None (no truncation)")
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

    all_bout_data, regression_results = process_all_conditions(
        dfAutoPI,
        CONDITIONS,
        SPEED_THRESHOLDS,
        MIN_BOUT_LENGTH,
        MAX_INTEGRATED_THRESHOLD
    )

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_dir = os.path.join(PROJECT_DATA_PATH, "results")

    if len(all_bout_data) > 0:
        bout_data_fn = os.path.join(results_dir, "integrated_ang_vel_bout_data.csv")
        all_bout_data.to_csv(bout_data_fn, index=False)
        print(f"Saved bout data: {bout_data_fn}")
        print(f"  Total rows: {len(all_bout_data):,}")

    if len(regression_results) > 0:
        reg_results_fn = os.path.join(results_dir, "integrated_ang_vel_regression_results.csv")
        regression_results.to_csv(reg_results_fn, index=False)
        print(f"Saved regression results: {reg_results_fn}")
        print(f"  Total rows: {len(regression_results)}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    if len(regression_results) > 0:
        print("\n" + "="*80)
        print("PRIMARY ANALYSIS: SIGNED REGRESSION")
        print("="*80)
        print("\nDirect test of hypothesis: Does signed cumulative turning predict signed heading deviation?")
        print("Expected: β < 0 (accumulated left turns → rightward heading error)\n")

        # Show signed regression results
        signed_cols = ['condition', 'speed_threshold', 'beta_signed', 'r_squared_signed',
                      'p_signed', 'ci_lower_95', 'ci_upper_95']
        available_cols = [col for col in signed_cols if col in regression_results.columns]
        print(regression_results[available_cols].to_string(index=False))

        # Count significant signed relationships
        sig_signed = (regression_results['p_signed'] < 0.05).sum()
        total_signed = len(regression_results[~regression_results['p_signed'].isna()])
        print(f"\nSignificant signed relationships (p < 0.05): {sig_signed}/{total_signed}")

        # Count negative betas (expected direction)
        negative_betas = (regression_results['beta_signed'] < 0).sum()
        print(f"Negative β (expected direction): {negative_betas}/{total_signed}")

        print("\n" + "="*80)
        print("COMPARISON: OLD APPROACH (Split by Direction)")
        print("="*80)
        print("\nNote: Old approach tests asymmetry (β_left ≠ β_right), not signed relationship")
        print("      A negative signed relationship appears as β_left < 0 AND β_right > 0\n")

        old_cols = ['condition', 'speed_threshold', 'beta_left', 'beta_right',
                   'beta_diff', 'p_asymmetry']
        available_old_cols = [col for col in old_cols if col in regression_results.columns]
        print(regression_results[available_old_cols].to_string(index=False))

        # Count significant asymmetries
        sig_asymmetries = (regression_results['p_asymmetry'] < 0.05).sum()
        total_asymmetry = len(regression_results[~regression_results['p_asymmetry'].isna()])
        print(f"\nSignificant asymmetries (p < 0.05): {sig_asymmetries}/{total_asymmetry}")

        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        print("Signed Regression (PRIMARY):")
        print("  β < 0: Accumulated left turns → rightward heading error (overestimation)")
        print("  β > 0: Accumulated left turns → leftward heading error (underestimation)")
        print("  β ≈ 0: No systematic relationship between cumulative turns and heading error")
        print("\nOld Approach (COMPARISON ONLY):")
        print("  Shows β_left < 0 AND β_right > 0 when there's a negative signed relationship")
        print("  This is NOT asymmetry - it's one consistent relationship!")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Review primary results: signed regression β and p-values")
    print(f"  2. Run visualization script: integrated_angular_velocity_visualizations.py")
    print(f"  3. If significant effects found, add controls (speed, phase, distance)")
    print(f"  4. Consider hierarchical models to account for session/trial clustering")
