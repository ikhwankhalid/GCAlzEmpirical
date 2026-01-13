"""
Trial-Level Analysis: Cumulative Turn vs Final Heading Error

This script analyzes the relationship between cumulative angular velocity
integrated over entire trials and the final heading error at homing endpoint.

Key difference from pure turn section analysis:
- Section analysis: One data point per continuous turn (same-direction rotation)
- Trial analysis: One data point per complete trial (trial start to homing endpoint)

Variables:
- X: Cumulative turn = integral of angular velocity over entire trial (radians)
- Y: Final heading error = mvtDirError at homing endpoint (radians)

Author: Analysis for Peng et al. 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DATA_PATH = '/workspace/Peng'
RESULTS_PATH = os.path.join(PROJECT_DATA_PATH, "results")

# Minimum trials per animal for regression
MIN_TRIALS = 5

# Animals in dataset
ANIMALS = ['jp486', 'jp3269', 'jp452', 'jp3120', 'jp451', 'mn8578', 'jp1686']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_angular_velocity(heading, time):
    """
    Compute angular velocity from heading time series.

    Parameters
    ----------
    heading : array
        Heading angles in radians
    time : array
        Time stamps

    Returns
    -------
    angular_velocity : array
        Angular velocity in rad/s (same length as input, first value = 0)
    """
    # Compute heading differences, handling wrap-around
    dh = np.diff(heading)
    dh = wrap_angle(dh)  # Handle wrap-around at +/- pi

    # Compute time differences
    dt = np.diff(time)
    dt = np.where(dt == 0, 1e-10, dt)  # Avoid division by zero

    # Angular velocity
    ang_vel = dh / dt

    # Prepend zero for first timepoint
    return np.concatenate([[0], ang_vel])


def compute_trial_metrics(trial_data):
    """
    Compute X and Y variables for a single trial.

    Parameters
    ----------
    trial_data : DataFrame
        All timepoints for one trial in homing phase

    Returns
    -------
    dict with cumulative_turn, final_heading_error, and metadata
    """
    if len(trial_data) < 2:
        return None

    # Sort by time
    trial_data = trial_data.sort_values('recTime').reset_index(drop=True)

    # Compute angular velocity from heading
    heading = trial_data['hdPose'].values
    time = trial_data['recTime'].values
    ang_vel = compute_angular_velocity(heading, time)

    # X: Cumulative turn = integral of |angular_velocity| over trial
    # Using absolute value to capture total turning magnitude
    dt = np.diff(time, prepend=time[0])
    cumulative_turn_abs = np.sum(np.abs(ang_vel) * dt)

    # Also compute signed cumulative turn (net rotation)
    cumulative_turn_signed = np.sum(ang_vel * dt)

    # Y: Final heading error at homing endpoint
    final_heading_error = trial_data['mvtDirError'].iloc[-1]

    # Metadata
    duration = time[-1] - time[0]
    n_timepoints = len(trial_data)
    mean_speed = trial_data['speed'].mean() if 'speed' in trial_data.columns else np.nan

    return {
        'cumulative_turn_abs': cumulative_turn_abs,
        'cumulative_turn_signed': cumulative_turn_signed,
        'final_heading_error': final_heading_error,
        'duration': duration,
        'n_timepoints': n_timepoints,
        'mean_speed': mean_speed,
        'start_time': time[0],
        'end_time': time[-1]
    }


def per_animal_regression(trial_df, x_col='cumulative_turn_abs', y_col='final_heading_error',
                          min_trials=5):
    """
    Run regression per animal on trial-level data.

    Parameters
    ----------
    trial_df : DataFrame
        Trial-level data with one row per trial
    x_col : str
        Column name for X variable
    y_col : str
        Column name for Y variable
    min_trials : int
        Minimum trials required per animal

    Returns
    -------
    results_df : DataFrame
        Per-animal regression results
    """
    results = []

    for animal_id in trial_df['mouse'].unique():
        animal_data = trial_df[trial_df['mouse'] == animal_id]

        X = animal_data[x_col].values
        Y = animal_data[y_col].values

        # Remove NaN
        valid = ~(np.isnan(X) | np.isnan(Y))
        X, Y = X[valid], Y[valid]

        n = len(X)
        if n >= min_trials:
            slope, intercept, r, p, se = stats.linregress(X, Y)

            # Compute 95% CI
            df = n - 2
            if df > 0:
                t_crit = t_dist.ppf(0.975, df)
                ci_lower = slope - t_crit * se
                ci_upper = slope + t_crit * se
            else:
                ci_lower = np.nan
                ci_upper = np.nan

            results.append({
                'animal_id': animal_id,
                'n_trials': n,
                'beta': slope,
                'intercept': intercept,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'r': r,
                'r_squared': r**2,
                'p_value': p,
                'x_mean': np.mean(X),
                'x_std': np.std(X),
                'y_mean': np.mean(Y),
                'y_std': np.std(Y)
            })

    return pd.DataFrame(results)


def per_animal_regression_by_condition(trial_df, conditions, x_col='cumulative_turn_abs',
                                        y_col='final_heading_error', min_trials=5):
    """
    Run per-animal regression separately for each condition.

    Parameters
    ----------
    trial_df : DataFrame
        Trial-level data
    conditions : list
        List of condition values to analyze
    x_col, y_col : str
        Column names for X and Y variables
    min_trials : int
        Minimum trials per animal/condition

    Returns
    -------
    results_df : DataFrame
    """
    results = []

    for condition in conditions:
        cond_data = trial_df[trial_df['light'] == condition]

        for animal_id in cond_data['mouse'].unique():
            animal_data = cond_data[cond_data['mouse'] == animal_id]

            X = animal_data[x_col].values
            Y = animal_data[y_col].values

            valid = ~(np.isnan(X) | np.isnan(Y))
            X, Y = X[valid], Y[valid]

            n = len(X)
            if n >= min_trials:
                slope, intercept, r, p, se = stats.linregress(X, Y)

                df = n - 2
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
                    'n_trials': n,
                    'beta': slope,
                    'intercept': intercept,
                    'se': se,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'r': r,
                    'r_squared': r**2,
                    'p_value': p
                })

    return pd.DataFrame(results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TRIAL-LEVEL ANALYSIS: Cumulative Turn vs Final Heading Error")
    print("="*80)

    # Load reconstruction data
    recon_fn = os.path.join(RESULTS_PATH, "reconstuctionDFAutoPI.csv")
    print(f"\nLoading reconstruction data from: {recon_fn}")
    print("(This may take a moment...)")

    recon = pd.read_csv(recon_fn)
    print(f"  Total rows: {len(recon):,}")
    print(f"  Columns: {list(recon.columns)[:10]}...")

    # Filter to homing phase only
    print("\nFiltering to homing phase...")
    homing = recon[recon['condition'].str.contains('homingFromLeavingLever', na=False)].copy()
    print(f"  Homing rows: {len(homing):,}")

    if len(homing) == 0:
        print("ERROR: No homing data found!")
        exit(1)

    # Create trial_id
    homing['trial_id'] = homing['session'] + '_T' + homing['trialNo'].astype(str)

    # Get unique trials
    unique_trials = homing['trial_id'].unique()
    print(f"  Unique trials: {len(unique_trials)}")
    print(f"  Unique animals: {homing['mouse'].nunique()}")
    print(f"  Animals: {list(homing['mouse'].unique())}")

    # Compute trial-level metrics
    print("\n" + "-"*60)
    print("COMPUTING TRIAL-LEVEL METRICS")
    print("-"*60)

    trial_results = []

    for i, trial_id in enumerate(unique_trials):
        if (i + 1) % 500 == 0:
            print(f"  Processing trial {i+1}/{len(unique_trials)}...")

        trial_data = homing[homing['trial_id'] == trial_id]
        metrics = compute_trial_metrics(trial_data)

        if metrics is not None:
            metrics['trial_id'] = trial_id
            metrics['mouse'] = trial_data['mouse'].iloc[0]
            metrics['session'] = trial_data['session'].iloc[0]
            metrics['light'] = trial_data['light'].iloc[0]
            metrics['trialNo'] = trial_data['trialNo'].iloc[0]
            trial_results.append(metrics)

    trial_df = pd.DataFrame(trial_results)
    print(f"\n  Trials with valid metrics: {len(trial_df)}")

    # Summary statistics
    print("\n" + "-"*60)
    print("TRIAL-LEVEL SUMMARY")
    print("-"*60)
    print(f"\n  Total trials: {len(trial_df)}")
    print(f"  Light trials: {len(trial_df[trial_df['light'] == 'light'])}")
    print(f"  Dark trials: {len(trial_df[trial_df['light'] == 'dark'])}")
    print(f"\n  Cumulative turn (abs):")
    print(f"    Mean: {trial_df['cumulative_turn_abs'].mean():.2f} rad")
    print(f"    Std:  {trial_df['cumulative_turn_abs'].std():.2f} rad")
    print(f"    Range: [{trial_df['cumulative_turn_abs'].min():.2f}, {trial_df['cumulative_turn_abs'].max():.2f}]")
    print(f"\n  Final heading error:")
    print(f"    Mean: {trial_df['final_heading_error'].mean():.4f} rad")
    print(f"    Std:  {trial_df['final_heading_error'].std():.4f} rad")

    # Save trial-level data
    trial_fn = os.path.join(RESULTS_PATH, "trial_level_data.csv")
    trial_df.to_csv(trial_fn, index=False)
    print(f"\n  Saved to: {trial_fn}")

    # ==========================================================================
    # POOLED REGRESSION
    # ==========================================================================
    print("\n" + "-"*60)
    print("POOLED REGRESSION (ALL ANIMALS)")
    print("-"*60)

    X = trial_df['cumulative_turn_abs'].values
    Y = trial_df['final_heading_error'].values
    valid = ~(np.isnan(X) | np.isnan(Y))
    X_valid, Y_valid = X[valid], Y[valid]

    slope, intercept, r, p, se = stats.linregress(X_valid, Y_valid)
    n = len(X_valid)
    df = n - 2
    t_crit = t_dist.ppf(0.975, df)
    ci_lower = slope - t_crit * se
    ci_upper = slope + t_crit * se

    print(f"\n  n = {n} trials")
    print(f"  beta = {slope:.6f} [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  R² = {r**2:.6f}")
    print(f"  p = {p:.4e}")

    # ==========================================================================
    # PER-ANIMAL REGRESSION
    # ==========================================================================
    print("\n" + "-"*60)
    print("PER-ANIMAL REGRESSION")
    print("-"*60)

    per_animal_results = per_animal_regression(trial_df, min_trials=MIN_TRIALS)

    if len(per_animal_results) > 0:
        print(f"\n  Animals with >= {MIN_TRIALS} trials: {len(per_animal_results)}")
        print("\n  Per-animal results:")
        for _, row in per_animal_results.iterrows():
            sig = "*" if row['p_value'] < 0.05 else " "
            print(f"    {row['animal_id']}: beta={row['beta']:.4f}, "
                  f"R²={row['r_squared']:.4f}, n={row['n_trials']}{sig}")

        # Save per-animal results
        per_animal_fn = os.path.join(RESULTS_PATH, "trial_level_per_animal.csv")
        per_animal_results.to_csv(per_animal_fn, index=False)
        print(f"\n  Saved to: {per_animal_fn}")

    # ==========================================================================
    # PER-ANIMAL BY CONDITION
    # ==========================================================================
    print("\n" + "-"*60)
    print("PER-ANIMAL REGRESSION BY CONDITION")
    print("-"*60)

    conditions = ['light', 'dark']
    per_animal_by_cond = per_animal_regression_by_condition(
        trial_df, conditions, min_trials=MIN_TRIALS
    )

    if len(per_animal_by_cond) > 0:
        print(f"\n  Total rows: {len(per_animal_by_cond)}")

        for cond in conditions:
            cond_data = per_animal_by_cond[per_animal_by_cond['condition'] == cond]
            print(f"\n  {cond.upper()} condition:")
            for _, row in cond_data.iterrows():
                sig = "*" if row['p_value'] < 0.05 else " "
                print(f"    {row['animal_id']}: beta={row['beta']:.4f}, n={row['n_trials']}{sig}")

        # Save
        per_animal_cond_fn = os.path.join(RESULTS_PATH, "trial_level_per_animal_by_condition.csv")
        per_animal_by_cond.to_csv(per_animal_cond_fn, index=False)
        print(f"\n  Saved to: {per_animal_cond_fn}")

    # ==========================================================================
    # COMPLETE
    # ==========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - trial_level_data.csv (one row per trial)")
    print(f"  - trial_level_per_animal.csv (per-animal regression)")
    print(f"  - trial_level_per_animal_by_condition.csv (per-animal by light/dark)")
