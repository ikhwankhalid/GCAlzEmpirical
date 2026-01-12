"""
Data Preprocessing Module for Hierarchical Heading Model

This module handles loading, filtering, and structuring data from the
reconstruction dataframe for the hierarchical Bayesian model.

Key functions:
- load_and_filter_data: Load and filter reconstruction CSV
- calculate_true_heading: Compute heading from position trajectory
- calculate_angular_velocity: Compute ω(t) from heading
- integrate_angular_velocity: Compute Ω(t) = ∫ω(τ)dτ
- calculate_decoded_heading: Compute heading from predictions
- structure_hierarchical_data: Structure data for PyMC model
"""

import numpy as np
import pandas as pd
import scipy.ndimage
import os
from typing import List, Optional, Dict, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Default usable sessions (grid score > 0.5)
DEFAULT_USEABLE_SESSIONS = [
    'jp486-19032023-0108', 'jp486-18032023-0108',
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
    'jp452-23112022-0108', 'jp1686-26042022-0108'
]

# Default project data path
DEFAULT_PROJECT_DATA_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng'


def load_and_filter_data(
    conditions: Optional[List[str]] = None,
    sessions: Optional[List[str]] = None,
    min_speed: float = 2.0,
    filter_mode: str = 'none',
    project_data_path: str = DEFAULT_PROJECT_DATA_PATH,
    max_nan_fraction: float = 0.2,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load and filter reconstruction data for hierarchical model.

    Parameters
    ----------
    conditions : list of str, optional
        Experimental conditions to include. If None, uses all conditions.
        Options: 'all_light', 'all_dark', 'searchToLeverPath_light',
                 'searchToLeverPath_dark', 'homingFromLeavingLever_light',
                 'homingFromLeavingLever_dark', 'atLever_light', 'atLever_dark'
    sessions : list of str, optional
        Session names to include. If None, uses DEFAULT_USEABLE_SESSIONS.
    min_speed : float, default=2.0
        Minimum speed threshold (cm/s). Interpretation depends on filter_mode.
    filter_mode : str, default='none'
        How to handle low-speed data:
        - 'none': Keep all data points (recommended for continuous model)
        - 'remove': Remove data points below min_speed (old behavior)
        - 'weight': Keep all data, add speed_weight column (future option)
    project_data_path : str
        Path to project data directory containing results folder.
    max_nan_fraction : float, default=0.2
        Maximum fraction of NaN values allowed in a trial (0-1).
    verbose : bool, default=True
        Print loading progress and statistics.

    Returns
    -------
    df : pd.DataFrame
        Filtered dataframe with columns:
        mouse, session, condition, trial, recTime, x, y, px, py,
        speed, mvtDirError, and others.
    """
    # Set defaults
    if sessions is None:
        sessions = DEFAULT_USEABLE_SESSIONS

    # Load data
    fn = os.path.join(project_data_path, "results", "reconstuctionDFAutoPI.csv")

    if verbose:
        print(f"Loading data from: {fn}")

    if not os.path.exists(fn):
        raise FileNotFoundError(f"Data file not found: {fn}")

    df = pd.read_csv(fn)

    if verbose:
        print(f"Loaded {len(df):,} rows")

    # Filter by sessions
    df = df[df.session.isin(sessions)].copy()

    if verbose:
        print(f"After session filter: {len(df):,} rows from {df.session.nunique()} sessions")

    # Filter by conditions if specified
    if conditions is not None:
        df = df[df.condition.isin(conditions)].copy()
        if verbose:
            print(f"After condition filter: {len(df):,} rows")

    # Handle speed filtering based on filter_mode
    if filter_mode == 'remove':
        # Old behavior: remove low-speed data points
        df = df[df.speed >= min_speed].copy()
        if verbose:
            print(f"After speed filter (>={min_speed} cm/s): {len(df):,} rows")
    elif filter_mode == 'weight':
        # Add speed weight column but keep all data
        df['speed_weight'] = np.clip(df['speed'] / min_speed, 0.0, 1.0)
        if verbose:
            print(f"Speed weighting enabled (threshold={min_speed} cm/s)")
    elif filter_mode == 'none':
        # Keep all data without modification
        if verbose:
            print(f"No speed filtering applied (keeping all {len(df):,} rows)")
    else:
        raise ValueError(f"Invalid filter_mode: {filter_mode}. Must be 'none', 'remove', or 'weight'.")

    # Check for required columns
    required_cols = ['mouse', 'session', 'trial', 'recTime', 'x', 'y', 'px', 'py', 'speed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create session_trial identifier
    df['session_trial'] = df['session'] + '_T' + df['trial'].astype(str)

    # Remove trials with too many NaNs
    initial_trials = df['session_trial'].nunique()

    trial_nan_fraction = df.groupby('session_trial').apply(
        lambda x: x[['x', 'y', 'px', 'py']].isna().any(axis=1).mean()
    )

    valid_trials = trial_nan_fraction[trial_nan_fraction <= max_nan_fraction].index
    df = df[df.session_trial.isin(valid_trials)].copy()

    if verbose:
        removed_trials = initial_trials - len(valid_trials)
        print(f"Removed {removed_trials} trials with >{max_nan_fraction*100:.0f}% NaN values")
        print(f"Final dataset: {len(df):,} rows, {df.session_trial.nunique()} trials, "
              f"{df.mouse.nunique()} animals")

    return df


def calculate_true_heading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate true heading from position trajectory.

    Heading is defined as the direction of movement: arctan2(Δy, Δx)

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates

    Returns
    -------
    heading : np.ndarray
        Heading angles in radians [-π, π]
        First value is NaN (no previous position to compute heading)
    """
    dx = np.diff(x)
    dy = np.diff(y)
    heading = np.arctan2(dy, dx)

    # Pad first value with NaN
    heading = np.insert(heading, 0, np.nan)

    return heading


def calculate_angular_velocity(
    heading: np.ndarray,
    recTime: np.ndarray,
    smooth_sigma: float = 1.0
) -> np.ndarray:
    """
    Calculate angular velocity ω(t) = dθ/dt from heading time series.
    
    Uses circular-aware smoothing by converting to complex exponentials to avoid
    artifacts at angular discontinuities.

    Parameters
    ----------
    heading : np.ndarray
        Heading angles in radians
    recTime : np.ndarray
        Time values (in seconds)
    smooth_sigma : float, default=1.0
        Gaussian smoothing sigma (0 = no smoothing)

    Returns
    -------
    omega : np.ndarray
        Angular velocity in rad/s
        First value is NaN
    """
    # Compute heading differences
    dheading = np.diff(heading)

    # Wrap to [-π, π] to handle circular discontinuities
    dheading = np.arctan2(np.sin(dheading), np.cos(dheading))

    # Compute time differences
    dt = np.diff(recTime)
    dt[dt == 0] = np.nan  # Avoid division by zero

    # Angular velocity
    omega = dheading / dt

    # Pad first value
    omega = np.insert(omega, 0, np.nan)

    # Apply circular-aware smoothing if requested
    if smooth_sigma > 0:
        valid_mask = ~np.isnan(omega)
        if np.sum(valid_mask) > 0:
            # Smooth by integrating, converting to complex exponential, smoothing, then differentiating
            # This preserves circular statistics
            # First, reconstruct angles from omega
            dt_array = np.diff(recTime, prepend=recTime[0])
            cumulative_angle = np.nancumsum(omega * dt_array)
            
            # Convert to complex exponential (circular representation)
            z = np.exp(1j * cumulative_angle)
            
            # Smooth the complex signal (separately for real and imaginary parts)
            z_smooth = z.copy()
            z_smooth[valid_mask] = (
                scipy.ndimage.gaussian_filter1d(np.real(z[valid_mask]), sigma=smooth_sigma) +
                1j * scipy.ndimage.gaussian_filter1d(np.imag(z[valid_mask]), sigma=smooth_sigma)
            )
            
            # Normalize to unit circle
            z_smooth = z_smooth / np.abs(z_smooth)
            
            # Convert back to angles
            smoothed_angle = np.angle(z_smooth)
            
            # Compute smoothed angular velocity
            dangle = np.diff(smoothed_angle)
            dangle = np.arctan2(np.sin(dangle), np.cos(dangle))  # Wrap
            omega_smoothed = dangle / dt
            omega_smoothed = np.insert(omega_smoothed, 0, np.nan)
            
            omega[valid_mask] = omega_smoothed[valid_mask]

    return omega


def integrate_angular_velocity(
    omega: np.ndarray,
    recTime: np.ndarray
) -> np.ndarray:
    """
    Compute integrated angular velocity Ω(t) = ∫ω(τ)dτ.

    Uses cumulative trapezoidal integration.

    Parameters
    ----------
    omega : np.ndarray
        Angular velocity in rad/s
    recTime : np.ndarray
        Time values (in seconds)

    Returns
    -------
    Omega : np.ndarray
        Integrated angular velocity (cumulative angle in radians)
    """
    # Compute time differences
    dt = np.diff(recTime, prepend=recTime[0])

    # Cumulative integration: Ω(t) = Σ ω(t) * Δt
    # Use nan-aware cumsum
    Omega = np.nancumsum(omega * dt)

    return Omega


def calculate_decoded_heading(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Calculate decoded heading from predicted positions.

    Parameters
    ----------
    px : np.ndarray
        Predicted X coordinates
    py : np.ndarray
        Predicted Y coordinates

    Returns
    -------
    decoded_heading : np.ndarray
        Decoded heading angles in radians [-π, π]
        First value is NaN
    """
    dpx = np.diff(px)
    dpy = np.diff(py)
    decoded_heading = np.arctan2(dpy, dpx)

    # Pad first value with NaN
    decoded_heading = np.insert(decoded_heading, 0, np.nan)

    return decoded_heading


def process_trial_data(
    trial_df: pd.DataFrame,
    smooth_sigma: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Process a single trial's data to compute all required quantities.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Data for a single trial (must be sorted by recTime)
    smooth_sigma : float, default=1.0
        Smoothing parameter for angular velocity

    Returns
    -------
    trial_data : dict
        Dictionary with keys:
        - 'recTime': Time array
        - 't': Relative time (from 0)
        - 'true_heading': True heading from (x, y)
        - 'decoded_heading': Decoded heading from (px, py)
        - 'omega': Angular velocity
        - 'Omega': Integrated angular velocity
        - 'theta_obs': Observed decoded heading (for model)
        - 'speed': Speed array (cm/s)
        - 'valid_mask': Boolean mask of valid (non-NaN) data points
    """
    # Sort by time
    trial_df = trial_df.sort_values('recTime')

    # Extract arrays
    recTime = trial_df['recTime'].values
    x = trial_df['x'].values
    y = trial_df['y'].values
    px = trial_df['px'].values
    py = trial_df['py'].values
    speed = trial_df['speed'].values

    # Calculate true heading
    true_heading = calculate_true_heading(x, y)

    # Calculate angular velocity from true movement
    omega = calculate_angular_velocity(true_heading, recTime, smooth_sigma)

    # Integrate angular velocity
    Omega = integrate_angular_velocity(omega, recTime)

    # Calculate decoded heading
    decoded_heading = calculate_decoded_heading(px, py)

    # Relative time (start from 0)
    t = recTime - recTime[0]

    # Valid data mask (no NaNs)
    valid_mask = ~(np.isnan(omega) | np.isnan(Omega) | np.isnan(decoded_heading))

    return {
        'recTime': recTime,
        't': t,
        'true_heading': true_heading,
        'decoded_heading': decoded_heading,
        'omega': omega,
        'Omega': Omega,
        'theta_obs': decoded_heading,  # This is what we're modeling
        'speed': speed,
        'valid_mask': valid_mask
    }


def structure_hierarchical_data(
    df: pd.DataFrame,
    smooth_sigma: float = 1.0,
    min_trial_length: int = 50,
    max_animals: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Structure data for hierarchical Bayesian model.

    Organizes data by animal → trial → timepoints

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe from load_and_filter_data()
    smooth_sigma : float, default=1.0
        Smoothing parameter for angular velocity
    min_trial_length : int, default=50
        Minimum number of valid timepoints required for a trial
    max_animals : int, optional
        Maximum number of animals to include (for testing)
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    data_dict : dict
        Structured data dictionary with keys:
        - 'animals': List of animal IDs
        - 'n_animals': Number of animals
        - 'trials_per_animal': Dict mapping animal → list of trial IDs
        - 'omega_integrated': Dict[animal][trial] → Ω(t) array
        - 'time': Dict[animal][trial] → t array
        - 'theta_obs': Dict[animal][trial] → θ̂_obs array
        - 'speed': Dict[animal][trial] → speed array
        - 'trial_lengths': Dict[animal][trial] → number of timepoints
        - 'trial_info': Dict with metadata
    """
    if verbose:
        print("\nStructuring hierarchical data...")

    # Get unique animals
    animals = sorted(df['mouse'].unique())

    if max_animals is not None:
        animals = animals[:max_animals]
        if verbose:
            print(f"Limiting to first {max_animals} animals")

    n_animals = len(animals)

    # Initialize data structures
    trials_per_animal = {}
    omega_integrated = {}
    time = {}
    theta_obs = {}
    speed = {}
    trial_lengths = {}
    trial_info = {}

    total_trials = 0
    total_timepoints = 0

    for animal in animals:
        if verbose:
            print(f"  Processing animal: {animal}")

        animal_df = df[df['mouse'] == animal]
        trial_ids = animal_df['session_trial'].unique()

        trials_per_animal[animal] = []
        omega_integrated[animal] = {}
        time[animal] = {}
        theta_obs[animal] = {}
        speed[animal] = {}
        trial_lengths[animal] = {}
        trial_info[animal] = {}

        for trial_id in trial_ids:
            trial_df = animal_df[animal_df['session_trial'] == trial_id]

            # Process trial
            try:
                trial_data = process_trial_data(trial_df, smooth_sigma)
            except Exception as e:
                if verbose:
                    print(f"    Warning: Failed to process trial {trial_id}: {e}")
                continue

            # Apply valid mask and check length
            valid_mask = trial_data['valid_mask']
            n_valid = np.sum(valid_mask)

            if n_valid < min_trial_length:
                continue

            # Store data (only valid timepoints)
            trials_per_animal[animal].append(trial_id)
            omega_integrated[animal][trial_id] = trial_data['Omega'][valid_mask]
            time[animal][trial_id] = trial_data['t'][valid_mask]
            theta_obs[animal][trial_id] = trial_data['theta_obs'][valid_mask]
            speed[animal][trial_id] = trial_data['speed'][valid_mask]
            trial_lengths[animal][trial_id] = n_valid

            # Store metadata
            trial_info[animal][trial_id] = {
                'session': trial_df['session'].iloc[0],
                'trial_no': trial_df['trial'].iloc[0],
                'condition': trial_df['condition'].iloc[0] if 'condition' in trial_df.columns else None,
                'n_timepoints': n_valid
            }

            total_trials += 1
            total_timepoints += n_valid

    # Remove animals with no valid trials
    animals_with_data = [a for a in animals if len(trials_per_animal[a]) > 0]

    if verbose:
        print(f"\nData structure complete:")
        print(f"  Animals: {len(animals_with_data)}")
        print(f"  Total trials: {total_trials}")
        print(f"  Total timepoints: {total_timepoints:,}")
        print(f"  Avg trials per animal: {total_trials / len(animals_with_data):.1f}")
        print(f"  Avg timepoints per trial: {total_timepoints / total_trials:.0f}")

    return {
        'animals': animals_with_data,
        'n_animals': len(animals_with_data),
        'trials_per_animal': {a: trials_per_animal[a] for a in animals_with_data},
        'omega_integrated': {a: omega_integrated[a] for a in animals_with_data},
        'time': {a: time[a] for a in animals_with_data},
        'theta_obs': {a: theta_obs[a] for a in animals_with_data},
        'speed': {a: speed[a] for a in animals_with_data},
        'trial_lengths': {a: trial_lengths[a] for a in animals_with_data},
        'trial_info': {a: trial_info[a] for a in animals_with_data}
    }


def get_data_summary(data_dict: Dict) -> pd.DataFrame:
    """
    Generate a summary dataframe of structured hierarchical data.

    Parameters
    ----------
    data_dict : dict
        Output from structure_hierarchical_data()

    Returns
    -------
    summary_df : pd.DataFrame
        Summary with one row per animal showing trial counts and stats
    """
    summary_rows = []

    for animal in data_dict['animals']:
        trials = data_dict['trials_per_animal'][animal]
        n_trials = len(trials)

        # Compute statistics
        lengths = [data_dict['trial_lengths'][animal][t] for t in trials]
        mean_length = np.mean(lengths)
        total_points = np.sum(lengths)

        summary_rows.append({
            'animal': animal,
            'n_trials': n_trials,
            'total_timepoints': total_points,
            'mean_trial_length': mean_length,
            'min_trial_length': np.min(lengths),
            'max_trial_length': np.max(lengths)
        })

    return pd.DataFrame(summary_rows)


if __name__ == '__main__':
    # Example usage
    print("="*80)
    print("Data Preprocessing Module - Example Usage")
    print("="*80)

    # Load and filter data
    df = load_and_filter_data(
        conditions=['all_light'],
        sessions=DEFAULT_USEABLE_SESSIONS[:5],  # First 5 sessions for demo
        min_speed=2.0
    )

    # Structure for hierarchical model
    data_dict = structure_hierarchical_data(
        df,
        smooth_sigma=1.0,
        min_trial_length=50,
        max_animals=3  # Limit for demo
    )

    # Print summary
    summary = get_data_summary(data_dict)
    print("\nData Summary:")
    print(summary.to_string(index=False))

    print("\n" + "="*80)
    print("Data preprocessing complete!")
    print("="*80)
