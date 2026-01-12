"""
Utility Functions for Hierarchical Heading Model

Helper functions for data manipulation, statistics, and reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def circular_mean(angles: np.ndarray) -> float:
    """
    Compute circular mean of angles.

    Parameters
    ----------
    angles : np.ndarray
        Angles in radians

    Returns
    -------
    mean_angle : float
        Circular mean in radians
    """
    return np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles)))


def circular_std(angles: np.ndarray) -> float:
    """
    Compute circular standard deviation.

    Parameters
    ----------
    angles : np.ndarray
        Angles in radians

    Returns
    -------
    std : float
        Circular standard deviation
    """
    r = np.sqrt(np.nanmean(np.cos(angles))**2 + np.nanmean(np.sin(angles))**2)
    return np.sqrt(-2 * np.log(r))


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angles to [-π, π] range.

    Parameters
    ----------
    angle : np.ndarray
        Angles in radians

    Returns
    -------
    wrapped : np.ndarray
        Wrapped angles in [-π, π]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values

    Returns
    -------
    r_squared : float
        R² value (0-1, higher is better)
    """
    ss_res = np.sum((observed - predicted)**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)

    if ss_tot == 0:
        return np.nan

    return 1 - (ss_res / ss_tot)


def compute_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute root mean squared error.

    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values

    Returns
    -------
    rmse : float
        Root mean squared error
    """
    return np.sqrt(np.mean((observed - predicted)**2))


def compute_mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute mean absolute error.

    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values

    Returns
    -------
    mae : float
        Mean absolute error
    """
    return np.mean(np.abs(observed - predicted))


def format_parameter_table(
    results_dict: Dict,
    hdi_prob: float = 0.95
) -> pd.DataFrame:
    """
    Format parameter estimates as a publication-ready table.

    Parameters
    ----------
    results_dict : dict
        Output from model.extract_parameters()
    hdi_prob : float, default=0.95
        HDI probability

    Returns
    -------
    table : pd.DataFrame
        Formatted table
    """
    rows = []

    # Population parameters
    for param_name, param_data in results_dict['population'].items():
        rows.append({
            'Parameter': param_name,
            'Level': 'Population',
            'Mean': f"{param_data['mean']:.3f}",
            f'{hdi_prob*100:.0f}% HDI': f"[{param_data['hdi_lower']:.3f}, {param_data['hdi_upper']:.3f}]"
        })

    # Animal parameters (first few)
    for i, (animal, animal_data) in enumerate(list(results_dict['animals'].items())[:5]):
        rows.append({
            'Parameter': 'alpha',
            'Level': f'Animal ({animal})',
            'Mean': f"{animal_data['alpha_mean']:.3f}",
            f'{hdi_prob*100:.0f}% HDI': f"[{animal_data['alpha_hdi_lower']:.3f}, {animal_data['alpha_hdi_upper']:.3f}]"
        })

    if len(results_dict['animals']) > 5:
        rows.append({
            'Parameter': '...',
            'Level': f'(+{len(results_dict["animals"])-5} more)',
            'Mean': '...',
            f'{hdi_prob*100:.0f}% HDI': '...'
        })

    return pd.DataFrame(rows)


def export_results_to_csv(
    model,
    filename: str,
    include_trials: bool = False
):
    """
    Export model results to CSV files.

    Parameters
    ----------
    model : HeadingHierarchicalModel
        Fitted model
    filename : str
        Base filename (will create multiple files)
    include_trials : bool, default=False
        Whether to include trial-level parameters
    """
    # Extract parameters
    params = model.extract_parameters()

    # Population parameters
    pop_df = pd.DataFrame([
        {
            'parameter': param_name,
            **param_data
        }
        for param_name, param_data in params['population'].items()
    ])

    pop_df.to_csv(f"{filename}_population.csv", index=False)
    print(f"Saved: {filename}_population.csv")

    # Animal parameters
    animal_df = pd.DataFrame([
        {
            'animal': animal,
            **animal_data
        }
        for animal, animal_data in params['animals'].items()
    ])

    animal_df.to_csv(f"{filename}_animals.csv", index=False)
    print(f"Saved: {filename}_animals.csv")

    # Fit statistics
    fit_stats = model.compute_model_fit_stats()

    fit_df = pd.DataFrame([fit_stats['overall']])
    fit_df.to_csv(f"{filename}_fit_stats.csv", index=False)
    print(f"Saved: {filename}_fit_stats.csv")

    if include_trials:
        # Trial-level statistics
        trial_df = fit_stats['trials']
        trial_df.to_csv(f"{filename}_trials.csv", index=False)
        print(f"Saved: {filename}_trials.csv")


def print_model_summary(model, detailed: bool = False):
    """
    Print a formatted summary of model results.

    Parameters
    ----------
    model : HeadingHierarchicalModel
        Fitted model
    detailed : bool, default=False
        Print detailed animal-level results
    """
    params = model.extract_parameters()
    fit_stats = model.compute_model_fit_stats()

    print("\n" + "="*80)
    print(f"MODEL SUMMARY: {model.condition}")
    print("="*80)

    print("\nDATA:")
    print(f"  Animals: {model.data['n_animals']}")
    print(f"  Trials: {model.n_total_trials}")
    print(f"  Total timepoints: {fit_stats['overall']['n_points']:,}")

    print("\nPOPULATION PARAMETERS:")

    for param_name, param_data in params['population'].items():
        print(f"  {param_name:12s}: {param_data['mean']:7.3f}  "
              f"95% HDI: [{param_data['hdi_lower']:6.3f}, {param_data['hdi_upper']:6.3f}]")

    print("\nMODEL FIT:")
    print(f"  R²: {fit_stats['overall']['r_squared']:.3f}")
    print(f"  RMSE: {fit_stats['overall']['rmse']:.3f}")
    print(f"  MAE: {fit_stats['overall']['mae']:.3f}")

    if detailed:
        print("\nANIMAL-SPECIFIC ALPHA:")
        for animal, animal_data in params['animals'].items():
            print(f"  {animal:25s}: {animal_data['alpha_mean']:6.3f}  "
                  f"95% HDI: [{animal_data['alpha_hdi_lower']:6.3f}, {animal_data['alpha_hdi_upper']:6.3f}]")

    print("="*80)


if __name__ == '__main__':
    print("="*80)
    print("Utility Functions Module")
    print("="*80)
    print("\nThis module provides helper functions for the heading model.")
