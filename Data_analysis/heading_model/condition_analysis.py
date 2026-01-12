"""
Condition-Based Analysis for Hierarchical Heading Model

This module provides functions for fitting models across multiple experimental
conditions and comparing parameter estimates between conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .data_preprocessing import (
    load_and_filter_data,
    structure_hierarchical_data,
    DEFAULT_USEABLE_SESSIONS
)
from .bayesian_model import HeadingHierarchicalModel


def fit_models_by_condition(
    conditions: List[str] = ['all_light', 'all_dark'],
    sessions: Optional[List[str]] = None,
    min_speed: float = 2.0,
    model_kwargs: Optional[Dict] = None,
    fit_kwargs: Optional[Dict] = None,
    max_animals: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, HeadingHierarchicalModel]:
    """
    Fit separate hierarchical models for each experimental condition.

    Parameters
    ----------
    conditions : list of str
        Experimental conditions to fit models for
    sessions : list of str, optional
        Session names to include
    min_speed : float, default=2.0
        Minimum speed threshold
    model_kwargs : dict, optional
        Keyword arguments for model.build_model()
    fit_kwargs : dict, optional
        Keyword arguments for model.fit()
    max_animals : int, optional
        Maximum number of animals (for testing)
    verbose : bool, default=True
        Print progress

    Returns
    -------
    models : dict
        Dictionary mapping condition → HeadingHierarchicalModel
    """
    if sessions is None:
        sessions = DEFAULT_USEABLE_SESSIONS

    if model_kwargs is None:
        model_kwargs = {}

    if fit_kwargs is None:
        fit_kwargs = {
            'draws': 2000,
            'tune': 1000,
            'chains': 4,
            'cores': 4
        }

    models = {}

    for condition in conditions:
        print("\n" + "="*80)
        print(f"FITTING MODEL FOR: {condition}")
        print("="*80)

        try:
            # Load and preprocess data
            if verbose:
                print(f"\n1. Loading data...")

            df = load_and_filter_data(
                conditions=[condition],
                sessions=sessions,
                min_speed=min_speed,
                verbose=verbose
            )

            # Structure for hierarchical model
            if verbose:
                print(f"\n2. Structuring hierarchical data...")

            data_dict = structure_hierarchical_data(
                df,
                smooth_sigma=1.0,
                min_trial_length=50,
                max_animals=max_animals,
                verbose=verbose
            )

            # Build model
            if verbose:
                print(f"\n3. Building PyMC model...")

            model = HeadingHierarchicalModel(data_dict, condition_name=condition)
            model.build_model(**model_kwargs)

            # Fit model
            if verbose:
                print(f"\n4. Fitting with MCMC...")

            model.fit(**fit_kwargs)

            # Check convergence
            if verbose:
                print(f"\n5. Checking convergence...")

            model.check_convergence()

            # Compute fit statistics
            if verbose:
                print(f"\n6. Computing model fit...")

            model.compute_model_fit_stats()

            models[condition] = model

            if verbose:
                print(f"\n✓ Model for {condition} complete!\n")

        except Exception as e:
            print(f"\n✗ Error fitting model for {condition}: {e}")
            print("Skipping this condition...")
            continue

    print("\n" + "="*80)
    print(f"COMPLETED: {len(models)}/{len(conditions)} models fitted successfully")
    print("="*80)

    return models


def compare_conditions(
    model_dict: Dict[str, HeadingHierarchicalModel],
    param: str = 'alpha',
    hdi_prob: float = 0.95
) -> pd.DataFrame:
    """
    Compare parameter estimates across conditions.

    Parameters
    ----------
    model_dict : dict
        Dictionary from fit_models_by_condition()
    param : str, default='alpha'
        Parameter to compare ('alpha', 'mu_alpha', 'mu_gamma', etc.)
    hdi_prob : float, default=0.95
        HDI probability

    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table with means and HDIs
    """
    results = []

    for condition, model in model_dict.items():
        params = model.extract_parameters(hdi_prob=hdi_prob)

        if param == 'alpha':
            # Animal-specific alphas
            for animal, animal_params in params['animals'].items():
                results.append({
                    'condition': condition,
                    'animal': animal,
                    'parameter': 'alpha',
                    'mean': animal_params['alpha_mean'],
                    'sd': animal_params['alpha_sd'],
                    'hdi_lower': animal_params['alpha_hdi_lower'],
                    'hdi_upper': animal_params['alpha_hdi_upper']
                })
        else:
            # Population-level parameter
            if param in params['population']:
                pop_param = params['population'][param]
                results.append({
                    'condition': condition,
                    'animal': 'population',
                    'parameter': param,
                    'mean': pop_param['mean'],
                    'sd': pop_param['sd'],
                    'hdi_lower': pop_param['hdi_lower'],
                    'hdi_upper': pop_param['hdi_upper']
                })

    return pd.DataFrame(results)


def compute_condition_differences(
    model_dict: Dict[str, HeadingHierarchicalModel],
    condition_pairs: Optional[List[Tuple[str, str]]] = None,
    param: str = 'mu_alpha'
) -> pd.DataFrame:
    """
    Compute posterior distributions of differences between conditions.

    Parameters
    ----------
    model_dict : dict
        Dictionary of fitted models
    condition_pairs : list of tuples, optional
        Pairs of conditions to compare. If None, compares all pairs.
    param : str, default='mu_alpha'
        Parameter to compare

    Returns
    -------
    differences_df : pd.DataFrame
        Posterior statistics for condition differences
    """
    if condition_pairs is None:
        conditions = list(model_dict.keys())
        condition_pairs = [(c1, c2) for i, c1 in enumerate(conditions)
                          for c2 in conditions[i+1:]]

    results = []

    for cond1, cond2 in condition_pairs:
        if cond1 not in model_dict or cond2 not in model_dict:
            continue

        # Extract posterior samples
        model1 = model_dict[cond1]
        model2 = model_dict[cond2]

        # Get parameter samples
        samples1 = model1.trace.posterior[param].values.flatten()
        samples2 = model2.trace.posterior[param].values.flatten()

        # Compute difference
        diff = samples1 - samples2

        # Statistics
        mean_diff = np.mean(diff)
        hdi_lower, hdi_upper = np.percentile(diff, [2.5, 97.5])
        prob_positive = np.mean(diff > 0)

        results.append({
            'condition_1': cond1,
            'condition_2': cond2,
            'parameter': param,
            'mean_difference': mean_diff,
            'hdi_lower': hdi_lower,
            'hdi_upper': hdi_upper,
            'prob_positive': prob_positive,
            'significant': (hdi_lower > 0) or (hdi_upper < 0)
        })

    return pd.DataFrame(results)


def summarize_models(model_dict: Dict[str, HeadingHierarchicalModel]) -> pd.DataFrame:
    """
    Create summary table of key parameters across all models.

    Parameters
    ----------
    model_dict : dict
        Dictionary of fitted models

    Returns
    -------
    summary_df : pd.DataFrame
        Summary table with key statistics
    """
    summaries = []

    for condition, model in model_dict.items():
        params = model.extract_parameters()
        fit_stats = model.compute_model_fit_stats()

        summary = {
            'condition': condition,
            'n_animals': model.data['n_animals'],
            'n_trials': model.n_total_trials,
            'mu_alpha_mean': params['population']['mu_alpha']['mean'],
            'mu_alpha_hdi_lower': params['population']['mu_alpha']['hdi_lower'],
            'mu_alpha_hdi_upper': params['population']['mu_alpha']['hdi_upper'],
            'mu_gamma_mean': params['population']['mu_gamma']['mean'],
            'sigma_obs_mean': params['population']['sigma_obs']['mean'],
            'r_squared': fit_stats['overall']['r_squared'],
            'rmse': fit_stats['overall']['rmse']
        }

        summaries.append(summary)

    return pd.DataFrame(summaries)


if __name__ == '__main__':
    print("="*80)
    print("Condition Analysis Module")
    print("="*80)
    print("\nThis module provides functions for multi-condition analysis.")
    print("See notebooks for usage examples.")
