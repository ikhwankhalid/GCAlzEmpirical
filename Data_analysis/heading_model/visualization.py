"""
Visualization Module for Hierarchical Heading Model

This module provides plotting functions for diagnostics, parameter estimates,
and model fits.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from typing import Dict, Optional, List
from .bayesian_model import HeadingHierarchicalModel

# Set plotting style
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


def plot_trace_diagnostics(
    model: HeadingHierarchicalModel,
    var_names: Optional[List[str]] = None,
    figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Plot MCMC trace diagnostics.

    Parameters
    ----------
    model : HeadingHierarchicalModel
        Fitted model
    var_names : list of str, optional
        Variables to plot. If None, plots hyperparameters.
    figsize : tuple, default=(12, 10)
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    if var_names is None:
        var_names = ['mu_alpha', 'sigma_alpha', 'mu_gamma',
                    'sigma_gamma', 'sigma_obs']

    fig = az.plot_trace(
        model.trace,
        var_names=var_names,
        figsize=figsize,
        compact=True
    )

    fig.suptitle(f'MCMC Trace Diagnostics - {model.condition}',
                fontsize=14, fontweight='bold', y=1.01)

    return fig


def plot_posterior_distributions(
    model: HeadingHierarchicalModel,
    var_names: Optional[List[str]] = None,
    hdi_prob: float = 0.95,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot posterior distributions for key parameters.

    Parameters
    ----------
    model : HeadingHierarchicalModel
        Fitted model
    var_names : list of str, optional
        Variables to plot
    hdi_prob : float, default=0.95
        HDI probability
    figsize : tuple, default=(12, 8)
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    if var_names is None:
        var_names = ['mu_alpha', 'sigma_alpha', 'mu_gamma', 'sigma_gamma']

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i, var_name in enumerate(var_names):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot posterior
        az.plot_posterior(
            model.trace,
            var_names=[var_name],
            hdi_prob=hdi_prob,
            ax=ax
        )

        ax.set_title(var_name, fontweight='bold')

    fig.suptitle(f'Posterior Distributions - {model.condition}',
                fontsize=14, fontweight='bold')
    fig.tight_layout()

    return fig


def plot_animal_parameters(
    model: HeadingHierarchicalModel,
    param: str = 'alpha',
    hdi_prob: float = 0.95,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot animal-specific parameter estimates with HDI.

    Parameters
    ----------
    model : HeadingHierarchicalModel
        Fitted model
    param : str, default='alpha'
        Parameter to plot
    hdi_prob : float, default=0.95
        HDI probability
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    params = model.extract_parameters(hdi_prob=hdi_prob)

    # Extract data
    animals = []
    means = []
    hdi_lowers = []
    hdi_uppers = []

    for animal, animal_params in params['animals'].items():
        animals.append(animal)
        means.append(animal_params[f'{param}_mean'])
        hdi_lowers.append(animal_params[f'{param}_hdi_lower'])
        hdi_uppers.append(animal_params[f'{param}_hdi_upper'])

    # Sort by mean value
    sort_idx = np.argsort(means)
    animals = [animals[i] for i in sort_idx]
    means = [means[i] for i in sort_idx]
    hdi_lowers = [hdi_lowers[i] for i in sort_idx]
    hdi_uppers = [hdi_uppers[i] for i in sort_idx]

    # Compute error bars
    errors_lower = np.array(means) - np.array(hdi_lowers)
    errors_upper = np.array(hdi_uppers) - np.array(means)
    errors = np.array([errors_lower, errors_upper])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(animals))

    ax.barh(y_pos, means, xerr=errors, capsize=3,
           color='steelblue', alpha=0.7, edgecolor='black')

    # Add reference line at perfect integration (α=1)
    if param == 'alpha':
        ax.axvline(x=1.0, color='red', linestyle='--',
                  linewidth=2, label='Perfect integration')
        ax.legend()

    ax.set_yticks(y_pos)
    ax.set_yticklabels(animals)
    ax.set_xlabel(f'{param} (mean ± {hdi_prob*100:.0f}% HDI)', fontweight='bold')
    ax.set_ylabel('Animal', fontweight='bold')
    ax.set_title(f'Animal-Specific {param} Estimates - {model.condition}',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()

    return fig


def plot_model_fits(
    model: HeadingHierarchicalModel,
    n_trials: int = 6,
    random_seed: Optional[int] = None,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Plot observed vs. predicted headings for random trials.

    Parameters
    ----------
    model : HeadingHierarchicalModel
        Fitted model
    n_trials : int, default=6
        Number of trials to plot
    random_seed : int, optional
        Random seed for trial selection
    figsize : tuple, default=(15, 10)
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get posterior means
    posterior_means = model.trace.posterior.mean(dim=['chain', 'draw'])
    alpha_means = posterior_means['alpha'].values
    gamma_means = posterior_means['gamma'].values
    theta0_means = posterior_means['theta_0'].values

    # Randomly select trials
    trial_indices = np.random.choice(model.n_total_trials,
                                    size=min(n_trials, model.n_total_trials),
                                    replace=False)

    # Create subplots
    ncols = 3
    nrows = int(np.ceil(n_trials / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_trials > 1 else [axes]

    for i, trial_idx in enumerate(trial_indices):
        animal_idx = model.trial_to_animal[trial_idx]
        trial_id = model.trial_to_id[trial_idx]
        animal = model.data['animals'][animal_idx]

        # Get data
        Omega = model.data['omega_integrated'][animal][trial_id]
        t = model.data['time'][animal][trial_id]
        theta_obs = model.data['theta_obs'][animal][trial_id]

        # Predict
        theta_pred = (theta0_means[trial_idx] +
                     alpha_means[animal_idx] * Omega +
                     gamma_means[trial_idx] * t)
        theta_pred = np.arctan2(np.sin(theta_pred), np.cos(theta_pred))

        # Plot
        ax = axes[i]

        ax.plot(t, theta_obs, 'o-', alpha=0.6, markersize=3,
               label='Observed', color='darkblue')
        ax.plot(t, theta_pred, 'o-', alpha=0.8, markersize=3,
               label='Predicted', color='orange')

        # Compute circular RMSE (accounts for circular nature of angles)
        angular_diff = np.arctan2(np.sin(theta_obs - theta_pred), np.cos(theta_obs - theta_pred))
        circular_rmse = np.sqrt(np.mean(angular_diff**2))

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (rad)')
        ax.set_title(f'{animal} - Trial {trial_id.split("_")[-1]}\nCircular RMSE={circular_rmse:.3f}',
                    fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Model Fits - {model.condition}',
                fontsize=14, fontweight='bold')
    fig.tight_layout()

    return fig


def plot_posterior_predictive_checks(
    model: HeadingHierarchicalModel,
    ppc: az.InferenceData,
    n_samples: int = 50,
    figsize: tuple = (12, 5)
) -> plt.Figure:
    """
    Plot posterior predictive checks.

    Parameters
    ----------
    model : HeadingHierarchicalModel
        Fitted model
    ppc : az.InferenceData
        Posterior predictive samples
    n_samples : int, default=50
        Number of posterior samples to plot
    figsize : tuple, default=(12, 5)
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Histogram of observations vs. predictions
    ax = axes[0]

    # Collect all observed data
    all_obs = []
    for animal in model.data['animals']:
        for trial_id in model.data['trials_per_animal'][animal]:
            all_obs.extend(model.data['theta_obs'][animal][trial_id])

    all_obs = np.array(all_obs)

    # Plot observed distribution
    ax.hist(all_obs, bins=50, alpha=0.5, label='Observed',
           color='darkblue', density=True)

    # Plot predicted distributions (multiple samples)
    # Note: This requires extracting predictions from ppc
    # For simplicity, showing placeholder

    ax.set_xlabel('Heading (rad)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Predictive Check', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Residuals histogram
    ax = axes[1]

    # Compute residuals
    posterior_means = model.trace.posterior.mean(dim=['chain', 'draw'])
    alpha_means = posterior_means['alpha'].values
    gamma_means = posterior_means['gamma'].values
    theta0_means = posterior_means['theta_0'].values

    all_residuals = []

    for trial_idx in range(model.n_total_trials):
        animal_idx = model.trial_to_animal[trial_idx]
        trial_id = model.trial_to_id[trial_idx]
        animal = model.data['animals'][animal_idx]

        Omega = model.data['omega_integrated'][animal][trial_id]
        t = model.data['time'][animal][trial_id]
        theta_obs = model.data['theta_obs'][animal][trial_id]

        theta_pred = (theta0_means[trial_idx] +
                     alpha_means[animal_idx] * Omega +
                     gamma_means[trial_idx] * t)

        residuals = theta_obs - theta_pred
        all_residuals.extend(residuals)

    all_residuals = np.array(all_residuals)

    ax.hist(all_residuals, bins=50, color='darkred', alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Residuals (rad)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution', fontweight='bold')
    ax.grid(alpha=0.3)

    fig.suptitle(f'Posterior Predictive Checks - {model.condition}',
                fontsize=14, fontweight='bold')
    fig.tight_layout()

    return fig


def plot_condition_comparison(
    model_dict: Dict[str, HeadingHierarchicalModel],
    param: str = 'mu_alpha',
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Compare parameter estimates across conditions.

    Parameters
    ----------
    model_dict : dict
        Dictionary of models
    param : str, default='mu_alpha'
        Parameter to compare
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    conditions = list(model_dict.keys())
    means = []
    hdi_lowers = []
    hdi_uppers = []

    for condition in conditions:
        model = model_dict[condition]
        params = model.extract_parameters()

        means.append(params['population'][param]['mean'])
        hdi_lowers.append(params['population'][param]['hdi_lower'])
        hdi_uppers.append(params['population'][param]['hdi_upper'])

    # Compute error bars
    errors_lower = np.array(means) - np.array(hdi_lowers)
    errors_upper = np.array(hdi_uppers) - np.array(means)
    errors = np.array([errors_lower, errors_upper])

    # Plot
    x_pos = np.arange(len(conditions))

    ax.bar(x_pos, means, yerr=errors, capsize=5,
          color='steelblue', alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel(f'{param} (mean ± 95% HDI)', fontweight='bold')
    ax.set_title(f'{param} Across Conditions', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    if param == 'mu_alpha':
        ax.axhline(y=1.0, color='red', linestyle='--',
                  linewidth=2, label='Perfect integration')
        ax.legend()

    fig.tight_layout()

    return fig


if __name__ == '__main__':
    print("="*80)
    print("Visualization Module")
    print("="*80)
    print("\nThis module provides plotting functions for model results.")
    print("See notebooks for usage examples.")
