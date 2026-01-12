"""
Hierarchical Bayesian Model for Heading Dynamics

This module implements the PyMC hierarchical model for estimating heading
dynamics parameters (α, γ, θ₀).

Model equation:
    θ̂'(t) = θ₀ + α·Ω(t) + γ·t

where:
    θ̂'(t): Predicted decoded heading
    θ₀: Initial heading (trial-specific)
    α: Gain parameter (animal-specific)
    Ω(t): Integrated angular velocity
    γ: Drift parameter (trial-specific)
"""

import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from typing import Dict, Optional, Tuple
import pickle
import warnings


class HeadingHierarchicalModel:
    """
    Hierarchical Bayesian model for heading dynamics.

    Estimates animal-specific gain (α) and trial-specific drift (γ) and
    initial heading (θ₀) parameters that best predict decoded heading from
    integrated angular velocity.

    Attributes
    ----------
    data : dict
        Structured data from structure_hierarchical_data()
    condition : str
        Name/label for this model instance
    model : pm.Model
        PyMC model object
    trace : az.InferenceData
        MCMC trace after fitting
    """

    def __init__(self, data_dict: Dict, condition_name: str = "all"):
        """
        Initialize hierarchical model.

        Parameters
        ----------
        data_dict : dict
            Structured data from data_preprocessing.structure_hierarchical_data()
        condition_name : str, default="all"
            Name for this model instance (e.g., 'all_light', 'all_dark')
        """
        self.data = data_dict
        self.condition = condition_name
        self.model = None
        self.trace = None
        self.vectorized_data = None  # Cache vectorized data for reuse

        # Build trial index mapping
        self._build_trial_index()

    def _build_trial_index(self):
        """Build mapping from trial indices to animal/trial IDs."""
        self.trial_to_animal = []
        self.trial_to_id = []
        self.animal_to_idx = {animal: i for i, animal in enumerate(self.data['animals'])}

        for animal_idx, animal in enumerate(self.data['animals']):
            for trial_id in self.data['trials_per_animal'][animal]:
                self.trial_to_animal.append(animal_idx)
                self.trial_to_id.append(trial_id)

        self.n_total_trials = len(self.trial_to_animal)

    def _prepare_vectorized_data(self):
        """
        Prepare vectorized data structures for efficient model computation.

        Concatenates all trial observations into single arrays and creates
        index mappings for animal and trial parameters.

        Returns
        -------
        vectorized_data : dict
            Dictionary containing:
            - omega_all: Concatenated integrated angular velocity
            - t_all: Concatenated time arrays
            - theta_obs_all: Concatenated observed heading
            - speed_all: Concatenated speed values
            - mean_speed: Mean speed across all observations
            - animal_idx_all: Animal index for each observation
            - trial_idx_all: Trial index for each observation
            - n_obs: Total number of observations
        """
        omega_list = []
        t_list = []
        theta_obs_list = []
        speed_list = []
        animal_idx_list = []
        trial_idx_list = []

        for trial_idx in range(self.n_total_trials):
            animal_idx = self.trial_to_animal[trial_idx]
            trial_id = self.trial_to_id[trial_idx]
            animal = self.data['animals'][animal_idx]

            # Get data for this trial
            Omega = self.data['omega_integrated'][animal][trial_id]
            t = self.data['time'][animal][trial_id]
            theta_obs = self.data['theta_obs'][animal][trial_id]
            speed = self.data['speed'][animal][trial_id]

            n_points = len(theta_obs)

            # Append to lists
            omega_list.append(Omega)
            t_list.append(t)
            theta_obs_list.append(theta_obs)
            speed_list.append(speed)
            animal_idx_list.append(np.full(n_points, animal_idx, dtype=int))
            trial_idx_list.append(np.full(n_points, trial_idx, dtype=int))

        # Concatenate all arrays
        speed_all = np.concatenate(speed_list)

        vectorized_data = {
            'omega_all': np.concatenate(omega_list),
            't_all': np.concatenate(t_list),
            'theta_obs_all': np.concatenate(theta_obs_list),
            'speed_all': speed_all,
            'mean_speed': np.mean(speed_all),
            'animal_idx_all': np.concatenate(animal_idx_list),
            'trial_idx_all': np.concatenate(trial_idx_list),
            'n_obs': sum(len(x) for x in theta_obs_list)
        }

        return vectorized_data

    def build_model(self,
                   alpha_prior_mean: float = 1.0,
                   alpha_prior_sd: float = 0.5,
                   gamma_prior_sd: float = 0.1,
                   theta0_prior_sd: float = 1.0,
                   obs_noise_beta: float = 0.1,
                   use_noncentered: bool = False) -> pm.Model:
        """
        Build PyMC hierarchical model.

        Parameters
        ----------
        alpha_prior_mean : float, default=1.0
            Prior mean for gain parameter (expect perfect integration)
        alpha_prior_sd : float, default=0.5
            Prior SD for gain parameter prior
        gamma_prior_sd : float, default=0.1
            Prior SD for drift parameter
        theta0_prior_sd : float, default=1.0
            Prior SD for initial heading
        obs_noise_beta : float, default=0.1
            Scale parameter for observation noise (HalfCauchy)
        use_noncentered : bool, default=False
            Use non-centered parameterization for better sampling

        Returns
        -------
        model : pm.Model
            Configured PyMC model
        """
        print(f"\nBuilding hierarchical model for: {self.condition}")
        print(f"  Animals: {self.data['n_animals']}")
        print(f"  Trials: {self.n_total_trials}")
        
        # Prepare vectorized data BEFORE entering PyMC context
        print("  Preparing vectorized data...")
        vdata = self._prepare_vectorized_data()
        self.vectorized_data = vdata  # Cache for later use
        print(f"  Total observations: {vdata['n_obs']}")

        with pm.Model() as model:
            # ============================================================
            # HYPERPARAMETERS (Population Level)
            # ============================================================

            # Alpha (gain) - animal-specific
            mu_alpha = pm.Normal('mu_alpha',
                                mu=alpha_prior_mean,
                                sigma=alpha_prior_sd)
            sigma_alpha = pm.HalfCauchy('sigma_alpha', beta=0.25)

            # Gamma (drift) - trial-specific with global mean
            mu_gamma = pm.Normal('mu_gamma', mu=0.0, sigma=gamma_prior_sd)
            sigma_gamma = pm.HalfCauchy('sigma_gamma', beta=gamma_prior_sd)

            # Theta_0 (initial heading) - trial-specific
            mu_theta0 = pm.Normal('mu_theta0', mu=0.0, sigma=np.pi)
            sigma_theta0 = pm.HalfCauchy('sigma_theta0', beta=theta0_prior_sd)

            # ============================================================
            # OBSERVATION NOISE (SPEED-DEPENDENT)
            # ============================================================

            # Base observation noise (at reference speed)
            sigma_base = pm.HalfCauchy('sigma_base', beta=obs_noise_beta)

            # Speed-noise relationship parameter
            # beta_speed > 0: noise decreases with speed
            # beta_speed = 0: speed-independent noise (reverts to old model)
            beta_speed = pm.HalfNormal('beta_speed', sigma=1.0)

            # ============================================================
            # ANIMAL-LEVEL PARAMETERS
            # ============================================================

            if use_noncentered:
                # Non-centered parameterization for better sampling
                alpha_offset = pm.Normal('alpha_offset',
                                        mu=0, sigma=1,
                                        shape=self.data['n_animals'])
                alpha = pm.Deterministic('alpha',
                                        mu_alpha + sigma_alpha * alpha_offset)
            else:
                # Centered parameterization
                alpha = pm.Normal('alpha',
                                mu=mu_alpha,
                                sigma=sigma_alpha,
                                shape=self.data['n_animals'])

            # ============================================================
            # TRIAL-LEVEL PARAMETERS
            # ============================================================

            if use_noncentered:
                # Non-centered parameterization
                gamma_offset = pm.Normal('gamma_offset',
                                        mu=0, sigma=1,
                                        shape=self.n_total_trials)
                gamma = pm.Deterministic('gamma',
                                        mu_gamma + sigma_gamma * gamma_offset)

                theta0_offset = pm.Normal('theta0_offset',
                                         mu=0, sigma=1,
                                         shape=self.n_total_trials)
                theta_0 = pm.Deterministic('theta_0',
                                          mu_theta0 + sigma_theta0 * theta0_offset)
            else:
                # Centered parameterization
                gamma = pm.Normal('gamma',
                                mu=mu_gamma,
                                sigma=sigma_gamma,
                                shape=self.n_total_trials)

                theta_0 = pm.Normal('theta_0',
                                  mu=mu_theta0,
                                  sigma=sigma_theta0,
                                  shape=self.n_total_trials)

            # ============================================================
            # FORWARD MODEL & LIKELIHOOD (VECTORIZED)
            # ============================================================

            # Vectorized forward model: θ̂'(t) = θ₀ + α·Ω(t) + γ·t
            # Use advanced indexing to map parameters to observations
            theta_pred_unwrapped = (
                theta_0[vdata['trial_idx_all']] +
                alpha[vdata['animal_idx_all']] * vdata['omega_all'] +
                gamma[vdata['trial_idx_all']] * vdata['t_all']
            )

            # Wrap predictions to [-π, π] to respect circular nature
            import pytensor.tensor as pt
            theta_pred = pt.arctan2(pt.sin(theta_pred_unwrapped),
                                    pt.cos(theta_pred_unwrapped))

            # Speed-dependent observation noise
            # sigma_obs(speed) = sigma_base * exp(-beta_speed * speed_normalized)
            # At low speed: sigma_obs is large (high uncertainty)
            # At high speed: sigma_obs is small (low uncertainty)
            # Clip speeds to minimum threshold (0.5 cm/s) to avoid numerical issues
            speed_clipped = pt.maximum(vdata['speed_all'], 0.5)  # Minimum 0.5 cm/s
            speed_normalized = speed_clipped / vdata['mean_speed']
            sigma_obs = sigma_base * pt.exp(-beta_speed * speed_normalized)

            # Use von Mises distribution for circular data
            # von Mises is the circular analog of the Normal distribution
            # Convert sigma to kappa (concentration parameter): kappa ≈ 1/sigma²
            kappa = 1.0 / (sigma_obs ** 2 + 1e-6)  # Add small constant for stability

            pm.VonMises('theta_obs_all',
                        mu=theta_pred,
                        kappa=kappa,
                        observed=vdata['theta_obs_all'])

        self.model = model
        print("Model built successfully!")

        return model

    def fit(self,
           draws: int = 2000,
           tune: int = 1000,
           chains: int = 4,
           target_accept: float = 0.95,
           cores: int = 4,
           random_seed: Optional[int] = None) -> az.InferenceData:
        """
        Fit model using MCMC (NUTS sampler).

        Parameters
        ----------
        draws : int, default=2000
            Number of samples to draw (per chain)
        tune : int, default=1000
            Number of tuning/warmup samples
        chains : int, default=4
            Number of MCMC chains
        target_accept : float, default=0.95
            Target acceptance rate for NUTS
        cores : int, default=4
            Number of CPU cores for parallel chains
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        trace : az.InferenceData
            Posterior samples and diagnostics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"\nFitting model with MCMC...")
        print(f"  Chains: {chains}")
        print(f"  Tune: {tune}, Draws: {draws}")
        print(f"  Target accept: {target_accept}")

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True
            )

        print("Sampling complete!")
        return self.trace

    def check_convergence(self, var_names: Optional[list] = None) -> pd.DataFrame:
        """
        Check MCMC convergence diagnostics.

        Parameters
        ----------
        var_names : list of str, optional
            Variables to check. If None, checks hyperparameters and first few alphas.

        Returns
        -------
        summary : pd.DataFrame
            Summary statistics with R-hat, ESS, etc.
        """
        if self.trace is None:
            raise ValueError("No trace available. Run fit() first.")

        if var_names is None:
            # Check hyperparameters and first 5 animal alphas
            var_names = ['mu_alpha', 'sigma_alpha',
                        'mu_gamma', 'sigma_gamma',
                        'mu_theta0', 'sigma_theta0',
                        'sigma_base', 'beta_speed']
            # Add first few alphas
            n_alpha_check = min(5, self.data['n_animals'])
            var_names += [f'alpha' for _ in range(n_alpha_check)]

        print("\n" + "="*80)
        print("Convergence Diagnostics")
        print("="*80)

        summary = az.summary(self.trace, var_names=var_names, hdi_prob=0.95)

        # Check R-hat
        max_rhat = summary['r_hat'].max()
        print(f"\nMax R-hat: {max_rhat:.4f} (should be < 1.01)")

        if max_rhat > 1.01:
            print("  ⚠️  WARNING: Convergence issues detected!")
            problem_vars = summary[summary['r_hat'] > 1.01].index.tolist()
            print(f"  Problem variables: {problem_vars}")
        else:
            print("  ✓ All R-hat values < 1.01")

        # Check effective sample size
        min_ess_bulk = summary['ess_bulk'].min()
        min_ess_tail = summary['ess_tail'].min()
        print(f"\nMin ESS (bulk): {min_ess_bulk:.0f} (should be > 400)")
        print(f"Min ESS (tail): {min_ess_tail:.0f} (should be > 400)")

        if min_ess_bulk < 400 or min_ess_tail < 400:
            print("  ⚠️  WARNING: Low effective sample size!")
        else:
            print("  ✓ Adequate effective sample size")

        print("="*80)

        return summary

    def posterior_predictive_check(self, n_samples: int = 100) -> az.InferenceData:
        """
        Generate posterior predictive samples.

        Parameters
        ----------
        n_samples : int, default=100
            Number of posterior samples to use for predictions

        Returns
        -------
        ppc : az.InferenceData
            Posterior predictive samples
        """
        if self.trace is None:
            raise ValueError("No trace available. Run fit() first.")

        print(f"\nGenerating posterior predictive samples (n={n_samples})...")

        with self.model:
            ppc = pm.sample_posterior_predictive(
                self.trace,
                samples=n_samples,
                return_inferencedata=True
            )

        print("Posterior predictive check complete!")
        return ppc

    def extract_parameters(self, hdi_prob: float = 0.95) -> Dict:
        """
        Extract parameter estimates with credible intervals.

        Parameters
        ----------
        hdi_prob : float, default=0.95
            Probability for highest density interval

        Returns
        -------
        results : dict
            Dictionary with population, animal, and trial-level parameters
        """
        if self.trace is None:
            raise ValueError("No trace available. Run fit() first.")

        summary = az.summary(self.trace, hdi_prob=hdi_prob)

        results = {
            'condition': self.condition,
            'population': {},
            'animals': {},
            'summary': summary
        }

        # Extract population-level parameters
        for param in ['mu_alpha', 'sigma_alpha', 'mu_gamma', 'sigma_gamma',
                     'mu_theta0', 'sigma_theta0', 'sigma_base', 'beta_speed']:
            if param in summary.index:
                lower_col = f'hdi_{(1-hdi_prob)/2*100:.1f}%'
                upper_col = f'hdi_{(1 - (1-hdi_prob)/2)*100:.1f}%'
                results['population'][param] = {
                    'mean': summary.loc[param, 'mean'],
                    'sd': summary.loc[param, 'sd'],
                    'hdi_lower': summary.loc[param, lower_col],
                    'hdi_upper': summary.loc[param, upper_col]
                }

        # Extract animal-specific alphas
        for i, animal in enumerate(self.data['animals']):
            alpha_key = f'alpha[{i}]'
            if alpha_key in summary.index:
                lower_col = f'hdi_{(1-hdi_prob)/2*100:.1f}%'
                upper_col = f'hdi_{(1 - (1-hdi_prob)/2)*100:.1f}%'
                results['animals'][animal] = {
                    'alpha_mean': summary.loc[alpha_key, 'mean'],
                    'alpha_sd': summary.loc[alpha_key, 'sd'],
                    'alpha_hdi_lower': summary.loc[alpha_key, lower_col],
                    'alpha_hdi_upper': summary.loc[alpha_key, upper_col],
                    'n_trials': len(self.data['trials_per_animal'][animal])
                }

        return results

    def compute_model_fit_stats(self) -> Dict:
        """
        Compute model fit statistics (RMSE, R²) using circular residuals.
        
        Uses circular distance for residuals to properly handle angular data.

        Returns
        -------
        stats : dict
            Dictionary with overall and per-trial fit statistics
        """
        if self.trace is None:
            raise ValueError("No trace available. Run fit() first.")

        print("\nComputing model fit statistics (circular residuals)...")

        # Get posterior mean predictions
        posterior_means = self.trace.posterior.mean(dim=['chain', 'draw'])

        alpha_means = posterior_means['alpha'].values
        gamma_means = posterior_means['gamma'].values
        theta0_means = posterior_means['theta_0'].values

        # Compute predictions and residuals for each trial
        all_residuals_circular = []
        all_obs = []
        trial_stats = []

        for trial_idx in range(self.n_total_trials):
            animal_idx = self.trial_to_animal[trial_idx]
            trial_id = self.trial_to_id[trial_idx]
            animal = self.data['animals'][animal_idx]

            # Get data
            Omega = self.data['omega_integrated'][animal][trial_id]
            t = self.data['time'][animal][trial_id]
            theta_obs = self.data['theta_obs'][animal][trial_id]

            # Predict (unwrapped)
            theta_pred_unwrapped = (theta0_means[trial_idx] +
                                    alpha_means[animal_idx] * Omega +
                                    gamma_means[trial_idx] * t)
            
            # Wrap predictions to [-π, π]
            theta_pred = np.arctan2(np.sin(theta_pred_unwrapped),
                                    np.cos(theta_pred_unwrapped))

            # Circular residuals: wrap difference to [-π, π]
            residuals = theta_obs - theta_pred
            residuals_circular = np.arctan2(np.sin(residuals),
                                           np.cos(residuals))

            all_residuals_circular.extend(residuals_circular)
            all_obs.extend(theta_obs)

            # Per-trial stats (using circular residuals)
            rmse = np.sqrt(np.mean(residuals_circular**2))
            mae = np.mean(np.abs(residuals_circular))

            trial_stats.append({
                'animal': animal,
                'trial_id': trial_id,
                'rmse': rmse,
                'mae': mae,
                'n_points': len(residuals_circular)
            })

        # Overall statistics (using circular residuals)
        all_residuals_circular = np.array(all_residuals_circular)
        all_obs = np.array(all_obs)

        overall_rmse = np.sqrt(np.mean(all_residuals_circular**2))
        overall_mae = np.mean(np.abs(all_residuals_circular))

        # R² using circular residuals
        # For circular data, we compute R² based on circular variance
        ss_res = np.sum(all_residuals_circular**2)
        
        # Circular mean for observed data
        mean_obs = np.arctan2(np.mean(np.sin(all_obs)),
                             np.mean(np.cos(all_obs)))
        obs_dev = all_obs - mean_obs
        obs_dev_circular = np.arctan2(np.sin(obs_dev), np.cos(obs_dev))
        ss_tot = np.sum(obs_dev_circular**2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        stats = {
            'overall': {
                'rmse': overall_rmse,
                'mae': overall_mae,
                'r_squared': r_squared,
                'n_trials': self.n_total_trials,
                'n_points': len(all_residuals_circular)
            },
            'trials': pd.DataFrame(trial_stats)
        }

        print(f"  Overall RMSE (circular): {overall_rmse:.4f} rad")
        print(f"  Overall MAE (circular): {overall_mae:.4f} rad")
        print(f"  R² (circular): {r_squared:.4f}")

        return stats

    def save(self, filename: str):
        """Save model and trace to file."""
        data = {
            'condition': self.condition,
            'data_dict': self.data,
            'trace': self.trace,
            'model': self.model
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to: {filename}")

    @classmethod
    def load(cls, filename: str):
        """Load model and trace from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        model_instance = cls(data['data_dict'], data['condition'])
        model_instance.trace = data['trace']
        model_instance.model = data['model']

        print(f"Model loaded from: {filename}")
        return model_instance


if __name__ == '__main__':
    # Example usage
    print("="*80)
    print("Hierarchical Bayesian Model - Example Usage")
    print("="*80)

    # This would typically follow data preprocessing
    print("\nNote: This example requires preprocessed data.")
    print("See data_preprocessing.py for data preparation.")
