"""
Run per-trial regression analysis for dark conditions.

This script tests whether the positive beta (slope) observed in pooled
dark condition analysis is robust when analyzed at the trial level.

Key approach:
- Instead of pooling all sections across trials, run regression within each trial
- Each trial is an independent "experiment"
- Test if mean beta across trials differs from zero (one-sample t-test)
- Use 2.0 cm/s speed threshold (more trials with sufficient data than 5.0 cm/s)

Author: Analysis generated for Peng et al. 2025
Date: 2025-12-10
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Import analysis and visualization functions
from pure_turn_sections_analysis_relaxed import per_trial_regression
from pure_turn_sections_visualizations_relaxed import (
    plot_trial_beta_distribution,
    plot_light_vs_dark_trial_comparison
)

# Paths
RESULTS_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng\\results'
FIGURES_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng\\figures\\pure_turn_sections_analysis_relaxed\\per_trial'

# Configuration
SPEED_THRESHOLD = 2.0  # cm/s - 787 trials with >= 5 sections
MIN_SECTIONS = 5  # Minimum sections per trial to include


if __name__ == "__main__":
    print("=" * 80)
    print("PER-TRIAL REGRESSION ANALYSIS FOR DARK CONDITIONS")
    print("=" * 80)
    print(f"\nSpeed threshold: {SPEED_THRESHOLD} cm/s")
    print(f"Minimum sections per trial: {MIN_SECTIONS}")

    # Load endpoint data
    endpoints_fn = os.path.join(RESULTS_PATH, "pure_turn_section_endpoints_relaxed.csv")
    print(f"\nLoading endpoint data: {endpoints_fn}")
    endpoints = pd.read_csv(endpoints_fn)
    print(f"Loaded {len(endpoints):,} section endpoints")

    # Create output directory
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # =========================================================================
    # DARK CONDITIONS ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("DARK CONDITIONS")
    print("=" * 80)

    dark_conditions = ['all_dark', 'searchToLeverPath_dark', 'homingFromLeavingLever_dark', 'atLever_dark']
    all_dark_results = []
    all_dark_aggregate = []

    for condition in dark_conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        trial_results, aggregate_stats = per_trial_regression(
            endpoints,
            condition=condition,
            speed_threshold=SPEED_THRESHOLD,
            min_sections=MIN_SECTIONS
        )

        if 'error' in aggregate_stats:
            print(f"  Error: {aggregate_stats['error']}")
            continue

        # Store results
        trial_results['condition'] = condition
        all_dark_results.append(trial_results)
        aggregate_stats['condition'] = condition
        all_dark_aggregate.append(aggregate_stats)

        # Print summary
        print(f"\n  Trials analyzed: {aggregate_stats['n_trials']}")
        print(f"  Total sections: {aggregate_stats['n_total_sections']}")
        print(f"  Mean beta: {aggregate_stats['mean_beta']:.4f}")
        print(f"  95% CI: [{aggregate_stats['ci_low']:.4f}, {aggregate_stats['ci_high']:.4f}]")
        print(f"  One-sample t: {aggregate_stats['one_sample_t']:.3f}")
        print(f"  p-value: {aggregate_stats['one_sample_p']:.6f}")
        print(f"  Positive beta trials: {aggregate_stats['n_positive_beta']} ({aggregate_stats['prop_positive']*100:.1f}%)")

        # Generate visualization
        fig_path = os.path.join(FIGURES_PATH, f"per_trial_beta_distribution_{condition}_speed{SPEED_THRESHOLD}.png")
        fig = plot_trial_beta_distribution(
            trial_results, aggregate_stats, condition, SPEED_THRESHOLD, fig_path
        )
        if fig:
            plt.close(fig)

    # =========================================================================
    # LIGHT CONDITIONS ANALYSIS (for comparison)
    # =========================================================================
    print("\n" + "=" * 80)
    print("LIGHT CONDITIONS (for comparison)")
    print("=" * 80)

    light_conditions = ['all_light', 'searchToLeverPath_light', 'homingFromLeavingLever_light', 'atLever_light']
    all_light_results = []
    all_light_aggregate = []

    for condition in light_conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        trial_results, aggregate_stats = per_trial_regression(
            endpoints,
            condition=condition,
            speed_threshold=SPEED_THRESHOLD,
            min_sections=MIN_SECTIONS
        )

        if 'error' in aggregate_stats:
            print(f"  Error: {aggregate_stats['error']}")
            continue

        # Store results
        trial_results['condition'] = condition
        all_light_results.append(trial_results)
        aggregate_stats['condition'] = condition
        all_light_aggregate.append(aggregate_stats)

        # Print summary
        print(f"\n  Trials analyzed: {aggregate_stats['n_trials']}")
        print(f"  Total sections: {aggregate_stats['n_total_sections']}")
        print(f"  Mean beta: {aggregate_stats['mean_beta']:.4f}")
        print(f"  95% CI: [{aggregate_stats['ci_low']:.4f}, {aggregate_stats['ci_high']:.4f}]")
        print(f"  One-sample t: {aggregate_stats['one_sample_t']:.3f}")
        print(f"  p-value: {aggregate_stats['one_sample_p']:.6f}")
        print(f"  Negative beta trials: {aggregate_stats['n_negative_beta']} ({(1-aggregate_stats['prop_positive'])*100:.1f}%)")

        # Generate visualization
        fig_path = os.path.join(FIGURES_PATH, f"per_trial_beta_distribution_{condition}_speed{SPEED_THRESHOLD}.png")
        fig = plot_trial_beta_distribution(
            trial_results, aggregate_stats, condition, SPEED_THRESHOLD, fig_path
        )
        if fig:
            plt.close(fig)

    # =========================================================================
    # LIGHT VS DARK COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("LIGHT VS DARK COMPARISON")
    print("=" * 80)

    # all_light vs all_dark
    if len(all_light_results) > 0 and len(all_dark_results) > 0:
        # Find 'all_light' and 'all_dark' results
        light_all = None
        light_all_stats = None
        for i, agg in enumerate(all_light_aggregate):
            if agg['condition'] == 'all_light':
                light_all = all_light_results[i]
                light_all_stats = agg
                break

        dark_all = None
        dark_all_stats = None
        for i, agg in enumerate(all_dark_aggregate):
            if agg['condition'] == 'all_dark':
                dark_all = all_dark_results[i]
                dark_all_stats = agg
                break

        if light_all is not None and dark_all is not None:
            fig_path = os.path.join(FIGURES_PATH, f"light_vs_dark_comparison_speed{SPEED_THRESHOLD}.png")
            fig = plot_light_vs_dark_trial_comparison(
                light_all, light_all_stats,
                dark_all, dark_all_stats,
                SPEED_THRESHOLD, fig_path
            )
            if fig:
                plt.close(fig)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Combine all trial-level results
    if len(all_dark_results) > 0:
        dark_trial_df = pd.concat(all_dark_results, ignore_index=True)
        dark_trial_fn = os.path.join(RESULTS_PATH, f"pure_turn_section_per_trial_dark_speed{SPEED_THRESHOLD}.csv")
        dark_trial_df.to_csv(dark_trial_fn, index=False)
        print(f"Saved dark trial results: {dark_trial_fn}")
        print(f"  {len(dark_trial_df)} trial-level regressions")

    if len(all_light_results) > 0:
        light_trial_df = pd.concat(all_light_results, ignore_index=True)
        light_trial_fn = os.path.join(RESULTS_PATH, f"pure_turn_section_per_trial_light_speed{SPEED_THRESHOLD}.csv")
        light_trial_df.to_csv(light_trial_fn, index=False)
        print(f"Saved light trial results: {light_trial_fn}")
        print(f"  {len(light_trial_df)} trial-level regressions")

    # Save aggregate statistics
    if len(all_dark_aggregate) > 0:
        dark_agg_df = pd.DataFrame(all_dark_aggregate)
        dark_agg_fn = os.path.join(RESULTS_PATH, f"pure_turn_section_per_trial_aggregate_dark_speed{SPEED_THRESHOLD}.csv")
        dark_agg_df.to_csv(dark_agg_fn, index=False)
        print(f"Saved dark aggregate stats: {dark_agg_fn}")

    if len(all_light_aggregate) > 0:
        light_agg_df = pd.DataFrame(all_light_aggregate)
        light_agg_fn = os.path.join(RESULTS_PATH, f"pure_turn_section_per_trial_aggregate_light_speed{SPEED_THRESHOLD}.csv")
        light_agg_df.to_csv(light_agg_fn, index=False)
        print(f"Saved light aggregate stats: {light_agg_fn}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'Condition':<35} {'n_trials':>10} {'Mean Beta':>12} {'95% CI':>25} {'p-value':>12} {'Sig?':>8}")
    print("-" * 105)

    for agg in all_light_aggregate + all_dark_aggregate:
        condition = agg['condition']
        n_trials = agg['n_trials']
        mean_beta = agg['mean_beta']
        ci = f"[{agg['ci_low']:.4f}, {agg['ci_high']:.4f}]"
        p = agg['one_sample_p']
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"{condition:<35} {n_trials:>10} {mean_beta:>12.4f} {ci:>25} {p:>12.6f} {sig:>8}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {FIGURES_PATH}")
    print(f"Results saved to: {RESULTS_PATH}")
