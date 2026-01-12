"""
Run statistical validation on sparse high-R^2 conditions.

This script validates the bootstrap CIs and permutation tests for the
conditions with highest per-animal R^2 values.
"""

import pandas as pd
import os

# Import validation functions
from pure_turn_sections_analysis_relaxed import validate_sparse_conditions
from pure_turn_sections_visualizations_relaxed import plot_validation_forest

# Paths
RESULTS_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng\\results'
FIGURES_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng\\figures\\pure_turn_sections_analysis_relaxed\\per_animal'

if __name__ == "__main__":
    print("="*80)
    print("STATISTICAL VALIDATION OF HIGH R^2 CONDITIONS")
    print("="*80)

    # Load endpoint data
    endpoints_fn = os.path.join(RESULTS_PATH, "pure_turn_section_endpoints_relaxed.csv")
    print(f"\nLoading endpoint data: {endpoints_fn}")
    endpoints = pd.read_csv(endpoints_fn)
    print(f"Loaded {len(endpoints):,} section endpoints")

    # Run validation
    print("\n" + "="*80)
    print("RUNNING BOOTSTRAP CIs AND PERMUTATION TESTS")
    print("="*80)

    validation_results = validate_sparse_conditions(
        endpoints,
        n_bootstrap=1000,
        n_permutations=10000
    )

    # Save results
    output_fn = os.path.join(RESULTS_PATH, "pure_turn_section_validation_relaxed.csv")
    validation_results.to_csv(output_fn, index=False)
    print(f"\nValidation results saved to: {output_fn}")

    # Display summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(validation_results[['condition', 'speed_threshold', 'n', 'beta', 'beta_ci_low',
                               'beta_ci_high', 'perm_p_value', 'ci_excludes_zero']].to_string())

    # Generate forest plot
    print("\n" + "="*80)
    print("GENERATING FOREST PLOT")
    print("="*80)

    os.makedirs(FIGURES_PATH, exist_ok=True)
    forest_path = os.path.join(FIGURES_PATH, "validation_forest_plot.png")
    fig = plot_validation_forest(validation_results, forest_path)

    if fig:
        import matplotlib.pyplot as plt
        plt.close(fig)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
