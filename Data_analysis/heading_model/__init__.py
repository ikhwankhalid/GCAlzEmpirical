"""
Hierarchical Bayesian Model for Heading Dynamics

This package implements a hierarchical Bayesian model to estimate parameters
(α, γ, θ₀) that minimize the difference between predicted and observed headings
across multiple animals and trials.

Model equation:
    θ̂'(t) = θ₀ + α·Ω(t) + γ·t

where:
    θ̂'(t): Predicted decoded heading at time t
    θ₀: Initial heading (trial-specific)
    α: Gain parameter (animal-specific)
    Ω(t): Integrated angular velocity from true movement
    γ: Drift parameter (trial-specific)
"""

__version__ = "0.1.0"
__author__ = "Peng et al. 2025"

from . import data_preprocessing
from . import bayesian_model
from . import utils

__all__ = ['data_preprocessing', 'bayesian_model', 'utils']
