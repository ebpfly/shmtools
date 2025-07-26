"""
Classification and outlier detection module for SHMTools.

This module provides machine learning algorithms for structural health
monitoring including outlier detection and damage classification methods.
"""

from .outlier_detection import *
from .nonparametric import *
from .semiparametric import *

__all__ = [
    # Outlier detection functions
    "learn_mahalanobis",
    "score_mahalanobis", 
    "learn_pca",
    "score_pca",
    "learn_svd",
    "score_svd",
    "learn_factor_analysis",
    "score_factor_analysis",
    "roc",
    # Nonparametric detection functions
    "gaussian_kernel_shm",
    "epanechnikov_kernel_shm",
    "learn_kernel_density_shm",
    "score_kernel_density_shm",
    "roc_shm",
    # Semi-parametric detection functions
    "k_medians_shm",
    "learn_gmm_shm",
    "score_gmm_shm",
    "learn_gmm_semiparametric_model_shm",
    "score_gmm_semiparametric_model_shm",
]