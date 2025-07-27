"""
Classification and outlier detection module for SHMTools.

This module provides machine learning algorithms for structural health
monitoring including outlier detection and damage classification methods.
"""

from .outlier_detection import *
from .nonparametric import *
from .semiparametric import *

# Import NLPCA functions if TensorFlow is available
try:
    from .nlpca import learn_nlpca_shm, score_nlpca_shm
    NLPCA_AVAILABLE = True
except ImportError:
    NLPCA_AVAILABLE = False

__all__ = [
    # Outlier detection functions
    "learn_mahalanobis_shm",
    "score_mahalanobis_shm", 
    "learn_pca",
    "score_pca",
    "learn_svd_shm",
    "score_svd_shm",
    "learn_factor_analysis_shm",
    "score_factor_analysis_shm",
    "roc",
    "roc_shm",
    # Nonparametric detection functions
    "gaussian_kernel_shm",
    "epanechnikov_kernel_shm",
    "uniform_kernel_shm",
    "quartic_kernel_shm",
    "triangle_kernel_shm",
    "triweight_kernel_shm",
    "cosine_kernel_shm",
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

# Add NLPCA functions to __all__ if available
if NLPCA_AVAILABLE:
    __all__.extend(["learn_nlpca_shm", "score_nlpca_shm"])