"""
Classification and outlier detection module for SHMTools.

This module provides machine learning algorithms for structural health
monitoring including outlier detection and damage classification methods.
"""

from .outlier_detection import *

__all__ = [
    # MATLAB compatible functions
    "learn_mahalanobis_shm",
    "score_mahalanobis_shm", 
    "learn_pca_shm",
    "score_pca_shm",
    "learn_svd_shm",
    "score_svd_shm",
    "roc_shm",
    # Modern Python interface
    "mahalanobis_distance",
    "pca_detector", 
    "learn_mahalanobis",
    "score_mahalanobis",
    "learn_pca",
    "score_pca",
]