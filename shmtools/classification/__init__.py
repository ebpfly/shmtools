"""
Classification and outlier detection module for SHMTools.

This module provides machine learning algorithms for structural health
monitoring including outlier detection and damage classification methods.
"""

from .outlier_detection import *

__all__ = [
    # Outlier detection functions
    "learn_mahalanobis",
    "score_mahalanobis", 
    "learn_pca",
    "score_pca",
    "learn_svd",
    "score_svd",
    "roc",
]