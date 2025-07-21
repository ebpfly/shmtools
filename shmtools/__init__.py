"""
SHMTools: Python-based Structural Health Monitoring Toolkit

A comprehensive library for structural health monitoring, signal processing,
and damage detection algorithms. Converted from the original MATLAB SHMTools
library developed by Los Alamos National Laboratory.
"""

__version__ = "0.1.0"
__author__ = "SHMTools Development Team"

# Import main modules for easy access
from . import core
from . import features
from . import classification
from . import modal
from . import active_sensing
from . import hardware
from . import plotting
from . import utils

# Common functions for convenience
from .core.spectral import psd_welch, stft
from .core.filtering import filter_signal, bandpass_filter
from .core.statistics import statistical_moments, rms, crest_factor
from .features.time_series import ar_model, arx_model
from .classification.outlier_detection import mahalanobis_distance, pca_detector

__all__ = [
    # Modules
    "core",
    "features", 
    "classification",
    "modal",
    "active_sensing", 
    "hardware",
    "plotting",
    "utils",
    # Common functions
    "psd_welch",
    "stft", 
    "filter_signal",
    "bandpass_filter",
    "statistical_moments",
    "rms",
    "crest_factor",
    "ar_model",
    "arx_model", 
    "mahalanobis_distance",
    "pca_detector",
]