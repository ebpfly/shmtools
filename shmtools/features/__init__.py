"""
Feature extraction module for SHMTools.

This module contains functions for extracting damage-sensitive features
from structural response signals including time series modeling,
modal analysis, and active sensing features.
"""

from .time_series import *
# from .modal_features import *
# from .active_sensing_features import *

__all__ = [
    # Time series features - MATLAB compatible
    "ar_model_shm",
    # Time series features - Modern Python interface
    "ar_model",
    "arx_model", 
    "ar_model_order",
    # Modal features (TODO)
    # Active sensing features (TODO)
]