"""
Core signal processing functions for SHMTools.

This module provides fundamental signal processing operations including
spectral analysis, filtering, statistics, and signal conditioning.
"""

# Spectral analysis functions
from .spectral import psd_welch_shm, stft_shm, cwt_analysis_shm

# Signal filtering functions
from .signal_filtering import (
    residual_signal_shm,
    difference_signal_shm,
    bandpass_condition_signal_shm,
    gear_mesh_filter_shm,
    envelope_signal_shm,
)

# Signal processing functions
from .signal_processing import ars_tach_shm, fir1_shm

# Statistical analysis functions
from .statistics import (
    fm0_shm,
    fm4_shm,
    peak_factor_shm,
    impulse_factor_shm,
    clearance_factor_shm,
    shape_factor_shm,
    compute_damage_features_shm,
)

# Preprocessing functions
from .preprocessing import scale_min_max_shm


__all__ = [
    # Spectral analysis
    "psd_welch_shm",
    "stft_shm",
    "cwt_analysis_shm",
    # Signal filtering
    "residual_signal_shm",
    "difference_signal_shm",
    "bandpass_condition_signal_shm",
    "gear_mesh_filter_shm",
    "envelope_signal_shm",
    # Signal processing
    "ars_tach_shm",
    "fir1_shm",
    # Statistics
    "fm0_shm",
    "fm4_shm",
    "peak_factor_shm",
    "impulse_factor_shm",
    "clearance_factor_shm",
    "shape_factor_shm",
    "compute_damage_features_shm",
    # Preprocessing
    "scale_min_max_shm",
]
