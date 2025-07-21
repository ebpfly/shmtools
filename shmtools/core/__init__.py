"""
Core signal processing functions for SHMTools.

This module contains fundamental signal processing operations including
spectral analysis, filtering, and statistical computations.
"""

from .spectral import *
from .filtering import *
from .statistics import *
from .preprocessing import *
from .signal_processing import ars_tach, fir1
from .signal_filtering import (
    residual_signal, difference_signal, bandpass_condition_signal,
    gear_mesh_filter, envelope_signal
)

__all__ = [
    # Spectral analysis
    "psd_welch_shm",
    "psd_welch",
    "stft", 
    "spectrogram",
    "cepstrum",
    "fast_kurtogram",
    "cwt_analysis",
    # Filtering
    "filter_signal",
    "bandpass_filter",
    "lowpass_filter", 
    "highpass_filter",
    "fir1",
    # Statistics - Basic
    "statistical_moments",
    "rms",
    "crest_factor",
    "peak_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    # Statistics - Damage indicators
    "fm0_shm", 
    "fm4_shm",
    # TODO: Add m6a_shm, m8a_shm, na4m_shm, nb4m_shm when properly implemented
    "compute_damage_features",
    # Preprocessing
    "demean",
    "window_signal",
    "envelope",
    "analytic_signal",
    "scale_min_max_shm",
    # Signal processing
    "ars_tach",
    # Signal filtering
    "residual_signal",
    "difference_signal", 
    "bandpass_condition_signal",
    "gear_mesh_filter",
    "envelope_signal",
]