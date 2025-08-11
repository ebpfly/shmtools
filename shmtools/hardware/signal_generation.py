"""
Signal generation functions for hardware excitation.

This module provides functions to generate excitation signals for
structural health monitoring experiments.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def band_lim_white_noise_shm(array_size: Tuple[int, int], 
                            cutoffs: np.ndarray,
                            rms: float,
                            filter_order: int = 5) -> np.ndarray:
    """
    Generate band-limited white noise for structural excitation.
    
    .. meta::
        :category: Data Acquisition - Signal Generation
        :matlab_equivalent: bandLimWhiteNoise_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Signal
        :display_name: Band-Limited White Noise
        :verbose_call: [Excitation Signal] = Band-Limited White Noise (Array Size, Cutoff Frequencies, RMS Level, Filter Order)
    
    Parameters
    ----------
    array_size : tuple of int
        Shape of output array (n_samples, n_signals).
        
        .. gui::
            :widget: dimension_input
            :default: (1024, 1)
            
    cutoffs : array_like, shape (2,)
        Normalized cutoff frequencies [low, high] (0 to 1).
        For example, [20, 150] Hz at 320 Hz sampling → [0.125, 0.9375].
        
        .. gui::
            :widget: frequency_range
            :min: 0.0
            :max: 1.0
            :default: [0.1, 0.8]
            
    rms : float
        Desired RMS level of output signal.
        
        .. gui::
            :widget: number_input
            :min: 0.0
            :max: 10.0
            :default: 1.0
            :step: 0.1
            
    filter_order : int, optional
        Order of Butterworth bandpass filter. Default is 5.
        
        .. gui::
            :widget: number_input
            :min: 2
            :max: 10
            :default: 5
    
    Returns
    -------
    excitation : ndarray, shape (n_samples, n_signals)
        Band-limited white noise signals with specified RMS level.
    
    Notes
    -----
    The function generates white Gaussian noise and applies a Butterworth
    bandpass filter to limit the frequency content. The signal is then
    scaled to achieve the desired RMS level.
    
    Examples
    --------
    >>> # Generate 1 second of band-limited noise at 1000 Hz
    >>> fs = 1000  # Sampling frequency
    >>> duration = 1.0  # seconds
    >>> n_samples = int(fs * duration)
    >>> 
    >>> # 20-100 Hz bandpass
    >>> cutoffs = np.array([20, 100]) / (fs / 2)  # Normalize to Nyquist
    >>> excitation = band_lim_white_noise_shm((n_samples, 1), cutoffs, rms=2.0)
    >>> 
    >>> print(f"Signal RMS: {np.sqrt(np.mean(excitation**2)):.3f}")
    Signal RMS: 2.000
    """
    n_samples, n_signals = array_size
    
    # Validate inputs
    if len(cutoffs) != 2:
        raise ValueError("Cutoffs must be a 2-element array [low, high]")
    
    if cutoffs[0] >= cutoffs[1]:
        raise ValueError("Low cutoff must be less than high cutoff")
    
    if np.any(cutoffs <= 0) or np.any(cutoffs >= 1):
        raise ValueError("Cutoff frequencies must be between 0 and 1")
    
    # Generate white noise
    white_noise = np.random.randn(n_samples, n_signals)
    
    # Design Butterworth bandpass filter
    sos = signal.butter(filter_order, cutoffs, btype='band', output='sos')
    
    # Apply filter to each signal
    filtered_noise = np.zeros_like(white_noise)
    for i in range(n_signals):
        # Apply forward-backward filtering for zero phase shift
        filtered_noise[:, i] = signal.sosfiltfilt(sos, white_noise[:, i])
    
    # Calculate current RMS
    current_rms = np.sqrt(np.mean(filtered_noise**2, axis=0))
    
    # Scale to desired RMS
    scaling_factors = rms / (current_rms + 1e-10)  # Avoid division by zero
    excitation = filtered_noise * scaling_factors
    
    return excitation