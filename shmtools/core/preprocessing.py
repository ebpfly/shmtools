"""
Signal preprocessing functions for SHM applications.

This module provides basic signal conditioning operations including
detrending, windowing, and envelope detection.
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
from typing import Union, Optional


def demean(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Remove mean from signal.
    
    Python equivalent of MATLAB's demean_shm function.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    axis : int, optional
        Axis along which to remove mean. Default is 0.
        
    Returns
    -------
    y : np.ndarray
        Zero-mean signal.
    """
    return x - np.mean(x, axis=axis, keepdims=True)


def window_signal(x: np.ndarray, window: str = "hann") -> np.ndarray:
    """
    Apply window function to signal.
    
    Python equivalent of MATLAB's window_shm function.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    window : str, optional
        Window type ('hann', 'hamming', 'blackman', etc.). Default is 'hann'.
        
    Returns
    -------
    y : np.ndarray
        Windowed signal.
    """
    N = len(x)
    
    if window == "hann":
        w = signal.windows.hann(N)
    elif window == "hamming":
        w = signal.windows.hamming(N)
    elif window == "blackman":
        w = signal.windows.blackman(N)
    elif window == "bartlett":
        w = signal.windows.bartlett(N)
    elif window == "kaiser":
        w = signal.windows.kaiser(N, 5.0)
    else:
        raise ValueError(f"Unknown window type: {window}")
    
    if x.ndim == 1:
        return x * w
    else:
        # Apply window to each column
        return x * w[:, np.newaxis]


def envelope(x: np.ndarray, method: str = "hilbert") -> np.ndarray:
    """
    Compute envelope of signal.
    
    Python equivalent of MATLAB's envelope_shm function.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    method : str, optional
        Method for envelope computation ('hilbert', 'peaks'). Default is 'hilbert'.
        
    Returns
    -------
    env : np.ndarray
        Signal envelope.
    """
    if method == "hilbert":
        analytic = signal.hilbert(x, axis=0)
        return np.abs(analytic)
    elif method == "peaks":
        # TODO: Implement peak-based envelope detection
        raise NotImplementedError("Peak-based envelope not yet implemented")
    else:
        raise ValueError(f"Unknown envelope method: {method}")


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """
    Compute analytic signal using Hilbert transform.
    
    Python equivalent of MATLAB's analyticSignal_shm function.
    
    Parameters
    ----------
    x : np.ndarray
        Input real signal.
        
    Returns
    -------
    z : np.ndarray
        Complex analytic signal.
    """
    return signal.hilbert(x, axis=0)


def scale_min_max(x: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:
    """
    Scale signal to specified range.
    
    Python equivalent of MATLAB's scaleMinMax_shm function.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    feature_range : tuple, optional
        Target range (min, max). Default is (0, 1).
        
    Returns
    -------
    x_scaled : np.ndarray
        Scaled signal.
    """
    x_min = np.min(x)
    x_max = np.max(x) 
    
    # Avoid division by zero
    if x_max == x_min:
        return np.full_like(x, feature_range[0])
    
    # Scale to [0, 1] then to target range
    x_std = (x - x_min) / (x_max - x_min)
    x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return x_scaled


def scale_min_max_shm(X: np.ndarray, scaling_dimension: int = 1, scale_range: Optional[tuple] = None) -> np.ndarray:
    """
    Scale data to a minimum and maximum value.
    
    This function scales the data between the values in the scaling range
    along the specified dimension. MATLAB-compatible version of scaleMinMax_shm.
    
    .. meta::
        :category: Core - Preprocessing
        :matlab_equivalent: scaleMinMax_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Time Series
        
    Parameters
    ----------
    X : array_like
        Data matrix to scale.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    scaling_dimension : int, optional
        Dimension of X to apply scaling along (default: 1)
        
        .. gui::
            :widget: spinner
            :min: 1
            :max: 3
            :default: 1
            
    scale_range : tuple, optional
        Minimum and maximum value for scaled output. If None, data is 
        normalized to z-score. Format: (min_value, max_value)
        
        .. gui::
            :widget: range_input
            :default: [0, 1]
            
    Returns
    -------
    scaled_X : ndarray
        Scaled data matrix.
        
    References
    ----------
    MATLAB scaleMinMax_shm function from SHMTools.
    """
    X = np.asarray(X, dtype=np.float64)
    
    # Handle default scaling dimension (MATLAB uses 1-based indexing)
    if scaling_dimension < 1:
        scaling_dimension = 1
    
    # Convert to 0-based indexing for Python
    scaling_axis = scaling_dimension - 1
    
    data_shape = X.shape
    
    if len(data_shape) < scaling_dimension:
        raise ValueError("Specified scaling dimension does not exist in data.")
    
    # If no scale_range specified, use z-score normalization
    if scale_range is None:
        scaled_X = zscore(X, axis=scaling_axis, ddof=0)  # Use ddof=0 to match MATLAB
        return scaled_X
    
    # Min-max scaling
    x_min = np.min(X, axis=scaling_axis, keepdims=True)
    x_max = np.max(X, axis=scaling_axis, keepdims=True)
    x_range = x_max - x_min
    
    # Handle division by zero (when min == max)
    x_range = np.where(x_range == 0, 1.0, x_range)
    
    scale_min = scale_range[0]
    scale_max = scale_range[1]
    scale_range_val = scale_max - scale_min
    
    # Apply scaling: ((X - min) / range) * scale_range + scale_min
    scaled_X = (X - x_min) / x_range
    scaled_X = scale_range_val * scaled_X + scale_min
    
    return scaled_X