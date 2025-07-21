"""
Digital filtering functions for signal processing.

This module provides various filtering operations including bandpass,
lowpass, highpass filters and other signal conditioning functions.
"""

import numpy as np
from scipy import signal
from typing import Union, Optional, Tuple


def filter_signal(
    x: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    axis: int = -1,
    zi: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply digital filter to signal.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    b : np.ndarray
        Numerator coefficients of filter.
    a : np.ndarray
        Denominator coefficients of filter.
    axis : int, optional
        Axis along which to apply filter. Default is -1.
    zi : np.ndarray, optional
        Initial conditions for filter delays.
        
    Returns
    -------
    y : np.ndarray
        Filtered signal.
    zf : np.ndarray, optional
        Final conditions for filter delays (if zi provided).
    """
    if zi is not None:
        y, zf = signal.lfilter(b, a, x, axis=axis, zi=zi)
        return y, zf
    else:
        y = signal.lfilter(b, a, x, axis=axis)
        return y


def bandpass_filter(
    x: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
    ftype: str = "butter",
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order. Default is 4.
    ftype : str, optional
        Filter type ('butter', 'cheby1', 'cheby2', 'ellip'). Default is 'butter'.
    zero_phase : bool, optional
        If True, use filtfilt for zero-phase filtering. Default is True.
        
    Returns
    -------
    y : np.ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if ftype == "butter":
        b, a = signal.butter(order, [low, high], btype="band")
    elif ftype == "cheby1":
        b, a = signal.cheby1(order, 1, [low, high], btype="band")
    elif ftype == "cheby2":
        b, a = signal.cheby2(order, 20, [low, high], btype="band")
    elif ftype == "ellip":
        b, a = signal.ellip(order, 1, 20, [low, high], btype="band")
    else:
        raise ValueError(f"Unknown filter type: {ftype}")
    
    if zero_phase:
        y = signal.filtfilt(b, a, x, axis=0)
    else:
        y = signal.lfilter(b, a, x, axis=0)
    
    return y


def lowpass_filter(
    x: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
    ftype: str = "butter",
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Apply lowpass filter to signal.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order. Default is 4.
    ftype : str, optional
        Filter type. Default is 'butter'.
    zero_phase : bool, optional
        If True, use filtfilt for zero-phase filtering. Default is True.
        
    Returns
    -------
    y : np.ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    if ftype == "butter":
        b, a = signal.butter(order, normal_cutoff, btype="low")
    else:
        raise ValueError(f"Filter type {ftype} not implemented")
    
    if zero_phase:
        y = signal.filtfilt(b, a, x, axis=0)
    else:
        y = signal.lfilter(b, a, x, axis=0)
    
    return y


def highpass_filter(
    x: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
    ftype: str = "butter",
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Apply highpass filter to signal.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order. Default is 4.
    ftype : str, optional
        Filter type. Default is 'butter'.
    zero_phase : bool, optional
        If True, use filtfilt for zero-phase filtering. Default is True.
        
    Returns
    -------
    y : np.ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    if ftype == "butter":
        b, a = signal.butter(order, normal_cutoff, btype="high")
    else:
        raise ValueError(f"Filter type {ftype} not implemented")
    
    if zero_phase:
        y = signal.filtfilt(b, a, x, axis=0)
    else:
        y = signal.lfilter(b, a, x, axis=0)
    
    return y