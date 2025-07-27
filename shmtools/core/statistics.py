"""
Statistical analysis functions for signal processing.

This module provides statistical measures and damage indicators
commonly used in structural health monitoring.
"""

import numpy as np
from typing import Union, Dict, Any, Optional


def statistical_moments(x: np.ndarray, axis: int = 0) -> Dict[str, float]:
    """
    Compute statistical moments of a signal.

    Python equivalent of MATLAB's statMoments_shm function.

    Parameters
    ----------
    x : np.ndarray
        Input signal array.
    axis : int, optional
        Axis along which to compute moments. Default is 0.

    Returns
    -------
    moments : dict
        Dictionary containing statistical moments:
        - mean: First moment (mean)
        - var: Second central moment (variance)
        - skew: Third standardized moment (skewness)
        - kurt: Fourth standardized moment (kurtosis)
    """
    mean = np.mean(x, axis=axis)
    var = np.var(x, axis=axis, ddof=1)

    # Center the data
    x_centered = x - np.expand_dims(mean, axis)

    # Compute higher order moments
    std = np.sqrt(var)
    skew = np.mean((x_centered / np.expand_dims(std, axis)) ** 3, axis=axis)
    kurt = np.mean((x_centered / np.expand_dims(std, axis)) ** 4, axis=axis)

    return {"mean": mean, "var": var, "skew": skew, "kurt": kurt}


def rms(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute root mean square (RMS) value.

    Python equivalent of MATLAB's rms_shm function.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    axis : int, optional
        Axis along which to compute RMS. Default is 0.

    Returns
    -------
    rms_val : np.ndarray
        RMS value of input signal.
    """
    return np.sqrt(np.mean(x**2, axis=axis))


def crest_factor(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute crest factor (peak-to-RMS ratio).

    Python equivalent of MATLAB's crestFactor_shm function.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    axis : int, optional
        Axis along which to compute crest factor. Default is 0.

    Returns
    -------
    cf : np.ndarray
        Crest factor of input signal.
    """
    peak = np.max(np.abs(x), axis=axis)
    rms_val = rms(x, axis=axis)
    return peak / rms_val


def fm0_shm(
    X: np.ndarray,
    fund_mesh_freq: float,
    track_orders: Optional[np.ndarray] = None,
    nfft: int = 512,
    n_bin_search: int = 2,
) -> np.ndarray:
    """
    Feature Extraction: Compute FM0 gear damage indicator.

    The FM0 feature is the magnitude of a signal's peak-to-peak amplitude
    divided by the sum of the frequency amplitudes corresponding to its
    fundamental gearmesh frequencies.

    Parameters
    ----------
    X : ndarray, shape (samples, channels, instances)
        Input signal matrix.
    fund_mesh_freq : float
        Fundamental gear mesh frequency (normalized).
    track_orders : ndarray, optional
        Orders of fundamental mesh frequency to track. Default is [1].
    nfft : int, optional
        Number of bins in FFT power spectral density. Default is 512.
    n_bin_search : int, optional
        FFT bin search parameter. Default is 2.

    Returns
    -------
    fm0 : ndarray, shape (instances, channels)
        FM0 feature vectors in concatenated format.

    Notes
    -----
    FM0 is used for gear damage detection:
    - Not good for detecting minor tooth damage
    - Adequate for larger tooth damage
    - More responsive to minor tooth damage when used with Hoelder exponent

    See Also
    --------
    psd_welch : Power spectral density estimation
    """
    from .spectral import psd_welch

    # Handle defaults
    if track_orders is None:
        track_orders = np.array([1])

    # Get dimensions
    n_signal, n_channel, n_instance = X.shape
    fm0 = np.zeros((n_instance, n_channel))

    # Process each channel and instance
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            # Calculate peak-to-peak amplitude
            signal = X[:, i_channel, i_instance]
            peak_to_peak_amp = abs(np.max(signal) - np.min(signal))

            # Compute power spectral density using defaults
            f_psd, psd = psd_welch(signal, fs=1.0, nfft=nfft)

            # Sum amplitudes at tracked orders
            sum_amp = _sum_amplitudes(
                psd, f_psd, fund_mesh_freq, track_orders, n_bin_search
            )

            # Calculate FM0 feature
            fm0[i_instance, i_channel] = peak_to_peak_amp / sum_amp

    return fm0


def _sum_amplitudes(
    psd: np.ndarray,
    f_psd: np.ndarray,
    fund_mesh_freq: float,
    track_orders: np.ndarray,
    n_bin_search: int,
) -> float:
    """
    Sum amplitudes at tracked gear mesh order frequencies.

    Parameters
    ----------
    psd : ndarray
        Power spectral density.
    f_psd : ndarray
        Frequency vector.
    fund_mesh_freq : float
        Fundamental mesh frequency.
    track_orders : ndarray
        Orders to track.
    n_bin_search : int
        Search window half-width in bins.

    Returns
    -------
    sum_amp : float
        Sum of amplitudes at tracked orders.
    """
    sum_amp = 0.0

    for order in track_orders:
        target_freq = order * fund_mesh_freq

        # Find closest frequency bin
        I = np.where((f_psd - target_freq) >= 0)[0]
        if len(I) > 0:
            I = I[0]

            # Search window bounds
            start_idx = max(0, I - n_bin_search)
            end_idx = min(len(psd), I + n_bin_search + 1)

            # Add maximum amplitude in search window
            sum_amp += np.max(psd[start_idx:end_idx])

    return sum_amp


def fm4_shm(D: np.ndarray) -> np.ndarray:
    """
    Feature Extraction: Compute FM4 damage indicator from difference signal.

    FM4 computes the fourth statistical moment about the mean of a difference
    signal normalized by its variance squared. Specifically developed to detect
    surface damage on a limited number of gear teeth.

    Parameters
    ----------
    D : ndarray, shape (samples, channels, instances)
        Difference signal matrix.

    Returns
    -------
    fm4 : ndarray, shape (instances, channels)
        FM4 feature vectors in concatenated format.

    Notes
    -----
    FM4 = n * Σ(x - μ)⁴ / [Σ(x - μ)²]²

    Where:
    - n = number of samples
    - x = difference signal values
    - μ = mean of the difference signal

    FM4 is particularly sensitive to outliers and spikes in the signal
    due to the 4th power, making it effective for damage detection.
    """
    # Get dimensions
    n_signal, n_channel, n_instance = D.shape
    fm4 = np.zeros((n_instance, n_channel))

    # Process each channel and instance
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            # Extract difference signal
            signal = D[:, i_channel, i_instance]

            # Calculate mean
            mean_val = np.mean(signal)

            # Center the signal
            centered = signal - mean_val

            # Calculate fourth and second moments
            fourth_moment = np.sum(centered**4)
            second_moment_squared = (np.sum(centered**2)) ** 2

            # Calculate FM4 with normalization factor
            fm4[i_instance, i_channel] = (
                n_signal * fourth_moment / second_moment_squared
            )

    return fm4


# TODO: Implement M6A, M8A, NA4M, NB4M functions
# These need to be implemented based on exact MATLAB algorithms
# Current placeholder implementations have been removed to prevent incorrect usage


def peak_factor_shm(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute peak factor (maximum value normalized by RMS).

    Python equivalent of MATLAB's peakFactor_shm function. Similar to
    crest factor but uses maximum value instead of absolute maximum.

    .. meta::
        :category: Statistics - Basic Indicators
        :matlab_equivalent: peakFactor_shm
        :complexity: Basic
        :sensitivity: Peak, Outlier

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for peak factor analysis"

    axis : int, optional, default=0
        Axis along which to compute peak factor.

    Returns
    -------
    pf : np.ndarray
        Peak factor of input signal.

        .. gui::
            :plot_type: "scalar"
            :description: "Peak factor"
    """
    peak = np.max(x, axis=axis)
    rms_val = rms(x, axis=axis)
    return peak / (rms_val + 1e-12)


def impulse_factor_shm(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute impulse factor (peak divided by mean absolute value).

    Python equivalent of MATLAB's impulseFactor_shm function. Sensitive
    to impulsive content and outliers.

    .. meta::
        :category: Statistics - Basic Indicators
        :matlab_equivalent: impulseFactor_shm
        :complexity: Basic
        :sensitivity: Impulse, Outlier

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for impulse factor analysis"

    axis : int, optional, default=0
        Axis along which to compute impulse factor.

    Returns
    -------
    if_val : np.ndarray
        Impulse factor of input signal.

        .. gui::
            :plot_type: "scalar"
            :description: "Impulse factor"
    """
    peak = np.max(np.abs(x), axis=axis)
    mean_abs = np.mean(np.abs(x), axis=axis)
    return peak / (mean_abs + 1e-12)


def clearance_factor_shm(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute clearance factor (peak divided by square of mean square root).

    Python equivalent of MATLAB's clearanceFactor_shm function. Highly
    sensitive to impulses and outliers.

    .. meta::
        :category: Statistics - Basic Indicators
        :matlab_equivalent: clearanceFactor_shm
        :complexity: Basic
        :sensitivity: Impulse, Outlier

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for clearance factor analysis"

    axis : int, optional, default=0
        Axis along which to compute clearance factor.

    Returns
    -------
    cf : np.ndarray
        Clearance factor of input signal.

        .. gui::
            :plot_type: "scalar"
            :description: "Clearance factor"
    """
    peak = np.max(np.abs(x), axis=axis)
    mean_sqrt = np.mean(np.sqrt(np.abs(x)), axis=axis)
    return peak / (mean_sqrt**2 + 1e-12)


def shape_factor_shm(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute shape factor (RMS divided by mean absolute value).

    Python equivalent of MATLAB's shapeFactor_shm function. Measures
    the "shape" of the amplitude distribution.

    .. meta::
        :category: Statistics - Basic Indicators
        :matlab_equivalent: shapeFactor_shm
        :complexity: Basic
        :sensitivity: Distribution Shape

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for shape factor analysis"

    axis : int, optional, default=0
        Axis along which to compute shape factor.

    Returns
    -------
    sf : np.ndarray
        Shape factor of input signal.

        .. gui::
            :plot_type: "scalar"
            :description: "Shape factor"
    """
    rms_val = rms(x, axis=axis)
    mean_abs = np.mean(np.abs(x), axis=axis)
    return rms_val / (mean_abs + 1e-12)


def compute_damage_features_shm(
    x: np.ndarray, fs: float = None, axis: int = 0
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute comprehensive set of damage indicators.

    Python equivalent of MATLAB's damageFeaturesAll_shm function. Computes
    all available statistical damage indicators in one function call.

    .. meta::
        :category: Statistics - Feature Extraction
        :matlab_equivalent: damageFeaturesAll_shm
        :complexity: Intermediate
        :output_type: Feature Vector

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for feature extraction"

    fs : float, optional
        Sampling frequency for frequency-dependent features.

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 100000.0
            :description: "Sampling frequency (Hz)"

    axis : int, optional, default=0
        Axis along which to compute features.

    Returns
    -------
    features : dict
        Dictionary containing all damage indicators:
        - Basic: RMS, peak, crest, shape, impulse, clearance factors
        - Statistical: FM0, FM4, M6A, M8A
        - Advanced: NA4M (if scipy available)

        .. gui::
            :plot_type: "bar"
            :description: "Comprehensive damage features"
    """
    features = {
        "rms": rms(x, axis=axis),
        "peak_factor": peak_factor(x, axis=axis),
        "crest_factor": crest_factor(x, axis=axis),
        "shape_factor": shape_factor(x, axis=axis),
        "impulse_factor": impulse_factor(x, axis=axis),
        "clearance_factor": clearance_factor(x, axis=axis),
        "fm0": fm0(x, axis=axis),
        "fm4": fm4(x, axis=axis),
        "m6a": m6a(x, axis=axis),
        "m8a": m8a(x, axis=axis),
    }

    # Add frequency-dependent features if fs is provided
    if fs is not None:
        # Add NA4M (envelope-based)
        try:
            features["na4m"] = na4m(x, axis=axis)
        except ImportError:
            pass  # scipy not available

    return features
