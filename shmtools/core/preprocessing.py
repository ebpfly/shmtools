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


def scale_min_max_shm(
    X: np.ndarray, scaling_dimension: int = 1, scale_range: Optional[tuple] = None
) -> np.ndarray:
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


def analytic_signal_shm(X: np.ndarray) -> np.ndarray:
    """
    Convert signals to their analytic form using Hilbert transform via FFT.
    
    .. meta::
        :category: Core - Preprocessing
        :matlab_equivalent: analyticSignal_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Time Series
        :display_name: Analytic Signal
        :verbose_call: [Analytic Signal Matrix] = Analytic Signal (Signal Matrix)
        
    Returns a matrix of analytic signals converted by Hilbert transform
    using the FFT. The analytic signal consists of a real part of a signal
    as well as its imaginary parts. To compute analytic signal, the FFT
    of the real signal is calculated and its positive frequency components
    are doubled; its negative frequency components are set to zero. By
    taking the inverse Fourier transform of the manipulated frequency
    domain, the analytic signal is formed.
    
    Parameters
    ----------
    X : ndarray, shape (samples, channels, instances)
        Input signal matrix.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Signal matrix for analytic signal conversion
    
    Returns
    -------
    analytic_matrix : ndarray, shape (samples, channels, instances), dtype=complex
        Analytic signal matrix with real and imaginary components.
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.preprocessing import analytic_signal_shm
    >>> 
    >>> # Generate synthetic signal
    >>> t = np.linspace(0, 1, 100)
    >>> x = np.sin(2*np.pi*5*t)  # 5 Hz sine wave
    >>> X = x.reshape(-1, 1, 1)
    >>> 
    >>> # Compute analytic signal
    >>> analytic = analytic_signal_shm(X)
    >>> # Real part should be original signal, imaginary part is Hilbert transform
    >>> print(f"Original signal RMS: {np.sqrt(np.mean(x**2)):.3f}")
    >>> print(f"Analytic magnitude RMS: {np.sqrt(np.mean(np.abs(analytic)**2)):.3f}")
    
    References
    ----------
    [1] Oppenheim, A.V., and R.W. Schafer, Discrete-Time Signal Processing,
    Third Edition. Prentice-Hall, 1989, Ch. 12 p.949-955
    
    [2] Randall, Robert., Vibration-based Condition Monitoring, Wiley and
    Sons, 2011. Ch. 3.3.2 p.97.
    """
    # Get matrix dimension sizes
    n_signal, n_channel, n_instance = X.shape
    
    # Allocate memory for analytic signal matrix
    analytic_matrix = np.zeros((n_signal, n_channel, n_instance), dtype=complex)
    
    # Compute analytic signal for each channel and instance
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            signal = X[:, i_channel, i_instance]
            analytic_matrix[:, i_channel, i_instance] = _analytic_single(signal)
    
    return analytic_matrix


def _analytic_single(x_n: np.ndarray) -> np.ndarray:
    """
    Convert a single signal to analytic signal using Hilbert transform via FFT.
    
    This is the core algorithm from the MATLAB analyticSignal_shm function.
    
    Parameters
    ----------
    x_n : ndarray
        Input real signal.
    
    Returns
    -------
    x_h : ndarray, dtype=complex
        Complex analytic signal.
    """
    # Take only the real part
    x_r = np.real(x_n)
    n_xr = len(x_r)
    
    # Compute FFT
    x_e = np.fft.fft(x_r, n_xr)
    
    # Create the multiplier vector u_n
    u_n = np.zeros(n_xr)
    
    if n_xr % 2 == 0:  # When n_xr is even
        u_n[0] = 1  # DC component
        u_n[n_xr // 2] = 1  # Nyquist frequency (if present)
        u_n[1:n_xr // 2] = 2  # Positive frequencies
        # Negative frequencies remain 0
    else:  # When n_xr is odd
        u_n[0] = 1  # DC component
        u_n[1:(n_xr + 1) // 2] = 2  # Positive frequencies
        # Negative frequencies remain 0
    
    # Apply the multiplier and take inverse FFT
    x_h = np.fft.ifft(x_e * u_n)
    
    return x_h


def envelope_shm(X: np.ndarray) -> np.ndarray:
    """
    Calculate envelope signals from signal matrix.
    
    .. meta::
        :category: Core - Preprocessing
        :matlab_equivalent: envelope_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Time Series
        :display_name: Envelope Signal
        :verbose_call: [Enveloped Signal Matrix] = Envelope Signal (Signal Matrix)
        
    Returns a matrix of enveloped signals. The envelope signal is the
    magnitude of the analytic signal. It is used in feature NB4M which
    takes the 4th order statistical moment of the enveloped signal and is
    normalized by an averaged variance of a baseline enveloped signal
    squared. Envelope analysis is also a useful tool in bearing analysis as
    not much can be gained from the frequency domain alone for bearing
    failures. Oftentimes changes in the frequency domain due to bearing
    damage are buried by deterministic gear mesh components. As such,
    frequency bands of vibration signals are bandpass filtered and the
    envelope of the band pass signal is then used for bearing diagnostics.
    Frequency bands of interest can be determined using the fast kurtogram
    method which looks at the spectral kurtosis of different frequency
    bands.
    
    Parameters
    ----------
    X : ndarray, shape (samples, channels, instances)
        Input signal matrix.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Signal matrix for envelope calculation
    
    Returns
    -------
    envelope_matrix : ndarray, shape (samples, channels, instances)
        Enveloped signal matrix (magnitude of analytic signal).
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.preprocessing import envelope_shm
    >>> 
    >>> # Generate AM modulated signal (typical bearing fault signature)
    >>> t = np.linspace(0, 1, 1000)
    >>> carrier = np.sin(2*np.pi*100*t)  # 100 Hz carrier
    >>> modulator = 1 + 0.5*np.sin(2*np.pi*5*t)  # 5 Hz modulation
    >>> x = modulator * carrier
    >>> X = x.reshape(-1, 1, 1)
    >>> 
    >>> # Compute envelope
    >>> env = envelope_shm(X)
    >>> # Envelope should recover the modulation signal
    >>> print(f"Original modulator RMS: {np.sqrt(np.mean((modulator-1)**2)):.3f}")
    >>> print(f"Envelope variation RMS: {np.sqrt(np.mean((env[:,0,0]-np.mean(env[:,0,0]))**2)):.3f}")
    
    References
    ----------
    [1] Randall, Robert., Vibration-based Condition Monitoring, Wiley and
    Sons, 2011. Ch. 3.3.2 p.97. Ch 5.5 p.200
    
    See Also
    --------
    analytic_signal_shm : Compute analytic signal using Hilbert transform
    """
    # Get matrix dimension sizes
    n_signal, n_channel, n_instance = X.shape
    
    # Allocate memory for envelope matrix
    envelope_matrix = np.zeros((n_signal, n_channel, n_instance))
    
    # Compute envelope for each channel and instance
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            signal = X[:, i_channel, i_instance]
            # Envelope is the magnitude of the analytic signal
            analytic = _analytic_single(signal)
            envelope_matrix[:, i_channel, i_instance] = np.abs(analytic)
    
    return envelope_matrix
