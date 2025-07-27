"""
Spectral analysis functions for signal processing.

This module provides functions for frequency domain analysis including
power spectral density estimation, spectrograms, and time-frequency analysis.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union


def psd_welch_shm(
    X: np.ndarray,
    n_win: Optional[int] = None,
    n_ovlap: Optional[int] = None, 
    n_fft: Optional[int] = None,
    fs: Optional[float] = None,
    use_one_sided: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Feature Extraction: Estimate power spectral density via Welch's method.
    
    Estimate the power spectral density (PSD) of X using Welch's overlapped 
    segment averaging estimator. X is divided into overlapping segments. Each 
    segment is windowed by the Hamming window and its periodogram is computed. 
    The periodograms are then averaged to obtain the PSD estimate and reduce 
    the noise inherent in the periodogram.
    
    Parameters
    ----------
    X : ndarray, shape (time, channels, instances)
        Matrix of time series data.
    n_win : int, optional
        Window length in samples. Default varies based on signal length.
    n_ovlap : int, optional  
        Number of overlapping samples between windows. Default is 50% of window.
    n_fft : int, optional
        Number of FFT frequency bins. Default is max(256, 2^nextpow2(n_win)).
    fs : float, optional
        Sampling frequency in Hz. Default is 1.
    use_one_sided : bool, optional
        Create one-sided PSD instead of two-sided. Default is True.
        
    Returns
    -------
    psd_matrix : ndarray, shape (n_fft, channels, instances)
        Power spectral density matrix.
    f : ndarray, shape (n_fft,)
        Frequency vector corresponding to psd_matrix.
    use_one_sided : bool
        Whether one-sided PSD was computed.
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core import psd_welch_shm
    >>> 
    >>> # Generate test signal
    >>> fs = 1000  
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(fs)
    >>> X = signal.reshape(-1, 1, 1)  # (time, channels, instances)
    >>> 
    >>> # Compute PSD
    >>> psd, f, one_sided = psd_welch_shm(X, fs=fs)
    >>> peak_freq = f[np.argmax(psd[:, 0, 0])]
    >>> print(f"Peak frequency: {peak_freq:.1f} Hz")
    """
    # Handle input validation and defaults
    if X.ndim == 1:
        X = X.reshape(-1, 1, 1)
    elif X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
    n_time, n_channels, n_instances = X.shape
    
    # Set defaults following MATLAB behavior
    if fs is None:
        fs = 1.0
    if use_one_sided is None:
        use_one_sided = True
    if n_win is None:
        n_win = min(256, n_time)
    if n_ovlap is None:
        n_ovlap = n_win // 2
    if n_fft is None:
        n_fft = max(256, int(2 ** np.ceil(np.log2(n_win))))
        
    # Initialize output
    if use_one_sided and n_fft % 2 == 0:
        n_freq_out = n_fft // 2 + 1
    elif use_one_sided:
        n_freq_out = (n_fft + 1) // 2
    else:
        n_freq_out = n_fft
        
    psd_matrix = np.zeros((n_freq_out, n_channels, n_instances))
    
    # Compute PSD for each channel and instance using scipy.signal.welch
    for i_instance in range(n_instances):
        for i_channel in range(n_channels):
            x_current = X[:, i_channel, i_instance]
            
            # Use scipy.signal.welch with parameters matching MATLAB behavior
            f, psd_scipy = signal.welch(
                x_current,
                fs=fs,
                window='hamming',  # MATLAB default
                nperseg=n_win,
                noverlap=n_ovlap,
                nfft=n_fft,
                return_onesided=use_one_sided,
                scaling='density'
            )
            
            psd_matrix[:, i_channel, i_instance] = psd_scipy
    
    return psd_matrix, f, use_one_sided


def stft_shm(
    x: np.ndarray,
    fs: float = 1.0,
    window: str = "hann", 
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT).
    
    .. meta::
        :category: Core - Spectral Analysis
        :matlab_equivalent: stft_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Time-Frequency
        :display_name: Short-Time Fourier Transform
        :verbose_call: [Frequency, Time, STFT] = Short-Time Fourier Transform (Signal, Sampling Rate, Window, Segment Length, Overlap, FFT Length)
    
    Parameters
    ----------
    x : array_like
        Input signal array.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
        
        .. gui::
            :widget: numeric_input
            :min: 0.1
            :max: 1000000.0
            :step: 0.1
            :units: "Hz"
            
    window : str, optional
        Window function type. Default is "hann".
        
        .. gui::
            :widget: select
            :options: ["hann", "hamming", "blackman", "bartlett"]
            
    nperseg : int, optional
        Length of each segment. Default is 256.
        
        .. gui::
            :widget: numeric_input
            :min: 8
            :max: 8192
            :step: 1
            
    noverlap : int, optional
        Number of points to overlap between segments.
        
        .. gui::
            :widget: numeric_input
            :min: 0
            :max: 4096
            :step: 1
            :allow_none: True
            
    nfft : int, optional
        Length of FFT used.
        
        .. gui::
            :widget: numeric_input
            :min: 8
            :max: 8192
            :step: 1
            :allow_none: True
    
    Returns
    -------
    f : ndarray
        Frequency array.
    t : ndarray
        Time array.
    Zxx : ndarray
        STFT of x.
    """
    f, t, Zxx = signal.stft(x, fs=fs, window=window, nperseg=nperseg, 
                           noverlap=noverlap, nfft=nfft)
    return f, t, Zxx


def cwt_analysis_shm(
    x: np.ndarray,
    scales: np.ndarray,
    wavelet: str = "morlet",
    fs: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous Wavelet Transform analysis.
    
    .. meta::
        :category: Core - Spectral Analysis
        :matlab_equivalent: cwt_shm
        :complexity: Advanced
        :data_type: Time Series
        :output_type: Time-Frequency
        :display_name: Continuous Wavelet Transform
        :verbose_call: [CWT Coefficients, Frequencies] = Continuous Wavelet Transform (Signal, Scales, Wavelet, Sampling Rate)
    
    Parameters
    ----------
    x : array_like
        Input signal array.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    scales : array_like
        Scales for wavelet transform.
        
        .. gui::
            :widget: array_input
            :description: "Wavelet scales"
            
    wavelet : str, optional
        Wavelet type. Default is "morlet".
        
        .. gui::
            :widget: select
            :options: ["morlet", "mexh", "ricker"]
            
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
        
        .. gui::
            :widget: numeric_input
            :min: 0.1
            :max: 1000000.0
            :step: 0.1
            :units: "Hz"
    
    Returns
    -------
    coeffs : ndarray
        CWT coefficients.
    freqs : ndarray
        Frequencies corresponding to scales.
    """
    # This is a simplified implementation
    # In practice, you would use pywavelets or similar
    from scipy import signal as sig
    
    # Convert scales to frequencies (approximate for Morlet wavelet)
    if wavelet == "morlet":
        freqs = fs / (2 * scales)
    else:
        freqs = fs / scales
        
    # Placeholder implementation using continuous wavelet transform
    coeffs = np.zeros((len(scales), len(x)), dtype=complex)
    
    for i, scale in enumerate(scales):
        # Simple implementation - in practice use pywt.cwt
        if wavelet == "morlet":
            # Morlet wavelet approximation
            sigma = scale / (2 * np.pi)
            wavelet_func = lambda t: np.exp(1j * 2 * np.pi * t / scale) * np.exp(-t**2 / (2 * sigma**2))
        else:
            # Default to Ricker (Mexican hat)
            wavelet_func = sig.ricker
            
        # Convolution-based CWT (simplified)
        t_wav = np.arange(-3*scale, 3*scale+1)
        if len(t_wav) > len(x):
            t_wav = np.arange(-len(x)//2, len(x)//2+1)
            
        if wavelet == "morlet":
            wav = wavelet_func(t_wav)
        else:
            wav = wavelet_func(len(t_wav), scale)
            
        coeffs[i, :] = np.convolve(x, wav, mode='same')
    
    return coeffs, freqs