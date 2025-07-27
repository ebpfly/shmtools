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
    use_one_sided: Optional[bool] = None,
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
    is_one_sided : bool
        Specifies type of PSD (one-sided or two-sided).

    See Also
    --------
    frf_shm : Frequency response function

    Notes
    -----
    Default values:
    - n_win: Computed based on signal length for approximately 8 windows
    - n_ovlap: 50% window overlap
    - n_fft: Next power of 2 >= window length, minimum 256
    - fs: 1 Hz (normalized frequency)
    - use_one_sided: True

    Uses Hamming window for segment windowing.
    """
    # Handle input dimensions
    if X.ndim == 1:
        X = X[:, np.newaxis, np.newaxis]
    elif X.ndim == 2:
        X = X[:, :, np.newaxis]

    n_time, n_channels, n_instances = X.shape

    # Set defaults following MATLAB logic
    if fs is None:
        fs = 1.0

    if use_one_sided is None:
        use_one_sided = True

    # Default window length - approximately 8 windows
    if n_win is None:
        if n_fft is None and n_ovlap is None:
            n_win = int(np.floor(n_time / 4.5))
        elif n_ovlap is not None:
            n_win = int(np.floor((n_time + 7 * n_ovlap) / 8))
        elif n_fft is not None:
            n_win = n_fft
        else:
            n_win = int(np.floor(n_time / 4.5))

    # Default overlap - 50%
    if n_ovlap is None:
        n_ovlap = int(np.floor(n_win / 2))

    # Default FFT length
    if n_fft is None:
        n_fft = max(256, int(2 ** np.ceil(np.log2(n_win))))

    # Initialize output using a test call to determine actual output size
    # Use scipy.signal.welch to determine the actual frequency vector length
    test_sig = np.ones(n_win)  # Dummy signal for size determination
    f_test, psd_test = signal.welch(
        test_sig,
        fs=fs,
        window="hamming",
        nperseg=n_win,
        noverlap=n_ovlap,
        nfft=n_fft,
        return_onesided=use_one_sided,
        scaling="density",
    )

    n_freq = len(f_test)
    f = f_test

    # Initialize output
    psd_matrix = np.zeros((n_freq, n_channels, n_instances))

    # Process each channel and instance
    for i_channel in range(n_channels):
        for i_instance in range(n_instances):
            sig = X[:, i_channel, i_instance]

            # Use scipy.signal.welch with equivalent parameters
            f_scipy, psd_scipy = signal.welch(
                sig,
                fs=fs,
                window="hamming",
                nperseg=n_win,
                noverlap=n_ovlap,
                nfft=n_fft,
                return_onesided=use_one_sided,
                scaling="density",
            )

            psd_matrix[:, i_channel, i_instance] = psd_scipy

    return psd_matrix, f, use_one_sided

    """
    Estimate power spectral density using Welch's method.
    
    Python equivalent of MATLAB's psdWelch_shm function. Computes the power
    spectral density of input signals using Welch's overlapped segment averaging
    method for reduced noise and improved frequency resolution.
    
    .. meta::
        :category: Core - Spectral Analysis
        :matlab_equivalent: psdWelch_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Frequency Domain
        :interactive_plot: True
        :typical_usage: ["vibration_analysis", "frequency_content", "noise_analysis"]
    
    Parameters
    ----------
    x : array_like, shape (n_samples,) or (n_samples, n_channels)
        Input signal array. If 2D, PSD is computed for each column.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: "Upload time series data"
            
    fs : float, optional, default=1.0
        Sampling frequency in Hz.
        
        .. gui::
            :widget: numeric_input
            :min: 0.1
            :max: 1000000.0
            :step: 0.1
            :units: "Hz"
            :description: "Sampling frequency"
            
    window : {"hann", "hamming", "blackman", "bartlett"}, optional, default="hann"
        Window function applied to each segment.
        
        .. gui::
            :widget: select
            :options: ["hann", "hamming", "blackman", "bartlett"]
            :description: "Window function type"
            
    nperseg : int, optional, default=None
        Length of each segment for Welch's method. If None, uses scipy default.
        
        .. gui::
            :widget: numeric_input
            :min: 8
            :max: 8192
            :step: 1
            :allow_none: True
            :description: "Segment length (None for auto)"
            
    noverlap : int, optional, default=None
        Number of points to overlap between segments. If None, uses scipy default.
        
        .. gui::
            :widget: numeric_input
            :min: 0
            :max: 4096
            :step: 1
            :allow_none: True
            :description: "Overlap points (None for auto)"
            
    nfft : int, optional, default=None
        Length of the FFT used. If None, uses nperseg.
        
        .. gui::
            :widget: numeric_input
            :min: 8
            :max: 16384
            :step: 1
            :allow_none: True
            :description: "FFT length (None for auto)"
            
    scaling : {"density", "spectrum"}, optional, default="density"
        Return power spectral density ('density') or power spectrum ('spectrum').
        
        .. gui::
            :widget: select
            :options: ["density", "spectrum"]
            :description: "Scaling type"
    
    Returns
    -------
    f : ndarray, shape (n_freqs,)
        Frequency array in Hz.
    psd : ndarray, shape (n_freqs,) or (n_freqs, n_channels)
        Power spectral density array. Units are signal²/Hz if scaling='density',
        or signal² if scaling='spectrum'.
        
        .. gui::
            :plot_type: "line"
            :x_axis: "f"
            :y_axis: "psd"
            :log_scale: "y"
            :xlabel: "Frequency (Hz)"
            :ylabel: "PSD (Units²/Hz)"
            
    Raises
    ------
    ValueError
        If input signal is empty or sampling frequency is non-positive.
        
    See Also
    --------
    stft : Short-time Fourier transform
    spectrogram : Compute spectrogram using Welch's method
    
    Notes
    -----
    Welch's method computes an estimate of the power spectral density by
    dividing the data into overlapping segments, windowing each segment,
    computing the periodogram of each segment, and averaging the periodograms.
    
    This implementation uses scipy.signal.welch as the underlying computation
    engine, ensuring compatibility with the broader scientific Python ecosystem.
    
    References
    ----------
    .. [1] Welch, P. "The use of fast Fourier transform for the estimation of 
           power spectra: A method based on time averaging over short, modified 
           periodograms", IEEE Transactions on Audio and Electroacoustics, 
           Vol. 15, pp. 70-73, 1967.
    
    Examples
    --------
    Basic usage with synthetic signal:
    
    >>> import numpy as np
    >>> from shmtools.core import psd_welch
    >>> 
    >>> # Generate test signal with 50 Hz component
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> x = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(len(t))
    >>> 
    >>> # Compute PSD
    >>> f, psd = psd_welch(x, fs=fs)
    >>> 
    >>> # Find peak frequency
    >>> peak_idx = np.argmax(psd)
    >>> peak_freq = f[peak_idx]
    >>> print(f"Peak frequency: {peak_freq:.1f} Hz")
    
    Multi-channel analysis:
    
    >>> # Generate multi-channel data
    >>> x_multi = np.column_stack([
    ...     np.sin(2*np.pi*25*t) + 0.1*np.random.randn(len(t)),
    ...     np.sin(2*np.pi*75*t) + 0.1*np.random.randn(len(t))
    ... ])
    >>> f, psd_multi = psd_welch(x_multi, fs=fs)
    >>> # psd_multi.shape = (n_freqs, 2)
    
    Custom windowing and segment parameters:
    
    >>> f, psd = psd_welch(x, fs=fs, window='blackman', nperseg=512)
    """
    # TODO: Implement Welch's method PSD estimation
    # This is a placeholder for the actual implementation
    f, psd = signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
        axis=0,
    )
    return f, psd


def stft(
    x: np.ndarray,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT).

    Python equivalent of MATLAB's stft function. Provides time-frequency
    analysis by computing FFT over short, overlapping time windows.

    .. meta::
        :category: Core - Time-Frequency
        :matlab_equivalent: stft
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Time-Frequency

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for STFT analysis"

    fs : float, optional, default=1.0
        Sampling frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 0.1
            :max: 1000000.0
            :description: "Sampling frequency (Hz)"

    window : str, optional, default="hann"
        Window function.

        .. gui::
            :widget: select
            :options: ["hann", "hamming", "blackman", "bartlett"]
            :description: "Window function"

    nperseg : int, optional, default=256
        Length of each segment.

        .. gui::
            :widget: numeric_input
            :min: 8
            :max: 8192
            :description: "Segment length"

    noverlap : int, optional
        Number of points to overlap between segments.

        .. gui::
            :widget: numeric_input
            :min: 0
            :max: 4096
            :allow_none: True
            :description: "Overlap points"

    nfft : int, optional
        Length of the FFT used.

        .. gui::
            :widget: numeric_input
            :min: 8
            :max: 16384
            :allow_none: True
            :description: "FFT length"

    Returns
    -------
    f : np.ndarray
        Frequency array in Hz.

        .. gui::
            :plot_type: "reference"
            :description: "Frequency vector"

    t : np.ndarray
        Time array in seconds.

        .. gui::
            :plot_type: "reference"
            :description: "Time vector"

    Zxx : np.ndarray, complex
        STFT matrix (frequency x time).

        .. gui::
            :plot_type: "spectrogram"
            :x_axis: "t"
            :y_axis: "f"
            :xlabel: "Time (s)"
            :ylabel: "Frequency (Hz)"

    Examples
    --------
    Basic STFT analysis:

    >>> import numpy as np
    >>> from shmtools.core import stft
    >>>
    >>> # Generate chirp signal
    >>> fs = 1000
    >>> t_sig = np.linspace(0, 2, 2*fs, endpoint=False)
    >>> f0, f1 = 50, 200
    >>> x = np.sin(2*np.pi*(f0 + (f1-f0)*t_sig/2)*t_sig)
    >>>
    >>> # Compute STFT
    >>> f, t, Zxx = stft(x, fs=fs, nperseg=256)
    >>>
    >>> print(f"Frequency resolution: {f[1]-f[0]:.2f} Hz")
    >>> print(f"Time resolution: {t[1]-t[0]:.3f} s")
    """
    f, t, Zxx = signal.stft(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    return f, t, Zxx


def spectrogram(
    x: np.ndarray,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    scaling: str = "density",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram of input signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    window : str, optional
        Window function. Default is 'hann'.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int, optional
        Number of points to overlap between segments.
    nfft : int, optional
        Length of the FFT used.
    scaling : str, optional
        'density' or 'spectrum' scaling.

    Returns
    -------
    f : np.ndarray
        Frequency array.
    t : np.ndarray
        Time array.
    Sxx : np.ndarray
        Spectrogram of x.
    """
    # TODO: Implement spectrogram
    f, t, Sxx = signal.spectrogram(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
    )
    return f, t, Sxx


def cepstrum(x: np.ndarray) -> np.ndarray:
    """
    Compute the real cepstrum of a signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal.

    Returns
    -------
    c : np.ndarray
        Real cepstrum of x.
    """
    # TODO: Implement cepstrum calculation
    # c = ifft(log(abs(fft(x))))
    X = np.fft.fft(x)
    c = np.real(np.fft.ifft(np.log(np.abs(X) + 1e-12)))
    return c


def fast_kurtogram(
    x: np.ndarray, fs: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute fast kurtogram for bearing fault detection.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : float, optional
        Sampling frequency. Default is 1.0.

    Returns
    -------
    f : np.ndarray
        Frequency array.
    level : np.ndarray
        Level array.
    kurt : np.ndarray
        Kurtogram matrix.
    """
    # TODO: Implement fast kurtogram algorithm
    # This is a complex algorithm that will need a full implementation
    raise NotImplementedError("Fast kurtogram not yet implemented")


def cwt_analysis(
    x: np.ndarray,
    fs: float,
    frequencies: Optional[np.ndarray] = None,
    wavelet: str = "morlet2",
    w: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Continuous Wavelet Transform (CWT) analysis.

    Python equivalent of MATLAB's cwt function. Provides time-frequency
    analysis using continuous wavelets, excellent for analyzing transient
    events and non-stationary signals.

    .. meta::
        :category: Core - Time-Frequency
        :matlab_equivalent: cwt
        :complexity: Advanced
        :data_type: Time Series
        :output_type: Time-Frequency
        :sensitivity: Transient, Non-stationary

    Parameters
    ----------
    x : np.ndarray
        Input signal (1D array).

        .. gui::
            :widget: data_input
            :description: "Signal for CWT analysis"

    fs : float
        Sampling frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 0.1
            :max: 1000000.0
            :description: "Sampling frequency (Hz)"

    frequencies : np.ndarray, optional
        Array of frequencies to analyze. If None, creates logarithmic spacing.

        .. gui::
            :widget: frequency_range
            :description: "Frequency range for analysis"

    wavelet : str, optional, default='morlet2'
        Wavelet type ('morlet2', 'morlet', 'mexh').

        .. gui::
            :widget: select
            :options: ["morlet2", "morlet", "mexh"]
            :description: "Wavelet type"

    w : float, optional, default=6.0
        Wavelet parameter (bandwidth for Morlet).

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 20.0
            :description: "Wavelet bandwidth"

    Returns
    -------
    coefficients : np.ndarray, complex
        CWT coefficients matrix (frequency x time).

        .. gui::
            :plot_type: "scalogram"
            :x_axis: "t"
            :y_axis: "f"
            :xlabel: "Time (s)"
            :ylabel: "Frequency (Hz)"

    frequencies : np.ndarray
        Frequency vector in Hz.

        .. gui::
            :plot_type: "reference"
            :description: "Frequency vector"

    t : np.ndarray
        Time vector in seconds.

        .. gui::
            :plot_type: "reference"
            :description: "Time vector"

    Examples
    --------
    Basic CWT analysis:

    >>> import numpy as np
    >>> from shmtools.core import cwt_analysis
    >>>
    >>> # Generate signal with transient
    >>> fs = 1000
    >>> t_sig = np.linspace(0, 2, 2*fs, endpoint=False)
    >>> x = np.sin(2*np.pi*50*t_sig)
    >>> # Add transient at t=1s
    >>> impulse_idx = int(fs)
    >>> x[impulse_idx:impulse_idx+50] += 5*np.exp(-np.arange(50)/10)
    >>>
    >>> # Compute CWT
    >>> frequencies = np.logspace(1, 2.5, 50)  # 10-316 Hz
    >>> coeffs, freqs, t = cwt_analysis(x, fs, frequencies)
    >>>
    >>> print(f"CWT shape: {coeffs.shape}")
    >>> print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")

    Analyzing gear mesh harmonics:

    >>> # Focus on gear mesh frequency and harmonics
    >>> gear_mesh_freq = 300  # Hz
    >>> frequencies = np.array([gear_mesh_freq * i for i in range(1, 6)])
    >>> coeffs, freqs, t = cwt_analysis(x, fs, frequencies)
    """
    from scipy import signal
    import warnings

    # Validate inputs
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    if x.ndim > 1:
        raise ValueError("CWT currently supports only 1D signals")

    # Default frequency range if not provided
    if frequencies is None:
        # Create logarithmic frequency spacing from fs/100 to fs/4
        f_min = max(fs / 100, 1.0)
        f_max = min(fs / 4, fs / 2.56)  # Nyquist with some margin
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), 50)

    frequencies = np.asarray(frequencies)

    if len(frequencies) == 0:
        raise ValueError("Frequencies array cannot be empty")

    # Convert frequencies to scales for wavelets
    # For Morlet wavelet: scale = fc * fs / (frequency * 2*pi)
    # where fc is the center frequency of the wavelet
    if wavelet.lower() in ["morlet2", "morlet"]:
        # Morlet wavelet center frequency
        fc = w / (2 * np.pi)  # For morlet2
        scales = fc * fs / frequencies
    elif wavelet.lower() == "mexh":
        # Mexican hat wavelet
        fc = 1.0 / (2 * np.pi)
        scales = fc * fs / frequencies
    else:
        raise ValueError(f"Unsupported wavelet type: {wavelet}")

    # Remove any invalid scales
    valid_scales = scales > 0
    if not np.all(valid_scales):
        warnings.warn("Some frequencies resulted in invalid scales and will be ignored")
        scales = scales[valid_scales]
        frequencies = frequencies[valid_scales]

    # Select wavelet function
    if wavelet.lower() == "morlet2":
        wavelet_func = signal.morlet2
    elif wavelet.lower() == "morlet":
        wavelet_func = signal.morlet
    elif wavelet.lower() == "mexh":
        wavelet_func = signal.ricker
    else:
        raise ValueError(f"Unsupported wavelet type: {wavelet}")

    # Compute CWT
    if wavelet.lower() in ["morlet2", "morlet"]:
        coefficients = signal.cwt(x, wavelet_func, scales, w=w)
    else:
        coefficients = signal.cwt(x, wavelet_func, scales)

    # Create time vector
    t = np.arange(len(x)) / fs

    return coefficients, frequencies, t
