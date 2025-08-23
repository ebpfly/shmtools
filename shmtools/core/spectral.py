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
    Estimate power spectral density via Welch's method.

    .. meta::
        :category: Feature Extraction - Spectral Analysis
        :matlab_equivalent: psdWelch_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Spectral
        :display_name: Power Spectral Density via Welch's Method
        :verbose_call: [PSD Matrix, Frequency Vector, PSD Range] = Power Spectral Density via Welch's Method (Signal Matrix, Window Length, Overlap Length, FFT Bins, Sampling Frequency, Use One-Sided PSD)

    Computes the power spectral density of a digital signal using Welch's
    method. Welch's method segments a signal into a specified number of
    windowed signals with a specified number of overlapping samples and
    computes their Fourier transforms independently then averages these
    Fourier transforms to get an estimate of the power spectral density of
    a signal. Typically the number of overlapping segments should be
    between 50 to 75 percent of the window length.

    Parameters
    ----------
    X : ndarray, shape (time, channels, instances)
        Matrix of time series data.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Time series data matrix

    n_win : int, optional
        Samples per window. Default is 8 windows based on signal length.

        .. gui::
            :widget: number_input
            :min: 16
            :max: 2048
            :description: Window length in samples

    n_ovlap : int, optional
        Number of overlapping samples between windows. Default is 50% of window.

        .. gui::
            :widget: number_input
            :min: 0
            :description: Number of overlapping samples

    n_fft : int, optional
        Number of FFT frequency bins. Default is max(256, nextPow2(nWin)).

        .. gui::
            :widget: number_input
            :min: 64
            :max: 4096
            :description: Number of FFT frequency bins

    fs : float, optional
        Sampling frequency in Hz. Default is 1.

        .. gui::
            :widget: number_input
            :min: 0.1
            :max: 100000.0
            :default: 1.0
            :units: "Hz"
            :description: Sampling frequency

    use_one_sided : bool, optional
        Create one-sided PSD instead of two-sided. Default is True.

        .. gui::
            :widget: checkbox
            :default: true
            :description: Use one-sided PSD

    Returns
    -------
    psd_matrix : ndarray, shape (n_fft, channels, instances)
        Power spectral density matrix.

        .. gui::
            :plot_type: "spectral"
            :description: Power spectral density matrix

    f : ndarray, shape (n_fft,)
        Frequency vector corresponding to psd_matrix.

        .. gui::
            :plot_type: "line"
            :description: Frequency vector

    use_one_sided : bool
        Specifies type of PSD: one-sided or two-sided.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core import psd_welch_shm
    >>>
    >>> # Generate test signal with dominant frequency
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(fs)
    >>> X = signal.reshape(-1, 1, 1)  # (time, channels, instances)
    >>>
    >>> # Compute PSD
    >>> psd, f, one_sided = psd_welch_shm(X, fs=fs)
    >>> peak_freq = f[np.argmax(psd[:, 0, 0])]
    >>> print(f"Peak frequency: {peak_freq:.1f} Hz")

    References
    ----------
    [1] Welch P., The Use of Fast Fourier Transform for the Estimation of
    Power Spectra: A Method Based on Time Averaging Over Short, Modified
    Periodograms. IEEE Transactions on Audio and ElectroAcoustics, Vol.
    AU-15, No. 2. June 1967.

    See Also
    --------
    frf_shm : Frequency response function estimation
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
                window="hamming",  # MATLAB default
                nperseg=n_win,
                noverlap=n_ovlap,
                nfft=n_fft,
                return_onesided=use_one_sided,
                scaling="density",
            )

            psd_matrix[:, i_channel, i_instance] = psd_scipy

    return psd_matrix, f, use_one_sided


def stft_shm(
    X: np.ndarray,
    nWin: Optional[int] = None,
    nOvlap: Optional[int] = None,
    nFFT: Optional[int] = None,
    Fs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT).

    .. meta::
        :category: Feature Extraction - Spectral Analysis
        :matlab_equivalent: stft_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Time-Frequency
        :display_name: Short-Time Fourier Transform
        :verbose_call: [Frequency, Time, STFT] = Short-Time Fourier Transform (Signal, Sampling Rate, Window, Segment Length, Overlap, FFT Length)

    Parameters
    ----------
    X : array_like
        Input signal matrix (TIME, CHANNELS, INSTANCES).
    nWin : int, optional
        Short-time window length.
    nOvlap : int, optional
        Number of overlapping window samples.
    nFFT : int, optional
        Number of FFT frequency bins.
    Fs : float, optional
        Sampling frequency in Hz.

    Returns
    -------
    stftMatrix : ndarray
        STFT matrix (NFREQ, TIME, CHANNELS, INSTANCES).
    f : ndarray
        Frequency vector.
    t : ndarray
        Time vector.
    """
    from ..core.preprocessing import window_shm
    
    # Get matrix dimensions
    nSignal, nChannel, nInstance = X.shape
    
    # Set defaults based on MATLAB implementation
    if nWin is None:
        nWin = nSignal // 10
    if nOvlap is None:
        nOvlap = int(0.5 * nWin)
    if nFFT is None:
        nFFT = 2 ** int(np.ceil(np.log2(nWin)))
    if Fs is None:
        Fs = 1.0
    
    # Initialize output matrix
    stftMatrix = None
    
    # Compute STFT for each channel and instance
    for iChannel in range(nChannel):
        for iInstance in range(nInstance):
            # Extract single signal
            x = X[:, iChannel, iInstance]
            
            # Generate window
            win = window_shm('hamming', nWin, None)
            
            # Compute number of segments
            nSegments = int(np.floor((len(x) - nOvlap) / (nWin - nOvlap)))
            
            # Build windowed signal matrix
            xWinMTX = np.zeros((nWin, nSegments))
            for i in range(nSegments):
                start_idx = i * (nWin - nOvlap)
                end_idx = start_idx + nWin
                if end_idx <= len(x):
                    xWinMTX[:, i] = x[start_idx:end_idx] * win
            
            # Compute FFT
            kMtx = np.fft.fftshift(np.fft.fft(xWinMTX, nFFT, axis=0), axes=0)
            
            # Take positive frequencies only (one-sided)
            if nFFT % 2 == 1:  # nFFT is ODD
                mid_idx = (nFFT + 1) // 2
                kMtx = kMtx[mid_idx:, :]
            else:  # nFFT is EVEN
                mid_idx = (nFFT + 2) // 2 - 1
                kMtx = kMtx[mid_idx:, :]
            
            # Initialize output matrix on first iteration
            if stftMatrix is None:
                nF, nT = kMtx.shape
                stftMatrix = np.zeros((nF, nT, nChannel, nInstance), dtype=complex)
            
            # Store result
            stftMatrix[:, :, iChannel, iInstance] = kMtx
    
    # Generate frequency and time vectors
    f_full = np.linspace(-0.5, 0.5 - 1/nFFT, nFFT) * Fs
    if nFFT % 2 == 1:  # nFFT is ODD
        mid_idx = (nFFT + 1) // 2
        f = f_full[mid_idx:]
    else:  # nFFT is EVEN
        mid_idx = (nFFT + 2) // 2 - 1
        f = f_full[mid_idx:]
    
    t = (nWin / 2 / Fs) + np.arange(nSegments) * (nWin - nOvlap) / Fs
    
    return stftMatrix, f, t


def cwt_analysis_shm(
    x: np.ndarray, scales: np.ndarray, wavelet: str = "morlet", fs: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous Wavelet Transform analysis.

    .. meta::
        :category: Feature Extraction - Spectral Analysis
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
            wavelet_func = lambda t: np.exp(1j * 2 * np.pi * t / scale) * np.exp(
                -(t**2) / (2 * sigma**2)
            )
        else:
            # Default to Ricker (Mexican hat)
            wavelet_func = sig.ricker

        # Convolution-based CWT (simplified)
        t_wav = np.arange(-3 * scale, 3 * scale + 1)
        if len(t_wav) > len(x):
            t_wav = np.arange(-len(x) // 2, len(x) // 2 + 1)

        if wavelet == "morlet":
            wav = wavelet_func(t_wav)
        else:
            wav = wavelet_func(len(t_wav), scale)

        coeffs[i, :] = np.convolve(x, wav, mode="same")

    return coeffs, freqs


def wavelet_shm(wave_type: str, wave_param: np.ndarray) -> np.ndarray:
    """
    Generate wavelet of specified type.
    
    .. meta::
        :category: Feature Extraction - Spectral Analysis
        :matlab_equivalent: wavelet_shm
        :complexity: Intermediate
        :data_type: Parameters
        :output_type: Wavelet
        :display_name: Wavelet Generator
        :verbose_call: [Wavelet] = Wavelet Generator (Wavelet Type, Wavelet Parameters)
        
    Generates a specified wavelet of type 'Morlet', 'Shannon' or 'Bspline'.
    Wavelet parameters allow user to specify the size of the wavelet
    desired as well as its central frequency.
    
    Parameters
    ----------
    wave_type : str
        Wavelet type specification: 'morlet', 'shannon', or 'bspline'.
        
        .. gui::
            :widget: select
            :options: ["morlet", "shannon", "bspline"]
            :default: "morlet"
            :description: Type of mother wavelet to generate
            
    wave_param : array_like, shape (2,) or (3,)
        Wavelet parameter vector [Fc, Nw] or [Fc, Nw, useComplex]:
        - Fc: wavelet central frequency
        - Nw: half the wavelet length omitting the central element  
        - useComplex: complex (True) or real (False) valued wavelets
        
        .. gui::
            :widget: array_input
            :size: 3
            :description: [Central frequency, Half-length, Use complex (0/1)]
    
    Returns
    -------
    wavelet : ndarray, shape (2*nWavelet+1,)
        Symmetric wavelet array.
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.spectral import wavelet_shm
    >>> 
    >>> # Generate Morlet wavelet
    >>> fc = 0.1  # Central frequency
    >>> nw = 16   # Half-length
    >>> wave_param = [fc, nw, 1]  # Complex wavelet
    >>> 
    >>> wavelet = wavelet_shm('morlet', wave_param)
    >>> print(f"Wavelet length: {len(wavelet)}")
    >>> print(f"Wavelet is complex: {np.iscomplexobj(wavelet)}")
    
    References
    ----------
    [1] Liverpool John Moores University, "Mother Wavelets",
    http://www.ljmu.ac.uk/GERI/98293.htm. Web.
    """
    wave_type = wave_type.upper()
    
    # Handle different parameter formats
    if len(wave_param) == 1:
        fc = wave_param[0]
        n_wavelet = None
        analytic_flag = None
    elif len(wave_param) == 2:
        fc = wave_param[0]
        n_wavelet = wave_param[1]
        analytic_flag = None
    elif len(wave_param) >= 3:
        fc = wave_param[0]
        n_wavelet = wave_param[1]
        analytic_flag = wave_param[2]
    else:
        raise ValueError("Must provide at least wavelet central frequency")
    
    # Determine default wavelet order
    if n_wavelet is None:
        n_t = 2.25  # Number of periods in wavelet
        n_wavelet = int(np.ceil(n_t / fc))  # Number of samples in wavelet
    else:
        n_wavelet = int(n_wavelet)
    
    # Default: Real wavelet
    if analytic_flag is None:
        analytic_flag = False
    else:
        analytic_flag = bool(analytic_flag)
    
    # Create time vector
    x = np.arange(-n_wavelet, n_wavelet + 1)
    
    # Generate wavelets based on type
    if wave_type == 'MORLET':
        tol = 1e-3
        fb = np.sqrt((n_wavelet**2) / (-2 * np.log(tol)))
        wavelet = ((np.pi * fb**2)**(-0.25)) * np.exp(2*np.pi*1j*fc*x) * np.exp(-(x**2)/(2*fb**2))
        
    elif wave_type == 'SHANNON':
        fb = 4 / n_wavelet
        wavelet = np.sqrt(fb) * np.exp(2*np.pi*1j*fc*x) * np.sinc(fb*x)
        
    elif wave_type == 'BSPLINE':
        M = 3
        fb = 4 / n_wavelet
        wavelet = np.sqrt(fb) * np.exp(2*np.pi*1j*fc*x) * (np.sinc(fb*x/M))**M
        
    else:
        raise ValueError(f"Unknown wavelet type: {wave_type}. Must be 'morlet', 'shannon', or 'bspline'")
    
    # Convert to real if requested
    if not analytic_flag:  # Real valued wavelet
        wavelet = np.sqrt(2) * np.real(wavelet)
    
    return wavelet


def cwt_scalogram_shm(
    X: np.ndarray, 
    fs: Optional[float] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None, 
    n_scale: Optional[int] = None,
    wave_order: Optional[int] = None,
    wave_type: Optional[str] = None,
    use_analytic: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute continuous wavelet scalograms using mirrored Morlet wavelets.
    
    .. meta::
        :category: Feature Extraction - Spectral Analysis
        :matlab_equivalent: cwtScalogram_shm
        :complexity: Advanced
        :data_type: Time Series
        :output_type: Time-Frequency
        :display_name: Continuous Wavelet Scalogram
        :verbose_call: [Scalogram Matrix, Frequency Vector, Time Vector] = Continuous Wavelet Scalogram (Signal Matrix, Sampling Frequency, Minimum Frequency, Maximum Frequency, Number of Wavelet Scales, Wavelet Order Parameter, Wavelet Type, Use Complex Wavelet)
        
    Computes the continuous wavelet scalograms of a matrix of digital signals.
    The CWT scalogram has a dyadic time frequency output where the time resolution 
    and frequency resolutions vary over the time frequency domain. High frequency 
    scales have better time resolution and poor frequency resolution whereas the 
    low frequency scales have poor time resolution and better frequency resolution.
    
    Parameters
    ----------
    X : ndarray, shape (time, channels, instances)
        Matrix of time series data.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Time series data matrix
            
    fs : float, optional, default=1
        Sampling frequency in Hz.
        
        .. gui::
            :widget: number_input
            :min: 0.1
            :max: 100000.0
            :default: 1.0
            :description: Sampling frequency (Hz)
            
    f_min : float, optional, default=0.001
        Minimum scalogram frequency, f(1).
        
        .. gui::
            :widget: number_input
            :min: 0.0001
            :max: 1.0
            :default: 0.001
            :description: Minimum frequency (cycles/sample)
            
    f_max : float, optional, default=0.5
        Maximum scalogram frequency, f(n_scale).
        
        .. gui::
            :widget: number_input
            :min: 0.001
            :max: 0.5
            :default: 0.5
            :description: Maximum frequency (cycles/sample)
            
    n_scale : int, optional, default=64
        Number of scalogram frequency scales.
        
        .. gui::
            :widget: number_input
            :min: 8
            :max: 512
            :default: 64
            :description: Number of frequency scales
            
    wave_order : int, optional, default=16
        Wavelet order, imaginary implies analytic wavelets.
        
        .. gui::
            :widget: number_input
            :min: 4
            :max: 64
            :default: 16
            :description: Wavelet order parameter
            
    wave_type : str, optional, default='morlet'
        Wavelet type: 'morlet', 'shannon', or 'bspline'.
        
        .. gui::
            :widget: select
            :options: ["morlet", "shannon", "bspline"]
            :default: "morlet"
            :description: Type of mother wavelet
            
    use_analytic : bool, optional, default=False
        Use complex wavelets.
        
        .. gui::
            :widget: checkbox
            :default: false
            :description: Use complex-valued wavelets
    
    Returns
    -------
    scalo_matrix : ndarray, shape (n_scale, time, channels, instances)
        Continuous wavelet transform scalogram matrix.
        
        .. gui::
            :plot_type: "time_frequency"
            :description: CWT scalogram matrix
            
    f : ndarray, shape (n_scale,)
        Frequency vector corresponding to CWT scalogram.
        
        .. gui::
            :plot_type: "line"
            :description: Frequency vector
            
    t : ndarray, shape (time,)
        Time vector corresponding to CWT scalogram.
        
        .. gui::
            :plot_type: "line"
            :description: Time vector
            
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.spectral import cwt_scalogram_shm
    >>> 
    >>> # Generate test signal with gear mesh frequency
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs)
    >>> # Simulate gear mesh at 27 Hz with modulation
    >>> signal = np.sin(2*np.pi*27*t) + 0.5*np.sin(2*np.pi*54*t) + 0.1*np.random.randn(fs)
    >>> X = signal.reshape(-1, 1, 1)  # (time, channels, instances)
    >>> 
    >>> # Compute scalogram
    >>> scalo, freqs, time = cwt_scalogram_shm(X, fs=fs, n_scale=128)
    >>> print(f"Scalogram shape: {scalo.shape}")
    >>> print(f"Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    
    References
    ----------
    [1] Nguyen, D., et al., "Structural Damage Detection Using the
    Hölder Exponent", SPIE's 10th Annual International Symposium On Smart
    Structures and Materials San Diego, CA March 2-6, 2003.
    
    [2] Sohn, H., et. al. (2003), "Singularity Detection for Structural
    Health Monitoring Using Hölder Exponents", Los Alamos National
    Laboratory Report.
    """
    # Set defaults
    if fs is None:
        fs = 1.0
    if f_min is None:
        f_min = 0.001
    if f_max is None:
        f_max = 0.5
    if n_scale is None:
        n_scale = 64
    if wave_order is None:
        wave_order = 16
    if wave_type is None:
        wave_type = 'morlet'
    if use_analytic is None:
        use_analytic = False
    
    # Get matrix dimensions
    n_signal, n_channel, n_instance = X.shape
    
    # Process each channel and instance
    scalo_matrix = None
    f = None
    t = None
    
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            # Compute CWT scalogram for single signal
            scalo, f, t = _cwt_scalo_single(
                X[:, i_channel, i_instance], fs, f_min, f_max, n_scale,
                wave_order, wave_type, use_analytic
            )
            
            # Allocate memory on first iteration
            if scalo_matrix is None:
                n_f, n_t = scalo.shape
                scalo_matrix = np.zeros((n_f, n_t, n_channel, n_instance))
            
            # Store scalogram in the stack
            scalo_matrix[:, :, i_channel, i_instance] = scalo
    
    return scalo_matrix, f, t


def _cwt_scalo_single(
    X: np.ndarray, 
    fs: float, 
    f_min: float, 
    f_max: float, 
    n_scale: int,
    wave_order: int, 
    wave_type: str, 
    use_analytic: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CWT scalogram for a single signal.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples,)
        Input signal.
    fs : float
        Sampling frequency.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    n_scale : int
        Number of scales.
    wave_order : int
        Wavelet order.
    wave_type : str
        Wavelet type.
    use_analytic : bool
        Use analytic wavelets.
        
    Returns
    -------
    scalo : ndarray, shape (n_scale, n_samples)
        Scalogram matrix.
    f : ndarray, shape (n_scale,)
        Frequency vector.
    t : ndarray, shape (n_samples,)
        Time vector.
    """
    # Normalize frequencies
    f_min_norm = f_min / fs if f_min > 1 else f_min
    f_max_norm = f_max / fs if f_max > 1 else f_max
    
    # Check frequency bounds
    if f_min_norm >= f_max_norm:
        raise ValueError('f_max must be greater than f_min')
    
    # Get signal length
    n_signal = len(X)
    
    # Allocate memory for scalogram
    scalo = np.zeros((n_scale, n_signal))
    
    # Assign frequency scales and wavelet sample width factors
    f = 2**np.linspace(np.log2(f_max_norm), np.log2(f_min_norm), n_scale)
    a = 2**np.linspace(np.log2(1), np.log2(f_max_norm/f_min_norm), n_scale)
    
    # Assign wavelet order
    n_order = abs(wave_order)
    
    # Compute mirrored continuous wavelet scalogram
    for i in range(n_scale):
        n_wavelet = int(np.round(n_order * a[i]))
        
        # Generate wavelet
        wave_param = [f[i], n_wavelet, use_analytic]
        wavelet = wavelet_shm(wave_type, wave_param)
        
        # Create mirrored signal
        n_mirror = min(n_signal, n_wavelet)
        x_mirrored = np.concatenate([
            X[n_mirror-1:0:-1],  # Reversed beginning
            X,                   # Original signal
            X[-2:-n_mirror-1:-1] # Reversed end
        ])
        
        # Convolution using FFT
        n_mir = len(x_mirrored)
        n_wav = len(wavelet)
        n_fft = n_mir + n_wav - 1
        
        # Pad and compute FFT-based convolution
        X_fft = np.fft.fft(x_mirrored, n_fft)
        W_fft = np.fft.fft(wavelet, n_fft)
        CWT = np.fft.ifft(X_fft * W_fft, n_fft)
        
        # Extract relevant portion
        start_idx = n_wavelet + n_mirror
        end_idx = start_idx + n_signal
        scalo[i, :] = CWT[start_idx:end_idx]
    
    # Convert to power (magnitude squared)
    scalo = np.real(scalo * np.conj(scalo))
    
    # Assign scalogram frequency and time vectors
    f = f * fs
    t = np.arange(n_signal) / fs
    
    return scalo, f, t


def hoelder_exp_shm(scalo_matrix: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate Hoelder exponent series from time-frequency scalogram matrix.
    
    .. meta::
        :category: Feature Extraction - Spectral Analysis
        :matlab_equivalent: hoelderExp_shm
        :complexity: Advanced
        :data_type: Time-Frequency
        :output_type: Time Series
        :display_name: Hoelder Exponent
        :verbose_call: [Hoelder Exponent Matrix] = Hoelder Exponent (Scalogram, Scalogram Frequency Vector)
        
    Computes the matrix of Hoelder exponent series from a matrix of time frequency 
    scalograms. The Hoelder exponent is essentially the change in the slope along 
    the frequency domain of a time frequency matrix as a function of time. When 
    used with scalograms of a signal, if the signal contains non-linearities the 
    high frequency content in the time-frequency domain increases and increases 
    the slope. This allows for non-linearities to be more easily detected in 
    vibration signals.
    
    Parameters
    ----------
    scalo_matrix : ndarray, shape (n_freq, time, channels, instances)
        Continuous wavelet transform scalogram matrix (or other time-frequency matrix).
        
        .. gui::
            :widget: data_input
            :description: Time-frequency scalogram matrix
            
    f : ndarray, shape (n_freq,)
        CWT scalogram frequency vector.
        
        .. gui::
            :widget: array_input
            :description: Frequency vector corresponding to scalogram
    
    Returns
    -------
    hoelder_matrix : ndarray, shape (time, channels, instances)
        Hoelder exponent series matrix.
        
        .. gui::
            :plot_type: "time_series"
            :description: Hoelder exponent time series
            
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.spectral import cwt_scalogram_shm, hoelder_exp_shm
    >>> 
    >>> # Generate test signal with impacts (non-linearities)
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs)
    >>> signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(fs)
    >>> # Add impulses to simulate gear impacts
    >>> impact_times = [0.1, 0.3, 0.5, 0.7, 0.9]
    >>> for t_impact in impact_times:
    >>>     idx = int(t_impact * fs)
    >>>     signal[idx:idx+5] += 5.0  # Sharp impulse
    >>> X = signal.reshape(-1, 1, 1)
    >>> 
    >>> # Compute scalogram
    >>> scalo, freqs, time = cwt_scalogram_shm(X, fs=fs, n_scale=64)
    >>> 
    >>> # Extract Hoelder exponents
    >>> hoelder = hoelder_exp_shm(scalo, freqs)
    >>> print(f"Hoelder series shape: {hoelder.shape}")
    >>> print(f"Hoelder range: {hoelder.min():.3f} to {hoelder.max():.3f}")
    
    Notes
    -----
    The Hoelder exponent is computed as the slope of log(scalogram) vs log(frequency)
    at each time instant. Higher values indicate more non-linear, impulsive content
    which is characteristic of gear tooth impacts and bearing defects.
    
    The algorithm performs a linear regression in log-log space:
    log(scalogram) = alpha * log(frequency) + beta
    
    The Hoelder exponent h = -(alpha + 1) / 2
    
    References
    ----------
    [1] Nguyen, D., et al., "Structural Damage Detection Using the
    Hölder Exponent", SPIE's 10th Annual International Symposium On Smart
    Structures and Materials San Diego, CA March 2-6, 2003.
    
    [2] Sohn, H., et. al. (2003), "Singularity Detection for Structural
    Health Monitoring Using Hölder Exponents", Los Alamos National
    Laboratory Report.
    """
    # Get scalogram matrix dimension sizes
    n_freq, n_signal, n_channel, n_instance = scalo_matrix.shape
    
    # Allocate memory for Hoelder matrix
    hoelder_matrix = np.zeros((n_signal, n_channel, n_instance))
    
    # Compute feature values for (nChannel x nInstance) space
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            hoelder_matrix[:, i_channel, i_instance] = _hoelder_single(
                scalo_matrix[:, :, i_channel, i_instance], f
            )
    
    return hoelder_matrix


def _hoelder_single(scalo: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate Hoelder time series from single wavelet scalogram.
    
    Parameters
    ----------
    scalo : ndarray, shape (n_freq, n_time)
        Single scalogram matrix.
    f : ndarray, shape (n_freq,)
        Frequency vector.
        
    Returns
    -------
    h : ndarray, shape (n_time,)
        Hoelder exponent time series.
    """
    # Convert to log scale
    scalo = np.log10(np.abs(scalo + 1e-12))  # Add small constant to avoid log(0)
    f = np.log10(f).reshape(-1, 1)  # Column vector
    
    # Get dimensions
    N, S = scalo.shape  # N = n_freq, S = n_time
    
    # Compute regression coefficients for linear fit in log-log space
    # We're fitting: log(scalo) = alpha * log(f) + beta
    # Using least squares: alpha = (N*sum(f*scalo) - sum(f)*sum(scalo)) / (N*sum(f^2) - sum(f)^2)
    
    sum_f = np.sum(f)
    sum_f2 = np.sum(f**2)
    C = N * sum_f2 - sum_f**2
    
    # Vectorized computation for all time samples
    # sum(f * scalo) for each time sample
    sum_f_scalo = np.sum(f * scalo, axis=0)  # Shape: (n_time,)
    sum_scalo = np.sum(scalo, axis=0)        # Shape: (n_time,)
    
    # Compute slope (alpha) for each time sample
    alpha = (N * sum_f_scalo - sum_f * sum_scalo) / C
    
    # Convert slope to Hoelder exponent
    # Based on MATLAB: h = (-(alpha + 1) / 2)
    h = -(alpha + 1) / 2
    
    return h


def dwvd_shm(
    X: np.ndarray, 
    n_win: Optional[int] = None, 
    n_ovlap: Optional[int] = None, 
    n_fft: Optional[int] = None, 
    fs: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute discrete Wigner-Ville distributions from signal matrix.
    
    .. meta::
        :category: Feature Extraction - Spectral Analysis
        :matlab_equivalent: dwvd_shm
        :complexity: Advanced
        :data_type: Time Series
        :output_type: Time-Frequency
        :display_name: Discrete Wigner-Ville Distribution
        :verbose_call: [Wigner Ville Distribution Matrix, Frequency Vector, Time Vector] = Discrete Wigner-Ville Distribution (Signal Matrix, Samples per Window, Number of Overlapping Window Samples, Number of FFT Bins, Sampling Frequency)
        
    Computes the discrete Wigner-Ville matrix from signals. The function
    computes a Wigner-Ville distribution for each signal in the signal
    matrix by segmenting the signal into n_win length signals with n_ovlap
    overlapping samples. From the segments an analytic signal is computed
    and weighted kernel functions are formed to compute the Wigner-Ville
    Distribution via DFT. The Wigner Ville Distribution has better time and
    frequency resolution than the STFT but can suffer from interference
    negative values which are not actually present in the signal being
    analyzed.
    
    Parameters
    ----------
    X : ndarray, shape (time, channels, instances)
        Matrix of time series data.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Time series matrix for DWVD analysis
            
    n_win : int, optional
        Samples per window length. Default is signal_length/10.
        
        .. gui::
            :widget: number_input
            :min: 32
            :max: 2048
            :description: Window length in samples
            
    n_ovlap : int, optional
        Number of overlapping samples between windows. Default is 50% of window.
        
        .. gui::
            :widget: number_input
            :min: 0
            :description: Window overlap in samples
            
    n_fft : int, optional
        Number of frequency bins in FFT. Default is next power of 2 of n_win.
        
        .. gui::
            :widget: number_input
            :min: 64
            :max: 4096
            :description: FFT size
            
    fs : float, optional
        Sampling frequency in Hz. Default is 1.
        
        .. gui::
            :widget: number_input
            :min: 0.1
            :description: Sampling frequency (Hz)
    
    Returns
    -------
    dwvd_matrix : ndarray, shape (n_fft, n_time, channels, instances)
        Discrete Wigner-Ville distribution matrix.
        
    f : ndarray, shape (n_fft,)
        Frequency vector corresponding to dwvd_matrix.
        
    t : ndarray, shape (n_time,)
        Time vector corresponding to dwvd_matrix.
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.spectral import dwvd_shm
    >>> 
    >>> # Generate chirp signal (frequency sweep)
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs)
    >>> # Linear chirp from 50 to 200 Hz
    >>> signal = np.sin(2*np.pi*(50 + 75*t)*t)
    >>> X = signal.reshape(-1, 1, 1)
    >>> 
    >>> # Compute DWVD
    >>> dwvd, freqs, times = dwvd_shm(X, fs=fs, n_win=64)
    >>> print(f"DWVD shape: {dwvd.shape}")
    >>> print(f"Frequency range: {freqs.min():.1f} to {freqs.max():.1f} Hz")
    
    References
    ----------
    [1] B. Boashash and P.J. Black, "An efficient real-time implementation
    of the Wigner-Ville distribution," IEEE Trans. Acoust. Speech Signal
    Process, ASSP-35, 1611-1618 (Nov. 1987).
    
    See Also
    --------
    analytic_signal_shm : Compute analytic signal using Hilbert transform
    """
    from .preprocessing import analytic_signal_shm
    
    # Get matrix dimension sizes
    n_signal, n_channel, n_instance = X.shape
    
    # Compute feature values for (nChannel X nInstance) space
    dwvd_matrix = None
    f = None  
    t = None
    
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            # Compute DWVD
            wvd, f, t = _dwvd_single(
                X[:, i_channel, i_instance], n_win, n_ovlap, n_fft, fs
            )
            
            # If first iteration, allocate memory for DWVD matrix
            if dwvd_matrix is None:
                n_f, n_t = wvd.shape
                dwvd_matrix = np.zeros((n_f, n_t, n_channel, n_instance))
            
            # Store DWVD in the stack
            dwvd_matrix[:, :, i_channel, i_instance] = wvd
    
    return dwvd_matrix, f, t


def _dwvd_single(
    X: np.ndarray, 
    n_win: Optional[int], 
    n_ovlap: Optional[int], 
    n_fft: Optional[int], 
    fs: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a single DWVD from signal.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples,)
        Input signal.
    n_win : int, optional
        Window length.
    n_ovlap : int, optional
        Overlap length.
    n_fft : int, optional
        FFT length.
    fs : float, optional
        Sampling frequency.
        
    Returns
    -------
    wvd : ndarray, shape (n_fft, n_segments)
        Wigner-Ville distribution.
    f : ndarray, shape (n_fft,)
        Frequency vector.
    t : ndarray, shape (n_segments,)
        Time vector.
    """
    # Make sure X is a column vector
    if X.ndim == 2 and X.shape[1] > 1:
        X = X.T
    X = X.flatten()
    
    # Set defaults
    # nWin Default: Window Size is 1/10th the length of the signal
    if n_win is None:
        n_win = len(X) // 10
    
    # Make sure window length is an odd integer
    n_win = int(n_win)
    if n_win % 2 == 0:
        n_win = n_win - 1
    
    # nOvlap Default: Default Overlap Length is half the window length
    if n_ovlap is None:
        n_ovlap = round(0.5 * n_win)
    n_ovlap = abs(int(n_ovlap))
    if n_ovlap >= n_win:
        n_ovlap = round(0.5 * n_win)
    
    # nFFT Default: Is the Next Power of 2 of the Windowing Size
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(n_win)))
    
    # Fs Default: Default Fs is 1Hz (Normalized Frequency)
    if fs is None:
        fs = 1.0
    
    # Compute Analytical Signal - Buffer with Zeros - Window Data
    from .preprocessing import _analytic_single
    Z = _analytic_single(X)
    # Buffer with zeros
    buffer_length = (n_win - 1) // 2
    Z = np.concatenate([
        np.zeros(buffer_length),
        Z,
        np.zeros(buffer_length)
    ])
    
    # Number of Segments so Final Window Does not Exceed Signal Length
    n_segments = (len(Z) - n_ovlap) // (n_win - n_ovlap)
    
    # nSegments Must be even
    if n_segments % 2 == 1:
        n_segments = n_segments - 1
    
    # Window Analytical Signal
    z_mtx = np.zeros((n_win, n_segments), dtype=complex)
    for i in range(n_segments):
        start_idx = i * (n_win - n_ovlap)
        end_idx = start_idx + n_win
        z_mtx[:, i] = Z[start_idx:end_idx]
    
    # Compute Kernel Sequences
    k_mtx = z_mtx * np.conj(z_mtx[::-1, :])
    
    # Modify Kernel Sequences - Make Periodic
    mid_point = (n_win - 1) // 2
    k_mtx_periodic = np.vstack([
        k_mtx[mid_point:, :],
        np.zeros((1, n_segments), dtype=complex),
        k_mtx[:mid_point, :]
    ])
    
    # Allocate Memory for WVD Matrix
    wvd = np.zeros((n_fft, n_segments))
    
    # Compute Wigner Ville Distribution
    for i in range(n_segments // 2):
        k_comb = k_mtx_periodic[:, 2*i] + 1j * k_mtx_periodic[:, 2*i + 1]
        wvd_fft = 2.0 * np.fft.fft(k_comb, n_fft)
        wvd[:, 2*i] = np.real(wvd_fft)
        wvd[:, 2*i + 1] = np.imag(wvd_fft)
    
    # Sample Frequency and Time Vectors for DWVD
    f = np.arange(0, 1, 1/n_fft) * (fs / 2)
    t = np.arange(0, n_segments + 1) * (n_win - n_ovlap) / fs
    
    return wvd, f, t


def lpc_spectrogram_shm(
    X: np.ndarray,
    model_order: Optional[int] = None,
    n_win: Optional[int] = None,
    n_ovlap: Optional[int] = None,
    n_fft: Optional[int] = None,
    fs: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram using Linear Predictive Coding (LPC) coefficients.
    
    .. meta::
        :category: Feature Extraction - Spectral Analysis
        :matlab_equivalent: lpcSpectrogram_shm
        :complexity: Advanced
        :data_type: Time Series
        :output_type: Time-Frequency
        :display_name: Linear Predictive Spectrogram
        :verbose_call: [LPC Spectrogram Matrix, Frequency Vector, Time Vector] = Linear Predictive Spectrogram (Signal Matrix, Linear Predictive Coefficient Model Order, Samples per Window, Number of Overlapping Window Samples, Number of FFT Bins, Sampling Frequency)
        
    Computes a matrix stack of the time frequency domain for a matrix
    of digital signals using a linear predictive coefficient spectrogram
    algorithm. The LPC spectrogram uses autoregressive modeling to estimate
    the spectral envelope, providing better frequency resolution than
    traditional periodogram-based methods.
    
    Parameters
    ----------
    X : ndarray, shape (time, channels, instances)
        Matrix of time series data.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Time series matrix for LPC spectrogram analysis
            
    model_order : int, optional
        Linear predictive model order. Default is 14.
        
        .. gui::
            :widget: number_input
            :min: 2
            :max: 50
            :default: 14
            :description: AR model order for LPC
            
    n_win : int, optional
        Number of samples per window. Default is signal_length/10.
        
        .. gui::
            :widget: number_input
            :min: 32
            :max: 2048
            :description: Window length in samples
            
    n_ovlap : int, optional
        Number of overlapping window samples. Default is 50% of window.
        
        .. gui::
            :widget: number_input
            :min: 0
            :description: Window overlap in samples
            
    n_fft : int, optional
        Number of FFT frequency bins. Default is next power of 2 of n_win.
        
        .. gui::
            :widget: number_input
            :min: 64
            :max: 4096
            :description: FFT size
            
    fs : float, optional
        Sampling frequency in Hz. Default is 1.
        
        .. gui::
            :widget: number_input
            :min: 0.1
            :description: Sampling frequency (Hz)
    
    Returns
    -------
    lpc_spec_matrix : ndarray, shape (n_freq, n_time, channels, instances)
        Linear predictive coefficient spectrogram matrix.
        
    f : ndarray, shape (n_freq,)
        Frequency vector corresponding to lpc_spec_matrix.
        
    t : ndarray, shape (n_time,)
        Time vector corresponding to lpc_spec_matrix.
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.spectral import lpc_spectrogram_shm
    >>> 
    >>> # Generate signal with formant structure (typical for LPC analysis)
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs)
    >>> # Two-formant signal
    >>> formant1 = np.sin(2*np.pi*200*t) * np.exp(-5*t)
    >>> formant2 = np.sin(2*np.pi*400*t) * np.exp(-3*t)
    >>> signal = formant1 + 0.5*formant2 + 0.1*np.random.randn(fs)
    >>> X = signal.reshape(-1, 1, 1)
    >>> 
    >>> # Compute LPC spectrogram
    >>> lpc_spec, freqs, times = lpc_spectrogram_shm(X, fs=fs, model_order=10)
    >>> print(f"LPC spectrogram shape: {lpc_spec.shape}")
    >>> print(f"Frequency range: {freqs.min():.1f} to {freqs.max():.1f} Hz")
    
    References
    ----------
    [1] Rabiner, L. R., & Schafer, R. W. (1978). Digital Processing of
    Speech Signals. Prentice Hall.
    
    See Also
    --------
    ar_model_shm : Autoregressive model estimation
    """
    from ..features import ar_model_shm
    
    # Get matrix dimension sizes
    n_signal, n_channel, n_instance = X.shape
    
    # Compute feature values for (nChannel X nInstance) space
    lpc_spec_matrix = None
    f = None
    t = None
    
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            # Compute LPC Spectrograms
            k_mtx, f, t = _lpc_spec_single(
                X[:, i_channel, i_instance], model_order, n_win, n_ovlap, n_fft, fs
            )
            
            # If first iteration, allocate memory for lpcSpec matrix
            if lpc_spec_matrix is None:
                n_f, n_t = k_mtx.shape
                lpc_spec_matrix = np.zeros((n_f, n_t, n_channel, n_instance))
            
            # Store lpcSpec in the stack
            lpc_spec_matrix[:, :, i_channel, i_instance] = k_mtx
    
    return lpc_spec_matrix, f, t


def _lpc_spec_single(
    X: np.ndarray,
    model_order: Optional[int],
    n_win: Optional[int],
    n_ovlap: Optional[int], 
    n_fft: Optional[int],
    fs: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a single LPC spectrogram.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples,)
        Input signal.
    model_order : int, optional
        LPC model order.
    n_win : int, optional
        Window length.
    n_ovlap : int, optional
        Overlap length.
    n_fft : int, optional
        FFT length.
    fs : float, optional
        Sampling frequency.
        
    Returns
    -------
    k_mtx : ndarray, shape (n_freq, n_segments)
        LPC spectrogram.
    f : ndarray, shape (n_freq,)
        Frequency vector.
    t : ndarray, shape (n_segments,)
        Time vector.
    """
    from ..features import ar_model_shm
    
    # Set defaults
    # modelOrder Default: LPC Model Order is 14
    if model_order is None:
        model_order = 14
    model_order = abs(int(model_order))
    
    # nWin Default: Is 1/10th Signal Length
    if n_win is None:
        n_win = len(X) // 10
    n_win = abs(int(n_win))
    
    # Hamming window
    win = signal.windows.hamming(n_win)
    
    # nOvlap Default: is half the length of the window. OVLAP Must be less than NWIN
    if n_ovlap is None:
        n_ovlap = round(0.5 * n_win)
    n_ovlap = abs(int(n_ovlap))
    if n_ovlap < 0 or n_ovlap > n_win:
        n_ovlap = round(0.5 * n_win)
    
    # NFFT Default: Is the Next Power of 2 of the Windowing Size
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(n_win)))
    
    # Fs Default: Is 1Hz
    if fs is None:
        fs = 1.0
    
    # Model order size check
    rec_max_order = int(n_win * (2/3))
    if not (1 <= model_order < n_win):
        raise ValueError(f'Inappropriate Model Order. Consider using model_order <= {rec_max_order} for window length n_win: {n_win}')
    
    if model_order > rec_max_order:
        print(f'WARNING: (model_order:n_win) > (2:3). Consider using model_order <= {rec_max_order} for window length n_win: {n_win}')
    
    # Number of Segments so Final Window Does not Exceed Signal Length
    n_segments = (len(X) - n_ovlap) // (n_win - n_ovlap)
    
    # Allocate Memory for Construction of Windowed Segment Matrix
    x_win_mtx = np.zeros((n_win, n_segments))
    
    # Build Windowed Signal Matrix
    for i in range(n_segments):
        i_win = np.arange(n_win)
        start_idx = i * (n_win - n_ovlap)
        x_win_mtx[:, i] = X[start_idx + i_win] * win
    
    # Allocate memory for LPC Spectrogram Matrix
    k_mtx = np.zeros((n_fft, n_segments), dtype=complex)
    
    # Calculate AR Model
    for i in range(n_segments):
        a, _ = ar_model_shm(x_win_mtx[:, i], model_order)
        # Compute LPC spectrum: 1 / A(z)
        ar_poly = np.concatenate([[1], -a])
        k_mtx[:, i] = np.fft.fft(ar_poly, n_fft) ** (-1)
        k_mtx[:, i] = np.fft.fftshift(k_mtx[:, i])  # Shift zero frequency to center
    
    # Sample Frequency and Time Vectors for LPC spectrogram
    f = np.linspace(-0.5, 0.5, n_fft, endpoint=False) * fs
    
    # Keep only positive frequencies
    if n_fft % 2 == 1:  # NFFT is ODD
        start_idx = (n_fft + 1) // 2
        f = f[start_idx:]
        k_mtx = k_mtx[start_idx:, :]
    else:  # NFFT is EVEN
        start_idx = (n_fft + 2) // 2 - 1
        f = f[start_idx:]
        k_mtx = k_mtx[start_idx:, :]
    
    t = (n_win / 2 / fs) + np.arange(n_segments) * (n_win - n_ovlap) / fs
    
    return np.abs(k_mtx), f, t
