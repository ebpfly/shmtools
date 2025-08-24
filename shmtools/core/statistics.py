"""
Statistical analysis functions for signal processing.

This module provides statistical measures and damage indicators
commonly used in structural health monitoring.
"""

import numpy as np
from typing import Union, Dict, Any, Optional, List, Tuple


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

    .. meta::
        :category: Feature Extraction - Condition Based Monitoring
        :matlab_equivalent: fm0_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        :display_name: FM0 Feature
        :verbose_call: [FM0 Feature Matrix] = FM0 Feature (Conditioned Raw Signal Ensemble, Fundamental Mesh Frequency, Harmonic Orders, FFT Bins, FFT Bin Search Width)

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
    from .spectral import psd_welch_shm

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
            psd, f_psd, _ = psd_welch_shm(
                signal.reshape(-1, 1, 1), None, None, nfft, 1.0, None
            )

            # Extract 1D arrays from 3D output
            psd_1d = psd[:, 0, 0]  # Extract (freq, channel=0, instance=0)

            # Sum amplitudes at tracked orders
            sum_amp = _sum_amplitudes(
                psd_1d, f_psd, fund_mesh_freq, track_orders, n_bin_search
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

    .. meta::
        :category: Feature Extraction - Condition Based Monitoring
        :matlab_equivalent: fm4_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        :display_name: FM4 Feature
        :verbose_call: [FM4 Feature Matrix] = FM4 Feature (Difference Signal Matrix)

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
        :category: Feature Extraction - Statistics
        :matlab_equivalent: peakFactor_shm
        :complexity: Basic
        :sensitivity: Peak, Outlier
        :verbose_call: [Peak Factor] = Peak Factor Feature (Signal)

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
        :category: Feature Extraction - Statistics
        :matlab_equivalent: impulseFactor_shm
        :complexity: Basic
        :sensitivity: Impulse, Outlier
        :verbose_call: [Impulse Factor] = Impulse Factor Feature (Signal)

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
        :category: Feature Extraction - Statistics
        :matlab_equivalent: clearanceFactor_shm
        :complexity: Basic
        :sensitivity: Impulse, Outlier
        :verbose_call: [Clearance Factor] = Clearance Factor Feature (Signal)

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
        :category: Feature Extraction - Statistics
        :matlab_equivalent: shapeFactor_shm
        :complexity: Basic
        :sensitivity: Distribution Shape
        :verbose_call: [Shape Factor] = Shape Factor Feature (Signal)

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


def crest_factor_shm(X: np.ndarray) -> np.ndarray:
    """
    Calculate crest factor feature matrix from raw signal matrix.

    .. meta::
        :category: Feature Extraction - Statistics
        :matlab_equivalent: crestFactor_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Features
        :display_name: Crest Factor Feature
        :verbose_call: [Crest Factor Feature Matrix] = Crest Factor Feature (Conditioned Raw Signal Matrix)

    The crest factor operates on the conditioned raw signal. It is a measure of
    the peak amplitude of the signal divided by its root mean square. The
    crest factor has been found to be more sensitive to damage incurred in
    early stages of gear and bearing failure.

    Parameters
    ----------
    X : ndarray, shape (samples, channels, instances)
        Input signal matrix.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Raw vibration signal matrix

    Returns
    -------
    cf : ndarray, shape (instances, channels)
        Feature vectors of crest factor feature in concatenated format.
        FEATURES = CHANNELS

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.statistics import crest_factor_shm
    >>>
    >>> # Generate synthetic signal with impulses (high crest factor)
    >>> t = np.linspace(0, 1, 1000)
    >>> x = np.sin(2*np.pi*10*t) + 0.1*np.random.randn(1000)
    >>> # Add impulses to simulate damage
    >>> x[::100] += 5.0  # Add impulses every 100 samples
    >>> X = x.reshape(-1, 1, 1)  # (samples, channels, instances)
    >>>
    >>> cf = crest_factor_shm(X)
    >>> print(f"Crest factor: {cf[0,0]:.3f}")

    References
    ----------
    [1] Lebold, M.; McClintic, K.; Campbell, R.; Byington, C.; Maynard, K.,
    Review of Vibration Analysis Methods for Gearbox Diagnostics and
    Prognostics, Proceedings of the 54th Meeting of the Society for
    Machinery Failure Prevention Technology, Virginia Beach, VA, May 1-4,
    2000, p. 623-634.
    """
    # Get matrix dimension sizes
    n_signal, n_channel, n_instance = X.shape

    # Allocate memory for feature space
    cf = np.zeros((n_instance, n_channel))

    # Compute feature values for (nChannel X nInstance) space
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            signal = X[:, i_channel, i_instance]
            # Crest factor = peak / RMS
            peak = np.max(np.abs(signal))  # Use absolute value for peak
            rms_val = np.sqrt(np.mean(signal**2))
            # Handle zero RMS case
            if rms_val > 0:
                cf[i_instance, i_channel] = peak / rms_val
            else:
                cf[i_instance, i_channel] = 0.0

    return cf


def stat_moments_shm(X: np.ndarray) -> np.ndarray:
    """
    Calculate first four statistical moments as damage sensitive features.

    .. meta::
        :category: Feature Extraction - Statistics
        :matlab_equivalent: statMoments_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Features
        :display_name: Statistical Moments
        :verbose_call: [Statistics Feature Vectors] = Statistical Moments (Time Series Data)

    Returns the first four statistical moments as damage sensitive features:
    (1) mean, (2) standard deviation, (3) skewness, (4) kurtosis

    Parameters
    ----------
    X : ndarray, shape (time, channels, instances)
        Time series matrix.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Time series data for statistical analysis

    Returns
    -------
    statistics_fv : ndarray, shape (instances, features)
        First four statistical moments of each channel grouped by moments.
        FEATURES = 4*CHANNELS (mean, std, skew, kurt for each channel)

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.statistics import stat_moments_shm
    >>>
    >>> # Generate synthetic signals with different statistical properties
    >>> t = np.linspace(0, 1, 1000)
    >>> # Normal signal
    >>> x1 = np.random.randn(1000)
    >>> # Signal with outliers (high kurtosis)
    >>> x2 = np.random.randn(1000)
    >>> x2[::100] += 10.0  # Add outliers
    >>> X = np.stack([x1, x2], axis=1).reshape(1000, 2, 1)
    >>>
    >>> stats = stat_moments_shm(X)
    >>> print("Channel 1 - Mean: {:.3f}, Std: {:.3f}, Skew: {:.3f}, Kurt: {:.3f}".format(*stats[0, :4]))
    >>> print("Channel 2 - Mean: {:.3f}, Std: {:.3f}, Skew: {:.3f}, Kurt: {:.3f}".format(*stats[0, 4:]))

    References
    ----------
    Figueiredo, E., Park, G., Figueiras, J., Farrar, C., and Worden, K.
    (2009). Structural Health Monitoring Algorithm Comparisons Using
    Standard Data Sets, Los Alamos National Laboratory Report: LA-14393.
    """
    # Get dimensions: [time, channels, instances]
    n, m, t = X.shape

    # Allocate output: [instances, features] where features = 4*channels
    statistics_fv = np.zeros((t, m * 4))

    # Compute the first four statistical moments for each instance
    for i in range(t):
        # Extract data for instance i: [time, channels]
        data = X[:, :, i]

        # Compute mean
        mean_vals = np.sum(data, axis=0) / n

        # Center the data
        centered = data - mean_vals[np.newaxis, :]

        # Compute standard deviation (with bias correction)
        std_vals = np.sqrt(np.sum(centered**2, axis=0) / (n - 1))

        # Compute skewness
        skew_vals = (np.sum(centered**3, axis=0) / (n - 1)) / (std_vals**3)

        # Compute kurtosis
        kurt_vals = (np.sum(centered**4, axis=0) / (n - 1)) / (std_vals**4)

        # Handle division by zero for constant signals
        skew_vals = np.where(std_vals == 0, 0.0, skew_vals)
        kurt_vals = np.where(std_vals == 0, 0.0, kurt_vals)

        # Concatenate all moments: [mean, std, skew, kurt] for all channels
        statistics_fv[i, :] = np.concatenate(
            [mean_vals, std_vals, skew_vals, kurt_vals]
        )

    return statistics_fv


def rms_shm(X: np.ndarray) -> np.ndarray:
    """
    Calculate root mean square (RMS) feature matrix from signal matrix.

    .. meta::
        :category: Feature Extraction - Statistics
        :matlab_equivalent: rms_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Features
        :display_name: Root Mean Square Feature
        :verbose_call: [RMS Feature Matrix] = Root Mean Square Feature (Signal Matrix)

    Parameters
    ----------
    X : ndarray, shape (samples, channels, instances)
        Input signal matrix.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Signal matrix for RMS calculation

    Returns
    -------
    rms_vals : ndarray, shape (instances, channels)
        RMS feature vectors in concatenated format.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.statistics import rms_shm
    >>>
    >>> # Generate synthetic signal
    >>> t = np.linspace(0, 1, 1000)
    >>> x = 2.0 * np.sin(2*np.pi*10*t)  # Amplitude 2, RMS should be ~1.414
    >>> X = x.reshape(-1, 1, 1)
    >>>
    >>> rms_val = rms_shm(X)
    >>> print(f"RMS value: {rms_val[0,0]:.3f}")
    """
    # Get matrix dimension sizes
    n_signal, n_channel, n_instance = X.shape

    # Allocate memory for feature space
    rms_vals = np.zeros((n_instance, n_channel))

    # Compute RMS values for each channel and instance
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            signal = X[:, i_channel, i_instance]
            # RMS = sqrt(mean(x^2))
            rms_vals[i_instance, i_channel] = np.sqrt(np.mean(signal**2))

    return rms_vals


def m6a_shm(D: np.ndarray) -> np.ndarray:
    """
    Calculate M6A feature from difference signal matrix.

    .. meta::
        :category: Feature Extraction - Statistics
        :matlab_equivalent: m6a_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        :display_name: M6A Feature
        :verbose_call: [M6A Feature Matrix] = M6A Feature (Difference Signal Matrix)

    The M6A feature is defined as the sixth statistical moment about the mean
    of a difference signal normalized by its variance to the 3rd power. M6A is
    similar to the FM4 feature but is reported to be more sensitive to damage.

    Parameters
    ----------
    D : ndarray, shape (samples, channels, instances)
        Difference signal matrix.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Difference signal matrix for M6A feature extraction

    Returns
    -------
    m6a : ndarray, shape (instances, channels)
        Feature vectors of M6A feature in concatenated format.
        FEATURES = CHANNELS

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.statistics import m6a_shm
    >>>
    >>> # Generate synthetic difference signal with outliers
    >>> np.random.seed(42)
    >>> t = np.linspace(0, 1, 1000)
    >>> # Difference signal with high moments due to outliers
    >>> signal = 0.1*np.random.randn(1000)
    >>> signal[::100] += 2.0  # Add outliers
    >>> D = signal.reshape(-1, 1, 1)
    >>>
    >>> m6a_val = m6a_shm(D)
    >>> print(f"M6A value: {m6a_val[0,0]:.3f}")

    References
    ----------
    [1] Lebold, M.; McClintic, K.; Campbell, R.; Byington, C.; Maynard, K.,
    Review of Vibration Analysis Methods for Gearbox Diagnostics and
    Prognostics, Proceedings of the 54th Meeting of the Society for
    Machinery Failure Prevention Technology, Virginia Beach, VA, May 1-4,
    2000, p. 623-634.
    """
    # Get dimension sizes
    n_signal, n_channel, n_instance = D.shape

    # Allocate memory for feature space
    m6a = np.zeros((n_instance, n_channel))

    # Compute feature values for (nChannel x nInstance) space
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            signal = D[:, i_channel, i_instance]

            # Calculate mean
            mean_val = np.mean(signal)

            # Center the signal
            centered = signal - mean_val

            # Calculate sixth moment and second moment
            sixth_moment = np.sum(centered**6)
            second_moment = np.sum(centered**2)

            # M6A formula: (n^2 * sum((x-mu)^6)) / (sum((x-mu)^2)^3)
            if second_moment > 0:
                m6a[i_instance, i_channel] = (n_signal**2 * sixth_moment) / (
                    second_moment**3
                )
            else:
                m6a[i_instance, i_channel] = 0.0

    return m6a


def m8a_shm(D: np.ndarray) -> np.ndarray:
    """
    Calculate M8A feature from difference signal matrix.

    .. meta::
        :category: Feature Extraction - Statistics
        :matlab_equivalent: m8a_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        :display_name: M8A Feature
        :verbose_call: [M8A Feature Matrix] = M8A Feature (Difference Signal Space)

    The M8A feature is defined as the eighth statistical moment about the mean
    of a difference signal normalized by its variance to the 4th power. M8A is
    similar to the FM4 feature but is reported to be more sensitive to damage.

    Parameters
    ----------
    D : ndarray, shape (samples, channels, instances)
        Difference signal matrix.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Difference signal matrix for M8A feature extraction

    Returns
    -------
    m8a : ndarray, shape (instances, channels)
        Feature vectors of M8A feature in concatenated format.
        FEATURES = CHANNELS

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.statistics import m8a_shm
    >>>
    >>> # Generate synthetic difference signal with extreme outliers
    >>> np.random.seed(42)
    >>> t = np.linspace(0, 1, 1000)
    >>> # Difference signal with very high moments due to extreme outliers
    >>> signal = 0.1*np.random.randn(1000)
    >>> signal[::200] += 5.0  # Add extreme outliers
    >>> D = signal.reshape(-1, 1, 1)
    >>>
    >>> m8a_val = m8a_shm(D)
    >>> print(f"M8A value: {m8a_val[0,0]:.3f}")

    References
    ----------
    [1] Lebold, M.; McClintic, K.; Campbell, R.; Byington, C.; Maynard, K.,
    Review of Vibration Analysis Methods for Gearbox Diagnostics and
    Prognostics, Proceedings of the 54th Meeting of the Society for
    Machinery Failure Prevention Technology, Virginia Beach, VA, May 1-4,
    2000, p. 623-634.
    """
    # Get dimension sizes
    n_signal, n_channel, n_instance = D.shape

    # Allocate memory for feature space
    m8a = np.zeros((n_instance, n_channel))

    # Compute feature values for (nChannel x nInstance) space
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            signal = D[:, i_channel, i_instance]

            # Calculate mean
            mean_val = np.mean(signal)

            # Center the signal
            centered = signal - mean_val

            # Calculate eighth moment and second moment
            eighth_moment = np.sum(centered**8)
            second_moment = np.sum(centered**2)

            # M8A formula: (n^3 * sum((x-mu)^8)) / (sum((x-mu)^2)^4)
            if second_moment > 0:
                m8a[i_instance, i_channel] = (n_signal**3 * eighth_moment) / (
                    second_moment**4
                )
            else:
                m8a[i_instance, i_channel] = 0.0

    return m8a


def compute_damage_features_shm(
    x: np.ndarray, features: List[str] = None, axis: int = 0
) -> Dict[str, np.ndarray]:
    """
    Compute multiple damage-sensitive features from vibration signals.

    .. meta::
        :category: Feature Extraction - Statistics
        :matlab_equivalent: Multiple feature functions
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        :display_name: Compute Damage Features
        :verbose_call: [Feature Dictionary] = Compute Damage Features (Signal Matrix, Feature List, Axis)

    Parameters
    ----------
    x : ndarray, shape (..., samples, ...)
        Input signal matrix.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Input vibration signals for feature extraction

    features : list of str, optional
        List of features to compute. Available features:
        ['peak_factor', 'impulse_factor', 'clearance_factor', 'shape_factor', 'fm0', 'fm4', 'crest_factor', 'rms', 'stat_moments', 'm6a', 'm8a']
        If None, computes all features.

        .. gui::
            :widget: multiselect
            :options: ["peak_factor", "impulse_factor", "clearance_factor", "shape_factor", "fm0", "fm4", "crest_factor", "rms", "stat_moments", "m6a", "m8a"]
            :default: ["peak_factor", "impulse_factor", "clearance_factor", "shape_factor", "crest_factor", "rms", "m6a", "m8a"]
            :description: Select damage features to compute

    axis : int, optional
        Axis along which to compute features (default: 0).

        .. gui::
            :widget: number_input
            :min: 0
            :max: 2
            :default: 0
            :description: Axis for feature computation (0=time, 1=channels, 2=instances)

    Returns
    -------
    features_dict : dict
        Dictionary containing computed features with feature names as keys
        and feature arrays as values.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.statistics import compute_damage_features_shm
    >>>
    >>> # Generate synthetic vibration signal
    >>> t = np.linspace(0, 1, 1000)
    >>> x = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(1000)
    >>> x = x.reshape(-1, 1, 1)  # (samples, channels, instances)
    >>>
    >>> # Compute damage features
    >>> features = compute_damage_features_shm(x, features=['crest_factor', 'rms', 'm6a', 'm8a'])
    >>> print(f"Crest factor: {features['crest_factor'][0,0]:.3f}")
    >>> print(f"RMS: {features['rms'][0,0]:.3f}")
    >>> print(f"M6A: {features['m6a'][0,0]:.3f}")
    >>> print(f"M8A: {features['m8a'][0,0]:.3f}")
    """
    if features is None:
        features = [
            "peak_factor",
            "impulse_factor",
            "clearance_factor",
            "shape_factor",
            "fm0",
            "fm4",
            "crest_factor",
            "rms",
            "stat_moments",
            "m6a",
            "m8a",
        ]

    feature_functions = {
        "peak_factor": peak_factor_shm,
        "impulse_factor": impulse_factor_shm,
        "clearance_factor": clearance_factor_shm,
        "shape_factor": shape_factor_shm,
        "crest_factor": crest_factor_shm,
        "rms": rms_shm,
        "stat_moments": stat_moments_shm,
        "m6a": m6a_shm,
        "m8a": m8a_shm,
        "fm0": lambda x, axis=axis: fm0_shm(
            x, fund_mesh_freq=1.0, track_orders=[1], n_fft=None, n_bin_search=3
        ),
        "fm4": fm4_shm,
    }

    results = {}
    for feature_name in features:
        if feature_name in feature_functions:
            try:
                results[feature_name] = feature_functions[feature_name](x, axis=axis)
            except TypeError:
                # Handle functions that don't take axis parameter
                results[feature_name] = feature_functions[feature_name](x)
        else:
            raise ValueError(f"Unknown feature: {feature_name}")

    return results


def na4m_shm(
    R: np.ndarray, m2: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate NA4M damage feature from residual signal matrix.

    .. meta::
        :category: Feature Extraction - Statistics
        :matlab_equivalent: na4m_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        :display_name: NA4M Feature
        :verbose_call: [NA4M Feature Matrix, Average Baseline Signal Variance] = NA4M Feature (Residual Signal Matrix, Average Baseline Signal Variance)

    Computes the NA4M damage feature for a matrix of residual signals. The
    NA4M damage feature is the fourth order statistical moment normalized
    by an average baseline signal variance to the second power. The
    residual signal is the raw signal with the gear mesh and shaft
    frequencies removed from the signal.

    Parameters
    ----------
    R : ndarray, shape (samples, channels, instances)
        Residual signal matrix.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Residual signal matrix for NA4M feature extraction

    m2 : ndarray, shape (1, channels), optional
        Average baseline signal variance. If None, computed from all instances.

        .. gui::
            :widget: array_input
            :description: Average baseline signal variance (leave empty to compute from data)

    Returns
    -------
    na4m : ndarray, shape (instances, channels)
        Feature vectors of NA4M feature in concatenated format.
        FEATURES = CHANNELS

    m2 : ndarray, shape (1, channels)
        Average baseline signal variance.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.statistics import na4m_shm
    >>>
    >>> # Generate synthetic residual signal with damage
    >>> np.random.seed(42)
    >>> t = np.linspace(0, 1, 1000)
    >>> # Clean residual signal (baseline)
    >>> baseline = 0.1*np.random.randn(1000, 2, 10)  # 2 channels, 10 instances
    >>> # Damaged residual with higher moments
    >>> damaged = baseline.copy()
    >>> damaged[::50, :, :] += 1.0  # Add periodic impacts
    >>> R = np.concatenate([baseline, damaged], axis=2)
    >>>
    >>> na4m_val, m2_val = na4m_shm(R)
    >>> print(f"Baseline NA4M (first 10): {np.mean(na4m_val[:10, 0]):.3f}")
    >>> print(f"Damaged NA4M (last 10): {np.mean(na4m_val[10:, 0]):.3f}")

    References
    ----------
    [1] Lebold, M.; McClintic, K.; Campbell, R.; Byington, C.; Maynard, K.,
    Review of Vibration Analysis Methods for Gearbox Diagnostics and
    Prognostics, Proceedings of the 54th Meeting of the Society for
    Machinery Failure Prevention Technology, Virginia Beach, VA, May 1-4,
    2000, p. 623-634.

    [2] Decker, H.J., Handschuh, R.F., Zakrajsek, J.J. An Enhancement to the
    NA4 Gear Vibration Diagnostic Parameter. U.S. Army Research Laboratory/
    NASA Glenn Research Center. 18th Annual Meeting of the Vibration
    Institute, NASA TM 106553, ARL-TR-389, June 1994.
    """
    # Get dimension sizes
    n_signal, n_channel, n_instance = R.shape

    # Allocate memory for feature space
    na4m = np.zeros((n_instance, n_channel))

    # If not supplied, compute M2 for all channels
    if m2 is None:
        # Compute mean variance across all instances and samples
        # MATLAB: m2 = ((1/nInstance)*sum(((1/nSignal).*sum((R-repmat(mean(R,1),[nSignal,1,1])).^2,1)),3));
        mean_R = np.mean(R, axis=0, keepdims=True)  # Shape: (1, n_channel, n_instance)
        centered_R = R - mean_R  # Shape: (n_signal, n_channel, n_instance)
        variance_per_instance = (
            np.sum(centered_R**2, axis=0) / n_signal
        )  # Shape: (n_channel, n_instance)
        m2 = (
            np.sum(variance_per_instance, axis=1, keepdims=True) / n_instance
        )  # Shape: (n_channel, 1)
        m2 = m2.T  # Shape: (1, n_channel) to match MATLAB

    # Compute feature values for (nChannel x nInstance) space
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            signal = R[:, i_channel, i_instance]
            mean_val = np.mean(signal)
            centered = signal - mean_val

            # Fourth moment about mean
            fourth_moment = np.sum(centered**4)

            # NA4M formula: (1/n * sum((x-mu)^4)) / (m2^2)
            na4m[i_instance, i_channel] = (fourth_moment / n_signal) / (
                m2[0, i_channel] ** 2
            )

    return na4m, m2


def nb4m_shm(
    X: np.ndarray, m2: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate NB4M feature from band passed mesh signal matrix.

    .. meta::
        :category: Feature Extraction - Statistics
        :matlab_equivalent: nb4m_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        :display_name: NB4M Feature
        :verbose_call: [NB4M Feature Vectors] = NB4M Feature (Band Passed Mesh Signal, Average Envelope Variance)

    Computes the NB4M feature matrix for a 3D matrix of band pass mesh
    signals. Traditionally NB4 is calculated continuously over the course
    of a machine's life and M2 is a runtime average of the envelope signal
    variance. For the purposes of this module the numerator of the feature
    (m2) is to be an averaged envelope signal variance from a baseline
    state, similar to the NA4* damage feature, as the data supplied is a
    binary data set with only a good condition and bad condition and not a
    run to failure data set. NB4 was developed to detect damage on gear
    teeth. The NB4M feature provided in this function is the 4th order
    statistical moment of the band passed mesh signal normalized by an
    average variance of a baseline gearbox signal squared.

    Parameters
    ----------
    X : ndarray, shape (samples, channels, instances)
        Band pass mesh signal matrix.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Band passed mesh signal matrix for NB4M feature extraction

    m2 : ndarray, shape (1, channels), optional
        Average envelope signal variance from a baseline gearbox state.
        If None, computed for each channel from all instances.

        .. gui::
            :widget: array_input
            :description: Average envelope signal variance (leave empty to compute from data)

    Returns
    -------
    nb4m : ndarray, shape (instances, channels)
        Feature vectors of NB4M feature in concatenated format.
        FEATURES = CHANNELS

    m2 : ndarray, shape (1, channels)
        Average envelope signal variance.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.statistics import nb4m_shm
    >>>
    >>> # Generate synthetic band-passed mesh signal with gear damage
    >>> np.random.seed(42)
    >>> t = np.linspace(0, 1, 1000)
    >>> # Simulate gear mesh vibration with modulation
    >>> gear_freq = 50  # Hz
    >>> modulation_freq = 2  # Hz (damage frequency)
    >>> baseline_signal = np.sin(2*np.pi*gear_freq*t) * (1 + 0.1*np.random.randn(1000))
    >>> damaged_signal = np.sin(2*np.pi*gear_freq*t) * (1 + 0.5*np.sin(2*np.pi*modulation_freq*t))
    >>> X_baseline = baseline_signal.reshape(-1, 1, 1)
    >>> X_damaged = damaged_signal.reshape(-1, 1, 1)
    >>> X = np.concatenate([X_baseline, X_damaged], axis=2)
    >>>
    >>> nb4m_val, m2_val = nb4m_shm(X)
    >>> print(f"Baseline NB4M: {nb4m_val[0, 0]:.3f}")
    >>> print(f"Damaged NB4M: {nb4m_val[1, 0]:.3f}")

    References
    ----------
    [1] Lebold, M.; McClintic, K.; Campbell, R.; Byington, C.; Maynard, K.,
    Review of Vibration Analysis Methods for Gearbox Diagnostics and
    Prognostics, Proceedings of the 54th Meeting of the Society for
    Machinery Failure Prevention Technology, Virginia Beach, VA, May 1-4,
    2000, p. 623-634.

    See Also
    --------
    envelope_shm : Compute envelope signal using Hilbert transform
    analytic_signal_shm : Compute analytic signal using Hilbert transform
    """
    from .preprocessing import envelope_shm

    # Get dimension size
    n_signal, n_channel, n_instance = X.shape

    # Allocate memory for feature space
    nb4m = np.zeros((n_instance, n_channel))

    # Get enveloped signal
    E = envelope_shm(X)

    # If not supplied, compute M2 for all channels from current data
    if m2 is None:
        # MATLAB: m2 = ((1/nInstance)*sum(((1/nSignal).*sum((E-repmat(mean(E,1),[nSignal,1,1])).^2,1)),3));
        mean_E = np.mean(E, axis=0, keepdims=True)  # Shape: (1, n_channel, n_instance)
        centered_E = E - mean_E  # Shape: (n_signal, n_channel, n_instance)
        variance_per_instance = (
            np.sum(centered_E**2, axis=0) / n_signal
        )  # Shape: (n_channel, n_instance)
        m2 = (
            np.sum(variance_per_instance, axis=1, keepdims=True) / n_instance
        )  # Shape: (n_channel, 1)
        m2 = m2.T  # Shape: (1, n_channel) to match MATLAB

    # Compute feature values for (nChannel x nInstance) space
    for i_channel in range(n_channel):
        for i_instance in range(n_instance):
            envelope_signal = E[:, i_channel, i_instance]
            mean_val = np.mean(envelope_signal)
            centered = envelope_signal - mean_val

            # Fourth moment of envelope signal
            fourth_moment = np.sum(centered**4)

            # NB4M formula: (1/n * sum((E-mu)^4)) / (m2^2)
            nb4m[i_instance, i_channel] = (fourth_moment / n_signal) / (
                m2[0, i_channel] ** 2
            )

    return nb4m, m2
