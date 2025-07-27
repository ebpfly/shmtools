"""
Core signal processing functions for SHMTools.

This module provides essential signal processing operations including
angular resampling, filtering, and signal conditioning.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import signal, interpolate
import warnings


def ars_tach_shm(
    data: np.ndarray,
    tach: np.ndarray,
    fs: float,
    ppr: int = 1,
    k: float = 1.0,
    angres: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Angular resampling using tachometer signal.

    Python equivalent of MATLAB's ars_tach function. Converts time-based
    vibration signals to angle-based signals using tachometer reference.
    Critical for analyzing rotating machinery under varying speed conditions.

    .. meta::
        :category: Core - Signal Processing
        :matlab_equivalent: ars_tach
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Angle Domain
        :rotating_machinery: True

    Parameters
    ----------
    data : array_like, shape (n_samples,) or (n_samples, n_channels)
        Vibration signal(s) to be resampled.

        .. gui::
            :widget: data_input
            :description: "Vibration signal for angular resampling"

    tach : array_like, shape (n_samples,)
        Tachometer signal (pulse per revolution or analog).

        .. gui::
            :widget: data_input
            :description: "Tachometer reference signal"

    fs : float
        Sampling frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 100000.0
            :description: "Sampling frequency (Hz)"

    ppr : int, optional, default=1
        Pulses per revolution in tachometer signal.

        .. gui::
            :widget: numeric_input
            :min: 1
            :max: 1000
            :description: "Pulses per revolution"

    k : float, optional, default=1.0
        Gear ratio or speed multiplier.

        .. gui::
            :widget: numeric_input
            :min: 0.1
            :max: 100.0
            :description: "Gear ratio"

    angres : int, optional, default=1024
        Angular resolution (samples per revolution).

        .. gui::
            :widget: numeric_input
            :min: 64
            :max: 4096
            :description: "Angular resolution"

    Returns
    -------
    data_ars : ndarray, shape (angres * n_revs, n_channels)
        Angular resampled vibration data.

        .. gui::
            :plot_type: "line"
            :x_axis: "angle"
            :y_axis: "amplitude"
            :xlabel: "Angle (degrees)"
            :ylabel: "Amplitude"

    angle : ndarray, shape (angres * n_revs,)
        Angle vector in degrees.

        .. gui::
            :plot_type: "reference"
            :description: "Angle reference vector"

    Raises
    ------
    ValueError
        If tachometer signal is invalid or contains insufficient pulses.

    Examples
    --------
    Basic angular resampling:

    >>> import numpy as np
    >>> from shmtools.core import ars_tach
    >>> from shmtools.utils import import_cbm_data
    >>>
    >>> # Load CBM data
    >>> dataset, _, _, fs = import_cbm_data()
    >>> vibration = dataset[:, 1, 0]  # Accelerometer channel
    >>> tachometer = dataset[:, 0, 0]  # Tachometer channel
    >>>
    >>> # Perform angular resampling
    >>> vibration_ars, angle = ars_tach(vibration, tachometer, fs, ppr=1)
    >>> print(f"Original length: {len(vibration)}")
    >>> print(f"Resampled length: {len(vibration_ars)}")
    >>> print(f"Angular resolution: {len(vibration_ars) / (angle[-1] / 360):.1f} samples/rev")

    Multi-channel resampling with gear ratio:

    >>> # Multi-channel data
    >>> vibration_multi = dataset[:, 1:3, 0]  # Multiple accelerometer channels
    >>> vibration_ars, angle = ars_tach(vibration_multi, tachometer, fs,
    ...                                 ppr=1, k=3.71, angres=2048)
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_samples, n_channels = data.shape

    # Validate inputs
    if len(tach) != n_samples:
        raise ValueError("Data and tachometer signals must have same length")

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    if ppr <= 0:
        raise ValueError("Pulses per revolution must be positive")

    # Time vector
    t = np.arange(n_samples) / fs

    # Find tachometer pulses using peak detection
    tach_normalized = (tach - np.mean(tach)) / np.std(tach)

    # Detect pulses (positive peaks above threshold)
    pulse_threshold = 2.0  # 2 standard deviations
    pulse_indices, _ = signal.find_peaks(
        tach_normalized,
        height=pulse_threshold,
        distance=int(fs / (200 * ppr)),  # Minimum distance between pulses (max 200 Hz)
    )

    if len(pulse_indices) < 2:
        raise ValueError(
            f"Insufficient tachometer pulses detected: {len(pulse_indices)}"
        )

    # Convert pulse indices to time
    pulse_times = pulse_indices / fs

    # Calculate instantaneous speed between pulses
    dt_pulses = np.diff(pulse_times)
    instantaneous_period = dt_pulses * ppr  # Time per revolution

    # Handle edge cases and smooth speed variations
    if len(instantaneous_period) > 2:
        # Remove outliers (more than 2x median period)
        median_period = np.median(instantaneous_period)
        valid_mask = np.abs(instantaneous_period - median_period) < 2 * median_period

        if np.sum(valid_mask) < len(instantaneous_period) * 0.5:
            warnings.warn(
                "High number of outlier periods detected in tachometer signal"
            )

        # Smooth the instantaneous period
        if len(instantaneous_period) > 5:
            window_length = min(5, len(instantaneous_period) // 2 * 2 + 1)  # Odd number
            instantaneous_period = signal.savgol_filter(
                instantaneous_period, window_length, 2, mode="nearest"
            )

    # Create cumulative angle vector
    # Start from first pulse
    start_idx = pulse_indices[0]
    end_idx = pulse_indices[-1]

    # Time vector from first to last pulse
    t_segment = t[start_idx : end_idx + 1]
    data_segment = data[start_idx : end_idx + 1, :]

    # Interpolate instantaneous period over entire time segment
    pulse_times_segment = pulse_times[:-1]  # Remove last pulse (no period after it)
    period_interp = interpolate.interp1d(
        pulse_times_segment,
        instantaneous_period,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(t_segment)

    # Calculate cumulative angle
    dt = 1.0 / fs
    instantaneous_freq = 1.0 / (
        period_interp + 1e-12
    )  # Add small value to avoid division by zero
    cumulative_angle = np.cumsum(
        instantaneous_freq * dt * 360.0 * k
    )  # Convert to degrees
    cumulative_angle = cumulative_angle - cumulative_angle[0]  # Start from 0

    # Determine number of complete revolutions
    total_angle = cumulative_angle[-1]
    n_revs = int(total_angle / 360.0)

    if n_revs < 1:
        raise ValueError("Insufficient data for one complete revolution")

    # Create uniform angle grid
    angle_uniform = np.linspace(0, n_revs * 360.0, n_revs * angres, endpoint=False)

    # Interpolate data to uniform angle grid
    data_ars = np.zeros((len(angle_uniform), n_channels))

    for ch in range(n_channels):
        # Remove any duplicate angles for interpolation
        unique_mask = np.diff(cumulative_angle, prepend=-1) > 0
        angle_unique = cumulative_angle[unique_mask]
        data_unique = data_segment[unique_mask, ch]

        if len(angle_unique) < 2:
            raise ValueError("Insufficient unique angle points for interpolation")

        # Clip angle_uniform to available range
        angle_min, angle_max = angle_unique[0], angle_unique[-1]
        valid_mask = (angle_uniform >= angle_min) & (angle_uniform <= angle_max)

        if np.sum(valid_mask) < len(angle_uniform) * 0.5:
            warnings.warn("Limited angle range available for resampling")

        # Interpolate
        interp_func = interpolate.interp1d(
            angle_unique,
            data_unique,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,  # Zero-pad outside range
        )

        data_ars[:, ch] = interp_func(angle_uniform)

    # Squeeze if single channel
    if n_channels == 1:
        data_ars = data_ars.squeeze()

    return data_ars, angle_uniform


def fir1_shm(
    n: int,
    Wn: Union[float, Tuple[float, float]],
    ftype: str = "low",
    window: str = "hamming",
    fs: float = 2.0,
) -> np.ndarray:
    """
    Design FIR filter using window method.

    Python equivalent of MATLAB's fir1 function. Designs finite impulse
    response filters using windowing method.

    .. meta::
        :category: Core - Filtering
        :matlab_equivalent: fir1
        :complexity: Basic
        :data_type: Filter Coefficients
        :output_type: Filter

    Parameters
    ----------
    n : int
        Filter order (number of taps - 1).

        .. gui::
            :widget: numeric_input
            :min: 1
            :max: 1000
            :description: "Filter order"

    Wn : float or tuple of floats
        Critical frequency/frequencies. For lowpass and highpass filters,
        Wn is a scalar. For bandpass and bandstop filters, Wn is a 2-element sequence.

        .. gui::
            :widget: frequency_input
            :description: "Critical frequency (Hz)"

    ftype : {'low', 'high', 'band', 'stop'}, optional
        Filter type: 'low' for lowpass, 'high' for highpass,
        'band' for bandpass, 'stop' for bandstop.

        .. gui::
            :widget: select
            :options: ["low", "high", "band", "stop"]
            :description: "Filter type"

    window : str or tuple, optional, default='hamming'
        Window function to use.

        .. gui::
            :widget: select
            :options: ["hamming", "hann", "blackman", "kaiser"]
            :description: "Window function"

    fs : float, optional, default=2.0
        Sampling frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 100000.0
            :description: "Sampling frequency (Hz)"

    Returns
    -------
    b : ndarray
        Filter coefficients.

        .. gui::
            :plot_type: "line"
            :x_axis: "sample"
            :y_axis: "coefficient"
            :xlabel: "Sample"
            :ylabel: "Coefficient"

    Examples
    --------
    Design lowpass filter:

    >>> from shmtools.core import fir1
    >>> import numpy as np
    >>>
    >>> # 50-tap lowpass filter with cutoff at 100 Hz (fs=1000 Hz)
    >>> b = fir1(50, 100, ftype='low', fs=1000)
    >>> print(f"Filter length: {len(b)}")

    Design bandpass filter:

    >>> # Bandpass filter 50-200 Hz
    >>> b = fir1(100, (50, 200), ftype='band', fs=1000)
    """
    # Normalize frequency to Nyquist frequency
    nyquist = fs / 2.0

    if isinstance(Wn, (list, tuple)):
        Wn_norm = [w / nyquist for w in Wn]
    else:
        Wn_norm = Wn / nyquist

    # Validate frequency range
    if isinstance(Wn_norm, (list, tuple)):
        if any(w <= 0 or w >= 1 for w in Wn_norm):
            raise ValueError(
                "Critical frequencies must be between 0 and Nyquist frequency"
            )
    else:
        if Wn_norm <= 0 or Wn_norm >= 1:
            raise ValueError(
                "Critical frequency must be between 0 and Nyquist frequency"
            )

    # Map filter types to scipy equivalents
    filter_types = {
        "low": "lowpass",
        "high": "highpass",
        "band": "bandpass",
        "stop": "bandstop",
    }

    if ftype not in filter_types:
        raise ValueError(
            f"Invalid filter type. Must be one of: {list(filter_types.keys())}"
        )

    # Design filter using scipy
    b = signal.firwin(
        n + 1,  # scipy uses number of taps, MATLAB uses order
        Wn_norm,
        pass_zero=filter_types[ftype],
        window=window,
    )

    return b
