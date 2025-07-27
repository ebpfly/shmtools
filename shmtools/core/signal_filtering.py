"""
Signal filtering and conditioning functions for SHMTools.

This module provides specialized filtering functions for extracting
residual, difference, and bandpass filtered signals commonly used
in gearbox condition monitoring.
"""

import numpy as np
from typing import Tuple, Union, Optional
from scipy import signal
import warnings


def residual_signal_shm(
    x: np.ndarray,
    fs: float,
    shaft_freq: float,
    harmonics: int = 3,
    bandwidth: float = 0.1,
) -> np.ndarray:
    """
    Extract residual signal by removing shaft harmonics.

    Python equivalent of MATLAB's residualSignal_shm function. Removes
    shaft harmonics and their sidebands to highlight fault-related content.

    .. meta::
        :category: Core - Signal Filtering
        :matlab_equivalent: residualSignal_shm
        :complexity: Intermediate
        :sensitivity: Fault Detection

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for residual analysis"

    fs : float
        Sampling frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 100000.0
            :description: "Sampling frequency (Hz)"

    shaft_freq : float
        Shaft rotational frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 0.1
            :max: 1000.0
            :description: "Shaft frequency (Hz)"

    harmonics : int, optional, default=3
        Number of shaft harmonics to remove.

        .. gui::
            :widget: numeric_input
            :min: 1
            :max: 10
            :description: "Number of harmonics"

    bandwidth : float, optional, default=0.1
        Relative bandwidth for notch filters (fraction of center frequency).

        .. gui::
            :widget: numeric_input
            :min: 0.01
            :max: 0.5
            :description: "Filter bandwidth (fraction)"

    Returns
    -------
    x_residual : np.ndarray
        Residual signal with shaft harmonics removed.

        .. gui::
            :plot_type: "line"
            :xlabel: "Time (s)"
            :ylabel: "Amplitude"

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core import residual_signal
    >>>
    >>> # Generate signal with shaft harmonics + fault
    >>> fs = 1000
    >>> t = np.linspace(0, 2, 2*fs, endpoint=False)
    >>> shaft_freq = 30  # 30 Hz shaft
    >>> x = (np.sin(2*np.pi*shaft_freq*t) +
    ...      0.5*np.sin(2*np.pi*2*shaft_freq*t) +
    ...      0.2*np.random.randn(len(t)))  # Add noise/fault content
    >>>
    >>> # Extract residual signal
    >>> x_res = residual_signal(x, fs, shaft_freq, harmonics=2)
    """
    nyquist = fs / 2
    x_residual = x.copy()

    # Apply notch filters for each harmonic
    for h in range(1, harmonics + 1):
        center_freq = h * shaft_freq

        if center_freq >= nyquist:
            warnings.warn(
                f"Harmonic {h} ({center_freq:.1f} Hz) exceeds Nyquist frequency"
            )
            continue

        # Calculate bandwidth in Hz
        bw_hz = bandwidth * center_freq
        low_freq = max(center_freq - bw_hz / 2, 1.0)
        high_freq = min(center_freq + bw_hz / 2, nyquist - 1)

        # Design notch filter (bandstop)
        try:
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist

            # Butterworth bandstop filter
            b, a = signal.butter(2, [low_norm, high_norm], btype="bandstop")
            x_residual = signal.filtfilt(b, a, x_residual)

        except Exception as e:
            warnings.warn(f"Failed to design notch filter for harmonic {h}: {e}")
            continue

    return x_residual


def difference_signal_shm(
    x_baseline: np.ndarray, x_test: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """
    Compute difference signal between baseline and test conditions.

    Python equivalent of MATLAB's differenceSignal_shm function. Highlights
    changes between baseline and current conditions.

    .. meta::
        :category: Core - Signal Filtering
        :matlab_equivalent: differenceSignal_shm
        :complexity: Basic
        :sensitivity: Condition Change

    Parameters
    ----------
    x_baseline : np.ndarray
        Baseline (healthy) signal.

        .. gui::
            :widget: data_input
            :description: "Baseline signal"

    x_test : np.ndarray
        Test (current) signal.

        .. gui::
            :widget: data_input
            :description: "Test signal"

    normalize : bool, optional, default=True
        Whether to normalize signals before subtraction.

        .. gui::
            :widget: checkbox
            :description: "Normalize signals"

    Returns
    -------
    x_diff : np.ndarray
        Difference signal highlighting changes.

        .. gui::
            :plot_type: "line"
            :xlabel: "Time (s)"
            :ylabel: "Difference"

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core import difference_signal
    >>>
    >>> # Generate baseline and damaged signals
    >>> t = np.linspace(0, 1, 1000)
    >>> x_baseline = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(1000)
    >>> x_damaged = x_baseline + 0.2*np.sin(2*np.pi*200*t)  # Add fault frequency
    >>>
    >>> # Compute difference
    >>> x_diff = difference_signal(x_baseline, x_damaged)
    """
    if len(x_baseline) != len(x_test):
        raise ValueError("Baseline and test signals must have same length")

    if normalize:
        # Normalize by RMS
        rms_baseline = np.sqrt(np.mean(x_baseline**2))
        rms_test = np.sqrt(np.mean(x_test**2))

        x_baseline_norm = x_baseline / (rms_baseline + 1e-12)
        x_test_norm = x_test / (rms_test + 1e-12)

        x_diff = x_test_norm - x_baseline_norm
    else:
        x_diff = x_test - x_baseline

    return x_diff


def bandpass_condition_signal_shm(
    x: np.ndarray, fs: float, freq_range: Tuple[float, float], filter_order: int = 4
) -> np.ndarray:
    """
    Extract bandpass filtered signal around fault frequencies.

    Python equivalent of MATLAB's bandpassCondition_shm function. Isolates
    specific frequency bands related to fault conditions.

    .. meta::
        :category: Core - Signal Filtering
        :matlab_equivalent: bandpassCondition_shm
        :complexity: Basic
        :sensitivity: Frequency-specific Faults

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for bandpass filtering"

    fs : float
        Sampling frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 100000.0
            :description: "Sampling frequency (Hz)"

    freq_range : tuple of floats
        (low_freq, high_freq) bandpass frequency range in Hz.

        .. gui::
            :widget: frequency_range
            :description: "Frequency range (Hz)"

    filter_order : int, optional, default=4
        Filter order for Butterworth filter.

        .. gui::
            :widget: numeric_input
            :min: 1
            :max: 10
            :description: "Filter order"

    Returns
    -------
    x_filtered : np.ndarray
        Bandpass filtered signal.

        .. gui::
            :plot_type: "line"
            :xlabel: "Time (s)"
            :ylabel: "Amplitude"

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core import bandpass_condition_signal
    >>>
    >>> # Generate signal with multiple frequency components
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> x = (np.sin(2*np.pi*50*t) +
    ...      np.sin(2*np.pi*150*t) +
    ...      np.sin(2*np.pi*300*t) +
    ...      0.1*np.random.randn(fs))
    >>>
    >>> # Extract bearing fault frequency band (100-200 Hz)
    >>> x_bearing = bandpass_condition_signal(x, fs, (100, 200))
    """
    low_freq, high_freq = freq_range
    nyquist = fs / 2

    # Validate frequency range
    if low_freq <= 0:
        raise ValueError("Low frequency must be positive")
    if high_freq >= nyquist:
        raise ValueError(f"High frequency must be less than Nyquist ({nyquist} Hz)")
    if low_freq >= high_freq:
        raise ValueError("Low frequency must be less than high frequency")

    # Normalize frequencies
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist

    # Design Butterworth bandpass filter
    b, a = signal.butter(filter_order, [low_norm, high_norm], btype="band")

    # Apply zero-phase filtering
    x_filtered = signal.filtfilt(b, a, x)

    return x_filtered


def gear_mesh_filter_shm(
    x: np.ndarray,
    fs: float,
    gear_mesh_freq: float,
    sideband_range: float = 50.0,
    harmonics: int = 3,
) -> np.ndarray:
    """
    Extract gear mesh frequency components and sidebands.

    Python equivalent of MATLAB's gearMeshFilter_shm function. Focuses on
    gear mesh frequencies and their modulation sidebands for fault detection.

    .. meta::
        :category: Core - Signal Filtering
        :matlab_equivalent: gearMeshFilter_shm
        :complexity: Intermediate
        :sensitivity: Gear Faults

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for gear mesh analysis"

    fs : float
        Sampling frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 100000.0
            :description: "Sampling frequency (Hz)"

    gear_mesh_freq : float
        Gear mesh frequency in Hz.

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 10000.0
            :description: "Gear mesh frequency (Hz)"

    sideband_range : float, optional, default=50.0
        Frequency range around mesh frequency to include (Hz).

        .. gui::
            :widget: numeric_input
            :min: 1.0
            :max: 500.0
            :description: "Sideband range (Hz)"

    harmonics : int, optional, default=3
        Number of mesh harmonics to include.

        .. gui::
            :widget: numeric_input
            :min: 1
            :max: 5
            :description: "Number of harmonics"

    Returns
    -------
    x_mesh : np.ndarray
        Gear mesh filtered signal.

        .. gui::
            :plot_type: "line"
            :xlabel: "Time (s)"
            :ylabel: "Amplitude"

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core import gear_mesh_filter
    >>>
    >>> # Generate gearbox signal
    >>> fs = 10000
    >>> t = np.linspace(0, 2, 2*fs, endpoint=False)
    >>> gear_mesh = 1200  # Hz
    >>> x = (0.5*np.sin(2*np.pi*gear_mesh*t) +
    ...      0.2*np.sin(2*np.pi*(gear_mesh+30)*t) +  # Sideband
    ...      0.2*np.sin(2*np.pi*(gear_mesh-30)*t) +  # Sideband
    ...      0.1*np.random.randn(len(t)))
    >>>
    >>> # Extract gear mesh components
    >>> x_mesh = gear_mesh_filter(x, fs, gear_mesh, sideband_range=100)
    """
    nyquist = fs / 2
    x_mesh = np.zeros_like(x)

    # Extract each gear mesh harmonic with sidebands
    for h in range(1, harmonics + 1):
        center_freq = h * gear_mesh_freq

        if center_freq + sideband_range / 2 >= nyquist:
            warnings.warn(f"Gear mesh harmonic {h} exceeds Nyquist frequency")
            continue

        # Define frequency range including sidebands
        low_freq = max(center_freq - sideband_range / 2, 1.0)
        high_freq = min(center_freq + sideband_range / 2, nyquist - 1)

        try:
            # Extract this harmonic component
            x_harmonic = bandpass_condition_signal(
                x, fs, (low_freq, high_freq), filter_order=6
            )
            x_mesh += x_harmonic

        except Exception as e:
            warnings.warn(f"Failed to extract gear mesh harmonic {h}: {e}")
            continue

    return x_mesh


def envelope_signal_shm(
    x: np.ndarray, method: str = "hilbert", smooth_length: Optional[int] = None
) -> np.ndarray:
    """
    Compute signal envelope for amplitude modulation analysis.

    Python equivalent of MATLAB's envelopeSignal_shm function. Extracts
    amplitude envelope for detecting amplitude modulation patterns.

    .. meta::
        :category: Core - Signal Filtering
        :matlab_equivalent: envelopeSignal_shm
        :complexity: Basic
        :sensitivity: Amplitude Modulation

    Parameters
    ----------
    x : np.ndarray
        Input signal.

        .. gui::
            :widget: data_input
            :description: "Signal for envelope analysis"

    method : str, optional, default='hilbert'
        Envelope extraction method ('hilbert', 'rectify').

        .. gui::
            :widget: select
            :options: ["hilbert", "rectify"]
            :description: "Envelope method"

    smooth_length : int, optional
        Smoothing window length for rectification method.

        .. gui::
            :widget: numeric_input
            :min: 1
            :max: 1000
            :allow_none: True
            :description: "Smoothing length"

    Returns
    -------
    envelope : np.ndarray
        Signal envelope.

        .. gui::
            :plot_type: "line"
            :xlabel: "Time (s)"
            :ylabel: "Envelope"

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core import envelope_signal
    >>>
    >>> # Generate amplitude modulated signal
    >>> fs = 1000
    >>> t = np.linspace(0, 2, 2*fs, endpoint=False)
    >>> carrier = np.sin(2*np.pi*100*t)
    >>> modulation = 0.5 * (1 + np.sin(2*np.pi*5*t))
    >>> x = carrier * modulation
    >>>
    >>> # Extract envelope
    >>> env = envelope_signal(x, method='hilbert')
    """
    if method.lower() == "hilbert":
        from scipy.signal import hilbert

        analytic_signal = hilbert(x)
        envelope = np.abs(analytic_signal)

    elif method.lower() == "rectify":
        # Full-wave rectification
        envelope = np.abs(x)

        # Apply smoothing if requested
        if smooth_length is not None and smooth_length > 1:
            # Moving average smoothing
            window = np.ones(smooth_length) / smooth_length
            envelope = signal.convolve(envelope, window, mode="same")

    else:
        raise ValueError(f"Unknown envelope method: {method}")

    return envelope
