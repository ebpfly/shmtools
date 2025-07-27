"""
Matched filtering functions for active sensing.

This module provides coherent and incoherent matched filtering functions
for guided wave active sensing applications.
"""

import numpy as np
from typing import Union
from scipy.signal import hilbert, correlate


def coherent_matched_filter_shm(
    waveform: np.ndarray, matched_waveform: np.ndarray
) -> np.ndarray:
    """
    Coherent matched filter for guided wave analysis.

    .. meta::
        :category: Active Sensing - Signal Processing
        :matlab_equivalent: coherentMatchedFilter_shm
        :complexity: Intermediate
        :data_type: Waveforms
        :output_type: Filter Response
        :display_name: Coherent Matched Filter
        :verbose_call: Filter Result = Coherent Matched Filter (Waveform, Matched Waveform)

    Parameters
    ----------
    waveform : array_like
        Input waveform to be filtered. Shape: (TIME_POINTS,) or (TIME_POINTS, CHANNELS).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Input waveform data

    matched_waveform : array_like
        Template waveform for matching. Must have same number of channels as waveform.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Template waveform for matching

    Returns
    -------
    filter_result : ndarray
        Coherent matched filter output. Same shape as waveform.

    Notes
    -----
    The coherent matched filter computes the normalized cross-correlation between
    the input waveform and the matched waveform. This preserves phase information
    and is optimal for detecting known signals in noise.

    The filter output is computed as:
    result = conv(waveform, fliplr(matched_waveform)) / sqrt(energy(matched_waveform))

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import coherent_matched_filter_shm
    >>>
    >>> # Generate test signals
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2*np.pi*10*t) + 0.1*np.random.randn(1000)
    >>> template = np.sin(2*np.pi*10*t[:100])
    >>>
    >>> # Apply coherent matched filter
    >>> result = coherent_matched_filter_shm(signal, template)
    >>> print(f"Filter output shape: {result.shape}")

    References
    ----------
    Turin, G. L. (1960). An introduction to matched filters. IRE transactions on
    Information theory, 6(3), 311-329.
    """
    waveform = np.asarray(waveform, dtype=np.float64)
    matched_waveform = np.asarray(matched_waveform, dtype=np.float64)

    # Handle different input dimensions
    if waveform.ndim == 1:
        waveform = waveform[:, np.newaxis]
    if matched_waveform.ndim == 1:
        matched_waveform = matched_waveform[:, np.newaxis]

    # Check dimensions
    if waveform.shape[1] != matched_waveform.shape[1]:
        raise ValueError("Waveform and matched waveform must have same number of channels")

    time_points, n_channels = waveform.shape
    template_length = matched_waveform.shape[0]

    # Initialize output
    filter_result = np.zeros_like(waveform)

    # Process each channel
    for ch in range(n_channels):
        signal = waveform[:, ch]
        template = matched_waveform[:, ch]

        # Normalize template energy
        template_energy = np.sum(template**2)
        if template_energy > 0:
            template_normalized = template / np.sqrt(template_energy)
        else:
            template_normalized = template

        # Compute cross-correlation using scipy correlate
        # 'full' mode gives the full discrete linear cross-correlation
        correlation = correlate(signal, template_normalized, mode='same')
        filter_result[:, ch] = correlation

    # Return original shape if input was 1D
    if filter_result.shape[1] == 1:
        filter_result = filter_result.flatten()

    return filter_result


def incoherent_matched_filter_shm(
    waveform: np.ndarray, matched_waveform: np.ndarray
) -> np.ndarray:
    """
    Incoherent matched filter for guided wave analysis.

    .. meta::
        :category: Active Sensing - Signal Processing
        :matlab_equivalent: incoherentMatchedFilter_shm
        :complexity: Intermediate
        :data_type: Waveforms
        :output_type: Filter Response
        :display_name: Incoherent Matched Filter
        :verbose_call: Filter Result = Incoherent Matched Filter (Waveform, Matched Waveform)

    Parameters
    ----------
    waveform : array_like
        Input waveform to be filtered. Shape: (TIME_POINTS,) or (TIME_POINTS, CHANNELS).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Input waveform data

    matched_waveform : array_like
        Template waveform for matching. Must have same number of channels as waveform.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Template waveform for matching

    Returns
    -------
    filter_result : ndarray
        Incoherent matched filter output (magnitude). Same shape as waveform.

    Notes
    -----
    The incoherent matched filter computes the magnitude of the complex-valued
    matched filter output using the analytic signal (Hilbert transform). This
    removes phase sensitivity and is useful when phase relationships are unknown.

    The filter output is computed as:
    1. Compute coherent matched filter result
    2. Generate analytic signal using Hilbert transform
    3. Return magnitude: |result + j*hilbert(result)|

    This approach is particularly useful for detecting scattered waves where
    phase relationships may be disrupted by damage.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import incoherent_matched_filter_shm
    >>>
    >>> # Generate test signals with phase shifts
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2*np.pi*10*t + np.pi/4) + 0.1*np.random.randn(1000)
    >>> template = np.sin(2*np.pi*10*t[:100])
    >>>
    >>> # Apply incoherent matched filter
    >>> result = incoherent_matched_filter_shm(signal, template)
    >>> print(f"Filter output shape: {result.shape}")

    References
    ----------
    Ricker, D. W. (2003). Echo signal processing. Springer Science & Business Media.
    """
    waveform = np.asarray(waveform, dtype=np.float64)
    matched_waveform = np.asarray(matched_waveform, dtype=np.float64)

    # First compute coherent matched filter
    coherent_result = coherent_matched_filter_shm(waveform, matched_waveform)

    # Handle different input dimensions for Hilbert transform
    if coherent_result.ndim == 1:
        coherent_result = coherent_result[:, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False

    # Initialize output
    incoherent_result = np.zeros_like(coherent_result)

    # Process each channel
    for ch in range(coherent_result.shape[1]):
        # Compute analytic signal using Hilbert transform
        analytic_signal = hilbert(coherent_result[:, ch])
        
        # Take magnitude for incoherent result
        incoherent_result[:, ch] = np.abs(analytic_signal)

    # Return original shape if input was 1D
    if squeeze_output:
        incoherent_result = incoherent_result.flatten()

    return incoherent_result