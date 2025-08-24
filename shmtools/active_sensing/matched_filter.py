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
        :category: Feature Extraction - Active Sensing
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

    # Extract template (MATLAB: matched_waveform is single column template)
    if matched_waveform.shape[1] > 1:
        # Use first column as template
        template = matched_waveform[:, 0]
    else:
        template = matched_waveform.flatten()

    time_points, n_channels = waveform.shape
    template_length = len(template)

    # Initialize output
    filter_result = np.zeros_like(waveform)

    # Normalize template energy
    template_energy = np.sum(template**2)
    if template_energy > 0:
        template_normalized = template / np.sqrt(template_energy)
    else:
        template_normalized = template

    # Process each channel with the same template
    for ch in range(n_channels):
        signal = waveform[:, ch]

        # Compute cross-correlation using scipy correlate
        # 'full' mode gives the full discrete linear cross-correlation
        correlation = correlate(signal, template_normalized, mode="same")
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
        :category: Feature Extraction - Active Sensing
        :matlab_equivalent: incoherentMatchedFilter_shm
        :complexity: Intermediate
        :data_type: Waveforms
        :output_type: Filter Response
        :display_name: Incoherent Matched Filter
        :verbose_call: Filter Result = Incoherent Matched Filter (Waveform, Matched Waveform)

    Parameters
    ----------
    waveform : array_like
        Input waveform to be filtered. Shape: (TIME_POINTS, CHANNELS).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Input waveform data

    matched_waveform : array_like
        Template waveform for matching. Shape: (TIME_TEMPLATE, 1) - single channel template.

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

    Replicates the exact MATLAB incoherentMatchedFilter_shm behavior:
    1. Apply single template to each channel independently
    2. Use sqrt((template'*waveform)^2 + (template90'*waveform)^2) for same lengths
    3. Use convolution with truncation for different lengths

    This approach is particularly useful for detecting scattered waves where
    phase relationships may be disrupted by damage.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import incoherent_matched_filter_shm
    >>>
    >>> # Generate test signals
    >>> waveform = np.random.randn(1000, 10)  # 10 channels
    >>> template = np.sin(np.linspace(0, 2*np.pi, 100))[:, np.newaxis]  # Single template
    >>>
    >>> # Apply incoherent matched filter
    >>> result = incoherent_matched_filter_shm(waveform, template)
    >>> print(f"Filter output shape: {result.shape}")

    References
    ----------
    Ricker, D. W. (2003). Echo signal processing. Springer Science & Business Media.
    """
    waveform = np.asarray(waveform, dtype=np.float64)
    matched_waveform = np.asarray(matched_waveform, dtype=np.float64)

    # Handle empty inputs (MATLAB behavior)
    if waveform.size == 0:
        return np.array([])
    if matched_waveform.size == 0:
        return waveform

    # Ensure proper dimensions
    if waveform.ndim == 1:
        waveform = waveform[:, np.newaxis]
    if matched_waveform.ndim == 1:
        matched_waveform = matched_waveform[:, np.newaxis]

    # Extract the template (should be single column)
    if matched_waveform.shape[1] > 1:
        # Use first column as template
        template = matched_waveform[:, 0]
    else:
        template = matched_waveform.flatten()

    time_points_wave, n_channels = waveform.shape
    template_length = len(template)

    # Generate 90-degree phase-shifted version (Hilbert transform approach)
    template_90 = _shift_90_degrees(template)

    # Check if waveform and template have same length
    if time_points_wave == template_length:
        # Apply filter once per channel (MATLAB: filterResult(:,i)=sqrt((matchedWaveform'*waveform).^2+(mWaveform90'*waveform).^2))
        filter_result = np.zeros((1, n_channels))
        for i in range(n_channels):
            channel_signal = waveform[:, i]
            # Dot products
            dot1 = np.dot(template, channel_signal)
            dot2 = np.dot(template_90, channel_signal)
            filter_result[0, i] = np.sqrt(dot1**2 + dot2**2)
    else:
        # Apply filter through convolution (MATLAB behavior)
        filter_result = np.zeros_like(waveform)
        L = template_length

        for i in range(n_channels):
            channel_signal = waveform[:, i]

            # Convolution with template and 90-degree version
            conv1 = np.convolve(channel_signal, template, mode="full")
            conv2 = np.convolve(channel_signal, template_90, mode="full")

            # Combined magnitude
            conv_combined = np.sqrt(conv1**2 + conv2**2)

            # Truncate ends (MATLAB: filtTemp(round(L/2):round(end-L/2)))
            start_idx = round(L / 2)
            end_idx = len(conv_combined) - round(L / 2)
            filter_result[:, i] = conv_combined[start_idx:end_idx]

    return filter_result


def _shift_90_degrees(waveform: np.ndarray) -> np.ndarray:
    """
    Shift waveform by 90 degrees (quadrature phase).

    Replicates the MATLAB shift90 function behavior using FFT-based approach.
    """
    waveform = waveform.flatten()
    L = len(waveform)

    # Zero-pad to next power of 2 (MATLAB behavior)
    power2L = 2 ** int(np.ceil(np.log2(L)))

    # Create frequency domain multiplier for 90-degree shift
    fh = np.zeros(power2L, dtype=complex)
    fh[: power2L // 2] = 1j  # +j for positive frequencies
    fh[power2L // 2 + 1 :] = -1j  # -j for negative frequencies

    # Zero-pad waveform
    waveform_padded = np.concatenate([waveform, np.zeros(power2L - L)])

    # Apply 90-degree phase shift in frequency domain
    waveform_fft = np.fft.fft(waveform_padded)
    shifted_fft = waveform_fft * fh
    shifted_padded = np.fft.ifft(shifted_fft)

    # Extract original length and take real part
    waveform_90 = np.real(shifted_padded[:L])

    return waveform_90
