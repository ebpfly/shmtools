"""
Condition-Based Monitoring (CBM) functions for rotating machinery analysis.

This module provides functions for angular resampling, time synchronous averaging,
and discrete/random separation used in condition-based monitoring of rotating
machinery, particularly for bearing and gear fault detection.
"""

import numpy as np


def time_sync_avg_shm(x_ars_matrix: np.ndarray, samples_per_rev: int) -> np.ndarray:
    """
    Time-synchronous average of angularly sampled signals.

    .. meta::
        :category: Feature Extraction - Condition Based Monitoring
        :matlab_equivalent: timeSyncAvg_shm
        :complexity: Basic
        :data_type: Angular Signals
        :output_type: Averaged Signals
        :display_name: Time Synchronous Average
        :verbose_call: [TSA Matrix] = Time Sync Avg (Angular Matrix, SPR)

    Parameters
    ----------
    x_ars_matrix : array_like
        Matrix of angular resampled signals. Shape: (SAMPLES, CHANNELS, INSTANCES)
        where SAMPLES = samples_per_rev * REVOLUTIONS.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Angular resampled signal matrix

    samples_per_rev : int
        Integer valued samples per revolution.

        .. gui::
            :widget: number_input
            :min: 8
            :max: 8192
            :default: 256
            :description: Samples per revolution

    Returns
    -------
    x_tsa_matrix : ndarray
        Matrix of time synchronous averaged signals.
        Shape: (SAMPLESPERREV, CHANNELS, INSTANCES).

    Notes
    -----
    This function takes a matrix of angular resampled signals and averages
    each cycle of angular rotation to a single synchronous averaged signal
    of one revolution for each signal in the input signal matrix. This
    method is used to remove noise from signals and extract periodic elements.

    The time synchronous average can be used to:
    1. Extract periodic components from noisy signals
    2. Remove periodic components when subtracted from original signal
    3. Enhance gear mesh frequencies and suppress random noise

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.features.condition_based_monitoring import time_sync_avg_shm
    >>>
    >>> # Create synthetic angular resampled data
    >>> samples_per_rev = 256
    >>> n_revolutions = 10
    >>> n_channels = 3
    >>> n_instances = 5
    >>>
    >>> # Generate signal with periodic component
    >>> t_angular = np.linspace(0, n_revolutions * 2 * np.pi,
    ...                        samples_per_rev * n_revolutions)
    >>> periodic_signal = np.sin(t_angular) + 0.5 * np.sin(2 * t_angular)
    >>> noise = 0.1 * np.random.randn(len(t_angular))
    >>>
    >>> # Create signal matrix
    >>> x_ars = np.zeros((len(t_angular), n_channels, n_instances))
    >>> for ch in range(n_channels):
    ...     for inst in range(n_instances):
    ...         x_ars[:, ch, inst] = periodic_signal + noise
    >>>
    >>> # Compute time synchronous average
    >>> x_tsa = time_sync_avg_shm(x_ars, samples_per_rev)
    >>> print(f"TSA shape: {x_tsa.shape}")
    >>> print(f"Original shape: {x_ars.shape}")

    References
    ----------
    .. [1] Randall, Robert. "Vibration-based Condition Monitoring."
           Wiley and Sons, 2011. Ch. 3.3.2 p.97. Ch 3.6.2 p.120
    """
    x_ars_matrix = np.asarray(x_ars_matrix, dtype=np.float64)
    samples_per_rev = int(samples_per_rev)

    # Get matrix dimension sizes
    n_signal, n_channel, n_instance = x_ars_matrix.shape

    # Allocate memory for TSA matrix
    x_tsa_matrix = np.zeros((samples_per_rev, n_channel, n_instance))

    # Get number of cycles
    n_cycles = n_signal // samples_per_rev

    if n_cycles == 0:
        raise ValueError(
            f"Signal length {n_signal} is shorter than "
            f"samples_per_rev {samples_per_rev}"
        )

    # Truncate to min number of cycles
    x_ars_matrix = x_ars_matrix[: samples_per_rev * n_cycles, :, :]

    # Sum and average over all cycles
    for i_cycle in range(n_cycles):
        start_idx = samples_per_rev * i_cycle
        end_idx = samples_per_rev * (i_cycle + 1)
        x_tsa_matrix += x_ars_matrix[start_idx:end_idx, :, :]

    x_tsa_matrix = x_tsa_matrix / n_cycles

    return x_tsa_matrix


# For backwards compatibility and consistency with other modules
timeSyncAvg_shm = time_sync_avg_shm
