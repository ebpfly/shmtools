"""
Utility functions for active sensing signal processing.

This module provides helper functions for data manipulation and processing
in active sensing applications.
"""

import numpy as np
from typing import Union, Optional, List, Tuple


def extract_subsets_shm(
    data: np.ndarray, start_indices: np.ndarray, subset_length: int
) -> np.ndarray:
    """
    Extract data subsets from array using start indices and fixed length.

    .. meta::
        :category: Active Sensing - Utilities
        :matlab_equivalent: extractSubsets_shm
        :complexity: Basic
        :data_type: Arrays
        :output_type: Data Subsets
        :display_name: Extract Subsets
        :verbose_call: Data Subsets = Extract Subsets (Data, Start Indices, Subset Window)

    Parameters
    ----------
    data : array_like
        Input data array. Can be 1D, 2D, or 3D.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Input data array

    start_indices : array_like
        Starting indices for each subset extraction.

        .. gui::
            :widget: array_input
            :description: Start indices for extraction

    subset_length : int
        Length of each subset to extract.

        .. gui::
            :widget: number_input
            :min: 1
            :max: 10000
            :default: 100
            :description: Subset length

    Returns
    -------
    subsets : ndarray
        Extracted data subsets. Shape depends on input dimensionality.

    Notes
    -----
    Extracts fixed-length subsets from data starting at specified indices.
    Useful for extracting time windows around time-of-flight locations.

    If start_index + subset_length exceeds data bounds, the subset is
    zero-padded or truncated as appropriate.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import extract_subsets_shm
    >>>
    >>> # Create test data
    >>> data = np.arange(100)
    >>> start_indices = np.array([10, 20, 30])
    >>>
    >>> # Extract subsets
    >>> subsets = extract_subsets_shm(data, start_indices, 5)
    >>> print(f"Subsets shape: {subsets.shape}")
    """
    data = np.asarray(data)
    start_indices = np.asarray(start_indices, dtype=int)

    # Handle different data dimensions
    if data.ndim == 1:
        return _extract_subsets_1d(data, start_indices, subset_length)
    elif data.ndim == 2:
        return _extract_subsets_2d(data, start_indices, subset_length)
    elif data.ndim == 3:
        return _extract_subsets_3d(data, start_indices, subset_length)
    else:
        raise ValueError("Data must be 1D, 2D, or 3D array")


def _extract_subsets_1d(
    data: np.ndarray, start_indices: np.ndarray, subset_length: int
) -> np.ndarray:
    """Extract subsets from 1D data."""
    n_subsets = len(start_indices)
    data_length = len(data)

    # Initialize output
    subsets = np.zeros((n_subsets, subset_length))

    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + subset_length

        if start_idx >= 0 and start_idx < data_length:
            # Extract available data
            actual_end = min(end_idx, data_length)
            actual_length = actual_end - start_idx

            if actual_length > 0:
                subsets[i, :actual_length] = data[start_idx:actual_end]

    return subsets


def _extract_subsets_2d(
    data: np.ndarray, start_indices: np.ndarray, subset_length: int
) -> np.ndarray:
    """Extract subsets from 2D data (TIME, CHANNELS)."""
    n_subsets = len(start_indices)
    data_length, n_channels = data.shape

    # Initialize output
    subsets = np.zeros((n_subsets, subset_length, n_channels))

    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + subset_length

        if start_idx >= 0 and start_idx < data_length:
            # Extract available data
            actual_end = min(end_idx, data_length)
            actual_length = actual_end - start_idx

            if actual_length > 0:
                subsets[i, :actual_length, :] = data[start_idx:actual_end, :]

    return subsets


def _extract_subsets_3d(
    data: np.ndarray, start_indices: np.ndarray, subset_length: int
) -> np.ndarray:
    """Extract subsets from 3D data (TIME, CHANNELS, INSTANCES)."""
    n_subsets = len(start_indices)
    data_length, n_channels, n_instances = data.shape

    # Initialize output
    subsets = np.zeros((n_subsets, subset_length, n_channels, n_instances))

    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + subset_length

        if start_idx >= 0 and start_idx < data_length:
            # Extract available data
            actual_end = min(end_idx, data_length)
            actual_length = actual_end - start_idx

            if actual_length > 0:
                subsets[i, :actual_length, :, :] = data[start_idx:actual_end, :, :]

    return subsets


def flex_logic_filter_shm(
    data: np.ndarray, logic_filter: np.ndarray, dims: Optional[List[int]] = None
) -> np.ndarray:
    """
    Apply flexible logical filtering to multi-dimensional data.

    .. meta::
        :category: Active Sensing - Utilities
        :matlab_equivalent: flexLogicFilter_shm
        :complexity: Intermediate
        :data_type: Arrays
        :output_type: Filtered Data
        :display_name: Flexible Logic Filter
        :verbose_call: Filtered Data = Flexible Logic Filter (Data, Logic Filter, Dimensions)

    Parameters
    ----------
    data : array_like
        Input data array to be filtered.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Input data array

    logic_filter : array_like
        Boolean filter array.

        .. gui::
            :widget: array_input
            :description: Boolean filter mask

    dims : list of int, optional
        Dimensions along which to apply filter. If None, applies to all.

        .. gui::
            :widget: array_input
            :description: Dimensions to filter (optional)

    Returns
    -------
    filtered_data : ndarray
        Data with logic filter applied.

    Notes
    -----
    Applies boolean filtering along specified dimensions, useful for
    removing invalid data points or applying geometric constraints.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import flex_logic_filter_shm
    >>>
    >>> # Create test data and filter
    >>> data = np.random.randn(10, 5)
    >>> filter_mask = np.array([True, False, True, False, True])
    >>>
    >>> # Apply filter
    >>> filtered = flex_logic_filter_shm(data, filter_mask, dims=[1])
    >>> print(f"Filtered shape: {filtered.shape}")
    """
    data = np.asarray(data)
    logic_filter = np.asarray(logic_filter, dtype=bool)

    if dims is None:
        # Apply filter to appropriate dimension based on shapes
        if data.ndim == 1:
            return data[logic_filter]
        elif data.ndim == 2:
            if len(logic_filter) == data.shape[0]:
                return data[logic_filter, :]
            elif len(logic_filter) == data.shape[1]:
                return data[:, logic_filter]
            else:
                raise ValueError("Filter length doesn't match any data dimension")
        else:
            raise ValueError("Must specify dims for >2D data")
    else:
        # Apply filter along specified dimensions
        filtered_data = data.copy()
        for dim in dims:
            filtered_data = np.take(filtered_data, np.where(logic_filter)[0], axis=dim)
        return filtered_data


def sum_mult_dims_shm(data: np.ndarray, dimensions: List[int]) -> np.ndarray:
    """
    Sum array along multiple dimensions.

    .. meta::
        :category: Active Sensing - Utilities
        :matlab_equivalent: sumMultDims_shm
        :complexity: Basic
        :data_type: Arrays
        :output_type: Summed Data
        :display_name: Sum Multiple Dimensions
        :verbose_call: [Data Sum] = Sum Multiple Dimensions (Data, Dimensions)

    Parameters
    ----------
    data : array_like
        Input data array.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Input data array

    dimensions : list of int
        Dimensions along which to sum.

        .. gui::
            :widget: array_input
            :description: Dimensions to sum

    Returns
    -------
    data_sum : ndarray
        Array with specified dimensions summed.

    Notes
    -----
    Sums data along multiple dimensions simultaneously. Useful for
    combining contributions from multiple sensor pairs or time windows.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import sum_mult_dims_shm
    >>>
    >>> # Create 3D test data
    >>> data = np.random.randn(5, 10, 3)
    >>>
    >>> # Sum along dimensions 0 and 2
    >>> result = sum_mult_dims_shm(data, [0, 2])
    >>> print(f"Result shape: {result.shape}")
    """
    data = np.asarray(data)

    # Sort dimensions in descending order to avoid index shifting
    dimensions = sorted(set(dimensions), reverse=True)

    result = data.copy()
    for dim in dimensions:
        result = np.sum(result, axis=dim)

    return result


def estimate_group_velocity_shm(
    waveform: np.ndarray,
    pair_list: np.ndarray,
    sensor_layout: np.ndarray,
    sample_rate: float,
    actuation_width: float,
    line_of_sight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Estimate group velocity from guided wave measurements.

    .. meta::
        :category: Active Sensing - Signal Processing
        :matlab_equivalent: estimateGroupVelocity_shm
        :complexity: Advanced
        :data_type: Waveforms
        :output_type: Velocity
        :display_name: Estimate Wavespeed
        :verbose_call: [Estimated Speed, Speed List] = Estimate Wavespeed (Waveform, Pair List, Sensor Layout, Sampling Rate, Actuation Width, Line of Sight)

    Parameters
    ----------
    waveform : array_like
        Guided wave measurement data. Shape: (TIME_POINTS, CHANNELS, INSTANCES).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Waveform measurement data

    pair_list : array_like
        Sensor pair indices of shape (N_PAIRS, 2).

        .. gui::
            :widget: array_input
            :description: Sensor pair indices

    sensor_layout : array_like
        Sensor coordinates of shape (N_SENSORS, 2).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Sensor layout coordinates

    sample_rate : float
        Sampling rate in Hz.

        .. gui::
            :widget: number_input
            :min: 1000
            :max: 10000000
            :default: 1000000
            :description: Sampling rate (Hz)

    actuation_width : float
        Width of actuation pulse in seconds.

        .. gui::
            :widget: number_input
            :min: 0.000001
            :max: 0.001
            :default: 0.00001
            :description: Actuation pulse width (seconds)

    line_of_sight : array_like, optional
        Line-of-sight matrix of shape (N_PAIRS, N_PAIRS). If None,
        assumes all pairs have line-of-sight.

        .. gui::
            :widget: array_input
            :description: Line-of-sight matrix (optional)

    Returns
    -------
    estimated_speed : float
        Estimated group velocity in m/s.

    speed_list : ndarray
        Individual velocity estimates for each valid pair.

    Notes
    -----
    Estimates guided wave group velocity by analyzing time-of-flight between
    sensor pairs. Uses cross-correlation to find peak arrival times and
    calculates velocity from known distances.

    The algorithm:
    1. Calculate distances between all sensor pairs
    2. Cross-correlate waveforms to find time delays
    3. Compute velocity = distance / time_delay
    4. Filter outliers and return robust estimate

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import estimate_group_velocity_shm
    >>>
    >>> # Generate synthetic data
    >>> waveform = np.random.randn(1000, 4, 1)  # 4 sensors
    >>> pairs = np.array([[1, 2], [1, 3], [2, 4]])  # MATLAB indexing
    >>> layout = np.array([[0, 0], [0.1, 0], [0.2, 0], [0.3, 0]])
    >>>
    >>> # Estimate velocity
    >>> velocity, speeds = estimate_group_velocity_shm(
    ...     waveform, pairs, layout, 1e6, 1e-5
    ... )
    >>> print(f"Estimated velocity: {velocity:.0f} m/s")
    """
    waveform = np.asarray(waveform, dtype=np.float64)
    pair_list = np.asarray(pair_list, dtype=int)
    sensor_layout = np.asarray(sensor_layout, dtype=np.float64)

    if waveform.ndim == 2:
        waveform = waveform[:, :, np.newaxis]

    time_points, n_channels, n_instances = waveform.shape
    n_pairs = pair_list.shape[0]

    # Calculate distances between sensor pairs
    distances = []
    valid_pairs = []

    for i, pair in enumerate(pair_list):
        # Convert from MATLAB 1-based to Python 0-based indexing
        sensor1_idx = pair[0] - 1
        sensor2_idx = pair[1] - 1

        if sensor1_idx >= 0 and sensor1_idx < n_channels and \
           sensor2_idx >= 0 and sensor2_idx < n_channels:
            
            coord1 = sensor_layout[sensor1_idx, :]
            coord2 = sensor_layout[sensor2_idx, :]
            distance = np.sqrt(np.sum((coord1 - coord2) ** 2))
            
            distances.append(distance)
            valid_pairs.append((sensor1_idx, sensor2_idx))

    distances = np.array(distances)
    
    if len(valid_pairs) == 0:
        return 0.0, np.array([])

    # Estimate time delays using cross-correlation
    time_delays = []
    
    for i, (sensor1_idx, sensor2_idx) in enumerate(valid_pairs):
        # Use first instance for velocity estimation
        signal1 = waveform[:, sensor1_idx, 0]
        signal2 = waveform[:, sensor2_idx, 0]
        
        # Cross-correlation to find time delay
        correlation = np.correlate(signal1, signal2, mode='full')
        lag_max = np.argmax(np.abs(correlation))
        
        # Convert lag to time delay
        max_lag = len(signal1) - 1
        actual_lag = lag_max - max_lag
        time_delay = actual_lag / sample_rate
        
        # Only use positive delays that make physical sense
        if time_delay > 0 and time_delay < distances[i] / 100:  # Minimum reasonable velocity: 100 m/s
            time_delays.append(time_delay)
        else:
            time_delays.append(np.nan)

    time_delays = np.array(time_delays)
    
    # Calculate velocities
    valid_mask = ~np.isnan(time_delays) & (time_delays > 0)
    
    if not np.any(valid_mask):
        return 0.0, np.array([])
        
    valid_distances = distances[valid_mask]
    valid_delays = time_delays[valid_mask]
    
    speed_list = valid_distances / valid_delays
    
    # Remove outliers (simple approach: remove speeds outside 1-3 std from median)
    median_speed = np.median(speed_list)
    std_speed = np.std(speed_list)
    
    outlier_mask = np.abs(speed_list - median_speed) < 3 * std_speed
    if np.any(outlier_mask):
        filtered_speeds = speed_list[outlier_mask]
        estimated_speed = np.mean(filtered_speeds)
    else:
        estimated_speed = median_speed
    
    return estimated_speed, speed_list