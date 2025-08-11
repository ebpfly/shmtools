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
        :category: Feature Extraction - Active Sensing
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
    data_length, n_channels = data.shape
    
    # Handle different start_indices formats
    if start_indices.ndim == 1:
        # 1D indices: extract same indices for all channels
        n_subsets = len(start_indices)
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
    
    elif start_indices.ndim == 2:
        # 2D indices: different indices for each pair-POI combination
        # start_indices shape: (N_PAIRS, N_POIS)
        # data shape: (TIME, N_PAIRS)
        n_pairs, n_pois = start_indices.shape
        
        # Output shape: (N_PAIRS, N_POIS, subset_length)
        subsets = np.zeros((n_pairs, n_pois, subset_length))
        
        for pair_idx in range(n_pairs):
            for poi_idx in range(n_pois):
                start_idx = start_indices[pair_idx, poi_idx]
                end_idx = start_idx + subset_length
                
                if start_idx >= 0 and start_idx < data_length and pair_idx < n_channels:
                    # Extract available data from the specific channel/pair
                    actual_end = min(end_idx, data_length)
                    actual_length = actual_end - start_idx
                    
                    if actual_length > 0:
                        subsets[pair_idx, poi_idx, :actual_length] = data[start_idx:actual_end, pair_idx]
        
        return subsets
    
    else:
        raise ValueError(f"Unsupported start_indices dimensions: {start_indices.ndim}")


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
        :category: Feature Extraction - Active Sensing
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
        elif data.ndim == 3:
            # For 3D data, try to match filter shape to first two dimensions
            if logic_filter.shape == data.shape[:2]:
                # Filter matches first two dimensions (pairs, POIs)
                # Apply element-wise filtering
                filtered_data = data.copy()
                filtered_data[~logic_filter] = 0  # Zero out where filter is False
                return filtered_data
            else:
                raise ValueError(f"Filter shape {logic_filter.shape} doesn't match data shape {data.shape[:2]}")
        else:
            raise ValueError("Must specify dims for >3D data")
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
        :category: Feature Extraction - Active Sensing
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
        :category: Feature Extraction - Active Sensing
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
    
    # Handle different pair_list formats
    if pair_list.shape[0] == 2:  # MATLAB format: [actuator_ids, sensor_ids]
        pair_indices = pair_list.T  # Transpose to (N_PAIRS, 2)
    else:  # Standard format: (N_PAIRS, 2)
        pair_indices = pair_list
    
    n_pairs = pair_indices.shape[0]

    # Handle different sensor_layout formats
    if sensor_layout.shape[0] == 3:  # MATLAB format: [sensorID, xCoord, yCoord]
        sensor_coords = sensor_layout[1:3, :].T  # Extract coordinates and transpose to (N_SENSORS, 2)
        sensor_ids = sensor_layout[0, :]  # Extract sensor IDs
    else:  # Standard format: (N_SENSORS, 2)
        sensor_coords = sensor_layout
        sensor_ids = np.arange(sensor_layout.shape[0])  # Assume sequential IDs

    # Calculate distances between sensor pairs
    distances = []
    valid_pairs = []

    for i in range(n_pairs):
        pair = pair_indices[i]
        # Convert from MATLAB 1-based to Python 0-based indexing
        actuator_id = pair[0]
        sensor_id = pair[1]
        
        # Find indices in the sensor layout
        if sensor_layout.shape[0] == 3:  # Had sensor IDs
            actuator_idx = np.where(sensor_ids == actuator_id)[0]
            sensor_idx = np.where(sensor_ids == sensor_id)[0]
            
            if len(actuator_idx) > 0 and len(sensor_idx) > 0:
                actuator_idx = actuator_idx[0]
                sensor_idx = sensor_idx[0]
            else:
                # Fallback: treat as 1-based indices
                actuator_idx = actuator_id - 1 if actuator_id > 0 else 0
                sensor_idx = sensor_id - 1 if sensor_id > 0 else 0
        else:
            # Direct indexing
            actuator_idx = actuator_id - 1 if actuator_id > 0 else actuator_id
            sensor_idx = sensor_id - 1 if sensor_id > 0 else sensor_id

        # Check bounds
        if (
            actuator_idx >= 0
            and actuator_idx < sensor_coords.shape[0]
            and sensor_idx >= 0
            and sensor_idx < sensor_coords.shape[0]
        ):
            coord1 = sensor_coords[actuator_idx, :]
            coord2 = sensor_coords[sensor_idx, :]
            distance = np.sqrt(np.sum((coord1 - coord2) ** 2))

            distances.append(distance)
            # Map to waveform channel indices (this assumes the waveform channels correspond to pair indices)
            valid_pairs.append((i, i))  # For cross-correlation, use the same pair index for both signals

    distances = np.array(distances)

    if len(valid_pairs) == 0:
        return 0.0, np.array([])

    # Estimate time delays using cross-correlation
    time_delays = []

    for i, (pair_idx1, pair_idx2) in enumerate(valid_pairs):
        # For group velocity estimation, we need to analyze the waveform for each pair
        # The waveform channels correspond to different actuator-sensor pairs
        if pair_idx1 < n_channels:
            # Use the waveform from this pair (each column is a different actuator-sensor pair)
            signal = waveform[:, pair_idx1, 0]
            
            # For group velocity, we typically look at the envelope or peak arrival time
            # Here we'll use a simple approach: find the peak arrival time
            signal_envelope = np.abs(signal)
            peak_idx = np.argmax(signal_envelope)
            
            # Convert peak time to time delay
            time_delay = peak_idx / sample_rate
            
            # Only use delays that make physical sense
            if (
                time_delay > 0 and distances[i] > 0 and time_delay < distances[i] / 100
            ):  # Minimum reasonable velocity: 100 m/s
                time_delays.append(time_delay)
            else:
                time_delays.append(np.nan)
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


def reduce_2_pair_subset_shm(
    sensor_subset: np.ndarray,
    sensor_layout: np.ndarray,
    pair_list: np.ndarray,
    waveform_base: Optional[np.ndarray] = None,
    waveform_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract parameter and data subsets based on sensor subset.

    .. meta::
        :category: Feature Extraction - Active Sensing
        :matlab_equivalent: reduce2PairSubset_shm
        :complexity: Intermediate
        :data_type: Arrays
        :output_type: Data Subsets
        :display_name: Reduce to Pair Subset
        :verbose_call: [Pair Subset, Layout Subset, Baseline Waveform, Test Waveform] = Reduce to Pair Subset (Sensor Subset, Sensor Layout, Pair List, Baseline Waveforms, Test Waveforms)

    Parameters
    ----------
    sensor_subset : array_like
        List of sensor subset IDs (0-based indexing).

        .. gui::
            :widget: array_input
            :description: Sensor subset IDs

    sensor_layout : array_like
        Sensor layout with shape (N_SENSORS, 3) containing [sensorID, xCoord, yCoord].

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Sensor layout coordinates

    pair_list : array_like
        Matrix of actuator-sensor pairs with shape (2, N_PAIRS) containing [actuatorID, sensorID].

        .. gui::
            :widget: array_input
            :description: Sensor pair indices

    waveform_base : array_like, optional
        Baseline waveform data with shape (TIME, N_PAIRS, ...).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Baseline waveform data (optional)

    waveform_test : array_like, optional
        Test waveform data with shape (TIME, N_PAIRS, ...).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Test waveform data (optional)

    Returns
    -------
    pair_subset : ndarray
        Subset of sensor pairs containing only pairs using sensors in subset.

    layout_subset : ndarray
        Subset of sensor layout containing only sensors in subset.

    waveform_base_sub : ndarray or None
        Subset of baseline waveform data corresponding to pair subset.

    waveform_test_sub : ndarray or None
        Subset of test waveform data corresponding to pair subset.

    Notes
    -----
    Extracts parameter and data subsets according to sensor subset. Only
    pairs where both actuator and sensor are in the subset are retained.

    This function expects MATLAB-style data layout where sensor_layout has
    shape (3, N_SENSORS) with rows [sensorID, xCoord, yCoord] and pair_list
    has shape (2, N_PAIRS) with rows [actuatorID, sensorID].

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import reduce_2_pair_subset_shm
    >>>
    >>> # Create sensor layout (MATLAB format: 3 x N_SENSORS)
    >>> sensor_layout = np.array([[0, 1, 2, 3], [0, 1, 2, 0], [0, 0, 1, 1]])
    >>> 
    >>> # Create pair list (MATLAB format: 2 x N_PAIRS) 
    >>> pair_list = np.array([[0, 1, 2], [1, 2, 3]])
    >>>
    >>> # Define subset
    >>> subset = np.array([0, 1, 2])
    >>>
    >>> # Extract subset
    >>> pairs_sub, layout_sub, _, _ = reduce_2_pair_subset_shm(
    ...     subset, sensor_layout, pair_list
    ... )
    >>> print(f"Subset pairs shape: {pairs_sub.shape}")
    """
    sensor_subset = np.asarray(sensor_subset, dtype=int)
    sensor_layout = np.asarray(sensor_layout)
    pair_list = np.asarray(pair_list, dtype=int)

    # Find pairs where both sensors are in the subset
    # pair_list has shape (2, N_PAIRS) with [actuatorID, sensorID] rows
    actuator_in_subset = np.isin(pair_list[0, :], sensor_subset)
    sensor_in_subset = np.isin(pair_list[1, :], sensor_subset)
    
    # Keep pairs where both actuator and sensor are in subset
    sub_pair_mask = actuator_in_subset & sensor_in_subset
    
    # Extract pair subset
    pair_subset = pair_list[:, sub_pair_mask]
    
    # Find sensors that are in the subset
    # sensor_layout has shape (3, N_SENSORS) with [sensorID, xCoord, yCoord] rows
    sensor_ids = sensor_layout[0, :] if sensor_layout.shape[0] > 2 else np.arange(sensor_layout.shape[1])
    sub_sensor_mask = np.isin(sensor_ids, sensor_subset)
    
    # Extract layout subset
    layout_subset = sensor_layout[:, sub_sensor_mask]
    
    # Extract waveform subsets if provided
    waveform_base_sub = None
    if waveform_base is not None:
        waveform_base = np.asarray(waveform_base)
        if waveform_base.size > 0:
            waveform_base_sub = waveform_base[:, sub_pair_mask]
    
    waveform_test_sub = None
    if waveform_test is not None:
        waveform_test = np.asarray(waveform_test)
        if waveform_test.size > 0:
            waveform_test_sub = waveform_test[:, sub_pair_mask]
    
    return pair_subset, layout_subset, waveform_base_sub, waveform_test_sub
