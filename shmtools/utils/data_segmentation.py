"""
Data segmentation utilities for structural health monitoring.

This module provides functions for segmenting time series data to increase
sample size and improve statistical analysis.
"""

import numpy as np
from typing import Tuple, Optional


def segment_time_series(
    data: np.ndarray,
    segment_length: int,
    overlap: int = 0,
    preserve_states: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Segment long time series into multiple shorter segments.

    This function is useful for increasing the number of training/testing
    instances by breaking long time series into shorter segments.

    .. meta::
        :category: Utilities - Data Processing
        :complexity: Basic
        :data_type: Time Series
        :output_type: Time Series
        :display_name: Segment Time Series

    Parameters
    ----------
    data : array_like
        Time series data to segment.
        Shape: (TIME, CHANNELS, INSTANCES)

    segment_length : int
        Length of each segment in time steps.

        .. gui::
            :widget: number_input
            :min: 100
            :max: 10000
            :default: 2048

    overlap : int, optional
        Number of overlapping time steps between segments. Default is 0.

        .. gui::
            :widget: number_input
            :min: 0
            :max: 1000
            :default: 0

    preserve_states : array_like, optional
        State labels for each instance. If provided, states are replicated
        for each segment from the same original instance.
        Shape: (INSTANCES,) or (1, INSTANCES)

    Returns
    -------
    segmented_data : ndarray
        Segmented time series data.
        Shape: (segment_length, CHANNELS, INSTANCES * n_segments)

    segmented_states : ndarray or None
        Replicated state labels for segmented data if preserve_states provided.
        Shape: (INSTANCES * n_segments,) or (1, INSTANCES * n_segments)

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.utils import segment_time_series
    >>>
    >>> # Create sample data: 8192 time points, 4 channels, 10 instances
    >>> data = np.random.randn(8192, 4, 10)
    >>> states = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>>
    >>> # Segment into 2048-point segments (4 segments per instance)
    >>> segmented, seg_states = segment_time_series(data, 2048, preserve_states=states)
    >>> print(f"Original shape: {data.shape}")
    >>> print(f"Segmented shape: {segmented.shape}")
    >>> print(f"Number of segments per instance: {segmented.shape[2] // data.shape[2]}")
    """
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(
            f"Data must be 3D (TIME, CHANNELS, INSTANCES), got shape {data.shape}"
        )

    time_points, n_channels, n_instances = data.shape

    if segment_length > time_points:
        raise ValueError(
            f"Segment length ({segment_length}) cannot exceed time series length ({time_points})"
        )

    if overlap >= segment_length:
        raise ValueError("Overlap must be less than segment length")

    # Calculate step size and number of segments per instance
    step_size = segment_length - overlap
    n_segments_per_instance = (time_points - segment_length) // step_size + 1

    # Pre-allocate output array
    total_segments = n_instances * n_segments_per_instance
    segmented_data = np.zeros((segment_length, n_channels, total_segments))

    # Segment each instance
    segment_idx = 0
    for instance in range(n_instances):
        for seg in range(n_segments_per_instance):
            start_idx = seg * step_size
            end_idx = start_idx + segment_length

            # Extract segment
            segmented_data[:, :, segment_idx] = data[start_idx:end_idx, :, instance]
            segment_idx += 1

    # Handle state preservation if requested
    segmented_states = None
    if preserve_states is not None:
        preserve_states = np.asarray(preserve_states)

        # Handle both (INSTANCES,) and (1, INSTANCES) shapes
        if preserve_states.ndim == 1:
            states_1d = preserve_states
            states_shape_is_1d = True
        else:
            states_1d = preserve_states.ravel()
            states_shape_is_1d = False

        # Replicate states for each segment
        segmented_states = np.repeat(states_1d, n_segments_per_instance)

        # Restore 2D shape if original was 2D
        if not states_shape_is_1d:
            segmented_states = segmented_states.reshape(1, -1)

    return segmented_data, segmented_states


def prepare_train_test_split(
    features: np.ndarray,
    states: np.ndarray,
    undamaged_states: list,
    train_fraction: float = 0.8,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare train/test split for outlier detection with undamaged/damaged labels.

    Splits undamaged data into train/test sets and combines remaining undamaged
    with all damaged data for testing.

    Parameters
    ----------
    features : array_like
        Feature matrix. Shape: (INSTANCES, FEATURES)

    states : array_like
        State labels for each instance. Shape: (INSTANCES,)

    undamaged_states : list
        List of state values that represent undamaged conditions.

    train_fraction : float, optional
        Fraction of undamaged data to use for training. Default is 0.8.

    random_seed : int, optional
        Random seed for reproducible splits.

    Returns
    -------
    train_features : ndarray
        Training features (undamaged only). Shape: (n_train, FEATURES)

    test_features : ndarray
        Test features (remaining undamaged + all damaged). Shape: (n_test, FEATURES)

    test_labels : ndarray
        Binary test labels (0=undamaged, 1=damaged). Shape: (n_test,)

    Examples
    --------
    >>> features = np.random.randn(170, 60)  # 170 instances, 60 features
    >>> states = np.repeat(np.arange(1, 18), 10)  # States 1-17, 10 each
    >>> undamaged_states = list(range(1, 10))  # States 1-9 are undamaged
    >>>
    >>> X_train, X_test, y_test = prepare_train_test_split(
    ...     features, states, undamaged_states, train_fraction=0.8
    ... )
    """
    features = np.asarray(features)
    states = np.asarray(states)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Identify undamaged instances
    is_undamaged = np.isin(states, undamaged_states)
    undamaged_features = features[is_undamaged]
    damaged_features = features[~is_undamaged]

    # Split undamaged data
    n_undamaged = undamaged_features.shape[0]
    n_train = int(np.round(train_fraction * n_undamaged))

    # Shuffle undamaged indices
    undamaged_indices = np.random.permutation(n_undamaged)

    # Create train/test splits
    train_features = undamaged_features[undamaged_indices[:n_train]]
    test_undamaged = undamaged_features[undamaged_indices[n_train:]]

    # Combine test data
    test_features = np.vstack([test_undamaged, damaged_features])

    # Create binary labels (0=undamaged, 1=damaged)
    n_test_undamaged = test_undamaged.shape[0]
    n_test_damaged = damaged_features.shape[0]
    test_labels = np.concatenate(
        [np.zeros(n_test_undamaged, dtype=int), np.ones(n_test_damaged, dtype=int)]
    )

    return train_features, test_features, test_labels
