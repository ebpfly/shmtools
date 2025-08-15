"""
LADPackage Active Sensing utility functions.

This module provides LADPackage-specific wrappers for active sensing functionality,
matching the MATLAB LADPackage interface exactly.
"""

import numpy as np
import scipy.io
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.path import Path as mpath

# Add project root to path for imports
module_path = Path(__file__).resolve()
project_root = module_path.parent.parent.parent
sys.path.insert(0, str(project_root))

from shmtools.active_sensing import (
    incoherent_matched_filter_shm,
    propagation_dist_2_points_shm,
    distance_2_index_shm,
    build_contained_grid_shm,
    fill_2d_map_shm,
    get_prop_dist_2_boundary_shm,
    struct_cell_2_mat_shm,
    extract_subsets_shm,
    flex_logic_filter_shm,
    sum_mult_dims_shm,
    reduce_2_pair_subset_shm,
)


def import_active_sense_data(filename: Optional[str] = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, float, np.ndarray, np.ndarray
]:
    """
    Import active sensing dataset.
    
    LADPackage wrapper for importing active sensing data from .mat files.
    
    .. meta::
        :category: LAD
        :matlab_equivalent: importActiveSenseData
        :complexity: Basic
        :data_type: Active Sensing
        :output_type: Data
        :display_name: Import Active Sensing Dataset
        :verbose_call: [Baseline Waveforms, Test Waveforms, Sensor Layout, Sensor Pair List, Border Structure, Sample Rate, Actuation Waveform, Damage Location] = Import Active Sensing Dataset(File Name)
    
    Parameters
    ----------
    filename : str, optional
        Path to active sensing .mat data file. If None, uses default example data.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat"]
            :description: Active sensing data file
    
    Returns
    -------
    waveform_base : ndarray, shape (time, sensor_pairs)
        Matrix of waveforms acquired before damage was introduced
    waveform_test : ndarray, shape (time, sensor_pairs)
        Matrix of waveforms acquired after damage was introduced
    sensor_layout : ndarray, shape (3, n_sensors)
        Sensor layout IDs and coordinates [sensorID, xCoord, yCoord]
    pair_list : ndarray, shape (2, n_pairs)
        Matrix of actuator-sensor pairs [actuatorID, sensorID]
    border_struct : dict
        Dictionary containing 'outside' and 'inside' border definitions
    sample_rate : float
        Sampling rate (Hz)
    actuation_waveform : ndarray, shape (time,)
        Waveform used for actuation
    damage_location : ndarray, shape (2,)
        X and Y coordinates of damage location
    
    Notes
    -----
    MATLAB Compatibility: Direct conversion from LADPackage importActiveSenseData.m
    Default dataset is 'data_example_ActiveSense.mat'
    
    Examples
    --------
    >>> waveform_base, waveform_test, sensor_layout, pair_list, border_struct, sample_rate, actuation_waveform, damage_location = import_active_sense_data()
    >>> print(f"Loaded {waveform_base.shape[1]} sensor pairs")
    """
    if filename is None:
        # Try multiple possible locations for the data file
        possible_paths = [
            Path(__file__).parent / "data" / "data_example_ActiveSense.mat",
            project_root / "examples" / "data" / "data_example_ActiveSense.mat",
            project_root / "LADPackage" / "active_sensing" / "data" / "data_example_ActiveSense.mat",
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                "Could not find data_example_ActiveSense.mat. Please download it to:\n"
                f"  {possible_paths[0]}\n"
                "or provide an explicit filename parameter."
            )
    else:
        data_path = Path(filename)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {filename}")
    
    # Load the MATLAB data
    mat_data = scipy.io.loadmat(str(data_path))
    
    # Extract all required variables
    waveform_base = mat_data['waveformBase']
    waveform_test = mat_data['waveformTest']
    sensor_layout = mat_data['sensorLayout']
    pair_list = mat_data['pairList']
    border_struct = mat_data['borderStruct']
    sample_rate = float(mat_data['sampleRate'].flatten()[0])
    actuation_waveform = mat_data['actuationWaveform'].flatten()
    damage_location = mat_data['damageLocation'].flatten()
    
    # Convert border_struct from MATLAB structure to Python dictionary
    # Handle the nested structure properly
    border_dict = {}
    if hasattr(border_struct, 'dtype') and border_struct.dtype.names:
        # It's a structured array - extract fields from (1,1) array
        struct_item = border_struct[0, 0] if border_struct.ndim == 2 else border_struct[0]
        for field in border_struct.dtype.names:
            field_value = struct_item[field]
            # Convert cell arrays to list if needed
            if isinstance(field_value, np.ndarray) and field_value.dtype == object:
                border_dict[field] = [item for item in field_value.flatten()]
            else:
                border_dict[field] = field_value
    else:
        # Already a dictionary or simple array
        border_dict = border_struct
    
    return (
        waveform_base, 
        waveform_test, 
        sensor_layout, 
        pair_list, 
        border_dict, 
        sample_rate, 
        actuation_waveform, 
        damage_location
    )


def process_active_sensing_waveforms(
    sensor_subset: np.ndarray,
    sensor_layout: np.ndarray,
    pair_list: np.ndarray,
    waveform_base: np.ndarray,
    waveform_test: np.ndarray,
    matched_waveform: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process active sensing waveforms with baseline subtraction and matched filtering.
    
    LADPackage wrapper for active sensing waveform processing pipeline.
    
    .. meta::
        :category: LAD
        :matlab_equivalent: ProcessActiveSensingWaveforms
        :complexity: Intermediate
        :data_type: Active Sensing
        :output_type: Features
        :display_name: Process Active Sensing Waveforms
        :verbose_call: [Filter Result, Layout Subset, Sensor Pair Subset] = Process Active Sensing Waveforms (Sensor Subset List, Sensor Layout, Sensor Pair List, Baseline Waveforms, Test Waveforms, Excitation Waveform)
    
    Parameters
    ----------
    sensor_subset : ndarray, shape (n_subset,)
        List of sensor subset IDs (1-indexed for MATLAB compatibility)
        
        .. gui::
            :widget: array_input
            :description: Sensor IDs to include
    
    sensor_layout : ndarray, shape (3, n_sensors)
        Sensor layout IDs and coordinates
        
        .. gui::
            :widget: data_input
            :description: Sensor positions
    
    pair_list : ndarray, shape (2, n_pairs)
        Matrix of actuator-sensor pairs
        
        .. gui::
            :widget: data_input
            :description: Sensor pair definitions
    
    waveform_base : ndarray, shape (time, pairs)
        Baseline waveforms
        
        .. gui::
            :widget: data_input
            :description: Reference waveforms
    
    waveform_test : ndarray, shape (time, pairs)
        Test waveforms
        
        .. gui::
            :widget: data_input
            :description: Test waveforms
    
    matched_waveform : ndarray, shape (time,)
        Excitation waveform for matched filtering
        
        .. gui::
            :widget: data_input
            :description: Actuation signal
    
    Returns
    -------
    filter_result : ndarray
        Matched filter output
    layout_subset : ndarray
        Subset of sensor layout
    pair_subset : ndarray
        Subset of sensor pairs
    
    Notes
    -----
    Performs three steps:
    1. Reduce to pair subset based on selected sensors
    2. Baseline subtraction (test - baseline)
    3. Incoherent matched filtering
    """
    # Step 1: Reduce to pair subset
    pair_subset, layout_subset, waveform_base_sub, waveform_test_sub = reduce_2_pair_subset_shm(
        sensor_subset, sensor_layout, pair_list, waveform_base, waveform_test
    )
    
    # Step 2: Baseline subtraction
    dif_waveform = waveform_test_sub - waveform_base_sub
    
    # Step 3: Incoherent matched filter
    filter_result = incoherent_matched_filter_shm(dif_waveform, matched_waveform)
    
    return filter_result, layout_subset, pair_subset


def arrival_filter(
    waveforms: np.ndarray,
    front_clip: int = 0,
    arrival_offset: int = 0
) -> np.ndarray:
    """
    Filter guided wave envelopes to first arrival.
    
    LADPackage implementation of arrival time filtering for guided waves.
    
    .. meta::
        :category: LAD
        :matlab_equivalent: arrivalFilter
        :complexity: Advanced
        :data_type: Active Sensing
        :output_type: Features
        :display_name: Arrival Filter
        :verbose_call: [filteredWaveforms] = Arrival Filter (waveforms, frontClip, arrivalOffset)
    
    Parameters
    ----------
    waveforms : ndarray, shape (n_samples, n_channels)
        Enveloped, differenced guided wave signals
        
        .. gui::
            :widget: data_input
            :description: Input waveform envelopes
    
    front_clip : int, default=0
        Number of samples to ignore at beginning of waveform
        
        .. gui::
            :widget: number_input
            :min: 0
            :max: 1000
            :default: 450
            :description: Samples to skip at start
    
    arrival_offset : int, default=0
        Number of samples to shift result by (approximately half excitation width)
        
        .. gui::
            :widget: number_input
            :min: 0
            :max: 1000
            :default: 450
            :description: Arrival time offset
    
    Returns
    -------
    filtered_waveforms : ndarray
        Filtered waveform envelopes
    
    Notes
    -----
    Algorithm from Flynn et al. 2011, Proceedings of Royal Society A.
    Uses maximum likelihood estimation for arrival time detection.
    """
    N = front_clip
    M = arrival_offset
    x = waveforms.copy()
    
    # Remove front clipping region
    x = x[N:, :]
    x = x + 1e-16  # Avoid log(0)
    
    num_time, num_trans = x.shape
    
    # Create index arrays
    n1 = np.tile(np.arange(1, num_time + 1)[:, np.newaxis], (1, num_trans))
    n2 = num_time - n1 + 1
    
    # Calculate log sum
    x1sL = np.sum(np.log(x), axis=0)
    
    # Calculate cumulative squared sums
    x1s2 = np.cumsum(x**2, axis=0)
    x2s2 = np.flipud(np.cumsum(np.flipud(x**2), axis=0))
    
    # Calculate variances
    sig1 = 1.0 / n1 / 2.0 * x1s2
    sig2 = 1.0 / n2 / 2.0 * x2s2
    
    # Calculate likelihood function
    y = 2 * x1sL - (n1 * np.log(sig1) + n2 * np.log(sig2) + n1[-1, 0])
    
    # Handle edge cases
    y[0, :] = y[1, :]
    y[-1, :] = y[-2, :]
    
    # Add padding
    pad = np.zeros((M + N, num_trans)) + np.min(y, axis=0)
    if M > 0:
        y = np.vstack([pad, y[:-M, :]])
    else:
        y = np.vstack([pad, y])
    
    return y


def map_active_sensing_geometry(
    velocity: float,
    subset_window: Optional[np.ndarray],
    distance_allowance: float,
    border_struct: Dict,
    x_spacing: Optional[float],
    y_spacing: Optional[float],
    sample_rate: float,
    offset: np.ndarray,
    data: np.ndarray,
    pair_list: np.ndarray,
    sensor_layout: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Map processed active sensing waveforms to geometry.
    
    LADPackage wrapper for active sensing geometry mapping.
    
    .. meta::
        :category: LAD
        :matlab_equivalent: MapActiveSensingGeometry
        :complexity: Advanced
        :data_type: Active Sensing
        :output_type: Image
        :display_name: Map Active Sensing Geometry
        :verbose_call: [X Matrix, Y Matrix, Combined Geometry, Data Map 2D] = Map Active Sensing Map (Velocity, Subset Window, Distance Allowance, Geometry, X Spacing, Y Spacing, Sample Rate, Actuation Waveform, Data, Sensor Pair List, Sensor Layout)
    
    Parameters
    ----------
    velocity : float
        Propagation velocity (distance/time)
    subset_window : ndarray or None
        Window to apply to subsets
    distance_allowance : float
        Distance past nearest border that can be mapped
    border_struct : dict
        Border structure defining geometry
    x_spacing : float or None
        Desired spacing between X coordinates
    y_spacing : float or None
        Desired spacing between Y coordinates
    sample_rate : float
        Sample rate (Hz)
    offset : ndarray
        Offset of impulse from start of waveform
    data : ndarray
        Full data matrix
    pair_list : ndarray
        Sensor pair list
    sensor_layout : ndarray
        Sensor positions
    
    Returns
    -------
    x_matrix : ndarray
        Matrix of X coordinates
    y_matrix : ndarray
        Matrix of Y coordinates
    combined_matrix : ndarray
        Combined geometry
    data_map_2d : ndarray
        2D data map with NaNs for empty space
    """
    # Set defaults
    if velocity is None or velocity == 0:
        velocity = 66000.0
    if subset_window is None:
        subset_window = 1  # Default window length of 1
    elif isinstance(subset_window, np.ndarray):
        subset_window = int(subset_window[0]) if subset_window.size > 0 else 1
    if distance_allowance is None:
        distance_allowance = np.inf
    if x_spacing is None or x_spacing == 0:
        x_spacing = 0.5
    if y_spacing is None or y_spacing == 0:
        y_spacing = 0.5
    
    # Step 1: Pass through pair list and sensor layout
    pair_list_out = pair_list
    sensor_layout_out = sensor_layout
    
    # Step 2: Combine structure-cell to matrix
    # Handle border structure from MATLAB
    if isinstance(border_struct, dict):
        # Extract the 'outside' field which contains the border segments
        if 'outside' in border_struct:
            border_data = border_struct['outside']
            if isinstance(border_data, list) and len(border_data) > 0:
                combined_matrix = border_data[0] if isinstance(border_data[0], np.ndarray) else np.array(border_data[0])
            elif isinstance(border_data, np.ndarray):
                combined_matrix = border_data
            else:
                combined_matrix = struct_cell_2_mat_shm(border_struct)
        else:
            combined_matrix = struct_cell_2_mat_shm(border_struct)
    else:
        combined_matrix = struct_cell_2_mat_shm(border_struct)
    
    # Step 3: Build contained grid
    # Convert line segments to polygon vertices for grid building
    if combined_matrix.shape[0] == 4:
        # Format is [x1; y1; x2; y2] per column - convert to polygon
        n_segments = combined_matrix.shape[1]
        polygon_points = []
        for i in range(n_segments):
            x1, y1, x2, y2 = combined_matrix[:, i]
            if i == 0:
                polygon_points.append([x1, y1])
            polygon_points.append([x2, y2])
        polygon_border = np.array(polygon_points)
    else:
        polygon_border = combined_matrix

    poly_path = mpath(polygon_border)
    
    # Build grid using simplified approach
    x_min = np.min(polygon_border[:, 0])
    x_max = np.max(polygon_border[:, 0])
    y_min = np.min(polygon_border[:, 1])
    y_max = np.max(polygon_border[:, 1])
    
    x_range = np.arange(x_min, x_max + x_spacing, x_spacing)
    y_range = np.arange(y_min, y_max + y_spacing, y_spacing)
    x_matrix, y_matrix = np.meshgrid(x_range, y_range)
    
    # Create point list and mask. Mask is all points inside the polygon.
    point_list = np.column_stack([x_matrix.flatten(), y_matrix.flatten()])
    point_mask = np.asarray([poly_path.contains_point(pt) for pt in point_list])
    point_list = point_list[point_mask,:]
    point_mask = point_mask.reshape(x_matrix.shape)



    # Step 4: Propagation distance to POIs
    # Ensure correct shapes for function call
    if pair_list.ndim == 1:
        pair_list_out = pair_list_out.reshape(-1, 1)
    if sensor_layout.ndim == 1:
        sensor_layout_out = sensor_layout_out.reshape(-1, 1)
    
    prop_distance = propagation_dist_2_points_shm(
        pair_list_out, sensor_layout_out, point_list
    )
    
    # Step 5: Distance to index
    # The offset should be the length of the actuation waveform, not the waveform itself
    if isinstance(offset, np.ndarray) and offset.size > 1:
        # Use the length of the actuation waveform as the offset
        offset_value = len(offset)
    else:
        offset_value = offset if np.isscalar(offset) else 0
    
    indices = distance_2_index_shm(prop_distance, sample_rate, velocity, offset_value)
    
    # Step 6: Extract subsets
    data_subsets = extract_subsets_shm(data, indices, subset_window)
    
    # Step 9: Get propagation distance to boundary
    prop_dist_boundary, min_prop_dist = get_prop_dist_2_boundary_shm(
        pair_list_out, sensor_layout_out, combined_matrix
    )
    
    # Step 10: Build logical array for distance filtering
    # MATLAB: below_max_distance = bsxfun(@lt, distance - distance_allowance, max_distance)
    below_max_distance = (prop_distance - distance_allowance) < min_prop_dist[:, np.newaxis]
    
    # Step 11: Flexible logic filter
    filtered_data = flex_logic_filter_shm(data_subsets, below_max_distance)
    
    # Step 12: Sum multiple dimensions
    filtered_data = data_subsets
    data_sum = sum_mult_dims_shm(filtered_data, [0, 2])  # Sum along first two dimensions
    
    # Step 13: Fill 2D map
    data_map_2d = fill_2d_map_shm(data_sum, point_mask)
    
    return x_matrix, y_matrix, combined_matrix, data_map_2d


def plot_as_result(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    data_map_2d: np.ndarray,
    border: np.ndarray,
    sensor_layout: np.ndarray
) -> None:
    """
    Plot active sensing map with geometry.
    
    LADPackage visualization for active sensing results.
    
    .. meta::
        :category: LAD
        :matlab_equivalent: plotASResult
        :complexity: Basic
        :data_type: Image
        :output_type: Plot
        :display_name: Plot Active Sensing Map
        :verbose_call: Plot Active Sensing Map (X Matrix, Y Matrix, Data Map 2D, Border, Sensor Subset Layout)
    
    Parameters
    ----------
    x_matrix : ndarray
        Matrix of X coordinates
    y_matrix : ndarray
        Matrix of Y coordinates
    data_map_2d : ndarray
        2D data map
    border : ndarray
        Border line segments
    sensor_layout : ndarray
        Sensor positions
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot the data map
    im = ax.pcolormesh(x_matrix, y_matrix, data_map_2d, shading='auto', cmap='jet')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Signal Amplitude')
    
    # Plot border if provided
    if border is not None and len(border) > 0:
        for segment in border.T:
            if len(segment) >= 4:
                ax.plot([segment[0], segment[2]], [segment[1], segment[3]], 'k-', linewidth=2)
    
    # Plot sensor positions if provided
    if sensor_layout is not None and sensor_layout.shape[1] > 0:
        # sensor_layout is (3, n_sensors) with [ID, x, y]
        sensor_x = sensor_layout[1, :]
        sensor_y = sensor_layout[2, :]
        ax.plot(sensor_x, sensor_y, 'ko', markersize=20, markerfacecolor='white', 
                markeredgewidth=2, label='Sensors')
        
        # Add sensor IDs
        for i in range(len(sensor_x)):
            sensor_id = int(sensor_layout[0, i])
            ax.text(sensor_x[i], sensor_y[i], str(sensor_id), 
                   ha='center', va='center', fontsize=8)
    
    # Labels and title
    ax.set_xlabel('X Position (cm)')
    ax.set_ylabel('Y Position (cm)')
    ax.set_title('Active Sensing Image')
    ax.set_aspect('equal')
    
    if sensor_layout is not None and sensor_layout.shape[1] > 0:
        ax.legend()
    
    plt.tight_layout()
    # plt.show()