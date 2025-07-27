"""
Geometry and spatial analysis functions for active sensing.

This module provides geometric calculations for guided wave propagation
including distance calculations, line-of-sight analysis, and grid generation.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from matplotlib.path import Path


def propagation_dist_2_points_shm(
    pair_list: np.ndarray, sensor_layout: np.ndarray, point_list: np.ndarray
) -> np.ndarray:
    """
    Calculate propagation distances from sensor pairs to points of interest.

    .. meta::
        :category: Active Sensing - Geometry
        :matlab_equivalent: propagationDist2Points_shm
        :complexity: Intermediate
        :data_type: Coordinates
        :output_type: Distances
        :display_name: Propagation Distance to POIs
        :verbose_call: [Propagation Distance] = Propagation Distance to POIs (Pair List, Sensor Layout, Points of Interest)

    Parameters
    ----------
    pair_list : array_like
        Sensor pair indices of shape (N_PAIRS, 2). Each row contains
        [actuator_index, sensor_index] (1-based MATLAB indexing).

        .. gui::
            :widget: array_input
            :description: Sensor pair indices (1-based)

    sensor_layout : array_like
        Sensor coordinates of shape (N_SENSORS, 2). Each row contains
        [x_coordinate, y_coordinate] for each sensor.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Sensor layout coordinates

    point_list : array_like
        Points of interest coordinates of shape (N_POINTS, 2). Each row
        contains [x_coordinate, y_coordinate] for each POI.

        .. gui::
            :widget: array_input
            :description: Points of interest coordinates

    Returns
    -------
    prop_distance : ndarray
        Propagation distances of shape (N_PAIRS, N_POINTS). Each element
        [i,j] contains the total propagation distance from actuator to
        sensor for pair i through point j.

    Notes
    -----
    Calculates the total propagation distance for guided waves traveling
    from an actuator through a point of interest to a sensor. This assumes
    the wave scatters at the POI and continues to the receiver.

    The distance is computed as:
    distance = ||actuator - POI|| + ||POI - sensor||

    This is fundamental for time-of-flight calculations in active sensing.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import propagation_dist_2_points_shm
    >>>
    >>> # Define sensor layout (4 sensors in square pattern)
    >>> sensor_layout = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>>
    >>> # Define sensor pairs (MATLAB 1-based indexing)
    >>> pair_list = np.array([[1, 3], [2, 4]])  # diagonal pairs
    >>>
    >>> # Define points of interest
    >>> point_list = np.array([[0.5, 0.5], [0.3, 0.7]])
    >>>
    >>> # Calculate propagation distances
    >>> distances = propagation_dist_2_points_shm(pair_list, sensor_layout, point_list)
    >>> print(f"Distance matrix shape: {distances.shape}")
    """
    pair_list = np.asarray(pair_list, dtype=int)
    sensor_layout = np.asarray(sensor_layout, dtype=np.float64)
    point_list = np.asarray(point_list, dtype=np.float64)

    n_pairs = pair_list.shape[0]
    n_points = point_list.shape[0]

    # Initialize output matrix
    prop_distance = np.zeros((n_pairs, n_points))

    # Process each sensor pair
    for i in range(n_pairs):
        # Convert from MATLAB 1-based to Python 0-based indexing
        actuator_idx = pair_list[i, 0] - 1
        sensor_idx = pair_list[i, 1] - 1

        # Get actuator and sensor coordinates
        actuator_coord = sensor_layout[actuator_idx, :]
        sensor_coord = sensor_layout[sensor_idx, :]

        # Calculate distances to each point of interest
        for j in range(n_points):
            poi_coord = point_list[j, :]

            # Distance from actuator to POI
            dist_act_to_poi = np.sqrt(np.sum((actuator_coord - poi_coord) ** 2))

            # Distance from POI to sensor
            dist_poi_to_sens = np.sqrt(np.sum((poi_coord - sensor_coord) ** 2))

            # Total propagation distance
            prop_distance[i, j] = dist_act_to_poi + dist_poi_to_sens

    return prop_distance


def distance_2_index_shm(
    prop_distance: np.ndarray,
    sample_rate: float,
    velocity: float,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Convert propagation distances to waveform sample indices.

    .. meta::
        :category: Active Sensing - Signal Processing
        :matlab_equivalent: distance2Index_shm
        :complexity: Basic
        :data_type: Distances
        :output_type: Indices
        :display_name: Distance to Index
        :verbose_call: [Indices] = Distance To Index (Propagation Distance, Sample Rate, Velocity, Offset)

    Parameters
    ----------
    prop_distance : array_like
        Propagation distances in meters.

        .. gui::
            :widget: array_input
            :description: Propagation distances (meters)

    sample_rate : float
        Sampling rate in Hz.

        .. gui::
            :widget: number_input
            :min: 1000
            :max: 10000000
            :default: 1000000
            :description: Sampling rate (Hz)

    velocity : float
        Wave propagation velocity in m/s.

        .. gui::
            :widget: number_input
            :min: 100
            :max: 10000
            :default: 3000
            :description: Wave velocity (m/s)

    offset : float, optional
        Time offset in seconds (default: 0.0).

        .. gui::
            :widget: number_input
            :min: 0
            :max: 0.001
            :default: 0.0
            :description: Time offset (seconds)

    Returns
    -------
    indices : ndarray
        Waveform sample indices corresponding to time-of-flight.
        Same shape as prop_distance, rounded to nearest integer.

    Notes
    -----
    Converts physical distances to waveform sample indices using:
    time = distance / velocity + offset
    index = round(time * sample_rate)

    This is essential for extracting features at the correct time-of-flight
    locations in guided wave analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import distance_2_index_shm
    >>>
    >>> # Example distances
    >>> distances = np.array([0.1, 0.2, 0.3, 0.4])  # meters
    >>>
    >>> # Convert to indices
    >>> indices = distance_2_index_shm(distances, 1e6, 3000)
    >>> print(f"Sample indices: {indices}")
    """
    prop_distance = np.asarray(prop_distance, dtype=np.float64)

    # Calculate time-of-flight
    time_of_flight = prop_distance / velocity + offset

    # Convert to sample indices
    indices = np.round(time_of_flight * sample_rate).astype(int)

    return indices


def build_contained_grid_shm(
    border_struct: List[np.ndarray], x_spacing: float, y_spacing: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build grid of points contained within structure borders.

    .. meta::
        :category: Active Sensing - Geometry
        :matlab_equivalent: buildContainedGrid_shm
        :complexity: Advanced
        :data_type: Geometry
        :output_type: Grid Points
        :display_name: Build Contained Grid
        :verbose_call: [Grid Points, Points of Interest Mask, X Matrix, Y Matrix] = Build Contained Grid (Border Structure, X Spacing, Y Spacing)

    Parameters
    ----------
    border_struct : list of array_like
        List of border line segments. Each element is an array of shape
        (N_POINTS, 2) defining a border polygon.

        .. gui::
            :widget: geometry_input
            :description: Structure border definition

    x_spacing : float
        Grid spacing in x-direction.

        .. gui::
            :widget: number_input
            :min: 0.001
            :max: 1.0
            :default: 0.01
            :description: X-direction spacing

    y_spacing : float
        Grid spacing in y-direction.

        .. gui::
            :widget: number_input
            :min: 0.001
            :max: 1.0
            :default: 0.01
            :description: Y-direction spacing

    Returns
    -------
    grid_points : ndarray
        Coordinates of grid points inside structure, shape (N_INSIDE, 2).

    poi_mask : ndarray
        Boolean mask indicating which grid points are inside structure.

    x_matrix : ndarray
        X-coordinate matrix for 2D grid.

    y_matrix : ndarray
        Y-coordinate matrix for 2D grid.

    Notes
    -----
    Creates a regular grid and tests which points lie inside the structure
    boundary. Uses polygon containment testing to determine interior points.

    This is fundamental for creating spatial imaging grids for damage detection.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import build_contained_grid_shm
    >>>
    >>> # Define rectangular structure border
    >>> border = [np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
    >>>
    >>> # Build grid
    >>> points, mask, x_mat, y_mat = build_contained_grid_shm(border, 0.1, 0.1)
    >>> print(f"Grid points inside structure: {len(points)}")
    """
    # Find overall bounding box
    all_points = np.vstack(border_struct)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    # Create regular grid
    x_range = np.arange(x_min, x_max + x_spacing, x_spacing)
    y_range = np.arange(y_min, y_max + y_spacing, y_spacing)
    x_matrix, y_matrix = np.meshgrid(x_range, y_range)

    # Flatten for point-in-polygon testing
    grid_points_all = np.column_stack([x_matrix.flatten(), y_matrix.flatten()])

    # Test which points are inside ALL border polygons
    poi_mask_flat = np.ones(len(grid_points_all), dtype=bool)

    for border_segment in border_struct:
        if len(border_segment) > 2:  # Valid polygon
            # Create matplotlib Path object for polygon containment testing
            polygon_path = Path(border_segment)
            # Points inside this polygon
            inside_this_polygon = polygon_path.contains_points(grid_points_all)
            # Keep only points inside this polygon
            poi_mask_flat &= inside_this_polygon

    # Reshape mask to grid shape
    poi_mask = poi_mask_flat.reshape(x_matrix.shape)

    # Extract points that are inside structure
    grid_points = grid_points_all[poi_mask_flat]

    return grid_points, poi_mask, x_matrix, y_matrix


def sensor_pair_line_of_sight_shm(
    pair_list: np.ndarray,
    sensor_layout: np.ndarray,
    point_list: np.ndarray,
    border: List[np.ndarray],
) -> np.ndarray:
    """
    Determine line-of-sight visibility for sensor pairs to points of interest.

    .. meta::
        :category: Active Sensing - Geometry
        :matlab_equivalent: sensorPairLineOfSight_shm
        :complexity: Advanced
        :data_type: Geometry
        :output_type: Visibility Matrix
        :display_name: Sensor Pair Line of Sight
        :verbose_call: [Line of Sight Matrix] = Sensor Pair Line of Sight (Pair List, Sensor Layout, Points of Interest, Border)

    Parameters
    ----------
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

    point_list : array_like
        Points of interest coordinates of shape (N_POINTS, 2).

        .. gui::
            :widget: array_input
            :description: Points of interest coordinates

    border : list of array_like
        Structure border line segments.

        .. gui::
            :widget: geometry_input
            :description: Structure border definition

    Returns
    -------
    line_of_sight : ndarray
        Boolean matrix of shape (N_PAIRS, N_POINTS) indicating visibility.
        True means both actuator->POI and POI->sensor paths are clear.

    Notes
    -----
    Tests whether guided wave propagation paths are geometrically feasible
    by checking if line segments intersect with structure boundaries.

    For each sensor pair and POI, tests:
    1. Actuator to POI line segment
    2. POI to sensor line segment

    Both paths must be clear for line-of-sight to be True.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import sensor_pair_line_of_sight_shm
    >>>
    >>> # Define geometry
    >>> sensors = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> pairs = np.array([[1, 3]])  # MATLAB 1-based
    >>> points = np.array([[0.5, 0.5]])
    >>> border = [np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
    >>>
    >>> # Test line of sight
    >>> los = sensor_pair_line_of_sight_shm(pairs, sensors, points, border)
    >>> print(f"Line of sight matrix: {los}")
    """
    pair_list = np.asarray(pair_list, dtype=int)
    sensor_layout = np.asarray(sensor_layout, dtype=np.float64)
    point_list = np.asarray(point_list, dtype=np.float64)

    n_pairs = pair_list.shape[0]
    n_points = point_list.shape[0]

    # Initialize line-of-sight matrix
    line_of_sight = np.ones((n_pairs, n_points), dtype=bool)

    # Process each sensor pair
    for i in range(n_pairs):
        # Convert from MATLAB 1-based to Python 0-based indexing
        actuator_idx = pair_list[i, 0] - 1
        sensor_idx = pair_list[i, 1] - 1

        actuator_coord = sensor_layout[actuator_idx, :]
        sensor_coord = sensor_layout[sensor_idx, :]

        # Check each point of interest
        for j in range(n_points):
            poi_coord = point_list[j, :]

            # Check if both paths are clear
            path1_clear = _check_line_segment_clear(actuator_coord, poi_coord, border)
            path2_clear = _check_line_segment_clear(poi_coord, sensor_coord, border)

            line_of_sight[i, j] = path1_clear and path2_clear

    return line_of_sight


def _check_line_segment_clear(
    point1: np.ndarray, point2: np.ndarray, border: List[np.ndarray]
) -> bool:
    """
    Check if line segment between two points intersects with border.

    Parameters
    ----------
    point1 : ndarray
        Start point coordinates.
    point2 : ndarray
        End point coordinates.
    border : list of ndarray
        Border line segments.

    Returns
    -------
    clear : bool
        True if line segment doesn't intersect border.
    """
    # Simple implementation: check if line segment intersects any border segment
    # This is a simplified version - more sophisticated algorithms could be used

    for border_segment in border:
        if len(border_segment) < 2:
            continue

        # Check intersection with each border edge
        for k in range(len(border_segment) - 1):
            border_p1 = border_segment[k]
            border_p2 = border_segment[k + 1]

            if _line_segments_intersect(point1, point2, border_p1, border_p2):
                return False

    return True


def _line_segments_intersect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> bool:
    """
    Check if two line segments intersect.

    Uses the orientation-based method to determine intersection.
    """

    def orientation(p, q, r):
        """Find orientation of ordered triplet (p, q, r)."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0  # collinear
        return 1 if val > 0 else 2  # clockwise or counterclockwise

    def on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        return (
            q[0] <= max(p[0], r[0])
            and q[0] >= min(p[0], r[0])
            and q[1] <= max(p[1], r[1])
            and q[1] >= min(p[1], r[1])
        )

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases for collinear points
    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return False


def fill_2d_map_shm(data_1d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill 2D map from 1D data using boolean mask.

    .. meta::
        :category: Active Sensing - Utilities
        :matlab_equivalent: fill2DMap_shm
        :complexity: Basic
        :data_type: Arrays
        :output_type: 2D Map
        :display_name: Fill 2D Map
        :verbose_call: [Data Map 2D] = Fill 2D Map (Data 1D, Mask)

    Parameters
    ----------
    data_1d : array_like
        1D data values corresponding to True mask locations.

        .. gui::
            :widget: array_input
            :description: 1D data values

    mask : array_like
        2D boolean mask indicating where to place data.

        .. gui::
            :widget: array_input
            :description: 2D boolean mask

    Returns
    -------
    data_map_2d : ndarray
        2D map with data_1d values at mask==True locations,
        zeros elsewhere. Same shape as mask.

    Notes
    -----
    This function maps 1D results back to a 2D spatial grid for visualization.
    Essential for creating damage images from point-wise calculations.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.active_sensing import fill_2d_map_shm
    >>>
    >>> # Create mask and data
    >>> mask = np.array([[True, False], [False, True]])
    >>> data = np.array([1.0, 2.0])
    >>>
    >>> # Fill 2D map
    >>> map_2d = fill_2d_map_shm(data, mask)
    >>> print(f"2D map:\\n{map_2d}")
    """
    data_1d = np.asarray(data_1d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    # Initialize output with zeros
    data_map_2d = np.zeros(mask.shape, dtype=np.float64)

    # Fill data at mask locations
    data_map_2d[mask] = data_1d

    return data_map_2d