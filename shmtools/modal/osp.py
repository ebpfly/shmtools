"""
Optimal sensor placement functions for modal analysis.

This module provides functions for determining optimal sensor locations
based on modal analysis criteria including Fisher Information Matrix
and Maximum Norm methods.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import warnings


def response_interp_shm(
    geom_layout: np.ndarray, 
    disp_vec: np.ndarray, 
    resp_dof: np.ndarray, 
    use_3d_interp: bool = False
) -> np.ndarray:
    """
    Interpolate response from DOF indices to node XYZ coordinates.
    
    Converts 1D response vector indexed by degrees of freedom (DOF) to
    3D response array indexed by node coordinates.
    
    .. meta::
        :category: Auxiliary - Plotting
        :matlab_equivalent: responseInterp_shm
        :complexity: Basic
        :data_type: Modal Data
        :output_type: Visualization Data
        :display_name: Response Interpolation
        :verbose_call: [Response XYZ] = Response Interpolation (Geometry Layout, Displacement Vector, Response DOF, Use 3D Interpolation)
        
    Parameters
    ----------
    geom_layout : array_like, shape (n_nodes, 3) or (4, n_nodes)
        Node coordinates [X, Y, Z] for each node. If shape is (4, n_nodes),
        assumes first row is node indices and rows 2-4 are X, Y, Z coordinates.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: Node coordinate matrix
            
    disp_vec : array_like, shape (n_dof,)
        Response values at each DOF.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: Response/displacement vector
            
    resp_dof : array_like, shape (n_dof, 2)
        DOF definitions [node_index, direction] where direction is:
        1=X, 2=Y, 3=Z
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: DOF to node mapping
            
    use_3d_interp : bool, optional
        Whether to use 3D interpolation (not implemented).
        Default: False
        
        .. gui::
            :widget: checkbox
            :default: false
            
    Returns
    -------
    resp_xyz : array_like, shape (n_nodes, 3)
        Response values [X, Y, Z] at each node.
    """
    # Handle different input formats
    if geom_layout.shape[0] == 4:
        # Format: [node_indices; X; Y; Z]
        n_nodes = geom_layout.shape[1]
        geom_xyz = geom_layout[1:4, :].T  # Extract X,Y,Z and transpose
    else:
        # Format: [X, Y, Z] per row
        n_nodes = geom_layout.shape[0]
        geom_xyz = geom_layout
    
    resp_xyz = np.zeros((n_nodes, 3))
    
    # Convert DOF vector to node XYZ responses
    for i, (node_idx, direction) in enumerate(resp_dof):
        # Convert from MATLAB 1-based to Python 0-based indexing
        node_idx = int(node_idx) - 1
        direction = int(direction) - 1  # 1,2,3 -> 0,1,2 for X,Y,Z
        
        if 0 <= node_idx < n_nodes and 0 <= direction < 3:
            resp_xyz[node_idx, direction] = disp_vec[i]
    
    return resp_xyz


def add_resp_2_geom_shm(
    geom_layout: np.ndarray, 
    resp_xyz: np.ndarray, 
    scale: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Add response vector to geometry for deformed shape visualization.
    
    .. meta::
        :category: Auxiliary - Plotting
        :matlab_equivalent: addResp2Geom_shm
        :complexity: Basic
        :data_type: Modal Data
        :output_type: Visualization Data
        :display_name: Add Response to Geometry
        :verbose_call: [Response Layout, Response Scale] = Add Response to Geometry (Geometry Layout, Response XYZ, Scale)
        
    Parameters
    ----------
    geom_layout : array_like, shape (n_nodes, 3) or (4, n_nodes)
        Original node coordinates [X, Y, Z]. If shape is (4, n_nodes),
        assumes first row is node indices and rows 2-4 are X, Y, Z coordinates.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: Node coordinate matrix
            
    resp_xyz : array_like, shape (n_nodes, 3)
        Response/displacement values [X, Y, Z] at each node.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: Response XYZ values
            
    scale : float, optional
        Scale factor for response. If None, automatically scaled
        to 10% of structure size.
        
        .. gui::
            :widget: number_input
            :min: 0.0
            :max: 100.0
            :default: null
            :description: Response scale factor
            
    Returns
    -------
    resp_layout : array_like, shape (n_nodes, 3) or (4, n_nodes)
        Deformed node coordinates (same format as input).
    resp_scale : float
        Applied scale factor.
    """
    # Handle different input formats
    if geom_layout.shape[0] == 4:
        # Format: [node_indices; X; Y; Z]
        geom_xyz = geom_layout[1:4, :].T  # Extract X,Y,Z and transpose
        output_4row = True
    else:
        # Format: [X, Y, Z] per row
        geom_xyz = geom_layout
        output_4row = False
    
    if scale is None:
        # Auto-scale to 10% of structure size
        geom_range = np.max(geom_xyz, axis=0) - np.min(geom_xyz, axis=0)
        max_resp = np.max(np.abs(resp_xyz))
        
        if max_resp > 0:
            scale = 0.1 * np.max(geom_range) / max_resp
        else:
            scale = 1.0
    
    # Apply deformation
    deformed_xyz = geom_xyz + scale * resp_xyz
    
    # Return in same format as input
    if output_4row:
        resp_layout = geom_layout.copy()
        resp_layout[1:4, :] = deformed_xyz.T
    else:
        resp_layout = deformed_xyz
    
    return resp_layout, scale


def osp_fisher_info_eiv_shm(
    num_sensors: int,
    mode_shapes: np.ndarray,
    cov_matrix: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimal sensor placement using Fisher Information Matrix and Effective Independence method.
    
    Determines optimal sensor locations by maximizing the determinant of the
    Fisher Information Matrix using the Effective Independence (EI) algorithm.
    
    .. meta::
        :category: Auxiliary - Sensor Support - Optimal Sensor Placement
        :matlab_equivalent: OSP_FisherInfoEIV_shm
        :complexity: Advanced
        :data_type: Modal Data
        :output_type: Sensor Locations
        :display_name: OSP Fisher Information EI
        :verbose_call: [Optimal DOF List, Determinant Q] = OSP Fisher Information EI (Number of Sensors, Mode Shapes, Covariance Matrix)
        
    Parameters
    ----------
    num_sensors : int
        Number of sensors to place.
        
        .. gui::
            :widget: number_input
            :min: 1
            :max: 1000
            :default: 12
            :description: Number of sensors
            
    mode_shapes : array_like, shape (n_dof, n_modes)
        Mode shape matrix with DOFs as rows and modes as columns.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: Mode shape matrix
            
    cov_matrix : array_like, shape (n_modes, n_modes), optional
        Modal covariance matrix. If None, identity matrix is used.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: Modal covariance matrix
            :required: false
            
    Returns
    -------
    op_list : array_like, shape (num_sensors,)
        Optimal DOF indices (1-based for MATLAB compatibility).
    det_q : array_like, shape (iterations,)
        Determinant of Fisher Information Matrix at each iteration.
        
    References
    ----------
    D. Kammer, "Sensor placement for on-orbit modal identification and
    correlation of large space structures," Journal of Guidance, Control, 
    and Dynamics, vol. 14, pp. 251-259, 1991.
    """
    n_dof, n_modes = mode_shapes.shape
    
    if num_sensors > n_dof:
        raise ValueError(f"Number of sensors ({num_sensors}) cannot exceed DOFs ({n_dof})")
    
    # Initialize covariance matrix if not provided
    if cov_matrix is None:
        cov_matrix = np.eye(n_modes)
    
    # Initialize with all DOFs as candidates
    candidate_set = np.arange(n_dof)
    selected_dofs = []
    
    # Effective Independence Method iterations
    phi = mode_shapes.copy()
    det_history = []
    
    # Iteratively remove DOFs with smallest contribution
    while len(candidate_set) > num_sensors:
        # Current mode shape matrix
        phi_c = phi[candidate_set, :]
        
        # Fisher Information Matrix with improved numerical stability
        q_matrix = phi_c.T @ phi_c
        
        # Add stronger regularization for numerical stability
        reg_term = 1e-8 * np.eye(n_modes)
        q_regularized = q_matrix + reg_term
        
        # Store determinant
        det_q = np.linalg.det(q_regularized)
        det_history.append(det_q)
        
        # Effective independence distribution with better error handling
        try:
            q_inv = np.linalg.inv(q_regularized)
            eid = np.diag(phi_c @ q_inv @ phi_c.T)
            
            # Check for numerical issues
            if np.any(~np.isfinite(eid)):
                raise np.linalg.LinAlgError("Non-finite values in EID calculation")
                
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix encountered, using pseudo-inverse with regularization")
            # Use pseudo-inverse with stronger regularization
            q_inv = np.linalg.pinv(q_matrix + 1e-6 * np.eye(n_modes))
            eid = np.diag(phi_c @ q_inv @ phi_c.T)
            
            # Replace any non-finite values with small positive values
            eid = np.where(np.isfinite(eid), eid, 1e-12)
        
        # Find DOF with minimum contribution
        min_idx = np.argmin(eid)
        
        # Remove from candidate set
        candidate_set = np.delete(candidate_set, min_idx)
    
    # Final Fisher Information Matrix
    phi_final = phi[candidate_set, :]
    q_final = phi_final.T @ phi_final
    det_final = np.linalg.det(q_final + 1e-8 * np.eye(n_modes))
    det_history.append(det_final)
    
    # Convert to 1-based indexing for MATLAB compatibility
    op_list = candidate_set + 1
    
    return op_list, np.array(det_history)


def get_sensor_layout_shm(
    op_list: np.ndarray,
    resp_dof: np.ndarray, 
    node_layout: np.ndarray
) -> np.ndarray:
    """
    Convert optimal DOF indices to sensor XYZ coordinates.
    
    .. meta::
        :category: Auxiliary - Sensor Support - Optimal Sensor Placement
        :matlab_equivalent: getSensorLayout_shm
        :complexity: Basic
        :data_type: Sensor Indices
        :output_type: Sensor Locations
        :display_name: Get Sensor Layout
        :verbose_call: [Sensor Layout] = Get Sensor Layout (Optimal DOF List, Response DOF, Node Layout)
        
    Parameters
    ----------
    op_list : array_like, shape (n_sensors,)
        Optimal DOF indices (1-based).
        
        .. gui::
            :widget: array_input
            :description: Optimal DOF indices
            
    resp_dof : array_like, shape (n_dof, 2) 
        DOF definitions [node_index, direction].
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: DOF to node mapping
            
    node_layout : array_like, shape (n_nodes, 3) or (4, n_nodes)
        Node coordinates [X, Y, Z]. If shape is (4, n_nodes),
        assumes first row is node indices and rows 2-4 are X, Y, Z coordinates.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            :description: Node coordinate matrix
            
    Returns
    -------
    sensor_layout : array_like, shape (n_sensors, 3)
        Sensor XYZ coordinates.
    """
    # Handle different input formats
    if node_layout.shape[0] == 4:
        # Format: [node_indices; X; Y; Z]
        node_xyz = node_layout[1:4, :].T  # Extract X,Y,Z and transpose
        n_nodes = node_layout.shape[1]
    else:
        # Format: [X, Y, Z] per row
        node_xyz = node_layout
        n_nodes = node_layout.shape[0]
    
    n_sensors = len(op_list)
    sensor_layout = np.zeros((n_sensors, 3))
    
    for i, dof_idx in enumerate(op_list):
        # Convert from 1-based to 0-based indexing
        dof_idx = int(dof_idx) - 1
        
        if 0 <= dof_idx < len(resp_dof):
            node_idx = int(resp_dof[dof_idx, 0]) - 1  # Convert to 0-based
            
            if 0 <= node_idx < n_nodes:
                sensor_layout[i, :] = node_xyz[node_idx, :]
    
    return sensor_layout


def osp_max_norm_shm(
    num_sensors: int,
    mode_shapes: np.ndarray,
    weights: np.ndarray,
    dualing_distance: float,
    resp_dof: np.ndarray,
    geom_layout: np.ndarray
) -> np.ndarray:
    """
    Optimal sensor placement using Maximum Norm method.
    
    Determines optimal sensor locations by maximizing the weighted norm
    of mode shapes with minimum separation constraints.
    
    .. meta::
        :category: Auxiliary - Sensor Support - Optimal Sensor Placement
        :matlab_equivalent: OSP_MaxNorm_shm
        :complexity: Advanced
        :data_type: Modal Data
        :output_type: Sensor Locations
        :display_name: OSP Maximum Norm
        :verbose_call: [Optimal DOF List] = OSP Maximum Norm (Number of Sensors, Mode Shapes, Weights, Dualing Distance, Response DOF, Geometry Layout)
        
    Parameters
    ----------
    num_sensors : int
        Number of sensors to place.
        
        .. gui::
            :widget: number_input
            :min: 1
            :max: 1000
            :default: 12
            
    mode_shapes : array_like, shape (n_dof, n_modes)
        Mode shape matrix.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            
    weights : array_like, shape (n_modes,)
        Mode importance weights.
        
        .. gui::
            :widget: array_input
            :description: Mode weights (e.g., [13, 12, 11, ...])
            
    dualing_distance : float
        Minimum separation distance between sensors.
        
        .. gui::
            :widget: number_input
            :min: 0.0
            :default: 20.0
            :description: Minimum sensor separation
            
    resp_dof : array_like, shape (n_dof, 2)
        DOF definitions [node_index, direction].
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            
    geom_layout : array_like, shape (n_nodes, 3) or (4, n_nodes)
        Node coordinates for distance calculations. If shape is (4, n_nodes),
        assumes first row is node indices and rows 2-4 are X, Y, Z coordinates.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat", ".npy"]
            
    Returns
    -------
    op_list : array_like, shape (num_sensors,)
        Optimal DOF indices (1-based).
        
    References
    ----------
    M. Meo and G. Zumpano, "On the optimal sensor placement techniques
    for a bridge structure," Engineering Structures, vol. 27, pp. 1488-1497, 2005.
    """
    n_dof, n_modes = mode_shapes.shape
    
    if len(weights) != n_modes:
        raise ValueError(f"Number of weights ({len(weights)}) must match modes ({n_modes})")
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    # Calculate weighted norm for each DOF
    weighted_norm = np.zeros(n_dof)
    for i in range(n_dof):
        weighted_norm[i] = np.sum(weights * mode_shapes[i, :]**2)
    
    # Handle different input formats for geometry
    if geom_layout.shape[0] == 4:
        # Format: [node_indices; X; Y; Z]
        node_xyz = geom_layout[1:4, :].T  # Extract X,Y,Z and transpose
        n_nodes = geom_layout.shape[1]
    else:
        # Format: [X, Y, Z] per row
        node_xyz = geom_layout
        n_nodes = geom_layout.shape[0]
    
    # Get node coordinates for each DOF
    dof_coords = np.zeros((n_dof, 3))
    for i in range(n_dof):
        node_idx = int(resp_dof[i, 0]) - 1  # Convert to 0-based
        if 0 <= node_idx < n_nodes:
            dof_coords[i, :] = node_xyz[node_idx, :]
    
    # Greedy selection with minimum separation constraint
    selected_dofs = []
    available_dofs = list(range(n_dof))
    
    while len(selected_dofs) < num_sensors and available_dofs:
        # Sort by weighted norm
        sorted_indices = sorted(available_dofs, key=lambda x: weighted_norm[x], reverse=True)
        
        # Find best DOF that satisfies separation constraint
        selected = False
        for dof_idx in sorted_indices:
            # Check separation from already selected sensors
            valid = True
            for sel_dof in selected_dofs:
                dist = np.linalg.norm(dof_coords[dof_idx] - dof_coords[sel_dof])
                if dist < dualing_distance:
                    valid = False
                    break
            
            if valid:
                selected_dofs.append(dof_idx)
                available_dofs.remove(dof_idx)
                selected = True
                break
        
        # If no valid DOF found, select best remaining (relaxing constraint)
        if not selected and available_dofs:
            best_dof = sorted_indices[0]
            selected_dofs.append(best_dof)
            available_dofs.remove(best_dof)
            warnings.warn(f"Relaxed separation constraint for sensor {len(selected_dofs)}")
    
    # Convert to 1-based indexing
    op_list = np.array(selected_dofs) + 1
    
    return op_list


# Placeholder visualization functions - these would typically use matplotlib
def node_element_plot_shm(
    node_layout: np.ndarray,
    elements: np.ndarray,
    resp_scale: Optional[float] = None,
    ax_handle: Optional[Any] = None
) -> Any:
    """
    Plot structural geometry with optional deformed shape.
    
    .. meta::
        :category: Auxiliary - Plotting
        :matlab_equivalent: nodeElementPlot_shm
        :complexity: Basic
        :data_type: Geometry Data
        :output_type: Plot
        :display_name: Node Element Plot
        :verbose_call: [Axis Handle] = Node Element Plot (Node Layout, Elements, Response Scale, Axis Handle)
    """
    # This is a placeholder - actual implementation would create matplotlib plot
    print("Plotting nodes and elements...")
    return None


def plot_sensors_shm(
    sensor_layout: np.ndarray,
    ax_handle: Optional[Any] = None
) -> None:
    """
    Plot sensor locations on structure.
    
    .. meta::
        :category: Auxiliary - Plotting
        :matlab_equivalent: plotSensors_shm
        :complexity: Basic
        :data_type: Sensor Locations
        :output_type: Plot
        :display_name: Plot Sensors
        :verbose_call: Plot Sensors (Sensor Layout, Axis Handle)
    """
    # This is a placeholder - actual implementation would add sensors to plot
    print(f"Plotting {len(sensor_layout)} sensors...")