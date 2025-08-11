"""
LADPackage-specific plotting functions for optimal sensor placement.

This module provides plotting utilities for visualizing modal analysis results,
sensor layouts, and optimal sensor placement configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Union, Any
import sys
from pathlib import Path

# Add project root to path to access shmtools
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from shmtools.modal.osp import response_interp_shm


def _plot_element_edges(ax, elements, x_coords, y_coords, z_coords, displacement_scale):
    """
    Plot finite element edges following MATLAB nodeElementPlot_shm logic.
    
    Based on MATLAB analysis:
    - elements[0,:] = element IDs
    - elements[1:,:] = 8 node indices per element (8-node hexahedra)
    - MATLAB draws 6 faces per hexahedron using specific connectivity
    
    8-node hexahedron face connectivity (MATLAB ilist for elType=8):
    Face 1: [1 2 3 4] (nodes 1-4, bottom face)
    Face 2: [5 6 7 8] (nodes 5-8, top face)  
    Face 3: [1 2 6 5] (nodes 1,2,6,5, front face)
    Face 4: [3 7 8 4] (nodes 3,7,8,4, back face)
    Face 5: [1 5 8 4] (nodes 1,5,8,4, left face)
    Face 6: [2 6 7 3] (nodes 2,6,7,3, right face)
    """
    if elements.size == 0:
        return
        
    n_nodes_total = len(x_coords)
    
    # MATLAB 8-node hexahedron face definitions (0-based indexing)
    hex_faces = [
        [0, 1, 2, 3],  # Bottom face: nodes 1-2-3-4
        [4, 5, 6, 7],  # Top face: nodes 5-6-7-8
        [0, 1, 5, 4],  # Front face: nodes 1-2-6-5  
        [2, 6, 7, 3],  # Back face: nodes 3-7-8-4
        [0, 4, 7, 3],  # Left face: nodes 1-5-8-4
        [1, 5, 6, 2],  # Right face: nodes 2-6-7-3
    ]
    
    for elem_idx in range(elements.shape[1]):
        # Extract element nodes (skip first row which is element ID)
        elem_nodes = elements[1:, elem_idx].astype(int) - 1  # Convert to 0-based
        
        # Check if all nodes are valid
        if len(elem_nodes) == 8 and np.all(elem_nodes >= 0) and np.all(elem_nodes < n_nodes_total):
            # Draw each face of the hexahedron
            for face in hex_faces:
                # Get the 4 nodes that define this face
                face_nodes = [elem_nodes[i] for i in face]
                
                # Draw the face edges: 0->1->2->3->0
                for i in range(4):
                    n1 = face_nodes[i]
                    n2 = face_nodes[(i + 1) % 4]  # Next node (wrap around)
                    
                    # Plot edge
                    ax.plot([x_coords[n1], x_coords[n2]], 
                           [y_coords[n1], y_coords[n2]], 
                           [z_coords[n1], z_coords[n2]], 
                           'k-', alpha=0.4, linewidth=0.5)


def plot_nodal_response(
    geometry_layout: np.ndarray,
    mode_shapes: np.ndarray, 
    resp_dof: np.ndarray,
    elements: np.ndarray,
    mode_number: int
) -> plt.Axes:
    """
    Plot nodal response with element mesh for a specific mode.
    
    LADPackage wrapper: Node-element plot with response values
    
    .. meta::
        :category: Plotting - Optimal Sensor Placement
        :matlab_equivalent: PlotNodalResponse
        :complexity: Intermediate
        :data_type: Modal Data
        :output_type: 3D Plot
        :display_name: Plot Nodal Response
        :verbose_call: [Axes Handle] = Plot Nodal Response (Geometry Layout, Mode Shapes, Response DOF, Elements, Mode Number)
    
    Parameters
    ----------
    geometry_layout : ndarray, shape (3, num_nodes)
        Node coordinates array where each column is [x, y, z] for one node
        
        .. gui::
            :widget: data_input
            :description: Node geometry layout (3 x num_nodes)
            
    mode_shapes : ndarray, shape (num_dofs, num_modes)
        Mode shape matrix with DOF responses for each mode
        
        .. gui::
            :widget: data_input
            :description: Mode shapes matrix
            
    resp_dof : ndarray, shape (num_dofs, 2) 
        DOF definitions [node_id, direction] for each DOF
        
        .. gui::
            :widget: data_input
            :description: Response DOF definitions
            
    elements : ndarray, shape (num_nodes_per_elem, num_elements)
        Element connectivity matrix
        
        .. gui::
            :widget: data_input
            :description: Element connectivity
            
    mode_number : int
        Mode number to plot (1-based indexing)
        
        .. gui::
            :widget: number_input
            :min: 1
            :description: Mode number to visualize
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes handle for the 3D plot
    
    Notes
    -----
    MATLAB Compatibility: This function provides the same interface as the 
    original LADPackage PlotNodalResponse function. It creates a 3D surface
    plot showing modal response amplitudes across the structure geometry.
    
    Examples
    --------
    >>> # Load modal data
    >>> node_layout, elements, mode_shapes, resp_dof = import_modal_osp_shm()
    >>> 
    >>> # Plot mode 3 response
    >>> ax = plot_nodal_response(node_layout, mode_shapes, resp_dof, elements, 3)
    >>> plt.title('Mode 3 Response')
    >>> plt.show()
    """
    # Convert to 0-based indexing
    mode_idx = mode_number - 1
    
    if mode_idx >= mode_shapes.shape[1] or mode_idx < 0:
        raise ValueError(f"Mode number {mode_number} is out of range (1-{mode_shapes.shape[1]})")
    
    # Extract mode shape for specified mode
    mode_vector = mode_shapes[:, mode_idx]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get node positions - note: row 0 is node IDs, actual coords are rows 1,2,3
    if geometry_layout.shape[0] == 4:
        # Format: [node_ids, x, y, z]
        x_nodes = geometry_layout[1, :]  # X coordinates
        y_nodes = geometry_layout[2, :]  # Y coordinates  
        z_nodes = geometry_layout[3, :]  # Z coordinates
    else:
        # Format: [x, y, z] - standard case
        x_nodes = geometry_layout[0, :]
        y_nodes = geometry_layout[1, :]
        z_nodes = geometry_layout[2, :]
    
    # Interpolate response to node locations and apply displacement
    try:
        response_nodes = response_interp_shm(geometry_layout, mode_vector, resp_dof)
        
        # Calculate displacement magnitude for color mapping
        displacement_magnitude = np.sqrt(np.sum(response_nodes**2, axis=1))
        
        # Apply displacement to node positions (scaled for visibility)
        displacement_scale = 10.0  # Scale factor for visual effect
        x_displaced = x_nodes + displacement_scale * response_nodes[:, 0]
        y_displaced = y_nodes + displacement_scale * response_nodes[:, 1] 
        z_displaced = z_nodes + displacement_scale * response_nodes[:, 2]
        
    except Exception as e:
        print(f"Warning: Response interpolation failed ({e}), using undeformed geometry")
        # Fallback: simple mapping for basic visualization
        response_nodes = np.zeros((geometry_layout.shape[1], 3))
        displacement_magnitude = np.zeros(geometry_layout.shape[1])
        for i, (node_id, direction) in enumerate(resp_dof):
            node_idx = int(node_id) - 1  # Convert to 0-based
            if 0 <= node_idx < len(displacement_magnitude):
                displacement_magnitude[node_idx] += abs(mode_vector[i])
                
        x_displaced = x_nodes
        y_displaced = y_nodes  
        z_displaced = z_nodes
        displacement_scale = 0.0
    
    # Create scatter plot with color mapping based on displacement magnitude
    scatter = ax.scatter(x_displaced, y_displaced, z_displaced, 
                        c=displacement_magnitude, 
                        cmap='RdYlBu_r',  # Red-Yellow-Blue colormap
                        s=60, 
                        alpha=0.8,
                        edgecolors='black',
                        linewidth=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, 
                label=f'Mode {mode_number} Response Magnitude')
    
    # Plot mesh edges if elements are available
    if elements.size > 0:
        _plot_element_edges(ax, elements, x_displaced, y_displaced, z_displaced, displacement_scale)
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position') 
    ax.set_zlabel('Z Position')
    ax.set_title(f'Mode {mode_number} Nodal Response')
    
    # Set equal aspect ratio
    max_range = np.array([x_nodes.max()-x_nodes.min(), 
                         y_nodes.max()-y_nodes.min(), 
                         z_nodes.max()-z_nodes.min()]).max() / 2.0
    mid_x = (x_nodes.max()+x_nodes.min()) * 0.5
    mid_y = (y_nodes.max()+y_nodes.min()) * 0.5
    mid_z = (z_nodes.max()+z_nodes.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax


def plot_sensors_with_mesh(
    elements: np.ndarray,
    nodes: np.ndarray, 
    sensor_indices: np.ndarray,
    resp_dof: np.ndarray
) -> Tuple[plt.Axes, plt.Axes, Any]:
    """
    Plot sensor locations overlaid on structural mesh.
    
    LADPackage wrapper: Node-element plot with sensors
    
    .. meta::
        :category: Plotting - Optimal Sensor Placement
        :matlab_equivalent: PlotSensorsWithMesh
        :complexity: Intermediate
        :data_type: Modal Data
        :output_type: 3D Plot
        :display_name: Plot Sensors with Mesh
        :verbose_call: [Axes Handle, Axes Handle, Sensor Handle] = Plot Sensors with Mesh (Elements, Nodes, Sensor Indices, Response DOF)
    
    Parameters
    ----------
    elements : ndarray, shape (nodes_per_elem, num_elements)
        Element connectivity matrix defining mesh topology
        
        .. gui::
            :widget: data_input
            :description: Element connectivity matrix
            
    nodes : ndarray, shape (3, num_nodes)
        Node coordinate array [x, y, z] for each node
        
        .. gui::
            :widget: data_input
            :description: Node geometry layout
            
    sensor_indices : ndarray, shape (num_sensors,)
        DOF indices where sensors should be placed
        
        .. gui::
            :widget: data_input
            :description: Optimal sensor DOF indices
            
    resp_dof : ndarray, shape (num_dofs, 2)
        DOF definitions [node_id, direction] for each DOF
        
        .. gui::
            :widget: data_input
            :description: Response DOF definitions
    
    Returns
    -------
    ax1 : matplotlib.axes.Axes
        Primary axes handle for 3D mesh plot
    ax2 : matplotlib.axes.Axes
        Secondary axes handle (same as ax1 for compatibility)
    sensor_handle : matplotlib collection
        Handle to sensor scatter plot for legend/styling
    
    Notes
    -----
    MATLAB Compatibility: This function replicates the interface of the original
    LADPackage PlotSensorsWithMesh function. It creates a 3D wireframe mesh
    with optimal sensor locations highlighted.
    
    Examples
    --------
    >>> # Load data and compute optimal sensor placement
    >>> node_layout, elements, mode_shapes, resp_dof = import_modal_osp_shm()
    >>> optimal_dofs, det_q = osp_fisher_info_eiv_shm(12, mode_shapes)
    >>> 
    >>> # Plot sensors on mesh
    >>> ax1, ax2, sensors = plot_sensors_with_mesh(elements, node_layout, 
    ...                                           optimal_dofs, resp_dof)
    >>> plt.title('Optimal Sensor Placement')
    >>> plt.show()
    """
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get node positions - note: row 0 is node IDs, actual coords are rows 1,2,3
    if nodes.shape[0] == 4:
        # Format: [node_ids, x, y, z]
        x_nodes = nodes[1, :]  # X coordinates
        y_nodes = nodes[2, :]  # Y coordinates
        z_nodes = nodes[3, :]  # Z coordinates
    else:
        # Format: [x, y, z] - standard case
        x_nodes = nodes[0, :]
        y_nodes = nodes[1, :]
        z_nodes = nodes[2, :]
    
    # Plot mesh structure with proper element connectivity
    if elements.size > 0:
        _plot_element_edges(ax, elements, x_nodes, y_nodes, z_nodes, 0.0)
    
    # Plot all nodes as small dots
    ax.scatter(x_nodes, y_nodes, z_nodes, 
              c='lightgray', s=10, alpha=0.6, label='Nodes')
    
    # Determine sensor locations from DOF indices
    sensor_x, sensor_y, sensor_z = [], [], []
    
    for sensor_dof in sensor_indices:
        # Find corresponding node for this DOF (0-based indexing)
        dof_idx = int(sensor_dof) - 1  # Convert to 0-based
        if 0 <= dof_idx < len(resp_dof):
            node_id = int(resp_dof[dof_idx, 0]) - 1  # Convert to 0-based
            if 0 <= node_id < len(x_nodes):
                sensor_x.append(x_nodes[node_id])
                sensor_y.append(y_nodes[node_id])
                sensor_z.append(z_nodes[node_id])
    
    # Plot optimal sensor locations
    sensor_handle = ax.scatter(sensor_x, sensor_y, sensor_z,
                              c='red', s=100, marker='o',
                              edgecolors='black', linewidth=2,
                              label=f'Optimal Sensors ({len(sensor_indices)})',
                              alpha=0.9)
    
    # Add sensor number labels
    for i, (x, y, z) in enumerate(zip(sensor_x, sensor_y, sensor_z)):
        ax.text(x, y, z, f'  {i+1}', fontsize=10, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position') 
    ax.set_title(f'Optimal Sensor Placement ({len(sensor_indices)} sensors)')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([x_nodes.max()-x_nodes.min(), 
                         y_nodes.max()-y_nodes.min(), 
                         z_nodes.max()-z_nodes.min()]).max() / 2.0
    mid_x = (x_nodes.max()+x_nodes.min()) * 0.5
    mid_y = (y_nodes.max()+y_nodes.min()) * 0.5
    mid_z = (z_nodes.max()+z_nodes.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Return axes handles for MATLAB compatibility
    return ax, ax, sensor_handle