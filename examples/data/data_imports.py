"""
MATLAB-compatible data import functions for SHMTools example datasets.

These functions replicate the exact behavior and signatures of the original
MATLAB import functions from matlab/SHMTools/Examples/ExampleData/
"""

import numpy as np
import scipy.io
from pathlib import Path
from typing import Tuple, Dict


def _get_data_path(filename: str) -> Path:
    """Get path to data file in the data_files subdirectory."""
    current_dir = Path(__file__).parent
    return current_dir / "data_files" / filename


def import_3story_structure_shm() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Import 3-story structure experimental data.
    
    Loads experimental data from a 3-story structure with various damage states.
    The dataset contains vibration measurements from 5 channels across 170 test
    instances, including both undamaged and damaged conditions.
    
    .. meta::
        :category: Data Import
        :matlab_equivalent: import_3StoryStructure_shm
        :complexity: Basic
        :data_type: Vibration Data
        :output_type: Dataset
        
    Returns
    -------
    dataset : np.ndarray
        Shape (8192, 5, 170). Time series observations from 5 channels for 170 trials.
        
        .. gui::
            :widget: data_display
            :display_type: 3d_array
            :preview_dims: [time, channels, instances]
            
    damage_states : np.ndarray  
        Shape (170, 1). Binary classification vector of known damage states
        (0=undamaged, 1=damaged). First 90 instances are undamaged, last 80 are damaged.
        
        .. gui::
            :widget: data_display
            :display_type: binary_vector
            :labels: ["Undamaged", "Damaged"]
            
    state_list : np.ndarray
        Shape (1, 170). Detailed state identifier for each instance (1-17).
        
        .. gui::
            :widget: data_display
            :display_type: state_vector
            :range: [1, 17]
            
    Examples
    --------
    >>> dataset, damage_states, state_list = import_3story_structure_shm()
    >>> print(f"Dataset shape: {dataset.shape}")
    >>> print(f"Number of undamaged: {np.sum(damage_states == 0)}")
    >>> print(f"Number of damaged: {np.sum(damage_states == 1)}")
    
    Notes
    -----
    The data file 'data3SS.mat' must be present in the same directory as this module.
    This dataset is commonly used for testing outlier detection and classification
    algorithms in structural health monitoring applications.
    
    References
    ----------
    .. [1] Los Alamos National Laboratory, "Experimental Data from a 3-Story Structure",
           LA-CC-14-046, 2014.
           
    Raises
    ------
    FileNotFoundError
        If data3SS.mat cannot be found in the data directory.
    """
    data_file = _get_data_path("data3SS.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data3SS.mat at {data_file}. "
            f"Please download the file and place it in the examples/data directory."
        )
    
    # Load the MATLAB file
    mat_data = scipy.io.loadmat(str(data_file))
    
    # Extract dataset and states (following MATLAB exactly)
    dataset = mat_data["dataset"]  # Shape: (8192, 5, 170)
    states = mat_data["states"].flatten()  # Convert to 1D array
    
    # Create state_list as row vector (1, INSTANCES) - MATLAB: states'
    state_list = states.reshape(1, -1)
    
    # Create damage_states as column vector (INSTANCES, 1)
    # MATLAB: logical([zeros(90,1);ones(80,1)])
    damage_states = np.concatenate([
        np.zeros(90, dtype=bool),
        np.ones(80, dtype=bool)
    ]).reshape(-1, 1).astype(np.float64)
    
    return dataset, damage_states, state_list


def import_cbm_data_shm() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Import condition-based monitoring experimental data.
    
    Loads vibration and tachometer data from rotating machinery for
    condition-based monitoring applications. Includes both healthy and
    faulty bearing conditions.
    
    .. meta::
        :category: Data Import
        :matlab_equivalent: import_CBMData_shm
        :complexity: Intermediate
        :data_type: Vibration Data
        :output_type: Dataset
        
    Returns
    -------
    dataset : np.ndarray
        Shape (TIME, 4, INSTANCES). Multi-channel time series data where
        channel 1 is tachometer and channels 2-4 are accelerometers.
        
        .. gui::
            :widget: data_display
            :display_type: 3d_array
            :channel_labels: ["Tachometer", "Accel X", "Accel Y", "Accel Z"]
            
    damage_states : np.ndarray
        Shape (INSTANCES, 1). Binary classification vector
        (0=healthy bearing, 1=faulty bearing).
        
        .. gui::
            :widget: data_display
            :display_type: binary_vector
            :labels: ["Healthy", "Faulty"]
            
    state_list : np.ndarray
        Shape (INSTANCES, 1). Detailed condition state for each instance.
        
        .. gui::
            :widget: data_display
            :display_type: state_vector
            
    fs : float
        Sampling frequency in Hz.
        
        .. gui::
            :widget: number_display
            :units: Hz
            :format: .1f
            
    Examples
    --------
    >>> dataset, damage_states, state_list, fs = import_cbm_data_shm()
    >>> print(f"Sampling rate: {fs} Hz")
    >>> print(f"Recording duration: {dataset.shape[0]/fs:.2f} seconds")
    >>> print(f"Number of channels: {dataset.shape[1]}")
    
    Notes
    -----
    The data file 'data_CBM.mat' must be present in the same directory.
    This dataset is ideal for testing bearing fault detection algorithms
    and condition-based monitoring approaches.
    
    References
    ----------
    .. [1] Robinson, L., "Condition-Based Monitoring Dataset", 
           Los Alamos National Laboratory, 2013.
           
    Raises
    ------
    FileNotFoundError
        If data_CBM.mat cannot be found in the data directory.
    """
    data_file = _get_data_path("data_CBM.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data_CBM.mat at {data_file}. "
            f"Please download the file and place it in the examples/data directory."
        )
    
    # Load the MATLAB file
    mat_data = scipy.io.loadmat(str(data_file))
    
    # Extract variables (names from MATLAB file)
    dataset = mat_data["dataset"]
    damage_states = mat_data["damageStates"]
    state_list = mat_data["stateList"]
    fs = float(mat_data["Fs"][0, 0])  # Extract scalar from array
    
    return dataset, damage_states, state_list, fs


def import_active_sense1_shm() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        Dict, float, np.ndarray, np.ndarray]:
    """
    Import active sensing experimental dataset #1.
    
    Loads guided wave propagation data from a plate structure with 32 piezoelectric
    transducers. The test structure was a 0.01 inch concave-shaped plate approximately
    48 inches on one side. Damage was simulated using a 2-inch neodymium magnet.
    
    .. meta::
        :category: Data Import
        :matlab_equivalent: import_ActiveSense1_shm
        :complexity: Advanced
        :data_type: Guided Waves
        :output_type: Dataset
        
    Returns
    -------
    waveform_base : np.ndarray
        Shape (TIME, SENSORPAIRS). Baseline waveforms acquired before damage.
        
        .. gui::
            :widget: data_display
            :display_type: waveform_matrix
            :label: Baseline Waveforms
            
    waveform_test : np.ndarray
        Shape (TIME, SENSORPAIRS). Test waveforms acquired after damage.
        
        .. gui::
            :widget: data_display
            :display_type: waveform_matrix
            :label: Damaged Waveforms
            
    sensor_layout : np.ndarray
        Shape (3, SENSORS). Sensor IDs and coordinates [ID, X, Y].
        
        .. gui::
            :widget: scatter_plot
            :x_col: 1
            :y_col: 2
            :label_col: 0
            
    pair_list : np.ndarray
        Shape (2, PAIRS). Actuator-sensor pair definitions [actuatorID, sensorID].
        
        .. gui::
            :widget: data_display
            :display_type: pair_matrix
            
    border_struct : dict
        Plate border definitions with 'outside' and 'inside' fields.
        
        .. gui::
            :widget: geometry_display
            :display_type: boundary_lines
            
    sample_rate : float
        Sampling rate in points/second.
        
        .. gui::
            :widget: number_display
            :units: Hz
            
    actuation_waveform : np.ndarray
        Shape (TIME, 1). Gaussian windowed sinusoid used for actuation.
        
        .. gui::
            :widget: waveform_plot
            :label: Actuation Signal
            
    damage_location : np.ndarray
        Shape (2, 1). Damage coordinates [X, Y].
        
        .. gui::
            :widget: point_marker
            :color: red
            :label: Damage Location
            
    Examples
    --------
    >>> (waveform_base, waveform_test, sensor_layout, pair_list,
    ...  border_struct, sample_rate, actuation_waveform, 
    ...  damage_location) = import_active_sense1_shm()
    >>> print(f"Number of sensor pairs: {pair_list.shape[1]}")
    >>> print(f"Sampling rate: {sample_rate} Hz")
    >>> print(f"Damage at: ({damage_location[0,0]:.1f}, {damage_location[1,0]:.1f})")
    
    Notes
    -----
    The data file 'data_example_ActiveSense.mat' must be present.
    The acquisition system cycled through actuator-sensor pairs one at a time,
    inducing the actuation waveform and sensing the propagated wave.
    
    References
    ----------
    .. [1] Flynn, E., "Active Sensing Dataset for Guided Wave Analysis",
           LA-CC-14-046, Los Alamos National Laboratory, 2010.
           
    Raises
    ------
    FileNotFoundError
        If data_example_ActiveSense.mat cannot be found.
    """
    data_file = _get_data_path("data_example_ActiveSense.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data_example_ActiveSense.mat at {data_file}. "
            f"Please download the file and place it in the examples/data directory."
        )
    
    # Load specific variables (following MATLAB exactly)
    mat_data = scipy.io.loadmat(
        str(data_file),
        variable_names=[
            "waveformBase", "waveformTest", "sensorLayout", "pairList",
            "borderStruct", "sampleRate", "actuationWaveform", "damageLocation"
        ]
    )
    
    waveform_base = mat_data["waveformBase"]
    waveform_test = mat_data["waveformTest"]
    sensor_layout = mat_data["sensorLayout"]
    pair_list = mat_data["pairList"]
    border_struct_raw = mat_data["borderStruct"]
    sample_rate = float(mat_data["sampleRate"][0, 0])
    actuation_waveform = mat_data["actuationWaveform"]
    damage_location = mat_data["damageLocation"]
    
    # Convert borderStruct from MATLAB struct to Python dict
    border_struct = {}
    if border_struct_raw.dtype.names:
        for field_name in border_struct_raw.dtype.names:
            border_struct[field_name] = border_struct_raw[field_name][0, 0]
    
    return (waveform_base, waveform_test, sensor_layout, pair_list,
            border_struct, sample_rate, actuation_waveform, damage_location)


def import_sensor_diagnostic_shm() -> Tuple[np.ndarray, np.ndarray]:
    """
    Import sensor diagnostic dataset for health assessment.
    
    Loads electrical impedance (susceptance) measurements from piezoelectric
    sensors in various health states including healthy, debonded, and broken conditions.
    
    .. meta::
        :category: Data Import
        :matlab_equivalent: import_SensorDiagnostic_shm
        :complexity: Intermediate
        :data_type: Impedance Data
        :output_type: Dataset
        
    Returns
    -------
    healthy_data : np.ndarray
        Shape (FREQUENCY, SENSORS). Susceptance data from healthy sensors only.
        First column contains frequency values in Hz.
        
        .. gui::
            :widget: spectrum_plot
            :x_col: 0
            :y_cols: [1, 2, 3, ...]
            :x_label: Frequency (Hz)
            :y_label: Susceptance
            :label: Healthy Sensors
            
    example_sensor_data : np.ndarray
        Shape (FREQUENCY, SENSORS). Susceptance data from sensors in various
        health states (healthy, debonded, broken). First column contains frequencies.
        
        .. gui::
            :widget: spectrum_plot
            :x_col: 0
            :y_cols: [1, 2, 3, ...]
            :x_label: Frequency (Hz)
            :y_label: Susceptance
            :label: Mixed Health States
            :line_styles: ["solid", "dashed", "dotted"]
            
    Examples
    --------
    >>> healthy_data, example_sensor_data = import_sensor_diagnostic_shm()
    >>> frequencies = healthy_data[:, 0]
    >>> print(f"Frequency range: {frequencies[0]:.0f} - {frequencies[-1]:.0f} Hz")
    >>> print(f"Number of healthy sensors: {healthy_data.shape[1] - 1}")
    >>> print(f"Number of test sensors: {example_sensor_data.shape[1] - 1}")
    
    Notes
    -----
    The data file 'dataSensorDiagnostic.mat' must be present.
    This dataset is useful for developing and testing sensor self-diagnostic
    algorithms based on electromechanical impedance measurements.
    
    References
    ----------
    .. [1] Harvey, D., "Sensor Diagnostic Dataset using Impedance Measurements",
           LA-CC-14-046, Los Alamos National Laboratory, 2010.
           
    Raises
    ------
    FileNotFoundError
        If dataSensorDiagnostic.mat cannot be found.
    """
    data_file = _get_data_path("dataSensorDiagnostic.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find dataSensorDiagnostic.mat at {data_file}. "
            f"Please download the file and place it in the examples/data directory."
        )
    
    # Load specific variables (following MATLAB exactly)
    mat_data = scipy.io.loadmat(
        str(data_file),
        variable_names=["sd_ex", "sd_ex_broken"]
    )
    
    # Following MATLAB exactly:
    # healthyData = sd_ex;
    # exampleSensorData = sd_ex_broken;
    healthy_data = mat_data["sd_ex"]
    example_sensor_data = mat_data["sd_ex_broken"]
    
    return healthy_data, example_sensor_data


def import_modal_osp_shm() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Import modal analysis data for optimal sensor placement studies.
    
    Loads finite element model data including node layouts, element connectivity,
    mode shapes, and degree-of-freedom definitions for structural dynamics analysis.
    
    .. meta::
        :category: Data Import
        :matlab_equivalent: import_ModalOSP_shm
        :complexity: Advanced
        :data_type: Modal Data
        :output_type: Dataset
        
    Returns
    -------
    node_layout : np.ndarray
        Shape (4, NODES). Node information [ID, X, Y, Z] for each node.
        
        .. gui::
            :widget: 3d_scatter
            :x_col: 1
            :y_col: 2
            :z_col: 3
            :label_col: 0
            :label: FEM Nodes
            
    elements : np.ndarray
        Shape (NODES_PER_ELEMENT, ELEMENTS). Node indices defining each element.
        
        .. gui::
            :widget: mesh_display
            :node_data: node_layout
            :connectivity: elements
            :display_type: wireframe
            
    mode_shapes : np.ndarray
        Shape (DOFS, MODES). Modal displacement vectors for each mode.
        
        .. gui::
            :widget: mode_animator
            :node_data: node_layout
            :dof_data: resp_dof
            :animation_speed: 1.0
            
    resp_dof : np.ndarray
        Shape (DOFS, 2). DOF definitions [nodeID, direction].
        Direction codes: 1=X, 2=Y, 3=Z, 4=YZ, 5=XZ, 6=XY.
        
        .. gui::
            :widget: data_display
            :display_type: dof_table
            :headers: ["Node ID", "Direction"]
            :direction_labels: {1: "X", 2: "Y", 3: "Z", 4: "YZ", 5: "XZ", 6: "XY"}
            
    Examples
    --------
    >>> node_layout, elements, mode_shapes, resp_dof = import_modal_osp_shm()
    >>> print(f"Number of nodes: {node_layout.shape[1]}")
    >>> print(f"Number of elements: {elements.shape[1]}")
    >>> print(f"Number of modes: {mode_shapes.shape[1]}")
    >>> print(f"Number of DOFs: {resp_dof.shape[0]}")
    
    Notes
    -----
    The data file 'data_OSPExampleModal.mat' must be present.
    This dataset is designed for testing optimal sensor placement algorithms
    and modal analysis techniques in structural dynamics.
    
    References
    ----------
    .. [1] Harvey, D., "Modal Analysis Dataset for Optimal Sensor Placement",
           LA-CC-14-046, Los Alamos National Laboratory, 2010.
           
    Raises
    ------
    FileNotFoundError
        If data_OSPExampleModal.mat cannot be found.
    """
    data_file = _get_data_path("data_OSPExampleModal.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data_OSPExampleModal.mat at {data_file}. "
            f"Please download the file and place it in the examples/data directory."
        )
    
    # Load all variables (following MATLAB exactly)
    mat_data = scipy.io.loadmat(str(data_file))
    
    # Extract the required variables
    node_layout = mat_data["nodeLayout"]
    elements = mat_data["elements"]
    mode_shapes = mat_data["modeShapes"]
    resp_dof = mat_data["respDOF"]
    
    return node_layout, elements, mode_shapes, resp_dof