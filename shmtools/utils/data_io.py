"""
MATLAB-compatible data import functions for SHMTools example datasets.

These functions replicate the exact behavior and signatures of the original 
MATLAB import functions from shmtool-matlab/SHMTools/Examples/ExampleData/
"""

import numpy as np
import scipy.io
from pathlib import Path
from typing import Tuple, Dict, Any, List
import os


def _get_example_data_path(filename: str) -> Path:
    """Get path to example data file, searching multiple possible locations."""
    # Get current file directory
    current_dir = Path(__file__).parent
    
    # Try different possible locations
    search_paths = [
        # From shmtools/utils/ -> examples/data/
        current_dir.parent.parent / "examples" / "data" / filename,
        # From shmtools/utils/ -> ../../shmtool-matlab/SHMTools/Examples/ExampleData/
        current_dir.parent.parent.parent / "shmtool-matlab" / "SHMTools" / "Examples" / "ExampleData" / filename,
        # From current working directory
        Path.cwd() / "examples" / "data" / filename,
        Path.cwd() / filename,
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    # If not found, return the preferred location for error message
    return current_dir.parent.parent / "examples" / "data" / filename


def import_3StoryStructure_shm() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Data Import: Import 3 story structure data
    
    Replicates import_3StoryStructure_shm.m exactly.
    
    Returns
    -------
    dataset : np.ndarray
        Shape (TIME, CHANNELS, INSTANCES). Observations from C channels for T trials.
    damageStates : np.ndarray  
        Shape (INSTANCES, 1). Binary classification vector of known damage states 
        (0-undamaged and 1-damaged).
    stateList : np.ndarray
        Shape (1, INSTANCES). State for each instance.
        
    Raises
    ------
    FileNotFoundError
        If data3SS.mat cannot be found.
    """
    data_file = _get_example_data_path("data3SS.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data3SS.mat at {data_file}. "
            f"Please ensure the file exists in the examples/data directory."
        )
    
    # Load the MATLAB file
    mat_data = scipy.io.loadmat(str(data_file))
    
    # Extract dataset and states (following MATLAB exactly)
    dataset = mat_data['dataset']  # Shape should be (8192, 5, 170)
    states = mat_data['states'].flatten()  # Convert to 1D array
    
    # Create stateList as row vector (1, INSTANCES) - MATLAB: states'
    stateList = states.reshape(1, -1)
    
    # Create damageStates as column vector (INSTANCES, 1) - MATLAB: logical([zeros(90,1);ones(80,1)])
    damageStates = np.concatenate([
        np.zeros(90, dtype=bool),
        np.ones(80, dtype=bool)
    ]).reshape(-1, 1)
    
    return dataset, damageStates, stateList


def import_CBMData_shm() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Data Import: Import condition-based monitoring example data
    
    Replicates import_CBMData_shm.m exactly.
    
    Returns
    -------
    dataset : np.ndarray
        Shape (TIME, CHANNELS, INSTANCES). Data set in concatenated format,
        4 channels are tachometer (ch 1) and 3 accelerometers.
    damageStates : np.ndarray
        Shape (INSTANCES, 1). Binary classification vector of known damage states 
        (0-undamaged and 1-damaged).
    stateList : np.ndarray  
        Shape (INSTANCES, 1). State for each instance.
    Fs : float
        Sampling frequency in Hz.
        
    Raises
    ------
    FileNotFoundError
        If data_CBM.mat cannot be found.
    """
    data_file = _get_example_data_path("data_CBM.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data_CBM.mat at {data_file}. "
            f"Please ensure the file exists in the examples/data directory."
        )
    
    # Load the MATLAB file
    mat_data = scipy.io.loadmat(str(data_file))
    
    # Extract variables (names from MATLAB file)
    dataset = mat_data['dataset']
    damageStates = mat_data['damageStates'] 
    stateList = mat_data['stateList']
    Fs = float(mat_data['Fs'][0, 0])  # Extract scalar from array
    
    return dataset, damageStates, stateList, Fs


def import_ActiveSense1_shm() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, float, np.ndarray, np.ndarray]:
    """
    Data Import: Import active sensing dataset #1
    
    Replicates import_ActiveSense1_shm.m exactly.
    
    Returns
    -------
    waveformBase : np.ndarray
        Shape (TIME, SENSORPAIRS). Matrix of waveforms acquired before damage was introduced.
    waveformTest : np.ndarray  
        Shape (TIME, SENSORPAIRS). Matrix of waveforms acquired after damage was introduced.
    sensorLayout : np.ndarray
        Shape (SENSORINFO, SENSORS). Sensor layout IDs and coordinates,
        SENSORINFO is the ordered set [sensorID, xCoord, yCoord].
    pairList : np.ndarray
        Shape (SENSORIDS, PAIRS). Matrix of actuator-sensor PAIRS,
        SENSORIDS is the ordered set of integer IDs [actuatorID, sensorID].
    borderStruct : dict
        Contains 'outside' and 'inside' fields with plate border definitions.
    sampleRate : float
        Sampling rate (points/time).
    actuationWaveform : np.ndarray
        Shape (TIME, 1). Waveform used for actuation.
    damageLocation : np.ndarray
        Shape (COORDINATES, 1). x and y coordinates of damage location,
        COORDINATES is the ordered set [x, y].
        
    Raises
    ------
    FileNotFoundError
        If data_example_ActiveSense.mat cannot be found.
    """
    data_file = _get_example_data_path("data_example_ActiveSense.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data_example_ActiveSense.mat at {data_file}. "
            f"Please ensure the file exists in the examples/data directory."
        )
    
    # Load the MATLAB file with specific variables (following MATLAB exactly)
    mat_data = scipy.io.loadmat(
        str(data_file), 
        variable_names=['waveformBase', 'waveformTest', 'sensorLayout', 'pairList',
                       'borderStruct', 'sampleRate', 'actuationWaveform', 'damageLocation']
    )
    
    waveformBase = mat_data['waveformBase']
    waveformTest = mat_data['waveformTest'] 
    sensorLayout = mat_data['sensorLayout']
    pairList = mat_data['pairList']
    borderStruct = mat_data['borderStruct']
    sampleRate = float(mat_data['sampleRate'][0, 0])  # Extract scalar
    actuationWaveform = mat_data['actuationWaveform']
    damageLocation = mat_data['damageLocation']
    
    # Convert borderStruct from MATLAB struct to Python dict
    # MATLAB structs are loaded as numpy structured arrays
    border_dict = {}
    if borderStruct.dtype.names:
        for field_name in borderStruct.dtype.names:
            border_dict[field_name] = borderStruct[field_name][0, 0]
    
    return waveformBase, waveformTest, sensorLayout, pairList, border_dict, sampleRate, actuationWaveform, damageLocation


def import_SensorDiagnostic_shm() -> Tuple[np.ndarray, np.ndarray]:
    """
    Data Import: Import sensor diagnostic dataset
    
    Replicates import_SensorDiagnostic_shm.m exactly.
    
    Returns
    -------
    healthyData : np.ndarray
        Shape (FREQUENCY, SENSORS). Susceptance data from healthy sensors only,
        first column contains frequencies.
    exampleSensorData : np.ndarray
        Shape (FREQUENCY, SENSORS). Susceptance data from healthy, debonded, 
        and broken sensors, first column contains frequencies.
        
    Raises
    ------
    FileNotFoundError
        If dataSensorDiagnostic.mat cannot be found.
    """
    data_file = _get_example_data_path("dataSensorDiagnostic.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find dataSensorDiagnostic.mat at {data_file}. "
            f"Please ensure the file exists in the examples/data directory."
        )
    
    # Load the MATLAB file with specific variables
    mat_data = scipy.io.loadmat(str(data_file), variable_names=['sd_ex', 'sd_ex_broken'])
    
    # Following MATLAB exactly:
    # healthyData = sd_ex;
    # exampleSensorData = sd_ex_broken;
    healthyData = mat_data['sd_ex']
    exampleSensorData = mat_data['sd_ex_broken']
    
    return healthyData, exampleSensorData


def import_ModalOSP_shm() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Data Import: Import Modal Optimal Sensor Placement Data
    
    Replicates import_ModalOSP_shm.m exactly.
    
    Returns
    -------
    nodeLayout : np.ndarray
        Shape (NODEINFO, NODES, INSTANCES). ID and coordinates of each node,
        NODEINFO is the ordered set [ID XCoord YCoord ZCoord].
    elements : np.ndarray
        Shape (NODES, ELEMENTS). Indices of the nodes that construct each element.
    modeShapes : np.ndarray
        Shape (DOFS, MODES). Mode shapes of interest.
    respDOF : np.ndarray
        Shape (DOFS, DOFINFO). Definitions of the degrees of freedom for the response vector.
        DOFINFO is the ordered set [coordID, responseDirection].
        
    Raises
    ------
    FileNotFoundError
        If data_OSPExampleModal.mat cannot be found.
    """
    data_file = _get_example_data_path("data_OSPExampleModal.mat")
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data_OSPExampleModal.mat at {data_file}. "
            f"Please ensure the file exists in the examples/data directory."
        )
    
    # Load the MATLAB file (load all variables like MATLAB)
    mat_data = scipy.io.loadmat(str(data_file))
    
    # Extract the required variables
    nodeLayout = mat_data['nodeLayout']
    elements = mat_data['elements']
    modeShapes = mat_data['modeShapes']
    respDOF = mat_data['respDOF']
    
    return nodeLayout, elements, modeShapes, respDOF