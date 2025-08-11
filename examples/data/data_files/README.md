# SHMTools Example Datasets

This directory contains the standard example datasets used throughout the SHMTools library. These datasets were originally distributed with the MATLAB SHMTools library and provide benchmark data for testing structural health monitoring algorithms.

## Available Datasets

### 1. 3-Story Structure Dataset (`data3SS.mat`)
**Size**: ~25 MB  
**Description**: Base-excited 3-story aluminum frame structure with various damage conditions  
**Structure**: (8192 time points, 5 channels, 170 test conditions)  
**Sampling Rate**: 2000 Hz  
**Damage States**: 17 different states (9 undamaged, 8 damaged)  
**Reference**: Figueiredo et al. (2009) LA-14393 report  

**Usage:**
```python
from examples.data import import_3story_structure_shm, load_3story_data

# MATLAB-compatible import
dataset, damage_states, state_list = import_3story_structure_shm()

# Dictionary format with metadata
data = load_3story_data()
signals = data['dataset'][:, 1:, :]  # Channels 2-5 (accelerometers)
```

### 2. Condition-Based Monitoring Dataset (`data_CBM.mat`)
**Size**: ~54 MB  
**Description**: Rotating machinery vibration data for bearing and gearbox fault analysis  
**Structure**: (10240 time points, 4 channels, 384 test instances)  
**Sampling Rate**: 2048 Hz  
**Channels**: Tachometer, Gearbox Accel, Top Bearing Accel, Side Bearing Accel  
**Damage States**: 6 fault conditions (healthy bearings, ball faults, gear faults)  

**Usage:**
```python
from examples.data import import_cbm_data_shm, load_cbm_data

# MATLAB-compatible import  
dataset, damage_states, state_list, fs = import_cbm_data_shm()

# Dictionary format
data = load_cbm_data()
```

### 3. Active Sensing Dataset (`data_example_ActiveSense.mat`)
**Size**: ~32 MB  
**Description**: Guided wave propagation data from plate structure with piezoelectric transducers  
**Structure**: Baseline and test waveforms from 32 sensors (492 actuator-sensor pairs)  
**Sampling Rate**: 5 MHz  
**Test Structure**: 0.01" concave plate (~48" side), damage simulated with magnet  

**Usage:**
```python
from examples.data import import_active_sense1_shm

(waveform_base, waveform_test, sensor_layout, pair_list,
 border_struct, sample_rate, actuation_waveform, 
 damage_location) = import_active_sense1_shm()
```

### 4. Sensor Diagnostic Dataset (`dataSensorDiagnostic.mat`)
**Size**: ~63 KB  
**Description**: Piezoelectric sensor impedance measurements for sensor health monitoring  
**Data Types**: Susceptance measurements from healthy, debonded, and broken sensors  
**Frequency Range**: Swept frequency impedance measurements  

**Usage:**
```python
from examples.data import import_sensor_diagnostic_shm

healthy_data, example_sensor_data = import_sensor_diagnostic_shm()
frequencies = healthy_data[:, 0]  # First column contains frequencies
```

### 5. Modal Analysis Dataset (`data_OSPExampleModal.mat`)
**Size**: ~50 KB  
**Description**: Finite element model data for optimal sensor placement studies  
**Components**: Node layouts, element connectivity, mode shapes, DOF definitions  
**Application**: Structural dynamics analysis and sensor placement optimization  

**Usage:**
```python
from examples.data import import_modal_osp_shm

node_layout, elements, mode_shapes, resp_dof = import_modal_osp_shm()
```

## Data Organization

The data import functions provide two interfaces:

1. **MATLAB-compatible functions** (`import_*_shm`): Return raw arrays matching original MATLAB function signatures
2. **Dictionary loaders** (`load_*_data`): Return dictionaries with metadata and descriptive field names

## Original Source

These datasets were originally distributed with:
- **SHMTools**: MATLAB library by Los Alamos National Laboratory
- **References**: Various technical reports and publications from LANL structural health monitoring research
- **License**: Research and educational use

## File Format

All files are MATLAB `.mat` format (HDF5-based) and can be loaded using:
- `scipy.io.loadmat()` (Python)
- `load()` (MATLAB)
- HDF5 viewers for inspection

## Size and Storage

**Total size**: ~161 MB  
These files are **not tracked in git** due to size. They must be manually downloaded and placed in this directory for the examples to work.

## Getting the Data

The data files should be downloaded and placed in this directory. Check the main repository README for download instructions or contact the repository maintainers.