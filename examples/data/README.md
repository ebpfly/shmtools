# SHMTools Example Datasets

## Dataset Files

This directory contains the example datasets required for SHMTools Python notebooks:

| File | Size | Description | Used By |
|------|------|-------------|---------|
| `data3SS.mat` | 25MB | 3-story structure base excitation data | PCA, Mahalanobis, SVD, NLPCA, Factor Analysis examples |
| `data_CBM.mat` | 54MB | Condition-based monitoring (rotating machinery) | CBM bearing/gearbox analysis examples |
| `data_example_ActiveSense.mat` | 32MB | Guided wave ultrasonic measurements | Active sensing feature extraction |
| `dataSensorDiagnostic.mat` | 63KB | Piezoelectric sensor impedance measurements | Sensor diagnostics example |
| `data_OSPExampleModal.mat` | 50KB | Modal analysis and optimal sensor placement | Modal features, OSP examples |

**Total**: ~161MB

## Primary Dataset: data3SS.mat

The main dataset used by most examples contains acceleration measurements from a 3-story base-excited structure:

- **Format**: `(8192 time points, 5 channels, 170 test conditions)`
- **Sampling Rate**: 2000 Hz
- **Channels**: Force, Ch2, Ch3, Ch4, Ch5 (accelerometers)
- **Test Conditions**: 17 damage states × 10 tests each
  - **States 1-9**: Undamaged baseline conditions
  - **States 10-17**: Progressive damage scenarios (nonlinear bumper contact)

### Structure Description
The test structure consists of:
- Aluminum columns and plates with bolted joints
- Four-column frame design (essentially 4-DOF system)
- Suspended center column for damage simulation
- Electrodynamic shaker for base excitation
- Adjustable bumper to create nonlinear damage behavior

## Data Loading

Use the provided data loading utilities:

```python
from shmtools.utils.data_loading import load_3story_data

# Load primary dataset
data = load_3story_data()
dataset = data['dataset']  # Shape: (8192, 5, 170)
fs = data['fs']           # 2000.0 Hz

# Extract channels 2-5 (accelerometers only)
signals = dataset[:, 1:, :]  # Skip channel 0 (force)

# Split by damage condition
baseline = signals[:, :, :90]   # Undamaged (states 1-9, 10 tests each)
damaged = signals[:, :, 90:]    # Damaged (states 10-17, 10 tests each)
```

Other datasets:
```python
from shmtools.utils.data_loading import (
    load_cbm_data,
    load_active_sensing_data, 
    load_sensor_diagnostic_data,
    load_modal_osp_data
)
```

## Copyright and Licensing

All datasets are from the original SHMTools library developed by Los Alamos National Laboratory and are distributed under the same license terms (LA-CC-14-046).

**Copyright Notice**  
Copyright (c) 2014, Los Alamos National Security, LLC. All rights reserved.

These datasets are redistributed under BSD-3-Clause-like terms. The data content remains unchanged from the original LANL releases, only converted to Python-compatible loading.

## References

- Figueiredo, E., Park, G., Figueiras, J., Farrar, C., & Worden, K. (2009). Structural Health Monitoring Algorithm Comparisons using Standard Data Sets. Los Alamos National Laboratory Report: LA-14393.

- Original MATLAB SHMTools library: https://github.com/lanl/SHMTools (LA-CC-14-046)

## File Integrity

If you encounter loading issues, verify file integrity:

```bash
# Check file sizes match expected values
ls -lh *.mat

# Expected sizes:
# data3SS.mat: ~25MB
# data_CBM.mat: ~54MB  
# data_example_ActiveSense.mat: ~32MB
# dataSensorDiagnostic.mat: ~63KB
# data_OSPExampleModal.mat: ~50KB
```

For any data loading issues, see the data loading utilities in `shmtools/utils/data_loading.py`.