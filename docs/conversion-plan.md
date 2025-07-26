# Example-Driven Conversion Plan: MATLAB to Python

## Overview

This plan converts MATLAB ExampleUsageScripts to Python Jupyter notebooks one by one, ensuring each example works end-to-end with all dependencies properly converted. Each conversion validates functional equivalence with the original MATLAB and publishes a working notebook to HTML.

## Conversion Strategy

### Core Principle: Example-Driven Development
Each phase converts **one complete example** with all its dependencies, validates the output matches MATLAB, and publishes a working Jupyter notebook. This ensures:

- ✅ **Functional Parity**: Python results match MATLAB exactly  
- ✅ **Dependency Validation**: All required functions work together
- ✅ **Publication Ready**: Clean notebook with explanations and visualizations
- ✅ **Progressive Complexity**: Build from simple to advanced examples

### Quality Gates for Each Example
1. **MATLAB Analysis**: Read and understand the original `.m` file completely
2. **MATLAB Metadata Extraction**: Extract "VERBOSE FUNCTION CALL" and other UI metadata from original MATLAB files
3. **Dependency Mapping**: Identify all required SHMTools functions and locate their MATLAB source files
4. **Function Conversion**: Convert each dependency following `docs/docstring-format.md` including exact VERBOSE FUNCTION CALL reproduction
5. **Algorithm Verification**: Validate Python output matches MATLAB with same inputs
6. **UI Metadata Validation**: Ensure human-readable names match original MATLAB exactly
7. **Notebook Creation**: **CRITICAL**: Direct translation of MATLAB workflow preserving ALL educational comments and explanations from original
8. **Execution Testing**: Ensure notebook runs end-to-end without errors
9. **HTML Publishing**: Export to clean HTML with proper formatting

### Notebook Conversion Rules
**⚠️ MANDATORY**: Notebooks must be direct translations of MATLAB examples, preserving ALL educational content:

- **Section Headings**: Match MATLAB `%% Section Name` comments exactly as `## Section Name`
- **Educational Comments**: Convert ALL MATLAB comments to markdown cells - preserve introduction, methodology explanations, and step descriptions
- **Workflow Preservation**: Include only the workflow steps present in the original MATLAB file (no additions)
- **Function Calls**: Use exact Python equivalents of MATLAB function calls with same parameters
- **Code Comments**: Preserve MATLAB code comments as Python comments where appropriate
- **Educational Value**: Maintain the instructional purpose of the original MATLAB examples

**Example Translation**:
```matlab
%% Load data
% The data here is in the form of time series in a 3 dimensional matrix
% (time, sensors, instances) and also a state vector representing the
% various environmental conditions under which the data is collected.
load('data3SS.mat');
```

**Python Translation**:
```markdown
## Load data

The data here is in the form of time series in a 3 dimensional matrix
(time, sensors, instances) and also a state vector representing the
various environmental conditions under which the data is collected.
```
```python
data = load_3story_data()
dataset = data['dataset'] 
states = data['damage_states']
```

### Development Order: Simple to Complex

Start with fundamental outlier detection methods that share common dependencies, then progress to specialized algorithms.

## MATLAB Reference Files: Critical Requirement

**⚠️ MANDATORY**: Every function conversion MUST reference the original MATLAB file for complete algorithm understanding and UI metadata extraction.

### MATLAB File Locations

The original MATLAB SHMTools library is located at:
```
/Users/eric/repo/shm/shmtool-matlab/SHMTools/
```

**Core function directories:**
- **Feature Extraction**: `SHMTools/SHMFunctions/FeatureExtraction/`
  - `TimeSeriesModels/arModel_shm.m` - AR model parameter estimation
  - `SpectralAnalysis/psdWelch_shm.m` - Power spectral density 
  - `Statistics/rms_shm.m`, `crestFactor_shm.m` - Statistical moments
  - `Preprocessing/filter_shm.m` - Digital filtering

- **Feature Classification**: `SHMTools/SHMFunctions/FeatureClassification/`
  - `OutlierDetection/ParametricDetectors/learnPCA_shm.m`, `scorePCA_shm.m`
  - `OutlierDetection/ParametricDetectors/learnMahalanobis_shm.m`, `scoreMahalanobis_shm.m`
  - `OutlierDetection/ParametricDetectors/learnSVD_shm.m`, `scoreSVD_shm.m`

- **Data Acquisition**: `SHMTools/SHMFunctions/DataAcquisition/`
- **Auxiliary Functions**: `SHMTools/SHMFunctions/Auxiliary/`

### Required MATLAB Metadata Extraction

Every MATLAB function contains essential UI metadata in its header:

```matlab
function [output1, output2] = functionName_shm (input1, input2)
% Category: Description of function purpose
%
% VERBOSE FUNCTION CALL:
%   [Output Description 1, Output Description 2] = Function Display Name (Input Description 1, Input Description 2)
%
% CALL METHODS:
%   [output1, output2] = functionName_shm (input1, input2)
%
% DESCRIPTION:
%   Detailed algorithm description...
%
% INPUTS:
%   input1 (dimensions) : description
%   input2 (dimensions) : description  
%
% OUTPUTS:
%   output1 (dimensions) : description
%   output2 (dimensions) : description
```

### Function Conversion Protocol

**🔍 Step 1: Locate MATLAB File**
- Search in appropriate `SHMFunctions/` subdirectory
- If not found, search recursively through entire `SHMTools/` directory
- Use file system search if necessary - **DO NOT give up**
- MATLAB functions may be in unexpected locations

**📋 Step 2: Extract Complete Metadata**
- **VERBOSE FUNCTION CALL**: Copy exactly for `:verbose_call:` field
- **Category**: Map to `:category:` field 
- **Function description**: Use for docstring brief description
- **Input descriptions**: Use for parameter documentation with `:description:` in `.. gui::` blocks
- **Output descriptions**: Use for return value documentation
- **Algorithm details**: Include in docstring body

**✅ Step 3: Validate Metadata Accuracy**
```python
# Python docstring must match MATLAB exactly:
.. meta::
    :display_name: Function Display Name
    :verbose_call: [Output Description 1, Output Description 2] = Function Display Name (Input Description 1, Input Description 2)
```

### Example: ar_model Conversion

**MATLAB Source** (`TimeSeriesModels/arModel_shm.m`):
```matlab
% VERBOSE FUNCTION CALL:
%   [AR Parameters Feature Vectors, RMS Residuals Feature Vectors, AR
%   Parameters, AR Residuals, AR Prediction] = AR Model (Time Series Data,
%   AR Model Order)
```

**Python Result** (must match exactly):
```python
:verbose_call: [AR Parameters Feature Vectors, RMS Residuals Feature Vectors, AR Parameters, AR Residuals, AR Prediction] = AR Model (Time Series Data, AR Model Order)
```

### File Search Strategy

If a function cannot be found in the expected location:

1. **Search by pattern**: `find /Users/eric/repo/shm/shmtool-matlab -name "*functionname*.m"`
2. **Search by content**: `grep -r "function.*functionname" /Users/eric/repo/shm/shmtool-matlab/`  
3. **Check alternative names**: Some functions may have variations or be in wrapper directories
4. **Verify in mFUSE library**: Check `mFUSE/mFUSElibrary.txt` for function locations
5. **Search example scripts**: Function may be referenced in `Examples/ExampleUsageScripts/`

**⚠️ CRITICAL**: If a MATLAB function truly cannot be found, this indicates a missing dependency that must be resolved before conversion can proceed.

## Phase 1: PCA Outlier Detection
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/examplePCA.m` (165 lines)
- **Python Output**: `examples/notebooks/basic/pca_outlier_detection.ipynb` 
- **HTML Output**: `examples/published/html/pca_outlier_detection.html`

### Required Dependencies

#### Core Functions to Convert

**⚠️ MANDATORY FOR EACH FUNCTION**: Read the complete MATLAB source file and extract all UI metadata before conversion.

1. **`arModel_shm`** → `shmtools.features.ar_model_shm()`
   - **MATLAB Source**: `/Users/eric/repo/shm/shmtool-matlab/SHMTools/SHMFunctions/FeatureExtraction/TimeSeriesModels/arModel_shm.m`
   - **Purpose**: AR model parameter estimation and RMSE calculation
   - **VERBOSE CALL**: Must extract from MATLAB header (lines 4-7)
   - **Dependencies**: Basic matrix operations only

2. **`learnPCA_shm`** → `shmtools.classification.learn_pca_shm()`
   - **MATLAB Source**: `/Users/eric/repo/shm/shmtool-matlab/SHMTools/SHMFunctions/FeatureClassification/OutlierDetection/ParametricDetectors/learnPCA_shm.m`
   - **Purpose**: Train PCA-based outlier detection model  
   - **VERBOSE CALL**: Must extract from MATLAB header (lines 4-6)
   - **Dependencies**: PCA decomposition, normalization

3. **`scorePCA_shm`** → `shmtools.classification.score_pca_shm()`
   - **Source**: `../shmtool-matlab/SHMTools/SHMFunctions/FeatureClassification/OutlierDetection/ParametricDetectors/scorePCA_shm.m`
   - **Purpose**: Score new data against trained PCA model
   - **Dependencies**: `learnPCA_shm` model format

#### Support Functions
4. **Data loading utilities** → `shmtools.utils.load_mat_data()`
   - **Purpose**: Load `data3SS.mat` format consistently
   - **Implementation**: Using `scipy.io.loadmat` with proper reshaping

### Example Analysis
The `examplePCA.m` script demonstrates:
1. **Data Loading**: 3-story structure dataset with 4 channels, multiple conditions  
2. **Feature Extraction**: AR(15) model RMSE values from channels 2-5
3. **Train/Test Split**: Training on conditions 1-9, testing on conditions 1-9 (baseline) + 10-17 (damage)
4. **PCA Modeling**: Learn PCA transformation from training features
5. **Damage Detection**: Score test data and apply 95% threshold for classification
6. **Visualization**: Time histories, feature plots, damage indicator bar charts

### Conversion Requirements

#### Function Signatures (MATLAB-compatible)
```python
def ar_model_shm(data: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate autoregressive model parameters and compute RMSE.
    
    .. meta::
        :category: Features - Time Series Models  
        :matlab_equivalent: arModel_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
    """

def learn_pca_shm(features: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Learn PCA-based outlier detection model from training features.
    
    .. meta::
        :category: Classification - Parametric Detectors
        :matlab_equivalent: learnPCA_shm  
        :complexity: Basic
        :data_type: Features
        :output_type: Model
    """

def score_pca_shm(features: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """
    Score features using trained PCA outlier detection model.
    
    .. meta::
        :category: Classification - Parametric Detectors
        :matlab_equivalent: scorePCA_shm
        :complexity: Basic  
        :data_type: Features
        :output_type: Scores
    """
```

#### Modern Python Interfaces
```python
# Convenience functions for Pythonic usage
def pca_outlier_detection(
    train_features: np.ndarray,
    test_features: np.ndarray, 
    n_components: Optional[int] = None,
    threshold_percentile: float = 95.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Modern interface combining learn + score + threshold."""
```

### Success Criteria
- [ ] `arModel_shm` produces identical RMSE values to MATLAB
- [ ] `learnPCA_shm` model parameters match MATLAB output
- [ ] `scorePCA_shm` damage indicators match MATLAB exactly
- [ ] Jupyter notebook runs without errors
- [ ] All 17 test conditions classified correctly (9 undamaged, 8 damaged)
- [ ] HTML export renders properly with all plots

### Dependencies for Future Examples
This phase establishes foundation functions reused in later examples:
- AR modeling → Used in Mahalanobis, SVD, Factor Analysis examples
- PCA framework → Extended to NLPCA and other parametric detectors
- Train/test data splitting → Standard pattern across all outlier detection

---

## Phase 2: Mahalanobis Distance Outlier Detection  
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleMahalanobis.m` (169 lines)
- **Python Output**: `examples/notebooks/basic/mahalanobis_outlier_detection.ipynb`

### Additional Dependencies
- **`learnMahalanobis_shm`** → `shmtools.classification.learn_mahalanobis_shm()`
- **`scoreMahalanobis_shm`** → `shmtools.classification.score_mahalanobis_shm()`

**Reuses**: `arModel_shm` from Phase 1, same data loading patterns

---

## Phase 3: SVD Outlier Detection
*Target: 1-2 weeks*

### Target Example  
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleSVD.m` (170 lines)
- **Python Output**: `examples/notebooks/basic/svd_outlier_detection.ipynb`

### Additional Dependencies
- **`learnSVD_shm`** → `shmtools.classification.learn_svd_shm()`
- **`scoreSVD_shm`** → `shmtools.classification.score_svd_shm()`

**Reuses**: `arModel_shm`, data patterns from Phases 1-2

---

## Phase 4: Factor Analysis Outlier Detection
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleFactorAnalysis.m` (164 lines) 
- **Python Output**: `examples/notebooks/intermediate/factor_analysis_outlier_detection.ipynb`

### Additional Dependencies
- **`learnFactorAnalysis_shm`** → `shmtools.classification.learn_factor_analysis_shm()`
- **`scoreFactorAnalysis_shm`** → `shmtools.classification.score_factor_analysis_shm()`

---

## Phase 5: Nonlinear PCA (NLPCA) Outlier Detection
*Target: 3-4 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleNLPCA.m` (163 lines)
- **Python Output**: `examples/notebooks/advanced/nlpca_outlier_detection.ipynb`

### Additional Dependencies  
- **`learnNLPCA_shm`** → `shmtools.classification.learn_nlpca_shm()`
- **`scoreNLPCA_shm`** → `shmtools.classification.score_nlpca_shm()`
- Neural network components for nonlinear PCA

---

## Phase 6: AR Model Order Selection
*Target: 1-2 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleARModelOrder.m` (91 lines)
- **Python Output**: `examples/notebooks/basic/ar_model_order_selection.ipynb`

### Additional Dependencies
- **`arModelOrder_shm`** → `shmtools.features.ar_model_order_shm()`
- Information criteria (AIC, BIC) for model selection

---

## Phase 7: Nonparametric Outlier Detection
*Target: 3-4 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleDirectUseOfNonParametric.m` (149 lines)
- **Python Output**: `examples/notebooks/advanced/nonparametric_outlier_detection.ipynb`

### Additional Dependencies
- **Kernel density estimation functions**
- **Fast metric kernel density** → `shmtools.classification.fast_metric_kernel_density_shm()`

---

## Phase 8: Semi-Parametric Outlier Detection  
*Target: 3-4 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleDirectUseOfSemiParametric.m` (147 lines)
- **Python Output**: `examples/notebooks/advanced/semiparametric_outlier_detection.ipynb`

### Additional Dependencies
- **Gaussian Mixture Model functions**
- **Semi-parametric density estimation**

---

## Phase 9: Active Sensing Feature Extraction
*Target: 4-5 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleActiveSensingFeature.m` (289 lines)
- **Python Output**: `examples/notebooks/advanced/active_sensing_feature_extraction.ipynb`

### Additional Dependencies
- **Guided wave analysis functions**
- **Matched filtering algorithms** 
- **Geometry and propagation utilities**

---

## Phase 10: Condition-Based Monitoring
*Target: 3-4 weeks*

### Target Examples
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/example_CBM_Bearing_Analysis.m`
- **Python Output**: `examples/notebooks/specialized/cbm_bearing_analysis.ipynb`

### Additional Dependencies
- **Time-synchronous averaging**
- **Order tracking algorithms**
- **Kurtogram analysis**

---

## Later Phases: Specialized Examples

### Phase 11: Sensor Diagnostics
- **exampleSensorDiagnostics.m** → `sensor_diagnostics.ipynb`

### Phase 12: Modal Analysis
- **exampleModalFeatures.m** → `modal_analysis_features.ipynb`

### Phase 13: Hardware Integration
- **example_NI_multiplex.m** → `ni_daq_integration.ipynb`
- **example_DAQ_ARModel_Mahalanobis.m** → `daq_real_time_monitoring.ipynb`

---

## Data Management Strategy

### Manual Data Setup (Simple Approach)

Users manually download all example datasets once and place them in the repository. No fancy automation needed.

#### Required Datasets
Based on ExampleUsageScripts analysis:

1. **`data3SS.mat`** (25MB) - 3-story structure data
   - **Used by**: 9+ examples (PCA, Mahalanobis, SVD, NLPCA, etc.)
   - **Format**: `(8192 time points, 5 channels, 170 conditions)`
   - **Description**: Base-excited 3-story structure with 17 damage states (10 tests each)

2. **`dataSensorDiagnostic.mat`** (63KB) - Sensor health data
3. **`data_CBM.mat`** (54MB) - Condition-based monitoring data  
4. **`data_example_ActiveSense.mat`** (32MB) - Active sensing data
5. **`data_OSPExampleModal.mat`** (50KB) - Modal analysis data

**Total**: ~161MB (reasonable for manual download)

### Repository Structure
```bash
shmtools-python/
├── examples/
│   ├── data/                          # User puts downloaded .mat files here
│   │   ├── README.md                  # Download instructions
│   │   ├── .gitignore                 # Ignore *.mat files  
│   │   ├── data3SS.mat               # ← User downloads these
│   │   ├── dataSensorDiagnostic.mat  # ← 
│   │   ├── data_CBM.mat              # ←
│   │   ├── data_example_ActiveSense.mat # ←
│   │   └── data_OSPExampleModal.mat  # ←
│   └── notebooks/
│       ├── basic/
│       ├── intermediate/
│       └── advanced/
└── shmtools/
    └── utils/
        └── data_loading.py           # Simple .mat file loading
```

### Data Loading Interface
```python
# shmtools/utils/data_loading.py
import scipy.io
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

def get_data_dir() -> Path:
    """Get the standard data directory."""
    # Assume we're always called from project root or examples/
    current = Path.cwd()
    if current.name == "shmtools-python":
        return current / "examples" / "data"
    elif current.name == "examples":
        return current / "data"
    elif (current / "examples" / "data").exists():
        return current / "examples" / "data"
    else:
        # Try to find it relative to this file
        this_file = Path(__file__).parent
        project_root = this_file.parent.parent  # shmtools/utils -> shmtools-python
        return project_root / "examples" / "data"

def load_3story_data() -> Dict[str, Any]:
    """
    Load 3-story structure dataset.
    
    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'dataset': (8192, 5, 170) acceleration data  
        - 'fs': sampling frequency (Hz)
        - 'channels': channel names
        - 'damage_states': damage condition for each test
        
    Examples
    --------
    >>> data = load_3story_data()
    >>> signals = data['dataset'][:, 1:, :]  # Channels 2-5 only
    >>> baseline = signals[:, :, :90]        # Undamaged conditions
    >>> damaged = signals[:, :, 90:]         # Damaged conditions
    """
    data_dir = get_data_dir()
    data_path = data_dir / "data3SS.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            f"Please download data3SS.mat and place it in {data_dir}/\n"
            f"See {data_dir}/README.md for download instructions."
        )
    
    # Load MATLAB file
    mat_data = scipy.io.loadmat(str(data_path))
    dataset = mat_data['dataset']  # Shape: (8192, 5, 170)
    
    return {
        'dataset': dataset,
        'fs': 2000.0,  # Sampling frequency from original documentation
        'channels': ['Force', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
        'conditions': np.arange(1, 171),  # State conditions 1-170
        'damage_states': np.repeat(np.arange(1, 18), 10),  # 17 states, 10 tests each
        'description': "3-story structure base excitation data (LANL)"
    }

def load_sensor_diagnostic_data() -> Dict[str, Any]:
    """Load sensor diagnostic dataset."""
    data_dir = get_data_dir()
    data_path = data_dir / "dataSensorDiagnostic.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    return dict(scipy.io.loadmat(str(data_path)))

def load_cbm_data() -> Dict[str, Any]:
    """Load condition-based monitoring dataset."""
    data_dir = get_data_dir()
    data_path = data_dir / "data_CBM.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    return dict(scipy.io.loadmat(str(data_path)))

def load_active_sensing_data() -> Dict[str, Any]:
    """Load active sensing dataset."""
    data_dir = get_data_dir()
    data_path = data_dir / "data_example_ActiveSense.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    return dict(scipy.io.loadmat(str(data_path)))

def load_modal_osp_data() -> Dict[str, Any]:
    """Load modal analysis and optimal sensor placement dataset."""
    data_dir = get_data_dir()
    data_path = data_dir / "data_OSPExampleModal.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    return dict(scipy.io.loadmat(str(data_path)))
```

### User Setup Instructions
```markdown
# examples/data/README.md

# SHMTools Example Datasets

## Quick Setup

1. **Download the dataset package** from [provide link to Dropbox/Google Drive/etc.]
   - Or individual files from the original MATLAB SHMTools distribution

2. **Extract/copy the .mat files** to this directory:
   ```
   examples/data/
   ├── data3SS.mat               ← 25MB
   ├── dataSensorDiagnostic.mat  ← 63KB  
   ├── data_CBM.mat              ← 54MB
   ├── data_example_ActiveSense.mat ← 32MB
   └── data_OSPExampleModal.mat  ← 50KB
   ```

3. **That's it!** All notebooks will now work.

## Dataset Descriptions

### data3SS.mat (Primary Dataset)
- **Source**: 3-story base-excited structure (LANL test bed)
- **Format**: (8192 time points, 5 channels, 170 test conditions)
- **Content**: 17 damage states × 10 tests each
  - States 1-9: Undamaged baseline conditions
  - States 10-17: Progressive damage scenarios
- **Used by**: PCA, Mahalanobis, SVD, NLPCA, Factor Analysis examples

### Other Datasets
- **dataSensorDiagnostic.mat**: Piezoelectric sensor impedance measurements
- **data_CBM.mat**: Rotating machinery vibration for condition monitoring  
- **data_example_ActiveSense.mat**: Guided wave ultrasonic measurements
- **data_OSPExampleModal.mat**: Modal analysis and sensor placement data

## Copyright Notice
All datasets are from the original SHMTools library developed by Los Alamos 
National Laboratory (LA-CC-14-046) and are redistributed under the same 
BSD-3-Clause-like license terms.

Copyright (c) 2014, Los Alamos National Security, LLC. All rights reserved.

## References
- Figueiredo, E., Park, G., Figueiras, J., Farrar, C., & Worden, K. (2009).
  Structural Health Monitoring Algorithm Comparisons using Standard Data Sets.
  Los Alamos National Laboratory Report: LA-14393.
```

### Testing Strategy
```python
# tests/conftest.py
import pytest
from pathlib import Path

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_data: requires example datasets")

@pytest.fixture
def data_dir():
    """Get the data directory, skip tests if no data available."""
    data_path = Path(__file__).parent.parent / "examples" / "data" 
    if not (data_path / "data3SS.mat").exists():
        pytest.skip("Example datasets not available. See examples/data/README.md")
    return data_path
```

### Updated Phase 1: PCA Outlier Detection
```python
# In the Jupyter notebook:
from shmtools.utils.data_loading import load_3story_data

# Simple data loading - no downloads, no complexity
data = load_3story_data()
dataset = data['dataset']
fs = data['fs']

# Extract channels 2-5 as in original MATLAB  
signals = dataset[:, 1:, :]  # Skip channel 0 (force)
```

## Implementation Guidelines

### Directory Structure
```
shmtools-python/
├── examples/
│   ├── notebooks/
│   │   ├── basic/           # Phases 1-3, 6
│   │   ├── intermediate/    # Phase 4  
│   │   ├── advanced/        # Phases 5, 7-9
│   │   └── specialized/     # Phases 10+
│   ├── published/
│   │   ├── html/           # HTML exports
│   │   └── pdf/            # PDF exports (optional)
│   └── data/               # Example datasets
└── shmtools/
    ├── features/           # AR models, spectral analysis
    ├── classification/     # Outlier detection algorithms  
    └── utils/             # Data loading, MATLAB compatibility
```

### Conversion Workflow for Each Example

#### 1. Analysis Phase
```bash
# Read original MATLAB file completely
# Identify all function dependencies
# Map data flow and algorithm steps
# Note visualization requirements
```

#### 2. Function Conversion Phase  
```bash
# Convert each dependency function to Python
# Follow docs/docstring-format.md exactly
# Include both _shm and modern interfaces
# Add comprehensive docstrings with GUI metadata
```

#### 3. Validation Phase
```bash
# Test each function with identical MATLAB inputs
# Validate outputs match to numerical precision
# Create unit tests for regression protection
```

#### 4. Notebook Creation Phase
```bash
# Create educational Jupyter notebook
# Include background theory and references
# Add explanatory text between code cells
# Create publication-quality visualizations
```

#### 5. Testing and Publication Phase
```bash
# Method 1: Execute and convert in one step using ExecutePreprocessor
python -c "
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

# Add shmtools to path
sys.path.insert(0, os.getcwd())

# Read notebook
with open('examples/notebooks/{category}/{notebook_name}.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Execute notebook with proper kernel
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})

# Convert to HTML with executed outputs
html_exporter = HTMLExporter()
(body, resources) = html_exporter.from_notebook_node(nb)

# Save HTML with executed outputs
os.makedirs('examples/published/html', exist_ok=True)
with open('examples/published/html/{notebook_name}.html', 'w') as f:
    f.write(body)

print(f'Published {notebook_name}.html with executed outputs')
"

# Alternative Method 2: Use jupyter nbconvert directly (if nbconvert works)
# jupyter nbconvert --to html --execute {notebook_path} --output-dir=examples/published/html/

# Step 3: Validate HTML renders correctly
# - Check all plots display properly  
# - Verify code outputs are visible (text outputs, print statements)
# - Confirm execution results are shown (dataframes, arrays, plots)
# - Look for "Found shmtools at:" or similar output text in HTML
```

### Quality Assurance Checklist

For each converted example:
- [ ] All MATLAB functions identified and mapped
- [ ] Python functions follow exact MATLAB algorithms  
- [ ] Docstrings include all required GUI metadata
- [ ] Outputs match MATLAB within numerical tolerance
- [ ] Jupyter notebook runs without errors
- [ ] **HTML export created and renders cleanly**
- [ ] All visualizations display correctly
- [ ] Educational content explains the methodology

### Phase Completion Criteria

**A phase is considered COMPLETE when:**
1. ✅ All required functions are implemented and working
2. ✅ Jupyter notebook executes end-to-end without errors (`jupyter execute notebook.ipynb`)
3. ✅ **HTML export is created with executed outputs visible** and saved to `examples/published/html/`
4. ✅ All plots, code outputs, and execution results render correctly in HTML
5. ✅ Functions are exported in appropriate module `__init__.py` files

**Critical Note**: The HTML publication must show executed code outputs, plots, and results. Empty code cells or missing outputs indicate incomplete publication.

### Success Metrics

- **Functional Parity**: Python results match MATLAB exactly
- **Reusability**: Functions work across multiple examples  
- **Documentation Quality**: Notebooks suitable for publication
- **GUI Integration**: Docstring metadata enables automatic web interface
- **Performance**: Conversion maintains or improves execution speed

This approach ensures each example provides immediate value while building a robust foundation for the complete SHMTools conversion.

---

## Lessons Learned

### Phase 1: PCA Outlier Detection - Key Insights

Based on the successful completion of Phase 1, several critical lessons emerged that will improve subsequent phase implementations:

#### 1. MATLAB-to-Python Indexing Pitfalls

**Issue**: The most significant bug was in AR model parameter estimation due to incorrect indexing conversion.

**Root Cause**: 
- MATLAB uses 1-based indexing: `for k=1:arOrder; A(:,k)=X(arOrder+1-k:t-k,i,j); end`
- Direct formula translation to Python 0-based indexing caused off-by-one errors
- Initial implementation used `k` from 0 to arOrder-1 but applied MATLAB formula directly

**Solution**:
```python
# Correct approach: Explicit conversion from MATLAB to Python indexing
for k in range(ar_order):
    matlab_k = k + 1  # Convert to MATLAB 1-based indexing
    start_idx = ar_order + 1 - matlab_k - 1  # Convert to 0-based
    end_idx = t - matlab_k  # Python end is exclusive
    A[:, k] = X[start_idx:end_idx, i, j]
```

**Lesson**: Always validate numerical algorithms with synthetic test cases where true parameters are known.

#### 2. Numerical Stability Considerations

**Issue**: PCA SVD decomposition failed with small datasets due to numerical instability.

**Root Cause**: 
- Zero standard deviations in standardization step
- Singular covariance matrices
- Small sample sizes in test scenarios

**Solution**:
```python
# Handle zero standard deviation
data_std = np.where(data_std == 0, 1.0, data_std)

# Add regularization for numerical stability
R = R + 1e-10 * np.eye(R.shape[0])
```

**Lesson**: Add defensive programming for edge cases (constant features, small datasets, singular matrices).

#### 3. Execution Context Dependencies

**Issue**: Notebook imports failed during `nbconvert --execute` due to module path resolution.

**Root Cause**: Different working directories when notebook runs in various contexts (Jupyter, nbconvert, different CWDs).

**Solution**:
```python
# Robust path resolution for multiple execution contexts
possible_paths = [
    notebook_dir.parent.parent.parent,  # From examples/notebooks/basic/
    current_dir.parent.parent,          # From examples/notebooks/
    current_dir,                        # From project root
    Path('/absolute/fallback/path')     # Absolute fallback
]
```

**Lesson**: Design notebooks to be executable from multiple working directories and contexts.

#### 4. Visualization Best Practices

**Issue**: Hardcoded `plt.text()` positioning breaks across different figure sizes and contexts.

**Root Cause**: Fixed coordinates don't adapt to different DPI, figure sizes, or data ranges.

**Solution**: Use `plt.legend()` for categorical labels rather than absolute positioning.

**Lesson**: Prefer matplotlib's automatic layout mechanisms over manual positioning.

#### 5. Testing Strategy Refinements

**Insights**:
- **Synthetic Data Validation**: Critical for catching algorithmic errors early
- **Real Data Integration**: Essential for finding practical issues (path resolution, data loading)
- **Cross-Platform Testing**: Module imports and paths behave differently across systems

**Recommended Testing Workflow**:
1. **Unit Tests**: Individual functions with synthetic data
2. **Integration Tests**: Full workflow with real data
3. **Notebook Execution**: End-to-end validation including publication
4. **Algorithm Validation**: Compare with known analytical solutions

#### 6. Documentation and Metadata Standards

**Successes**:
- Machine-readable docstrings enable automatic GUI generation
- Two-tier architecture (MATLAB-compatible + modern Python) provides flexibility
- Comprehensive examples with theory and implementation details

**Refinements**:
- Include algorithm validation steps in docstring examples
- Add numerical stability notes for edge cases
- Document execution context requirements

#### 7. Publication Workflow Optimization

**Key Steps**:
1. Develop and test functions individually
2. Create notebook with comprehensive workflow
3. Execute notebook with `nbconvert --execute` for full validation
4. Verify all plots and outputs render correctly
5. Check accessibility (alt text for images)

**Quality Gates Validated**:
- All functions pass individual and integration tests
- Notebook executes without errors from project root
- HTML export includes all outputs and visualizations
- Algorithm produces expected results on known test cases

#### 8. Data Management Insights

**Effective Approach**:
- Simple manual download strategy works well for research datasets
- Comprehensive error messages guide users to data setup
- Availability checking prevents cryptic failures

**Lessons**:
- Provide clear data setup instructions with exact file locations
- Include data availability checking in utility functions
- Design graceful degradation when datasets are missing

### Recommendations for Future Phases

#### Updated Quality Gates
1. **MATLAB Analysis**: Complete algorithmic understanding
2. **Dependency Mapping**: All required functions identified
3. **Function Conversion**: MATLAB-compatible implementations with defensive programming
4. **Algorithm Validation**: Test with synthetic data where ground truth is known
5. **Integration Testing**: Full workflow with real data
6. **Notebook Creation**: Educational content with robust execution context handling
7. **Execution Testing**: Notebook runs end-to-end from multiple working directories
8. **Publication Validation**: HTML export with all outputs and accessible visualizations

#### Implementation Best Practices
- **Start with synthetic test cases** for algorithm validation
- **Add numerical stability safeguards** for edge cases
- **Use relative imports and robust path resolution** for notebooks
- **Prefer matplotlib automatic layout** over manual positioning
- **Include comprehensive error handling** with helpful messages
- **Test execution contexts** (Jupyter, nbconvert, different working directories)

#### Testing Workflow Enhancement
```bash
# Recommended testing sequence for each phase
pytest tests/test_functions/      # Unit tests with synthetic data
pytest tests/test_integration/    # Real data integration tests  
python -m nbconvert --execute examples/notebooks/*/example.ipynb  # Notebook validation
pytest tests/test_notebooks/     # Notebook-specific tests
```

These lessons learned will significantly accelerate development in subsequent phases and improve the overall quality and robustness of the conversion.