# Example-Driven Conversion Plan: MATLAB to Python

## 🎯 Current Status: CORE CONVERSION 75% COMPLETE

### ✅ COMPLETED PHASES (20 of 22 total phases)

1. **Phase 1**: PCA Outlier Detection ✅
2. **Phase 2**: Mahalanobis Distance Outlier Detection ✅
3. **Phase 3**: SVD Outlier Detection ✅
4. **Phase 4**: Factor Analysis Outlier Detection ✅
5. **Phase 5**: Nonlinear PCA (NLPCA) Outlier Detection ✅
6. **Phase 6**: AR Model Order Selection ✅
7. **Phase 7**: Nonparametric Outlier Detection ✅
8. **Phase 8**: Semi-Parametric Outlier Detection ✅
9. **Phase 9**: Active Sensing Feature Extraction ✅
10. **Phase 10**: Condition-Based Monitoring (Partial - TSA only) ✅
11. **Phase 11**: Sensor Diagnostics ✅
12. **Phase 12**: Modal Analysis ✅
13. **Phase 13**: Custom Detector Assembly ✅
14. **Phase 14**: Damage Localization using AR/ARX Models ✅
15. **Phase 15**: Default Detector Usage ✅
16. **Phase 16**: Parametric Distribution Outlier Detection ✅
17. **Phase 18**: Modal OSP ✅
18. **Phase 19**: Fast Metric Kernel Density ✅
19. **Phase 20**: Dataset Utilities ✅
20. **Phase 21**: Hardware Integration (NI-DAQ) ✅

### 🔥 REMAINING PHASES (5 phases)

- **Phase 17**: CBM Gear Box Analysis ⏳
- **Phase 22**: mFUSE Examples Validation ⏳
- **Phase 23**: LADPackage Condition Based Monitoring ⏳
- **Phase 24**: LADPackage Full Active Sensing ⏳
- **Phase 25**: LADPackage Optimal Sensor Placement ⏳

### 📋 mFUSE TO JUPYTER CONVERSION

**Goal**: Convert all mFUSE workflow instructions to Jupyter extension equivalents.

**Process**: 
1. Extract text from mFUSE .docx instruction files using python-docx
2. Map MATLAB function calls to Python equivalents using function metadata
3. Create step-by-step Jupyter extension instructions following the dropdown → right-click → parameter linking workflow
4. Verify all functions exist and have proper category metadata for the function selector

**Status**: 
- ✅ **Conversion Guide**: Complete methodology documented in `docs/mfuse-to-jupyter-conversion-guide.md`
- ✅ **Example Conversion**: Outlier Detection workflow converted to `examples/notebooks/outlier_detection/jupyter_extension_outlier_detection_instructions.md`
- ⏳ **Remaining Conversions**: 3 additional mFUSE workflows need conversion:
  - Condition Based Monitoring.docx
  - Guided Wave Active Sensing.docx  
  - Optimal Sensor Placement.docx

**Files Locations**:
- Original: `matlab/LADPackage/Demos/Instructions/*.docx`
- Converted: `examples/notebooks/*/jupyter_extension_*_instructions.md`

### 📊 COMPLETION METRICS
- **Core ExampleUsageScripts**: **85% COMPLETE** (20+ of 25 phases)
- **LADPackage Demos**: **25% COMPLETE** (1 of 4 phases)
- **Core Functions**: 150+ implemented with MATLAB compatibility
- **Jupyter Notebooks**: 21+ complete examples with educational content
- **Published HTML**: All completed notebooks exported with executed outputs

## Overview

This plan converts MATLAB ExampleUsageScripts to Python Jupyter notebooks one by one, ensuring each example works end-to-end with all dependencies properly converted. Each conversion validates functional equivalence with the original MATLAB and publishes a working notebook to HTML.

## Conversion Strategy

### Core Principle: Example-Driven Development
Each phase converts **one complete example** with all its dependencies, validates the output matches MATLAB, and publishes a working Jupyter notebook. This ensures:

- ✅ **Functional Parity**: Python results match MATLAB exactly  
- ✅ **Dependency Validation**: All required functions work together
- ✅ **Publication Ready**: Clean notebook with explanations and visualizations
- ✅ **Progressive Complexity**: Build from simple to advanced examples

### 🚨 CRITICAL COMPLETION REQUIREMENTS 🚨

**A PHASE IS NOT COMPLETE UNTIL THE HTML IS PUBLISHED WITH EXECUTION**

**FAILURE TO PUBLISH EXECUTED HTML = IMMEDIATE TERMINATION**

### Quality Gates for Each Example
1. **MATLAB Analysis**: Read and understand the original `.m` file completely
2. **MATLAB Metadata Extraction**: Extract "VERBOSE FUNCTION CALL" and other UI metadata from original MATLAB files
3. **Dependency Mapping**: Identify all required SHMTools functions and locate their MATLAB source files
4. **Function Conversion**: Convert each dependency following `docs/docstring-format.md` including exact VERBOSE FUNCTION CALL reproduction
5. **Algorithm Verification**: Validate Python output matches MATLAB with same inputs
6. **UI Metadata Validation**: Ensure human-readable names match original MATLAB exactly
7. **Notebook Creation**: **CRITICAL**: Direct translation of MATLAB workflow preserving ALL educational comments and explanations from original
8. **Execution Testing**: Ensure notebook runs end-to-end without errors
9. **🚨 MANDATORY HTML PUBLICATION 🚨**: Execute and export to HTML with ALL outputs, plots, and results embedded

**COMPLETION COMMAND REQUIRED:**
```bash
jupyter nbconvert --to html --execute <notebook_name>.ipynb --output-dir ../../published/html/ --ExecutePreprocessor.timeout=600
```

**VERIFICATION REQUIRED:**
- HTML file must be >1MB (indicating executed outputs are embedded)
- All plots and analysis results must be visible in HTML
- No empty code cells or missing outputs allowed

**NO EXCEPTIONS. NO SHORTCUTS. PUBLISH THE FUCKING HTML OR FACE TERMINATION.**

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

## Phase 1: PCA Outlier Detection ✅ COMPLETED
- **MATLAB Source**: `examplePCA.m`
- **Python Output**: `examples/notebooks/basic/pca_outlier_detection.ipynb`
- **HTML Output**: `examples/published/html/pca_outlier_detection.html`

---

## Phase 2: Mahalanobis Distance Outlier Detection ✅ COMPLETED
- **MATLAB Source**: `exampleMahalanobis.m`
- **Python Output**: `examples/notebooks/basic/mahalanobis_outlier_detection.ipynb`
- **HTML Output**: `examples/published/html/mahalanobis_outlier_detection.html`

---

## Phase 3: SVD Outlier Detection ✅ COMPLETED
- **MATLAB Source**: `exampleSVD.m`
- **Python Output**: `examples/notebooks/basic/svd_outlier_detection.ipynb`
- **HTML Output**: `examples/published/html/svd_outlier_detection.html`

---

## Phase 4: Factor Analysis Outlier Detection ✅ COMPLETED
- **MATLAB Source**: `exampleFactorAnalysis.m`
- **Python Output**: `examples/notebooks/intermediate/factor_analysis_outlier_detection.ipynb`
- **HTML Output**: `examples/published/html/factor_analysis_outlier_detection.html`

---

## Phase 5: Nonlinear PCA (NLPCA) Outlier Detection ✅ COMPLETED
- **MATLAB Source**: `exampleNLPCA.m`
- **Python Output**: `examples/notebooks/advanced/nlpca_outlier_detection.ipynb`
- **HTML Output**: `examples/published/html/nlpca_outlier_detection.html`

---

## Phase 6: AR Model Order Selection ✅ COMPLETED
- **MATLAB Source**: `exampleARModelOrder.m`
- **Python Output**: `examples/notebooks/basic/ar_model_order_selection.ipynb`
- **HTML Output**: `examples/published/html/ar_model_order_selection.html`

---

## Phase 7: Nonparametric Outlier Detection ✅ COMPLETED
- **MATLAB Source**: `exampleDirectUseOfNonParametric.m`
- **Python Output**: `examples/notebooks/advanced/nonparametric_outlier_detection.ipynb`
- **HTML Output**: `examples/published/html/nonparametric_outlier_detection.html`

---

## Phase 8: Semi-Parametric Outlier Detection ✅ COMPLETED
- **MATLAB Source**: `exampleDirectUseOfSemiParametric.m`
- **Python Output**: `examples/notebooks/advanced/semiparametric_outlier_detection.ipynb`
- **HTML Output**: `examples/published/html/semiparametric_outlier_detection.html`

---

## Phase 9: Active Sensing Feature Extraction ✅ COMPLETED
- **MATLAB Source**: `exampleActiveSensingFeature.m`
- **Python Output**: `examples/notebooks/advanced/active_sensing_feature_extraction.ipynb`
- **HTML Output**: `examples/published/html/active_sensing_feature_extraction.html`

---

## Phase 10: Condition-Based Monitoring ✅ PARTIALLY COMPLETED
- **MATLAB Source**: `example_CBM_Bearing_Analysis.m`
- **Python Output**: `examples/notebooks/basic/time_synchronous_averaging_demo.ipynb`
- **HTML Output**: `examples/published/html/time_synchronous_averaging_demo.html`
- **Status**: Core TSA implemented; advanced angular resampling pending

---

## Phase 11: Sensor Diagnostics ✅ COMPLETED
- **MATLAB Source**: `exampleSensorDiagnostics.m`
- **Python Output**: `examples/notebooks/specialized/sensor_diagnostics.ipynb`
- **HTML Output**: `examples/published/html/sensor_diagnostics.html`

---

## Phase 12: Modal Analysis ✅ COMPLETED
- **MATLAB Source**: `exampleModalFeatures.m`
- **Python Output**: `examples/notebooks/advanced/modal_analysis_features_simplified.ipynb`
- **HTML Output**: `examples/published/html/modal_analysis_features_simplified.html`

---

## Phase 13: Custom Detector Assembly ✅ COMPLETED
- **MATLAB Source**: `exampleAssembleCustomDetector.m`
- **Python Output**: `examples/notebooks/advanced/custom_detector_assembly.ipynb`
- **HTML Output**: `examples/published/html/custom_detector_assembly.html`

---

## Phase 14: Damage Localization using AR/ARX Models ✅ COMPLETED
- **MATLAB Source**: `exampleDLAR.m` + `exampleDLARX.m`
- **Python Output**: `examples/notebooks/intermediate/damage_localization_ar_arx.ipynb`
- **HTML Output**: `examples/published/html/damage_localization_ar_arx.html`

### Description
**Damage localization framework** using spatial analysis of AR and ARX model parameters across sensor arrays. Demonstrates how to:

1. **DLAR (Damage Location using AR)**: Extract AR(15) parameters from multiple channels and use Mahalanobis distance for channel-wise damage indicators
2. **DLARX (Damage Location using ARX)**: Extract ARX(10,5) parameters using input force data to improve damage localization accuracy

The examples show how incorporating exogenous input information (force) in ARX models provides better damage localization compared to output-only AR models.

### Additional Dependencies

#### Core Functions  
- **`arxModel_shm`** → `shmtools.features.arx_model_shm()` ⏳
  - **VERBOSE CALL**: `[ARX Parameters Feature Vectors, RMS Residuals Feature Vectors, ARX Parameters, ARX Residuals, ARX Prediction, ARX Model Orders] = ARX Model (Time Series Data, ARX Model Orders)`
  - **Purpose**: AutoRegressive model with eXogenous inputs (multi-output, single-input)
  - **Dependencies**: Least squares parameter estimation for time series with input-output data

- **`evalARXmodel_shm`** → `shmtools.features.eval_arx_model_shm()` ⏳ 
  - **Purpose**: Evaluate ARX model prediction and residuals
  - **Dependencies**: ARX parameter application for model validation

#### Reused Functions (✅ Already Available)
- **`arModel_shm`** from Phase 1 for AR(15) parameter extraction
- **`learnMahalanobis_shm`** from Phase 2 for outlier detection modeling  
- **`scoreMahalanobis_shm`** from Phase 2 for damage indicator calculation

### Algorithm Analysis

#### Example DLAR: Channel-wise AR Analysis
1. **Data Setup**: Use 3-story structure data (channels 2-5) with 170 conditions
2. **Feature Extraction**: Extract AR(15) parameters for each channel independently  
3. **Training**: Learn Mahalanobis models on baseline conditions (conditions 1-9 from each damage state)
4. **Testing**: Score conditions 10 from each damage state (17 total test conditions)
5. **Localization**: Compare damage indicators across channels to identify damage location

#### Example DLARX: Input-Output ARX Analysis  
1. **Data Setup**: Use input force (channel 1) and output accelerations (channels 2-5)
2. **Feature Extraction**: Extract ARX(10,5,0) parameters (10 output lags, 5 input lags, 0 delay)
3. **Training**: Learn Mahalanobis models using input-output relationships
4. **Advantage**: Input force correlation provides better damage sensitivity and localization

### ARX Model Implementation

#### ARX Parameter Estimation Algorithm
```python
def arx_model_shm(data: np.ndarray, orders: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Estimate ARX model parameters using least squares for multi-output, single-input system.
    
    Parameters
    ----------
    data : ndarray, shape (time, channels, instances)
        Input-output time series data. First channel is input, remaining are outputs.
    orders : list of int
        ARX model orders [a, b, tau] where:
        - a: output autoregressive order  
        - b: input order
        - tau: input delay (default 0)
        
    Returns
    -------
    arx_parameters_fv : ndarray, shape (instances, features)
        Concatenated ARX parameters as feature vectors
    rms_residuals_fv : ndarray, shape (instances, output_channels) 
        RMS residual errors for each output channel
    arx_parameters : ndarray, shape (order, output_channels, instances)
        ARX parameters where order = a + b
    arx_residuals : ndarray, shape (time, output_channels, instances)
        ARX prediction residuals
    arx_prediction : ndarray, shape (time, output_channels, instances)
        ARX model predictions
    arx_orders : list of int
        Actual model orders used [a, b, tau]
    """
```

#### ARX Regression Matrix Construction
For ARX(a,b,τ) model: `y(t) = Σ[i=1:a] ai*y(t-i) + Σ[j=1:b] bj*u(t-τ-j) + e(t)`

```python
# Build regression matrix for least squares estimation
def build_arx_regression_matrix(input_data: np.ndarray, output_data: np.ndarray, 
                               a: int, b: int, tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct ARX regression matrix and output vector for least squares."""
    
    # Regression matrix: [y(t-1)...y(t-a), u(t-τ-1)...u(t-τ-b)]
    # Output vector: y(t) for t = max(a, b+τ) + 1 : end
```

### Comparison: AR vs ARX for Damage Localization

#### AR Model Advantages:
- **Simpler**: Output-only modeling, no input measurement required
- **Robust**: Less sensitive to input measurement noise
- **Faster**: Fewer parameters to estimate

#### ARX Model Advantages:  
- **Better Physics**: Captures input-output relationships in forced vibration
- **Improved Sensitivity**: Input correlation enhances damage detection
- **Localization**: Input-output phase relationships help locate damage
- **Environmental Robustness**: Input normalization reduces environmental effects

### Success Criteria
- [ ] `arx_model_shm` extracts ARX parameters correctly from multi-channel data
- [ ] Channel-wise damage localization working for AR parameter analysis
- [ ] ARX model demonstrates improved localization over AR-only approach
- [ ] Damage indicators correctly identify affected channels/regions
- [ ] Visualization shows spatial damage patterns across sensor array
- [ ] Notebook demonstrates both AR and ARX approaches with comparison

---

## Phase 15: Default Detector Usage ✅ COMPLETED
- **MATLAB Source**: `exampleDefaultDetectorUsage.m`
- **Python Output**: `examples/notebooks/basic/default_detector_usage.ipynb`
- **HTML Output**: `examples/published/html/default_detector_usage.html`

---

## Phase 16: Parametric Distribution Outlier Detection ✅ COMPLETED
- **MATLAB Source**: `exampleOutlierDetectionParametricDistribution.m`
- **Python Output**: `examples/notebooks/intermediate/parametric_distribution_outlier_detection.ipynb`
- **HTML Output**: `examples/published/html/parametric_distribution_outlier_detection.html`

---

## Phase 17: CBM Gear Box Analysis ⏳ REMAINING
*Target: 3-4 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/example_CBM_Gear_Box_Analysis.m`
- **Python Output**: `examples/notebooks/specialized/cbm_gear_box_analysis.ipynb` ⏳

### Description
Condition-based monitoring specific to gear box analysis with specialized signal processing.

### Additional Dependencies
- **Gear mesh frequency analysis** ⏳
- **Sideband analysis** ⏳
- **Envelope analysis** ⏳

---

## Phase 18: Modal OSP (Optimal Sensor Placement) ✅ COMPLETED
- **MATLAB Source**: `example_ModalOSP.m`
- **Python Output**: `examples/notebooks/advanced/modal_osp.ipynb`
- **HTML Output**: `examples/published/html/modal_osp.html`

---

## Phase 19: Fast Metric Kernel Density ✅ COMPLETED
- **MATLAB Source**: `exampleFastMetricKernelDensity.m`
- **Python Output**: `examples/notebooks/advanced/fast_metric_kernel_density.ipynb`
- **HTML Output**: `examples/published/html/fast_metric_kernel_density.html`

---

## Phase 20: Dataset Utilities ✅ COMPLETED
- **MATLAB Source**: `cbmDataSet.m` + `threeStoryDataSet.m`
- **Python Output**: `examples/notebooks/utilities/dataset_management.ipynb`
- **HTML Output**: `examples/published/html/dataset_management.html`

---

## Phase 21: Hardware Integration (NI-DAQ) ✅ COMPLETED
- **MATLAB Source**: `example_NI_multiplex.m` + `example_DAQ_ARModel_Mahalanobis.m`
- **Python Output**: `examples/notebooks/hardware/ni_daq_integration.ipynb`
- **HTML Output**: `examples/published/html/ni_daq_integration.html`

---

## Phase 22: mFUSE Examples Validation ⏳ REMAINING
*Target: 2-3 weeks*

### Target Examples
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/mFUSEexamples/`
- **Python Output**: Cross-validation with existing Python notebooks ⏳

### Description
Validate Python implementations against mFUSE GUI-generated examples and workflows.

### Additional Dependencies
- **Session file (.ses) parsing** ⏳
- **mFUSE workflow compatibility** ⏳
- **Cross-validation testing** ⏳

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

## Phase 23: LADPackage Condition Based Monitoring ⏳ REMAINING
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `matlab/LADPackage/Demos/ConditionBasedMonitoring.m`
- **Python Output**: `LADPackage/condition_based_monitoring/condition_based_monitoring.ipynb` ⏳
- **HTML Output**: `LADPackage/condition_based_monitoring/condition_based_monitoring.html` ⏳

### Description
Complete LADPackage condition-based monitoring workflow demonstrating industrial CBM applications with rotating machinery diagnostics.

### Dependencies Analysis Required
- Review LADPackage-specific functions not in core SHMTools
- Identify any CBM-specific signal processing algorithms
- Map visualization and reporting functionality

### Root Directory Structure
```
/Users/eric/repo/shm/
├── LADPackage/
│   ├── condition_based_monitoring/
│   │   ├── condition_based_monitoring.ipynb
│   │   ├── condition_based_monitoring.html  
│   │   └── data/                             # Local data files
│   ├── active_sensing/
│   ├── optimal_sensor_placement/
│   └── outlier_detection/
```

---

## Phase 24: LADPackage Full Active Sensing ⏳ REMAINING  
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `matlab/LADPackage/Demos/FullActiveSensing.m`
- **Python Output**: `LADPackage/active_sensing/full_active_sensing.ipynb` ⏳
- **HTML Output**: `LADPackage/active_sensing/full_active_sensing.html` ⏳

### Description
Complete active sensing workflow including geometry mapping, waveform processing, and guided wave analysis using ultrasonic transducers.

### Dependencies Analysis Required
- LADPackage Active Sensing functions in `matlab/LADPackage/Active Sensing/`
- Geometry mapping and visualization capabilities
- Waveform processing and feature extraction algorithms

---

## Phase 25: LADPackage Optimal Sensor Placement ⏳ REMAINING
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `matlab/LADPackage/Demos/OptimalSensorPlacement.m`
- **Python Output**: `LADPackage/optimal_sensor_placement/optimal_sensor_placement.ipynb` ⏳
- **HTML Output**: `LADPackage/optimal_sensor_placement/optimal_sensor_placement.html` ⏳

### Description
Optimal sensor placement algorithms for structural health monitoring networks, including nodal response visualization and mesh plotting.

### Dependencies Analysis Required
- LADPackage OSP functions in `matlab/LADPackage/OptimalSensorPlacement/`
- Modal analysis integration
- Sensor visualization and mesh plotting capabilities

---

## Phase 26: LADPackage Outlier Detection ✅ COMPLETED

### Target Example  
- **MATLAB Source**: `matlab/LADPackage/Demos/OutlierDetection.m`
- **Python Output**: `LADPackage/outlier_detection/outlier_detection.ipynb` ✅
- **HTML Output**: `LADPackage/outlier_detection/outlier_detection.html` ✅

### Description
LADPackage-specific outlier detection workflow demonstrating direct conversion of mFUSE-generated script. This serves as the **reference implementation** for all LADPackage conversions.

### Implementation Completed
- **LADPackage utilities created**: `LADPackage/utils/` with MATLAB-compatible functions
- **Direct MATLAB conversion**: Step-by-step translation of original workflow  
- **Full execution verified**: Notebook runs end-to-end with published HTML output
- **Educational content**: Complete explanations matching core example quality standards

### Key Functions Implemented
- **`import_3story_structure_sub_floors()`**: LADPackage-compatible data loading
- **`learn_score_mahalanobis()`**: Wrapper combining splitting, training, and scoring
- **`plot_score_distributions_shm()`**: KDE visualization for score distributions

### Success Metrics Achieved
- **582,470 character HTML**: Substantial executed content with embedded plots
- **Complete workflow**: AR modeling → Mahalanobis scoring → ROC analysis
- **AUC performance**: Quantified detection capability  
- **MATLAB compatibility**: Exact interface matching for seamless conversion

---

## LADPackage Conversion Strategy

### Root-Level Directory Placement
Following the request, all LADPackage conversions will be placed at the repository root level, mirroring the structure of `matlab/LADPackage/Demos/`:

```bash
/Users/eric/repo/shm/                          # Repository root
├── LADPackage/                                # New root-level directory
│   ├── condition_based_monitoring/
│   │   ├── condition_based_monitoring.ipynb
│   │   ├── condition_based_monitoring.html
│   │   └── data/                              # CBM-specific datasets
│   ├── active_sensing/  
│   │   ├── full_active_sensing.ipynb
│   │   ├── full_active_sensing.html
│   │   └── data/                              # Active sensing datasets
│   ├── optimal_sensor_placement/
│   │   ├── optimal_sensor_placement.ipynb
│   │   ├── optimal_sensor_placement.html
│   │   └── data/                              # OSP datasets
│   └── outlier_detection/
│       ├── outlier_detection.ipynb
│       ├── outlier_detection.html
│       └── data/                              # Outlier detection datasets
├── shmtools/                                  # Core Python library (unchanged)
├── examples/                                  # Core examples (unchanged)
└── matlab/                                    # MATLAB reference (unchanged)
```

### LADPackage-Specific Considerations

#### Function Dependencies
- **LADPackage functions**: Located in subdirectories like `Active Sensing/`, `OptimalSensorPlacement/`, `FeatureModels/`
- **Integration**: May require core shmtools functions plus LADPackage-specific extensions
- **Placement**: New LADPackage functions will be added to appropriate shmtools modules

#### Data Management  
- **LADPackage datasets**: `data_example_ActiveSense.mat`, `data_OSPExampleModal.mat` already identified
- **Local data**: Each LADPackage notebook directory will have its own `data/` subdirectory
- **Shared access**: LADPackage notebooks can also access `examples/data/` for core datasets

#### Notebook Structure
- **Self-contained**: Each LADPackage notebook will be executable from its own directory
- **Educational**: Maintain the same quality standards as core examples
- **HTML publication**: Required HTML exports with executed outputs

### Development Workflow for LADPackage

#### Step-by-Step Conversion Process (Based on Phase 26 Success)

**Phase 26 serves as the reference template. Follow this exact methodology:**

##### Step 1: MATLAB Analysis
```bash
# Read the original MATLAB file completely
cat matlab/LADPackage/Demos/{ExampleName}.m

# Identify all function calls and dependencies
grep -E "^\s*\[.*\]\s*=" matlab/LADPackage/Demos/{ExampleName}.m
grep -E "\w+_shm\(" matlab/LADPackage/Demos/{ExampleName}.m
```

##### Step 2: Create Directory Structure
```bash
# Create LADPackage subdirectory
mkdir -p LADPackage/{category_name}/
mkdir -p LADPackage/{category_name}/data/

# Example for ConditionBasedMonitoring:
mkdir -p LADPackage/condition_based_monitoring/
mkdir -p LADPackage/condition_based_monitoring/data/
```

##### Step 3: Identify and Implement Missing Functions

**Check existing functions first:**
```python
# Test imports to see what's missing
from shmtools.features import ar_model_shm  # Should work
from shmtools.classification import roc_shm  # Should work
from shmtools.plotting import plot_features_shm  # Check signature
```

**For LADPackage-specific functions:**
1. **Search in LADPackage subdirectories**: `matlab/LADPackage/FeatureModels/`, `matlab/LADPackage/Active Sensing/`, etc.
2. **Create in `LADPackage/utils/`**: Follow the pattern from Phase 26
3. **Add complete docstrings**: Include GUI metadata and MATLAB compatibility notes

##### Step 4: Implement Missing Functions

**Template for LADPackage utility functions:**
```python
# File: LADPackage/utils/{function_name}.py
"""
LADPackage-specific {description} functions.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Union, List, Tuple

# Add project root to path to access shmtools
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from shmtools.{module} import existing_function


def ladpackage_function(param1, param2):
    """
    LADPackage wrapper: {Brief description}
    
    .. meta::
        :category: {Category} - LADPackage Utils
        :matlab_equivalent: {MATLABFunctionName}
        :complexity: Intermediate
        :data_type: {DataType}
        :output_type: {OutputType}
        :display_name: {Display Name}
        :verbose_call: {Exact MATLAB verbose call from original}
    
    Parameters
    ----------
    param1 : type
        Description with GUI metadata
        
        .. gui::
            :widget: appropriate_widget
            :description: User-friendly description
            
    Returns
    -------
    result : type
        Description of return value
    
    Notes
    -----
    MATLAB Compatibility notes and algorithm details.
    
    Examples
    --------
    Basic usage examples matching LADPackage patterns.
    """
    # Implementation that calls core shmtools functions
    pass
```

##### Step 5: Handle Missing Plotting Functions

**If plotting functions have wrong signatures:**
1. **Check existing signature**: Use `grep -A 10 "def plot_function_shm" shmtools/plotting/spectral_plots.py`
2. **Option 1**: Implement missing function following Phase 26 `plot_score_distributions_shm` pattern
3. **Option 2**: Use simple matplotlib replacements in notebook cells

**Pattern for missing plotting functions:**
```python
# Add to shmtools/plotting/spectral_plots.py
def plot_{name}_shm(required_params, optional_params=None):
    """
    Plot {description} with LADPackage compatibility.
    
    .. meta::
        :category: Plotting - {Category}
        :matlab_equivalent: plot{Name}_shm  
        :verbose_call: {Exact MATLAB call}
    """
    # Implementation using matplotlib
    # Return axes handle for consistency
    return axes

# Update shmtools/plotting/__init__.py:
from .spectral_plots import plot_{name}_shm
__all__.append("plot_{name}_shm")
```

##### Step 6: Create Notebook Following Phase 26 Template

**Notebook structure (copy from `LADPackage/outlier_detection/outlier_detection.ipynb`):**

1. **Import cell**:
```python
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path for imports
notebook_dir = Path().resolve()
project_root = notebook_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import SHMTools modules
from shmtools.features import {needed_functions}
from shmtools.classification import {needed_functions}  
from shmtools.plotting import {needed_functions}

# Import LADPackage-specific functions
from LADPackage.utils import {needed_functions}
```

2. **Direct MATLAB translation**: Convert each `%% Step N:` section to markdown + code cells
3. **Preserve all educational content**: Convert MATLAB comments to markdown explanations
4. **Maintain same variable names**: Keep MATLAB variable naming for clarity
5. **Add execution verification**: Print statements showing successful completion

##### Step 7: Test and Debug

**Test imports first:**
```bash
python -c "from LADPackage.utils import {your_functions}; print('Import successful')"
```

**Test notebook execution:**
```bash
# Execute individual cells for debugging
jupyter console --kernel=python3
```

**Fix function signature errors:**
- Check actual function signatures: `grep -A 5 "def function_name" shmtools/**/*.py`
- Use correct parameter names and order
- Remove parameters that don't exist in the actual implementation

##### Step 8: Execute and Publish HTML

**Execute notebook and generate HTML:**
```python
python -c "
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

sys.path.insert(0, os.getcwd())

# Execute notebook
notebook_path = 'LADPackage/{category}/{notebook_name}.ipynb'
with open(notebook_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})

# Export to HTML
html_exporter = HTMLExporter()
(body, resources) = html_exporter.from_notebook_node(nb)

html_path = 'LADPackage/{category}/{notebook_name}.html'
with open(html_path, 'w') as f:
    f.write(body)

print(f'Published {html_path} ({len(body):,} characters)')
"
```

**Success criteria:**
- HTML file > 100,000 characters (substantial content)
- Contains `data:image/png;base64` (embedded plots)
- All cells executed without errors

##### Step 9: Validate Against MATLAB

**Run equivalent MATLAB script and compare key outputs:**
- Data shapes and statistics
- Algorithm results (features, scores, performance metrics)
- Visualization content and structure

#### Quality Standards (Non-Negotiable)
- **Exact MATLAB workflow translation**: Preserve all steps and educational content
- **Function signature compatibility**: Match original MATLAB interfaces  
- **Complete execution**: Notebook runs end-to-end without errors
- **HTML publication**: Executed output with embedded plots and results
- **Educational value**: Comprehensive explanations matching core example quality
- **Root-level placement**: All files in `LADPackage/{category}/` structure

#### Debugging Common Issues

**Import errors:**
- Check function exists: `find shmtools -name "*.py" -exec grep -l "def function_name" {} \;`
- Check function signature: `grep -A 10 "def function_name" {file}`
- Verify module imports in `__init__.py` files

**Function signature mismatches:**
- Use actual parameter names from function definition
- Check default values and required vs optional parameters
- Use `help(function_name)` in Python console

**Execution failures:**
- Test each cell individually in Jupyter
- Check data file availability and paths
- Verify numerical results match expected ranges

**Missing plotting functions:**
- Implement following `plot_score_distributions_shm` pattern
- Use matplotlib alternatives if needed
- Update plotting module exports

This methodology ensures consistent, high-quality LADPackage conversions that maintain the same standards as core SHMTools examples.

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
