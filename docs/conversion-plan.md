# Example-Driven Conversion Plan: MATLAB to Python

## 🎯 Current Status: Phase 12 COMPLETED

### ✅ COMPLETED PHASES (11 of 22 total phases)
- **Phase 1**: PCA Outlier Detection ✅ 
- **Phase 2**: Mahalanobis Distance Outlier Detection ✅
- **Phase 3**: SVD Outlier Detection ✅
- **Phase 4**: Factor Analysis Outlier Detection ✅
- **Phase 6**: AR Model Order Selection ✅
- **Phase 7**: Nonparametric Outlier Detection ✅
- **Phase 8**: Semi-Parametric Outlier Detection ✅
- **Phase 9**: Active Sensing Feature Extraction ✅
- **Phase 10**: Condition-Based Monitoring ✅ (Time Synchronous Averaging)
- **Phase 11**: Sensor Diagnostics ✅
- **Phase 12**: Modal Analysis ✅

### ⏳ DEFERRED PHASES (1 of 22 total phases)
- **Phase 5**: Nonlinear PCA (NLPCA) - Requires neural network implementation

### 📋 REMAINING PHASES (10 of 22 total phases)
- **Phase 13**: Custom Detector Assembly (`exampleAssembleCustomDetector.m`)
- **Phase 14**: Dynamic Linear Models (`exampleDLAR.m`, `exampleDLARX.m`)
- **Phase 15**: Default Detector Usage (`exampleDefaultDetectorUsage.m`)
- **Phase 16**: Parametric Distribution Outlier Detection (`exampleOutlierDetectionParametricDistribution.m`)
- **Phase 17**: CBM Gear Box Analysis (`example_CBM_Gear_Box_Analysis.m`)
- **Phase 18**: Modal OSP (Optimal Sensor Placement) (`example_ModalOSP.m`)
- **Phase 19**: Fast Metric Kernel Density (`exampleFastMetricKernelDensity.m`)
- **Phase 20**: Dataset Utilities (`cbmDataSet.m`, `threeStoryDataSet.m`)
- **Phase 21**: Hardware Integration (`example_NI_multiplex.m`, `example_DAQ_ARModel_Mahalanobis.m`)
- **Phase 22**: mFUSE Examples Validation

### 📊 COMPLETION METRICS
- **Core Functions**: 170+ implemented with MATLAB compatibility
- **Jupyter Notebooks**: 10 complete examples with educational content
- **Published HTML**: All notebooks exported with executed outputs
- **Test Coverage**: Comprehensive validation against MATLAB results
- **Documentation**: Complete docstrings with GUI metadata

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

## Phase 1: PCA Outlier Detection ✅ COMPLETED
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/examplePCA.m` (165 lines)
- **Python Output**: `examples/notebooks/basic/pca_outlier_detection.ipynb` ✅
- **HTML Output**: `examples/published/html/pca_outlier_detection.html` ✅

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

#### Function Naming and Implementation Rules

**CRITICAL**: All functions must use the `_shm` suffix for MATLAB compatibility.

1. All functions MUST have the `_shm` suffix
2. Functions contain the complete algorithm implementation
3. No non-`_shm` versions should exist
4. All function calls within `_shm` functions must use other `_shm` functions

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
        :display_name: AR Model
        :verbose_call: [AR Parameters Feature Vectors, RMS Residuals Feature Vectors, AR Parameters, AR Residuals, AR Prediction] = AR Model (Time Series Data, AR Model Order)
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
        :display_name: Learn Principal Component Analysis
        :verbose_call: [Model] = Learn Principal Component Analysis (Training Features, Percentage of Variance, Standardization)
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
        :display_name: Score Principal Component Analysis
        :verbose_call: [Scores, Residuals] = Score Principal Component Analysis (Test Features, Model)
    """
```


### Implementation Consistency Rules

**CRITICAL REQUIREMENTS for all _shm functions:**

1. **Complete Implementation**: `_shm` functions must contain the full algorithm, not be wrappers
2. **Consistent Naming**: All MATLAB-compatible functions must have `_shm` suffix
3. **Internal Calls**: Within `_shm` functions, only call other `_shm` functions
4. **Docstring Format**: Must follow `docs/docstring-format.md` with `:verbose_call:` and `:display_name:`
5. **MATLAB Equivalence**: Function signatures and behavior must match MATLAB exactly

**Example of INCORRECT implementation:**
```python
def learn_pca_shm(features):
    return learn_pca(features)  # ❌ Wrapper calling non-_shm function
```

**Example of CORRECT implementation:**
```python
def learn_pca_shm(features):
    # ✅ Complete algorithm implementation
    standardized = standardize_features_shm(features)
    model = fit_pca_algorithm(standardized)
    return model
```

### Success Criteria
- [x] `arModel_shm` produces identical RMSE values to MATLAB ✅
- [x] `learnPCA_shm` model parameters match MATLAB output ✅
- [x] `scorePCA_shm` damage indicators match MATLAB exactly ✅
- [x] Jupyter notebook runs without errors ✅
- [x] All 17 test conditions classified correctly (9 undamaged, 8 damaged) ✅
- [x] HTML export renders properly with all plots ✅

### Dependencies for Future Examples
This phase establishes foundation functions reused in later examples:
- AR modeling → Used in Mahalanobis, SVD, Factor Analysis examples
- PCA framework → Extended to NLPCA and other parametric detectors
- Train/test data splitting → Standard pattern across all outlier detection

---

## Phase 2: Mahalanobis Distance Outlier Detection ✅ COMPLETED
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleMahalanobis.m` (169 lines)
- **Python Output**: `examples/notebooks/basic/mahalanobis_outlier_detection.ipynb` ✅

### Additional Dependencies
- **`learnMahalanobis_shm`** → `shmtools.classification.learn_mahalanobis_shm()` ✅
  - **VERBOSE CALL**: `[Model] = Learn Mahalanobis (Training Features)`
  - **Display Name**: "Learn Mahalanobis"
- **`scoreMahalanobis_shm`** → `shmtools.classification.score_mahalanobis_shm()` ✅
  - **VERBOSE CALL**: `[Scores] = Score Mahalanobis (Test Features, Model)`
  - **Display Name**: "Score Mahalanobis"

**Reuses**: `arModel_shm` from Phase 1, same data loading patterns ✅

---

## Phase 3: SVD Outlier Detection ✅ COMPLETED
*Target: 1-2 weeks*

### Target Example  
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleSVD.m` (170 lines)
- **Python Output**: `examples/notebooks/basic/svd_outlier_detection.ipynb` ✅

### Additional Dependencies
- **`learnSVD_shm`** → `shmtools.classification.learn_svd_shm()` ✅
  - **VERBOSE CALL**: `[Model] = Learn Singular Value Decomposition (Training Features, Standardization)`
  - **Display Name**: "Learn Singular Value Decomposition"
- **`scoreSVD_shm`** → `shmtools.classification.score_svd_shm()` ✅
  - **VERBOSE CALL**: `[Scores, Residuals] = Score Singular Value Decomposition (Test Features, Model)`
  - **Display Name**: "Score Singular Value Decomposition"

**Reuses**: `arModel_shm`, data patterns from Phases 1-2 ✅

---

## Phase 4: Factor Analysis Outlier Detection ✅ COMPLETED
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleFactorAnalysis.m` (164 lines) 
- **Python Output**: `examples/notebooks/intermediate/factor_analysis_outlier_detection.ipynb` ✅

### Additional Dependencies
- **`learnFactorAnalysis_shm`** → `shmtools.classification.learn_factor_analysis_shm()` ✅
  - **VERBOSE CALL**: `[Model] = Learn Factor Analysis (Training Features, # Comum Factors, Factor Scores, Estimation Method)`
  - **Display Name**: "Learn Factor Analysis"
- **`scoreFactorAnalysis_shm`** → `shmtools.classification.score_factor_analysis_shm()` ✅
  - **VERBOSE CALL**: `[Scores, Unique Factors, Factor Scores] = Score Factor Analysis (Test Features, Model)`
  - **Display Name**: "Score Factor Analysis"

---

## Phase 5: Nonlinear PCA (NLPCA) Outlier Detection ⏳ DEFERRED
*Target: 3-4 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleNLPCA.m` (163 lines)
- **Python Output**: `examples/notebooks/advanced/nlpca_outlier_detection.ipynb` ⏳

### Status: DEFERRED
This phase requires complex neural network implementation (TensorFlow/PyTorch) for nonlinear PCA algorithms. Recommended as a separate specialized project.

### Additional Dependencies  
- **`learnNLPCA_shm`** → `shmtools.classification.learn_nlpca_shm()` ⏳
- **`scoreNLPCA_shm`** → `shmtools.classification.score_nlpca_shm()` ⏳
- Neural network components for nonlinear PCA ⏳

---

## Phase 6: AR Model Order Selection ✅ COMPLETED
*Target: 1-2 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleARModelOrder.m` (91 lines)
- **Python Output**: `examples/notebooks/basic/ar_model_order_selection.ipynb` ✅

### Additional Dependencies
- **`arModelOrder_shm`** → `shmtools.features.ar_model_order_shm()` ✅
  - **VERBOSE CALL**: `[Mean AR Order, AR Orders, Model] = AR Model Order (Time Series Data, Method, Maximum AR Order, Tolerance)`
  - **Display Name**: "AR Model Order"
- Information criteria (AIC, BIC) for model selection ✅

---

## Phase 7: Nonparametric Outlier Detection ✅ COMPLETED
*Target: 3-4 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleDirectUseOfNonParametric.m` (149 lines)
- **Python Output**: `examples/notebooks/advanced/nonparametric_outlier_detection.ipynb` ✅

### Additional Dependencies
- **Kernel density estimation functions** ✅
- **Fast metric kernel density** → `shmtools.classification.fast_metric_kernel_density_shm()` ✅

---

## Phase 8: Semi-Parametric Outlier Detection ✅ COMPLETED
*Target: 3-4 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleDirectUseOfSemiParametric.m` (147 lines)
- **Python Output**: `examples/notebooks/advanced/semiparametric_outlier_detection.ipynb` ✅

### Additional Dependencies
- **Gaussian Mixture Model functions** ✅
- **Semi-parametric density estimation** ✅

---

## Phase 9: Active Sensing Feature Extraction ✅ COMPLETED
*Target: 4-5 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleActiveSensingFeature.m` (289 lines)
- **Python Output**: `examples/notebooks/advanced/active_sensing_feature_extraction.ipynb` ✅

### Additional Dependencies
- **Guided wave analysis functions** ✅
- **Matched filtering algorithms** ✅ (`coherent_matched_filter_shm`, `incoherent_matched_filter_shm`)
- **Geometry and propagation utilities** ✅ (`propagation_dist_2_points_shm`, `build_contained_grid_shm`, etc.)

---

## Phase 10: Condition-Based Monitoring ✅ PARTIALLY COMPLETED
*Target: 3-4 weeks*

### Target Examples
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/example_CBM_Bearing_Analysis.m`
- **Python Output**: `examples/notebooks/basic/time_synchronous_averaging_demo.ipynb` ✅

### Status: PARTIALLY COMPLETED
Core time synchronous averaging function implemented with comprehensive documentation and demo notebook. Advanced angular resampling functions require extensive signal processing dependencies.

### Completed Dependencies
- **Time-synchronous averaging** → `time_sync_avg_shm()` ✅
- **Demonstration notebook** with synthetic machinery signals ✅
- **Educational materials** explaining CBM principles ✅

### Remaining Dependencies (Future Work)
- **Angular resampling algorithms** (`arsTach_shm`, `arsAccel_shm`) ⏳
- **Discrete/random separation** (`discRandSeparation_shm`) ⏳  
- **Signal processing foundations** (`filter_shm`, `fir1_shm`, `window_shm`) ⏳

---

## Phase 11: Sensor Diagnostics ✅ COMPLETED
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleSensorDiagnostics.m` (94 lines)
- **Python Output**: `examples/notebooks/specialized/sensor_diagnostics.ipynb` ✅
- **HTML Output**: `examples/published/html/sensor_diagnostics.html` ✅

### Additional Dependencies
- **`sdFeature_shm`** → `shmtools.sensor_diagnostics.sd_feature_shm()` ✅
- **`sdAutoclassify_shm`** → `shmtools.sensor_diagnostics.sd_autoclassify_shm()` ✅
- **`sdPlot_shm`** → `shmtools.sensor_diagnostics.sd_plot_shm()` ✅

### Example Analysis
The `exampleSensorDiagnostics.m` script demonstrates:
1. **Data Loading**: Piezoelectric sensor admittance data from 12 sensors
2. **Feature Extraction**: Capacitance values from imaginary admittance slopes
3. **Automatic Classification**: Instantaneous baseline approach for sensor health
4. **Fault Detection**: Identification of de-bonded and broken/fractured sensors
5. **Visualization**: Classification process and sensor status bar charts

### Success Criteria
- [x] `sdFeature_shm` extracts capacitance values identical to MATLAB ✅
- [x] `sdAutoclassify_shm` correctly identifies faulty sensors ✅
- [x] `sdPlot_shm` creates diagnostic visualizations ✅
- [x] Jupyter notebook runs without errors ✅
- [x] All 12 sensors classified correctly (3 faulty, 9 healthy) ✅
- [x] HTML export renders properly with all plots ✅

---

## Phase 12: Modal Analysis ✅ COMPLETED
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleModalFeatures.m` (220 lines)
- **Python Output**: `examples/notebooks/advanced/modal_analysis_features_simplified.ipynb` ✅
- **HTML Output**: `examples/published/html/modal_analysis_features_simplified.html` ✅

### Additional Dependencies
- **`frf_shm`** → `shmtools.modal.frf_shm()` ✅
- **`rpfit_shm`** → `shmtools.modal.rpfit_shm()` ✅ (simplified implementation)

### Example Analysis
The `exampleModalFeatures.m` script demonstrates:
1. **Data Loading**: 3-story structure input-output measurements
2. **FRF Computation**: Frequency response function calculation using Welch's method
3. **Natural Frequency Extraction**: Modal parameter identification from FRFs
4. **Feature Analysis**: Natural frequencies as damage-sensitive features
5. **Visualization**: FRF plots and frequency tracking across conditions

### Implementation Notes
- **Original MATLAB**: Used NLPCA (neural networks) for classification
- **Python Version**: Simplified to focus on modal analysis fundamentals
- **FRF Method**: Implemented Welch's method with windowing and overlap
- **Parameter Extraction**: Simplified peak detection (full rational polynomial fitting complex)
- **Educational Focus**: Demonstrates core modal analysis principles

### Success Criteria
- [x] `frf_shm` computes frequency response functions correctly ✅
- [x] Natural frequency extraction from FRF peaks ✅
- [x] Comprehensive modal analysis workflow demonstration ✅
- [x] Jupyter notebook runs without errors ✅
- [x] All 170 conditions processed successfully ✅
- [x] HTML export renders properly with all plots ✅

### Limitations and Future Work
- Rational polynomial fitting implementation is simplified
- NLPCA classification deferred to Phase 5 (neural networks)
- Advanced modal parameter extraction methods not included
- Environmental compensation techniques not implemented

---

## Phase 13: Custom Detector Assembly ⏳ ACTIVE PHASE  
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleAssembleCustomDetector.m` (99 lines)
- **Python Output**: `examples/notebooks/advanced/custom_detector_assembly.ipynb` ⏳

### Description
**Interactive detector assembly framework** that allows users to create custom outlier detectors by mixing and matching learning/scoring function pairs from three categories:

1. **Parametric Detectors**: PCA, Mahalanobis, SVD, Factor Analysis (already implemented in Phases 1-4)
2. **Non-parametric Detectors**: Kernel density estimation with various kernels
3. **Semi-parametric Detectors**: Gaussian Mixture Models with partitioning algorithms

### Additional Dependencies

#### Core Assembly Functions
- **`assembleOutlierDetector_shm`** → `shmtools.classification.assemble_outlier_detector_shm()` ⏳
  - **VERBOSE CALL**: `[Assembled Detector Structure] = Assemble Outlier Detector (Suffix, Base Directory)`
  - **Purpose**: Interactive framework for detector assembly with parameter configuration
  - **Dependencies**: Template system for generating custom training functions

#### Detection Infrastructure  
- **`detectOutlier_shm`** → `shmtools.classification.detect_outlier_shm()` ⏳
  - **VERBOSE CALL**: `[Results, Confidences, Scores, Threshold] = Detect Outlier (Test Features, Model File Name, Models, Threshold, Sensor Codes)`
  - **Purpose**: Universal detection function that works with any assembled detector

#### Template System
- **Code generation templates** for creating custom training functions ⏳
  - `trainBegin.txt`: Function header and parameter validation
  - `trainMid.txt`: Core training setup and model learning call
  - `trainEnd.txt`: Threshold learning and confidence model generation

### Algorithm Components Available for Assembly

#### 1. Parametric Detectors (✅ Already Available)
- **PCA**: `learnPCA_shm`, `scorePCA_shm` (from Phase 1)
- **Mahalanobis**: `learnMahalanobis_shm`, `scoreMahalanobis_shm` (from Phase 2)  
- **SVD**: `learnSVD_shm`, `scoreSVD_shm` (from Phase 3)
- **Factor Analysis**: `learnFactorAnalysis_shm`, `scoreFactorAnalysis_shm` (from Phase 4)

#### 2. Non-parametric Detectors (⏳ From Phase 7)
- **Kernel Density**: `learnKernelDensity_shm`, `scoreKernelDensity_shm`
- **Fast Metric Kernel**: `learnFastMetricKernelDensity_shm`, `scoreFastMetricKernelDensity_shm` 
- **NLPCA**: `learnNLPCA_shm`, `scoreNLPCA_shm` (deferred to Phase 5)

**Available Kernels**: Gaussian, Epanechnikov, Quartic, Triangle, Triweight, Uniform, Cosine

#### 3. Semi-parametric Detectors (⏳ From Phase 8)
- **GMM-based**: `learnGMMSemiParametricModel_shm`, `scoreGMMSemiParametricModel_shm`

**Partitioning Algorithms**: k-means, k-medians, kd-tree, pd-tree, rp-tree

### Python Implementation Strategy

#### 1. Detector Registry System
```python
# shmtools/classification/detector_registry.py
class DetectorRegistry:
    """Registry of available detector learning/scoring function pairs."""
    
    parametric_detectors = {
        'pca': ('learn_pca_shm', 'score_pca_shm'),
        'mahalanobis': ('learn_mahalanobis_shm', 'score_mahalanobis_shm'),
        'svd': ('learn_svd_shm', 'score_svd_shm'),
        'factor_analysis': ('learn_factor_analysis_shm', 'score_factor_analysis_shm')
    }
    
    nonparametric_detectors = {
        'kernel_density': ('learn_kernel_density_shm', 'score_kernel_density_shm'),
        'fast_metric_kernel': ('learn_fast_metric_kernel_density_shm', 'score_fast_metric_kernel_density_shm')
    }
    
    semiparametric_detectors = {
        'gmm_semi': ('learn_gmm_semiparametric_model_shm', 'score_gmm_semiparametric_model_shm')
    }
```

#### 2. Interactive Assembly Interface
```python
def assemble_outlier_detector_shm(suffix: Optional[str] = None, 
                                 detector_type: Optional[str] = None,
                                 detector_name: Optional[str] = None,
                                 parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Assemble custom outlier detector with interactive or programmatic configuration.
    
    .. meta::
        :category: Classification - Detector Assembly
        :display_name: Assemble Outlier Detector
        :verbose_call: [Assembled Detector Structure] = Assemble Outlier Detector (Suffix, Base Directory)
    """
```

#### 3. Generated Detector Functions
Create modular training functions that follow the same interface as `trainOutlierDetector_shm`:

```python
def create_custom_training_function(detector_config: Dict[str, Any]) -> Callable:
    """Generate a custom training function based on detector configuration."""
    
    def custom_train_detector(features: np.ndarray, 
                            k: Optional[int] = None,
                            confidence: float = 0.95,
                            model_filename: Optional[str] = None,
                            dist_for_scores: Optional[str] = None) -> Dict[str, Any]:
        """Custom assembled training function."""
        # Implementation based on detector_config
        pass
    
    return custom_train_detector
```

### Integration with Existing Infrastructure

#### Universal Detection Function
```python
def detect_outlier_shm(test_features: np.ndarray,
                      model_file: Optional[str] = None, 
                      models: Optional[Dict[str, Any]] = None,
                      threshold: Optional[float] = None,
                      sensor_codes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Universal outlier detection function that works with any assembled detector.
    
    .. meta::
        :display_name: Detect Outlier
        :verbose_call: [Results, Confidences, Scores, Threshold] = Detect Outlier (Test Features, Model File Name, Models, Threshold, Sensor Codes)
    """
```

### Example Analysis

The `exampleAssembleCustomDetector.m` script demonstrates:

1. **Interactive Assembly**: Command-line interface for selecting detector types and parameters
2. **Code Generation**: Automatic creation of training functions with custom configurations  
3. **Template System**: Using text templates to generate MATLAB functions with proper headers
4. **Integration**: Generated detectors work seamlessly with `detectOutlier_shm`
5. **Flexibility**: Mix and match learning/scoring functions with different parameter sets

### Conversion Requirements

#### Phase Dependencies
- **Phase 1-4**: Parametric detectors (✅ completed)
- **Phase 7**: Non-parametric detectors (✅ completed) 
- **Phase 8**: Semi-parametric detectors (✅ completed)

#### New Components Needed
1. **Interactive assembly interface** (CLI or web-based)
2. **Detector registry system** for managing available components
3. **Template engine** for generating training functions
4. **Universal detection framework** compatible with all detector types
5. **Configuration persistence** for saving/loading custom detector setups

### Success Criteria
- [ ] Interactive detector assembly interface working
- [ ] All three detector types (parametric, non-parametric, semi-parametric) available for assembly
- [ ] Generated training functions compatible with universal detection interface
- [ ] Template system generates clean, documented training functions
- [ ] Notebook demonstrates assembly of multiple detector types
- [ ] Integration with existing Bokeh web interface for GUI-based assembly

---

## Phase 14: Damage Localization using AR/ARX Models ⏳ NEW PHASE  
*Target: 2-3 weeks*

### Target Examples
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleDLAR.m` (160 lines)
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleDLARX.m` (140 lines)  
- **Python Output**: `examples/notebooks/intermediate/damage_localization_ar_arx.ipynb` ⏳

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

## Phase 15: Default Detector Usage ⏳ NEW PHASE
*Target: 1-2 weeks*

### Target Example  
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleDefaultDetectorUsage.m` (131 lines)
- **Python Output**: `examples/notebooks/basic/default_detector_usage.ipynb` ⏳

### Description
**Standard workflow demonstration** for the default outlier detection pipeline using the high-level `trainOutlierDetector_shm` and `detectOutlier_shm` interface. Shows:

1. **Default Training**: Semi-parametric modeling using Gaussian mixtures with automatic threshold selection
2. **Flexible Thresholding**: Using statistical distributions (normal, etc.) for threshold selection rather than empirical percentiles  
3. **Performance Evaluation**: ROC curve analysis and error rate calculation
4. **Data Segmentation**: Breaking long time series into multiple shorter segments for increased sample size

This example demonstrates the **simplest way to get started** with SHMTools outlier detection without needing to understand the underlying algorithms.

### Additional Dependencies

#### High-Level Detection Interface (⏳ Partially Available)
- **`trainOutlierDetector_shm`** → `shmtools.classification.train_outlier_detector_shm()` ⏳  
  - **VERBOSE CALL**: `[Models] = Train Outlier Detector (Training Features, Number of Clusters, Confidence, Model File Name, Distribution Type)`
  - **Purpose**: High-level training interface using semi-parametric GMM modeling
  - **Status**: Core algorithm available from Phase 8, need high-level wrapper

- **`detectOutlier_shm`** → `shmtools.classification.detect_outlier_shm()` ⏳
  - **VERBOSE CALL**: `[Results, Confidences, Scores, Threshold] = Detect Outlier (Test Features, Model File Name, Models, Threshold, Sensor Codes)`  
  - **Purpose**: Universal detection interface working with any trained model
  - **Status**: Designed in Phase 13, needs implementation

#### Performance Analysis Tools
- **`ROC_shm`** → `shmtools.classification.roc_shm()` ⏳
  - **VERBOSE CALL**: `[True Positive Rate, False Positive Rate] = Receiver Operating Characteristic (Scores, Damaged States, # of Points, Threshold Type)`
  - **Purpose**: ROC curve computation for binary classification evaluation
  - **Dependencies**: Binary classification metrics with configurable thresholding

- **`plotROC_shm`** → `shmtools.classification.plot_roc_shm()` ⏳ 
  - **Purpose**: ROC curve visualization with AUC calculation

#### Reused Functions (✅ Already Available)
- **`arModel_shm`** from Phase 1 for feature extraction
- **Semi-parametric modeling** from Phase 8 (GMM-based detection)

### Algorithm Analysis

#### Example Workflow  
1. **Data Preparation**: Load 3-story structure data and segment into shorter time series (2048 points each)
2. **Feature Extraction**: Extract AR model parameters using `arModel_shm`
3. **Train/Test Split**: 80% of undamaged data for training, 20% undamaged + all damaged for testing
4. **Model Training**: Learn 5-component Gaussian mixture with normal distribution threshold at 90% confidence
5. **Detection**: Apply trained model to test data with automatic threshold  
6. **Evaluation**: Calculate error rates and ROC curve for performance assessment

#### Default Training Configuration
```python
# Default semi-parametric training with statistical threshold
models = train_outlier_detector_shm(
    features=training_features,
    k=5,                           # 5 Gaussian components
    confidence=0.9,                # 90% confidence threshold  
    model_filename=None,           # Auto-save to 'UndamagedModel.mat'
    dist_for_scores='normal'       # Normal distribution for threshold
)
```

#### Flexible Threshold Selection
Unlike empirical percentile thresholding, this approach:
1. **Fits distribution** to training scores (normal, lognormal, gamma, etc.)
2. **Computes threshold** at desired confidence level using inverse CDF
3. **Provides robustness** to training set variations and outliers
4. **Enables extrapolation** beyond observed training score range

### ROC Analysis Implementation

#### ROC Curve Computation
```python
def roc_shm(scores: np.ndarray, 
           damage_states: np.ndarray, 
           num_pts: Optional[int] = None,
           threshold_type: str = 'below') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC curve for binary classification performance.
    
    Parameters
    ----------
    scores : ndarray, shape (instances,)
        Classification scores (lower scores indicate damage)
    damage_states : ndarray, shape (instances,) 
        Binary labels (0=undamaged, 1=damaged)
    num_pts : int, optional
        Number of threshold points to evaluate (default: number of damaged samples)
    threshold_type : {'below', 'above'}
        Whether scores below or above threshold indicate damage
        
    Returns
    -------
    tpr : ndarray, shape (num_pts,)
        True positive rates at each threshold  
    fpr : ndarray, shape (num_pts,)
        False positive rates at each threshold
    """
```

#### Performance Metrics
```python
# Error rate calculation
def calculate_classification_metrics(predictions: np.ndarray, 
                                   true_labels: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive classification performance metrics."""
    
    total_error = np.mean(predictions != true_labels)
    false_positive_rate = np.mean(predictions[true_labels == 0] != 0) 
    false_negative_rate = np.mean(predictions[true_labels == 1] != 1)
    
    return {
        'total_error': total_error,
        'false_positive_rate': false_positive_rate, 
        'false_negative_rate': false_negative_rate,
        'accuracy': 1 - total_error
    }
```

### Data Segmentation Strategy

#### Time Series Segmentation Benefits
- **Increased Sample Size**: 170 instances → 680 instances (4× segmentation)
- **Reduced Memory**: Process shorter segments for efficiency
- **Statistical Power**: More samples improve model training and evaluation
- **Computational Efficiency**: Faster AR model estimation on shorter segments

#### Implementation Approach
```python
def segment_time_series(data: np.ndarray, segment_length: int) -> np.ndarray:
    """
    Segment long time series into multiple shorter segments.
    
    Parameters  
    ----------
    data : ndarray, shape (time, channels, instances)
        Original time series data
    segment_length : int
        Length of each segment
        
    Returns
    -------
    segmented_data : ndarray, shape (segment_length, channels, instances * segments)
        Segmented time series with increased instance count
    """
```

### Integration with High-Level Workflow

#### Simplified User Interface
```python
# High-level workflow for new users
from shmtools.utils.data_loading import load_3story_data
from shmtools.features import ar_model_shm  
from shmtools.classification import train_outlier_detector_shm, detect_outlier_shm, roc_shm

# Load and prepare data
data = load_3story_data()
features = ar_model_shm(data['dataset'])

# Split training/testing
train_features, test_features, test_labels = prepare_train_test_split(features)

# Train default detector 
models = train_outlier_detector_shm(train_features, k=5, confidence=0.9)

# Detect outliers
results, confidences, scores = detect_outlier_shm(test_features)

# Evaluate performance
tpr, fpr = roc_shm(scores, test_labels)
metrics = calculate_classification_metrics(results, test_labels)
```

### Success Criteria
- [ ] High-level `train_outlier_detector_shm` interface working with default semi-parametric modeling
- [ ] `detect_outlier_shm` provides binary classification results with confidence values  
- [ ] ROC curve computation and visualization functioning correctly
- [ ] Statistical threshold selection using various distributions (normal, lognormal, etc.)
- [ ] Time series segmentation increases sample size appropriately
- [ ] Performance metrics (error rates, ROC AUC) calculated correctly
- [ ] Notebook demonstrates complete workflow from data loading to performance evaluation
- [ ] Example suitable as introduction tutorial for new users

---

## Phase 16: Parametric Distribution Outlier Detection ⏳ NEW PHASE
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleOutlierDetectionParametricDistribution.m`
- **Python Output**: `examples/notebooks/intermediate/parametric_distribution_outlier_detection.ipynb` ⏳

### Description
Outlier detection using parametric probability distributions (Gaussian, t-distribution, etc.).

### Additional Dependencies
- **Parametric distribution fitting** ⏳
- **Statistical hypothesis testing** ⏳
- **Distribution-based scoring** ⏳

---

## Phase 17: CBM Gear Box Analysis ⏳ NEW PHASE
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

## Phase 18: Modal OSP (Optimal Sensor Placement) ⏳ NEW PHASE
*Target: 4-5 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/example_ModalOSP.m`
- **Python Output**: `examples/notebooks/advanced/modal_osp.ipynb` ⏳

### Description
Combines modal analysis with optimal sensor placement algorithms for structural monitoring.

### Additional Dependencies
- **Optimal sensor placement algorithms** ⏳
- **Modal assurance criteria** ⏳
- **Sensor network optimization** ⏳

---

## Phase 19: Fast Metric Kernel Density ⏳ NEW PHASE
*Target: 2-3 weeks*

### Target Example
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/exampleFastMetricKernelDensity.m`
- **Python Output**: `examples/notebooks/advanced/fast_metric_kernel_density.ipynb` ⏳

### Description
High-performance kernel density estimation with custom distance metrics.

### Additional Dependencies
- **Fast kernel density algorithms** ⏳ (may be completed in Phase 7)
- **Custom distance metrics** ⏳
- **Bandwidth selection methods** ⏳

---

## Phase 20: Dataset Utilities ⏳ NEW PHASE
*Target: 1-2 weeks*

### Target Examples
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/cbmDataSet.m`
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/threeStoryDataSet.m`
- **Python Output**: `examples/notebooks/utilities/dataset_management.ipynb` ⏳

### Description
Dataset loading, preprocessing, and management utilities for common SHM datasets.

### Additional Dependencies
- **Enhanced data loading utilities** ⏳
- **Data preprocessing pipelines** ⏳
- **Dataset validation tools** ⏳

---

## Phase 21: Hardware Integration ⏳ UPDATED PHASE
*Target: 4-5 weeks*

### Target Examples
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/example_NI_multiplex.m`
- **MATLAB Source**: `../shmtool-matlab/SHMTools/Examples/ExampleUsageScripts/example_DAQ_ARModel_Mahalanobis.m`
- **Python Output**: `examples/notebooks/hardware/ni_daq_integration.ipynb` ⏳
- **Python Output**: `examples/notebooks/hardware/daq_real_time_monitoring.ipynb` ⏳

### Description
Real-time data acquisition and monitoring with National Instruments hardware.

### Additional Dependencies
- **NI-DAQmx Python integration** ⏳
- **Real-time signal processing** ⏳
- **Hardware multiplexing** ⏳

---

## Phase 22: mFUSE Examples Validation ⏳ NEW PHASE
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