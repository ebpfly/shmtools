# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains **two parallel structural health monitoring (SHM) toolkits**:

1. **`shmtools-python/`** - Modern Python conversion with Bokeh web interface (in development)
2. **`shmtools-matlab/`** - Original MATLAB SHMTools library with Java mFUSE GUI

The Python version is being developed using **example-driven development**, converting MATLAB functionality phase by phase while maintaining API compatibility.

## Working Directory Structure

```
/Users/eric/repo/shm/
├── shmtools-python/     # Main Python development (use this for most work)
├── shmtools-matlab/     # Original MATLAB reference (read-only reference)
└── CLAUDE.md           # This file
```

**IMPORTANT**: Always work in `shmtools-python/` unless specifically directed to examine MATLAB reference code.

## Python Development Commands

### Setup and Installation
```bash
cd shmtools-python/

# Install dependencies
pip install -r requirements.txt

# Install in development mode  
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=shmtools --cov=bokeh_shmtools

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m "not hardware"      # Skip hardware tests  
pytest -m integration         # Run only integration tests
pytest -m "requires_data"     # Run tests requiring example datasets
```

### Code Quality
```bash
# Format code
black shmtools/ bokeh_shmtools/

# Lint code
flake8 shmtools/ bokeh_shmtools/

# Type checking
mypy shmtools/ bokeh_shmtools/
```

### Running the Web Interface
```bash
# Activate virtual environment first
source venv/bin/activate

# Start Bokeh server (preferred method)
bokeh serve bokeh_shmtools/app.py --show

# Alternative entry point
shmtools-gui serve

# Access at http://localhost:5006
```

### JupyterLab Extension Development
**CRITICAL**: The JupyterLab extension requires a specific 3-step build process that must be followed exactly:

```bash
# ALWAYS work from the extension directory
cd shm_function_selector/

# Step 1: Compile TypeScript to JavaScript (REQUIRED FIRST)
npm run build:lib

# Step 2: Build the JupyterLab extension (uses compiled JS)
npm run build:labextension:dev

# Step 3: Integrate extension into JupyterLab (from parent directory)
cd ..
source venv/bin/activate
jupyter lab build
```

**IMPORTANT BUILD NOTES**:
- **NEVER** skip Step 1 - TypeScript changes won't appear without it
- **NEVER** run `jupyter lab build` alone - it won't pick up extension changes
- If changes don't appear, clear cache: `rm -rf shm_function_selector/shm_function_selector/labextension/static/*.js`
- Always check the generated file hash changes to confirm new build
- Browser refresh may be required after `jupyter lab build`

**Debugging Extension Issues**:
```bash
# Check if TypeScript compiled correctly
cd shm_function_selector/
npm run build:lib
grep "your_debug_text" lib/index.js

# Check if extension built correctly  
npm run build:labextension:dev
grep "your_debug_text" shm_function_selector/labextension/static/lib_index_js.*.js

# Force complete rebuild if stuck
rm -rf shm_function_selector/labextension/static/*.js
npm run build:lib
npm run build:labextension:dev
cd .. && source venv/bin/activate && jupyter lab build
```

## Development Architecture

### Core Python Library (`shmtools/`)

**MATLAB-compatible function architecture**: All functions use the `_shm` suffix for MATLAB compatibility:

- **`core/`** - Signal processing fundamentals (spectral analysis, filtering, statistics)
- **`features/`** - Feature extraction (time series modeling, AR models) 
- **`classification/`** - ML and outlier detection (Mahalanobis, PCA, SVD)
- **`modal/`** - Modal analysis and structural dynamics
- **`active_sensing/`** - Guided wave analysis and propagation
- **`hardware/`** - Data acquisition interfaces (NI-DAQmx, serial)
- **`plotting/`** - Bokeh-specific visualization utilities  
- **`utils/`** - Data I/O, MATLAB compatibility, general utilities

### Web Interface (`bokeh_shmtools/`)

**4-panel Bokeh application** replicating original mFUSE workflow:

- **`app.py`** - Main Bokeh server with panel layout management
- **`panels/`** - UI components:
  - `function_library.py` - Function browser with category tree
  - `workflow_builder.py` - Drag-and-drop workflow creation
  - `parameter_controls.py` - Dynamic parameter forms with validation  
  - `results_viewer.py` - Interactive plotting and data visualization
- **`workflows/`** - Workflow execution engine and step management
- **`sessions/`** - Session file management (.ses format compatibility)
- **`utils/docstring_parser.py`** - Extracts GUI metadata from function docstrings

## Critical Development Principles

### Example-Driven Development Strategy

**Completion marked by working Jupyter notebooks**, not just function stubs. Each phase converts **one complete example** with all its dependencies, validates the output matches MATLAB, and publishes a working notebook.

### Quality Gates for Each Example
1. **MATLAB Analysis**: Read and understand the original `.m` file completely
2. **Dependency Mapping**: Identify all required SHMTools functions
3. **Function Conversion**: Convert each dependency with defensive programming and numerical stability
4. **Algorithm Validation**: Test with synthetic data where ground truth is known
5. **Integration Testing**: Full workflow validation with real data
6. **Notebook Creation**: Create educational Jupyter notebook with robust execution context handling
7. **Execution Testing**: Ensure notebook runs end-to-end from multiple working directories
8. **Publication Validation**: HTML export with all outputs and accessible visualizations

### Current Phase Priorities

**Development Order: Simple to Complex**
Start with fundamental outlier detection methods that share common dependencies, then progress to specialized algorithms.

1. **Phase 1: PCA Outlier Detection** *(2-3 weeks)*
   - **Target**: `examplePCA.m` → `examples/notebooks/basic/pca_outlier_detection.ipynb`
   - **Core Functions**: `arModel_shm`, `learnPCA_shm`, `scorePCA_shm`
   - **Dataset**: `data3SS.mat` (3-story structure, 4 channels, 17 damage states)
   - **Foundation**: Establishes AR modeling and PCA framework for later examples

2. **Phase 2: Mahalanobis Distance Outlier Detection** *(2-3 weeks)*
   - **Target**: `exampleMahalanobis.m` → `examples/notebooks/basic/mahalanobis_outlier_detection.ipynb`
   - **Core Functions**: `learnMahalanobis_shm`, `scoreMahalanobis_shm`
   - **Reuses**: `arModel_shm` from Phase 1, same data patterns

3. **Phase 3: SVD Outlier Detection** *(1-2 weeks)*
   - **Target**: `exampleSVD.m` → `examples/notebooks/basic/svd_outlier_detection.ipynb`
   - **Core Functions**: `learnSVD_shm`, `scoreSVD_shm`
   - **Reuses**: `arModel_shm`, data patterns from Phases 1-2

4. **Phase 4: Factor Analysis Outlier Detection** *(2-3 weeks)*
   - **Target**: `exampleFactorAnalysis.m` → `examples/notebooks/intermediate/factor_analysis_outlier_detection.ipynb`
   - **Core Functions**: `learnFactorAnalysis_shm`, `scoreFactorAnalysis_shm`

5. **Phase 5: Nonlinear PCA (NLPCA) Outlier Detection** *(3-4 weeks)*
   - **Target**: `exampleNLPCA.m` → `examples/notebooks/advanced/nlpca_outlier_detection.ipynb`
   - **Core Functions**: `learnNLPCA_shm`, `scoreNLPCA_shm`
   - **Dependencies**: Neural network components for nonlinear PCA

6. **Phase 6: AR Model Order Selection** *(1-2 weeks)*
   - **Target**: `exampleARModelOrder.m` → `examples/notebooks/basic/ar_model_order_selection.ipynb`
   - **Core Functions**: `arModelOrder_shm`
   - **Dependencies**: Information criteria (AIC, BIC) for model selection

7. **Phase 7: Nonparametric Outlier Detection** *(3-4 weeks)*
   - **Target**: `exampleDirectUseOfNonParametric.m` → `examples/notebooks/advanced/nonparametric_outlier_detection.ipynb`
   - **Dependencies**: Kernel density estimation, fast metric kernel density

8. **Phase 8: Semi-Parametric Outlier Detection** *(3-4 weeks)*
   - **Target**: `exampleDirectUseOfSemiParametric.m` → `examples/notebooks/advanced/semiparametric_outlier_detection.ipynb`
   - **Dependencies**: Gaussian Mixture Model functions, semi-parametric density estimation

9. **Phase 9: Active Sensing Feature Extraction** *(4-5 weeks)*
   - **Target**: `exampleActiveSensingFeature.m` → `examples/notebooks/advanced/active_sensing_feature_extraction.ipynb`
   - **Dependencies**: Guided wave analysis, matched filtering algorithms, geometry and propagation utilities

10. **Phase 10: Condition-Based Monitoring** *(3-4 weeks)*
    - **Target**: `example_CBM_Bearing_Analysis.m` → `examples/notebooks/specialized/cbm_bearing_analysis.ipynb`
    - **Dependencies**: Time-synchronous averaging, order tracking algorithms, kurtogram analysis

### Later Phases: Specialized Examples

- **Phase 11**: Sensor Diagnostics (`exampleSensorDiagnostics.m`)
- **Phase 12**: Modal Analysis (`exampleModalFeatures.m`)
- **Phase 13**: Hardware Integration (`example_NI_multiplex.m`, `example_DAQ_ARModel_Mahalanobis.m`)

## Data Management Strategy

### Manual Data Setup (Simple Approach)

Users manually download all example datasets once and place them in the repository.

#### Required Datasets
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
Example data loading functions in `shmtools/utils/data_loading.py`:

```python
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
    """
```

## Key Lessons from Phase 1

**Critical Issues Identified and Resolved:**

1. **MATLAB Indexing Conversion**: Most significant bug was incorrect 1-based to 0-based indexing conversion in AR model. Always use explicit `matlab_k = k + 1` conversion approach.

2. **Numerical Stability**: Add defensive programming for zero standard deviations (`data_std = np.where(data_std == 0, 1.0, data_std)`) and singular matrices (small regularization term).

3. **Execution Context**: Notebooks must handle multiple working directories and execution contexts (Jupyter, nbconvert, different CWDs). Use robust path resolution with fallback options.

4. **Visualization**: Prefer `plt.legend()` over hardcoded `plt.text()` positioning for better portability across contexts.

5. **Testing Strategy**: Always validate algorithms with synthetic test cases where ground truth is known before testing with real data.

**See `docs/conversion-plan.md` for complete lessons learned documentation.**

## MATLAB Function Conversion Rules

**CRITICAL**: Before implementing ANY function:

1. **Read Original MATLAB File**: Must examine the complete `.m` file in `shmtools-matlab/SHMFunctions/`
2. **Extract Exact Algorithm**: Document mathematical steps precisely  
3. **Preserve All Information**: Only convert what exists in original MATLAB
4. **Verify Function Signature**: Match input/output parameters exactly
5. **Check Dependencies**: Identify all MATLAB function dependencies

### Function Naming Convention

**CRITICAL**: All functions must use the `_shm` suffix for MATLAB compatibility.

- **All functions**: `psd_welch_shm()` (with `_shm` suffix)
- **Export pattern**: `shmtools.core.psd_welch_shm()` 
- **No modern Python versions**: Only `_shm` functions should exist

### Docstring Format

Uses **machine-readable docstrings** for automatic GUI generation:

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
        
    Parameters
    ----------
    data : array_like
        Input time series data.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    order : int
        AR model order.
        
        .. gui::
            :widget: number_input
            :min: 1
            :max: 50
            :default: 15
    """
```

## Key Dependencies and Technologies

- **Core**: numpy, scipy, scikit-learn, matplotlib, pandas
- **Web Interface**: bokeh >=2.4.0 (4-panel layout)
- **Optional Hardware**: nidaqmx, pyserial
- **Advanced Features**: pywavelets, numba, joblib
- **Development**: pytest, black, flake8, mypy, pre-commit

## Migration from MATLAB

### Compatibility Features
- Function signatures match MATLAB interfaces where possible
- Session files (.ses) from mFUSE can be imported to workflows
- Data formats support MATLAB .mat file loading via scipy.io
- Parameter naming preserves MATLAB conventions

### Reference Documentation
- **Conversion Plan**: `docs/conversion-plan.md` - detailed phase planning
- **Docstring Format**: `docs/docstring-format.md` - GUI integration specs
- **MATLAB Reference**: Use `shmtools-matlab/` for algorithm verification

## Common Development Workflows

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
# Run complete notebook end-to-end
pytest examples/test_notebooks/test_*.py

# Export to HTML
jupyter nbconvert --to html notebook.ipynb

# Validate HTML renders correctly
# Check all plots display properly
```

### Adding New Functions
1. Identify target MATLAB function in `shmtools-matlab/SHMFunctions/`
2. Read complete `.m` file to understand algorithm
3. Implement both `_shm` and modern versions in appropriate `shmtools/` module
4. Add machine-readable docstring with GUI specifications
5. Write tests in `tests/test_shmtools/`
6. Update function exports in module `__init__.py`

### Testing Hardware Functions
```bash
# Skip hardware tests during normal development
pytest -m "not hardware"

# Run hardware tests when DAQ equipment available
pytest -m hardware
```

### Working with Bokeh Interface
```bash
# Start development server with live reload
bokeh serve --dev bokeh_shmtools/app.py --show

# Test individual panels
python -m bokeh_shmtools.panels.function_library
```

## Success Criteria

### Quality Assurance Checklist

For each converted example:
- [ ] All MATLAB functions identified and mapped
- [ ] Python functions follow exact MATLAB algorithms  
- [ ] Docstrings include all required GUI metadata
- [ ] Outputs match MATLAB within numerical tolerance
- [ ] Jupyter notebook runs without errors
- [ ] HTML export renders cleanly
- [ ] All visualizations display correctly
- [ ] Educational content explains the methodology

### Success Metrics

- **Functional Parity**: Python results match MATLAB exactly
- **Reusability**: Functions work across multiple examples  
- **Documentation Quality**: Notebooks suitable for publication
- **GUI Integration**: Docstring metadata enables automatic web interface
- **Performance**: Conversion maintains or improves execution speed

This approach ensures each example provides immediate value while building a robust foundation for the complete SHMTools conversion.