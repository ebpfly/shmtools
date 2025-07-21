# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHMTools Python is a comprehensive structural health monitoring toolkit converted from the original MATLAB SHMTools library. It provides 165+ signal processing and ML functions with a modern Bokeh web interface, replacing the original Java mFUSE GUI.

## Development Commands

### Virtual Environment Setup (REQUIRED)
**IMPORTANT**: Always use the virtual environment to avoid package conflicts:

```bash
# Activate virtual environment (REQUIRED for all Python work)
source venv/bin/activate

# If venv doesn't exist, create it first:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Installation and Setup
```bash
# Activate virtual environment first
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode  
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Testing
```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests
pytest

# Run tests with coverage
pytest --cov=shmtools --cov=bokeh_shmtools

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m "not hardware"      # Skip hardware tests
pytest -m integration         # Run only integration tests
```

### Code Quality
```bash
# Activate virtual environment first
source venv/bin/activate

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

### Jupyter Notebook Operations
```bash
# Activate virtual environment first
source venv/bin/activate

# Convert notebook to HTML with execution
jupyter nbconvert --to html --execute examples/notebooks/basic/notebook_name.ipynb --output-dir=examples/html/ --ExecutePreprocessor.timeout=300

# Start Jupyter notebook server
jupyter notebook

# Start JupyterLab
jupyter lab
```

## Architecture

### Core Library Structure (`shmtools/`)

**Two-tier architecture**: Core functions + web interface
- **`core/`** - Signal processing fundamentals (spectral, filtering, statistics)
- **`features/`** - Feature extraction (time series modeling, AR models) 
- **`classification/`** - ML and outlier detection (Mahalanobis, PCA)
- **`modal/`** - Modal analysis and structural dynamics
- **`active_sensing/`** - Guided wave analysis
- **`hardware/`** - Data acquisition interfaces
- **`plotting/`** - Bokeh-specific visualization utilities
- **`utils/`** - Data I/O and general utilities

### Web Interface Structure (`bokeh_shmtools/`)

**Panel-based Bokeh application** mimicking original mFUSE workflow:
- **`app.py`** - Main Bokeh server entry point with 4-panel layout
- **`panels/`** - UI components:
  - `function_library.py` - Function browser and selection
  - `workflow_builder.py` - Drag-and-drop workflow creation
  - `parameter_controls.py` - Dynamic parameter input forms
  - `results_viewer.py` - Visualization and output display
- **`workflows/`** - Workflow execution engine
- **`sessions/`** - Session file management (.ses compatibility)

### Function Naming Convention
- All functions maintain original `_shm` suffix for MATLAB compatibility
- Core functions exported without suffix in main `__init__.py` for convenience
- Example: `psd_welch_shm()` → available as `shmtools.psd_welch()`

## Development Approach

### Example-Driven Development
The project follows **example-driven development** where completion of each phase is marked by converting a MATLAB example to a working Jupyter notebook:

1. **Phase 1**: Basic spectral analysis (`examplePSD.m` → `spectral_analysis_example.ipynb`)
2. **Phase 2**: Statistical features (`exampleStatisticalMoments.m`)
3. **Phase 3**: Digital filtering (`exampleFiltering.m`)
4. **Phase 4**: Mahalanobis outlier detection (`exampleMahalanobis.m`)
5. **Phase 5**: PCA outlier detection (`examplePCA.m`)
6. **Phase 6**: Complete 3-story structure analysis (`example_3StoryStructure.m`)

### Key Dependencies
- **Core**: numpy, scipy, scikit-learn, matplotlib, pandas
- **Web Interface**: bokeh >=2.4.0
- **Optional Hardware**: nidaqmx, pyserial
- **Advanced Features**: pywavelets, numba, joblib

### Testing Strategy
- **Test markers**: `slow`, `integration`, `hardware`
- **Coverage**: Targets both `shmtools/` and `bokeh_shmtools/` modules
- **Hardware tests**: Require actual DAQ hardware, marked separately

### Migration Compatibility
- Function signatures match original MATLAB interfaces
- Session files (.ses) from mFUSE can be imported
- Data formats support MATLAB .mat file loading
- Parameter naming preserves MATLAB conventions