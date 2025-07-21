# SHMTools Python

A comprehensive Python-based structural health monitoring toolkit, converted from the original MATLAB SHMTools library developed by Los Alamos National Laboratory.

## Overview

SHMTools Python provides a modern, web-based platform for structural health monitoring analysis, featuring:

- **165+ signal processing and ML functions** converted from MATLAB
- **Modern Bokeh web interface** replacing the original Java mFUSE GUI
- **Interactive workflow builder** for creating analysis sequences
- **Real-time data visualization** with zoom, pan, and selection
- **Cross-platform deployment** - runs in any web browser

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shmtools-python.git
cd shmtools-python

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Launch the Web Interface

```bash
# Start the Bokeh server
bokeh serve bokeh_shmtools/app.py --show

# Or use the command line entry point
shmtools-gui serve
```

Open your browser to `http://localhost:5006` to access the workflow builder.

### Basic Usage

```python
import numpy as np
import shmtools

# Generate example signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(fs)

# Compute power spectral density
f, psd = shmtools.psd_welch(signal, fs=fs)

# Apply bandpass filter
filtered = shmtools.bandpass_filter(signal, 40, 60, fs)

# Extract AR model features
ar_coeffs = shmtools.ar_model(signal, order=10)

# Outlier detection
scores = shmtools.mahalanobis_distance(ar_coeffs.reshape(1, -1))
```

## Architecture

### Core Library (`shmtools/`)

- **`core/`**: Basic signal processing (spectral analysis, filtering, statistics)
- **`features/`**: Feature extraction (time series modeling, modal analysis)
- **`classification/`**: Machine learning and outlier detection
- **`modal/`**: Modal analysis and structural dynamics
- **`active_sensing/`**: Guided wave analysis
- **`hardware/`**: Data acquisition interfaces
- **`plotting/`**: Bokeh-specific visualization utilities
- **`utils/`**: General utilities and data management

### Web Interface (`bokeh_shmtools/`)

- **`app.py`**: Main Bokeh server application
- **`panels/`**: UI panel components (function library, workflow builder, etc.)
- **`workflows/`**: Workflow execution engine
- **`sessions/`**: Session file management (.ses compatibility)

## Development Status

### Phase 1: Core Library (In Progress)
- ✅ Project structure and packaging
- ✅ Core signal processing stubs
- 🔄 Spectral analysis functions
- 🔄 Filtering and preprocessing
- 🔄 Statistical analysis
- ⏳ Time series modeling
- ⏳ Outlier detection algorithms

### Phase 2: Web Interface (In Progress)  
- ✅ Bokeh application structure
- ✅ Function library panel
- ✅ Workflow builder panel
- ✅ Parameter controls panel
- ✅ Results viewer panel
- ⏳ Workflow execution engine
- ⏳ Session file support

### Phase 3: Advanced Features (Planned)
- ⏳ Modal analysis functions
- ⏳ Active sensing algorithms
- ⏳ Hardware integration
- ⏳ Advanced visualization
- ⏳ Real-time data streaming

## Contributing

We welcome contributions! Please see our [development guide](docs/development.md) for details on:

- Setting up the development environment
- Code style and testing guidelines
- Adding new functions and algorithms
- Contributing to the web interface

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black shmtools/ bokeh_shmtools/
flake8 shmtools/ bokeh_shmtools/
```

## Migration from MATLAB

This project maintains compatibility with existing MATLAB workflows:

- **Function names** preserve the original `_shm` suffix pattern
- **Parameter interfaces** match MATLAB function signatures
- **Session files** (.ses) from mFUSE can be imported
- **Data formats** support MATLAB .mat file loading

See the [migration guide](docs/matlab-migration.md) for detailed conversion information.

## Documentation

- [API Documentation](docs/api/)
- [User Guide](docs/user-guide.md) 
- [Conversion Plan](docs/conversion-plan.md)
- [MATLAB Function Mapping](docs/matlab-mapping.md)

## License

This project is licensed under the BSD 3-Clause License, consistent with the original MATLAB SHMTools library.

## Acknowledgments

This work builds upon the original SHMTools library developed by Los Alamos National Laboratory. We gratefully acknowledge the contributions of the original authors and the structural health monitoring research community.

## Citation

If you use SHMTools Python in your research, please cite:

```bibtex
@software{shmtools_python,
  title={SHMTools Python: A Web-Based Structural Health Monitoring Toolkit},
  author={SHMTools Development Team},
  year={2024},
  url={https://github.com/yourusername/shmtools-python}
}
```