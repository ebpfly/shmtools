# SHMTools Python

A comprehensive Python-based structural health monitoring toolkit, converted from the original MATLAB SHMTools library developed by Los Alamos National Laboratory.

## Overview

SHMTools Python provides a modern, JupyterLab-based platform for structural health monitoring analysis, featuring:

- **108+ signal processing and ML functions** converted from MATLAB with full numerical parity
- **jFUSE JupyterLab extension** - an intelligent function selector replacing the original Java mFUSE GUI
- **Interactive Jupyter notebooks** with guided workflows and educational examples
- **Real-time data visualization** with rich plotting capabilities
- **Cloud deployment** - AWS JupyterHub instances with complete SHMTools environment
- **Example-driven development** - comprehensive notebooks covering all SHM applications

## Quick Start

### One-Command Installation

```bash
# Clone and install with automatic jFUSE extension build
git clone https://github.com/ebpfly/shmtools.git
cd shm
source venv/bin/activate  # CRITICAL: Always activate first
pip install -e .

# Install Jupyter kernel
python -m ipykernel install --user --name=shm-venv --display-name="SHM Python (venv)"

# Launch JupyterLab
jupyter lab
```

**That's it!** The `pip install -e .` automatically:
- ✅ Installs all Python dependencies
- ✅ Builds the TypeScript components (`npm install`, `npm run build:lib`)
- ✅ Compiles the JupyterLab extension (`npm run build:labextension:dev`)
- ✅ Integrates with JupyterLab (`jupyter lab build`)

Open your browser to `http://localhost:8888` and look for the 🔍 **SHM Functions** panel in the left sidebar!

### Cloud Deployment (Recommended)

Deploy a complete SHMTools environment on AWS in ~5-10 minutes:

```bash
cd jupyterhub/
./setup_jupyterhub_aws.sh
# Access at http://<PUBLIC_IP> with HTTPS enabled
```

### Basic Usage

#### Using jFUSE Extension in JupyterLab
1. **One-time setup**: `pip install -e .` (automatically builds everything)
2. **Launch**: `jupyter lab`
3. **Create notebook**: Use "SHM Python (venv)" kernel
4. **Find functions**: Click 🔍 **SHM Functions** in left sidebar
5. **Insert code**: Search and click any function to add pre-configured code

#### Direct Python API

```python
import numpy as np
import shmtools

# Generate example signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(fs)

# Compute power spectral density
f, psd = shmtools.psd_welch_shm(signal, fs)

# Apply bandpass filter
filtered = shmtools.bandpass_shm(signal, 40, 60, fs)

# Extract AR model features
ar_coeffs = shmtools.ar_model_shm(signal, 10)

# Outlier detection using Mahalanobis distance
scores = shmtools.mahalanobis_squared_shm(ar_coeffs.reshape(1, -1))
```

#### Example Workflows
Explore complete analysis workflows in `examples/notebooks/`:
- **Basic SHM**: Signal processing fundamentals
- **Modal Analysis**: Structural dynamics and mode shapes
- **Active Sensing**: Guided wave damage detection
- **Condition Monitoring**: Health state classification
- **Outlier Detection**: Anomaly detection techniques

## Architecture

### Core Library (`shmtools/`)

- **`core/`**: Basic signal processing (spectral analysis, filtering, statistics)
- **`features/`**: Feature extraction (time series modeling, modal analysis)
- **`classification/`**: Machine learning and outlier detection
- **`modal/`**: Modal analysis and structural dynamics
- **`active_sensing/`**: Guided wave analysis and ultrasonic testing
- **`hardware/`**: Data acquisition interfaces (NI-DAQ, serial communication)
- **`plotting/`**: Matplotlib and Bokeh visualization utilities
- **`utils/`**: General utilities and data management

### jFUSE JupyterLab Extension (`shm_function_selector/`)

- **TypeScript Frontend**: Interactive function selector with smart search and parameter forms
- **Python Backend**: Server extension providing function metadata and code generation
- **Integration**: Seamless insertion of SHMTools functions into Jupyter notebooks
- **Educational**: Context-aware help and example code snippets

### Educational Content (`examples/`)

- **`notebooks/`**: Comprehensive tutorials by category (basic, intermediate, advanced)
- **`data/`**: Real-world datasets (.mat files) for hands-on learning
- **Published HTML**: Static exports for offline viewing and teaching

### Cloud Infrastructure (`jupyterhub/`)

- **AWS Deployment**: Automated EC2 setup with JupyterHub, HTTPS, and complete SHMTools environment
- **User Management**: The Littlest JupyterHub (TLJH) with multi-user support
- **Remote Updates**: Continuous deployment from GitHub repository

## Development Status

### ✅ Core Library (Completed)
- ✅ 108+ functions with full MATLAB parity
- ✅ Spectral analysis (PSD, coherence, transfer functions)
- ✅ Filtering and preprocessing (bandpass, highpass, lowpass)
- ✅ Statistical analysis (moments, distributions, correlation)
- ✅ Time series modeling (AR, ARMA, state-space)
- ✅ Outlier detection algorithms (Mahalanobis, novelty detection)
- ✅ Modal analysis (frequency domain decomposition, stochastic subspace)
- ✅ Active sensing (guided waves, damage detection)

### ✅ jFUSE JupyterLab Extension (Production Ready)
- ✅ TypeScript frontend with intelligent search
- ✅ Python server extension with function metadata
- ✅ Seamless code insertion and parameter forms
- ✅ Context-sensitive help and documentation
- ✅ Educational examples and workflows

### ✅ Educational Content (Comprehensive)
- ✅ 20+ example notebooks covering all SHM applications
- ✅ Real-world datasets with complete analysis workflows
- ✅ HTML exports for offline access and teaching
- ✅ Progressive learning path (basic → intermediate → advanced)

### ✅ Cloud Deployment (Production)
- ✅ Automated AWS EC2 deployment with JupyterHub
- ✅ HTTPS with Let's Encrypt certificates
- ✅ Multi-user environment with complete SHMTools stack
- ✅ Remote update capabilities for continuous deployment

### 🔄 Current Development
- 🔄 Hardware integration testing (NI-DAQ, serial devices)
- 🔄 Additional example datasets and use cases

## Contributing

We welcome contributions! Please see our [development guide](docs/development.md) for details on:

- Setting up the development environment
- Code style and testing guidelines
- Adding new functions and algorithms
- Contributing to the web interface

### Development Setup

```bash
# Activate virtual environment (CRITICAL)
cd /path/to/shm && source venv/bin/activate

# Install in development mode (includes extension auto-build)
pip install -e .[dev]

# Run tests
pytest                                 # All tests
pytest -m "not hardware"              # Skip hardware tests

# Code quality
black shmtools/ && flake8 shmtools/   # Format & lint

# After making extension changes, rebuild quickly:
shmtools-build-extension              # Build only the extension
# OR
./restart_jupyterlab.sh               # Full rebuild + restart
```

### Manual Extension Commands

If you need more control over the extension build process:

```bash
# Build extension components only
shmtools-build-extension

# Full extension install/reinstall
install-jfuse

# Remove extension
uninstall-jfuse

# Traditional manual build (if needed)
cd shm_function_selector/
npm install && npm run build:lib && npm run build:labextension:dev
cd .. && jupyter lab build
```

## AWS Cloud Deployment

### Quick Deploy
Create a complete SHMTools cloud environment in ~5-10 minutes:

```bash
cd jupyterhub/
./setup_jupyterhub_aws.sh
# Creates Ubuntu 22.04 EC2 with JupyterHub, Node.js 20.x, Python 3.10/3.12
# Installs 108+ SHM functions, jFUSE extension, example datasets
# Enables HTTPS with Let's Encrypt certificate
# Access at http://<PUBLIC_IP>
```

### Configuration Options
Edit `jupyterhub/setup_jupyterhub_aws.sh` for custom settings:

```bash
AWS_REGION="us-east-2"
INSTANCE_TYPE="t3.medium"  # ~$30/month
GITHUB_OWNER="your-username"
GITHUB_REPO="shmtools"
```

### Update Deployed Instance
Keep cloud instances synchronized with latest code:

```bash
# From local machine (recommended)
cd jupyterhub/
./remote_update.sh 3.130.148.209        # Update specific IP
./remote_update.sh                       # Auto-detect running instance

# Or directly on server
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<IP>
cd /srv/classrepo
./jupyterhub/update_deployment.sh       # Full update workflow
```

### User Management
```bash
# Add users to JupyterHub
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<IP>
sudo tljh-config set users.allowed username1 username2
sudo tljh-config reload
```

### Cost Management
```bash
# Stop instance when not in use
aws ec2 stop-instances --instance-ids <INSTANCE_ID>

# Terminate when done
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

## Migration from MATLAB

SHMTools Python provides seamless transition from MATLAB workflows:

### 🔄 Function Compatibility
- **Naming**: All functions use `_shm` suffix (e.g., `psd_welch_shm`, `ar_model_shm`)
- **Signatures**: Parameter interfaces match MATLAB exactly
- **Numerics**: Results validated to machine precision against MATLAB
- **Documentation**: Complete docstrings with GUI metadata for jFUSE integration

### 📁 Data Compatibility
- **MATLAB .mat files**: Direct loading with `scipy.io.loadmat`
- **Real datasets**: 161MB of example data covering all SHM applications
- **Synthetic validation**: Extensive testing with known-answer problems

### 🎓 Learning Resources
- **Side-by-side comparisons**: MATLAB and Python implementations
- **Conversion examples**: Complete workflows showing before/after
- **Best practices**: Modern Python patterns while preserving MATLAB logic

Explore `examples/notebooks/` for hands-on migration examples.

## Documentation & Resources

- **CLAUDE.md**: Essential development instructions and deployment guides
- **Example Notebooks**: 20+ comprehensive tutorials in `examples/notebooks/`
- **Published HTML**: Offline-viewable exports in `published_notebooks/`
- **Function Reference**: Built-in documentation accessible through jFUSE extension
- **AWS Deployment**: Complete cloud setup guides in `jupyterhub/README.md`

## License

This project is licensed under the BSD 3-Clause License, consistent with the original MATLAB SHMTools library.

## Acknowledgments

This work builds upon the original SHMTools library developed by Los Alamos National Laboratory. We gratefully acknowledge the contributions of the original authors and the structural health monitoring research community.

## Key Features

### 🔬 Complete MATLAB Parity
- All 108+ functions maintain exact numerical compatibility with MATLAB originals
- Comprehensive test suite validates results against reference implementations
- Supports existing .mat datasets and analysis workflows

### 🎯 Intelligent Function Discovery
- jFUSE extension provides smart search across all SHM functions
- Context-aware parameter forms with validation and defaults
- Instant code insertion with proper imports and documentation

### 📚 Educational Excellence
- Progressive learning path from basic signal processing to advanced SHM
- Real-world datasets with complete analysis workflows
- Interactive notebooks combining theory, code, and visualization

### ☁️ Cloud-Ready Deployment
- One-command AWS deployment with complete SHMTools environment
- HTTPS-enabled JupyterHub with multi-user support
- Automated updates and maintenance scripts

## Citation

If you use SHMTools Python in your research, please cite:

```bibtex
@software{shmtools_python,
  title={SHMTools Python: A JupyterLab-Based Structural Health Monitoring Toolkit},
  author={SHMTools Development Team},
  year={2024},
  url={https://github.com/ebpfly/shmtools}
}
```