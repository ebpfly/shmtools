# SHMTools Example Notebooks

This directory contains Jupyter notebooks demonstrating SHMTools functionality, designed to replicate and enhance the original MATLAB "Example Usage Scripts" with publication-quality formatting.

## Structure

### `/basic/`
Core examples demonstrating individual function usage:
- `spectral_analysis_example.ipynb` - PSD, STFT, spectrogram examples
- `filtering_example.ipynb` - Digital filtering demonstrations  
- `statistical_features_example.ipynb` - Damage indicators and statistics
- `ar_modeling_example.ipynb` - Time series feature extraction
- `outlier_detection_example.ipynb` - Mahalanobis and PCA methods

### `/advanced/`
Complex analysis workflows combining multiple functions:
- `complete_shm_analysis.ipynb` - End-to-end damage detection workflow
- `modal_analysis_workflow.ipynb` - Modal parameter identification
- `active_sensing_analysis.ipynb` - Guided wave damage localization
- `condition_monitoring.ipynb` - Rotating machinery analysis

### `/tutorials/`
Step-by-step learning materials:
- `getting_started.ipynb` - Introduction to SHMTools Python
- `matlab_to_python_migration.ipynb` - Converting existing MATLAB workflows
- `web_interface_tutorial.ipynb` - Using the Bokeh workflow builder

## Publication Features

All notebooks are designed with publication-quality output:

- **Professional formatting** with proper headings and structure
- **High-quality plots** with publication-ready styling
- **Comprehensive documentation** with theory and implementation details
- **Reproducible results** with fixed random seeds and example data
- **Export capabilities** for PDF, HTML, and presentation formats

## Usage

### Running Notebooks
```bash
# Launch Jupyter Lab
jupyter lab examples/notebooks/

# Or use Jupyter Notebook
jupyter notebook examples/notebooks/
```

### Publishing Examples
```bash
# Convert to HTML (with plots)
jupyter nbconvert --to html examples/notebooks/basic/spectral_analysis_example.ipynb

# Convert to PDF (requires LaTeX)
jupyter nbconvert --to pdf examples/notebooks/basic/spectral_analysis_example.ipynb

# Generate presentation slides
jupyter nbconvert --to slides examples/notebooks/tutorials/getting_started.ipynb
```

## Notebook Standards

To maintain consistency and publication quality:

1. **Structure**: Follow the MATLAB code cell format with clear sections
2. **Imports**: Group all imports at the top
3. **Data**: Use example datasets included in `examples/data/`
4. **Plots**: Include both interactive (Bokeh) and static (matplotlib) versions
5. **Documentation**: Explain theory, parameters, and results
6. **Reproducibility**: Set random seeds and include version information

## Example Template

```python
# %% [markdown]
# # Example Title
# 
# Brief description of what this example demonstrates.
# 
# ## Theory
# Mathematical background and explanation.

# %% [python]
# Import required packages
import numpy as np
import matplotlib.pyplot as plt
import shmtools

# Set parameters for reproducibility
np.random.seed(42)

# %% [markdown]
# ## Data Generation
# Description of the test data.

# %% [python]
# Generate or load example data
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(fs)

# %% [markdown]
# ## Analysis
# Description of the analysis steps.

# %% [python]
# Perform analysis
f, psd = shmtools.psd_welch(signal, fs=fs)

# Create publication-quality plot
plt.figure(figsize=(10, 6))
plt.semilogy(f, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V²/Hz)')
plt.title('Power Spectral Density Example')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Results
# Discussion of results and conclusions.
```

This approach ensures that all examples can be easily converted to professional documentation while maintaining the interactive and educational benefits of Jupyter notebooks.