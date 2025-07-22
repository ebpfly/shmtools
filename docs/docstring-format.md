# SHMTools Python Docstring Format

## Overview

SHMTools Python uses structured docstrings that are both PEP 257/287 compliant and machine-readable for the Bokeh workflow builder. This format allows automatic extraction of function metadata, parameter specifications, and usage examples.

## Docstring Structure

### Required Sections (PEP 287 compliant)

```python
def example_function(x, fs=1000.0, window="hann", order=None):
    """
    Brief one-line description of the function.
    
    Longer description providing more detail about the function's purpose,
    algorithm, and applications in structural health monitoring.
    
    Parameters
    ----------
    x : array_like
        Input signal array. If 2D, analysis is performed on each column.
    fs : float, optional
        Sampling frequency in Hz. Default is 1000.0.
    window : {"hann", "hamming", "blackman"}, optional
        Window function type. Default is "hann".
    order : int, optional
        Filter order. If None, automatically determined.
        
    Returns
    -------
    result : ndarray
        Processed signal or analysis result.
    metadata : dict
        Dictionary containing analysis metadata and parameters.
        
    Raises
    ------
    ValueError
        If input signal is empty or sampling frequency is non-positive.
    TypeError
        If window type is not supported.
        
    See Also
    --------
    related_function : Brief description of related function.
    shmtools.core.other_function : Cross-reference to related functionality.
    
    Notes
    -----
    Additional technical information, algorithm details, or implementation
    notes. Mathematical equations can be included using LaTeX notation.
    
    The algorithm implements the method described in [1]_.
    
    References
    ----------
    .. [1] Author, A. "Title of Paper." Journal Name, vol. X, pp. Y-Z, Year.
    
    Examples
    --------
    Basic usage:
    
    >>> import numpy as np
    >>> import shmtools
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(fs)
    >>> result, metadata = shmtools.example_function(signal, fs=fs)
    
    Advanced usage with custom parameters:
    
    >>> result = shmtools.example_function(signal, fs=fs, window="blackman", order=4)
    """
```

## Machine-Readable Metadata Extensions

### Function Classification Tags

Add structured metadata in the docstring for the workflow builder:

```python
def psd_welch(x, fs=1.0, window="hann", nperseg=None):
    """
    Estimate power spectral density using Welch's method.
    
    Python equivalent of MATLAB's psdWelch_shm function. Computes the power
    spectral density of input signals using Welch's overlapped segment averaging
    method for reduced noise and improved frequency resolution.
    
    .. meta::
        :category: Core - Spectral Analysis
        :matlab_equivalent: psdWelch_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Frequency Domain
        :interactive_plot: True
        :typical_usage: ["vibration_analysis", "frequency_content", "noise_analysis"]
        :display_name: Power Spectral Density (Welch Method)
        :verbose_call: Compute PSD(signal, sampling_rate, window_type, segment_length)
    
### Meta Field Specifications

The `.. meta::` section contains machine-readable metadata for the workflow builder:

- **`:category:`** - Functional classification (e.g., "Core - Spectral Analysis", "Classification - Parametric Detectors")
- **`:matlab_equivalent:`** - Original MATLAB function name for reference  
- **`:complexity:`** - Difficulty level: "Basic", "Intermediate", "Advanced"
- **`:data_type:`** - Expected input type: "Time Series", "Features", "Frequency Domain"
- **`:output_type:`** - Output data type: "Features", "Scores", "Model", "Frequency Domain"
- **`:interactive_plot:`** - Boolean, whether function generates interactive visualizations
- **`:typical_usage:`** - List of common use cases for search and categorization
- **`:display_name:`** - Short, human-readable function name for UI display (NEW)
- **`:verbose_call:`** - Human-readable function call template with parameter names (NEW)

The new `:display_name:` and `:verbose_call:` fields provide the UI with short, descriptive names that match the original MATLAB "VERBOSE FUNCTION CALL" metadata.

    Parameters
    ----------
    x : array_like, shape (n_samples,) or (n_samples, n_channels)
        Input signal array. If 2D, PSD is computed for each column.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: "Upload time series data"
            
    fs : float, optional, default=1.0
        Sampling frequency in Hz.
        
        .. gui::
            :widget: numeric_input
            :min: 0.1
            :max: 1000000.0
            :step: 0.1
            :units: "Hz"
            :description: "Sampling frequency"
            
    window : {"hann", "hamming", "blackman", "bartlett"}, optional, default="hann"
        Window function applied to each segment.
        
        .. gui::
            :widget: select
            :options: ["hann", "hamming", "blackman", "bartlett"]
            :description: "Window function type"
            
    nperseg : int, optional
        Length of each segment for Welch's method. If None, uses scipy default.
        
        .. gui::
            :widget: numeric_input
            :min: 8
            :max: 8192
            :step: 1
            :allow_none: True
            :description: "Segment length (None for auto)"
    
    Returns
    -------
    f : ndarray, shape (n_freqs,)
        Frequency array in Hz.
    psd : ndarray, shape (n_freqs,) or (n_freqs, n_channels)
        Power spectral density array. Units are signal²/Hz.
        
        .. gui::
            :plot_type: "line"
            :x_axis: "f"
            :y_axis: "psd"
            :log_scale: "y"
            :xlabel: "Frequency (Hz)"
            :ylabel: "PSD (Units²/Hz)"
    """
```

## Parsing Implementation

### Docstring Parser for Workflow Builder

```python
# bokeh_shmtools/utils/docstring_parser.py

import inspect
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ParameterSpec:
    """Parameter specification for GUI generation."""
    name: str
    type_hint: str
    description: str
    default: Any
    widget: str = "text_input"
    widget_params: Dict[str, Any] = None
    
@dataclass
class FunctionMetadata:
    """Complete function metadata for workflow builder."""
    name: str
    brief_description: str
    full_description: str
    category: str
    matlab_equivalent: Optional[str]
    complexity: str
    parameters: List[ParameterSpec]
    returns: Dict[str, Any]
    examples: List[str]
    interactive_plot: bool = False
    typical_usage: List[str] = None

def parse_shmtools_docstring(func) -> FunctionMetadata:
    """
    Parse SHMTools function docstring to extract metadata.
    
    Parameters
    ----------
    func : callable
        Function to parse.
        
    Returns
    -------
    metadata : FunctionMetadata
        Parsed function metadata.
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return None
    
    # Extract meta section
    meta_pattern = r'\.\. meta::\s*\n((?:\s{4}.*\n?)*)'
    meta_match = re.search(meta_pattern, docstring)
    meta_info = {}
    
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lstrip(':')
                meta_info[key] = eval(value.strip()) if value.strip().startswith('[') else value.strip()
    
    # Parse parameters with GUI specifications
    parameters = _parse_parameters_with_gui(docstring)
    
    # Extract other sections
    brief_desc = docstring.split('\n')[0]
    
    return FunctionMetadata(
        name=func.__name__,
        brief_description=brief_desc,
        full_description=_extract_description(docstring),
        category=meta_info.get('category', 'Uncategorized'),
        matlab_equivalent=meta_info.get('matlab_equivalent'),
        complexity=meta_info.get('complexity', 'Unknown'),
        parameters=parameters,
        returns=_parse_returns(docstring),
        examples=_extract_examples(docstring),
        interactive_plot=meta_info.get('interactive_plot', False),
        typical_usage=meta_info.get('typical_usage', [])
    )

def _parse_parameters_with_gui(docstring: str) -> List[ParameterSpec]:
    """Parse parameters section with GUI widget specifications."""
    # Implementation to parse Parameters section and extract GUI specs
    pass

# Additional parsing functions...
```

## Usage in Workflow Builder

```python
# bokeh_shmtools/panels/function_library.py

from bokeh_shmtools.utils.docstring_parser import parse_shmtools_docstring
import shmtools

def discover_shmtools_functions():
    """Discover and categorize all SHMTools functions."""
    functions = {}
    
    for module_name in ['core', 'features', 'classification']:
        module = getattr(shmtools, module_name)
        for func_name in dir(module):
            if not func_name.startswith('_'):
                func = getattr(module, func_name)
                if callable(func):
                    metadata = parse_shmtools_docstring(func)
                    if metadata:
                        category = metadata.category
                        if category not in functions:
                            functions[category] = []
                        functions[category].append(metadata)
    
    return functions
```

## Key Features

### 1. **PEP Compliance**
- Follows PEP 257 (Docstring Conventions)
- Uses PEP 287 (reStructuredText) formatting
- Compatible with Sphinx documentation generation

### 2. **Machine Readable**
- `.. meta::` sections for workflow builder metadata
- `.. gui::` sections for parameter widget specifications
- Structured parameter and return value descriptions

### 3. **Automatic GUI Generation**
- Widget type and parameters specified in docstring
- Validation rules (min/max, options) embedded
- Plot specifications for result visualization

### 4. **Documentation Integration**
- Examples section for Jupyter notebook generation
- Cross-references to related functions
- Mathematical notation support with LaTeX

### 5. **Migration Support**
- `matlab_equivalent` field for cross-referencing
- Complexity indicators for conversion planning
- Usage tags for categorization

This format provides the same structured information that mFUSE extracted from MATLAB headers while being fully PEP-compliant and enabling automatic GUI generation for the Bokeh interface.