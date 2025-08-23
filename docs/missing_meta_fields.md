# Functions Missing `.. meta:` Field in Docstrings

This document lists all functions ending with `_shm` that are missing the required `.. meta:` field in their docstrings.

**Generated on:** 2025-08-23

## Summary

- **Total functions ending with `_shm`:** 107
- **Functions missing `.. meta:` field:** 1

## Functions Missing Meta Field

1. **`plot_spectrogram_shm`**
   - **File:** `/Users/eric/repo/shm/shmtools/plotting/spectral_plots.py`
   - **Line:** 446
   - **Status:** Missing `.. meta:` field in docstring

## Notes

- The `.. meta:` field is required for automatic GUI generation and function categorization
- This field should include category, MATLAB equivalent, complexity, data type, output type, and display name
- For plotting functions like `plot_spectrogram_shm`, the meta field should specify category as "Plotting - Spectral Analysis"

## Resolution

The missing meta field should be added to the `plot_spectrogram_shm` function docstring following the project's docstring format specifications found in `docs/docstring-format.md`.

Example format for a plotting function:
```python
"""
Plot spectrogram with proper formatting.

.. meta::
    :category: Plotting - Spectral Analysis
    :matlab_equivalent: plot_spectrogram_shm
    :complexity: Basic
    :data_type: Spectral Data
    :output_type: Visualization
    :display_name: Plot Spectrogram
    
Parameters
----------
...
```