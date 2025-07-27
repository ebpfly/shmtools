# Notebook Publishing Guide

This guide explains how to use the `publish_notebooks.py` script to find, execute, and convert all SHMTools example notebooks to publishable HTML format.

## Overview

The notebook publishing system provides:

- **Automatic Discovery**: Finds all `.ipynb` files in the examples directory
- **Robust Execution**: Executes notebooks with proper error handling and timeouts
- **HTML Conversion**: Converts to publication-quality HTML with custom styling
- **Categorization**: Organizes notebooks by difficulty level (basic, intermediate, advanced, specialized)
- **Index Generation**: Creates a comprehensive index page linking all published notebooks
- **Error Handling**: Gracefully handles execution failures and provides detailed reporting

## Quick Start

### Basic Usage

```bash
# Publish all notebooks with default settings
python publish_notebooks.py

# Publish with custom timeout and skip execution errors
python publish_notebooks.py --timeout 1200 --skip-errors

# Clean output directory and republish everything
python publish_notebooks.py --clean

# Use custom directories
python publish_notebooks.py --examples-dir custom/notebooks --output-dir published
```

### Prerequisites

Ensure you have the required dependencies installed:

```bash
# Install development dependencies (includes nbconvert)
pip install -r requirements-dev.txt

# Install shmtools in development mode
pip install -e .
```

## Command Line Options

### Basic Options

- `--examples-dir PATH`: Path to examples notebooks directory (default: `examples/notebooks`)
- `--output-dir PATH`: Output directory for published HTML files (default: `published_notebooks`)
- `--timeout SECONDS`: Timeout in seconds for notebook execution (default: 900)

### Execution Options

- `--skip-errors`: Publish notebooks even if execution fails (default: False)
- `--clean`: Clean output directory before publishing (default: False)

### Example Commands

```bash
# Standard publication
python publish_notebooks.py

# Fast publication (skip failed executions)
python publish_notebooks.py --skip-errors --timeout 300

# Complete rebuild
python publish_notebooks.py --clean --timeout 1800

# Development mode (shorter timeout, skip errors)
python publish_notebooks.py --timeout 180 --skip-errors
```

## Output Structure

The script creates a structured output directory:

```
published_notebooks/
├── index.html                 # Main index page
├── basic/                     # Basic level notebooks
│   ├── pca_outlier_detection.html
│   ├── mahalanobis_outlier_detection.html
│   └── ...
├── intermediate/              # Intermediate level notebooks
│   └── factor_analysis_outlier_detection.html
├── advanced/                  # Advanced level notebooks
│   ├── active_sensing_feature_extraction.html
│   └── ...
├── specialized/               # Specialized notebooks
│   └── sensor_diagnostics.html
├── other/                     # Miscellaneous notebooks
│   └── dataloader_demo.html
└── assets/                    # Static assets (if any)
```

## Features

### Automatic Categorization

Notebooks are automatically categorized based on their directory structure:

- **Basic**: Fundamental outlier detection and time series analysis
- **Intermediate**: More complex analysis techniques  
- **Advanced**: Specialized algorithms and computationally intensive methods
- **Specialized**: Domain-specific applications
- **Other**: Utility notebooks and demonstrations

### Enhanced HTML Output

Each published notebook includes:

- **Professional Styling**: Clean, responsive design with SHMTools branding
- **Navigation**: Links back to index and external resources
- **Metadata**: Generation timestamp, source notebook information
- **Syntax Highlighting**: Properly formatted code cells with syntax highlighting
- **Preserved Outputs**: All plots, tables, and results from execution

### Error Handling

The system provides robust error handling:

- **Execution Timeouts**: Prevents hanging on infinite loops or long computations
- **Graceful Degradation**: Can publish notebooks even if execution fails
- **Detailed Error Reporting**: Shows exactly which cells failed and why
- **Partial Success**: Successfully executed portions are still converted to HTML

### Index Page Features

The generated index page includes:

- **Statistics Dashboard**: Overview of total notebooks, successful publications, categories
- **Category Organization**: Notebooks grouped by difficulty level with descriptions
- **Status Indicators**: Visual indication of which notebooks published successfully
- **Responsive Design**: Works well on desktop and mobile devices

## Execution Process

### 1. Discovery Phase

```
Discovering notebooks...
   Found 13 notebooks across 5 categories
   Basic: 5 notebooks
   Intermediate: 1 notebooks
   Advanced: 5 notebooks
   Specialized: 1 notebooks
   Other: 1 notebooks
```

### 2. Execution Phase

For each notebook:

1. **Load Notebook**: Parse the .ipynb file
2. **Set Working Directory**: Configure execution context
3. **Execute Cells**: Run all code cells with timeout protection
4. **Collect Metadata**: Track execution time, errors, cell counts
5. **Report Results**: Show success/failure status

```
Processing Basic notebooks:
    Executing notebook: pca_outlier_detection.ipynb
      ✓ Executed in 45.2s (15 cells)
      ✓ HTML saved to: published_notebooks/basic/pca_outlier_detection.html
```

### 3. Conversion Phase

- **HTML Generation**: Convert executed notebook to HTML
- **Style Enhancement**: Apply custom CSS and layout
- **Asset Management**: Handle plots, images, and other outputs
- **Navigation Addition**: Add header, footer, and navigation links

### 4. Index Generation

- **Statistics Compilation**: Count successes, failures, categories
- **Page Generation**: Create comprehensive index with links
- **Category Descriptions**: Add explanatory text for each section

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'shmtools'
   ```
   **Solution**: Install shmtools in development mode: `pip install -e .`

2. **Missing Dependencies**
   ```
   ImportError: cannot import name 'ars_tach' from 'shmtools.core.signal_processing'
   ```
   **Solution**: Missing functions need to be implemented in the codebase

3. **Execution Timeouts**
   ```
   CellTimeoutError: A cell timed out while it was being executed
   ```
   **Solution**: Increase timeout with `--timeout 1800` or use `--skip-errors`

4. **Data File Errors**
   ```
   FileNotFoundError: data3SS.mat not found
   ```
   **Solution**: Ensure required data files are in `examples/data/`

### Debug Mode

For detailed debugging information:

```bash
# Run with verbose output
python publish_notebooks.py --skip-errors 2>&1 | tee publish.log

# Check specific notebook execution
python -c "
from publish_notebooks import NotebookPublisher
from pathlib import Path
pub = NotebookPublisher()
result = pub.execute_notebook(Path('examples/notebooks/basic/pca_outlier_detection.ipynb'))
print(result)
"
```

### Partial Publication

If some notebooks fail, you can still get useful output:

```bash
# Skip failed executions and publish what works
python publish_notebooks.py --skip-errors

# Focus on specific categories by moving notebooks temporarily
mkdir temp_advanced
mv examples/notebooks/advanced/* temp_advanced/
python publish_notebooks.py  # Publish only basic/intermediate
mv temp_advanced/* examples/notebooks/advanced/
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Publish Notebooks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Publish notebooks
      run: python publish_notebooks.py --skip-errors --timeout 600
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: published-notebooks
        path: published_notebooks/
```

### Local Development Workflow

```bash
# Quick development check (fast, skip errors)
python publish_notebooks.py --skip-errors --timeout 180

# Full publication test (before committing)
python publish_notebooks.py --clean --timeout 900

# Check specific category
python publish_notebooks.py --examples-dir examples/notebooks/basic --output-dir basic_only
```

## Customization

### Custom Styling

To modify the HTML appearance, edit the `enhance_html()` method in `publish_notebooks.py`:

```python
# Add custom CSS
custom_css = """
<style>
/* Your custom styles here */
.notebook-header {
    background: linear-gradient(135deg, #your-color1, #your-color2);
}
</style>
"""
```

### Custom Categories

To add new notebook categories, modify the `categorize_notebooks()` method:

```python
categories = {
    'basic': [],
    'intermediate': [],
    'advanced': [],
    'specialized': [],
    'experimental': [],  # New category
    'other': []
}
```

### Execution Configuration

Modify execution settings in the `NotebookPublisher` class:

```python
# Custom execution processor
self.executor = ExecutePreprocessor(
    timeout=your_timeout,
    kernel_name="python3",
    allow_errors=True,  # Allow execution errors
    interrupt_on_timeout=True
)
```

## Output Examples

### Successful Execution

```
SHMTools Notebook Publisher
==================================================

1. Discovering notebooks...
   Found 13 notebooks across 5 categories
   Basic: 5 notebooks
   Intermediate: 1 notebooks
   Advanced: 5 notebooks
   Specialized: 1 notebooks
   Other: 1 notebooks

2. Executing and publishing notebooks...

  Processing Basic notebooks:
    Executing notebook: pca_outlier_detection.ipynb
      ✓ Executed in 12.3s (19 cells)
      ✓ HTML saved to: published_notebooks/basic/pca_outlier_detection.html

3. Creating index page...

✓ Index page created: published_notebooks/index.html

==================================================
PUBLICATION SUMMARY
==================================================
Total notebooks processed: 13
Successful executions: 10
Successful publications: 13
Output directory: /path/to/published_notebooks

✅ Successfully published 13 notebooks!
📖 Open published_notebooks/index.html to view the published notebooks
```

### With Execution Errors

```
  Processing Basic notebooks:
    Executing notebook: ar_model_order_selection.ipynb
      ✗ Execution failed: ImportError: cannot import name 'ars_tach'
      ✓ HTML saved to: published_notebooks/basic/ar_model_order_selection.html

==================================================
PUBLICATION SUMMARY
==================================================
Total notebooks processed: 13
Successful executions: 8
Successful publications: 13
Output directory: /path/to/published_notebooks

Failed to publish (0):

✅ Successfully published 13 notebooks!
```

This comprehensive publishing system ensures that SHMTools example notebooks are always available in a professional, accessible format for users, documentation, and educational purposes.