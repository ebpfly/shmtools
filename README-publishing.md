# Notebook Publishing

This directory contains scripts to automatically find, execute, and publish all SHMTools example notebooks to HTML format.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# Publish all notebooks (recommended for development)
python publish_notebooks.py --skip-errors

# Full publication (for release)
python publish_notebooks.py --clean --timeout 1200
```

## What It Does

The `publish_notebooks.py` script:

1. **Discovers** all `.ipynb` files in `examples/notebooks/`
2. **Categorizes** them by directory (basic, intermediate, advanced, specialized)
3. **Executes** each notebook with proper error handling
4. **Converts** to HTML with professional styling and navigation
5. **Creates** an index page linking all published notebooks

## Output

Creates a `published_notebooks/` directory with:

- Professional HTML versions of all notebooks
- Organized by difficulty level
- Comprehensive index page with statistics
- Preserved plots, outputs, and formatting

## Options

- `--skip-errors`: Publish notebooks even if execution fails (recommended during development)
- `--clean`: Start fresh by cleaning output directory
- `--timeout 1800`: Set execution timeout (default: 900 seconds)
- `--examples-dir PATH`: Use different notebooks directory
- `--output-dir PATH`: Specify output directory

## Example Output

```
Found 13 notebooks across 5 categories
  Basic: 5 notebooks (PCA, Mahalanobis, SVD outlier detection, etc.)
  Intermediate: 1 notebook (Factor analysis)
  Advanced: 5 notebooks (Active sensing, modal analysis, etc.)
  Specialized: 1 notebook (Sensor diagnostics)
  Other: 1 notebook (Data loader demo)

✅ Successfully published 13 notebooks!
📖 Open published_notebooks/index.html to view results
```

## Troubleshooting

- **Import errors**: Make sure `pip install -e .` was run
- **Missing functions**: Some notebooks may fail due to unimplemented functions
- **Timeouts**: Increase timeout or use `--skip-errors` for development
- **Data files**: Ensure required `.mat` files are in `examples/data/`

For detailed documentation, see `docs/notebook-publishing.md`.

## Integration

- **CI/CD**: Can be integrated into GitHub Actions for automatic publication
- **Documentation**: Generated HTML can be deployed to GitHub Pages or documentation sites
- **Development**: Use `--skip-errors` for quick iteration during development