# Notebook Testing

This directory contains comprehensive tests for all example notebooks in the shmtools-python package.

## Test Organization

### Test Files

- **`test_all_notebooks.py`** - Main notebook execution tests organized by difficulty level
- **`test_data_validation.py`** - Data loading and availability validation tests
- **`conftest.py`** - Pytest configuration and shared fixtures
- **`utils/notebook_runner.py`** - Robust notebook execution utilities

### Test Categories

Tests are organized by notebook difficulty level and marked accordingly:

- **Basic** (`@pytest.mark.basic_notebook`) - Fundamental outlier detection methods
- **Intermediate** (`@pytest.mark.intermediate_notebook`) - More complex analysis methods  
- **Advanced** (`@pytest.mark.advanced_notebook`) - Specialized and computationally intensive methods
- **Specialized** (`@pytest.mark.specialized_notebook`) - Domain-specific applications

## Running Tests

### Prerequisites

Install test dependencies:
```bash
cd shmtools-python/
pip install -r requirements-dev.txt
```

### Basic Test Commands

Run all notebook tests:
```bash
pytest tests/test_notebooks/
```

Run only basic notebook tests:
```bash
pytest -m basic_notebook
```

Run tests that don't require data files:
```bash
pytest -m "not requires_data"
```

Run only data validation tests:
```bash
pytest tests/test_notebooks/test_data_validation.py
```

### Test Markers

Available pytest markers for filtering tests:

- **`notebook`** - All notebook-related tests
- **`basic_notebook`** - Basic level notebooks only
- **`intermediate_notebook`** - Intermediate level notebooks only  
- **`advanced_notebook`** - Advanced level notebooks only
- **`specialized_notebook`** - Specialized notebooks only
- **`requires_data`** - Tests requiring example datasets
- **`slow`** - Long-running tests (use `-m "not slow"` to skip)

### Example Test Commands

```bash
# Run all basic notebooks that don't require data
pytest -m "basic_notebook and not requires_data"

# Run data validation tests only
pytest -m requires_data tests/test_notebooks/test_data_validation.py

# Run all notebook tests except slow ones
pytest -m "notebook and not slow"

# Run notebook structure validation without execution
pytest tests/test_notebooks/test_all_notebooks.py::TestAllNotebooksQuick

# Run specific notebook test
pytest tests/test_notebooks/test_all_notebooks.py::TestBasicNotebooks::test_basic_notebook_execution[pca_outlier_detection.ipynb]
```

## Data Dependencies

### Required Data Files

The following data files are required for full test coverage:

1. **`data3SS.mat`** (25MB) - 3-story structure data
   - Used by: PCA, Mahalanobis, SVD outlier detection notebooks
   - Format: `(8192, 5, 170)` - time points, channels, conditions

2. **`dataSensorDiagnostic.mat`** (63KB) - Sensor health data
3. **`data_CBM.mat`** (54MB) - Condition-based monitoring data  
4. **`data_example_ActiveSense.mat`** (32MB) - Active sensing data
5. **`data_OSPExampleModal.mat`** (50KB) - Modal analysis data

### Data Setup

1. Create the data directory if it doesn't exist:
   ```bash
   mkdir -p examples/data/
   ```

2. Download data files to `examples/data/` directory

3. Tests will automatically detect available data files and skip tests for missing data

### Data-Independent Tests

Many tests can run without data files:

- Notebook structure validation
- JSON format validation  
- Basic execution tests for non-data-dependent notebooks
- Data loading infrastructure tests

## Test Results and Reporting

### Successful Test Output

```
✓ pca_outlier_detection.ipynb executed successfully:
  - Execution time: 12.34 seconds
  - Total cells: 25
  - Executed cells: 15
```

### Failed Test Output

```
FAILED: Notebook pca_outlier_detection.ipynb failed to execute:
Execution time: 5.67 seconds
Total cells: 25
Executed cells: 10
Error cells: 1

Errors:
  Cell 12: NameError: name 'load_3story_data' is not defined
```

### Data Availability Report

```
Data file availability in /path/to/examples/data:
  ✓ data3SS.mat
  ✗ dataSensorDiagnostic.mat
  ✗ data_CBM.mat
  ✗ data_example_ActiveSense.mat
  ✗ data_OSPExampleModal.mat

Summary: 1/5 data files available
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Test notebooks without data
  run: pytest -m "notebook and not requires_data"

- name: Test notebooks with data (if available)  
  run: pytest -m "notebook and requires_data" || true
```

### Local Development

For fast development cycles:

```bash
# Quick validation without execution
pytest tests/test_notebooks/test_all_notebooks.py::TestAllNotebooksQuick

# Test only basic notebooks
pytest -m basic_notebook -v
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `shmtools` is installed in development mode:
   ```bash
   pip install -e .
   ```

2. **Missing Data**: Tests will skip gracefully, but download data files for full coverage

3. **Timeout Errors**: Increase timeout in `conftest.py` or use `-m "not slow"`

4. **Kernel Issues**: Ensure Jupyter is properly installed:
   ```bash
   pip install jupyter notebook
   ```

### Debug Mode

For detailed execution information:

```bash
pytest tests/test_notebooks/ -v -s --tb=long
```

## Contributing

When adding new notebooks:

1. Place in appropriate difficulty directory (`basic/`, `intermediate/`, `advanced/`, `specialized/`)
2. Add to the parametrized test list in `test_all_notebooks.py`
3. Document any new data dependencies
4. Ensure notebooks have robust error handling for missing data

## Architecture

The test framework provides:

- **Robust execution** - Handles various execution contexts and working directories
- **Detailed reporting** - Comprehensive metadata collection and error reporting  
- **Graceful degradation** - Tests skip appropriately when data or dependencies missing
- **Performance tracking** - Execution time and resource usage monitoring
- **Flexible filtering** - Extensive marker system for selective test execution