# Contributing to SHMTools

## Development Workflow

### 1. Issue-Based Development

All work starts with a GitHub issue:
- **Bug reports**: Use the bug report template
- **Feature requests**: Use the feature request template  
- **Notebook issues**: Use the notebook issue template

### 2. Branch and Fix Workflow

```bash
# 1. Create a branch for the issue
git checkout -b fix/issue-123-description

# 2. Make your changes
# ... edit code, add tests, update docs ...

# 3. Run quality checks
black shmtools/
flake8 shmtools/
pytest

# 4. Commit with descriptive message
git commit -m "Fix issue with Mahalanobis distance calculation

Resolves #123 by handling zero variance case properly.
- Add defensive programming for zero std deviation
- Update tests to cover edge case
- Validate against MATLAB reference"

# 5. Push and create pull request
git push -u origin fix/issue-123-description
gh pr create --title "Fix Mahalanobis zero variance issue" --body "Resolves #123"
```

### 3. Development Setup

```bash
# Clone and setup
git clone https://github.com/ebpfly/shm.git
cd shm/shmtools-python

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Code Standards

### Function Conversion Rules

When converting MATLAB functions to Python:

1. **Read the original MATLAB file completely** in `../shmtools-matlab/SHMFunctions/`
2. **Preserve exact algorithms** - don't "improve" the original logic
3. **Match function signatures** exactly where possible
4. **Add comprehensive docstrings** with GUI metadata
5. **Include MATLAB compatibility notes**

### Docstring Format

```python
def example_function_shm(data: np.ndarray, param: int) -> np.ndarray:
    """
    Brief description of the function.
    
    .. meta::
        :category: Core - Signal Processing
        :matlab_equivalent: exampleFunction_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Features
        
    Parameters
    ----------
    data : array_like
        Input data description.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    param : int
        Parameter description.
        
        .. gui::
            :widget: number_input
            :min: 1
            :max: 100
            :default: 10
    """
```

### Testing Requirements

- All functions must have unit tests
- Tests must cover edge cases (zero variance, empty arrays, etc.)
- Validate against MATLAB outputs where possible
- Notebook examples must run without errors

### Code Quality

```bash
# Format code
black shmtools/ bokeh_shmtools/

# Lint code  
flake8 shmtools/ bokeh_shmtools/

# Type checking (encouraged)
mypy shmtools/

# Run tests
pytest
pytest --cov=shmtools  # with coverage
```

## Pull Request Process

1. **Create descriptive PR title**: "Fix AR model indexing bug" not "Bug fix"
2. **Reference the issue**: "Resolves #123" in description  
3. **Fill out PR template** completely
4. **Ensure CI passes**: All GitHub Actions must pass
5. **Request review**: Tag relevant maintainers

### PR Requirements

- [ ] All tests pass
- [ ] Code follows style guidelines (black, flake8)
- [ ] Functions have proper docstrings
- [ ] No breaking changes (or clearly documented)
- [ ] Manual testing performed
- [ ] MATLAB compatibility verified (if applicable)

## Development Tips

### JupyterLab Extension Development

The extension requires a specific build process:

```bash
cd shm_function_selector/

# Step 1: Compile TypeScript
npm run build:lib

# Step 2: Build extension
npm run build:labextension:dev

# Step 3: Install in JupyterLab
cd .. && jupyter lab build
```

### Common Issues

- **Indexing**: Convert MATLAB 1-based to Python 0-based carefully
- **Numerical stability**: Add defensive programming for edge cases
- **Path handling**: Use robust path resolution for notebook execution
- **Data loading**: Ensure notebooks work from multiple working directories

### Validation Process

All changes must be validated against the original MATLAB implementation:

1. **Load identical test data** in both environments
2. **Compare outputs numerically** (within tolerance)
3. **Document any intentional differences**
4. **Update validation reports** in `validation/comparison_results/`

## Repository Structure

```
shmtools-python/
├── shmtools/          # Core Python library
├── bokeh_shmtools/    # Web interface (legacy)
├── shm_function_selector/  # JupyterLab extension
├── examples/          # Jupyter notebooks
│   ├── notebooks/     # Categorized examples
│   └── data/          # Shared datasets
├── tests/            # Test suite
└── validation/       # MATLAB comparison reports
```

## Getting Help

- **Documentation**: Check README.md and docs/ folder first
- **Discussions**: Use GitHub Discussions for general questions
- **Issues**: Create specific issues for bugs or feature requests
- **Code Review**: All PRs require review before merging

## Release Process

Releases follow semantic versioning (SemVer):
- **Patch** (0.1.1): Bug fixes, documentation updates
- **Minor** (0.2.0): New features, backward compatible
- **Major** (1.0.0): Breaking changes

Each release must include:
- Updated version numbers
- Release notes with changes
- Validation against MATLAB reference
- Working notebook examples