# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a **unified structural health monitoring (SHM) toolkit** that combines:

1. **Modern Python conversion** with JupyterLab extension interface (primary development)
2. **Original MATLAB reference code** for algorithm verification and compatibility

The Python version is being developed using **example-driven development**, converting MATLAB functionality phase by phase while maintaining API compatibility.

This software is life-safety critical. You cannot skip requirements just to get the code to run. You also cannot claim success unless requirements are met. You don't get to move the goal posts just because it's hard.

## Working Directory Structure

```
/Users/eric/repo/shm/
├── shmtools/             # Core Python library (signal processing, ML, modal analysis)
├── shm_function_selector/ # JupyterLab extension for interactive function selection
├── examples/             # Example notebooks and data
│   ├── notebooks/        # Jupyter notebooks organized by category
│   └── data/             # Example datasets (.mat files)
├── matlab/               # Original MATLAB reference code (read-only reference)
│   ├── SHMTools/         # Core MATLAB SHMTools library
│   ├── LADPackage/       # Legacy Academic Development package
│   └── mFUSE/            # Java GUI for MATLAB toolkit
├── jupyterhub/           # AWS cloud deployment infrastructure
├── tests/                # Test suites for validation
├── validation/           # MATLAB-Python comparison validation
└── published_notebooks/  # Published HTML versions of notebooks
```

**IMPORTANT**: 
- **The git repository root is `/Users/eric/repo/shm/`** - use this directory for all git operations
- **Primary development** happens in the root directory with the Python `shmtools/` package
- **MATLAB reference code** is in `matlab/` for algorithm verification only
- **JupyterLab extension** in `shm_function_selector/` provides interactive GUI functionality
- **AWS deployment is the primary deployment method** - use `jupyterhub/` for cloud infrastructure

## Python Development Commands

### Setup and Installation
```bash
# Work from repository root
cd /Users/eric/repo/shm/

# Install dependencies
pip install -r requirements.txt

# Install in development mode  
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=shmtools --cov=bokeh_shmtools

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m "not hardware"      # Skip hardware tests  
pytest -m integration         # Run only integration tests
pytest -m "requires_data"     # Run tests requiring example datasets
```

### Code Quality
```bash
# Format code
black shmtools/

# Lint code
flake8 shmtools/

# Type checking
mypy shmtools/
```

### Running JupyterLab with Extension
```bash
# Activate virtual environment first
source venv/bin/activate

# Start JupyterLab (preferred method)
jupyter lab

# Extension provides interactive function selection
# Access at http://localhost:8888
```

### JupyterLab Extension Development
**CRITICAL**: The JupyterLab extension requires a specific 3-step build process that must be followed exactly:

```bash
# ALWAYS work from the extension directory
cd shm_function_selector/

# Step 1: Compile TypeScript to JavaScript (REQUIRED FIRST)
npm run build:lib

# Step 2: Build the JupyterLab extension (uses compiled JS)
npm run build:labextension:dev

# Step 3: Integrate extension into JupyterLab (from parent directory)
cd ..
source venv/bin/activate
jupyter lab build
```

**IMPORTANT BUILD NOTES**:
- **NEVER** skip Step 1 - TypeScript changes won't appear without it
- **NEVER** run `jupyter lab build` alone - it won't pick up extension changes
- If changes don't appear, clear cache: `rm -rf shm_function_selector/shm_function_selector/labextension/static/*.js`
- Always check the generated file hash changes to confirm new build
- Browser refresh may be required after `jupyter lab build`

**Debugging Extension Issues**:
```bash
# Check if TypeScript compiled correctly
cd shm_function_selector/
npm run build:lib
grep "your_debug_text" lib/index.js

# Check if extension built correctly  
npm run build:labextension:dev
grep "your_debug_text" shm_function_selector/labextension/static/lib_index_js.*.js

# Force complete rebuild if stuck
rm -rf shm_function_selector/labextension/static/*.js
npm run build:lib
npm run build:labextension:dev
cd .. && source venv/bin/activate && jupyter lab build
```

## AWS Cloud Deployment (Primary Deployment Method)

### Overview

**SHMTools is designed for cloud deployment on AWS EC2 using JupyterHub.** This provides a complete, scalable research environment accessible through any web browser with no local installation required.

The cloud deployment includes:
- **JupyterHub server** for multi-user access
- **Complete SHMTools environment** with all dependencies
- **JupyterLab extension** with 108+ SHM functions
- **Claude Code integration** for AI-assisted development
- **Automatic security and access control**

### Quick Start

#### Prerequisites
- AWS CLI configured with appropriate credentials
- GitHub Personal Access Token (PAT) with repository access
- Command-line tools: `aws`, `jq`, `curl`, `ssh-keygen`

#### One-Command Deployment
```bash
cd jupyterhub/
./setup_jupyterhub_aws.sh
```

The script automatically:
1. Creates secure AWS infrastructure (IAM roles, security groups)
2. Launches Ubuntu 22.04 EC2 instance with JupyterHub
3. Installs complete SHMTools environment 
4. Builds and configures JupyterLab extension
5. Sets up Claude Code integration
6. Provides web access at `http://<PUBLIC_IP>`

**Total setup time**: ~5-10 minutes for a complete research environment.

### Architecture Components

#### Infrastructure Layer
- **EC2 Instance**: t3.medium Ubuntu 22.04 (configurable)
- **Storage**: 20GB GP3 volume with automatic scaling
- **Security**: Restricted SSH + open HTTP access
- **IAM**: Minimal-permission role for secure GitHub access
- **Region**: us-east-2 (configurable)

#### Software Stack
- **TLJH (The Littlest JupyterHub)**: Multi-user Jupyter environment
- **Node.js 20.x**: Required for JupyterLab 4.4+ compatibility
- **Python 3.10/3.12**: Dual environment (hub + user spaces)
- **SHMTools Package**: Complete library with all 108+ functions
- **JupyterLab Extension**: Interactive function selector and parameter linking
- **Claude Code**: AI-assisted development environment

#### User Experience
- **Web-based access**: No local installation required
- **Multi-user support**: Individual user sessions and data
- **Interactive development**: JupyterLab with SHM function integration
- **AI assistance**: Built-in Claude Code for code generation and debugging
- **Data persistence**: User data survives across sessions

### Configuration

#### Basic Configuration (`jupyterhub/setup_jupyterhub_aws.sh`)
```bash
# Edit these variables before deployment
AWS_PROFILE="default"
AWS_REGION="us-east-2"
INSTANCE_TYPE="t3.medium"
GITHUB_OWNER="your-github-username"
GITHUB_REPO="your-repo-name"
GIT_USER_NAME="Your Name"
GIT_USER_EMAIL="your.email@example.com"
```

#### Advanced Configuration
- **Instance scaling**: Modify `INSTANCE_TYPE` for more compute/memory
- **Storage expansion**: Adjust `VOLUME_SIZE_GB` for larger datasets
- **Multi-region**: Change `AWS_REGION` for global accessibility
- **Custom domains**: Configure Route 53 and SSL certificates

### Development Workflow

#### Standard Development Cycle
```bash
# 1. Deploy instance
cd jupyterhub/
./setup_jupyterhub_aws.sh

# 2. Access environment
# Navigate to http://<PUBLIC_IP>
# Login with username: ubuntu (set password on first login)

# 3. Develop in JupyterLab
# - Use SHM Function Selector for interactive development
# - Access Claude Code for AI assistance
# - All dependencies pre-installed and configured

# 4. Iterate and test
# - Changes persist across sessions
# - Real-time collaboration possible
# - Full Git integration available
```

#### Development Debugging
For development and testing without recreating instances:

```bash
# Use existing instance
DEBUG_MODE=true EXISTING_INSTANCE_IP=1.2.3.4 ./setup_jupyterhub_aws.sh

# Individual component installation
./debug_install.sh <IP> all        # Full reinstall
./debug_install.sh <IP> shmtools   # Just shmtools package
./debug_install.sh <IP> extension  # Just JupyterLab extension
./debug_install.sh <IP> claude     # Just Claude Code

# Status monitoring
./check_status.sh <IP>
```

### Security and Access Control

#### Built-in Security Features
- **Encrypted secrets**: GitHub PAT stored in AWS SSM Parameter Store
- **Network security**: SSH restricted to deployment IP, HTTP open for web access  
- **IAM permissions**: Minimal-privilege role with read-only SSM access
- **User isolation**: Individual user sessions with data separation
- **Automatic updates**: Security patches applied during deployment

#### User Management
```bash
# SSH to instance for user administration
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<PUBLIC_IP>

# Add users (admin required)
sudo tljh-config set users.allowed username
sudo tljh-config reload

# Grant admin access
sudo tljh-config set users.admin username
sudo tljh-config reload
```

### Cost Management

#### Instance Costs (us-east-2)
- **t3.medium**: ~$0.04/hour (~$30/month continuous)
- **t3.large**: ~$0.08/hour (~$60/month continuous) 
- **Storage**: ~$2/month for 20GB GP3

#### Cost Optimization
```bash
# Stop instance when not in use
aws ec2 stop-instances --instance-ids <INSTANCE_ID>

# Start when needed
aws ec2 start-instances --instance-ids <INSTANCE_ID>

# Terminate when project complete
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

### Production Considerations

#### Scaling for Classes/Workshops
- **Multi-user**: TLJH supports 50+ concurrent users on larger instances
- **Resource allocation**: Configure memory/CPU limits per user
- **Storage scaling**: EBS volumes can be expanded without downtime
- **Load balancing**: Deploy multiple instances behind ALB for 100+ users

#### Enterprise Features
- **Custom domains**: Route 53 + CloudFront for branded access
- **SSL/TLS**: Let's Encrypt integration available
- **Authentication**: LDAP/AD integration through TLJH
- **Monitoring**: CloudWatch integration for usage metrics
- **Backup**: EBS snapshots for data protection

### Troubleshooting

#### Common Issues
```bash
# Installation monitoring
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<IP> "sudo tail -f /var/log/cloud-init-output.log"

# Service status
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<IP> "sudo systemctl status jupyterhub"

# Extension debugging  
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<IP> "jupyter labextension list | grep shm"

# Function discovery test
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<IP> "python3 -c 'from shmtools.introspection import discover_functions_locally; print(len(discover_functions_locally()))'"
```

#### Recovery Procedures
- **Soft reset**: Restart JupyterHub service
- **Hard reset**: Reboot EC2 instance  
- **Full recovery**: Terminate and redeploy (5-10 minutes)
- **Data recovery**: EBS snapshots (configure separately)

### Why Cloud Deployment?

#### Advantages Over Local Installation
1. **Zero setup friction**: Complete environment in 5-10 minutes
2. **Consistent environment**: Same setup for all researchers
3. **Collaborative**: Multiple users can work simultaneously
4. **Scalable**: Easy to provision more compute/memory
5. **Accessible**: Works from any device with web browser
6. **Cost-effective**: Pay only when used, stop when idle
7. **Maintainable**: Infrastructure as code, reproducible deployments

#### Research Environment Benefits
1. **Complete SHMTools library**: All 108+ functions available immediately  
2. **Interactive development**: JupyterLab extension for function discovery
3. **AI assistance**: Claude Code integration for faster development
4. **Data handling**: Pre-configured for .mat files and large datasets
5. **Reproducibility**: Exact environment specifications in code
6. **Sharing**: Easy to spin up identical environments for collaborators

**The cloud deployment is the recommended approach for all SHMTools research and development work.**

## Development Architecture

### Core Python Library (`shmtools/`)

**Two-tier function architecture**: 

- **`core/`** - Signal processing fundamentals (spectral analysis, filtering, statistics)
- **`features/`** - Feature extraction (time series modeling, AR models) 
- **`classification/`** - ML and outlier detection (Mahalanobis, PCA, SVD)
- **`modal/`** - Modal analysis and structural dynamics
- **`active_sensing/`** - Guided wave analysis and propagation
- **`hardware/`** - Data acquisition interfaces (NI-DAQmx, serial)
- **`plotting/`** - Bokeh-specific visualization utilities  
- **`utils/`** - Data I/O, MATLAB compatibility, general utilities

### JupyterLab Extension (`shm_function_selector/`)

**Interactive function selection and parameter linking**:

- **Frontend (`src/`)** - TypeScript components for JupyterLab integration
- **Backend (`shm_function_selector/handlers.py`)** - Python API handlers for function discovery
- **Build System** - npm/webpack for TypeScript compilation and extension packaging

### Example Notebooks (`examples/notebooks/`)

**Organized educational notebooks** demonstrating SHMTools functionality:

- **`basic/`** - Introductory examples (AR models, PCA, Mahalanobis)
- **`intermediate/`** - Advanced outlier detection and feature extraction
- **`advanced/`** - Modal analysis and active sensing
- **`condition_based_monitoring/`** - CBM applications and rotating machinery
- **`modal_analysis/`** - Structural dynamics and optimal sensor placement
- **`active_sensing/`** - Guided wave analysis and sensor diagnostics
- **`outlier_detection/`** - Various detection algorithms and custom assemblies

## Critical Development Principles

### Example-Driven Development Strategy

**Completion marked by working Jupyter notebooks**, not just function stubs. Each phase converts **one complete example** with all its dependencies, validates the output matches MATLAB, and publishes a working notebook.

### Quality Gates for Each Example
1. **MATLAB Analysis**: Read and understand the original `.m` file completely
2. **Dependency Mapping**: Identify all required SHMTools functions
3. **Function Conversion**: Convert each dependency with defensive programming and numerical stability
4. **Algorithm Validation**: Test with synthetic data where ground truth is known
5. **Integration Testing**: Full workflow validation with real data
6. **Notebook Creation**: Create educational Jupyter notebook with robust execution context handling
7. **Execution Testing**: Ensure notebook runs end-to-end from multiple working directories
8. **Publication Validation**: HTML export with all outputs and accessible visualizations

## Data Management Strategy

### Manual Data Setup (Simple Approach)

Users manually download all example datasets once and place them in the repository.

#### Required Datasets
1. **`data3SS.mat`** (25MB) - 3-story structure data
   - **Used by**: 9+ examples (PCA, Mahalanobis, SVD, NLPCA, etc.)
   - **Format**: `(8192 time points, 5 channels, 170 conditions)`
   - **Description**: Base-excited 3-story structure with 17 damage states (10 tests each)

2. **`dataSensorDiagnostic.mat`** (63KB) - Sensor health data
3. **`data_CBM.mat`** (54MB) - Condition-based monitoring data  
4. **`data_example_ActiveSense.mat`** (32MB) - Active sensing data
5. **`data_OSPExampleModal.mat`** (50KB) - Modal analysis data

**Total**: ~161MB (reasonable for manual download)

### Repository Structure
```bash
/Users/eric/repo/shm/
├── examples/
│   ├── data/                          # User puts downloaded .mat files here
│   │   ├── README.md                  # Download instructions
│   │   ├── .gitignore                 # Ignore *.mat files  
│   │   ├── data3SS.mat               # ← User downloads these
│   │   ├── dataSensorDiagnostic.mat  # ← 
│   │   ├── data_CBM.mat              # ←
│   │   ├── data_example_ActiveSense.mat # ←
│   │   └── data_OSPExampleModal.mat  # ←
│   └── notebooks/
│       ├── basic/
│       ├── intermediate/
│       ├── advanced/
│       ├── condition_based_monitoring/
│       ├── modal_analysis/
│       ├── active_sensing/
│       └── outlier_detection/
└── shmtools/
    └── utils/
        └── data_loading.py           # Simple .mat file loading
```

## Key Lessons from Phase 1

**Critical Issues Identified and Resolved:**

1. **MATLAB Indexing Conversion**: Most significant bug was incorrect 1-based to 0-based indexing conversion in AR model. Always use explicit `matlab_k = k + 1` conversion approach.

2. **Numerical Stability**: Add defensive programming for zero standard deviations (`data_std = np.where(data_std == 0, 1.0, data_std)`) and singular matrices (small regularization term).

3. **Execution Context**: Notebooks must handle multiple working directories and execution contexts (Jupyter, nbconvert, different CWDs). Use robust path resolution with fallback options.

4. **Visualization**: Prefer `plt.legend()` over hardcoded `plt.text()` positioning for better portability across contexts.

5. **Testing Strategy**: Always validate algorithms with synthetic test cases where ground truth is known before testing with real data.

**See `docs/conversion-plan.md` for complete lessons learned documentation.**

## MATLAB Function Conversion Rules

**CRITICAL**: Before implementing ANY function:

1. **Read Original MATLAB File**: Must examine the complete `.m` file in `matlab/SHMTools/SHMFunctions/`
2. **Extract Exact Algorithm**: Document mathematical steps precisely  
3. **Preserve All Information**: Only convert what exists in original MATLAB
4. **Verify Function Signature**: Match input/output parameters exactly
5. **Check Dependencies**: Identify all MATLAB function dependencies

### Function Naming Convention

- **MATLAB-compatible**: `psdWelch_shm()` (with `_shm` suffix)
- **Example**: `shmtools.core.psd_welch_shm()` 

### Docstring Format

Uses **machine-readable docstrings** for automatic GUI generation:

```python
def ar_model_shm(data: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate autoregressive model parameters and compute RMSE.
    
    .. meta::
        :category: Features - Time Series Models  
        :matlab_equivalent: arModel_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        
    Parameters
    ----------
    data : array_like
        Input time series data.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    order : int
        AR model order.
        
        .. gui::
            :widget: number_input
            :min: 1
            :max: 50
            :default: 15
    """
```

## Key Dependencies and Technologies

- **Core**: numpy, scipy, scikit-learn, matplotlib, pandas
- **Web Interface**: bokeh >=2.4.0 (4-panel layout)
- **Optional Hardware**: nidaqmx, pyserial
- **Advanced Features**: pywavelets, numba, joblib
- **Development**: pytest, black, flake8, mypy, pre-commit

## Migration from MATLAB

### Compatibility Features
- Function signatures match MATLAB interfaces where possible
- Session files (.ses) from mFUSE can be imported to workflows
- Data formats support MATLAB .mat file loading via scipy.io
- Parameter naming preserves MATLAB conventions

### Reference Documentation
- **Conversion Plan**: `docs/conversion-plan.md` - detailed phase planning
- **Docstring Format**: `docs/docstring-format.md` - GUI integration specs
- **MATLAB Reference**: Use `matlab/` for algorithm verification

## Common Development Workflows

### Conversion Workflow for Each Example

#### 1. Analysis Phase
```bash
# Read original MATLAB file completely
# Identify all function dependencies
# Map data flow and algorithm steps
# Note visualization requirements
```

#### 2. Function Conversion Phase  
```bash
# Convert each dependency function to Python
# Follow docs/docstring-format.md exactly
# Add comprehensive docstrings with GUI metadata
```

#### 3. Validation Phase
```bash
# Test each function with identical MATLAB inputs
# Validate outputs match to numerical precision
# Create unit tests for regression protection
```

#### 4. Notebook Creation Phase
```bash
# Create educational Jupyter notebook
# Include background theory and references
# Add explanatory text between code cells
# Create publication-quality visualizations
```

#### 5. Testing and Publication Phase
```bash
# Run complete notebook end-to-end
pytest examples/test_notebooks/test_*.py

# Export to HTML
jupyter nbconvert --to html notebook.ipynb

# Validate HTML renders correctly
# Check all plots display properly
```

### Adding New Functions
1. Identify target MATLAB function in `matlab/SHMTools/SHMFunctions/`
2. Read complete `.m` file to understand algorithm
3. Add machine-readable docstring with GUI specifications
4. Write tests in `tests/test_shmtools/`
5. Update function exports in module `__init__.py`

### Testing Hardware Functions
```bash
# Skip hardware tests during normal development
pytest -m "not hardware"

# Run hardware tests when DAQ equipment available
pytest -m hardware
```

### Working with JupyterLab Extension
```bash
# Build and install extension for development
cd shm_function_selector/
npm run build:lib
npm run build:labextension:dev
cd ..
source venv/bin/activate
jupyter lab build

# Start JupyterLab with extension
jupyter lab
```

## Success Criteria

### Quality Assurance Checklist

For each converted example:
- [ ] All MATLAB functions identified and mapped
- [ ] Python functions follow exact MATLAB algorithms  
- [ ] Docstrings include all required GUI metadata
- [ ] Outputs match MATLAB within numerical tolerance
- [ ] Jupyter notebook runs without errors
- [ ] HTML export renders cleanly
- [ ] All visualizations display correctly
- [ ] Educational content explains the methodology

### Success Metrics

- **Functional Parity**: Python results match MATLAB exactly
- **Reusability**: Functions work across multiple examples  
- **Documentation Quality**: Notebooks suitable for publication
- **GUI Integration**: Docstring metadata enables automatic web interface
- **Performance**: Conversion maintains or improves execution speed

This approach ensures each example provides immediate value while building a robust foundation for the complete SHMTools conversion.

# GitHub Issue Resolution Workflow

## Overview

This repository uses a structured GitHub workflow for collaborative development. Claude Code can handle the complete issue-to-resolution cycle automatically.

## Issue Resolution Process

### 1. Identifying Issues

Issues can be found at: `https://github.com/ebpfly/shm/issues`

**Check issue status:**
```bash
# List all open issues
gh issue list

# View specific issue details
gh issue view 123

# Check issue labels and priority
gh issue view 123 --json labels,assignees,title,body
```

### 2. Branch Creation

**ALWAYS** create a descriptive branch for each issue:

```bash
# Branch naming convention: fix/issue-NUMBER-description
git checkout -b fix/issue-123-ar-model-error-handling

# For features: feature/issue-NUMBER-description  
git checkout -b feature/issue-124-new-psd-method

# For documentation: docs/issue-NUMBER-description
git checkout -b docs/issue-125-update-readme
```

### 3. Implementation Guidelines

#### For Bug Fixes:
1. **Reproduce the issue** first with a test case
2. **Locate the problem** in the codebase
3. **Implement defensive programming** (input validation, error handling)
4. **Maintain backward compatibility** unless explicitly breaking change
5. **Add test cases** to prevent regression

#### For New Features:
1. **Check MATLAB reference** in `matlab/SHMTools/SHMFunctions/` if applicable
2. **Follow existing patterns** in the codebase
3. **Add comprehensive docstrings** with GUI metadata
4. **Include examples** in docstring
5. **Update module `__init__.py`** to export new functions

#### For Notebook Issues:
1. **Test notebook execution** from multiple working directories
2. **Verify data file loading** works correctly
3. **Check all visualizations** render properly
4. **Validate HTML export** works without errors
5. **Update notebook metadata** if needed

### 4. Quality Assurance (MANDATORY)

**Before committing any changes:**

```bash
# Format code (required)
black shmtools/

# Lint code (required)
flake8 shmtools/

# Type checking (recommended)
mypy shmtools/

# Run tests (required for core changes)
pytest tests/test_shmtools/ -v

# Test notebooks (for notebook-related changes)
pytest tests/test_notebooks/ -k "basic" -v

# Test specific functionality
python -c "import shmtools; # test your changes"
```

### 5. Commit Standards

Use descriptive commit messages with issue references:

```bash
git commit -m "Fix AR model input validation

Resolves #123 by adding comprehensive error checking:
- Validate input array dimensions and size
- Add clear error messages for common mistakes  
- Maintain backward compatibility with valid inputs
- Add test cases for edge cases

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 6. Pull Request Creation

```bash
# Push branch
git push -u origin fix/issue-123-description

# Create PR with detailed description
gh pr create --title "Fix AR model input validation" --body "$(cat <<'EOF'
## Summary
Brief description of changes made.

## Issue Reference  
Resolves #123

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring

## Changes Made
- Specific change 1
- Specific change 2
- Specific change 3

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed
- [ ] Notebook examples still work (if applicable)

## Validation
- [ ] Code follows project style guidelines (black, flake8)
- [ ] Functions have proper docstrings with GUI metadata
- [ ] MATLAB compatibility verified (if applicable)
- [ ] No breaking changes to existing API

## Additional Notes
Any additional context about the changes.
EOF
)"
```

### 7. Merging Process

```bash
# Check PR status and tests
gh pr status

# View PR checks (GitHub Actions)
gh pr checks

# Merge after approval (squash merge preferred)
gh pr merge --squash --delete-branch
```

## Common Issue Types

### Bug Reports
- **Input validation errors**: Add defensive programming
- **MATLAB compatibility issues**: Check indexing (1-based vs 0-based)
- **Numerical stability**: Handle edge cases (zeros, infinities, NaN)
- **Notebook execution failures**: Check path handling, data loading

### Feature Requests  
- **New MATLAB functions**: Follow conversion rules exactly
- **GUI enhancements**: Update docstring metadata
- **Performance improvements**: Validate against original algorithms
- **API extensions**: Maintain backward compatibility

### Documentation Issues
- **Notebook problems**: Test execution contexts
- **Missing examples**: Add comprehensive usage examples
- **API documentation**: Update docstrings and type hints
- **Installation problems**: Update requirements and setup instructions

## Validation Requirements

### For All Changes:
- [ ] Code style compliance (black, flake8)
- [ ] Existing tests pass
- [ ] No breaking changes (or clearly documented)
- [ ] Proper error handling added
- [ ] Documentation updated if needed

### For Function Changes:
- [ ] MATLAB reference algorithm preserved
- [ ] Input validation added
- [ ] Comprehensive docstrings with GUI metadata  
- [ ] Type hints included
- [ ] Examples in docstring work correctly

### For Notebook Changes:
- [ ] Executes from multiple working directories
- [ ] All visualizations render correctly
- [ ] Data loading works reliably
- [ ] HTML export successful
- [ ] Educational content is clear

## Emergency Procedures

### Broken Main Branch:
```bash
# Create hotfix branch from main
git checkout -b hotfix/critical-fix main

# Make minimal fix
# ... implement fix ...

# Fast-track PR process
gh pr create --title "HOTFIX: Critical issue description" --body "Emergency fix for production issue"
gh pr merge --squash
```

### Failed Tests in PR:
1. **Never merge failing tests** - investigate and fix
2. **Check GitHub Actions logs** for specific failures  
3. **Run tests locally** to reproduce issues
4. **Update PR** with fixes, don't create new PR

### Rollback if Needed:
```bash
# Identify problematic commit
git log --oneline -10

# Create revert PR
git checkout -b revert/issue-123-rollback
git revert COMMIT_HASH
git push -u origin revert/issue-123-rollback
gh pr create --title "Revert problematic changes"
```

## Success Metrics

A successfully resolved issue should:
- [ ] **Close automatically** when PR is merged (use "Resolves #123" in PR)
- [ ] **Pass all automated tests** (GitHub Actions)
- [ ] **Maintain API compatibility** (no breaking changes)
- [ ] **Include proper documentation** (docstrings, comments)
- [ ] **Follow project conventions** (naming, style, patterns)
- [ ] **Be validated against MATLAB** (if applicable)

This structured approach ensures consistent, high-quality contributions while maintaining the project's reliability and usability.
```bash
# List all open issues
gh issue list

# View specific issue details
gh issue view 123

# Check issue labels and priority
gh issue view 123 --json labels,assignees,title,body
```

### 2. Branch Creation

**ALWAYS** create a descriptive branch for each issue:

```bash
# Branch naming convention: fix/issue-NUMBER-description
git checkout -b fix/issue-123-ar-model-error-handling

# For features: feature/issue-NUMBER-description  
git checkout -b feature/issue-124-new-psd-method

# For documentation: docs/issue-NUMBER-description
git checkout -b docs/issue-125-update-readme
```

### 3. Implementation Guidelines

#### For Bug Fixes:
1. **Reproduce the issue** first with a test case
2. **Locate the problem** in the codebase
3. **Implement defensive programming** (input validation, error handling)
4. **Maintain backward compatibility** unless explicitly breaking change
5. **Add test cases** to prevent regression

#### For New Features:
1. **Check MATLAB reference** in `matlab/SHMTools/SHMFunctions/` if applicable
2. **Follow existing patterns** in the codebase
3. **Add comprehensive docstrings** with GUI metadata
4. **Include examples** in docstring
5. **Update module `__init__.py`** to export new functions

#### For Notebook Issues:
1. **Test notebook execution** from multiple working directories
2. **Verify data file loading** works correctly
3. **Check all visualizations** render properly
4. **Validate HTML export** works without errors
5. **Update notebook metadata** if needed

### 4. Quality Assurance (MANDATORY)

**Before committing any changes:**

```bash
# Format code (required)
black shmtools/

# Lint code (required)
flake8 shmtools/

# Type checking (recommended)
mypy shmtools/

# Run tests (required for core changes)
pytest tests/test_shmtools/ -v

# Test notebooks (for notebook-related changes)
pytest tests/test_notebooks/ -k "basic" -v

# Test specific functionality
python -c "import shmtools; # test your changes"
```

### 5. Commit Standards

Use descriptive commit messages with issue references:

```bash
git commit -m "Fix AR model input validation

Resolves #123 by adding comprehensive error checking:
- Validate input array dimensions and size
- Add clear error messages for common mistakes  
- Maintain backward compatibility with valid inputs
- Add test cases for edge cases

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 6. Pull Request Creation

```bash
# Push branch
git push -u origin fix/issue-123-description

# Create PR with detailed description
gh pr create --title "Fix AR model input validation" --body "$(cat <<'EOF'
## Summary
Brief description of changes made.

## Issue Reference  
Resolves #123

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring

## Changes Made
- Specific change 1
- Specific change 2
- Specific change 3

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed
- [ ] Notebook examples still work (if applicable)

## Validation
- [ ] Code follows project style guidelines (black, flake8)
- [ ] Functions have proper docstrings with GUI metadata
- [ ] MATLAB compatibility verified (if applicable)
- [ ] No breaking changes to existing API

## Additional Notes
Any additional context about the changes.
EOF
)"
```

### 7. Merging Process

```bash
# Check PR status and tests
gh pr status

# View PR checks (GitHub Actions)
gh pr checks

# Merge after approval (squash merge preferred)
gh pr merge --squash --delete-branch
```

## Common Issue Types

### Bug Reports
- **Input validation errors**: Add defensive programming
- **MATLAB compatibility issues**: Check indexing (1-based vs 0-based)
- **Numerical stability**: Handle edge cases (zeros, infinities, NaN)
- **Notebook execution failures**: Check path handling, data loading

### Feature Requests  
- **New MATLAB functions**: Follow conversion rules exactly
- **GUI enhancements**: Update docstring metadata
- **Performance improvements**: Validate against original algorithms
- **API extensions**: Maintain backward compatibility

### Documentation Issues
- **Notebook problems**: Test execution contexts
- **Missing examples**: Add comprehensive usage examples
- **API documentation**: Update docstrings and type hints
- **Installation problems**: Update requirements and setup instructions

## Validation Requirements

### For All Changes:
- [ ] Code style compliance (black, flake8)
- [ ] Existing tests pass
- [ ] No breaking changes (or clearly documented)
- [ ] Proper error handling added
- [ ] Documentation updated if needed

### For Function Changes:
- [ ] MATLAB reference algorithm preserved
- [ ] Input validation added
- [ ] Comprehensive docstrings with GUI metadata  
- [ ] Type hints included
- [ ] Examples in docstring work correctly

### For Notebook Changes:
- [ ] Executes from multiple working directories
- [ ] All visualizations render correctly
- [ ] Data loading works reliably
- [ ] HTML export successful
- [ ] Educational content is clear

## Emergency Procedures

### Broken Main Branch:
```bash
# Create hotfix branch from main
git checkout -b hotfix/critical-fix main

# Make minimal fix
# ... implement fix ...

# Fast-track PR process
gh pr create --title "HOTFIX: Critical issue description" --body "Emergency fix for production issue"
gh pr merge --squash
```

### Failed Tests in PR:
1. **Never merge failing tests** - investigate and fix
2. **Check GitHub Actions logs** for specific failures  
3. **Run tests locally** to reproduce issues
4. **Update PR** with fixes, don't create new PR

### Rollback if Needed:
```bash
# Identify problematic commit
git log --oneline -10

# Create revert PR
git checkout -b revert/issue-123-rollback
git revert COMMIT_HASH
git push -u origin revert/issue-123-rollback
gh pr create --title "Revert problematic changes"
```

## Success Metrics

A successfully resolved issue should:
- [ ] **Close automatically** when PR is merged (use "Resolves #123" in PR)
- [ ] **Pass all automated tests** (GitHub Actions)
- [ ] **Maintain API compatibility** (no breaking changes)
- [ ] **Include proper documentation** (docstrings, comments)
- [ ] **Follow project conventions** (naming, style, patterns)
- [ ] **Be validated against MATLAB** (if applicable)

This structured approach ensures consistent, high-quality contributions while maintaining the project's reliability and usability.
