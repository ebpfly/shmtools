# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working with this repository.

## ⚠️ CRITICAL REQUIREMENTS

### 1. ALWAYS Activate Virtual Environment First
```bash
cd /Users/eric/repo/shm/
source venv/bin/activate
```
**NEVER** run Python/pip/jupyter/pytest commands without activation.

### 2. NO Local-Only Fixes
- **NO** hardcoded paths (`/Users/eric/repo/shm/`)
- **NO** config files in venv directories
- **NO** manual Jupyter configs bypassing proper packaging
- **ALWAYS** fix root cause in package code, not workarounds

**Wrong**: Creating `/Users/eric/repo/shm/venv/etc/jupyter/jupyter_server_config.py` with hardcoded paths  
**Right**: Fix `__init__.py` to properly export server extension hooks

## Project Overview

**Life-safety critical** structural health monitoring (SHM) toolkit:
- **Python conversion** of MATLAB SHMTools with JupyterLab extension
- **Example-driven development** maintaining MATLAB API compatibility
- **No shortcuts** - requirements must be fully met, not approximated

## Directory Structure

```
/Users/eric/repo/shm/              # Git repository root
├── shmtools/                      # Core Python library
├── shm_function_selector/         # JupyterLab extension
├── examples/                      # Notebooks and .mat datasets
├── matlab/                        # MATLAB reference (read-only)
├── jupyterhub/                    # AWS deployment scripts
├── tests/                         # Test suites
└── published_notebooks/           # HTML exports
```

## Development Commands

### Setup
```bash
cd /Users/eric/repo/shm/ && source venv/bin/activate  # ALWAYS FIRST
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -m ipykernel install --user --name=shm-venv --display-name="SHM Python (venv)"
```

### Testing & Quality
```bash
pytest                                 # All tests
pytest -m "not hardware"              # Skip hardware tests
black shmtools/ && flake8 shmtools/  # Format & lint
```

### JupyterLab
```bash
jupyter lab  # Access at http://localhost:8888
# Use "SHM Python (venv)" kernel for notebooks
```

### JupyterLab Extension Build (CRITICAL 3-Step Process)
```bash
cd /Users/eric/repo/shm/ && source venv/bin/activate
cd shm_function_selector/
npm run build:lib                     # 1. Compile TypeScript
npm run build:labextension:dev        # 2. Build extension  
cd .. && jupyter lab build            # 3. Integrate into JupyterLab
```

**Testing Extension Updates**:
```bash
./restart_jupyterlab.sh              # Automated: build + restart + open browser
```
*Use this script after every extension update for immediate testing*

**Debugging**:
- Clear cache if stuck: `rm -rf shm_function_selector/shm_function_selector/labextension/static/*.js`
- Check compiled JS: `grep "debug_text" shm_function_selector/labextension/static/lib_index_js.*.js`
- CSS issues often due to `.shm-enhanced-dropdown` container overriding element styles
- Server log: `tail -f jupyterlab.log`

## AWS Cloud Deployment

### Quick Deploy (~5-10 minutes)
```bash
cd jupyterhub/
./setup_jupyterhub_aws.sh  # Creates EC2 with JupyterHub + SHMTools
# Access at http://<PUBLIC_IP>
```

**Stack**: Ubuntu 22.04, TLJH, Node.js 20.x, Python 3.10/3.12, 108+ SHM functions, Claude Code

### Configuration
Edit `jupyterhub/setup_jupyterhub_aws.sh`:
```bash
AWS_REGION="us-east-2"
INSTANCE_TYPE="t3.medium"  # ~$30/month
GITHUB_OWNER="your-username"
```

### Update Deployed Instance
**From your local machine** (easiest):
```bash
cd jupyterhub/
./remote_update.sh 3.130.148.209        # Update specific IP
./remote_update.sh                       # Auto-detect running instance
./remote_update.sh --verbose <IP>        # Verbose output
```

**Directly on server**:
```bash
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<IP>
cd /srv/classrepo
./jupyterhub/update_deployment.sh       # Full update workflow
```

### Debug & Management
```bash
# Debug existing instance
DEBUG_MODE=true EXISTING_INSTANCE_IP=1.2.3.4 ./setup_jupyterhub_aws.sh
./debug_install.sh <IP> shmtools   # Reinstall component

# User management
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<IP>
sudo tljh-config set users.allowed username

# Cost control
aws ec2 stop-instances --instance-ids <ID>    # Stop when unused
aws ec2 terminate-instances --instance-ids <ID> # Delete
```

## Architecture

**`shmtools/`**: Core library with `core/`, `features/`, `classification/`, `modal/`, `active_sensing/`, `hardware/`, `plotting/`, `utils/`

**`shm_function_selector/`**: JupyterLab extension (TypeScript frontend + Python backend)

**`examples/notebooks/`**: Educational notebooks by category (basic, intermediate, advanced, CBM, modal, active sensing, outlier detection)

## Development Principles

**Example-Driven**: Complete working notebooks, not just function stubs

**Quality Gates**: MATLAB analysis → Dependency mapping → Conversion with defensive programming → Synthetic validation → Integration testing → Notebook creation → Multi-context testing → HTML publication

## Data Management

**Required Datasets** (~161MB total, download to `examples/data/`):
- `data3SS.mat` (25MB) - 3-story structure, 8192×5×170
- `dataSensorDiagnostic.mat` (63KB) - Sensor health
- `data_CBM.mat` (54MB) - Condition monitoring
- `data_example_ActiveSense.mat` (32MB) - Active sensing  
- `data_OSPExampleModal.mat` (50KB) - Modal analysis

## Key Lessons Learned

1. **MATLAB Indexing**: Use explicit `matlab_k = k + 1` conversion
2. **Numerical Stability**: Handle zeros/singularities with defensive programming
3. **Execution Context**: Support multiple working directories and environments
4. **Testing**: Validate with synthetic data before real data

## MATLAB Conversion Rules

**Before ANY function**: Read complete `.m` file → Extract exact algorithm → Match signatures → Check dependencies

**Naming**: `psd_welch_shm()` format with `_shm` suffix

**Docstrings**: Machine-readable with GUI metadata (`.. meta::` and `.. gui::` blocks)

## Dependencies

**Core**: numpy, scipy, scikit-learn, matplotlib, pandas, bokeh >=2.4.0  
**Hardware**: nidaqmx, pyserial (optional)  
**Dev**: pytest, black, flake8, mypy

## Workflows

### Conversion Steps
1. Analyze MATLAB → Map dependencies → Convert with GUI metadata → Validate numerically → Create notebook → Test & publish

### Quick Commands
```bash
pytest -m "not hardware"                      # Skip hardware tests
jupyter nbconvert --to html notebook.ipynb    # Export notebook
```

## Success Criteria

✓ MATLAB parity ✓ GUI metadata ✓ Numerical tolerance ✓ Clean HTML export ✓ Educational content

## GitHub Issue Workflow

### Branch & Implementation
```bash
gh issue list                                      # Find issues
git checkout -b fix/issue-123-description          # Create branch
# Implement: Reproduce → Fix → Test → Validate
black shmtools/ && flake8 shmtools/               # Format & lint
pytest tests/test_shmtools/ -v                    # Test
```

### Commit & PR
```bash
git commit -m "Fix description

Resolves #123 
- Change details

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>"

git push -u origin fix/issue-123-description
gh pr create --title "Fix title" --body "Resolves #123..."
gh pr merge --squash --delete-branch
```

### Emergency
```bash
# Hotfix: git checkout -b hotfix/critical main → fix → gh pr create/merge
# Rollback: git revert COMMIT_HASH → gh pr create
```
