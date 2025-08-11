#!/usr/bin/env bash
# Local setup script for SHM Function Selector extension
# Applies the same fixes used on AWS instance

set -euo pipefail

echo "========================================="
echo "Setting up SHM Function Selector locally"
echo "========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Current directory: $(pwd)"

# Install the server extension in editable mode
echo "Installing server extension..."
pip install -e .

# Test if the module can be imported
echo "Testing module import..."
python -c "import shm_function_selector; print('✓ Server extension module imported successfully')"

# Create Jupyter config directory if it doesn't exist
JUPYTER_CONFIG_DIR=~/.jupyter
mkdir -p "$JUPYTER_CONFIG_DIR"

# Add server extension configuration
echo "Configuring server extension..."
cat > "$JUPYTER_CONFIG_DIR/jupyter_server_config.py" << 'EOF'
# SHM Function Selector server extension configuration
c.ServerApp.jpserver_extensions = {"shm_function_selector": True}
EOF

echo "✓ Server extension configuration added to ~/.jupyter/jupyter_server_config.py"

# Rebuild JupyterLab to ensure extension is properly integrated
echo "Rebuilding JupyterLab..."
jupyter lab build

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Restart any running Jupyter Lab instances"
echo "2. Start Jupyter Lab: jupyter lab"
echo "3. The SHM Function Selector should now work without 404 errors"
echo ""
echo "If you still see issues, try:"
echo "  jupyter server extension list"
echo "  jupyter labextension list | grep shm"