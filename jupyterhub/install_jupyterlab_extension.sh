#!/bin/bash
set -e

# JupyterLab Extension Installation Script
# This script handles the installation of the SHM Function Selector JupyterLab extension

echo "========================================="
echo "Installing SHM JupyterLab Extension"
echo "========================================="

# Check if we're in the right directory
if [ ! -d "/srv/classrepo/shm_function_selector" ]; then
    echo "Error: /srv/classrepo/shm_function_selector not found!"
    echo "Please ensure the repository is cloned to /srv/classrepo"
    exit 1
fi

cd /srv/classrepo

echo "Installing JupyterLab extension using proper build process..."

# Set error handling for extension installation - don't let it stop the entire script
set +e  # Don't exit on errors in this section

# Install in development mode in both environments to ensure all files are available
echo "Installing extension in system environment..."
sudo -E pip3 install -e ./shm_function_selector/

echo "Installing extension in TLJH user environment..."
sudo -E /opt/tljh/user/bin/pip install -e ./shm_function_selector/

# The -e install should trigger the build hooks and create the labextension
echo "Checking if labextension was built..."
if [ ! -d "./shm_function_selector/shm_function_selector/labextension" ]; then
    echo "Labextension not built automatically, building manually..."
    cd /srv/classrepo/shm_function_selector
    
    # Install Node.js dependencies using jlpm (JupyterLab's package manager)
    echo "Installing dependencies with jlpm..."
    sudo -E /opt/tljh/user/bin/jlpm install || npm install
    
    # Build using jlpm commands
    echo "Building extension with jlpm..."
    sudo -E /opt/tljh/user/bin/jlpm run build:prod || (npm run build:lib && npm run build:labextension)
    
    cd /srv/classrepo
fi

# Fix import paths in handlers.py for relocated introspection module
echo "Fixing import paths in extension..."
sed -i 's/from shm_function_selector\.introspection/from ..introspection/g' /srv/classrepo/shm_function_selector/shm_function_selector/handlers.py 2>/dev/null || true

# Create missing directories/files that pip might expect
sudo mkdir -p /opt/tljh/user/share/jupyter/labextensions/shm-function-selector
sudo touch /opt/tljh/user/share/jupyter/labextensions/shm-function-selector/build_log.json 2>/dev/null || true

# The labextension should already be installed via pip editable install
echo "Verifying labextension installation..."
if sudo -E /opt/tljh/user/bin/jupyter labextension list | grep -q "shm-function-selector"; then
    echo "✓ Labextension already installed"
else
    echo "Installing labextension manually..."
    # Use develop mode for development installs
    sudo -E /opt/tljh/user/bin/jupyter labextension develop ./shm_function_selector/ --overwrite
fi

# Add custom server extension loader to config
echo "Configuring server extension..."
cat > /tmp/shm_extension_config.py << 'EOF'

# Load SHM Function Selector extension
import sys
sys.path.insert(0, '/srv/classrepo')

def load_shm_extension(server_app):
    try:
        from shm_function_selector.shm_function_selector.handlers import setup_handlers
        web_app = server_app.web_app
        setup_handlers(web_app)
        server_app.log.info("SHM Function Selector server extension loaded via custom config")
    except Exception as e:
        server_app.log.error(f"Failed to load SHM extension: {e}")

c.ServerApp.callable_extensions = [load_shm_extension]
EOF

# Append to config if not already present
if ! grep -q "load_shm_extension" /opt/tljh/user/etc/jupyter/jupyter_server_config.py 2>/dev/null; then
    cat /tmp/shm_extension_config.py | sudo tee -a /opt/tljh/user/etc/jupyter/jupyter_server_config.py > /dev/null
    echo "Server extension configuration added"
else
    echo "Server extension configuration already present"
fi

rm -f /tmp/shm_extension_config.py

# Rebuild JupyterLab to integrate the extension
echo "Building JupyterLab with extension..."
if sudo -E /opt/tljh/user/bin/jupyter lab build; then
    echo "✓ JupyterLab build successful"
else
    echo "⚠️  JupyterLab build failed, but proceeding..."
fi

# Verify extension installation
echo "Verifying extension installation..."
echo "Server extensions:"
sudo -E /opt/tljh/user/bin/jupyter server extension list | grep -i shm || echo "Server extension not found"
echo "Lab extensions:"
sudo -E /opt/tljh/user/bin/jupyter labextension list | grep -i shm || echo "Lab extension not found"

echo "JupyterLab extension installation complete!"

# Resume strict error handling
set -e

echo "========================================="
echo "Extension installation finished!"
echo "========================================="