#!/bin/bash
# Fix script to properly install SHM extension for all TLJH users
# Run this on the EC2 instance as ubuntu user

set -euxo pipefail

echo "========================================="
echo "Fixing SHM Extension for Non-Admin Users"
echo "========================================="

# Navigate to repository
cd /srv/classrepo

# Rebuild the extension first
echo "Rebuilding extension..."
cd shm_function_selector
npm run build:lib
npm run build:labextension:dev
cd ..

# Install server extension properly in TLJH user environment
echo "Installing server extension in TLJH user environment..."
# First install the extension package in user environment
sudo -E /opt/tljh/user/bin/pip install ./shm_function_selector/

# Copy config.json to ensure it's available
echo "Copying config.json..."
sudo cp shm_function_selector/config.json /opt/tljh/user/lib/python3.12/site-packages/

# Enable the server extension in user environment
echo "Enabling server extension..."
sudo -E /opt/tljh/user/bin/jupyter server extension enable shm_function_selector

# Install the labextension (frontend) in user environment
echo "Installing frontend extension..."
sudo -E /opt/tljh/user/bin/jupyter labextension develop --overwrite shm_function_selector/

# Create proper configuration directories
echo "Configuring server extension..."
sudo mkdir -p /opt/tljh/user/etc/jupyter
sudo mkdir -p /opt/tljh/user/share/jupyter/jupyter_server_config.d

# Create JSON config file for server extension
sudo tee /opt/tljh/user/share/jupyter/jupyter_server_config.d/shm_function_selector.json << 'EOF'
{
  "ServerApp": {
    "jpserver_extensions": {
      "shm_function_selector": true
    }
  }
}
EOF

# Also create Python config as backup
sudo tee /opt/tljh/user/etc/jupyter/jupyter_server_config.py << 'EOF'
# Server extension configuration
c.ServerApp.jpserver_extensions = {
    'shm_function_selector': True
}
# Allow the extension to run for all users
c.ServerApp.allow_origin = '*'
EOF

# Rebuild JupyterLab in user environment
echo "Rebuilding JupyterLab..."
sudo -E /opt/tljh/user/bin/jupyter lab build

# Restart JupyterHub to apply changes
echo "Restarting JupyterHub..."
sudo systemctl restart jupyterhub

echo "========================================="
echo "Fix Applied Successfully!"
echo "========================================="
echo ""
echo "The extension should now work for all users."
echo "Please have non-admin users:"
echo "1. Log out of JupyterHub"
echo "2. Log back in"
echo "3. Refresh their browser (Ctrl+F5)"
echo ""
echo "To verify the fix worked, run:"
echo "sudo -E /opt/tljh/user/bin/jupyter server extension list | grep shm"