#!/bin/bash

echo "Starting JupyterLab extension rebuild for JupyterHub..."
echo "This process may take several minutes..."

# Navigate to extension directory
cd /srv/classrepo/shm_function_selector

# Build the TypeScript library
echo "Step 1/3: Building TypeScript library..."
sudo -E npm run build:lib

# Build the lab extension
echo "Step 2/3: Building lab extension..."
sudo -E npm run build:labextension:dev

# Rebuild JupyterLab
echo "Step 3/3: Rebuilding JupyterLab..."
cd /srv/classrepo
sudo -E jupyter lab build

echo "Extension rebuild complete!"
echo "Users may need to refresh their browser to see updates."