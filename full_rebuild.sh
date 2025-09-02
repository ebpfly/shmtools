#!/bin/bash

echo "Full rebuild of SHM JupyterLab extension for JupyterHub..."
echo "================================================"

# Navigate to extension directory
cd /srv/classrepo/shm_function_selector

# Step 1: Build TypeScript library
echo "Step 1/5: Building TypeScript library..."
sudo -E npm run build:lib

# Step 2: Build lab extension
echo "Step 2/5: Building lab extension..."
sudo -E npm run build:labextension:dev

# Step 3: Install extension in development mode
echo "Step 3/5: Installing extension..."
cd /srv/classrepo
sudo -E pip install -e ./shm_function_selector

# Step 4: Clean old JupyterLab build
echo "Step 4/5: Cleaning JupyterLab cache..."
sudo jupyter lab clean --all

# Step 5: Rebuild JupyterLab
echo "Step 5/5: Rebuilding JupyterLab..."
sudo -E jupyter lab build --minimize=False

echo "================================================"
echo "Build complete!"
echo ""
echo "IMPORTANT: Users need to:"
echo "1. Hard refresh their browser (Ctrl+Shift+R or Cmd+Shift+R)"
echo "2. Or clear browser cache and reload"
echo ""
echo "The markdown cells should now appear when inserting functions."