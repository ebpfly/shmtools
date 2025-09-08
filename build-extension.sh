#!/bin/bash
# build-extension.sh - Automated JupyterLab extension builder for SHMTools
# Usage: ./build-extension.sh

set -e  # Exit on any error

echo "🔧 Building SHMTools JupyterLab Extension..."

# Check if we're in the right directory
if [ ! -d "shm_function_selector" ]; then
    echo "❌ Error: shm_function_selector directory not found"
    echo "   Please run this script from the shmtools repository root"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Please activate your virtual environment first:"
    echo "   source shmtools-venv/bin/activate"
    exit 1
fi

echo "📦 Installing Node.js dependencies..."
cd shm_function_selector

# Fix npm cache issues (common EACCES error)
echo "🔧 Fixing npm cache permissions..."
npm cache clear --force 2>/dev/null || true
mkdir -p ~/.npm && chown -R $USER ~/.npm 2>/dev/null || true

# Install with temporary cache to avoid permission issues
npm install --cache /tmp/npm-cache-$(whoami)-$$ || npm install

echo "🔨 Compiling TypeScript..."
npm run build:lib

echo "📱 Building JupyterLab extension..."
npm run build:labextension:dev

echo "📦 Installing extension package..."
cd ..
pip install -e shm_function_selector/

echo "🚀 Building JupyterLab..."
jupyter lab build

echo "✅ Extension build complete!"
echo ""
echo "📋 Verifying installation..."
if jupyter labextension list | grep -q "shm-function-selector"; then
    echo "✅ Extension successfully installed:"
    jupyter labextension list | grep "shm-function-selector"
    echo ""
    echo "🎉 Ready to use! Launch JupyterLab with: jupyter lab"
    echo "   Look for the 🔍 SHM Functions panel in the left sidebar"
else
    echo "❌ Extension not found in JupyterLab"
    echo "   Try running: jupyter lab build"
fi