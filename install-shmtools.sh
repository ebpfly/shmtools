#!/bin/bash
# install-shmtools.sh - Complete SHMTools installation script
# Usage: ./install-shmtools.sh

set -e  # Exit on any error

echo "🚀 SHMTools Complete Installation Script"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "shmtools" ]; then
    echo "❌ Error: This doesn't appear to be the SHMTools repository root"
    echo "   Please run this script from the shmtools directory"
    exit 1
fi

# Step 1: Create virtual environment
echo ""
echo "1️⃣ Creating virtual environment..."
if [ ! -d "shmtools-venv" ]; then
    python3 -m venv shmtools-venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Step 2: Activate virtual environment
echo ""
echo "2️⃣ Activating virtual environment..."
source shmtools-venv/bin/activate
echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Step 3: Install Python dependencies
echo ""
echo "3️⃣ Installing Python dependencies and SHMTools..."
pip install --upgrade pip
pip install -e .
echo "✅ Python installation complete"

# Step 4: Install Jupyter kernel
echo ""
echo "4️⃣ Installing Jupyter kernel..."
python -m ipykernel install --user --name=shmtools-venv --display-name="SHMTools Python"
echo "✅ Jupyter kernel installed"

# Step 5: Create _version.py if missing (common issue)
echo ""
echo "5️⃣ Ensuring extension files are present..."
python shm_function_selector/ensure_version.py

# Step 6: Build JupyterLab extension
echo ""
echo "6️⃣ Building JupyterLab extension..."
./build-extension.sh

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "🚀 To start using SHMTools:"
echo "   1. Activate environment: source shmtools-venv/bin/activate"
echo "   2. Launch JupyterLab:    jupyter lab"
echo "   3. Create new notebook with 'SHMTools Python' kernel"
echo "   4. Look for 🔍 SHM Functions panel in left sidebar"
echo ""
echo "📚 For examples and documentation, see:"
echo "   - examples/notebooks/ directory"
echo "   - Published notebooks at: https://shmtools.readthedocs.io"
echo ""