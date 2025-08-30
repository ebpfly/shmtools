#!/bin/bash
# Clean JupyterLab Extension Installation Script
# This script provides a clean, robust installation of the SHM Function Selector extension
# with all edge cases and common issues handled

set -euo pipefail

# Logging functions
log_step() {
    echo "[CLEAN-INSTALL $(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo "[CLEAN-ERROR $(date '+%Y-%m-%d %H:%M:%S')] ❌ $1"
}

log_success() {
    echo "[CLEAN-SUCCESS $(date '+%Y-%m-%d %H:%M:%S')] ✅ $1"
}

log_warning() {
    echo "[CLEAN-WARNING $(date '+%Y-%m-%d %H:%M:%S')] ⚠️ $1"
}

echo "========================================="
echo "🧩 CLEAN JUPYTERLAB EXTENSION INSTALLATION"
echo "========================================="

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root or with sudo"
    exit 1
fi

# Check if TLJH is installed
if [ ! -d "/opt/tljh" ]; then
    log_error "TLJH not found at /opt/tljh"
    exit 1
fi

# Check if repository exists
if [ ! -d "/srv/classrepo/shm_function_selector" ]; then
    log_error "Repository not found at /srv/classrepo/shm_function_selector"
    exit 1
fi

echo "========================================="
echo "STEP 1: FIXING PACKAGE CONFLICTS"
echo "========================================="

# Fix attrs package conflict that causes ImportError
log_step "🔧 Upgrading attrs package to fix conflicts..."
pip3 install --upgrade attrs 2>&1 | sed 's/^/[ATTRS-SYS] /'
/opt/tljh/user/bin/pip install --upgrade attrs 2>&1 | sed 's/^/[ATTRS-USER] /'
log_success "attrs package upgraded"

echo "========================================="
echo "STEP 2: CLEANING PREVIOUS INSTALLATIONS"
echo "========================================="

log_step "🧹 Cleaning previous installation attempts..."

# Remove node_modules to ensure clean npm install
if [ -d "/srv/classrepo/shm_function_selector/node_modules" ]; then
    log_step "Removing old node_modules..."
    rm -rf /srv/classrepo/shm_function_selector/node_modules
fi

# Remove old labextension files
if [ -d "/opt/tljh/user/share/jupyter/labextensions/shm-function-selector" ]; then
    log_step "Removing old labextension files..."
    rm -rf /opt/tljh/user/share/jupyter/labextensions/shm-function-selector
fi

# Uninstall any existing pip installation
log_step "Uninstalling any existing pip installation..."
/opt/tljh/user/bin/pip uninstall -y shm_function_selector 2>/dev/null || true

log_success "Previous installations cleaned"

echo "========================================="
echo "STEP 3: SETTING PROPER OWNERSHIP"
echo "========================================="

log_step "📂 Setting ownership to ubuntu user..."
chown -R ubuntu:ubuntu /srv/classrepo
log_success "Ownership set"

echo "========================================="
echo "STEP 4: BUILDING EXTENSION"
echo "========================================="

cd /srv/classrepo/shm_function_selector

# Check Node.js and npm
log_step "🔍 Checking Node.js environment..."
NODE_VERSION=$(node --version 2>/dev/null || echo "not found")
NPM_VERSION=$(npm --version 2>/dev/null || echo "not found")
log_step "Node.js: $NODE_VERSION, npm: $NPM_VERSION"

if [ "$NODE_VERSION" = "not found" ]; then
    log_error "Node.js is not installed"
    exit 1
fi

# Install dependencies with sudo
log_step "📦 Installing npm dependencies..."
if ! sudo npm install 2>&1 | sed 's/^/[NPM-INSTALL] /'; then
    log_error "npm install failed"
    exit 1
fi
log_success "Dependencies installed"

# Build the extension
log_step "🔨 Building TypeScript library..."
if ! sudo npm run build:lib 2>&1 | sed 's/^/[BUILD-LIB] /'; then
    log_error "TypeScript build failed"
    exit 1
fi
log_success "TypeScript library built"

log_step "🔨 Building JupyterLab extension..."
if ! sudo npm run build:labextension:dev 2>&1 | sed 's/^/[BUILD-EXT] /'; then
    log_error "Extension build failed"
    exit 1
fi
log_success "JupyterLab extension built"

echo "========================================="
echo "STEP 5: CREATING REQUIRED DIRECTORIES"
echo "========================================="

log_step "📁 Creating extension directories..."
EXT_DIR="/opt/tljh/user/share/jupyter/labextensions/shm-function-selector"
mkdir -p "$EXT_DIR"

# Create the build_log.json file that pip expects
log_step "📄 Creating build_log.json..."
touch "$EXT_DIR/build_log.json"

# Set proper ownership for the extension directory
chown -R root:root "$EXT_DIR"
log_success "Extension directories created"

echo "========================================="
echo "STEP 6: INSTALLING EXTENSION"
echo "========================================="

cd /srv/classrepo

# Install the extension in editable mode
log_step "📦 Installing extension in TLJH user environment..."
if ! /opt/tljh/user/bin/pip install -e shm_function_selector 2>&1 | sed 's/^/[PIP-INSTALL] /'; then
    log_error "pip install failed"
    exit 1
fi
log_success "Extension installed via pip"

echo "========================================="
echo "STEP 7: CONFIGURING SERVER EXTENSION"
echo "========================================="

log_step "⚙️ Configuring server extension..."

# Create config directory if it doesn't exist
CONFIG_DIR="/opt/tljh/user/etc/jupyter/jupyter_server_config.d"
mkdir -p "$CONFIG_DIR"

# Create server extension configuration
CONFIG_FILE="$CONFIG_DIR/shm_function_selector.json"
cat > "$CONFIG_FILE" << 'EOF'
{
  "ServerApp": {
    "jpserver_extensions": {
      "shm_function_selector": true
    }
  }
}
EOF

log_success "Server extension configured"

# Add Python path if needed
PYCONFIG_FILE="/opt/tljh/user/etc/jupyter/jupyter_server_config.py"
if ! grep -q "/srv/classrepo" "$PYCONFIG_FILE" 2>/dev/null; then
    log_step "📝 Adding Python path to server config..."
    echo "" >> "$PYCONFIG_FILE"
    echo "# Add SHMTools to Python path" >> "$PYCONFIG_FILE"
    echo "import sys" >> "$PYCONFIG_FILE"
    echo "sys.path.insert(0, '/srv/classrepo')" >> "$PYCONFIG_FILE"
    log_success "Python path added"
fi

echo "========================================="
echo "STEP 8: REBUILDING JUPYTERLAB"
echo "========================================="

log_step "🔨 Rebuilding JupyterLab..."
if /opt/tljh/user/bin/jupyter lab build 2>&1 | sed 's/^/[LAB-BUILD] /'; then
    log_success "JupyterLab rebuilt successfully"
else
    log_warning "JupyterLab rebuild had issues, but extension may still work"
fi

echo "========================================="
echo "STEP 9: VERIFICATION"
echo "========================================="

log_step "🔍 Verifying installation..."

# Check labextension
echo "📋 Installed JupyterLab extensions:"
/opt/tljh/user/bin/jupyter labextension list 2>&1 | sed 's/^/[LABEXT] /'

if /opt/tljh/user/bin/jupyter labextension list 2>/dev/null | grep -q "shm-function-selector"; then
    log_success "JupyterLab extension found"
else
    log_warning "JupyterLab extension not found in list"
fi

# Check server extension
echo "📋 Installed server extensions:"
/opt/tljh/user/bin/jupyter server extension list 2>&1 | sed 's/^/[SERVEREXT] /'

if /opt/tljh/user/bin/jupyter server extension list 2>/dev/null | grep -q "shm_function_selector"; then
    log_success "Server extension found"
else
    log_warning "Server extension not found in list"
fi

echo "========================================="
echo "STEP 10: RESTARTING SERVICES"
echo "========================================="

log_step "🔄 Restarting JupyterHub..."
systemctl restart jupyterhub
sleep 5

echo "========================================="
echo "✨ CLEAN INSTALLATION COMPLETE"
echo "========================================="
log_success "Extension installation completed"
log_step "Please test the extension in JupyterLab"