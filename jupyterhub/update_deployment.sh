#!/bin/bash

# update_deployment.sh - Update SHM deployment with latest changes
# This script pulls the latest code and reinitializes the SHMTools deployment
# Usage: ./update_deployment.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S')${NC} - $1"
}

error() {
    echo -e "${RED}$(date '+%Y-%m-%d %H:%M:%S')${NC} - ERROR: $1" >&2
}

success() {
    echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S')${NC} - $1"
}

warning() {
    echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S')${NC} - WARNING: $1"
}

# Check if running as the correct user
if [[ $EUID -eq 0 ]]; then
    error "Do not run this script as root. Run as ubuntu user instead."
    exit 1
fi

# Change to repository directory
REPO_DIR="/srv/classrepo"
if [[ ! -d "$REPO_DIR" ]]; then
    error "Repository directory $REPO_DIR not found!"
    exit 1
fi

cd "$REPO_DIR"

log "🔄 Starting SHM deployment update..."
echo "======================================================================"
echo "                    SHM DEPLOYMENT UPDATE SCRIPT"
echo "======================================================================"

# Step 1: Git operations
log "📥 Pulling latest changes from Git repository..."
if ! git fetch origin; then
    error "Failed to fetch from origin"
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
log "Current branch: $CURRENT_BRANCH"

if ! git pull origin "$CURRENT_BRANCH"; then
    error "Failed to pull latest changes"
    exit 1
fi

success "✅ Git repository updated successfully"

# Step 2: Update Python dependencies
log "📦 Updating Python dependencies..."

# Update requirements in both Python environments
log "Installing requirements.txt in local Python..."
if ! /usr/local/bin/python3 -m pip install --quiet --upgrade -r requirements.txt; then
    warning "Failed to update local Python requirements"
fi

log "Installing requirements.txt in TLJH user environment..."
if ! sudo /opt/tljh/user/bin/pip install --quiet --upgrade -r requirements.txt; then
    warning "Failed to update TLJH user environment requirements"
fi

log "Installing requirements-dev.txt in TLJH user environment..."
if ! sudo /opt/tljh/user/bin/pip install --quiet --upgrade -r requirements-dev.txt; then
    warning "Failed to update TLJH development requirements"
fi

success "✅ Python dependencies updated"

# Step 3: Reinstall shmtools package
log "🔧 Reinstalling shmtools package..."

# Reinstall in local environment (development mode)
log "Installing shmtools in local Python (development mode)..."
if ! /usr/local/bin/python3 -m pip install --quiet -e .; then
    error "Failed to install shmtools in local Python"
    exit 1
fi

# Reinstall in TLJH user environment (development mode)
log "Installing shmtools in TLJH user environment (development mode)..."
if ! sudo /opt/tljh/user/bin/pip install --quiet -e .; then
    error "Failed to install shmtools in TLJH user environment"
    exit 1
fi

success "✅ SHMTools package reinstalled successfully"

# Step 4: Update JupyterLab extension
log "🔌 Updating JupyterLab extension..."

cd "$REPO_DIR/shm_function_selector"

# Ensure proper permissions
sudo chown -R ubuntu:ubuntu .

# Install/update npm dependencies
log "Installing/updating npm dependencies..."
if ! npm install --silent; then
    error "Failed to install npm dependencies"
    exit 1
fi

# Build the extension
log "Building JupyterLab extension..."
if ! npm run build:lib; then
    error "Failed to build TypeScript library"
    exit 1
fi

if ! npm run build:labextension:dev; then
    error "Failed to build JupyterLab extension"
    exit 1
fi

# Reinstall server extension
log "Reinstalling server extension..."
if ! sudo /opt/tljh/user/bin/pip install --quiet -e .; then
    error "Failed to reinstall server extension"
    exit 1
fi

# Enable server extension
log "Enabling server extension..."
if ! sudo /opt/tljh/user/bin/jupyter server extension enable shm_function_selector --sys-prefix; then
    warning "Failed to enable server extension (may already be enabled)"
fi

# Rebuild JupyterLab
log "Rebuilding JupyterLab with updated extension..."
if ! timeout 300 sudo /opt/tljh/user/bin/jupyter lab build --dev-build=False; then
    error "Failed to rebuild JupyterLab (timed out after 5 minutes)"
    exit 1
fi

success "✅ JupyterLab extension updated successfully"

# Step 5: Restart services
log "🔄 Restarting JupyterHub service..."
if ! sudo systemctl restart jupyterhub; then
    error "Failed to restart JupyterHub service"
    exit 1
fi

# Wait for service to start
log "Waiting for JupyterHub to start..."
sleep 5

# Check if service is running
if sudo systemctl is-active --quiet jupyterhub; then
    success "✅ JupyterHub service restarted successfully"
else
    error "JupyterHub service failed to start properly"
    log "Check logs with: sudo journalctl -u jupyterhub -f"
    exit 1
fi

# Step 6: Verification
log "🔍 Verifying installation..."

# Check shmtools installation
SHMTOOLS_VERSION=$(/opt/tljh/user/bin/python -c "import shmtools; print(shmtools.__version__)" 2>/dev/null || echo "unknown")
log "SHMTools version: $SHMTOOLS_VERSION"

# Check extension installation
EXTENSION_STATUS=$(/opt/tljh/user/bin/jupyter labextension list 2>/dev/null | grep "shm-function-selector" || echo "not found")
if [[ "$EXTENSION_STATUS" == "not found" ]]; then
    warning "JupyterLab extension not detected in list"
else
    log "JupyterLab extension status: $EXTENSION_STATUS"
fi

# Test web accessibility
log "Testing web accessibility..."
if timeout 10 curl -s -o /dev/null -w "%{http_code}" http://localhost >/dev/null; then
    success "✅ JupyterHub web interface is accessible"
else
    warning "JupyterHub web interface may not be fully ready yet"
fi

cd "$REPO_DIR"

echo "======================================================================"
success "🎉 SHM DEPLOYMENT UPDATE COMPLETED SUCCESSFULLY!"
echo "======================================================================"
log "Summary of changes:"
log "  • Git repository updated to latest version"
log "  • Python dependencies updated"
log "  • SHMTools package reinstalled (v$SHMTOOLS_VERSION)"
log "  • JupyterLab extension rebuilt and updated"
log "  • JupyterHub service restarted"
echo ""
log "🌐 JupyterHub should be accessible at:"
EXTERNAL_IP=$(curl -s http://checkip.amazonaws.com/ || echo "unknown")
if [[ "$EXTERNAL_IP" != "unknown" ]]; then
    log "   External: http://$EXTERNAL_IP"
fi
log "   Internal: http://localhost"
echo ""
log "📋 Useful commands:"
log "   • Check JupyterHub status: sudo systemctl status jupyterhub"
log "   • View JupyterHub logs: sudo journalctl -u jupyterhub -f"
log "   • List extensions: /opt/tljh/user/bin/jupyter labextension list"
echo "======================================================================"