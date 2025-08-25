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

# Update requirements in TLJH user environment only (this is what matters)
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

# Reinstall in TLJH user environment (development mode) - this is the main environment
log "Installing shmtools in TLJH user environment (development mode)..."
if ! sudo /opt/tljh/user/bin/pip install --quiet -e .; then
    error "Failed to install shmtools in TLJH user environment"
    exit 1
fi

success "✅ SHMTools package reinstalled successfully"

# Step 4: Update JupyterLab extension
log "🔌 Updating JupyterLab extension..."
echo "====================================================================="
echo "                   JUPYTERLAB EXTENSION UPDATE"
echo "====================================================================="

cd "$REPO_DIR/shm_function_selector"
log "📂 Working directory: $(pwd)"

# Ensure proper permissions
log "🔧 Setting proper permissions..."
sudo chown -R ubuntu:ubuntu . 2>&1 | sed 's/^/[EXTENSION-PERMISSIONS] /'

# Install/update npm dependencies
log "📦 Installing/updating npm dependencies..."
if ! npm install 2>&1 | sed 's/^/[NPM-INSTALL] /'; then
    error "Failed to install npm dependencies"
    log "NPM error log (if available):"
    cat npm-debug.log 2>/dev/null | sed 's/^/[NPM-DEBUG] /' || log "No npm debug log found"
    exit 1
fi

# Build the extension
log "🔨 Building TypeScript library..."
if ! npm run build:lib 2>&1 | sed 's/^/[BUILD-LIB] /'; then
    error "Failed to build TypeScript library"
    log "Build output (last 20 lines):"
    tail -20 build.log 2>/dev/null | sed 's/^/[BUILD-ERROR] /' || log "No build log found"
    exit 1
fi

log "🔨 Building JupyterLab extension..."
if ! npm run build:labextension:dev 2>&1 | sed 's/^/[BUILD-EXTENSION] /'; then
    error "Failed to build JupyterLab extension"
    log "Extension build output (if available):"
    find . -name "*.log" -exec tail -10 {} \; 2>/dev/null | sed 's/^/[EXT-BUILD-ERROR] /' || log "No extension build logs found"
    exit 1
fi

# Fix import paths in handlers.py for relocated introspection module
log "🔧 Fixing import paths in extension..."
HANDLER_FILE="$REPO_DIR/shm_function_selector/shm_function_selector/handlers.py"
if [ -f "$HANDLER_FILE" ]; then
    log "📝 Updating import paths in handlers.py..."
    sed -i 's/from shm_function_selector\.introspection/from ..introspection/g' "$HANDLER_FILE" 2>&1 | sed 's/^/[IMPORT-FIX] /' || true
    log "✅ Import paths updated"
else
    warning "handlers.py not found at expected location: $HANDLER_FILE"
fi

# Reinstall server extension
log "🔄 Reinstalling server extension..."
# First create any missing directories/files that pip might expect
log "📁 Creating required directories..."
sudo mkdir -p /opt/tljh/user/share/jupyter/labextensions/shm-function-selector 2>&1 | sed 's/^/[MKDIR] /'
sudo touch /opt/tljh/user/share/jupyter/labextensions/shm-function-selector/build_log.json 2>&1 | sed 's/^/[TOUCH] /' || true

log "📦 Installing server extension via pip..."
if ! sudo /opt/tljh/user/bin/pip install --quiet -e . --no-deps 2>&1 | sed 's/^/[PIP-SERVER-EXT] /'; then
    warning "Failed to reinstall server extension via pip, trying alternative method"
    log "Pip error details:"
    sudo /opt/tljh/user/bin/pip install -e . --no-deps 2>&1 | sed 's/^/[PIP-ERROR] /' || true
fi

# Install labextension properly
log "🧩 Installing JupyterLab extension..."
if ! sudo /opt/tljh/user/bin/jupyter labextension develop . --overwrite 2>&1 | sed 's/^/[LABEXT-DEVELOP] /'; then
    warning "Failed to install labextension (may already be installed)"
    log "Extension development installation output:"
    sudo /opt/tljh/user/bin/jupyter labextension develop . --overwrite --debug 2>&1 | sed 's/^/[LABEXT-DEBUG] /' || true
fi

# Add custom server extension loader to config
log "⚙️ Configuring server extension..."
CONFIG_FILE="/opt/tljh/user/etc/jupyter/jupyter_server_config.py"
log "📝 Server config file: $CONFIG_FILE"

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
        import traceback
        server_app.log.error(f"Extension traceback: {traceback.format_exc()}")

c.ServerApp.callable_extensions = [load_shm_extension]
EOF

log "🔍 Checking if server extension config already exists..."
if ! grep -q "load_shm_extension" "$CONFIG_FILE" 2>/dev/null; then
    log "📝 Adding server extension configuration..."
    cat /tmp/shm_extension_config.py | sudo tee -a "$CONFIG_FILE" > /dev/null 2>&1 | sed 's/^/[CONFIG-APPEND] /'
    log "✅ Server extension configuration added"
else
    log "✅ Server extension configuration already present"
fi

log "🔍 Displaying current server config (last 20 lines):"
tail -20 "$CONFIG_FILE" 2>/dev/null | sed 's/^/[SERVER-CONFIG] /' || log "Could not read server config"

rm -f /tmp/shm_extension_config.py

# Rebuild JupyterLab
log "🔨 Rebuilding JupyterLab with updated extension..."
log "⚠️ This may take several minutes - JupyterLab build can be slow"
echo "====================================================================="
echo "                        JUPYTERLAB BUILD"
echo "====================================================================="
START_TIME=$(date +%s)
if ! timeout 300 sudo /opt/tljh/user/bin/jupyter lab build --dev-build=False 2>&1 | sed 's/^/[LAB-BUILD] /'; then
    END_TIME=$(date +%s)
    BUILD_DURATION=$((END_TIME - START_TIME))
    error "Failed to rebuild JupyterLab (timed out after 5 minutes, actual duration: ${BUILD_DURATION}s)"
    log "Checking JupyterLab build logs..."
    find /opt/tljh/user -name "*.log" -path "*jupyter*" -exec tail -20 {} \; 2>/dev/null | sed 's/^/[BUILD-LOG] /' || log "No build logs found"
    exit 1
fi
END_TIME=$(date +%s)
BUILD_DURATION=$((END_TIME - START_TIME))
log "✅ JupyterLab build completed successfully in ${BUILD_DURATION} seconds"

success "✅ JupyterLab extension updated successfully"

# Step 5: Restart services
log "🔄 Restarting JupyterHub service..."
echo "====================================================================="
echo "                      SERVICE RESTART"
echo "====================================================================="

log "⏹️ Stopping JupyterHub service first..."
sudo systemctl stop jupyterhub 2>&1 | sed 's/^/[SERVICE-STOP] /'
sleep 2

log "▶️ Starting JupyterHub service..."
if ! sudo systemctl start jupyterhub 2>&1 | sed 's/^/[SERVICE-START] /'; then
    error "Failed to start JupyterHub service"
    log "Service status:"
    sudo systemctl status jupyterhub 2>&1 | sed 's/^/[SERVICE-STATUS] /'
    log "Recent service logs:"
    sudo journalctl -u jupyterhub --no-pager -l -n 50 2>&1 | sed 's/^/[SERVICE-LOG] /'
    exit 1
fi

# Wait for service to start
log "⏳ Waiting for JupyterHub to start (15 seconds)..."
sleep 15

# Check if service is running
log "🔍 Checking service status..."
if sudo systemctl is-active --quiet jupyterhub; then
    success "✅ JupyterHub service restarted successfully"
    log "Service status details:"
    sudo systemctl status jupyterhub --no-pager -l 2>&1 | sed 's/^/[SERVICE-STATUS] /'
else
    error "JupyterHub service failed to start properly"
    log "Service status:"
    sudo systemctl status jupyterhub 2>&1 | sed 's/^/[SERVICE-STATUS] /'
    log "Recent service logs:"
    sudo journalctl -u jupyterhub --no-pager -l -n 50 2>&1 | sed 's/^/[SERVICE-LOG] /'
    log "Check logs with: sudo journalctl -u jupyterhub -f"
    exit 1
fi

# Step 6: Verification
log "🔍 Verifying installation..."
echo "====================================================================="
echo "                       VERIFICATION"
echo "====================================================================="

# Check shmtools installation
log "📦 Checking SHMTools installation..."
SHMTOOLS_VERSION=$(sudo /opt/tljh/user/bin/python -c "import shmtools; print(shmtools.__version__)" 2>&1 | sed 's/^/[SHMTOOLS-CHECK] /' || echo "unknown")
log "SHMTools version: $SHMTOOLS_VERSION"

log "📋 Verifying SHMTools import paths..."
sudo /opt/tljh/user/bin/python -c "
try:
    import shmtools
    print(f'SHMTools path: {shmtools.__file__}')
    print(f'SHMTools version: {shmtools.__version__}')
except Exception as e:
    print(f'SHMTools import error: {e}')
" 2>&1 | sed 's/^/[SHMTOOLS-IMPORT] /'

# Check extension installation
log "🧩 Checking JupyterLab extension installation..."
log "Server extensions:"
sudo /opt/tljh/user/bin/jupyter server extension list 2>&1 | sed 's/^/[SERVER-EXT-LIST] /'
log "Lab extensions:"
sudo /opt/tljh/user/bin/jupyter labextension list 2>&1 | sed 's/^/[LAB-EXT-LIST] /'

EXTENSION_STATUS=$(sudo /opt/tljh/user/bin/jupyter labextension list 2>/dev/null | grep "shm-function-selector" || echo "not found")
if [[ "$EXTENSION_STATUS" == "not found" ]]; then
    warning "JupyterLab extension not detected in extension list"
    log "🔍 Searching for extension files..."
    find /opt/tljh/user -name "*shm*" -type f 2>/dev/null | head -10 | sed 's/^/[EXT-FILES] /'
else
    log "✅ JupyterLab extension status: $EXTENSION_STATUS"
fi

# Test web accessibility
log "🌐 Testing web accessibility..."
HTTP_CODE=$(timeout 10 curl -s -o /dev/null -w "%{http_code}" http://localhost 2>&1 || echo "timeout")
log "HTTP response code: $HTTP_CODE"
if [[ "$HTTP_CODE" == "200" ]]; then
    success "✅ JupyterHub web interface is accessible"
elif [[ "$HTTP_CODE" == "timeout" ]]; then
    warning "Web interface test timed out - service may still be starting"
else
    warning "JupyterHub web interface returned HTTP $HTTP_CODE - may not be fully ready yet"
fi

# Check process status
log "🔍 Checking process status..."
ps aux | grep -E '(jupyter|hub)' | grep -v grep | sed 's/^/[PROCESSES] /' || log "No JupyterHub processes found"

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