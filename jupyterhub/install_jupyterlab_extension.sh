#!/bin/bash
set -e

# JupyterLab Extension Installation Script
# This script handles the installation of the SHM Function Selector JupyterLab extension

# Enhanced logging functions
log_step() {
    echo "[EXT-INSTALL $(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo "[EXT-ERROR $(date '+%Y-%m-%d %H:%M:%S')] ❌ $1"
}

log_success() {
    echo "[EXT-SUCCESS $(date '+%Y-%m-%d %H:%M:%S')] ✅ $1"
}

log_warning() {
    echo "[EXT-WARNING $(date '+%Y-%m-%d %H:%M:%S')] ⚠️ $1"
}

echo "========================================="
log_step "🧩 Installing SHM JupyterLab Extension"
echo "========================================="

# Check if we're in the right directory
log_step "🔍 Verifying repository structure..."
if [ ! -d "/srv/classrepo/shm_function_selector" ]; then
    log_error "Extension directory /srv/classrepo/shm_function_selector not found!"
    log_step "Available directories in /srv/classrepo:"
    ls -la /srv/classrepo 2>&1 | sed 's/^/[DIR-LIST] /' || log_error "Could not list /srv/classrepo"
    log_error "Please ensure the repository is cloned to /srv/classrepo"
    exit 1
fi

log_success "Repository structure verified"
cd /srv/classrepo
log_step "📂 Working directory: $(pwd)"

log_step "🔧 Installing JupyterLab extension using proper build process..."
echo "====================================================================="
echo "                    EXTENSION INSTALLATION"
echo "====================================================================="

# Set error handling for extension installation - don't let it stop the entire script
set +e  # Don't exit on errors in this section

# Install in development mode in both environments to ensure all files are available
log_step "📦 Installing extension in system environment..."
sudo -E pip3 install -e ./shm_function_selector/ 2>&1 | sed 's/^/[PIP-SYS] /' || log_warning "System environment installation had issues"

log_step "📦 Installing extension in TLJH user environment..."
sudo -E /opt/tljh/user/bin/pip install -e ./shm_function_selector/ 2>&1 | sed 's/^/[PIP-USER] /' || log_warning "TLJH user environment installation had issues"

# The -e install should trigger the build hooks and create the labextension
log_step "🔍 Checking if labextension was built..."
LABEXT_DIR="./shm_function_selector/shm_function_selector/labextension"
if [ ! -d "$LABEXT_DIR" ]; then
    log_warning "Labextension not built automatically, building manually..."
    cd /srv/classrepo/shm_function_selector
    log_step "📂 Changed to: $(pwd)"
    
    # Check Node.js and npm availability
    log_step "🔍 Checking Node.js environment..."
    node --version 2>&1 | sed 's/^/[NODE-VERSION] /' || log_error "Node.js not available"
    npm --version 2>&1 | sed 's/^/[NPM-VERSION] /' || log_error "NPM not available"
    
    # Install Node.js dependencies using jlpm (JupyterLab's package manager)
    log_step "📥 Installing dependencies with jlpm..."
    if ! sudo -E /opt/tljh/user/bin/jlpm install 2>&1 | sed 's/^/[JLPM-INSTALL] /'; then
        log_warning "jlpm install failed, trying npm install..."
        npm install 2>&1 | sed 's/^/[NPM-INSTALL-FALLBACK] /' || log_error "Both jlpm and npm install failed"
    fi
    
    # Build using jlpm commands
    log_step "🔨 Building extension with jlpm..."
    if ! sudo -E /opt/tljh/user/bin/jlpm run build:prod 2>&1 | sed 's/^/[JLPM-BUILD] /'; then
        log_warning "jlpm build:prod failed, trying npm build commands..."
        if ! npm run build:lib 2>&1 | sed 's/^/[NPM-BUILD-LIB] /'; then
            log_error "npm run build:lib failed"
        fi
        if ! npm run build:labextension 2>&1 | sed 's/^/[NPM-BUILD-EXT] /'; then
            log_error "npm run build:labextension failed"
        fi
    fi
    
    cd /srv/classrepo
    log_step "📂 Returned to: $(pwd)"
else
    log_success "Labextension directory found: $LABEXT_DIR"
fi

# Fix import paths in handlers.py for relocated introspection module
log_step "🔧 Fixing import paths in extension..."
HANDLER_FILE="/srv/classrepo/shm_function_selector/shm_function_selector/handlers.py"
if [ -f "$HANDLER_FILE" ]; then
    log_step "📝 Updating import paths in handlers.py..."
    sed -i 's/from shm_function_selector\.introspection/from ..introspection/g' "$HANDLER_FILE" 2>&1 | sed 's/^/[IMPORT-FIX] /' || log_warning "Import path fix failed"
    log_success "Import paths updated"
else
    log_warning "handlers.py not found at: $HANDLER_FILE"
fi

# Create missing directories/files that pip might expect
log_step "📁 Creating required directories and files..."
EXT_DIR="/opt/tljh/user/share/jupyter/labextensions/shm-function-selector"
sudo mkdir -p "$EXT_DIR" 2>&1 | sed 's/^/[MKDIR] /' || log_error "Failed to create extension directory"
sudo touch "$EXT_DIR/build_log.json" 2>&1 | sed 's/^/[TOUCH] /' || log_warning "Could not create build_log.json"
log_success "Required directories created"

# The labextension should already be installed via pip editable install
log_step "🔍 Verifying labextension installation..."
log_step "📋 Current labextension list:"
sudo -E /opt/tljh/user/bin/jupyter labextension list 2>&1 | sed 's/^/[LABEXT-LIST] /'

if sudo -E /opt/tljh/user/bin/jupyter labextension list 2>/dev/null | grep -q "shm-function-selector"; then
    log_success "Labextension already installed and detected"
else
    log_warning "Labextension not found in list, installing manually..."
    # Use develop mode for development installs
    log_step "🔧 Installing labextension in development mode..."
    sudo -E /opt/tljh/user/bin/jupyter labextension develop ./shm_function_selector/ --overwrite 2>&1 | sed 's/^/[LABEXT-DEVELOP] /' || log_error "Labextension development install failed"
fi

# Add custom server extension loader to config
log_step "⚙️ Configuring server extension..."
CONFIG_FILE="/opt/tljh/user/etc/jupyter/jupyter_server_config.py"
log_step "📄 Server config file: $CONFIG_FILE"

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

# Append to config if not already present
log_step "🔍 Checking if server extension config already exists..."
if ! grep -q "load_shm_extension" "$CONFIG_FILE" 2>/dev/null; then
    log_step "📝 Adding server extension configuration..."
    cat /tmp/shm_extension_config.py | sudo tee -a "$CONFIG_FILE" > /dev/null 2>&1 | sed 's/^/[CONFIG-APPEND] /'
    log_success "Server extension configuration added"
else
    log_success "Server extension configuration already present"
fi

log_step "📋 Current server config (last 15 lines):"
tail -15 "$CONFIG_FILE" 2>/dev/null | sed 's/^/[SERVER-CONFIG] /' || log_warning "Could not read server config"

rm -f /tmp/shm_extension_config.py

# Rebuild JupyterLab to integrate the extension
log_step "🔨 Building JupyterLab with extension..."
echo "====================================================================="
echo "                        JUPYTERLAB BUILD"
echo "====================================================================="
START_TIME=$(date +%s)
if sudo -E /opt/tljh/user/bin/jupyter lab build 2>&1 | sed 's/^/[LAB-BUILD] /'; then
    END_TIME=$(date +%s)
    BUILD_DURATION=$((END_TIME - START_TIME))
    log_success "JupyterLab build successful (took ${BUILD_DURATION} seconds)"
else
    END_TIME=$(date +%s)
    BUILD_DURATION=$((END_TIME - START_TIME))
    log_warning "JupyterLab build failed after ${BUILD_DURATION} seconds, but proceeding..."
    log_step "Checking for build logs..."
    find /opt/tljh/user -name "*.log" -path "*jupyter*" -exec tail -10 {} \; 2>/dev/null | sed 's/^/[BUILD-LOG] /' || log_step "No build logs found"
fi

# Verify extension installation
log_step "🔍 Verifying extension installation..."
echo "====================================================================="
echo "                       VERIFICATION"
echo "====================================================================="

log_step "📋 Server extensions:"
sudo -E /opt/tljh/user/bin/jupyter server extension list 2>&1 | sed 's/^/[SERVER-EXT] /' 
SERVER_EXT_FOUND=$(sudo -E /opt/tljh/user/bin/jupyter server extension list 2>/dev/null | grep -i shm || echo "not found")
if [[ "$SERVER_EXT_FOUND" == "not found" ]]; then
    log_warning "SHM server extension not found in list"
else
    log_success "SHM server extension detected: $SERVER_EXT_FOUND"
fi

log_step "📋 Lab extensions:"
sudo -E /opt/tljh/user/bin/jupyter labextension list 2>&1 | sed 's/^/[LAB-EXT] /'
LAB_EXT_FOUND=$(sudo -E /opt/tljh/user/bin/jupyter labextension list 2>/dev/null | grep -i shm || echo "not found")
if [[ "$LAB_EXT_FOUND" == "not found" ]]; then
    log_warning "SHM lab extension not found in list"
else
    log_success "SHM lab extension detected: $LAB_EXT_FOUND"
fi

# Check for extension files
log_step "🔍 Searching for extension files..."
find /opt/tljh/user -name "*shm*" -type f 2>/dev/null | head -10 | sed 's/^/[EXT-FILES] /' || log_step "No extension files found in search"

log_success "JupyterLab extension installation complete!"

# Resume strict error handling
set -e

echo "========================================="
log_success "🎉 Extension installation finished!"
echo "========================================="