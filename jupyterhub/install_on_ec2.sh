#!/bin/bash
# EC2 Installation Script for JupyterHub with SHMTools
# This script is executed from the cloned repository on EC2 instance
# It handles the complete installation and configuration of TLJH + SHMTools

set -euxo pipefail

# Log everything to cloud-init-output.log with timestamps
exec 1> >(logger -s -t user-data -p local6.info)
exec 2>&1

echo "========================================="
echo "🚀 Starting JupyterHub setup at $(date)"
echo "========================================="

# Enhanced logging functions for installation script
log_step() {
    echo "[SETUP-LOG $(date '+%Y-%m-%d %H:%M:%S')] ${1:-}"
}

log_error() {
    echo "[SETUP-ERROR $(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${1:-}"
}

log_success() {
    echo "[SETUP-SUCCESS $(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: ${1:-}"
}

log_warning() {
    echo "[SETUP-WARNING $(date '+%Y-%m-%d %H:%M:%S')] WARNING: ${1:-}"
}

# Parameters passed from bootstrap script as environment variables
# These should be set by the bootstrap script before calling this
JUPYTER_ADMIN_USER="${JUPYTER_ADMIN_USER:-ubuntu}"
AWS_REGION="${AWS_REGION:-us-east-2}"
GITHUB_PAT_SSM_PARAMETER_NAME="${GITHUB_PAT_SSM_PARAMETER_NAME:-/github/pat}"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"
GITHUB_OWNER="${GITHUB_OWNER:-}"
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
ENABLE_SSL="${ENABLE_SSL:-false}"
USE_DOMAIN="${USE_DOMAIN:-}"
SSL_EMAIL="${SSL_EMAIL:-}"
CERT_BACKUP_BUCKET="${CERT_BACKUP_BUCKET:-}"
CERT_BACKUP_KEY="${CERT_BACKUP_KEY:-}"

export DEBIAN_FRONTEND=noninteractive

log_step "📦 Installing TLJH (The Littlest JupyterHub)..."
curl -L https://tljh.jupyter.org/bootstrap.py | sudo python3 - --admin ${JUPYTER_ADMIN_USER} 2>&1 | sed 's/^/[TLJH-INSTALL] /'
log_success "TLJH installation complete!"

# Configure FirstUse Authenticator for self-service account creation
log_step "👥 Installing and configuring FirstUse Authenticator..."
sudo -E pip install jupyterhub-firstuseauthenticator 2>&1 | sed 's/^/[FIRSTUSE-INSTALL] /'
sudo tljh-config set auth.type firstuseauthenticator.FirstUseAuthenticator 2>&1 | sed 's/^/[AUTH-CONFIG] /'
sudo tljh-config set auth.FirstUseAuthenticator.create_users true 2>&1 | sed 's/^/[AUTH-CONFIG] /'
sudo tljh-config reload 2>&1 | sed 's/^/[CONFIG-RELOAD] /'
log_success "FirstUse Authenticator configured!"

# Install shmtools package and dependencies
log_step "📦 Installing shmtools package and dependencies..."
echo "========================================="
echo "📦 INSTALLING SHMTOOLS PACKAGE"
echo "========================================="
cd /srv/classrepo
if [ -f requirements.txt ]; then
  log_step "📋 Installing requirements.txt..."
  sudo -E pip3 install -r requirements.txt 2>&1 | sed 's/^/[PIP-REQUIREMENTS] /' || log_error "Failed to install requirements.txt"
fi
if [ -f requirements-dev.txt ]; then
  log_step "🛠️ Installing requirements-dev.txt..."
  sudo -E pip3 install -r requirements-dev.txt 2>&1 | sed 's/^/[PIP-DEV-REQUIREMENTS] /' || log_error "Failed to install requirements-dev.txt"
fi
# Install shmtools in development mode
log_step "🔧 Installing shmtools package in development mode..."
sudo -E pip3 install -e . 2>&1 | sed 's/^/[PIP-SHMTOOLS-SYS] /' || log_error "Failed to install shmtools in system environment"
# Also install in TLJH user environment for server extension access
log_step "🔧 Installing shmtools in TLJH user environment..."
sudo -E /opt/tljh/user/bin/pip install -e . 2>&1 | sed 's/^/[PIP-SHMTOOLS-USER] /' || log_error "Failed to install shmtools in TLJH user environment"
log_success "shmtools package installed!"

# Install JupyterLab extension
log_step "🧩 Building JupyterLab extension..."
echo "========================================="
echo "🧩 BUILDING JUPYTERLAB EXTENSION"
echo "========================================="
# Install Node.js 20.x (required for JupyterLab 4.4+)
log_step "📋 Installing Node.js 20.x..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - 2>&1 | sed 's/^/[NODEJS-INSTALL] /'
# Remove conflicting packages before installing nodejs 20.x
sudo apt-get remove -y libnode-dev 2>&1 | sed 's/^/[NODEJS-CLEANUP] /' || log_warning "No conflicting Node.js packages to remove"
sudo apt-get install -y nodejs 2>&1 | sed 's/^/[NODEJS-INSTALL] /'
NODE_VERSION=$(node --version)
log_success "Node.js installed: $NODE_VERSION"

# Install JupyterLab extension using separate script
log_step "🔧 Running JupyterLab extension installation script..."
if [ -f "/srv/classrepo/jupyterhub/install_jupyterlab_extension.sh" ]; then
    log_step "🛠️ Fixing script line endings (DOS/Unix compatibility)..."
    # Fix line endings in case of DOS format (defensive programming)
    if command -v dos2unix >/dev/null 2>&1; then
        dos2unix /srv/classrepo/jupyterhub/install_jupyterlab_extension.sh 2>&1 | sed 's/^/[DOS2UNIX] /'
    else
        # Fallback: remove carriage returns
        tr -d '\r' < /srv/classrepo/jupyterhub/install_jupyterlab_extension.sh > /tmp/install_jupyterlab_extension_fixed.sh
        mv /tmp/install_jupyterlab_extension_fixed.sh /srv/classrepo/jupyterhub/install_jupyterlab_extension.sh
        chmod +x /srv/classrepo/jupyterhub/install_jupyterlab_extension.sh
        log_step "Line endings fixed with fallback method"
    fi
    # Run the extension installation script with enhanced logging
    log_step "🚀 Executing extension installation script..."
    bash /srv/classrepo/jupyterhub/install_jupyterlab_extension.sh 2>&1 | sed 's/^/[EXTENSION-INSTALL] /' || log_error "Extension installation script failed"
    log_success "Extension installation script completed"
else
    log_error "Extension installation script not found at /srv/classrepo/jupyterhub/install_jupyterlab_extension.sh"
    log_warning "Skipping JupyterLab extension installation..."
fi

# Copy data files from local development machine to EC2 instance
log_step "📂 Setting up data files directory..."
mkdir -p /srv/classrepo/examples/data/data_files
chown -R ${JUPYTER_ADMIN_USER}:${JUPYTER_ADMIN_USER} /srv/classrepo/examples/data/data_files

log_step "📥 Downloading required data files..."
echo "========================================="
echo "📥 DOWNLOADING DATA FILES"
echo "========================================="

# Define data files to download (these would need to be hosted somewhere accessible)
# For now, create placeholder message since we need a way to get the files to EC2
DATA_FILES_DIR="/srv/classrepo/examples/data/data_files"

# Create a marker file indicating data setup is needed
cat > "$DATA_FILES_DIR/DATA_SETUP_REQUIRED.txt" <<'DATASETUP'
IMPORTANT: Data files are required for SHMTools examples to work properly.

Required files (~161MB total):
- data3SS.mat (25MB) - 3-story structure, 8192×5×170
- dataSensorDiagnostic.mat (63KB) - Sensor health
- data_CBM.mat (54MB) - Condition monitoring  
- data_example_ActiveSense.mat (32MB) - Active sensing
- data_OSPExampleModal.mat (50KB) - Modal analysis

To complete setup, copy these files from your local machine:
scp -i ~/.ssh/class-key-ssh-rsa /path/to/local/examples/data/data_files/*.mat ubuntu@PUBLIC_IP:/srv/classrepo/examples/data/data_files/

Or use the remote_update.sh script which will copy them automatically.
DATASETUP

log_warning "Data files not automatically copied - see DATA_SETUP_REQUIRED.txt for instructions"
log_step "💡 Use remote_update.sh script to copy data files automatically"

# Set proper ownership
log_step "🔧 Setting proper file ownership..."
chown -R ${JUPYTER_ADMIN_USER}:${JUPYTER_ADMIN_USER} /srv/classrepo 2>&1 | sed 's/^/[CHOWN] /'
log_success "File ownership configured"

# Claude Code (native installer) - Non-interactive installation
log_step "🤖 Installing Claude Code CLI..."
echo "========================================="
echo "🤖 INSTALLING CLAUDE CODE CLI"
echo "========================================="
# Run as ubuntu user without login shell to avoid TTY issues
sudo -u ${JUPYTER_ADMIN_USER} bash -c "curl -fsSL https://claude.ai/install.sh | bash" 2>&1 | sed 's/^/[CLAUDE-INSTALL] /' || log_warning "Claude Code installation failed (expected - requires manual auth), continuing..."
# Add Claude to PATH for all users
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/${JUPYTER_ADMIN_USER}/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/skel/.bashrc
log_success "Claude Code installer run - users will need to authenticate on first use"

# Keep port 80 open if ufw is present
log_step "🔥 Disabling UFW firewall if present..."
ufw disable 2>&1 | sed 's/^/[UFW] /' || log_step "UFW not installed or already disabled"

# Certificate restore from backup (before SSL setup)
if [ "${ENABLE_SSL}" = "true" ] && [ -n "${USE_DOMAIN}" ]; then
  log_step "🔐 Checking for certificate backup to restore..."
  
  # Try to restore existing certificate from S3
  if aws s3 cp "s3://${CERT_BACKUP_BUCKET}/${CERT_BACKUP_KEY}" /tmp/acme-restore.json 2>/dev/null; then
    log_step "Found certificate backup, validating..."
    
    # Validate that the backup contains valid Let's Encrypt certificates
    if jq -e '.letsencrypt.Certificates != null and (.letsencrypt.Certificates | length) > 0' /tmp/acme-restore.json >/dev/null 2>&1; then
      log_success "Certificate backup contains valid Let's Encrypt certificates, restoring..."
      
      # Show certificate details
      CERT_DOMAIN=$(jq -r '.letsencrypt.Certificates[0].domain.main // "unknown"' /tmp/acme-restore.json 2>/dev/null)
      log_success "Restoring certificate for domain: $CERT_DOMAIN"
      
      # Create the TLJH state directory if it doesn't exist
      mkdir -p /opt/tljh/state
      
      # Copy the certificate file to the correct location
      cp /tmp/acme-restore.json /opt/tljh/state/acme.json
      chmod 600 /opt/tljh/state/acme.json
      chown root:root /opt/tljh/state/acme.json
      
      # Also create the traefik acme directory structure that TLJH expects
      mkdir -p /opt/tljh/state/traefik/acme
      cp /tmp/acme-restore.json /opt/tljh/state/traefik/acme/acme.json
      chmod 600 /opt/tljh/state/traefik/acme/acme.json
      chown root:root /opt/tljh/state/traefik/acme/acme.json
      
      log_success "Valid Let's Encrypt certificate backup restored"
    else
      log_step "Certificate backup exists but contains no valid certificates"
      log_step "Will generate new certificates (backup likely from rate-limited attempt)"
    fi
    
    rm -f /tmp/acme-restore.json
  else
    log_step "No certificate backup found, will generate new certificates"
  fi
fi

# HTTPS setup 
if [ "${ENABLE_SSL}" = "true" ] && [ -n "${USE_DOMAIN}" ]; then
  log_step "🔒 Setting up HTTPS/SSL..."
  
  echo "========================================="
  echo "🔒 SSL/HTTPS SETUP FOR JUPYTERHUB"
  echo "========================================="
  log_step "Domain: ${USE_DOMAIN}"
  log_step "Email: ${SSL_EMAIL}"
  echo "========================================="
  
  # Configure HTTPS using TLJH's built-in support
  log_step "🔧 Configuring HTTPS settings..."
  tljh-config set https.enabled true
  tljh-config set https.letsencrypt.email "${SSL_EMAIL}"
  
  # Clear any existing domains first
  tljh-config unset https.letsencrypt.domains 2>/dev/null || true
  tljh-config add-item https.letsencrypt.domains "${USE_DOMAIN}"
  
  # Apply configuration changes - first general reload, then proxy specifically
  log_step "🔄 Applying TLJH configuration changes..."
  tljh-config reload
  
  # Wait for initial configuration to settle
  log_step "⏳ Waiting for initial configuration to settle..."
  sleep 10
  
  # Now reload the proxy specifically to ensure HTTPS is enabled
  log_step "🔄 Reloading proxy with HTTPS configuration..."
  tljh-config reload proxy
  
  # Wait for proxy reload to complete
  log_step "⏳ Waiting for proxy reload to complete..."
  sleep 15
  
  # Verify HTTPS is configured, if not force regeneration
  if ! grep -q ":443" /opt/tljh/state/traefik.toml 2>/dev/null; then
      log_warning "HTTPS not configured in traefik, forcing regeneration..."
      
      # Run the installer to ensure proper configuration
      log_step "Running TLJH installer to ensure proper configuration..."
      /opt/tljh/hub/bin/python3 -m tljh.installer --admin ${JUPYTER_ADMIN_USER}
      
      # Reload proxy again
      log_step "Reloading proxy after installer..."
      tljh-config reload proxy
      sleep 10
  fi
  
  # Final restart to ensure everything is working
  log_step "🔄 Final restart of services..."
  systemctl restart traefik || log_warning "Failed to restart Traefik"
  sleep 5
  systemctl restart jupyterhub || log_warning "Failed to restart JupyterHub"
  sleep 10
  
  # Verify HTTPS configuration  
  if grep -q "entryPoints.https" /opt/tljh/state/traefik.toml; then
      log_success "HTTPS entry point configured in Traefik"
  else
      log_error "HTTPS entry point NOT found in Traefik config!"
  fi
  
  # Test HTTPS connectivity
  log_step "🧪 Testing HTTPS connectivity..."
  HTTPS_RETRY=0
  HTTPS_MAX_RETRIES=24
  
  while [ $HTTPS_RETRY -lt $HTTPS_MAX_RETRIES ]; do
      if curl -s --connect-timeout 10 --max-time 15 "https://${USE_DOMAIN}" >/dev/null 2>&1; then
          log_success "HTTPS is responding at https://${USE_DOMAIN}"
          log_success "🎉 SSL SETUP COMPLETE!"
          echo "========================================="
          log_success "🌐 JupyterHub is now accessible at: https://${USE_DOMAIN}"
          echo "========================================="
          break
      fi
      
      log_step "⏳ Waiting for HTTPS... (attempt $((HTTPS_RETRY + 1))/$HTTPS_MAX_RETRIES)"
      
      # Show debug info every few attempts
      if [ $((HTTPS_RETRY % 6)) -eq 5 ]; then
          log_step "Debug - checking service status..."
          systemctl is-active traefik jupyterhub || true
      fi
      
      sleep 5
      HTTPS_RETRY=$((HTTPS_RETRY + 1))
  done
  
  if [ $HTTPS_RETRY -eq $HTTPS_MAX_RETRIES ]; then
      log_warning "HTTPS setup may need more time to complete"
  fi
  
  # Backup certificates after SSL setup (only if valid Let's Encrypt certificates exist)
  log_step "💾 Checking for valid Let's Encrypt certificates to backup..."
  CERT_FILE=""
  if [ -f "/opt/tljh/state/acme.json" ]; then
    CERT_FILE="/opt/tljh/state/acme.json"
  elif [ -f "/opt/tljh/state/traefik/acme/acme.json" ]; then
    CERT_FILE="/opt/tljh/state/traefik/acme/acme.json"
  fi
  
  if [ -n "$CERT_FILE" ]; then
    # Check if the file contains actual certificates (not just account info)
    if jq -e '.letsencrypt.Certificates != null and (.letsencrypt.Certificates | length) > 0' "$CERT_FILE" >/dev/null 2>&1; then
      log_step "Found valid Let's Encrypt certificates, backing up..."
      if aws s3 cp "$CERT_FILE" "s3://${CERT_BACKUP_BUCKET}/${CERT_BACKUP_KEY}" 2>/dev/null; then
        log_success "Let's Encrypt certificate backup saved to S3"
        
        # Show certificate details
        CERT_DOMAIN=$(jq -r '.letsencrypt.Certificates[0].domain.main // "unknown"' "$CERT_FILE" 2>/dev/null)
        log_success "Backed up certificate for domain: $CERT_DOMAIN"
      else
        log_warning "Failed to backup certificate to S3"
      fi
    else
      log_step "Certificate file exists but contains no valid Let's Encrypt certificates (likely due to rate limiting)"
      log_step "Skipping backup of incomplete certificate data"
    fi
  else
    log_step "No certificate file found to backup"
  fi
fi

echo "========================================="
log_success "🎉 SETUP COMPLETE at $(date)"
echo "========================================="
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
if [ "${ENABLE_SSL}" = "true" ] && [ -n "${USE_DOMAIN}" ]; then
  log_success "🌐 JupyterHub is ready at https://${USE_DOMAIN}"
  log_step "🔄 Backup access: http://$PUBLIC_IP"
else
  log_success "🌐 JupyterHub is ready at http://$PUBLIC_IP"
fi
log_step "👤 Admin login with username: ${JUPYTER_ADMIN_USER} (set password on first login)"
log_step "👥 Users can create accounts by choosing any username and password"
echo "========================================="
log_step "📊 FINAL SUMMARY:"
log_step "• JupyterHub URL: http://$PUBLIC_IP"
if [ "${ENABLE_SSL}" = "true" ] && [ -n "${USE_DOMAIN}" ]; then
  log_step "• HTTPS URL: https://${USE_DOMAIN}"
fi
log_step "• Repository: /srv/classrepo"
log_step "• Admin user: ${JUPYTER_ADMIN_USER}"
log_step "• SHMTools package: Installed in development mode"
log_step "• JupyterLab extension: Installed and configured"
log_step "• Data files: Available in /srv/classrepo/examples/data/data_files/"
log_step "• Claude Code CLI: Available after SSH login"
log_success "🚀 Ready for use!"
echo "========================================="