#!/bin/bash
set -euxo pipefail

# This script is downloaded and run by cloud_init_setup.sh
# Environment variables are passed from the bootstrap script

log_step() {
    echo "[SETUP-LOG $(date '+%Y-%m-%d %H:%M:%S')] ${1:-}"
}

log_error() {
    echo "[SETUP-ERROR $(date '+%Y-%m-%d %H:%M:%S')] ❌ ${1:-}"
}

log_success() {
    echo "[SETUP-SUCCESS $(date '+%Y-%m-%d %H:%M:%S')] ✅ ${1:-}"
}

log_warning() {
    echo "[SETUP-WARNING $(date '+%Y-%m-%d %H:%M:%S')] ⚠️ ${1:-}"
}

# Use environment variables with defaults
JUPYTER_ADMIN_USER=${JUPYTER_ADMIN_USER:-ubuntu}
GITHUB_OWNER=${GITHUB_OWNER:-ersimpson}
GITHUB_REPO=${GITHUB_REPO:-shm}
GITHUB_BRANCH=${GITHUB_BRANCH:-main}
AWS_REGION=${AWS_REGION:-us-east-2}
GITHUB_PAT_SSM_PARAMETER_NAME="/github/pat"
GIT_USER_NAME="SHMTools Admin"
GIT_USER_EMAIL="admin@shmtools.com"

log_step "🎯 Installing TLJH (The Littlest JupyterHub)..."
curl -L https://tljh.jupyter.org/bootstrap.py | sudo python3 - --admin $JUPYTER_ADMIN_USER
log_success "TLJH installation complete!"

# Configure FirstUse Authenticator
log_step "👥 Installing and configuring FirstUse Authenticator..."
sudo -E pip install jupyterhub-firstuseauthenticator
sudo tljh-config set auth.type firstuseauthenticator.FirstUseAuthenticator
sudo tljh-config set auth.FirstUseAuthenticator.create_users true
sudo tljh-config reload
log_success "FirstUse Authenticator configured!"

# Repo dir
log_step "📁 Creating repository directory..."
mkdir -p /srv/classrepo
chown $JUPYTER_ADMIN_USER:$JUPYTER_ADMIN_USER /srv/classrepo

# Git credential helper
log_step "🔐 Creating Git credential helper..."
cat >/usr/local/bin/gh-cred-helper <<'HLP'
#!/usr/bin/env bash
set -euo pipefail
action="${1:-get}"
while IFS= read -r line; do
  [ -z "$line" ] && break
  case "$line" in
    protocol=*) ;;
    host=*) host="${line#host=}" ;;
  esac
done
if [ "$action" = "get" ] && [ "${host:-}" = "github.com" ]; then
  token=$(aws ssm get-parameter --region ${AWS_REGION} --name "${GITHUB_PAT_SSM_PARAMETER_NAME}" --with-decryption --query Parameter.Value --output text)
  echo "username=x-access-token"
  echo "password=${token}"
fi
exit 0
HLP
chmod +x /usr/local/bin/gh-cred-helper

# Git identity + helper
log_step "🎛️ Configuring Git identity and credential helper..."
su - $JUPYTER_ADMIN_USER -c "git config --global user.name '$GIT_USER_NAME'"
su - $JUPYTER_ADMIN_USER -c "git config --global user.email '$GIT_USER_EMAIL'"
su - $JUPYTER_ADMIN_USER -c "git config --global credential.helper '/usr/local/bin/gh-cred-helper'"

# Clone repo
log_step "📥 Cloning repository from GitHub..."
su - $JUPYTER_ADMIN_USER -c "cd /srv && git clone https://github.com/$GITHUB_OWNER/$GITHUB_REPO.git classrepo || true"
su - $JUPYTER_ADMIN_USER -c "cd /srv/classrepo && git fetch && git checkout $GITHUB_BRANCH || true"
log_success "Repository cloned successfully!"

# Install shmtools package
log_step "📦 Installing shmtools package and dependencies..."
cd /srv/classrepo
if [ -f requirements.txt ]; then
  sudo -E pip3 install -r requirements.txt || log_error "Failed to install requirements.txt"
fi
if [ -f requirements-dev.txt ]; then
  sudo -E pip3 install -r requirements-dev.txt || log_error "Failed to install requirements-dev.txt"
fi
sudo -E pip3 install -e .
sudo -E /opt/tljh/user/bin/pip install -e .
log_success "shmtools package installed!"

# Install Node.js and extension
log_step "🧩 Building JupyterLab extension..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get remove -y libnode-dev || log_warning "No conflicting Node.js packages"
sudo apt-get install -y nodejs

# Install extension
if [ -f "/srv/classrepo/jupyterhub/install_jupyterlab_extension.sh" ]; then
    dos2unix /srv/classrepo/jupyterhub/install_jupyterlab_extension.sh || true
    bash /srv/classrepo/jupyterhub/install_jupyterlab_extension.sh || log_error "Extension installation failed"
    log_success "Extension installation completed"
else
    log_warning "Extension installation script not found"
fi

# Data files setup
log_step "📂 Setting up data files directory..."
mkdir -p /srv/classrepo/examples/data/data_files
cat > "/srv/classrepo/examples/data/data_files/DATA_SETUP_REQUIRED.txt" <<'DATASETUP'
IMPORTANT: Data files are required for SHMTools examples.
Required files (~161MB total):
- data3SS.mat (25MB)
- dataSensorDiagnostic.mat (63KB)
- data_CBM.mat (54MB)
- data_example_ActiveSense.mat (32MB)
- data_OSPExampleModal.mat (50KB)

To complete setup:
scp -i ~/.ssh/class-key-ssh-rsa /local/path/*.mat ubuntu@PUBLIC_IP:/srv/classrepo/examples/data/data_files/
DATASETUP

# Set ownership
chown -R $JUPYTER_ADMIN_USER:$JUPYTER_ADMIN_USER /srv/classrepo

# Install Claude Code
log_step "🤖 Installing Claude Code CLI..."
sudo -u $JUPYTER_ADMIN_USER bash -c "curl -fsSL https://claude.ai/install.sh | bash" || log_warning "Claude Code installation failed"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/$JUPYTER_ADMIN_USER/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/skel/.bashrc

# Disable UFW
ufw disable || log_step "UFW not installed or already disabled"

PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
log_success "🎉 SETUP COMPLETE!"
log_success "🌐 JupyterHub is ready at http://$PUBLIC_IP"
log_step "👤 Admin login: $JUPYTER_ADMIN_USER (set password on first login)"
log_step "👥 Users can create accounts by choosing any username/password"