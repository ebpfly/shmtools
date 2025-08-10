#!/usr/bin/env bash
# Debug installation script for existing JupyterHub instance
# Usage: ./debug_install.sh <PUBLIC_IP> [step]
# Steps: all, repo, shmtools, extension, claude

set -euo pipefail

# Configuration (should match main script)
KEY_PAIR_NAME="class-key-ssh-rsa"
JUPYTER_ADMIN_USER="ubuntu"
GITHUB_OWNER="ebpfly"
GITHUB_REPO="shm"
GITHUB_BRANCH="main"
GIT_USER_NAME="Eric Flynn"
GIT_USER_EMAIL="ericbflynn@gmail.com"
AWS_REGION="us-east-2"
GITHUB_PAT_SSM_PARAMETER_NAME="/github/pat"

# Usage check
if [ $# -lt 1 ]; then
  echo "Usage: $0 <PUBLIC_IP> [step]"
  echo "Steps: all, repo, shmtools, extension, claude, cleanup"
  echo "Example: $0 18.188.13.54 shmtools"
  exit 1
fi

PUBLIC_IP="$1"
STEP="${2:-all}"
LOCAL_PRIVATE_KEY="$HOME/.ssh/${KEY_PAIR_NAME}"

# Helper functions
ssh_run() {
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -i "${LOCAL_PRIVATE_KEY}" ${JUPYTER_ADMIN_USER}@${PUBLIC_IP} "$@"
}

echo_step() {
  echo "========================================="
  echo "$1"
  echo "========================================="
}

# Step: Clean up previous failed attempts
cleanup_step() {
  echo_step "CLEANUP: Removing previous installation artifacts"
  ssh_run "sudo rm -rf /srv/classrepo || true"
  ssh_run "sudo pip3 uninstall -y shmtools || true"
  ssh_run "sudo pip3 uninstall -y shm-function-selector || true"
  ssh_run "sudo jupyter labextension uninstall shm-function-selector || true"
  echo "Cleanup complete!"
}

# Step: Repository setup
repo_step() {
  echo_step "STEP: Repository Setup"
  
  # Setup git credential helper
  ssh_run "cat > /tmp/gh-cred-helper <<'HLP'
#!/usr/bin/env bash
set -euo pipefail
action=\"\${1:-get}\"
while IFS= read -r line; do
  [ -z \"\$line\" ] && break
  case \"\$line\" in
    protocol=*) ;;
    host=*) host=\"\${line#host=}\" ;;
  esac
done
if [ \"\$action\" = \"get\" ] && [ \"\${host:-}\" = \"github.com\" ]; then
  token=\$(aws ssm get-parameter --region ${AWS_REGION} --name \"${GITHUB_PAT_SSM_PARAMETER_NAME}\" --with-decryption --query Parameter.Value --output text)
  echo \"username=x-access-token\"
  echo \"password=\${token}\"
fi
exit 0
HLP"
  
  ssh_run "sudo mv /tmp/gh-cred-helper /usr/local/bin/gh-cred-helper"
  ssh_run "sudo chmod +x /usr/local/bin/gh-cred-helper"
  
  # Git configuration
  ssh_run "git config --global user.name '${GIT_USER_NAME}'"
  ssh_run "git config --global user.email '${GIT_USER_EMAIL}'"
  ssh_run "git config --global credential.helper '/usr/local/bin/gh-cred-helper'"
  
  # Clone repository
  ssh_run "sudo mkdir -p /srv/classrepo && sudo chown ${JUPYTER_ADMIN_USER}:${JUPYTER_ADMIN_USER} /srv/classrepo"
  ssh_run "cd /srv && git clone https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}.git classrepo || (cd classrepo && git pull)"
  ssh_run "cd /srv/classrepo && git checkout ${GITHUB_BRANCH}"
  
  echo "Repository setup complete!"
}

# Step: shmtools installation
shmtools_step() {
  echo_step "STEP: shmtools Installation"
  
  ssh_run "cd /srv/classrepo && if [ -f requirements.txt ]; then sudo -E pip3 install -r requirements.txt; fi"
  ssh_run "cd /srv/classrepo && if [ -f requirements-dev.txt ]; then sudo -E pip3 install -r requirements-dev.txt; fi"
  ssh_run "cd /srv/classrepo && sudo -E pip3 install -e ."
  
  # Test installation
  ssh_run "python3 -c 'import shmtools; print(\"shmtools imported successfully\")'"
  
  echo "shmtools installation complete!"
}

# Step: JupyterLab extension
extension_step() {
  echo_step "STEP: JupyterLab Extension"
  
  # Install Node.js if not present
  ssh_run "node --version || sudo apt-get update && sudo apt-get install -y nodejs npm"
  
  # Build extension
  ssh_run "cd /srv/classrepo/shm_function_selector && npm install"
  ssh_run "cd /srv/classrepo/shm_function_selector && npm run build:lib"
  ssh_run "cd /srv/classrepo/shm_function_selector && npm run build:labextension:dev"
  
  # Install extension
  ssh_run "cd /srv/classrepo && sudo -E pip3 install -e shm_function_selector/"
  ssh_run "cd /srv/classrepo && sudo -E jupyter labextension develop --overwrite shm_function_selector/"
  ssh_run "sudo -E jupyter lab build"
  
  # Test extension
  ssh_run "jupyter labextension list | grep shm-function-selector || echo 'Extension not found in list'"
  
  echo "JupyterLab extension complete!"
}

# Step: Claude Code installation
claude_step() {
  echo_step "STEP: Claude Code Installation"
  
  ssh_run "curl -fsSL https://claude.ai/install.sh | bash || true"
  ssh_run "which claude || echo 'Claude not in PATH'"
  
  echo "Claude Code installation complete!"
}

# Execute steps
case "$STEP" in
  "cleanup")
    cleanup_step
    ;;
  "repo")
    repo_step
    ;;
  "shmtools")
    shmtools_step
    ;;
  "extension")
    extension_step
    ;;
  "claude")
    claude_step
    ;;
  "all")
    cleanup_step
    repo_step
    shmtools_step
    extension_step
    claude_step
    ;;
  *)
    echo "Unknown step: $STEP"
    echo "Available steps: all, cleanup, repo, shmtools, extension, claude"
    exit 1
    ;;
esac

echo "========================================="
echo "STEP '$STEP' COMPLETED SUCCESSFULLY!"
echo "========================================="
echo "Instance: ${PUBLIC_IP}"
echo "JupyterHub: http://${PUBLIC_IP}"
echo ""
echo "Next steps:"
echo "  - Test individual steps: ./debug_install.sh ${PUBLIC_IP} <step>"
echo "  - SSH to instance: ssh -i ${LOCAL_PRIVATE_KEY} ${JUPYTER_ADMIN_USER}@${PUBLIC_IP}"
echo "  - View logs: ssh -i ${LOCAL_PRIVATE_KEY} ${JUPYTER_ADMIN_USER}@${PUBLIC_IP} 'sudo journalctl -u jupyterhub -f'"