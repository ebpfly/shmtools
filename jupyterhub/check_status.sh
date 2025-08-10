#!/usr/bin/env bash
# Status checking utility for JupyterHub instances
# Usage: ./check_status.sh <PUBLIC_IP>

set -euo pipefail

KEY_PAIR_NAME="class-key-ssh-rsa"
JUPYTER_ADMIN_USER="ubuntu"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <PUBLIC_IP>"
  exit 1
fi

PUBLIC_IP="$1"
LOCAL_PRIVATE_KEY="$HOME/.ssh/${KEY_PAIR_NAME}"

# Helper function
ssh_run() {
  ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -i "${LOCAL_PRIVATE_KEY}" ${JUPYTER_ADMIN_USER}@${PUBLIC_IP} "$@" 2>/dev/null
}

echo "========================================="
echo "STATUS CHECK for ${PUBLIC_IP}"
echo "========================================="

# Check SSH connectivity
echo -n "SSH Connection: "
if ssh_run "echo 'OK'" >/dev/null 2>&1; then
  echo "✅ Connected"
else
  echo "❌ Failed"
  exit 1
fi

# Check system status
echo -n "System Status: "
uptime=$(ssh_run "uptime -p" 2>/dev/null || echo "unknown")
echo "✅ Up $uptime"

# Check cloud-init status
echo -n "Cloud-init: "
cloud_init_status=$(ssh_run "sudo cloud-init status --format=json" 2>/dev/null | jq -r '.status' 2>/dev/null || echo "unknown")
case "$cloud_init_status" in
  "done") echo "✅ Complete" ;;
  "running") echo "🟡 Running" ;;
  "error") echo "❌ Error" ;;
  *) echo "❓ $cloud_init_status" ;;
esac

# Check JupyterHub service
echo -n "JupyterHub Service: "
if ssh_run "sudo systemctl is-active jupyterhub" >/dev/null 2>&1; then
  echo "✅ Running"
else
  echo "❌ Not running"
fi

# Check JupyterHub accessibility
echo -n "JupyterHub Web: "
if curl -s --connect-timeout 5 "http://${PUBLIC_IP}" | grep -q "JupyterHub" 2>/dev/null; then
  echo "✅ Accessible"
else
  echo "❌ Not accessible"
fi

# Check repository
echo -n "Repository: "
if ssh_run "test -d /srv/classrepo/.git" >/dev/null 2>&1; then
  repo_status=$(ssh_run "cd /srv/classrepo && git status --porcelain" 2>/dev/null | wc -l)
  echo "✅ Present ($repo_status modified files)"
else
  echo "❌ Missing"
fi

# Check shmtools installation
echo -n "shmtools Package: "
if ssh_run "python3 -c 'import shmtools; print(shmtools.__version__)'" >/dev/null 2>&1; then
  version=$(ssh_run "python3 -c 'import shmtools; print(shmtools.__version__)'" 2>/dev/null)
  echo "✅ Installed (v$version)"
else
  echo "❌ Not installed"
fi

# Check JupyterLab extension
echo -n "JupyterLab Extension: "
if ssh_run "jupyter labextension list 2>/dev/null | grep -q shm-function-selector"; then
  echo "✅ Installed"
else
  echo "❌ Not installed"
fi

# Check Claude Code
echo -n "Claude Code: "
if ssh_run "which claude" >/dev/null 2>&1; then
  echo "✅ Installed"
else
  echo "❌ Not installed"
fi

echo ""
echo "========================================="
echo "QUICK ACTIONS:"
echo "========================================="
echo "View logs:     ssh -i $LOCAL_PRIVATE_KEY $JUPYTER_ADMIN_USER@$PUBLIC_IP 'sudo tail -f /var/log/cloud-init-output.log'"
echo "JupyterHub:    ssh -i $LOCAL_PRIVATE_KEY $JUPYTER_ADMIN_USER@$PUBLIC_IP 'sudo journalctl -u jupyterhub -f'"
echo "Debug install: ./debug_install.sh $PUBLIC_IP <step>"
echo "Access Hub:    http://$PUBLIC_IP"
echo "SSH:           ssh -i $LOCAL_PRIVATE_KEY $JUPYTER_ADMIN_USER@$PUBLIC_IP"