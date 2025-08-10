# JupyterHub AWS Deployment

This directory contains scripts and resources for deploying JupyterHub with The Littlest JupyterHub (TLJH) on AWS EC2.

## Overview

The setup script automates the deployment of a JupyterHub server on AWS EC2 with:
- Ubuntu 22.04 base image
- TLJH (The Littlest JupyterHub) installation
- Automatic GitHub repository cloning and setup
- Claude Code integration
- Python dependencies installation
- JupyterLab extension support

## Contents

- `setup_jupyterhub_aws.sh` - Main deployment script that creates and configures an EC2 instance
- `keys/` - Legacy SSH key pairs directory (script now uses `~/.ssh/`)

## Key Features

### Security
- Creates IAM role with minimal permissions (SSM parameter read access only)
- Stores GitHub PAT securely in AWS SSM Parameter Store
- Configures security group with SSH (restricted to your IP) and HTTP access
- Uses credential helper to fetch PAT on-demand (no secrets in user data)
- SSH keys stored in standard `~/.ssh/` directory

### Repository Integration
- Automatically clones the specified GitHub repository (`ebpfly/shm`)
- Sets up Git configuration with user credentials
- Installs shmtools-python package with all dependencies
- Installs Node.js 20.x (required for JupyterLab 4.4+ compatibility)
- Builds and installs the JupyterLab SHM Function Selector extension
- Configures Claude Code with proper PATH setup
- Complete development environment ready for use

### Instance Configuration
- Instance type: t3.medium (configurable)
- Storage: 20GB GP3 volume
- Region: us-east-2 (configurable)
- Admin user: ubuntu

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Required command-line tools: `aws`, `jq`, `curl`, `ssh-keygen`
3. GitHub Personal Access Token (PAT) with repository access
4. AWS account with permissions to:
   - Create EC2 instances
   - Create/manage IAM roles and policies
   - Create/manage security groups
   - Store parameters in SSM Parameter Store

## Usage

1. Edit the configuration section at the top of `setup_jupyterhub_aws.sh`:
   ```bash
   AWS_PROFILE="default"
   AWS_REGION="us-east-2"
   GITHUB_OWNER="ebpfly"
   GITHUB_REPO="shm"
   GIT_USER_NAME="Your Name"
   GIT_USER_EMAIL="your.email@example.com"
   ```

2. Run the setup script:
   ```bash
   ./setup_jupyterhub_aws.sh
   ```

3. The script will:
   - Check AWS credentials and dependencies
   - Create or reuse SSH key pairs
   - Set up security groups and IAM roles
   - Prompt for GitHub PAT if not already stored
   - Launch EC2 instance with TLJH
   - Display connection information

4. Wait 5-10 minutes for installation to complete
   - The script automatically installs Node.js 20.x for JupyterLab compatibility
   - All components install without manual intervention

5. Access JupyterHub at: `http://<PUBLIC_IP>`

6. SSH access:
   ```bash
   ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<PUBLIC_IP>
   ```

## Monitoring Installation Progress

After launching the instance, you can monitor the setup progress in real-time using several methods:

### 1. Live Log Streaming (Recommended)
```bash
# SSH and tail the cloud-init logs
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<PUBLIC_IP> "sudo tail -f /var/log/cloud-init-output.log"
```

### 2. SSM Session Manager (if SSH not ready)
```bash
# Connect via AWS SSM
aws ssm start-session --target <INSTANCE_ID> --region us-east-2
# Then run:
sudo tail -f /var/log/cloud-init-output.log
```

### 3. Check Installation Status
```bash
# Quick status check
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@<PUBLIC_IP> "sudo cloud-init status"
# Returns "done" when complete
```

### 4. EC2 Console Output
```bash
# Get console output (may have delay)
aws ec2 get-console-output --instance-id <INSTANCE_ID> --region us-east-2 --output text
```

The installation logs will show detailed progress including:
- TLJH installation
- Repository cloning
- shmtools package installation
- JupyterLab extension building
- Claude Code setup

Look for "SETUP COMPLETE" message to confirm everything is ready.

## Debugging and Development

For iterative development without recreating instances:

### 1. Debug Mode for Main Script
Skip instance creation and use existing instance:
```bash
DEBUG_MODE=true EXISTING_INSTANCE_IP=18.188.13.54 ./setup_jupyterhub_aws.sh
```

### 2. Individual Step Installation
Run specific installation steps on existing instance:
```bash
# Run all steps (with cleanup)
./debug_install.sh 18.188.13.54 all

# Run individual steps
./debug_install.sh 18.188.13.54 cleanup    # Clean previous attempts
./debug_install.sh 18.188.13.54 repo       # Repository setup
./debug_install.sh 18.188.13.54 shmtools   # Install shmtools package
./debug_install.sh 18.188.13.54 extension  # Build JupyterLab extension  
./debug_install.sh 18.188.13.54 claude     # Install Claude Code
```

### 3. Status Checking
Check installation status and component health:
```bash
./check_status.sh 18.188.13.54
```

This shows:
- SSH connectivity
- Cloud-init completion status
- JupyterHub service status
- Web interface accessibility  
- Repository presence
- shmtools package installation
- JupyterLab extension status
- Claude Code installation

### 4. Manual Debugging Commands
```bash
# SSH to instance
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@18.188.13.54

# View installation logs
sudo tail -f /var/log/cloud-init-output.log

# Check JupyterHub service
sudo systemctl status jupyterhub
sudo journalctl -u jupyterhub -f

# Test shmtools import
python3 -c "import shmtools; print('OK')"

# Check JupyterLab extensions
jupyter labextension list

# Test repository access
cd /srv/classrepo && git status
```

## Post-Installation

- Set admin password on first JupyterHub login
- Sign into Claude Code: SSH to instance and run `claude`
- Repository location on server: `/srv/classrepo`
- shmtools package is installed and ready to use
- JupyterLab extension is available in all notebooks

## Architecture

The deployment creates:
- EC2 instance running Ubuntu 22.04
- IAM role with SSM parameter read permissions
- Security group allowing SSH (from your IP) and HTTP (public)
- SSM parameter storing GitHub PAT (encrypted)
- TLJH installation with Python environment
- Cloned repository with dependencies installed

## Security Notes

- GitHub PAT is stored encrypted in AWS SSM Parameter Store
- EC2 instance role has minimal permissions (read-only access to specific SSM parameter)
- SSH access restricted to deployment machine's IP address
- No secrets are embedded in EC2 user data