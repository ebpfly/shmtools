#!/bin/bash

# remote_update.sh - Trigger deployment update on remote SHM instance
# This script connects to a remote EC2 instance and runs the update_deployment.sh script
# Usage: ./remote_update.sh [IP_ADDRESS]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}$(date '+%H:%M:%S')${NC} - $1"
}

error() {
    echo -e "${RED}$(date '+%H:%M:%S')${NC} - ERROR: $1" >&2
}

success() {
    echo -e "${GREEN}$(date '+%H:%M:%S')${NC} - $1"
}

warning() {
    echo -e "${YELLOW}$(date '+%H:%M:%S')${NC} - WARNING: $1"
}

info() {
    echo -e "${CYAN}$(date '+%H:%M:%S')${NC} - $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SSH_KEY="$HOME/.ssh/class-key-ssh-rsa"
SSH_USER="ubuntu"
REMOTE_REPO_DIR="/srv/classrepo"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [IP_ADDRESS]

Update a remote SHM deployment by running the update script on the target instance.

OPTIONS:
    -h, --help              Show this help message
    -k, --key PATH          SSH key path (default: ~/.ssh/class-key-ssh-rsa)
    -u, --user USER         SSH user (default: ubuntu)
    -v, --verbose           Enable verbose SSH output
    -t, --timeout SECONDS   SSH connection timeout (default: 30)

ARGUMENTS:
    IP_ADDRESS              Target instance IP address
                           If not provided, will try to detect from AWS

EXAMPLES:
    $0 3.130.148.209                    # Update specific instance
    $0 --verbose 3.130.148.209          # Update with verbose output
    $0 --key ~/.ssh/my-key.pem 1.2.3.4  # Use custom SSH key
    $0                                   # Auto-detect instance IP

NOTES:
    - Requires SSH access to the target instance
    - The instance must have the SHM repository at $REMOTE_REPO_DIR
    - The update_deployment.sh script must exist on the target instance
EOF
}

# Function to detect running EC2 instance
detect_instance_ip() {
    log "Detecting running SHM EC2 instance..."
    
    # Check if AWS CLI is available
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install it or provide IP address manually."
        return 1
    fi
    
    # Look for running instances with SHM-related tags
    local instance_ip
    instance_ip=$(aws ec2 describe-instances \
        --filters "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[?Tags[?Key==`Name` && (contains(Value, `tljh`) || contains(Value, `shm`) || contains(Value, `class`))]].PublicIpAddress' \
        --output text 2>/dev/null | head -1)
    
    if [[ -z "$instance_ip" || "$instance_ip" == "None" ]]; then
        # Fallback: get any running instance
        instance_ip=$(aws ec2 describe-instances \
            --filters "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text 2>/dev/null)
    fi
    
    if [[ -n "$instance_ip" && "$instance_ip" != "None" ]]; then
        echo "$instance_ip" | tr -d '\n'
        return 0
    else
        return 1
    fi
}

# Function to validate SSH connectivity
validate_ssh() {
    local ip="$1"
    log "Testing SSH connectivity to $ip..."
    
    if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SSH_USER@$ip" 'echo "SSH connection successful"' &>/dev/null; then
        error "Cannot connect to $ip via SSH"
        error "Please check:"
        error "  - Instance is running and accessible"
        error "  - SSH key path: $SSH_KEY"
        error "  - Security group allows SSH from your IP"
        return 1
    fi
    
    success "SSH connectivity confirmed"
    return 0
}

# Function to check if update script exists on remote
check_remote_script() {
    local ip="$1"
    log "Checking if update script exists on remote instance..."
    
    if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SSH_USER@$ip" "test -f $REMOTE_REPO_DIR/jupyterhub/update_deployment.sh"; then
        error "Update script not found at $REMOTE_REPO_DIR/jupyterhub/update_deployment.sh"
        error "Please ensure the instance has the SHM repository with the update script"
        return 1
    fi
    
    success "Update script found on remote instance"
    return 0
}

# Function to run remote update
run_remote_update() {
    local ip="$1"
    local ssh_opts="-i $SSH_KEY -o StrictHostKeyChecking=no"
    
    if [[ "$VERBOSE" == "true" ]]; then
        ssh_opts="$ssh_opts -v"
    fi
    
    info "Starting remote update on $ip..."
    info "This will take several minutes. Please wait..."
    echo "======================================================================"
    
    # Execute the update script on remote instance
    if ssh $ssh_opts -t "$SSH_USER@$ip" "cd $REMOTE_REPO_DIR && ./jupyterhub/update_deployment.sh"; then
        echo "======================================================================"
        success "Remote update completed successfully!"
        info "JupyterHub should be accessible at: http://$ip"
    else
        echo "======================================================================"
        error "Remote update failed!"
        error "Check the output above for details"
        return 1
    fi
}

# Parse command line arguments
VERBOSE=false
IP_ADDRESS=""
SSH_TIMEOUT=30

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -k|--key)
            SSH_KEY="$2"
            shift 2
            ;;
        -u|--user)
            SSH_USER="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--timeout)
            SSH_TIMEOUT="$2"
            shift 2
            ;;
        -*)
            error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$IP_ADDRESS" ]]; then
                IP_ADDRESS="$1"
            else
                error "Multiple IP addresses provided: $IP_ADDRESS and $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate SSH key exists
if [[ ! -f "$SSH_KEY" ]]; then
    error "SSH key not found: $SSH_KEY"
    error "Please check the key path or use --key option"
    exit 1
fi

# Auto-detect IP if not provided
if [[ -z "$IP_ADDRESS" ]]; then
    info "No IP address provided, attempting auto-detection..."
    if ! IP_ADDRESS=$(detect_instance_ip); then
        error "Could not detect running instance IP"
        error "Please provide the IP address manually: $0 <IP_ADDRESS>"
        exit 1
    fi
    info "Detected instance IP: $IP_ADDRESS"
fi

# Validate IP address format
if [[ ! "$IP_ADDRESS" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
    error "Invalid IP address format: $IP_ADDRESS"
    exit 1
fi

# Main execution
echo "======================================================================"
echo "                    SHM REMOTE UPDATE SCRIPT"
echo "======================================================================"
log "Target instance: $IP_ADDRESS"
log "SSH key: $SSH_KEY"
log "SSH user: $SSH_USER"
echo "======================================================================"

# Validate prerequisites
validate_ssh "$IP_ADDRESS" || exit 1
check_remote_script "$IP_ADDRESS" || exit 1

# Ask for confirmation
echo ""
info "Ready to update SHM deployment on $IP_ADDRESS"
info "This will:"
info "  • Pull latest changes from Git"
info "  • Update Python dependencies"
info "  • Reinstall SHMTools package"  
info "  • Rebuild JupyterLab extension"
info "  • Restart JupyterHub service"
echo ""
read -p "Continue with update? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    info "Update cancelled by user"
    exit 0
fi

# Execute remote update
echo ""
run_remote_update "$IP_ADDRESS"

echo ""
success "🎉 Remote update process completed!"
info "You can now access the updated deployment at: http://$IP_ADDRESS"