#!/bin/bash
# Backup SSL certificates from a JupyterHub instance to S3
# Usage: ./backup_certificates.sh [instance_ip]

set -euo pipefail

# Configuration
CERT_BACKUP_BUCKET="shmtools-deployment-bucket"
CERT_BACKUP_KEY="certificates/acme.json"
SSH_KEY="~/.ssh/class-key-ssh-rsa"

# Expand tilde
SSH_KEY="${SSH_KEY/#\~/${HOME}}"

# Get instance IP
if [ $# -eq 1 ]; then
    INSTANCE_IP="$1"
else
    # Auto-detect running instance
    INSTANCE_IP=$(aws ec2 describe-instances \
        --filters "Name=instance-state-name,Values=running" "Name=tag:Name,Values=tljh-class-server" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null)
    
    if [ "$INSTANCE_IP" = "None" ] || [ -z "$INSTANCE_IP" ]; then
        echo "❌ No running instance found. Please provide IP address."
        echo "Usage: $0 [instance_ip]"
        exit 1
    fi
fi

echo "🔐 Backing up certificates from instance: $INSTANCE_IP"

# Check if certificate file exists on the instance
if ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "sudo test -f /opt/tljh/state/acme.json" 2>/dev/null; then
    echo "📄 Found certificate file, downloading..."
    
    # Copy to accessible location on remote instance
    ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "sudo cp /opt/tljh/state/acme.json /tmp/acme.json && sudo chown ubuntu:ubuntu /tmp/acme.json"
    
    # Download to local temp file
    scp -i "$SSH_KEY" ubuntu@"$INSTANCE_IP":/tmp/acme.json /tmp/acme-backup.json
    
    # Validate that the backup contains valid Let's Encrypt certificates
    if jq -e '.letsencrypt.Certificates != null and (.letsencrypt.Certificates | length) > 0' /tmp/acme-backup.json >/dev/null 2>&1; then
        echo "✅ Found valid Let's Encrypt certificates, backing up..."
        
        # Upload to S3
        if aws s3 cp /tmp/acme-backup.json "s3://$CERT_BACKUP_BUCKET/$CERT_BACKUP_KEY"; then
            echo "✅ Certificate backup saved to S3: s3://$CERT_BACKUP_BUCKET/$CERT_BACKUP_KEY"
            
            # Show certificate info
            echo "📋 Certificate details:"
            CERT_DOMAIN=$(jq -r '.letsencrypt.Certificates[0].domain.main // "unknown"' /tmp/acme-backup.json 2>/dev/null)
            echo "Domain: $CERT_DOMAIN"
        else
            echo "❌ Failed to upload certificate backup to S3"
            exit 1
        fi
    else
        echo "⚠️ Certificate file exists but contains no valid Let's Encrypt certificates"
        echo "This likely means the certificates failed due to rate limiting"
        echo "Skipping backup of incomplete certificate data"
        exit 1
    fi
    
    # Cleanup
    rm -f /tmp/acme-backup.json
    ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "rm -f /tmp/acme.json"
    
else
    echo "⚠️ No certificate file found on instance"
    exit 1
fi