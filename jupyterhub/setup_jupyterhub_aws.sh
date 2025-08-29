#!/usr/bin/env bash
# One-shot JupyterHub (TLJH) on AWS EC2 with repo + Claude Code.
# - Loud logging (no >/dev/null)
# - Creates/reads GitHub PAT in SSM (/github/pat). Prompts if missing.
# - No secrets in user data; EC2 role reads PAT on demand.

set -Eeuo pipefail
trap 'echo "❌ Error on line $LINENO: $BASH_COMMAND"; exit 1' ERR

############################################
#            FILL THESE IN FIRST           #
############################################
AWS_PROFILE="default"          # your CLI profile (usually "default")
AWS_REGION="us-east-2"         # e.g. us-east-2 / us-west-2
INSTANCE_TYPE="t3.medium"
VOLUME_SIZE_GB=20

KEY_PAIR_NAME="class-key-ssh-rsa"      # EC2 key pair name to create/use
SSH_PUBLIC_KEY_PATH="~/.ssh/class-key-ssh-rsa.pub"  # or ~/.ssh/id_rsa.pub

# Debug mode - set to "true" to skip instance creation and just monitor existing
DEBUG_MODE="${DEBUG_MODE:-false}"
EXISTING_INSTANCE_IP="${EXISTING_INSTANCE_IP:-}"

JUPYTER_ADMIN_USER="ubuntu"    # TLJH admin (also your SSH user)

GITHUB_OWNER="ebpfly"
GITHUB_REPO="shmtools"
GITHUB_BRANCH="main"

GIT_USER_NAME="Eric Flynn"
GIT_USER_EMAIL="ericbflynn@gmail.com"

GITHUB_PAT_SSM_PARAMETER_NAME="/github/pat"  # where we'll store/read your PAT
SECURITY_GROUP_NAME="tljh-sg"
INSTANCE_NAME_TAG="tljh-class-server"

# Elastic IP configuration
ELASTIC_IP_NAME="shmtools-static-ip"  # Name tag for the Elastic IP
USE_DOMAIN="jfuse.shmtools.com"       # Domain name (leave empty to disable SSL)

# SSL/HTTPS configuration
# IMPORTANT: SSL will only work if:
# 1. USE_DOMAIN is set to a valid domain you own
# 2. The domain's DNS A record points to the EC2 instance's IP address
# 3. Port 80 and 443 are accessible from the internet
ENABLE_SSL=true                       # Set to false to disable SSL setup
SSL_EMAIL="ericbflynn@gmail.com"      # Email for Let's Encrypt certificates

# Certificate backup configuration
CERT_BACKUP_BUCKET="shmtools-deployment-bucket"  # S3 bucket for certificate backups
CERT_BACKUP_KEY="certificates/acme.json"         # S3 key for certificate backup
############################################

# --- helpers ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 1; }; }

echo "🔎 Checking local deps…"
need aws; need jq; need curl; need ssh-keygen

echo "🔐 Checking AWS identity for profile: ${AWS_PROFILE}"
aws sts get-caller-identity --profile "${AWS_PROFILE}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile "${AWS_PROFILE}")
echo "✔ Account: ${ACCOUNT_ID}"

MY_IP=$(curl -fsSL https://checkip.amazonaws.com)
YOUR_IP_CIDR="${MY_IP}/32"
echo "🌍 Your IP (for SSH allowlist): ${YOUR_IP_CIDR}"

echo "🔧 Ensuring default VPC exists in ${AWS_REGION} …"
VPC_ID=$(aws ec2 describe-vpcs --region "${AWS_REGION}" --profile "${AWS_PROFILE}" \
  --query 'Vpcs[?IsDefault==`true`].VpcId | [0]' --output text)
[ -n "${VPC_ID}" ] && [ "${VPC_ID}" != "None" ] || { echo "No default VPC in ${AWS_REGION}"; exit 1; }

SUBNET_ID=$(aws ec2 describe-subnets --region "${AWS_REGION}" --profile "${AWS_PROFILE}" \
  --filters "Name=vpc-id,Values=${VPC_ID}" --query 'Subnets[0].SubnetId' --output text)
echo "✔ Using VPC: ${VPC_ID}  Subnet: ${SUBNET_ID}"

# --- SSH keypair handling (RSA only for EC2) ---
echo "🔑 Preparing EC2 key pair (RSA only)…"
# Resolve an absolute path if you provided one
PUB_PATH=""
# Expand tilde to home directory
SSH_PUBLIC_KEY_PATH="${SSH_PUBLIC_KEY_PATH/#\~/${HOME}}"
if [ -n "${SSH_PUBLIC_KEY_PATH}" ] && [ -f "${SSH_PUBLIC_KEY_PATH}" ]; then
  PUB_PATH="$(cd "$(dirname "${SSH_PUBLIC_KEY_PATH}")" && pwd)/$(basename "${SSH_PUBLIC_KEY_PATH}")"
  # Ensure it's RSA; EC2 key pairs must be ssh-rsa
  if ! head -1 "${PUB_PATH}" | grep -q '^ssh-rsa '; then
    echo "⚠ The public key at ${PUB_PATH} is not RSA. EC2 requires an ssh-rsa key."
    echo "→ Generating a new RSA key in ~/.ssh/${KEY_PAIR_NAME}{,.pub}…"
    mkdir -p ~/.ssh
    if [ -e "${HOME}/.ssh/${KEY_PAIR_NAME}" ]; then
      read -r -p "File ~/.ssh/${KEY_PAIR_NAME} exists. Overwrite? [y/N] " ans
      [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborting to avoid overwrite."; exit 1; }
    fi
    ssh-keygen -t rsa -b 2048 -C "tljh-${ACCOUNT_ID}" -N "" -f "${HOME}/.ssh/${KEY_PAIR_NAME}"
    PUB_PATH="${HOME}/.ssh/${KEY_PAIR_NAME}.pub"
  fi
else
  echo "ℹ No usable public key found at \$SSH_PUBLIC_KEY_PATH; generating RSA key in ~/.ssh/${KEY_PAIR_NAME}{,.pub}…"
  mkdir -p ~/.ssh
  if [ -e "${HOME}/.ssh/${KEY_PAIR_NAME}" ]; then
    read -r -p "File ~/.ssh/${KEY_PAIR_NAME} exists. Overwrite? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborting to avoid overwrite."; exit 1; }
  fi
  ssh-keygen -t rsa -b 2048 -C "tljh-${ACCOUNT_ID}" -N "" -f "${HOME}/.ssh/${KEY_PAIR_NAME}"
  PUB_PATH="${HOME}/.ssh/${KEY_PAIR_NAME}.pub"
fi

# Import (or reuse) the EC2 key pair
echo "🔗 Ensuring AWS EC2 key pair '${KEY_PAIR_NAME}' exists…"
if aws ec2 describe-key-pairs --key-names "${KEY_PAIR_NAME}" --region "${AWS_REGION}" --profile "${AWS_PROFILE}" >/dev/null 2>&1; then
  echo "✔ Key pair '${KEY_PAIR_NAME}' already exists in AWS; reusing."
else
  aws ec2 import-key-pair \
    --key-name "${KEY_PAIR_NAME}" \
    --public-key-material "fileb://${PUB_PATH}" \
    --region "${AWS_REGION}" --profile "${AWS_PROFILE}"
  echo "✔ Imported key pair '${KEY_PAIR_NAME}' from ${PUB_PATH}"
fi

# Figure out the local private key path for the SSH instructions at the end
if [[ "${PUB_PATH}" == *"/.ssh/${KEY_PAIR_NAME}.pub" ]]; then
  LOCAL_PRIVATE_KEY="${HOME}/.ssh/${KEY_PAIR_NAME}"
else
  LOCAL_PRIVATE_KEY="${PUB_PATH%.pub}"
fi


# --- Security group (SSH from your IP; HTTP open) ---
echo "🛡 Creating/reusing security group: ${SECURITY_GROUP_NAME}"
SG_ID=$(aws ec2 describe-security-groups --region "${AWS_REGION}" --profile "${AWS_PROFILE}" \
  --filters "Name=group-name,Values=${SECURITY_GROUP_NAME}" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "${SG_ID}" ] || [ "${SG_ID}" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group --group-name "${SECURITY_GROUP_NAME}" \
    --description "TLJH security group" --vpc-id "${VPC_ID}" \
    --region "$AWS_REGION" --profile "$AWS_PROFILE" \
    --query 'GroupId' --output text)
  echo "✔ Created SG ${SG_ID}"
else
  echo "✔ Reusing SG ${SG_ID}"
fi
echo "➡ Authorizing ingress rules (idempotent)…"
aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" \
  --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$YOUR_IP_CIDR,Description='Your IP'}]" \
  --region "$AWS_REGION" --profile "$AWS_PROFILE" || echo "ℹ SSH rule already present"
aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" \
  --ip-permissions "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0,Description='HTTP'}]" \
  --region "$AWS_REGION" --profile "$AWS_PROFILE" || echo "ℹ HTTP rule already present"

if [ "${ENABLE_SSL}" = "true" ]; then
  aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" \
    --ip-permissions "IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0,Description='HTTPS'}]" \
    --region "$AWS_REGION" --profile "$AWS_PROFILE" || echo "ℹ HTTPS rule already present"
fi

# --- IAM: role + instance profile with narrow SSM read ---
ROLE_NAME="tljh-ec2-role"
PROFILE_NAME="tljh-ec2-instance-profile"
echo "👤 Ensuring EC2 role '$ROLE_NAME' exists…"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'
  echo "✔ Created role"
else
  echo "✔ Role exists"
fi
echo "➡ Attaching AmazonSSMManagedInstanceCore (for SSM agent)…"
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore" || true

echo "➡ Adding inline policy to read ${GITHUB_PAT_SSM_PARAMETER_NAME}…"
cat > /tmp/allow-read-github-pat.json <<POL
{
  "Version":"2012-10-17",
  "Statement":[
    {
      "Sid":"ReadGithubPAT",
      "Effect":"Allow",
      "Action":["ssm:GetParameter"],
      "Resource":"arn:aws:ssm:${AWS_REGION}:${ACCOUNT_ID}:parameter${GITHUB_PAT_SSM_PARAMETER_NAME}"
    },
    {
      "Sid":"AllowDecryptViaSSM",
      "Effect":"Allow",
      "Action":["kms:Decrypt"],
      "Resource":"*",
      "Condition":{"StringEquals":{"kms:ViaService":"ssm.${AWS_REGION}.amazonaws.com"}}
    }
  ]
}
POL
aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name "AllowReadGithubPAT" \
  --policy-document file:///tmp/allow-read-github-pat.json

echo "👤 Ensuring instance profile '$PROFILE_NAME' exists…"
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME"
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
  echo "✔ Created instance profile and attached role"
else
  echo "✔ Instance profile exists"
fi

# --- SSM parameter: ensure PAT exists; prompt if missing ---
echo "🔒 Checking for SSM parameter ${GITHUB_PAT_SSM_PARAMETER_NAME} in $AWS_REGION …"
if aws ssm get-parameter --name "${GITHUB_PAT_SSM_PARAMETER_NAME}" --region "$AWS_REGION" --profile "$AWS_PROFILE" >/dev/null 2>&1; then
  echo "✔ Found ${GITHUB_PAT_SSM_PARAMETER_NAME}"
else
  echo "⚠️  ${GITHUB_PAT_SSM_PARAMETER_NAME} not found."
  read -r -p "Paste your GitHub PAT here (will be stored in SSM): " GH_PAT_INPUT
  echo "➡ Writing as SecureString…"
  if aws ssm put-parameter --name "${GITHUB_PAT_SSM_PARAMETER_NAME}" \
      --value "$GH_PAT_INPUT" --type SecureString --overwrite \
      --region "$AWS_REGION" --profile "$AWS_PROFILE"; then
    echo "✔ Stored PAT as SecureString"
  else
    echo "⚠️  SecureString failed (likely KMS restricted). Storing as String (less ideal)…"
    aws ssm put-parameter --name "${GITHUB_PAT_SSM_PARAMETER_NAME}" \
      --value "$GH_PAT_INPUT" --type String --overwrite \
      --region "$AWS_REGION" --profile "$AWS_PROFILE"
    echo "✔ Stored PAT as String"
  fi
fi

# --- Find Ubuntu 22.04 AMI (SSM public param, fallback to latest Canonical) ---
echo "🖼  Resolving Ubuntu 22.04 AMI in $AWS_REGION …"
SSM_PARAM="/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
if AMI_ID=$(aws ssm get-parameter --name "$SSM_PARAM" \
      --query 'Parameter.Value' --output text \
      --region "$AWS_REGION" --profile "$AWS_PROFILE" 2>/dev/null); then
  echo "✔ AMI via SSM: $AMI_ID"
else
  echo "ℹ SSM param missing; falling back to describe-images…"
  AMI_ID=$(aws ec2 describe-images --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=root-device-type,Values=ebs" "Name=virtualization-type,Values=hvm" \
    --query 'Images|sort_by(@,&CreationDate)[-1].ImageId' \
    --output text --region "$AWS_REGION" --profile "$AWS_PROFILE")
  echo "✔ AMI via describe-images: $AMI_ID"
fi

# --- Build minimal bootstrap user data ---
USERDATA_FILE="/tmp/userdata.sh"
cat > "$USERDATA_FILE" <<'USERDATA'
#!/bin/bash
# Minimal bootstrap script for EC2 instance
# This script clones the repository and executes the main installation script
set -euxo pipefail

# Log everything to cloud-init-output.log
exec 1> >(logger -s -t user-data -p local6.info)
exec 2>&1

echo "========================================="
echo "🚀 Starting bootstrap at $(date)"
echo "========================================"

# Install minimal dependencies needed for bootstrap
export DEBIAN_FRONTEND=noninteractive
echo "[BOOTSTRAP] Installing base packages..."
apt-get update -y
apt-get install -y python3 curl git awscli dos2unix jq

# Create repo directory
echo "[BOOTSTRAP] Creating repository directory..."
mkdir -p /srv/classrepo
chown ubuntu:ubuntu /srv/classrepo

# Create Git credential helper that fetches PAT from SSM
echo "[BOOTSTRAP] Setting up Git credential helper..."
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
  token=$(aws ssm get-parameter --region AWS_REGION_PLACEHOLDER --name "GITHUB_PAT_SSM_PLACEHOLDER" --with-decryption --query Parameter.Value --output text)
  echo "username=x-access-token"
  echo "password=${token}"
fi
exit 0
HLP

# Replace placeholders in credential helper
sed -i "s/AWS_REGION_PLACEHOLDER/AWS_REGION_VALUE/" /usr/local/bin/gh-cred-helper
sed -i "s|GITHUB_PAT_SSM_PLACEHOLDER|GITHUB_PAT_SSM_VALUE|" /usr/local/bin/gh-cred-helper
chmod +x /usr/local/bin/gh-cred-helper

# Configure Git
echo "[BOOTSTRAP] Configuring Git..."
su - ubuntu -c "git config --global user.name 'GIT_USER_NAME_VALUE'"
su - ubuntu -c "git config --global user.email 'GIT_USER_EMAIL_VALUE'"
su - ubuntu -c "git config --global credential.helper '/usr/local/bin/gh-cred-helper'"

# Clone repository
echo "[BOOTSTRAP] Cloning repository..."
su - ubuntu -c "cd /srv && git clone https://github.com/GITHUB_OWNER_VALUE/GITHUB_REPO_VALUE.git classrepo"
su - ubuntu -c "cd /srv/classrepo && git checkout GITHUB_BRANCH_VALUE"

# Fix line endings if needed
if [ -f "/srv/classrepo/jupyterhub/install_on_ec2.sh" ]; then
    echo "[BOOTSTRAP] Fixing line endings in installation script..."
    dos2unix /srv/classrepo/jupyterhub/install_on_ec2.sh 2>/dev/null || \
        tr -d '\r' < /srv/classrepo/jupyterhub/install_on_ec2.sh > /tmp/install_fixed.sh && \
        mv /tmp/install_fixed.sh /srv/classrepo/jupyterhub/install_on_ec2.sh
    chmod +x /srv/classrepo/jupyterhub/install_on_ec2.sh
fi

# Export environment variables for the installation script
export JUPYTER_ADMIN_USER="JUPYTER_ADMIN_USER_VALUE"
export AWS_REGION="AWS_REGION_VALUE"
export GITHUB_PAT_SSM_PARAMETER_NAME="GITHUB_PAT_SSM_VALUE"
export GIT_USER_NAME="GIT_USER_NAME_VALUE"
export GIT_USER_EMAIL="GIT_USER_EMAIL_VALUE"
export GITHUB_OWNER="GITHUB_OWNER_VALUE"
export GITHUB_REPO="GITHUB_REPO_VALUE"
export GITHUB_BRANCH="GITHUB_BRANCH_VALUE"
export ENABLE_SSL="ENABLE_SSL_VALUE"
export USE_DOMAIN="USE_DOMAIN_VALUE"
export SSL_EMAIL="SSL_EMAIL_VALUE"
export CERT_BACKUP_BUCKET="CERT_BACKUP_BUCKET_VALUE"
export CERT_BACKUP_KEY="CERT_BACKUP_KEY_VALUE"

# Execute main installation script
echo "[BOOTSTRAP] Executing main installation script..."
echo "========================================="
if [ -f "/srv/classrepo/jupyterhub/install_on_ec2.sh" ]; then
    bash /srv/classrepo/jupyterhub/install_on_ec2.sh
else
    echo "[BOOTSTRAP] ERROR: Installation script not found at /srv/classrepo/jupyterhub/install_on_ec2.sh"
    echo "[BOOTSTRAP] Repository contents:"
    ls -la /srv/classrepo/
    exit 1
fi
USERDATA

# Replace placeholders with actual values (macOS compatible)
sed -i '' "s/AWS_REGION_VALUE/${AWS_REGION}/g" "$USERDATA_FILE"
sed -i '' "s|GITHUB_PAT_SSM_VALUE|${GITHUB_PAT_SSM_PARAMETER_NAME}|g" "$USERDATA_FILE"
sed -i '' "s/GIT_USER_NAME_VALUE/${GIT_USER_NAME}/g" "$USERDATA_FILE"
sed -i '' "s/GIT_USER_EMAIL_VALUE/${GIT_USER_EMAIL}/g" "$USERDATA_FILE"
sed -i '' "s/GITHUB_OWNER_VALUE/${GITHUB_OWNER}/g" "$USERDATA_FILE"
sed -i '' "s/GITHUB_REPO_VALUE/${GITHUB_REPO}/g" "$USERDATA_FILE"
sed -i '' "s/GITHUB_BRANCH_VALUE/${GITHUB_BRANCH}/g" "$USERDATA_FILE"
sed -i '' "s/JUPYTER_ADMIN_USER_VALUE/${JUPYTER_ADMIN_USER}/g" "$USERDATA_FILE"
sed -i '' "s/ENABLE_SSL_VALUE/${ENABLE_SSL}/g" "$USERDATA_FILE"
sed -i '' "s/USE_DOMAIN_VALUE/${USE_DOMAIN}/g" "$USERDATA_FILE"
sed -i '' "s/SSL_EMAIL_VALUE/${SSL_EMAIL}/g" "$USERDATA_FILE"
sed -i '' "s/CERT_BACKUP_BUCKET_VALUE/${CERT_BACKUP_BUCKET}/g" "$USERDATA_FILE"
sed -i '' "s|CERT_BACKUP_KEY_VALUE|${CERT_BACKUP_KEY}|g" "$USERDATA_FILE"

echo "📝 User data written to $USERDATA_FILE"

# --- Launch instance (or use existing in debug mode) ---
if [ "$DEBUG_MODE" = "true" ]; then
  echo "🐛 DEBUG MODE: Skipping instance creation"
  if [ -z "$EXISTING_INSTANCE_IP" ]; then
    echo "❌ DEBUG_MODE=true but EXISTING_INSTANCE_IP not set"
    echo "Usage for debug mode:"
    echo "  DEBUG_MODE=true EXISTING_INSTANCE_IP=1.2.3.4 ./setup_jupyterhub_aws.sh"
    exit 1
  fi
  PUBLIC_IP="$EXISTING_INSTANCE_IP"
  INSTANCE_ID="debug-mode-unknown"
  echo "🐛 Using existing instance: $PUBLIC_IP"
else
  echo "🚀 Launching EC2 instance…"
  RUN_JSON=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_PAIR_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --iam-instance-profile Name="$PROFILE_NAME" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_SIZE_GB},VolumeType=gp3,DeleteOnTermination=true}" \
    --user-data "$(cat "$USERDATA_FILE")" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME_TAG}},{Key=Project,Value=TLJH}]" \
    --region "$AWS_REGION" --profile "$AWS_PROFILE" \
    --count 1)

  INSTANCE_ID=$(echo "$RUN_JSON" | jq -r '.Instances[0].InstanceId')
  echo "✔ Instance: $INSTANCE_ID"
  echo "⏳ Waiting for 'running'…"
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --profile "$AWS_PROFILE"

  # Check for existing Elastic IP
  echo "🔗 Checking for existing Elastic IP..."
  ELASTIC_IP_INFO=$(aws ec2 describe-addresses \
    --filters "Name=tag:Name,Values=$ELASTIC_IP_NAME" \
    --query 'Addresses[0]' \
    --region "$AWS_REGION" --profile "$AWS_PROFILE" \
    --output json 2>/dev/null || echo "{}")
  
  ELASTIC_IP=$(echo "$ELASTIC_IP_INFO" | jq -r '.PublicIp // empty')
  ALLOCATION_ID=$(echo "$ELASTIC_IP_INFO" | jq -r '.AllocationId // empty')
  
  if [ -n "$ELASTIC_IP" ] && [ "$ELASTIC_IP" != "null" ]; then
    echo "✔ Found existing Elastic IP: $ELASTIC_IP"
    echo "🔗 Associating Elastic IP with instance..."
    aws ec2 associate-address \
      --instance-id "$INSTANCE_ID" \
      --allocation-id "$ALLOCATION_ID" \
      --region "$AWS_REGION" --profile "$AWS_PROFILE" >/dev/null
    PUBLIC_IP="$ELASTIC_IP"
    echo "✔ Associated Elastic IP: $PUBLIC_IP"
  else
    echo "ℹ No existing Elastic IP found, using auto-assigned IP"
    PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text \
      --region "$AWS_REGION" --profile "$AWS_PROFILE")
    echo "🌐 Public IP: $PUBLIC_IP"
    echo "💡 To use a static IP, create an Elastic IP with Name tag: $ELASTIC_IP_NAME"
  fi
fi

cat <<NEXT
====================================================================
Instance launched successfully!
====================================================================
Instance ID: ${INSTANCE_ID}
Public IP: ${PUBLIC_IP}
JupyterHub URL: http://${PUBLIC_IP} (ready in ~5-10 minutes)$(if [ -n "${USE_DOMAIN}" ] && [ -n "$ELASTIC_IP" ]; then echo "\nDomain URL: http://jfuse.shmtools.com (configure DNS A record)$(if [ "${ENABLE_SSL}" = "true" ]; then echo "\nHTTPS URL: https://jfuse.shmtools.com (with SSL certificate)"; fi)"; fi)

Waiting for SSH to become available...
====================================================================
NEXT

# Wait for SSH to be available (with timeout)
echo "⏳ Waiting for SSH service to start (may take 1-2 minutes)..."
RETRY_COUNT=0
MAX_RETRIES=30
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -i "${LOCAL_PRIVATE_KEY}" ${JUPYTER_ADMIN_USER}@${PUBLIC_IP} "echo 'SSH ready'" 2>/dev/null; then
    echo "✔ SSH connection established!"
    break
  fi
  echo -n "."
  sleep 5
  RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo ""
  echo "⚠️  SSH connection timeout. The instance may still be initializing."
  echo "Try connecting manually with:"
  echo "  ssh -i ${LOCAL_PRIVATE_KEY} ${JUPYTER_ADMIN_USER}@${PUBLIC_IP}"
  exit 1
fi

echo ""
echo "======================================================================"
echo "📊 MONITORING INSTALLATION PROGRESS"
echo "======================================================================"
echo "Connecting to cloud-init logs..."
echo "Press Ctrl+C to exit monitoring (installation will continue)"
echo ""
echo "Quick reference while monitoring:"
echo "  - JupyterHub URL: http://${PUBLIC_IP}$(if [ -n "${USE_DOMAIN}" ] && [ "$PUBLIC_IP" = "$ELASTIC_IP" ]; then echo "\n  - Domain URL: http://jfuse.shmtools.com$(if [ "${ENABLE_SSL}" = "true" ]; then echo "\n  - Secure URL: https://jfuse.shmtools.com"; fi)"; fi)"
echo "  - Admin username: ${JUPYTER_ADMIN_USER}"
echo "  - Users can create their own accounts (any username + password)"
echo "  - Repo location: /srv/classrepo"
echo "======================================================================"
echo ""

# Start monitoring cloud-init logs
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "${LOCAL_PRIVATE_KEY}" ${JUPYTER_ADMIN_USER}@${PUBLIC_IP} \
    "sudo tail -f /var/log/cloud-init-output.log"
