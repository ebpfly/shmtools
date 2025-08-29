#!/bin/bash
set -euo pipefail

# Simple AWS JupyterHub deployment with minimal user data
echo "🚀 Setting up JupyterHub on AWS (Simple Version)"

# Configuration
AWS_REGION="${AWS_REGION:-us-east-2}"
AWS_PROFILE="${AWS_PROFILE:-default}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
JUPYTER_ADMIN_USER="${JUPYTER_ADMIN_USER:-ubuntu}"
GITHUB_OWNER="${GITHUB_OWNER:-ersimpson}"
GITHUB_REPO="${GITHUB_REPO:-shm}"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
KEY_PAIR_NAME="class-key-ssh-rsa"
INSTANCE_NAME_TAG="tljh-class-server"

# Check AWS CLI
if ! aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null; then
    echo "❌ AWS CLI not configured properly"
    exit 1
fi

# Get VPC info
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text --region "$AWS_REGION")
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0].SubnetId' --output text --region "$AWS_REGION")

# Get security group
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=tljh-sg" --query 'SecurityGroups[0].GroupId' --output text --region "$AWS_REGION")

# Get AMI
AMI_ID=$(aws ec2 describe-images --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=root-device-type,Values=ebs" "Name=virtualization-type,Values=hvm" \
  --query 'Images|sort_by(@,&CreationDate)[-1].ImageId' \
  --output text --region "$AWS_REGION")

# Create minimal user data
cat > /tmp/userdata.sh <<'USERDATA'
#!/bin/bash
set -euxo pipefail
exec 1> >(logger -s -t user-data -p local6.info)
exec 2>&1

echo "🚀 Starting JupyterHub bootstrap"
apt-get update -y
apt-get install -y python3 curl git awscli dos2unix

# Get instance info
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
AWS_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Get config from tags
JUPYTER_ADMIN_USER=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=JupyterAdminUser" --region $AWS_REGION --query 'Tags[0].Value' --output text)
GITHUB_OWNER=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=GitHubOwner" --region $AWS_REGION --query 'Tags[0].Value' --output text)
GITHUB_REPO=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=GitHubRepo" --region $AWS_REGION --query 'Tags[0].Value' --output text)
GITHUB_BRANCH=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=GitHubBranch" --region $AWS_REGION --query 'Tags[0].Value' --output text)

echo "📥 Downloading full setup script"
curl -fsSL "https://raw.githubusercontent.com/$GITHUB_OWNER/$GITHUB_REPO/$GITHUB_BRANCH/jupyterhub/full_setup.sh" -o /tmp/full_setup.sh
dos2unix /tmp/full_setup.sh || tr -d '\r' < /tmp/full_setup.sh > /tmp/full_setup_fixed.sh && mv /tmp/full_setup_fixed.sh /tmp/full_setup.sh
chmod +x /tmp/full_setup.sh

echo "🚀 Running full setup"
JUPYTER_ADMIN_USER="$JUPYTER_ADMIN_USER" GITHUB_OWNER="$GITHUB_OWNER" GITHUB_REPO="$GITHUB_REPO" GITHUB_BRANCH="$GITHUB_BRANCH" AWS_REGION="$AWS_REGION" bash /tmp/full_setup.sh

echo "🎉 Bootstrap complete"
USERDATA

echo "📝 User data size: $(wc -c < /tmp/userdata.sh) bytes"

# Launch instance
echo "🚀 Launching EC2 instance"
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_PAIR_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --user-data "file:///tmp/userdata.sh" \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=$INSTANCE_NAME_TAG},
        {Key=Project,Value=TLJH},
        {Key=JupyterAdminUser,Value=$JUPYTER_ADMIN_USER},
        {Key=GitHubOwner,Value=$GITHUB_OWNER},
        {Key=GitHubRepo,Value=$GITHUB_REPO},
        {Key=GitHubBranch,Value=$GITHUB_BRANCH}
    ]" \
    --region "$AWS_REGION" --profile "$AWS_PROFILE" \
    --query 'Instances[0].InstanceId' --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text \
  --region "$AWS_REGION")

echo "🎉 JupyterHub deployment started!"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "JupyterHub will be ready at: http://$PUBLIC_IP (in ~5-10 minutes)"
echo "Admin user: $JUPYTER_ADMIN_USER"