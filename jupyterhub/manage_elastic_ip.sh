#!/usr/bin/env bash
# Elastic IP management helper for SHMTools deployment
# Usage: ./manage_elastic_ip.sh <command>

set -Eeuo pipefail

AWS_PROFILE="default"
AWS_REGION="us-east-2"
ELASTIC_IP_NAME="shmtools-static-ip"

case "${1:-help}" in
  "create")
    echo "🔗 Creating new Elastic IP..."
    RESULT=$(aws ec2 allocate-address --domain vpc --region "$AWS_REGION" --profile "$AWS_PROFILE")
    ALLOCATION_ID=$(echo "$RESULT" | jq -r '.AllocationId')
    PUBLIC_IP=$(echo "$RESULT" | jq -r '.PublicIp')
    
    echo "🏷  Tagging Elastic IP..."
    aws ec2 create-tags --resources "$ALLOCATION_ID" \
      --tags Key=Name,Value="$ELASTIC_IP_NAME" Key=Project,Value=SHMTools \
      --region "$AWS_REGION" --profile "$AWS_PROFILE"
    
    echo "✅ Created Elastic IP: $PUBLIC_IP ($ALLOCATION_ID)"
    echo "💰 Cost: ~\$3-5/month when associated, \$3.65/month when unassociated"
    ;;
    
  "status")
    echo "🔍 Checking Elastic IP status..."
    ELASTIC_IP_INFO=$(aws ec2 describe-addresses \
      --filters "Name=tag:Name,Values=$ELASTIC_IP_NAME" \
      --region "$AWS_REGION" --profile "$AWS_PROFILE" \
      --output json 2>/dev/null || echo "[]")
    
    if [ "$(echo "$ELASTIC_IP_INFO" | jq '.Addresses | length')" = "0" ]; then
      echo "❌ No Elastic IP found with name: $ELASTIC_IP_NAME"
      echo "💡 Run: ./manage_elastic_ip.sh create"
    else
      ELASTIC_IP=$(echo "$ELASTIC_IP_INFO" | jq -r '.Addresses[0].PublicIp')
      ALLOCATION_ID=$(echo "$ELASTIC_IP_INFO" | jq -r '.Addresses[0].AllocationId')
      INSTANCE_ID=$(echo "$ELASTIC_IP_INFO" | jq -r '.Addresses[0].InstanceId // "unassociated"')
      
      echo "✅ Elastic IP: $ELASTIC_IP ($ALLOCATION_ID)"
      echo "🖥  Instance: $INSTANCE_ID"
      
      if [ "$INSTANCE_ID" != "unassociated" ] && [ "$INSTANCE_ID" != "null" ]; then
        echo "💚 Status: Associated"
      else
        echo "💛 Status: Unassociated (costing extra \$3.65/month)"
      fi
    fi
    ;;
    
  "release")
    echo "⚠️  This will permanently delete the Elastic IP!"
    echo "🌐 Domain DNS records pointing to this IP will break."
    read -p "Are you sure? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
      echo "Cancelled."
      exit 0
    fi
    
    ALLOCATION_ID=$(aws ec2 describe-addresses \
      --filters "Name=tag:Name,Values=$ELASTIC_IP_NAME" \
      --query 'Addresses[0].AllocationId' --output text \
      --region "$AWS_REGION" --profile "$AWS_PROFILE")
    
    if [ "$ALLOCATION_ID" = "None" ] || [ -z "$ALLOCATION_ID" ]; then
      echo "❌ No Elastic IP found to release"
      exit 1
    fi
    
    aws ec2 release-address --allocation-id "$ALLOCATION_ID" \
      --region "$AWS_REGION" --profile "$AWS_PROFILE"
    echo "✅ Released Elastic IP: $ALLOCATION_ID"
    ;;
    
  "dns")
    ELASTIC_IP=$(aws ec2 describe-addresses \
      --filters "Name=tag:Name,Values=$ELASTIC_IP_NAME" \
      --query 'Addresses[0].PublicIp' --output text \
      --region "$AWS_REGION" --profile "$AWS_PROFILE" 2>/dev/null || echo "None")
    
    if [ "$ELASTIC_IP" = "None" ] || [ -z "$ELASTIC_IP" ]; then
      echo "❌ No Elastic IP found"
      exit 1
    fi
    
    echo "🌐 DNS Configuration for jfuse.shmtools.com"
    echo "==========================================="
    echo "Type: A"
    echo "Name: jfuse"
    echo "Value: $ELASTIC_IP"
    echo "TTL: 300"
    echo ""
    echo "Alternative with www subdomain:"
    echo "Type: A"
    echo "Name: www.jfuse"
    echo "Value: $ELASTIC_IP"
    echo "TTL: 300"
    ;;
    
  *)
    echo "🚀 SHMTools Elastic IP Manager"
    echo "=============================="
    echo "Commands:"
    echo "  create   - Create new Elastic IP"
    echo "  status   - Check current Elastic IP status"
    echo "  release  - Delete Elastic IP (DANGEROUS)"
    echo "  dns      - Show DNS configuration"
    echo ""
    echo "Examples:"
    echo "  ./manage_elastic_ip.sh create"
    echo "  ./manage_elastic_ip.sh status"
    echo "  ./manage_elastic_ip.sh dns"
    ;;
esac