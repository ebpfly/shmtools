# SSL Certificate Management for JupyterHub Deployments

This document explains how SSL certificates are managed across JupyterHub deployments to avoid Let's Encrypt rate limiting.

## Problem

Let's Encrypt has a rate limit of 5 certificates per exact set of identifiers (domain names) in a 168-hour (7-day) window. When testing deployments, it's easy to hit this limit and be unable to get valid SSL certificates.

## Solution

The deployment system now automatically:

1. **Backs up certificates** to S3 after successful SSL setup
2. **Restores certificates** from S3 before attempting SSL setup on new deployments
3. **Reuses existing valid certificates** instead of requesting new ones

## How It Works

### Certificate Backup

After SSL setup completes, the deployment script:

1. Looks for certificate files in `/opt/tljh/state/acme.json` or `/opt/tljh/state/traefik/acme/acme.json`
2. **Validates that the file contains actual Let's Encrypt certificates** (not just account info or self-signed certificates)
3. If valid certificates are found, uploads the certificate file to S3: `s3://shmtools-deployment-bucket/certificates/acme.json`
4. If no valid certificates are found (e.g., due to rate limiting), skips backup and logs the reason

### Certificate Restore  

Before SSL setup begins, the deployment script:

1. Attempts to download the certificate backup from S3
2. **Validates that the backup contains actual Let's Encrypt certificates** (not just account info)
3. If valid certificates are found, places them in both expected locations:
   - `/opt/tljh/state/acme.json`
   - `/opt/tljh/state/traefik/acme/acme.json`
4. Sets proper permissions (600, root:root)
5. If no backup is found or backup contains no valid certificates, proceeds with normal certificate generation

## Configuration

The S3 bucket and key are configurable in the deployment script:

```bash
# Certificate backup configuration  
CERT_BACKUP_BUCKET="shmtools-deployment-bucket"  # S3 bucket for certificate backups
CERT_BACKUP_KEY="certificates/acme.json"         # S3 key for certificate backup
```

## Manual Certificate Management

### Backup Certificates from Existing Instance

Use the provided helper script:

```bash
# Backup certificates from a specific instance
./backup_certificates.sh 3.130.148.209

# Auto-detect running instance and backup
./backup_certificates.sh
```

### View Certificate Backup

```bash
# Download and examine certificate backup
aws s3 cp s3://shmtools-deployment-bucket/certificates/acme.json /tmp/acme.json
jq . /tmp/acme.json
```

### Manual Restore

If you need to manually restore certificates to an instance:

```bash
# Download certificate backup
aws s3 cp s3://shmtools-deployment-bucket/certificates/acme.json /tmp/acme.json

# Copy to instance
scp -i ~/.ssh/class-key-ssh-rsa /tmp/acme.json ubuntu@INSTANCE_IP:/tmp/

# Install on instance  
ssh -i ~/.ssh/class-key-ssh-rsa ubuntu@INSTANCE_IP "
sudo mkdir -p /opt/tljh/state/traefik/acme
sudo cp /tmp/acme.json /opt/tljh/state/acme.json
sudo cp /tmp/acme.json /opt/tljh/state/traefik/acme/acme.json
sudo chmod 600 /opt/tljh/state/acme.json /opt/tljh/state/traefik/acme/acme.json
sudo chown root:root /opt/tljh/state/acme.json /opt/tljh/state/traefik/acme/acme.json
sudo systemctl restart traefik jupyterhub
"
```

## Benefits

1. **Avoid Rate Limiting**: Reuse existing certificates instead of requesting new ones
2. **Faster Deployments**: Skip certificate generation when valid certificates exist  
3. **Reliable SSL**: No more "certificate limit exceeded" errors during testing
4. **Persistent Certificates**: Certificates survive instance termination and recreation

## Certificate Lifecycle

1. **First Deployment**: No backup exists, generates new certificate, backs up to S3
2. **Subsequent Deployments**: Restores certificate from S3, SSL works immediately
3. **Certificate Expiration**: Let's Encrypt auto-renewal still works, updated certificate gets backed up
4. **Rate Limit Hit**: Deployment still succeeds using backed-up certificate

## Troubleshooting

### Certificate Restore Failed

If certificate restore fails, check:

```bash
# Verify S3 backup exists
aws s3 ls s3://shmtools-deployment-bucket/certificates/

# Check S3 permissions
aws s3 cp s3://shmtools-deployment-bucket/certificates/acme.json /tmp/test.json

# Verify certificate format
jq . /tmp/test.json
```

### SSL Still Not Working

If SSL doesn't work after certificate restore:

1. Check Traefik logs: `sudo journalctl -u traefik -f`
2. Verify certificate placement: `sudo ls -la /opt/tljh/state/acme.json`
3. Restart services: `sudo systemctl restart traefik jupyterhub`
4. Check certificate expiration in the backup file

### Manual Certificate Generation

To force new certificate generation (ignoring backup):

1. Temporarily rename the S3 backup: `aws s3 mv s3://shmtools-deployment-bucket/certificates/acme.json s3://shmtools-deployment-bucket/certificates/acme.json.backup`
2. Run deployment (will generate new certificate)
3. Restore backup if needed: `aws s3 mv s3://shmtools-deployment-bucket/certificates/acme.json.backup s3://shmtools-deployment-bucket/certificates/acme.json`