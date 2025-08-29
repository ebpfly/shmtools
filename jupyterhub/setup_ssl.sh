#!/bin/bash
# Standalone SSL setup script for JupyterHub with TLJH and Let's Encrypt
# This script can be run independently to set up SSL on an existing TLJH instance

set -euo pipefail

# Configuration
USE_DOMAIN="${1:-jfuse.shmtools.com}"
SSL_EMAIL="${2:-ericbflynn@gmail.com}"

# Logging functions
log_step() {
    echo "[SSL-SETUP $(date '+%Y-%m-%d %H:%M:%S')] ${1:-}"
}

log_success() {
    echo "[SSL-SUCCESS $(date '+%Y-%m-%d %H:%M:%S')] ✅ ${1:-}"
}

log_error() {
    echo "[SSL-ERROR $(date '+%Y-%m-%d %H:%M:%S')] ❌ ${1:-}"
}

log_warning() {
    echo "[SSL-WARNING $(date '+%Y-%m-%d %H:%M:%S')] ⚠️ ${1:-}"
}

echo "========================================="
echo "🔒 SSL/HTTPS SETUP FOR JUPYTERHUB"
echo "========================================="
log_step "Domain: $USE_DOMAIN"
log_step "Email: $SSL_EMAIL"
echo "========================================="

# Check if we're running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root or with sudo"
    exit 1
fi

# Check if TLJH is installed
if ! command -v tljh-config >/dev/null 2>&1; then
    log_error "TLJH (The Littlest JupyterHub) is not installed"
    exit 1
fi

# Check if domain resolves to this server
log_step "🔍 Checking DNS resolution for $USE_DOMAIN..."
DOMAIN_IP=$(dig +short $USE_DOMAIN 2>/dev/null || echo "")
SERVER_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "")

if [ -z "$DOMAIN_IP" ]; then
    log_warning "Cannot resolve $USE_DOMAIN via DNS"
elif [ "$DOMAIN_IP" != "$SERVER_IP" ]; then
    log_warning "Domain $USE_DOMAIN resolves to $DOMAIN_IP but server IP is $SERVER_IP"
    log_warning "SSL certificate issuance may fail if DNS is not properly configured"
else
    log_success "DNS resolution looks good: $USE_DOMAIN -> $DOMAIN_IP"
fi

# Stop services that might interfere
log_step "🛑 Stopping services to avoid conflicts..."
systemctl stop traefik 2>/dev/null || true
systemctl stop jupyterhub 2>/dev/null || true

# Configure HTTPS using TLJH's built-in support
log_step "🔧 Configuring HTTPS settings..."
tljh-config set https.enabled true
tljh-config set https.letsencrypt.email "$SSL_EMAIL"

# Clear any existing domains first
tljh-config unset https.letsencrypt.domains 2>/dev/null || true
tljh-config add-item https.letsencrypt.domains "$USE_DOMAIN"

# Show configuration for verification
log_step "📋 Current TLJH configuration:"
tljh-config show

# Apply configuration changes - first general reload
log_step "🔄 Applying TLJH configuration changes..."
tljh-config reload

# Wait for initial configuration to settle
log_step "⏳ Waiting for initial configuration to settle..."
sleep 10

# Now reload the proxy specifically to ensure HTTPS is enabled
log_step "🔄 Reloading proxy with HTTPS configuration..."
tljh-config reload proxy

# Wait for proxy reload to complete
log_step "⏳ Waiting for proxy reload to complete..."
sleep 15

# Verify HTTPS is configured, if not force regeneration
if ! grep -q ":443" /opt/tljh/state/traefik.toml 2>/dev/null; then
    log_warning "HTTPS not configured in traefik, forcing regeneration..."
    
    # Run the installer to ensure proper configuration
    log_step "Running TLJH installer to ensure proper configuration..."
    /opt/tljh/hub/bin/python3 -m tljh.installer --admin ubuntu
    
    # Reload proxy again
    log_step "Reloading proxy after installer..."
    tljh-config reload proxy
    sleep 10
fi

# Final restart to ensure everything is working
log_step "🔄 Final restart of services..."
systemctl restart traefik
sleep 5
systemctl restart jupyterhub
sleep 10

# Verify HTTPS is in the config
if ! grep -q "https" /opt/tljh/state/traefik.toml; then
    log_error "HTTPS entrypoint still not found in Traefik config after regeneration"
    log_step "Attempting alternative approach..."
    systemctl stop traefik jupyterhub
    sleep 5
    /opt/tljh/hub/bin/python -c "from tljh import traefik; traefik.ensure_traefik_config('/opt/tljh/state')"
    systemctl start traefik jupyterhub
    sleep 15
fi

# Check service status
log_step "🔍 Checking service status..."
if systemctl is-active --quiet traefik; then
    log_success "Traefik is running"
else
    log_error "Traefik is not running"
    systemctl status traefik --no-pager -l
fi

if systemctl is-active --quiet jupyterhub; then
    log_success "JupyterHub is running"
else
    log_error "JupyterHub is not running"
    systemctl status jupyterhub --no-pager -l
fi

# Check if certificate directory exists
log_step "🔍 Checking certificate status..."
if [ -d "/opt/tljh/state/traefik/acme" ]; then
    log_step "Certificate directory exists:"
    ls -la /opt/tljh/state/traefik/acme/
    
    if [ -f "/opt/tljh/state/traefik/acme/acme.json" ]; then
        log_step "ACME configuration file exists"
        # Check if it contains our domain
        if grep -q "$USE_DOMAIN" /opt/tljh/state/traefik/acme/acme.json 2>/dev/null; then
            log_success "Domain $USE_DOMAIN found in ACME configuration"
        else
            log_warning "Domain $USE_DOMAIN not found in ACME configuration"
        fi
    else
        log_warning "ACME configuration file not found"
    fi
else
    log_warning "Certificate directory does not exist yet"
fi

# Test HTTPS connectivity
log_step "🧪 Testing HTTPS connectivity..."
HTTPS_RETRY=0
HTTPS_MAX_RETRIES=24  # 2 minutes with 5-second intervals

while [ $HTTPS_RETRY -lt $HTTPS_MAX_RETRIES ]; do
    if curl -s --connect-timeout 10 --max-time 15 "https://$USE_DOMAIN" >/dev/null 2>&1; then
        log_success "HTTPS is responding at https://$USE_DOMAIN"
        
        # Test certificate validity
        log_step "🔐 Checking SSL certificate..."
        CERT_INFO=$(echo | openssl s_client -servername "$USE_DOMAIN" -connect "$USE_DOMAIN:443" 2>/dev/null | openssl x509 -noout -subject -dates 2>/dev/null || echo "Certificate check failed")
        echo "$CERT_INFO"
        
        # Final test with full verification
        if curl -s --max-time 10 "https://$USE_DOMAIN" >/dev/null 2>&1; then
            log_success "SSL certificate is valid and working!"
            echo "========================================="
            log_success "🎉 SSL SETUP COMPLETE!"
            echo "========================================="
            log_success "🌐 JupyterHub is now accessible at: https://$USE_DOMAIN"
            echo "========================================="
            exit 0
        else
            log_warning "HTTPS responds but certificate may have issues"
        fi
        break
    fi
    
    log_step "⏳ Waiting for HTTPS... (attempt $((HTTPS_RETRY + 1))/$HTTPS_MAX_RETRIES)"
    
    # Show some debug info every few attempts
    if [ $((HTTPS_RETRY % 6)) -eq 5 ]; then
        log_step "Debug info - Traefik logs (last 10 lines):"
        journalctl -u traefik --no-pager -n 10 | tail -10
    fi
    
    sleep 5
    HTTPS_RETRY=$((HTTPS_RETRY + 1))
done

# If we get here, HTTPS setup failed
log_error "HTTPS setup failed after $HTTPS_MAX_RETRIES attempts"
log_step "Troubleshooting information:"
log_step "1. Check Traefik logs: sudo journalctl -u traefik -f"
log_step "2. Check JupyterHub logs: sudo journalctl -u jupyterhub -f"
log_step "3. Verify DNS: dig $USE_DOMAIN"
log_step "4. Check TLJH config: sudo tljh-config show"

exit 1