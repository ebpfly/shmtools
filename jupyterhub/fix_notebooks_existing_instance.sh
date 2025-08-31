#!/bin/bash
# Fix script to add example notebooks to existing TLJH instance
# Run this on the EC2 instance as ubuntu user

set -euo pipefail

echo "======================================================================"
echo "🔧 FIXING EXAMPLE NOTEBOOKS ON EXISTING INSTANCE"
echo "======================================================================"

# Copy notebooks to /etc/skel for future users
echo "📚 Setting up notebooks for future users..."
sudo mkdir -p /etc/skel/shmtools-examples
sudo cp -r /srv/classrepo/examples/notebooks/* /etc/skel/shmtools-examples/
sudo chown -R root:root /etc/skel/shmtools-examples
sudo chmod -R 755 /etc/skel/shmtools-examples
echo "✅ Future users will get notebooks in ~/shmtools-examples/"

# Copy notebooks to ALL existing jupyter-* users
echo "📚 Copying notebooks to existing users..."
for USER_HOME in /home/jupyter-*; do
    if [ -d "$USER_HOME" ]; then
        USERNAME=$(basename "$USER_HOME")
        if [ ! -d "$USER_HOME/shmtools-examples" ]; then
            echo "  → Copying notebooks for $USERNAME..."
            sudo cp -r /srv/classrepo/examples/notebooks "$USER_HOME/shmtools-examples"
            sudo chown -R "$USERNAME:$USERNAME" "$USER_HOME/shmtools-examples"
            echo "  ✅ Copied notebooks to $USER_HOME/shmtools-examples/"
        else
            echo "  ℹ️  Notebooks already exist for $USERNAME, skipping..."
        fi
    fi
done

# Install the post-spawn hook for extra reliability
echo "🔧 Installing post-spawn hook..."
if [ -f "/srv/classrepo/jupyterhub/tljh_config_post_install.py" ]; then
    sudo cp /srv/classrepo/jupyterhub/tljh_config_post_install.py /opt/tljh/config/jupyterhub_config.d/
    echo "✅ Post-spawn hook installed"
    
    # Reload JupyterHub configuration
    echo "🔄 Reloading JupyterHub configuration..."
    sudo tljh-config reload
    echo "✅ Configuration reloaded"
else
    echo "⚠️  Post-spawn hook file not found, skipping..."
fi

echo ""
echo "======================================================================"
echo "✅ FIX COMPLETE!"
echo "======================================================================"
echo ""
echo "Results:"
echo "  • Existing users: Notebooks copied to ~/shmtools-examples/"
echo "  • Future users: Will automatically get notebooks"
echo "  • Users may need to refresh their JupyterLab page to see changes"
echo ""
echo "Users affected:"
ls -d /home/jupyter-* 2>/dev/null | xargs -n1 basename || echo "  No jupyter users found yet"
echo ""
echo "======================================================================"