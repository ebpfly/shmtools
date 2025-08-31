#!/usr/bin/env python3
"""
Post-installation configuration for TLJH to ensure example notebooks are available.
This script configures JupyterHub spawner to copy examples on first login.
"""

import os
import shutil
from pathlib import Path

def post_spawn_hook(spawner):
    """Hook to run after a user's server is spawned.
    Ensures example notebooks are available for the user.
    """
    username = spawner.user.name
    user_home = Path(f"/home/jupyter-{username}")
    
    # Create symlink to shared examples if it doesn't exist
    shared_link = user_home / "shared-shmtools-examples"
    if not shared_link.exists():
        try:
            shared_link.symlink_to("/srv/data/shmtools-examples")
            os.system(f"chown -h jupyter-{username}:jupyter-{username} {shared_link}")
            spawner.log.info(f"Created shared examples symlink for {username}")
        except Exception as e:
            spawner.log.error(f"Failed to create symlink for {username}: {e}")
    
    # Copy personal examples if they don't exist
    personal_examples = user_home / "my-shmtools-examples"
    if not personal_examples.exists():
        try:
            shutil.copytree("/srv/data/shmtools-examples", personal_examples)
            os.system(f"chown -R jupyter-{username}:jupyter-{username} {personal_examples}")
            spawner.log.info(f"Copied personal examples for {username}")
        except Exception as e:
            spawner.log.error(f"Failed to copy examples for {username}: {e}")

# Configure the spawner hook
c.Spawner.post_start_hook = post_spawn_hook