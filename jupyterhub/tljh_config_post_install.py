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
    
    # Copy examples if they don't exist
    examples_dir = user_home / "shmtools-examples"
    if not examples_dir.exists():
        try:
            # Copy from the repository location
            source_dir = Path("/srv/classrepo/examples/notebooks")
            if source_dir.exists():
                shutil.copytree(source_dir, examples_dir)
                os.system(f"chown -R jupyter-{username}:jupyter-{username} {examples_dir}")
                spawner.log.info(f"Copied example notebooks for {username}")
            else:
                spawner.log.warning(f"Source notebooks not found at {source_dir}")
        except Exception as e:
            spawner.log.error(f"Failed to copy examples for {username}: {e}")

# Configure the spawner hook
c.Spawner.post_start_hook = post_spawn_hook