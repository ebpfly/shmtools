#!/bin/bash

# Script to restart JupyterLab server for testing extension updates
# Usage: ./restart_jupyterlab.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔄 Restarting JupyterLab for extension testing...${NC}"

# Navigate to project root and activate venv
cd /Users/eric/repo/shm/
echo -e "${YELLOW}📁 Activating virtual environment...${NC}"
source venv/bin/activate

# Kill any existing JupyterLab processes
echo -e "${YELLOW}🛑 Stopping existing JupyterLab processes...${NC}"
pkill -f "jupyter-lab" 2>/dev/null || true
sleep 2

# Build the extension (critical 3-step process from CLAUDE.md)
echo -e "${YELLOW}🔨 Building JupyterLab extension...${NC}"
cd shm_function_selector/

echo -e "  Step 1/3: Compiling TypeScript..."
npm run build:lib

echo -e "  Step 2/3: Building extension..."
npm run build:labextension:dev

echo -e "  Step 3/3: Integrating into JupyterLab..."
cd .. && jupyter lab build

# Start JupyterLab in background
echo -e "${YELLOW}🚀 Starting JupyterLab in background...${NC}"
nohup jupyter lab --no-browser > jupyterlab.log 2>&1 &
JUPYTER_PID=$!

# Wait for server to start
echo -e "${YELLOW}⏳ Waiting for server to start...${NC}"
sleep 3

# Check if server is running
if kill -0 $JUPYTER_PID 2>/dev/null; then
    echo -e "${GREEN}✅ JupyterLab started successfully (PID: $JUPYTER_PID)${NC}"
    echo -e "${GREEN}📋 Server log: tail -f jupyterlab.log${NC}"
    echo -e "${GREEN}🛑 To stop: pkill -f jupyter-lab${NC}"
    
    # Open browser automatically
    echo -e "${YELLOW}🌐 Opening browser...${NC}"
    sleep 2  # Give server a moment to fully initialize
    open http://localhost:8888
    
else
    echo -e "${RED}❌ Failed to start JupyterLab${NC}"
    cat jupyterlab.log
    exit 1
fi