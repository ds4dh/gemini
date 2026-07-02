#!/bin/bash
set -e

echo ">>> Updating system..."
apt-get update

echo ">>> Installing system dependencies..."
apt-get install -y \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ninja-build \
    screen

echo ">>> Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to the path for this session
export PATH="/root/.local/bin:$PATH"

echo ">>> Installing virutal environment..."
# Switching to data, the persistent folder
cd /data/llm-project || echo "Warning: Folder /data/llm-project cannot be found, staying in $(pwd)"

if [ ! -d ".venv" ]; then
    uv venv
    echo "Environment created in .venv."
else
    echo "Environment .venv already exists."
fi

# Activate your environment
source /data/gemini/.venv/bin/activate

# Build dependencies
uv sync --no-install-project