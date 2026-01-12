#!/bin/bash

if lspci | grep -iq "nvidia"; then
    PLATFORM="nvidia"
    echo "Detected NVIDIA GPU"
elif lspci | grep -iq "amd"; then
    PLATFORM="amd"
    echo "Detected AMD GPU"
else
    echo "No supported GPU detected, defaulting to CPU-only"
    PLATFORM="cpu"
fi

echo "Init environment using uv..."
uv venv --python 3.12
source .venv/bin/activate

REQUIREMENTS_FILE="requirements/requirements-${PLATFORM}.txt"

echo "Installing dependencies from $REQUIREMENTS_FILE using uv..."
uv pip install -r "$REQUIREMENTS_FILE"

echo "Installation complete for $PLATFORM platform."