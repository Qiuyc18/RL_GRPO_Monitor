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

case $PLATFORM in
    "nvidia")
        uv pip install torch torchvision torchaudio
        ;;
    "amd")
        # for MI250X, recommend using ROCm 6.0 or 6.2
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
        ;;
    "cpu")
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

uv pip install -r requirements.txt
uv pip install -e .

echo "Installation complete for $PLATFORM platform."