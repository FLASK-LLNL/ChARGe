#!/bin/bash

set -e  # Exit on error

echo "================================================================="
echo "Setting up ChARGe with GPT-OSS environment for MXFP4 quantization"
echo "================================================================="

echo "Creating conda environment..."
conda create -n charge_gptoss python=3.10 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate charge_gptoss

echo "Installing PyTorch..."
pip install torch torchvision

echo "Installing Triton..."
pip install triton

echo "Installing transformers and dependencies..."
pip install transformers accelerate bitsandbytes

echo "Installing local ChARGe (this should work from ChARGe root directory)..."
pip install -e .

echo "Installing AutoGen..."
pip install autogen-agentchat autogen-core autogen-ext

echo "Installing MCP..."
pip install mcp

echo "Installing RDKit..."
pip install rdkit

# Install kernels (needed for MXFP4)
echo "Installing kernels for MXFP4 support..."
pip install kernels

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add the following to your ~/.bashrc:"
echo ""
echo "# HuggingFace cache location (avoid quota issues in LC default directory)"
echo "export HF_HOME=/p/vast1/$USER/hf_cache"
echo "export HF_DATASETS_CACHE=/p/vast1/$USER/hf_cache"
echo "export TRANSFORMERS_CACHE=/p/vast1/$USER/hf_cache"
echo "export HUGGINGFACE_HUB_CACHE=/p/vast1/$USER/hf_cache"
echo ""
echo "2. Run: source ~/.bashrc"
echo "3. Create cache directory: mkdir -p /p/vast1/$USER/hf_cache"
echo "4. Activate environment: conda activate charge_gptoss"
echo "5. Run experiments!"
echo ""
