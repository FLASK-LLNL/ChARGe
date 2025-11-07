#!/bin/bash

# Configuration
PRODUCT_SMILES="c1cc(ccc1N)O"  # Default from main.py
LC_NODE="34"

# vLLM hosted model
## GPT-OSS-20b
#export VLLM_URL="http://192.168.128.${LC_NODE}:8010/v1"
#export VLLM_MODEL="/p/vast1/flask/models/marathe1/gpt-oss-20b"

## GPT-OSS-120b
export VLLM_URL="http://192.168.128.${LC_NODE}:8011/v1"
export VLLM_MODEL="/p/vast1/flask/models/marathe1/gpt-oss-120b"

# Reasoning level for GPT-OSS
export OSS_REASONING="medium" # Options: ["low", "medium", "high"]

cd examples/Retrosynthesis

python main.py \
    --client autogen \
    --backend vllm \
    --user-prompt "Generate a new reaction SMARTS and reactants for the product ${PRODUCT_SMILES}"
#    --server-path reaction_server.py \
