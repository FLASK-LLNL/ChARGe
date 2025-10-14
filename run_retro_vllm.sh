#!/bin/bash

# Configuration
PRODUCT_SMILES="c1cc(ccc1N)O"  # Default from main.py
#export VLLM_URL="http://localhost:8001/v1"
#export VLLM_MODEL="/p/vast1/flask/models/Meta-Llama-3.1-8B-Instruct"
#export VLLM_MODEL="/p/vast1/flask/team/tim/models/sft/llama-8b/pistachio/applications/v7-full-fwd/e5.lr1e-5.b128/checkpoint-78900"
#export VLLM_MODEL="/p/vast1/flask/team/tim/models/sft/llama-8b/pistachio/applications/v8-full-retro/e2.lr1e-5.b128/checkpoint-44000"

# Aniruddha hosting model with vLLM
## GPT-OSS-20b
export VLLM_URL="http://192.168.128.31:8000/v1"
export VLLM_MODEL="/p/vast1/flask/models/gpt-oss-20b"
## GPT-OSS-120b
#export VLLM_URL="http://192.168.128.32:8000/v1"
#export VLLM_MODEL="/p/vast1/flask/models/gpt-oss-120b"

# Reasoning level for GPT-OSS
OSS_REASONING="medium" # Options: ["low", "medium", "high"]


cd experiments/Retrosynthesis

python main.py \
    --client autogen \
    --backend vllm \
    --reasoning-effort $OSS_REASONING \
    --server-path reaction_server.py \
    --user-prompt "Generate a new reaction SMARTS and reactants for the product ${PRODUCT_SMILES}"
