#!/bin/bash

# Configuration
PRODUCT_SMILES="c1cc(ccc1N)O"  # Default from main.py
#export VLLM_URL="http://localhost:8001/v1"
#export VLLM_MODEL="/p/vast1/flask/models/Meta-Llama-3.1-8B-Instruct"
#export VLLM_MODEL="/p/vast1/flask/team/tim/models/sft/llama-8b/pistachio/applications/v7-full-fwd/e5.lr1e-5.b128/checkpoint-78900"
#export VLLM_MODEL="/p/vast1/flask/team/tim/models/sft/llama-8b/pistachio/applications/v8-full-retro/e2.lr1e-5.b128/checkpoint-44000"

MATRIX_NODE="34"

# Aniruddha hosting model with vLLM
## GPT-OSS-20b
#export VLLM_URL="http://192.168.128.${MATRIX_NODE}:8010/v1"
#export VLLM_MODEL="/p/vast1/flask/models/marathe1/gpt-oss-20b"

## GPT-OSS-120b
export VLLM_URL="http://192.168.128.${MATRIX_NODE}:8011/v1"
export VLLM_MODEL="/p/vast1/flask/models/marathe1/gpt-oss-120b"

# Reasoning level for GPT-OSS
export OSS_REASONING="medium" # Options: ["low", "medium", "high"]


cd experiments/Multi_Server_Experiments

python main.py \
    --server-urls "http://127.0.0.1:8000/sse" "http://127.0.0.1:8001/sse" \
    --backend vllm \
    --model gpt-oss-120b

#    --client autogen \
#    --server-path reaction_server.py \
#    --user-prompt "Generate a new reaction SMARTS and reactants for the product ${PRODUCT_SMILES}"
