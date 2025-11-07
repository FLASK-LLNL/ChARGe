#!/bin/bash

LC_NODE="34"

# vLLM hosted model
## GPT-OSS-20b
#export VLLM_URL="http://192.168.128.${LC_NODE}:8010/v1"
#export VLLM_MODEL="/p/vast1/flask/models/marathe1/gpt-oss-20b"

## GPT-OSS-120b
export VLLM_URL="http://192.168.128.${LC_NODE}:8011/v1"
export VLLM_MODEL="/p/vast1/flask/models/marathe1/gpt-oss-120b"

# Reasoning level for GPT-OSS
export OSS_REASONING="low" # Options: ["low", "medium", "high"]


cd examples/Multi_Server_Experiments

python main.py \
    --server-urls "http://127.0.0.1:8000/sse" "http://127.0.0.1:8001/sse" \
    --backend vllm \
    --model gpt-oss-120b
