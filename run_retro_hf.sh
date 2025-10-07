#!/bin/bash

# Configuration
LOCAL_MODEL_PATH="/p/vast1/flask/models/gpt-oss-120b"  # UPDATE THIS!
PRODUCT_SMILES="c1cc(ccc1N)O"  # Default from main.py

cd experiments/Retrosynthesis

python main.py \
    --client autogen \
    --backend huggingface \
    --server-path reaction_server.py \
    --user-prompt "Generate a new reaction SMARTS and reactants for the product ${PRODUCT_SMILES}"
#    --local-model-path "${LOCAL_MODEL_PATH}" \
#    --device auto \
#    --quantization 4bit
