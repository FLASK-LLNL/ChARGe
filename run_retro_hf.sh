#!/bin/bash

# Configuration
#export LOCAL_MODEL_PATH="/p/vast1/flask/models/marathe1/gpt-oss-20b"
export LOCAL_MODEL_PATH="/p/vast1/flask/models/marathe1/gpt-oss-120b"
PRODUCT_SMILES="c1cc(ccc1N)O"  # Default from main.py

cd examples/Retrosynthesis

python main.py \
    --client autogen \
    --backend huggingface \
    --user-prompt "Generate a new reaction SMARTS and reactants for the product ${PRODUCT_SMILES}"

#    --server-path reaction_server.py \
#    --device auto \
#    --quantization 4bit
