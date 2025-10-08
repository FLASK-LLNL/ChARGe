cd experiments/Retrosynthesis

# Set environment variables for vLLM
export VLLM_URL="http://192.168.128.31:8000/v1"
export VLLM_MODEL="/p/vast1/flask/models/gpt-oss-20b"

# Run with vLLM backend
python main.py \
    --client autogen \
    --backend vllm \
    --model "/p/vast1/flask/models/gpt-oss-20b" \
    --server-path reaction_server.py \
    --user-prompt "Generate a new reaction SMARTS and reactants for the target molecule: CC(=O)OC1=CC=CC=C1C(=O)O"
