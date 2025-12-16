## A RAG SSE Server with ChARGe

This RAG MCP server has functions to retrieve similar reactions, and expert predictions for forward or retrosynthesis.

In retrosynthesis, the user will input a dictionary like

`{"products": ["CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1"]}`


and may retrieve information like:

```
{
  "products": [
     "CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1"
   ],
  "similar": [
    {
      "reactants": [
        "CC(C)(C)OC(=O)n1ccc2cc(CO)ccc21"
      ],
      "products": [
        "CC(C)(C)OC(=O)n1ccc2cc(C=O)ccc21"
      ],
      "expert_prediction": "{\"products\": [\"CC(C)(C)OC(=O)n1ccc2cc(C=O)ccc21\"]}"
    },
    {
      "reactants": [
        "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
        "COC(=O)c1ccc2[nH]ccc2c1"
      ],
      "products": [
        "COC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1"
      ],
      "expert_prediction": "{\"products\": [\"COC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1\"]}"
    },
    {
      "reactants": [
        "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
        "COC(=O)c1ccc2cc[nH]c2c1"
      ],
      "products": [
        "COC(=O)c1ccc2ccn(C(=O)OC(C)(C)C)c2c1"
      ],
      "expert_prediction": "{\"products\": [\"COC(=O)c1ccc2ccn(C(=O)OC(C)(C)C)c2c1\"]}"
    }
  ],
  "expert_prediction": "{\"reactants\": [\"CC(=O)c1ccc2[nH]ccc2c1\", \"CC(C)(C)OC(=O)OC(=O)OC(C)(C)C\"], \"agents\": [\"CN(C)c1ccncc1\"], \"solvents\": [\"CC#N\"]}"
}

```

### Setup

You must first run the server on a different process. You can do this by running the following command in a terminal:

```bash
python "../../charge/rag/rag_mcp_server.py" \
    --database-path  <path-to-jsonl-file>  \
    --embedder-path <path-to-embedder-torchscript-file> \
    --embedder-vocab-path <path-to-emebedder-vocab-json>  \
    --forward-embedding-path <path-to-RAG-embedding-database-for-reagents> \
    --retro-embedding-path   <path-to-RAG-embedding-database-for-products> \
    --forward-expert-model-path <path-to-HF-expert-forward-model-checkpoint> \
    --retro-expert-model-path <path-to-HF-expert-forward-model-checkpoint> \
```

This will start an SSE MCP server locally. Note the address and port where the server is running (by default, it will be `http://127.0.0.1:8000`).


### Client Usage
You can then use the ChARGe client to connect to this server and perform operations.
Run the following script to see how to use the client with the RAG server. This performs retrosynthesis on the specified molecule.

```bash
python main.py --backend <backend> --model <model> --server-url <server_url>/sse --retrosynthesis --lead-molecules "CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1"

```

**Note:** The `--server-url` should point to the address where your SSE MCP server is running, appended with `/sse`.

To use the `vllm` backend, set the following environment variables before running:

```bash
export VLLM_URL="<url-of-vllm-model>"
export VLLM_MODEL="<path-to-model-weights>"  # e.g., /usr/workspace/gpt-oss-120b
export OSS_REASONING="low"                   # Options: ["low", "medium", "high"]
```
