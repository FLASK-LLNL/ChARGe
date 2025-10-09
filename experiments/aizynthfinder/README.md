# AiZynthFinder experiment

AiZynthFinder is a retrosynthesis tool based on Monte Carlo tree search that breaks down a target molecule recursively into purchasable precursors.

This experiment demonstrates how to make a tool call to the aizynth MCP server, and have the AI orchestrator summarize the tool call output.


## Requirement

You should have a separate python environment for running AiZynthFinder, because it has its own installation prerequisites that may not be compatible with the ChARGe environment.
Follow [these steps](https://github.com/MolecularAI/aizynthfinder?tab=readme-ov-file#installation) for AiZynthFinder installation. 

AiZynthFinder requires a config yaml file during initialization. Further, this config file contains paths to databases, reaction templates, and trained model files. This means you need to have access to these files. All these files are available at `/p/vast1/flask/team/tim/aizynth`. Alternatively, based on the [documentation](https://molecularai.github.io/aizynthfinder/index.html), you may be able to download these files from running `download_public_data .` (once you have AiZynthFinder installed). You can also specify different parameters in the config file for more advanced usage. See [here](https://molecularai.github.io/aizynthfinder/configuration.html) for more details regarding the config file.


## How to use

First set up the aizynth server:

```bash
python /path/to/aizynth_server.py --config /path/to/aizynth_config.yml --transport sse
```

Here is a specific example:

```bash
# cd to charge/servers
python aizynth_server.py --config /p/vast1/flask/team/tim/aizynth/config.yml --transport sse
```

This will start an SSE MCP server locally. The URL by default should be `http://127.0.0.1:8000/sse`.

You can then use the ChARGe client to connect to this server and perform operations:

```bash
python main.py --backend <backend> --model <model> --server-url <server_url>/sse
```

**Note:** The `--server-url` should point to the address where your SSE MCP server is running, appended with `/sse`.


## Example output (work-in-progress)

Here is one of the current output examplex for synthesizing caffeine:

```
Here are the distinct retrosynthetic disconnections returned by the tool, expressed as a nested list.  
Each inner list corresponds to one complete route (here every route is a single-step transformation from commercially available building blocks to the target).

[
  /* Route 1 – Late‐stage N-methylation with Me2SO4 */
  [
    "COS(=O)(=O)OC   +   Cn1cnc2c1c(=O)[nH]c(=O)n2C   →   CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
  ],

  /* Route 2 – Alternative N-methylation site (same reagent) */
  [
    "COS(=O)(=O)OC   +   Cn1c(=O)c2[nH]cnc2n(C)c1=O   →   CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
  ],

  /* Route 3 – Third N-methylation variant (same reagent) */
  [
    "COS(=O)(=O)OC   +   Cn1c(=O)[nH]c2ncn(C)c2c1=O   →   CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
  ],

  /* Route 4 – N-formylation / N-methylation using DMF as the C1-donor */
  [
    "CN(C)C=O   +   Cn1cnc2c1c(=O)[nH]c(=O)n2C   →   CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
  ],

  /* Route 5 – DMF variant with a different tautomer of the purine core */
  [
    "CN(C)C=O   +   Cn1c(=O)[nH]c2ncn(C)c2c1=O   →   CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
  ],

  /* Route 6 – Dialkyl-acetal mediated alkylation / protection sequence */
  [
    "CCOC(OCC)OCC   +   CNc1c(N)n(C)c(=O)n(C)c1=O   →   CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
  ]
]
```
