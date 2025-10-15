# Flaskv2 experiment


Flaskv2 refers to in-house languge models that have been finetuned to perform forward synthesis and retrosynthesis tasks.

This experiment demonstrates how to make a tool call to the flaskv2 MCP server, and have the AI orchestrator summarize the tool call output.


## Requirement

The current flaskv2 models were finetuned and interfaced with HuggingFace (HF) utilities. Although model inference does not stricly require these dependencies, many HF features (e.g., tokenization, `model.generate(...)`) are quite convenient to have, and are being used for setting up the MCP server. To install these dependencies, install ChARGe with the `[flask]` or `[all]` option. The HF dependencies are also listed in `pyproject.toml`.


## How to use

First set up the flaskv2 server:

```bash
python /path/to/FLASKv2_reactions.py --model-dir-retro /path/to/model/checkpoint/ --transport sse
```

The above is an example for setting up an SSE server that only exposes the retrosynthesis tool call (excluding forward synthesis). There are other command line arguments you can specify (please see `python FLASKv2_reactions.py --help`).

You can then use the ChARGe client to connect to this server and perform operations:

```bash
python main.py --backend <backend> --model <model> --server-urls <server_url>/sse --retrosynthesis --lead-molecules "{SMILES}"
```

The `--retrosynthesis` arguments specifies that a `RetrosynthesisExperiment` is performed. Otherwise a `ForwardSynthesisExperiment` will be performed.
The `--lead-molecules` argument can take a single SMILES string such as `"CC(=O)O[C@H](C)CCN"` (typically in the case of retrosynthesis) or multiple SMILES strings separated by space, such as `"CC" "O=O"` (typically in the case of forward synthesis). You should probably put `--lead-molecules` as the last argument on the command line.


## Example AI summary output


### Retrosynthesis for caffeine

```
[o3 orchestrated] Experiment completed. Results: Original tool output
1) {"reactants": ["CI", "CN1C=NC2=C1C(=O)N(C(=O)N2)C"], "agents": ["[NaH]"], "solvents": ["CN(C)C=O"]}
2) {"reactants": ["CN1C=NC2=C1C(=O)N(C)C(=O)N2C"], "agents": ["CC(C)O", "CCOC(C)=O", "O=C(O)[O][Na]", "[NaH]"], "solvents": ["C1CCOC1"]}
3) {"reactants": ["CI", "CN1C=NC2=C1C(=O)N(C(=O)N2)C"], "agents": ["[Na]"], "solvents": ["CO"]}

Overall retrosynthetic plan to make caffeine (CN1C=NC2=C1C(=O)N(C(=O)N2C)C)

Step 1. N-Methylation of theobromine to give caffeine  
 Theobromine (3,7-dimethylxanthine; SMILES: CN1C=NC2=C1C(=O)N(C(=O)N2)C)  
  + MeI (or Me2SO4) + strong base (NaH, NaOH, K2CO3) → caffeine  
 (Reaction indicated by tool outputs #1 and #3: “CI” = CH3I plus theobromine.)

Step 2. Preparation of theobromine from xanthine (if needed)  
 Xanthine + MeI (1 equiv) + base → 7-methylxanthine  
 7-Methylxanthine + MeI (1 equiv) + base → theobromine  

Commercially, xanthine itself can be obtained from uric acid by decarboxylation, or it can be built de novo from dimethylurea and malonic acid (Traube synthesis).  

Summary route:  
Uric acid → (decarboxylation) xanthine → (2 × MeI, base) theobromine → (MeI, base) caffeine.
```

