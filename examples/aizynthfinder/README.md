# AiZynthFinder experiment

AiZynthFinder is a retrosynthesis tool based on Monte Carlo tree search that breaks down a target molecule recursively into purchasable precursors.

This experiment demonstrates how to make a tool call to the aizynth MCP server, and have the AI orchestrator summarize the tool call output.


## Requirement

As noted in the README in `charge/servers/README.md` to install
AiZynthFinder you will need to perform a `pip install
charge[aizynthfinder]` or `pip install charge[all]` and follow it with
```
pip3 install --no-deps aizynthfinder reaction-utils
```

Alternatively, for an independent MCP server, You could have a separate python environment for running AiZynthFinder, because it has its own installation prerequisites that may not be compatible with the ChARGe environment.
Follow [these steps](https://github.com/MolecularAI/aizynthfinder?tab=readme-ov-file#installation) for AiZynthFinder installation. 

AiZynthFinder requires a config yaml file during
initialization. Further, this config file contains paths to databases,
reaction templates, and trained model files. This means you need to
have access to these files. The
[documentation](https://molecularai.github.io/aizynthfinder/index.html)
shows how to download these files from running `download_public_data
.` (once you have AiZynthFinder installed). You can also specify
different parameters in the config file for more advanced usage. See
[here](https://molecularai.github.io/aizynthfinder/configuration.html)
for more details regarding the config file.


## How to use

First set up the aizynth server:

```bash
python /path/to/aizynth_server.py --config /path/to/aizynth_config.yml --transport sse
```

This will start an SSE MCP server locally. The URL by default should be `http://127.0.0.1:8000/sse`.

You can then use the ChARGe client to connect to this server and perform operations:

```bash
python main.py --backend <backend> --model <model> --server-urls <server_url>/sse
```


## Tool call output format

AiZynthFinder typically returns multiple synthesis routes per target molecule. Only one route (for synthesizing caffeine) is shown below. This route is only single-step but should reveal all the dict key information. A route is a reaction tree in json/dict format. The reaction tree has two types of nodes: a molecule node and a reaction node. The tree root is the target molecule.

Notes:
- A molecule node's child is always a reaction node.
  - I'm not sure if a reaction node can have multiple parents (i.e., multiple products for a reaction). I haven't seen such a case from AiZynthFinder.
- A reaction node's children is always the reactant molecules of that reaction.
- For a relatively complex target molecule, a typical reaction tree from AiZynthFinder can be fairly deep (e.g., > 4-depth), resulting in lengthy outputs from the tool call.

```json
{
  "type": "mol",
  "hide": false,
  "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
  "is_chemical": true,
  "in_stock": false,
  "children": [
    {
      "type": "reaction",
      "hide": false,
      "smiles": "[C:1][n:4]([cH:3]:[NH:2])[cH2:5]>>COS(=O)(=O)O[C:1].[NH:2]:[cH:3][n:4][cH2:5]",
      "is_reaction": true,
      "metadata": {
        "template_hash": "93cbaf0b4b48881d9d7665df46a789855973bcb6da86f905ab1181a987e8abf0",
        "classification": "0.0 Unrecognized",
        "library_occurence": 99,
        "policy_probability": 0.014100000262260437,
        "policy_probability_rank": 7,
        "policy_name": "uspto",
        "template_code": 24589,
        "template": "[#7;a:2]:[c:3]:[n;H0;D3;+0:4](-[CH3;D1;+0:1]):[c:5]>>C-O-S(=O)(=O)-O-[CH3;D1;+0:1].[#7;a:2]:[c:3]:[nH;D2;+0:4]:[c:5]",
        "mapped_reaction_smiles": "[CH3:1][n:2]1[c:3](=[O:4])[c:5]2[c:6]([n:7][cH:8][n:9]2[CH3:10])[n:11]([CH3:12])[c:13]1=[O:14]>>[CH3:12][O:15][S:16]([O:17][CH3:18])(=[O:19])=[O:20].[CH3:1][n:2]1[c:3](=[O:4])[c:5]2[c:6]([n:7][cH:8][n:9]2[CH3:10])[nH:11][c:13]1=[O:14]"
      },
      "children": [
        {
          "type": "mol",
          "hide": false,
          "smiles": "COS(=O)(=O)OC",
          "is_chemical": true,
          "in_stock": true
        },
        {
          "type": "mol",
          "hide": false,
          "smiles": "Cn1c(=O)[nH]c2ncn(C)c2c1=O",
          "is_chemical": true,
          "in_stock": true
        }
      ]
    }
  ]
}
```

You can also get global information about the tree search process. This is an example for Ritonavir as the target molecule:
```json
{
  "target": "CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1",
  "search_time": 17.613009691238403,
  "first_solution_time": 0,
  "first_solution_iteration": 0,
  "number_of_nodes": 687,
  "max_transforms": 6,
  "max_children": 15,
  "number_of_routes": 109,
  "number_of_solved_routes": 0,
  "top_score": 0.8277327853542139,
  "is_solved": false,
  "number_of_steps": 5,
  "number_of_precursors": 7,
  "number_of_precursors_in_stock": 6,
  "precursors_in_stock": "O=C(Cl)C(=O)Cl, CNCc1csc(C(C)C)n1, O=C(Cl)Oc1ccc([N+](=O)[O-])cc1, COC(=O)[C@@H](N)C(C)C, O=C(Cl)Oc1ccc([N+](=O)[O-])cc1, OCc1cncs1",
  "precursors_not_in_stock": "CC(C)(C)[Si](C)(C)O[C@@H](C[C@H](Cc1ccccc1)N=[N+]=[N-])[C@@H](N)Cc1ccccc1",
  "precursors_availability": "zinc;zinc;zinc;zinc;Not in stock;zinc;zinc",
  "policy_used_counts": {
    "uspto": 686
  },
  "profiling": {
    "expansion_calls": 578,
    "reactants_generations": 1372,
    "iterations": 100
  }
}
```


## Example AI summary output

Here is one output example for synthesizing caffeine:

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

Here is one example for Ritonavir:

```
[o3 orchestrated] Experiment completed. Results: Below is a concise description of three reasonable disconnections that arise from the tool search.  All of the required “building blocks’’ (in italics) are commercially listed or available in ≤ 2-step literature procedures.  Standard peptide / carbamate chemistry is used throughout, so only the key bond-forming steps are highlighted.

────────────────────────
ROUTE A “Acid-chloride peptide coupling”

Key disconnections  
 1.  C-terminal amide (Ile–Phe junction)  
 2.  Carbamate to the 2-(hydroxymethyl)thiazolo[5,4-c]pyridine (“Het-CH₂OH”)  
 3.  Thiazole acid chloride coupling

Step outline  
1. *L-Isoleucine*  ─(Boc-protection, MeOH, SOCl₂)→  Boc-Ile-OMe  
2. Boc-Ile-OMe ─(DMF, (Me)₂NH, EDC)→  Boc-Ile-CONMe₂ (thiazole arm acceptor)  

3. Boc-Ile-CONMe₂  +  *5-isobutyl-2-(dimethylaminomethyl)thiazole*  
  ─(n-BuLi, then DMF, work-up)→  thiazole-(CH₂)-CONMe₂-Ile (free acid)  
4. Convert the Ile carboxylate to the acid chloride (oxalyl chloride, DMF).  

5. *H₂N-CH(Phe)-CH(OH)-CH₂-CH(Phe)-NH₂*  
  (prepared from L-Phe via reductive side-chain elaboration shown in Route C)  
  + acid chloride from step 4 ─(i-Pr₂NEt, –20 °C)→  target central amide.  

6. De-protect the N-terminus (TFA) to give the free amine.  

7. *Het-CH₂OH*  (2-(hydroxymethyl)thiazolo[5,4-c]pyridine)  
  → chloroformate (triphosgene, pyridine)  
  then add the free amine from step 6  → final carbamate → TARGET.  

Overall: three fragment union, all couplings with standard acid-chloride or chloroformate chemistry.  
Reagents and intermediates are bench-stable; no protecting-group sequence clashes.

────────────────────────
ROUTE B “Sequential peptide assembly”

1. Start from *H-Phe-CH₂OH* (reduction of L-phenylalanine methyl ester).  
2. Couple with Boc-Ile-OH (HATU / DIPEA) to give Boc-Ile-Phe-CH₂OH.  
3. Remove Boc (TFA) → H-Ile-Phe-CH₂OH (free N-terminus).  
4. Convert CH₂OH to carbamate with Het-ClCO₂ (chloroformate of Het-CH₂OH).  
5. Introduce the thiazole fragment exactly as in Route A (acid-chloride coupling of thiazole-CONMe₂-COCl).  

Advantage: only one protecting group (Boc) is manipulated; dipeptide prepared first and then decorated.

────────────────────────
ROUTE C “Late-stage carbamate then amide”

1. Prepare thiazole acid chloride as in Route A.  
2. Generate carbamate first:  
  H-Ile-NH-Phe-CH₂OH  +  Het-ClCO₂   →  Ile-NH-Phe-OCO-Het.  
3. Expose the Ile carboxylate (saponify if required), convert to acid chloride, then couple to the phenylalanine α-amine of *H-Phe-CH(OH)-CH₂-Phe-NH₂*.

This sequence delays formation of the congested amide until last, minimising side reactions during carbamate installation.

────────────────────────
Commercial / short-synthesis building blocks

• 5-Isobutyl-2-(dimethylaminomethyl)thiazole  
• 2-(Hydroxymethyl)thiazolo[5,4-c]pyridine  
• L-Isoleucine, L-Phenylalanine and their methyl esters  
• Dimethylamine, triphosgene, common peptide-coupling reagents (HATU, EDC, oxalyl chloride, etc.)

────────────────────────
Summary

All three routes converge on the same logic:  
(1) build the thiazole-N,N-dimethylamide fragment,  
(2) assemble an Ile–Phe dipeptide (or larger peptide) with a free α-amine,  
(3) couple the fragments via acid chloride, and  
(4) cap the terminal amine with the heteroaryl chloroformate to furnish the final carbamate.  
```
