# Exemplar ChARGe servers
Here is a collection of commonly used MCP servers in the ChARGe
framework.

- molecular_generation_server.py
- retrosynthesis_reaction_server.py
- SMILES.py
- SMARTS_reactions.py
- FLASKv2_reactions.py
- get_chemprop2_pred.py
- molecule_pricer.py

# Using AiZynthFinder tools
## Installation
After installing the ChARGe package with options [aizynthfinder], or
[all], run the additional commands to use the AiZynthFinder models.

1.) Install AiZynthFinder with pip.  Make sure not to install the
dependencies, they are locked to a very specific version that is
incompatible with ChARGe.
```
pip3 install --no-deps aizynthfinder reaction-utils
```

# Using Chemprop tools
## Installation
1.) Installing the ChARGe package with the [chemprop] or [all] options to use the Chemprop MPNN models.

2.) Set Chemprop model checkpoint path as environment variable
```
export CHEMPROP_BASE_PATH=<LC_PATH_TO_CHEMPROP_MODELS>
```
## Testing Chemprop Installation
```python
from charge.servers.molecular_property_utils import chemprop_preds_server
property='density'
chemprop_preds_server('COC(=O)COC=O','density')
```
Expected Result:
```
[[1.3979296684265137]]
```

## Usage
The `property` input variable in `chemprop_preds_server` must be set to one of the below properties.
```
valid_properties = {'density', 'hof', 'alpha','cv','gap','homo','lumo','mu','r2','zpve','lipo'}
```
# Using Chemprice tools
## Installation
After installing the ChARGe package, run the additional commands to use the Chemprice tools (getting the commercial price of a SMILES string).

1.) Install chemprice with pip.
```
pip3 install --no-deps chemprice
```

2.) Set API key for Chemspace as an environment variable
```
export CHEMSPACE_API_KEY=<ENTER_YOUR_CHEMSPACE_API_KEY>
```
## Testing Chemprice Installation
```python
from charge.servers.molecular_property_utils import get_molecule_price
smiles='CCO'
get_molecule_price(smiles)
```
Expected Result:
```
0.1056
```
