# Using Chemprop tools
## Installation (tested only on Dane)
After installing the ChARGe package, run the additional commands to use the Chemprop MPNN models.

1.) Install chemprop with pip.
```
pip install chemprop
```

2.) Set Chemprop model checkpoint path as environment variable
```
export CHEMPROP_BASE_PATH=<LC_PATH_TO_CHEMPROP_MODELS>
```
## Testing Chemprop Installation
```python
from molecular_property_utils import chemprop_preds_server
chemprop_preds_server('COC(=O)COC=O','10k_density')
```
Expected Result:
```
[[1.3979296684265137]]
```

## Usage
The `property` input variable in `chemprop_preds_server` must be set to one of the below properties.
```
valid_properties = {'10k_density', '10k_hof', 'qm9_alpha','qm9_cv','qm9_gap','qm9_homo','qm9_lumo','qm9_mu','qm9_r2','qm9_zpve','lipo'}
```
# Using Chemprice tools
## Installation
After installing the ChARGe package, run the additional commands to use the Chemprice tools (getting the commercial price of a SMILES string).

1.) Install chemprice with pip.
```
pip install chemprice
```

2.) Set API key for Chemspace as an environment variable
```
export CHEMSPACE_API_KEY=<ENTER_YOUR_CHEMSPACE_API_KEY>
```
## Testing Chemprice Installation
```python
from molecular_property_utils import get_molecule_price
smiles='CCO'
get_molecule_price(smiles)
```
Expected Result:
```
0.1056
```
