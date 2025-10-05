# Using Chemprop tools
## Installation (tested only on Tuo)
After installing the ChARGe package, run the additional commands to use the Chemprop MPNN models.

1.) Install chemprop. It is important to to install directly from this directory instead of the PyPI version- PyPI has some incompatible package dependencies.
```
cd external_tools/chemprop
pip install -e .
```

2.) (For usage on LC), Set Chemprop model checkpoint path as environment variable
```
export CHEMPROP_BASE_PATH=<LC_PATH_TO_CHEMPROP_MODELS>
```
## Testing Chemprop Installation
```python
from charge.servers.chemprop_make_prediction import get_chemprop_preds
from charge.servers.molecular_property_utils import chemprop_preds_server
smiles = [['CC(NCC1CO1)C#N'], ['CN(C)c1ccc(cc1)C(=C(C#N)C#N)C(c1ccc(cc1)N(=O)=O)=C1C(=O)c2ccccc2C1=O'], ['CC#CCC(OC(=O)c1ccc(cc1)N(=O)=O)C1(O)C(=O)OCC1(C)C=C']]
property='10k_density'
print(chemprop_preds_server(smiles,property))
```
Expected Result:
```
[[1.2157875703442804], [1.404070329005911], [1.3983503117725584]]
```

## Usage
The `property` input variable in `chemprop_preds_server` must be set to one of the below properties.
```
valid_properties = {'10k_density', '10k_hof', 'qm9_alpha','qm9_cv','qm9_gap','qm9_homo','qm9_lumo','qm9_mu','qm9_r2','qm9_zpve','lipo'}
```
