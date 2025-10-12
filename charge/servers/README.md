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
