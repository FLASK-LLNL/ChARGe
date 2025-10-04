################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    raise ImportError("Please install the rdkit package to use this module.")
from loguru import logger
from charge.servers.SMILES import SMILES_mcp
from charge.servers.SMILES_utils import get_synthesizability
import charge.servers.molecular_property_utils as mol_props


# Add some custom tools to the server
SMILES_mcp.tool()(mol_props.get_density)

SMILES_mcp.tool()(mol_props.get_density_and_synthesizability)

if __name__ == "__main__":
    SMILES_mcp.run(transport="stdio")
