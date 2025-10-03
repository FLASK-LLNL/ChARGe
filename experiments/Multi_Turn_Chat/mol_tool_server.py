################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from loguru import logger
from rdkit.Contrib.SA_Score import sascorer

import logging

from charge.servers.SMILES import SMILES_mcp

if __name__ == "__main__":
    SMILES_mcp.run(transport="sse")
