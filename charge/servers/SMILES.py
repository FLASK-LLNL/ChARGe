################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Contrib.SA_Score import sascorer
    HAS_SMILES = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_SMILES = False
    logger.warning(
        "Please install the rdkit support packages to use this module."
        "Install it with: pip install charge[rdkit]",
    )

from charge.servers.server_utils import add_server_arguments
import argparse

parser = argparse.ArgumentParser()
add_server_arguments(parser)
args = parser.parse_args()

SMILES_mcp = FastMCP(
    "[RDKit-SMILES] Chem and BioInformatics MCP Server",
    port=args.port,
    website_url=f"{args.host}",
)

logger.info("[RDKit-SMILES] Starting Chem and BioInformatics MCP Server")

import charge.servers.SMILES_utils as smiles

SMILES_mcp.tool()(smiles.canonicalize_smiles)

SMILES_mcp.tool()(smiles.verify_smiles)

SMILES_mcp.tool()(smiles.get_synthesizability)

SMILES_mcp.tool()(smiles.known_smiles)
