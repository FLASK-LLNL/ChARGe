################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    raise ImportError("Please install the rdkit package to use this module.")

from loguru import logger
import logging

from charge.servers.server_utils import args

SMILES_mcp = FastMCP(
    "[RDKit-SMILES] Chem and BioInformatics MCP Server",
    port=args.port,
    website_url=f"{args.host}",
)

logger.info("[RDKit-SMILES] Starting Chem and BioInformatics MCP Server")

logging.basicConfig(level=logging.DEBUG)

import charge.servers.SMILES_utils as smiles

SMILES_mcp.tool()(smiles.canonicalize_smiles)

SMILES_mcp.tool()(smiles.verify_smiles)

SMILES_mcp.tool()(smiles.get_synthesizability)

SMILES_mcp.tool()(smiles.known_smiles)

