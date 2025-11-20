################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdChemReactions
except ImportError:
    raise ImportError("Please install the rdkit package to use this module.")

from loguru import logger

from charge.servers.server_utils import add_server_arguments
import argparse

parser = argparse.ArgumentParser()
add_server_arguments(parser)
args = parser.parse_args()

SMARTS_mcp = FastMCP(
    "[RDKit-SMARTS] Chemistry and reaction verification MCP Server",
    port=args.port,
    website_url=f"{args.host}",
)

logger.info("[RDKit-SMARTS] Starting Chemistry and reaction verification MCP Server")

import charge.servers.SMARTS_reactions_utils as smarts
import charge.servers.SMILES_utils as smiles

SMARTS_mcp.tool()(smarts.verify_reaction_SMARTS)

SMARTS_mcp.tool()(smiles.verify_smiles)

SMARTS_mcp.tool()(smarts.verify_reaction)
