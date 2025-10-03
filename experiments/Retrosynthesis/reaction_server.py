################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions
from loguru import logger
from typing import Tuple

mcp = FastMCP("Chemistry and reaction verification MCP Server")

logger.info("Starting Chemistry and reaction verification MCP Server")

from charge.servers.SMARTS_reactions import SMARTS_mcp

if __name__ == "__main__":
    SMARTS_mcp.run(
        transport="stdio",
    )
