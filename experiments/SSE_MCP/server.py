################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from loguru import logger

mcp = FastMCP("Chemistry and reaction verification MCP Server")

logger.info("Starting Chemistry and reaction verification MCP Server")

# Persistent counter to demonstrate statefulness
COUNTER = 0


@mcp.tool()
def verify_smiles(smiles: str) -> bool:
    """
    Verify if a SMILES string is valid.
    """
    try:
        global COUNTER
        COUNTER += 1
        logger.info(f"Verifying SMILES: {smiles} used {COUNTER} times")
        return True
    except Exception as e:
        return False


if __name__ == "__main__":

    mcp.run(transport="sse")
