################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from loguru import logger
from charge.servers.ServerToolkit import ServerToolkit

try:
    from charge.servers.SMARTS_reactions_utils import verify_reaction_SMARTS
    from charge.servers.SMILES_utils import verify_smiles
    from charge.servers.SMARTS_reactions_utils import verify_reaction

    HAS_SMARTS = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_SMARTS = False
    logger.warning(
        "Please install the rdkit support packages to use this module."
        "Install it with: pip install charge[rdkit]",
    )


class SMARTSServer(ServerToolkit):
    """
    A ChARGe server that provides tools for Chemistry and reaction verification.
    """

    def __init__(self, mcp: FastMCP):
        """
        Initialize the SMARTSServer.

        Args:
            mcp (FastMCP): The MCP instance to register the function with.
        """
        super().__init__(mcp)

        if HAS_SMARTS:
            self.register_function_as_tool(self.mcp, verify_reaction_SMARTS)
            self.register_function_as_tool(self.mcp, verify_smiles)
            self.register_function_as_tool(self.mcp, verify_reaction)


if __name__ == "__main__":
    from charge.servers.server_utils import add_server_arguments
    import argparse

    logger.info(
        "[RDKit-SMARTS] Starting Chemistry and reaction verification MCP Server"
    )

    parser = argparse.ArgumentParser()
    add_server_arguments(parser)
    args = parser.parse_args()

    SMARTS_mcp = FastMCP(
        "[RDKit-SMARTS] Chemistry and reaction verification MCP Server",
        port=args.port,
        website_url=f"{args.host}",
    )

    sm = SMARTSServer(SMARTS_mcp)
    sm.run(transport=args.transport)
