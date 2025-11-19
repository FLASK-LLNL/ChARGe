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
    import charge.servers.SMILES_utils as smiles

    HAS_SMILES = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_SMILES = False
    logger.warning(
        "Please install the rdkit support packages to use this module."
        "Install it with: pip install charge[rdkit]",
    )


class SMILESServer(ServerToolkit):
    """
    A ChARGe server that provides SMILES based tools.
    """

    def __init__(self, mcp: FastMCP):
        """
        Initialize the SMILESServer.

        Args:
            mcp (FastMCP): The MCP instance to register the function with.
        """
        super().__init__(mcp)

        if HAS_SMILES:
            self.register_function_as_tool(self.mcp, smiles.canonicalize_smiles)
            self.register_function_as_tool(self.mcp, smiles.verify_smiles)
            self.register_function_as_tool(self.mcp, smiles.get_synthesizability)
            self.register_function_as_tool(self.mcp, smiles.known_smiles)


if __name__ == "__main__":
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

    sm = SMILESServer(SMILES_mcp)
    sm.run(transport=args.transport)
