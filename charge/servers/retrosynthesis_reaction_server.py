################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from charge.servers.SMILES_utils import verify_smiles, canonicalize_smiles
from charge.servers.log_progress import log_progress
from charge.servers.ServerToolkit import ServerToolkit
import argparse
import os

try:
    from charge.servers.AiZynthTools import is_molecule_synthesizable, RetroPlanner

    HAS_AIZYNTH = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_AIZYNTH = False
    logger.warning(
        "Please install the aiZynthFinder support packages to use this module."
        "Install it with: pip install charge[aiZynthFinder]",
    )


class RetroSynthesisServer(ServerToolkit):
    """
    A ChARGe server that provides common retrosynthesis reaction tools.
    """

    def __init__(self, mcp: FastMCP):
        """
        Initialize the RetroSynthesisServer.

        Args:
            mcp (FastMCP): The MCP instance to register the function with.
        """
        super().__init__(mcp)

        self.register_function_as_tool(self.mcp, verify_smiles)
        self.register_function_as_tool(self.mcp, canonicalize_smiles)
        self.register_function_as_tool(self.mcp, log_progress)


class TemplateFreeRetroSynthesisServer(RetroSynthesisServer):
    """
    A ChARGe server that provides common retrosynthesis reaction tools for template free retrosynthesis.
    """

    def __init__(self, mcp: FastMCP):
        """
        Initialize the TemplateFreeRetroSynthesisServer.

        Args:
            mcp (FastMCP): The MCP instance to register the function with.
        """
        super().__init__(mcp)


class TemplateRetroSynthesisServer(RetroSynthesisServer):
    """
    A ChARGe server that provides common retrosynthesis reaction tools for template retrosynthesis.
    """

    def __init__(self, mcp: FastMCP, configfile: str):
        """
        Initialize the TemplateRetroSynthesisServer.

        Args:
            mcp (FastMCP): The MCP instance to register the function with.
            configfile (str): The path to the configuration file for the AiZynthFinder.
        """
        super().__init__(mcp)
        if HAS_AIZYNTH:
            RetroPlanner.initialize(configfile=configfile)
            self.register_function_as_tool(self.mcp, is_molecule_synthesizable)
        else:
            raise ImportError(
                "Please install the aiZynthFinder support packages to use this module."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-type",
        default="template",
        choices=["template", "template-free"],
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.getcwd(), "config.yml"),
        help="Path to the configuration file for the AiZynthFinder",
    )
    add_server_arguments(parser)
    args = parser.parse_args()
    exp_type = args.exp_type

    mcp = FastMCP(
        "RetroSynthesis Reaction Server", port=args.port, website_url=f"{args.host}"
    )

    if exp_type == "template":
        logger.info("Starting Template RetroSynthesis Server")
        server = TemplateRetroSynthesisServer(mcp, args.config)
    elif exp_type == "template-free":
        logger.info("Starting Template Free RetroSynthesis Server")
        server = TemplateFreeRetroSynthesisServer(mcp)
    else:
        raise ValueError(f"Unknown task type: {exp_type}")

    server.run(
        transport=args.transport,
    )
