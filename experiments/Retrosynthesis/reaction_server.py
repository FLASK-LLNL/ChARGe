################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from charge.servers.SMILES_utils import verify_smiles, canonicalize_smiles
import argparse


template_free_mcp = FastMCP("template_free_reaction_server")

template_free_mcp.tool()(verify_smiles)
template_free_mcp.tool()(canonicalize_smiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-type",
        default="template",
        choices=["template", "template-free"],
    )
    args = parser.parse_args()
    exp_type = args.exp_type

    if exp_type == "template":
        from charge.servers.SMARTS_reactions import SMARTS_mcp

        SMARTS_mcp.run(
            transport="sse",
        )
    elif exp_type == "template-free":

        from charge.servers.AiZynthTools import is_molecule_synthesizable

        template_free_mcp.tool()(is_molecule_synthesizable)
        template_free_mcp.run(
            transport="sse",
        )
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
