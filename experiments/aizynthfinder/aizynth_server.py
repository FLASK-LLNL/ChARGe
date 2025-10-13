################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################


from charge.servers.server_utils import parser

# import argparse
# from aizynthfinder.aizynthfinder import AiZynthFinder
# from rdkit import Chem
from mcp.server.fastmcp import FastMCP

from charge.servers.AiZynthTools import is_molecule_synthesizable
parser.add_argument('--config', type=str, help='Config yaml file for initializing AiZynthFinder')

args = parser.parse_args()

#from charge.servers.server_utils import args

# mcp = FastMCP(
#     "Database MCP Server that keeps track of known molecules",
#     port=args.port,
#     website_url=f"{args.host}",
# )

# CLI arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--host', type=str, help='Server host', default='127.0.0.1')
# parser.add_argument('--port', type=int, help='Server port', default=8000)
# args = parser.parse_args()


# Initialize MCP server
mcp = FastMCP('AiZynthFinder', website_url=args.host, port=args.port)


# # Helper functions
# def is_valid_smiles(smiles: str) -> bool:
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return False
#     return True


# # Tools
# @mcp.tool()

mcp.tool()(is_molecule_synthesizable)

def main():
    # Initialize AiZynthFinder
    # global finder
    # finder = AiZynthFinder(configfile=args.config)
    # finder.stock.select('zinc')
    # finder.expansion_policy.select('uspto')
    # finder.filter_policy.select("uspto")

    # Run MCP server
    mcp.run(transport=args.transport)



if __name__ == "__main__":
    main()
