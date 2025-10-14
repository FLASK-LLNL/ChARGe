################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################


from charge.servers.server_utils import add_server_arguments, update_mcp_network
from mcp.server.fastmcp import FastMCP
from charge.servers.AiZynthTools import is_molecule_synthesizable, find_synthesis_routes
import argparse

parser = argparse.ArgumentParser()
add_server_arguments(parser)
parser.add_argument('--config', type=str, help='Config yaml file for initializing AiZynthFinder')
args = parser.parse_args()

# Initialize MCP server
mcp = FastMCP('AiZynthFinder')

mcp.tool()(is_molecule_synthesizable)
mcp.tool()(find_synthesis_routes)

def main():
    from charge.servers.AiZynthTools import RetroPlanner

    RetroPlanner.initialize(configfile=args.config)

    update_mcp_network(mcp, host=args.host, port=args.port)
    # Run MCP server
    mcp.run(transport=args.transport)

if __name__ == "__main__":
    main()
