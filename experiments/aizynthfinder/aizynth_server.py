################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################


from charge.servers.server_utils import parser
from mcp.server.fastmcp import FastMCP
from charge.servers.AiZynthTools import is_molecule_synthesizable

parser.add_argument('--config', type=str, help='Config yaml file for initializing AiZynthFinder')
args = parser.parse_args()

# Initialize MCP server
mcp = FastMCP('AiZynthFinder', website_url=args.host, port=args.port)

mcp.tool()(is_molecule_synthesizable)

def main():
    # Run MCP server
    mcp.run(transport=args.transport)

if __name__ == "__main__":
    main()
