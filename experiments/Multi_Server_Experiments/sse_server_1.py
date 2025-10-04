from mcp.server.fastmcp import FastMCP
from loguru import logger

from charge.servers.server_utils import args

mcp = FastMCP(
    "Database MCP Server that keeps track of known molecules",
    port=args.port,
    website_url=f"{args.host}",
)
logger.info("Starting Database MCP Server")

import charge.servers.SMILES_utils as smiles

mcp.tool()(smiles.known_smiles)

if __name__ == "__main__":
    mcp.run(transport="sse")
