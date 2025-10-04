from mcp.server.fastmcp import FastMCP
from loguru import logger

mcp = FastMCP("Chem server that verifies SMILES strings")
logger.info("Starting Chem server that verifies SMILES strings")

import charge.servers.SMILES_utils as smiles

mcp.tool()(smiles.verify_smiles)

if __name__ == "__main__":
    mcp.run(transport="stdio")
