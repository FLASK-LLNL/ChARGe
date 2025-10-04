from mcp.server.fastmcp import FastMCP
from loguru import logger

mcp = FastMCP("Chem Server that canonicalizes SMILES strings")
logger.info("Starting Chem Server that canonicalizes SMILES strings")

import charge.servers.SMILES_utils as smiles

mcp.tool()(smiles.canonicalize_smiles)

if __name__ == "__main__":
    mcp.run(transport="stdio")
