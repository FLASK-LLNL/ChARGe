from mcp.server.fastmcp import FastMCP
from rdkit import Chem

from loguru import logger
import argparse

parser = argparse.ArgumentParser(description="Run the Database MCP Server")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument(
    "--host", type=str, default="http://127.0.0.1", help="Host to run the server on"
)
args = parser.parse_args()


mcp = FastMCP(
    "Database MCP Server that keeps track of known molecules",
    port=args.port,
    website_url=f"{args.host}",
)
logger.info("Starting Database MCP Server")

database_of_smiles = []

NUM_HITS = 1


@mcp.tool()
def known_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is already known.
    """

    try:
        global NUM_HITS
        logger.info(f"Tool has been call: {NUM_HITS} times")

        NUM_HITS += 1
        logger.info(f"Checking if SMILES is known: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        if smiles in database_of_smiles:
            return True
        else:
            database_of_smiles.append(smiles)
            logger.info(f"SMILES not found in the database: {smiles}")
            return False

    except Exception as e:
        return False


if __name__ == "__main__":
    mcp.run(transport="sse")
