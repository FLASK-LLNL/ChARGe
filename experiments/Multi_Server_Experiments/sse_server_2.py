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
    "Database MCP Server that returns price of molecules",
    port=args.port,
    website_url=f"{args.host}",
)
logger.info("Starting Molecule Price MCP Server")


NUM_HITS = 1


@mcp.tool()
def known_smiles(smiles: str) -> float:
    """
    Returns the price of the molecule represented by the SMILES string.

    If the molecule is invalid returns -1.
    """

    try:
        global NUM_HITS
        logger.info(f"Tool has been call: {NUM_HITS} times")

        NUM_HITS += 1
        logger.info(f"Checking if SMILES is known: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        # Dummy pricing function: price is 10 times the length of the SMILES string
        price = len(smiles) * 10.0
        return price

    except Exception as e:
        return -1


if __name__ == "__main__":
    mcp.run(transport="sse")
