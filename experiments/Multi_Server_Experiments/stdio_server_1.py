from mcp.server.fastmcp import FastMCP
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from loguru import logger
from rdkit.Contrib.SA_Score import sascorer


mcp = FastMCP("Chem server that verifies SMILES strings")
logger.info("Starting Chem server that verifies SMILES strings")


@mcp.tool()
def verify_smiles(smiles: str) -> bool:
    """
    Verify if a SMILES string is valid.
    """
    try:
        logger.info(f"Verifying SMILES: {smiles}")
        Chem.MolFromSmiles(smiles)
        return True
    except Exception as e:
        return False


if __name__ == "__main__":
    mcp.run(transport="stdio")
