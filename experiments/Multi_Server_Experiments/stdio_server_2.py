from mcp.server.fastmcp import FastMCP
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from loguru import logger
from rdkit.Contrib.SA_Score import sascorer


mcp = FastMCP("Chem Server that canonicalizes SMILES strings")
logger.info("Starting Chem Server that canonicalizes SMILES strings")


@mcp.tool()
def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string.
    """
    try:
        logger.info(f"Canonicalizing SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        return smiles


if __name__ == "__main__":
    mcp.run(transport="stdio")
