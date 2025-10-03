from mcp.server.fastmcp import FastMCP
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from loguru import logger
from rdkit.Contrib.SA_Score import sascorer

import logging

mcp = FastMCP("Chem and BioInformatics MCP Server")
logger.info("Starting Chem and BioInformatics MCP Server")

logging.basicConfig(level=logging.DEBUG)


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


@mcp.tool()
def get_synthesizability(smiles: str) -> float:
    """
    Calculate the synthesizability of a molecule given its SMILES string.
    Values rankge from 1.0 (highly synthesizable) to 10.0 (not synthesizable).
    """

    try:
        # logger.info(f"Calculating synthesizability for SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES string or molecule could not be created.")
            return 10.0  # Default value for invalid SMILES
        score = sascorer.calculateScore(mol)
        logger.info(f"Synthesizability score for SMILES {smiles}: {score}")
        return score
    except Exception as e:
        logger.error(f"Error creating molecule from SMILES: {e}")
        return 10.0


database_of_smiles = []


@mcp.tool()
def known_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is already known.
    """

    try:
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
