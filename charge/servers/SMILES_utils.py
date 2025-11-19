################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Contrib.SA_Score import sascorer
    HAS_SMILES = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_SMILES = False
    logger.warning(
        "Please install the rdkit support packages to use this module."
        "Install it with: pip install charge[rdkit]",
    )

def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string. Returns the canonical SMILES.
    If the SMILES is invalid, returns "Invalid SMILES".

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The canonicalized SMILES string.
    """
    try:
        logger.info(f"Canonicalizing SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.error(f"Error canonicalizing SMILES: {smiles}")
        return "Invalid SMILES"


#        return smiles

# Persistent counter to demonstrate statefulness
SMILES_VERIFICATION_COUNTER = 0


def verify_smiles(smiles: str) -> bool:
    """
    Verify if a SMILES string is valid. Returns True if valid, False otherwise.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        bool: True if the SMILES is valid, False otherwise.
    """
    if not HAS_SMILES:
        raise ImportError("Please install the rdkit support packages to use this module.")
    try:
        global SMILES_VERIFICATION_COUNTER
        SMILES_VERIFICATION_COUNTER += 1
        logger.info(
            f"Verifying SMILES: {smiles} used {SMILES_VERIFICATION_COUNTER} times"
        )
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            logger.info(f"SMILES is valid: {smiles}")
            return True
        else:
            logger.error(f"SMILES is invalid: {smiles}")
            return False
    except Exception as e:
        logger.error(f"SMILES is invalid: {smiles}")
        return False


def get_synthesizability(smiles: str) -> float:
    """
    Calculate the synthesizability of a molecule given its SMILES string.
    Values range from 1.0 (highly synthesizable) to 10.0 (not synthesizable).

    Args:
        smiles (str): The input SMILES string.

    Returns:
        float: The synthesizability score.
    """
    if not HAS_SMILES:
        raise ImportError("Please install the rdkit support packages to use this module.")
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
NUM_HITS = 1


def known_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is already known.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        bool: True if the SMILES is known to this MCP server, False otherwise.
    """

    if not HAS_SMILES:
        raise ImportError("Please install the rdkit support packages to use this module.")
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
