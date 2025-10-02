################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from loguru import logger
from rdkit.Contrib.SA_Score import sascorer

mcp = FastMCP("Chem and BioInformatics MCP Server")

logger.info("Starting Chem and BioInformatics MCP Server")


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


@mcp.tool()
def get_density(smiles: str) -> float:
    """
    Calculate the density of a molecule given its SMILES string.
    """
    try:
        # logger.info(f"Calculating density for SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES string or molecule could not be created.")
            return 0.0
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

        if mol.GetNumConformers() == 0:
            logger.warning("No conformers found for the molecule.")
            return 0.0
        mw = Descriptors.MolWt(mol)
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            logger.warning("No atoms in the molecule.")
            return 0.0

        volume = AllChem.ComputeMolVolume(mol)
        density = volume / mw
        logger.info(f"Density for SMILES {smiles}: {density}")
        return density
    except Exception as e:
        return 0.0


@mcp.tool()
def get_density_and_synthesizability(smiles: str) -> tuple[float, float]:
    """
    Calculate the density and synthesizability of a molecule given its SMILES string.
    Returns a tuple of (density, synthesizability).
    """

    density = get_density(smiles)
    synthesizability = get_synthesizability(smiles)
    return density, synthesizability


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
    mcp.run(transport="stdio")
