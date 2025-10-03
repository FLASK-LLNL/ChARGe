################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    raise ImportError(
        "Please install the rdkit package to use this module."
    )
from loguru import logger
from charge.servers.SMILES import SMILES_mcp

# Add some custom tools to the server
@SMILES_mcp.tool()
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


@SMILES_mcp.tool()
def get_density_and_synthesizability(smiles: str) -> tuple[float, float]:
    """
    Calculate the density and synthesizability of a molecule given its SMILES string.
    Returns a tuple of (density, synthesizability).
    """

    density = get_density(smiles)
    synthesizability = SMILES_mcp.get_synthesizability(smiles)
    return density, synthesizability


if __name__ == "__main__":
    SMILES_mcp.run(transport="stdio")
