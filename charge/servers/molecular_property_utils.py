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
    raise ImportError("Please install the rdkit package to use this module.")

from loguru import logger
from charge.servers.SMILES_utils import get_synthesizability
from charge.servers.chemprop_make_prediction import get_chemprop_preds
import sys
import os

def get_density(smiles: str) -> float:
    """
    Calculate the density of a molecule given its SMILES string.
    Density is the molecular weight of the molecule per unit volume.
    In units of unified atomic mass (u) per cubic Angstroms (A^3)

    Args:
        smiles (str): The input SMILES string.
    Returns:
        float: Density of the molecule, returns 0.0 if there is an error.
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


def get_density_and_synthesizability(smiles: str) -> tuple[float, float]:
    """
    Calculate the density and synthesizability of a molecule given its SMILES string.
    Returns a tuple of (density, synthesizability).

    Density is the molecular weight of the molecule per unit volume.
    In units of unified atomic mass (u) per cubic Angstroms (A^3)
    Synthesizable values range from 1.0 (highly synthesizable) to 10.0 (not synthesizable).

    Args:
        smiles (str): The input SMILES string.
    Returns:
        A tuple containing:
            float: Density of the molecule, returns 0.0 if there is an error.
            float: Synthesizable score of the molecule, returns 10.0 if there is an error.
    """

    density = get_density(smiles)
    synthesizability = get_synthesizability(smiles)
    return density, synthesizability

def chemprop_preds_server(smiles,property):
    valid_properties = {'10k_density', '10k_hof', 'qm9_alpha','qm9_cv','qm9_gap','qm9_homo','qm9_lumo','qm9_mu','qm9_r2','qm9_zpve','lipo'}
    if property not in valid_properties:
        raise ValueError(
            f"Invalid property '{property}'. Must be one of {valid_properties}."
        )
    chemprop_base_path=os.environ.get("CHEMPROP_BASE_PATH")
    if(chemprop_base_path):
        model_path=chemprop_base_path+property+'_OOD/fold_0/'
        return(get_chemprop_preds(smiles,model_path))
    else:
        print('CHEMPROP_BASE_PATH environment variable not set!')
        sys.exit(2)
