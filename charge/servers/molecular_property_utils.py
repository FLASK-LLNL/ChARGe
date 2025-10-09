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
from charge.servers.get_chemprop2_preds import predict_with_chemprop
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
    
    """
    Predict molecular properties using pre-trained Chemprop models.

    This function serves as a server-side entry point for ChARGe to obtain property
    predictions from Chemprop models. It validates the requested property name,
    constructs the appropriate model path based on the `CHEMPROP_BASE_PATH`
    environment variable, and returns predictions for the provided SMILES input.

    Valid properties
    ----------------
    ChARGe can request any of the following property names:
      - 10k_density : Predicted density (g/cm³) from 10k dataset model
      - 10k_hof     : Heat of formation (kcal/mol) from 10k dataset model
      - qm9_alpha   : Polarizability (a0³)
      - qm9_cv      : Heat capacity at constant volume (cal/mol·K)
      - qm9_gap     : HOMO–LUMO energy gap (Hartree)
      - qm9_homo    : HOMO energy (Hartree)
      - qm9_lumo    : LUMO energy (Hartree)
      - qm9_mu      : Dipole moment (Debye)
      - qm9_r2      : Electronic spatial extent (a0^2)
      - qm9_zpve    : Zero-point vibrational energy (Hartree)
      - lipo        : Octanol–water partition coefficient (logD)

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecule to be evaluated.
    property : str
        The property to predict. Must be one of the valid property names listed above.

    Returns
    -------
    list[list[float]]
        A list of single-entry lists, where each inner list contains one float representing
        the predicted value for the specified property. For example:
        [[1.2158], [1.4041], [1.3984]].
        The outer list length corresponds to the number of input SMILES strings, and each
        inner list contains exactly one float prediction.

    Raises
    ------
    ValueError
        If the given property name is not recognized.
    SystemExit
        If the environment variable `CHEMPROP_BASE_PATH` is not set.

    Requirements
    ------------
    - The environment variable `CHEMPROP_BASE_PATH` must point to the base directory
      containing Chemprop model folders (e.g., ".../chemprop/Saved_Models/").
    - Each property model must exist under a path of the form:
      `{CHEMPROP_BASE_PATH}/{property}/model_0/best.pt`.

    Examples
    --------
    >>> chemprop_preds_server("CCO", "qm9_gap")
    6.73

    >>> chemprop_preds_server("c1ccccc1", "lipo")
    [[2.94]]
    """
    
    valid_properties = {'10k_density', '10k_hof', 'qm9_alpha','qm9_cv','qm9_gap','qm9_homo','qm9_lumo','qm9_mu','qm9_r2','qm9_zpve','lipo'}
    if property not in valid_properties:
        raise ValueError(
            f"Invalid property '{property}'. Must be one of {valid_properties}."
        )
    chemprop_base_path=os.environ.get("CHEMPROP_BASE_PATH")
    if(chemprop_base_path):
        model_path=os.path.join(chemprop_base_path, property)
        model_path=os.path.join(model_path, 'model_0/best.pt')
        return(predict_with_chemprop(model_path,[smiles]))
    else:
        print('CHEMPROP_BASE_PATH environment variable not set!')
        sys.exit(2)
