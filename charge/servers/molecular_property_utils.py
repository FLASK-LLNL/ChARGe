################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Contrib.SA_Score import sascorer

    HAS_RDKIT = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_RDKIT = False
    logger.warning(
        "Please install the rdkit support packages to use this module."
        "Install it with: pip install charge[rdkit]",
    )

from charge.servers.SMILES_utils import get_synthesizability
from charge.servers.get_chemprop2_preds import predict_with_chemprop
from charge.servers.molecule_pricer import get_chemspace_prices
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
    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
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

    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    density = get_density(smiles)
    synthesizability = get_synthesizability(smiles)
    return density, synthesizability


def chemprop_preds_server(smiles: str, property: str) -> float:
    """
    Predict molecular properties using pre-trained Chemprop models.
    This function returns property predictions from Chemprop models. It validates the requested property name,
    constructs the appropriate model, and returns predictions for the provided SMILES input.

    Valid properties
    ----------------
    ChARGe can request any of the following property names:
      - density : Predicted density (g/cm³)
      - hof     : Heat of formation (kcal/mol)
      - alpha   : Polarizability (a0³)
      - cv      : Heat capacity at constant volume (cal/mol·K)
      - gap     : HOMO–LUMO energy gap (Hartree)
      - homo    : HOMO energy (Hartree)
      - lumo    : LUMO energy (Hartree)
      - mu      : Dipole moment (Debye)
      - r2      : Electronic spatial extent (a0^2)
      - zpve    : Zero-point vibrational energy (Hartree)
      - lipo    : Octanol–water partition coefficient (logD)

    Args:
        smiles (str): A SMILES string representing the molecule to be evaluated.
        property (str): The property to predict. Must be one of the valid property names listed above.

    Returns:
        float
            A float representing the predicted value for the specified property.

    Raises:
        SystemExit
            If the environment variable `CHEMPROP_BASE_PATH` is not set.

    Examples:
        >>> chemprop_preds_server("CCO", "gap")
        6.73

        >>> chemprop_preds_server("c1ccccc1", "lipo")
        2.94
    """

    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    valid_properties = {
        "density",
        "hof",
        "alpha",
        "cv",
        "gap",
        "homo",
        "lumo",
        "mu",
        "r2",
        "zpve",
        "lipo",
    }
    if property not in valid_properties:
        raise ValueError(
            f"Invalid property '{property}'. Must be one of {valid_properties}."
        )
    chemprop_base_path = os.environ.get("CHEMPROP_BASE_PATH")
    if chemprop_base_path:
        model_path = os.path.join(chemprop_base_path, property)
        model_path = os.path.join(model_path, "model_0/best.pt")
        return predict_with_chemprop(model_path, [smiles])[0][0]
    else:
        print("CHEMPROP_BASE_PATH environment variable not set!")
        sys.exit(2)


def get_molecule_price(smiles):
    """
    Retrieve vendor pricing from ChemSpace for the molecule specified by the SMILES string, smiles.

    Args:
        smiles : str
            A SMILES string for the molecule of interest.

    Returns:
        float
            Returns float representing the lowest price (in USD/g) among all vendors for the specified molecules in SMILES_list.

    Examples:
        >>> get_molecule_price("CCO")
        0.1056
    """

    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    price = get_chemspace_prices([smiles])
    return price[0]
