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

import numpy as np
from ase import Atoms
from ase.optimize import MDMin
from ase.optimize.sciopt import SciPyFminCG
from ase.calculators.calculator import Calculator, all_changes
import torch
from torch_geometric.data import Data

from loguru import logger
from charge.servers.SMILES_utils import get_synthesizability

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


class GotenNetCalculator(Calculator):
    def __init__(self, path=None, model=None, device="cuda", weights_only=True, **kwargs):
        Calculator.__init__(self, **kwargs)
        if path is not None and model is not None:
            print("Both path and model specified, using model")
            self.model = model
        elif path is not None:
            if weights_only:
                from .nnp import EnergyForceModel
                self.model = EnergyForceModel()
                self.model.load_state_dict(torch.load(path, weights_only=weights_only))
            else:
                self.model = torch.load(path, weights_only=weights_only)
        elif model is not None:
            self.model = model

        self.model.to(device)
        self.device = device
        self.implemented_properties = [
            "energy",
            "forces",
        ]
        self.results = {}

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        z = torch.tensor(atoms.get_atomic_numbers(), device=self.device)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device, requires_grad=True)
        batch = torch.tensor([0] * z.size(0), dtype=torch.int64, device=self.device)

        inp = Data(z=z, pos=pos, batch=batch)

        out = self.model(inp)

        self.results = {}
        for property in properties:
            if property == "forces":
                # self.results["forces"] = -1 * pos.grad.cpu().detach().numpy()
                self.results["forces"] = out["forces"].cpu().detach().numpy()
            if property == "energy":
                # self.results["energy"] = float(out["property"].cpu().detach().numpy()[0])
                self.results["energy"] = float(out["energy"].cpu().detach().numpy()[0])

def AtomsFromMol(mol):
    """
    Get an ase.Atoms object from an RDKit.Mol, with preexisting conformers.
    """
    numbers = []

    for i, atom in enumerate(mol.GetAtoms()):
        numbers.append(atom.GetAtomicNum())

    positions = mol.GetConformer().GetPositions()
    numbers = np.array(numbers)
    return Atoms(
        numbers=numbers,
        positions=positions,
    )

_calcs = {
    "U0": None,
    "gap": None,
}

def get_gap(smiles: str) -> float:
    """
    Calculate the HOMO-LUMO gap of a molecule given its SMILES string.
    Returns a float of gap in eV.

    Args:
        smiles (str): The input SMILES string.
    Returns:
        float: The HOMO-LUMO gap of the molecule, returns NaN if there is an error.
    """

    if _calcs["U0"] is None:
        logger.info("Initializing GotenNet U0 calculator")
        _calcs["U0"] = GotenNetCalculator("models/U0.pt")
        logger.info("Initializing GotenNet gap calculator")
        _calcs["gap"] = GotenNetCalculator("models/gap.pt")

    elif _calcs["gap"] is None:
        logger.info("Initializing GotenNet gap calculator")
        _calcs["gap"] = GotenNetCalculator("models/U0.pt")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES string or molecule could not be created.")
        return float("nan")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if mol.GetNumConformers() == 0:
        logger.warning("No conformers found for the molecule.")
        return float("nan")

    AllChem.UFFOptimizeMolecule(mol, maxIters=490)
    atoms = AtomsFromMol(mol)
    atoms.calc = _calcs["U0"]
    dyn = SciPyFminCG(atoms, logfile="/dev/null")
    try:
        dyn.run(fmax=0.01, steps=10)
    except:
        # Sometimes the optimizer will raise an error about precision loss, despite the molecule being optimized successfully.
        # pass
        raise

    gap = _calcs["gap"].get_potential_energy(atoms)
    logger.info(f"Gap for SMILES {smiles}: {gap}")

    return gap
