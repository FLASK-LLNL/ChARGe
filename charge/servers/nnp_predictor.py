from loguru import logger
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    from ase import Atoms
    from ase.optimize.sciopt import SciPyFminCG
    from ase.calculators.calculator import Calculator, all_changes
    import torch
    from torch_geometric.data import Data
    from torch_geometric.datasets.qm9 import HAR2EV
    from gotennet.models.goten_model import GotenModel
    HAS_NNPS = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_NNPS = False
    logger.warning(
        "Please install the nnp support packages to use this module."
        "Install it with: pip install charge[nnp]",
    )

class GotenNetCalculator(Calculator):
    def __init__(self, model, device="cuda", weights_only=True, **kwargs):
        Calculator.__init__(self, **kwargs)
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
                out["property"].backward()
                self.results["forces"] = -1 * pos.grad.cpu().detach().numpy()
                # self.results["forces"] = out["forces"].cpu().detach().numpy()
            if property == "energy":
                self.results["energy"] = float(out["property"].cpu().detach().numpy()[0])
                # self.results["energy"] = float(out["energy"].cpu().detach().numpy()[0])

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
    "homo": None,
    "lumo": None,
}

def compute_band_gap(smiles: str) -> float:
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
        _calcs["U0"] = GotenNetCalculator(GotenModel.from_pretrained("QM9_small_U0"))
        logger.info("Initializing GotenNet homo calculator")
        _calcs["homo"] = GotenNetCalculator(GotenModel.from_pretrained("QM9_small_homo"))
        logger.info("Initializing GotenNet lumo calculator")
        _calcs["lumo"] = GotenNetCalculator(GotenModel.from_pretrained("QM9_small_lumo"))

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

    gap = _calcs["lumo"].get_potential_energy(atoms) - _calcs["homo"].get_potential_energy(atoms)
    logger.info(f"Gap for SMILES {smiles}: {gap}")

    # GotenNet predictions are improperly normalized for some values, see PyG source for QM9 dataset to see which ones
    return gap / HAR2EV

def main(smiles: str):
    print(f"{smiles} gap: {compute_band_gap(smiles)}")

if __name__ == "__main__":
    main("O=CCOC=O")
