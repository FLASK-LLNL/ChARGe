from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer


def canonicalize_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        return "Invalid SMILES"


def verify_smiles(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        return False


def get_synthesizability(smiles: str) -> float:
    """
    Calculate the synthesizability of a molecule given its SMILES string.
    Values range from 1.0 (highly synthesizable) to 10.0 (not synthesizable).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0  # Default value for invalid SMILES
        score = sascorer.calculateScore(mol)
        return score
    except Exception as e:
        return 10.0
