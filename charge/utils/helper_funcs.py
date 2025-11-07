from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import json
from charge.servers import SMILES_utils
from charge.servers.molecular_property_utils import get_density


def get_list_from_json_file(file_path: str) -> list:
    """
    Load a list of molecules from a JSON file.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        list: The list of molecules.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data["smiles"]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    except Exception as e:
        return []


def save_list_to_json_file(data: list, file_path: str) -> None:
    """
    Save a list of molecules to a JSON file.
    Args:
        data (list): The list of molecules.
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        pass


def post_process_smiles(smiles: str, parent_id: int, node_id: int) -> dict:
    """
    Post-process a solution SMILES string, add additional properties and return
    a dictionary that can be appended to the known molecules JSON file.

    Args:
        smiles (str): The input SMILES string.
    Returns:
        dict: The post-processed dictionary.
    """
    canonical_smiles = SMILES_utils.canonicalize_smiles(smiles)
    sascore = SMILES_utils.get_synthesizability(canonical_smiles)
    density = get_density(canonical_smiles)

    return {"smiles": canonical_smiles, "sascore": sascore, "density": density}
