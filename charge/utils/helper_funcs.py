import json


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
