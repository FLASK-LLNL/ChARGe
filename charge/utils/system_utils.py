import os
import os.path as osp
import re
from typing import Type, Union, Optional
import json
import warnings
import requests

def normalize_string(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def _load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def _prompt_from_json_file(file_path: str, key: str) -> str:
    data = _load_json(file_path)
    for k in data.keys():
        k = normalize_string(k)
        if k == key:
            return data[k]
    raise ValueError(f"Nothing resembling '{key}' key found in JSON file")


def _prompt_from_txt_file(file_path: str) -> str:

    with open(file_path, "r") as f:
        prompt = f.read()
    return prompt


def _check_file_exists(file_path: str) -> bool:
    return osp.isfile(file_path)


def read_from_file(self, file_path: str, key: str) -> str:
    assert osp.isfile(file_path), f"File {file_path} does not exist"
    if file_path.endswith(".txt"):
        return _prompt_from_txt_file(file_path)
    elif file_path.endswith(".json"):
        return _prompt_from_json_file(file_path, key)
    else:
        raise ValueError("Only .txt and .json files are supported")

def check_url_exists(url: str) -> bool:
    if not url.startswith("http://") and not url.startswith("https://"):
        return False

    if not url.endswith("/sse"):
        return False

    try:
        with requests.get(url, stream=True, timeout=5) as response:
            if response.status_code != 200:
                return False
    except requests.RequestException as e:
        warnings.warn(f"Error reaching URL '{url}': {e}")
        return False

    return True


def check_server_paths(server_paths: Optional[Union[str, list]]) -> list:
    """
    Gracefully handle errors in server paths provided by user.
    Args:
        server_paths (Optional[Union[str, list]]): The server paths to check.
    Returns:
        list: A list of valid server paths.
    Raises:
        FileNotFoundError: If any of the server paths do not exist and
        CHARGE_ERROR_ON_MISSING_SERVER is set to 1.
    """

    if server_paths is None:
        return []
    if not isinstance(server_paths, list) and not isinstance(server_paths, str):
        raise TypeError(
            "server_paths and server_urls must be a string or a list of strings"
        )

    _paths = []
    if isinstance(server_paths, str):
        _paths = [server_paths]
    else:
        _paths = server_paths

    valid_paths = []
    for path in _paths:
        if path.startswith("http://") or path.startswith("https://"):
            if check_url_exists(path):
                valid_paths.append(path)
            else:
                warnings.warn(f"Server URL '{path}' is not reachable.")
        else:
            if _check_file_exists(path):
                valid_paths.append(path)
            else:
                warnings.warn(f"Server path '{path}' does not exist.")

    CHARGE_RAISE_ON_MISSING_SERVER = (
        os.getenv("CHARGE_ERROR_ON_MISSING_SERVER", "0") == "1"
    )
    if len(valid_paths) != len(_paths):
        if CHARGE_RAISE_ON_MISSING_SERVER:
            raise ValueError("One or more server paths do not exist.")
    return valid_paths


