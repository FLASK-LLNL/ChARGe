from abc import ABC, abstractmethod
import os
from pydantic import BaseModel
from typing import Type, Union, Optional
import os.path as osp
import json
import re
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


def check_and_store_server_paths(server_paths: Optional[Union[str, list]]) -> list:
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


class Task(ABC):

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        verification_prompt: Optional[str] = None,
        refinement_prompt: Optional[str] = None,
        server_urls: Optional[Union[str, list]] = None,
        server_files: Optional[Union[str, list]] = None,
        structured_output_schema: Optional[Type[BaseModel]] = None,
        **kwargs,
    ):
        """
        Base class for defining an task, which is composed of a set of steps:
        e.g. prompts and tools. Users should inherit from this class.
        The Task class interfaces with the Client class to run tasks.
        At the very least, users should provide a system prompt and a user prompt.
        The system prompt is a high-level description of the task and provided
        to the reasoning engine at the start of the task. The user prompt
        is the specific task to be accomplished.

        Optionally, users can provide a verification prompt and a refinement prompt.
        When provided and check_response is set to True in the Client, the verification
        prompt along with all methods decorated with @verifier are provided to the
        reasoning engine for self verification. The refinement prompt is used to
        guide the reasoning engine to refine its response if the verification fails.

        **Note**: Automatic verification is an experimental feature and may not work as
        expected.

        The task class can also be extended to include hypothesis methods
        (decorated with @hypothesis) and verifier methods (decorated with @verifier).
        Appropriate MCPs are automatically generated for these methods and used by the
        Client class to call these methods in the HVR process. Prewritten functions
        (with type annotations and docstrings) can also be added to the Task
        via the register_<hypothesis/verifier>_tool functions.

        **Note**: Automatic MCP generation is an experimental feature and may not work as
        expected. All decorated methods must have proper type annotations and be static.
        The docstring of the methods is used as the docstring in the MCP server.
        Long running MCPs with high starting costs should be provided separately to the
        client or the method should call out to an external service / process.


        Args:
            system_prompt (str, optional): The system prompt for the task.
            user_prompt (str, optional): The user prompt for the task.
            verification_prompt (str, optional): The verification prompt for the task.
            refinement_prompt (str, optional): The refinement prompt for the task.
            server_urls (Union[str, list], optional): The MCP server URLs to use with over SSE protocol
                                                       for the task.
            server_files (Union[str, list], optional): The MCP server files to use with over STDIO protocl
                                                       for the task.
            **kwargs: Additional keyword arguments to be stored in the task.

        """
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verification_prompt = verification_prompt
        self.refinement_prompt = refinement_prompt

        self.server_urls = check_and_store_server_paths(server_urls)
        self.server_files = check_and_store_server_paths(server_files)

        self.structured_output_schema = structured_output_schema

        for key, value in kwargs.items():
            if hasattr(self, key):
                raise ValueError(f"Attribute {key} already exists in Task class.")
            setattr(self, key, value)

        self.constructor_args = {}

    def get_system_prompt(self) -> str:
        return self.system_prompt or ""

    def get_user_prompt(self) -> str:
        return self.user_prompt or ""

    def get_verification_prompt(self) -> str:
        return self.verification_prompt or ""

    def get_refinement_prompt(self) -> str:
        return self.refinement_prompt or ""

    def register_buffer(self, name: str, value: str):
        self.constructor_args[name] = value

    def get_structured_output_schema(self):
        assert (
            self.has_structured_output_schema()
        ), "structured_output_schema not implemented"

        return self.structured_output_schema  # type: ignore

    def set_structured_output_schema(self, schema: Type[BaseModel]):
        self.structured_output_schema = schema

    def has_structured_output_schema(self) -> bool:
        return (
            hasattr(self, "structured_output_schema")
            and self.structured_output_schema is not None
        )

    def read_from_file(self, file_path: str, key: str) -> str:
        assert osp.isfile(file_path), f"File {file_path} does not exist"
        if file_path.endswith(".txt"):
            return _prompt_from_txt_file(file_path)
        elif file_path.endswith(".json"):
            return _prompt_from_json_file(file_path, key)
        else:
            raise ValueError("Only .txt and .json files are supported")

    def set_system_prompt_from_file(self, file_path: str):
        """
        Set the system prompt from a file.
        Args:
            file_path (str): Path to the file containing the system prompt.
        Raises:
            ValueError: If the file is not a .txt or .json file.
        """
        assert osp.isfile(file_path), f"File {file_path} does not exist"
        self.system_prompt = self.read_from_file(file_path, "system_prompt")

    def set_user_prompt_from_file(self, file_path: str):
        """
        Set the user prompt from a file.
        Args:
            file_path (str): Path to the file containing the user prompt.
        Raises:
            ValueError: If the file is not a .txt or .json file.

        """
        assert osp.isfile(file_path), f"File {file_path} does not exist"
        self.user_prompt = self.read_from_file(file_path, "user_prompt")

    def set_verification_prompt_from_file(self, file_path: str):
        """
        Set the verification prompt from a file.
        Args:
            file_path (str): Path to the file containing the verification prompt.
        Raises:
            ValueError: If the file is not a .txt or .json file.
        """
        assert osp.isfile(file_path), f"File {file_path} does not exist"
        self.verification_prompt = self.read_from_file(file_path, "verification_prompt")

    def set_refinement_prompt_from_file(self, file_path: str):
        """
        Set the refinement prompt from a file.
        Args:
            file_path (str): Path to the file containing the refinement prompt.
        Raises:
            ValueError: If the file is not a .txt or .json file.
        """
        assert osp.isfile(file_path), f"File {file_path} does not exist"
        self.refinement_prompt = self.read_from_file(file_path, "refinement_prompt")

    def has_verification_prompt(self) -> bool:
        """
        Check if the task has a verification prompt.
        Returns:
            bool: True if the task has a verification prompt, False otherwise.
        """
        return self.verification_prompt is not None

    def has_refinement_prompt(self) -> bool:
        """
        Check if the task has a refinement prompt.
        Returns:
            bool: True if the task has a refinement prompt, False otherwise.
        """
        return self.refinement_prompt is not None

    def check_output_formatting(self, content: str | bytes | bytearray) -> bool:
        """
        Check if the task has output formatting requirements.
        Returns:
            bool: True if the task has output formatting requirements, False otherwise.
        """
        if (
            not self.has_structured_output_schema()
            or self.structured_output_schema is None
        ):
            return True

        structured_output_schema = self.structured_output_schema

        try:
            parsed_output = structured_output_schema.model_validate_json(content)
            return True
        except Exception as e:
            warnings.warn(
                f"Output formatting check failed with error: {e}. Content: {content}"
            )
            return False
