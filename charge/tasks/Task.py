from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type
import os.path as osp
import json
import re
import inspect
import os
import warnings


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


class Task(ABC):

    def __init__(
        self,
        system_prompt=None,
        user_prompt=None,
        verification_prompt=None,
        refinement_prompt=None,
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
            **kwargs: Additional keyword arguments to be stored in the task.

        """
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verification_prompt = verification_prompt
        self.refinement_prompt = refinement_prompt
        for key, value in kwargs.items():
            if hasattr(self, key):
                raise ValueError(f"Attribute {key} already exists in Task class.")
            setattr(self, key, value)
        self.constructor_args = {}

    def get_system_prompt(self) -> str:
        return self.system_prompt or ""

    def get_user_prompt(self) -> str:
        return self.user_prompt or ""

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
        return hasattr(self, "structured_output_schema")

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
