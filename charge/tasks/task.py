from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type, Union, Optional
import os
import os.path as osp
import warnings

from charge.utils.system_utils import read_from_file, check_server_paths


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

        self.server_urls = check_server_paths(server_urls)
        self.server_files = check_server_paths(server_files)

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

    def to_json(self) -> dict:
        """
        Serialize the task to a JSON-compatible dictionary.
        Returns:
            dict: JSON-compatible dictionary representation of the task.
        """
        # Get the base attributes
        data = {
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "verification_prompt": self.verification_prompt,
            "refinement_prompt": self.refinement_prompt,
            "server_urls": self.server_urls,
            "server_files": self.server_files,
            "constructor_args": self.constructor_args,
        }

        # Get any additional attributes that were set via kwargs
        # Exclude private attributes, methods, and known base attributes
        base_attrs = {
            "system_prompt",
            "user_prompt",
            "verification_prompt",
            "refinement_prompt",
            "server_urls",
            "server_files",
            "structured_output_schema",
            "constructor_args",
        }

        extra_attrs = {}
        for key, value in self.__dict__.items():
            if (
                not key.startswith("_")
                and key not in base_attrs
                and not callable(value)
            ):
                # Only include JSON-serializable values
                try:
                    import json

                    json.dumps(value)
                    extra_attrs[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable attributes
                    pass

        if extra_attrs:
            data["extra_attrs"] = extra_attrs

        return data

    @classmethod
    def from_json(cls, data: dict) -> "Task":
        """
        Deserialize a task from a JSON-compatible dictionary.
        Args:
            data (dict): JSON-compatible dictionary representation of the task.
        Returns:
            Task: Reconstructed task instance.
        """
        # Extract the base parameters for Task.__init__
        init_params = {
            "system_prompt": data.get("system_prompt"),
            "user_prompt": data.get("user_prompt"),
            "verification_prompt": data.get("verification_prompt"),
            "refinement_prompt": data.get("refinement_prompt"),
            "server_urls": data.get("server_urls"),
            "server_files": data.get("server_files"),
        }

        # Add any extra attributes as kwargs
        extra_attrs = data.get("extra_attrs", {})
        init_params.update(extra_attrs)

        # Create the instance
        instance = cls(**init_params)

        # Restore constructor_args if present
        if "constructor_args" in data:
            instance.constructor_args = data["constructor_args"]

        return instance
