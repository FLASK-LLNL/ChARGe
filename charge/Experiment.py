from abc import ABC, abstractmethod
import os.path as osp
import json
import re
import inspect
import os
from charge._to_mcp import experiment_to_mcp
from charge.inspector import inspect_class
from charge._tags import is_verifier, is_hypothesis
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


# def _prompt_from_json_file(file_path: str) -> str:


def _prompt_from_txt_file(file_path: str) -> str:

    with open(file_path, "r") as f:
        prompt = f.read()
    return prompt


class Experiment(ABC):

    def __init__(
        self,
        system_prompt=None,
        user_prompt=None,
        verification_prompt=None,
        refinement_prompt=None,
        **kwargs,
    ):
        """
        Base class for defining an experiment. Users should inherit from this class.
        The Experiment class interfaces with the Client class to run experiments.
        At the very least, users should provide a system prompt and a user prompt.
        The system prompt is a high-level description of the experiment and provided
        to the reasoning engine at the start of the experiment. The user prompt
        is the specific task to be accomplished.

        Optionally, users can provide a verification prompt and a refinement prompt.
        When provided and check_response is set to True in the Client, the verification
        prompt along with all methods decorated with @verifier are provided to the
        reasoning engine for self verification. The refinement prompt is used to
        guide the reasoning engine to refine its response if the verification fails.

        **Note**: Automatic verification is an experimental feature and may not work as
        expected.

        The experiment class can also be extended to include hypothesis methods
        (decorated with @hypothesis) and verifier methods (decorated with @verifier).
        Appropriate MCPs are automatically generated for these methods and used by the
        Client class to call these methods in the HVR process. Prewritten functions
        (with type annotations and docstrings) can also be added to the Experiment
        via the register_<hypothesis/verifier>_tool functions.

        **Note**: Automatic MCP generation is an experimental feature and may not work as
        expected. All decorated methods must have proper type annotations and be static.
        The docstring of the methods is used as the docstring in the MCP server.
        Long running MCPs with high starting costs should be provided separately to the
        client or the method should call out to an external service / process.


        Args:
            system_prompt (str, optional): The system prompt for the experiment.
            user_prompt (str, optional): The user prompt for the experiment.
            verification_prompt (str, optional): The verification prompt for the experiment.
            refinement_prompt (str, optional): The refinement prompt for the experiment.
            **kwargs: Additional keyword arguments to be stored in the experiment.

        """
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verification_prompt = verification_prompt
        self.refinement_prompt = refinement_prompt
        self.extra_args = kwargs
        self.constructor_args = {}

        self.registered_hypothesis_tools = []
        self.registered_verifier_tools = []

    def get_system_prompt(self) -> str:
        return self.system_prompt or ""

    def get_user_prompt(self) -> str:
        return self.user_prompt or ""

    def register_hypothesis_tool(self, func):
        """
        Register a hypothesis tool with the experiment.
        Args:
            func (callable): The hypothesis tool to register.
        """
        setattr(func, "_is_hypothesis", True)
        self.registered_hypothesis_tools.append(func)
        return func

    def register_verifier_tool(self, func):
        """
        Register a verifier tool with the experiment.
        Args:
            func (callable): The verifier tool to register.
        """
        setattr(func, "_is_verifier", True)
        self.registered_verifier_tools.append(func)
        return func

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

    def get_verifier_methods(self):
        cls_info = inspect_class(self)
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        name = cls_info["name"]

        verifier_methods = []
        for name, method in methods:
            if is_verifier(method):
                verifier_methods.append(method)
                # TODO: Currently only support (bool, str) return type for verifiers
                # In future, we may want to support more flexible return types
                if not method.__annotations__.get("return", None) == (bool, str):
                    warnings.warn(
                        f"Verifier method {name} should have return type (bool, str)."
                        + f" Found {method.__annotations__.get('return', None)} instead."
                        + " Automatic verification may fail."
                    )
        if len(verifier_methods) < 1:
            warnings.warn(
                f"Experiment class {name} has no verifier methods. "
                + "It's recommended to have at least one verifier method."
                + "Automatic verification will fail without any verifier methods."
            )
        self.verifier_methods = verifier_methods

    def setup_mcp_servers(self):

        class_info = inspect_class(self)
        name = class_info["name"]

        methods = inspect.getmembers(self, predicate=inspect.ismethod)

        verifier_methods = []
        hypothesis_methods = []
        for name, method in methods:
            if is_verifier(method):
                verifier_methods.append(method)
            if is_hypothesis(method):
                hypothesis_methods.append(method)
        verifier_methods += self.registered_verifier_tools
        hypothesis_methods += self.registered_hypothesis_tools
        if len(verifier_methods) < 1:
            warnings.warn(
                f"Experiment class {name} should have at least one verifier method."
                + " Automatic verification will fail without any verifier methods."
            )
        if len(hypothesis_methods) > 1:
            if not hasattr(self, "path"):
                self.path = os.getcwd()
            filename = os.path.join(self.path, f"{name}_hypotheses.py")
            with open(filename, "w") as f:
                f.write(experiment_to_mcp(class_info, hypothesis_methods))
            self.hypothesis_server_path = filename

        # Not used but generated for future
        # verifier_filename = os.path.join(self.path, f"{name}_verifiers.py")
        # with open(verifier_filename, "w") as f:
        #     f.write(experiment_to_mcp(class_info, verifier_methods))
        # self.verifier_server_path = verifier_filename
