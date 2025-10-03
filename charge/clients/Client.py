from typing import Type, Dict, Optional
from abc import ABC, abstractmethod
from charge.Experiment import Experiment
from charge._tags import is_verifier, is_hypothesis
from charge.inspector import inspect_class
import inspect
import os
from charge._to_mcp import experiment_to_mcp
import warnings
import argparse


class Client:
    def __init__(
        self, experiment_type: Experiment, path: str = ".", max_retries: int = 3
    ):
        self.experiment_type = experiment_type
        self.path = path
        self.max_retries = max_retries
        self._setup()

    def _setup(self):
        cls_info = inspect_class(self.experiment_type)
        methods = inspect.getmembers(self.experiment_type, predicate=inspect.ismethod)
        name = cls_info["name"]

        verifier_methods = []
        for name, method in methods:
            if is_verifier(method):
                verifier_methods.append(method)
        if len(verifier_methods) < 1:
            warnings.warn(
                f"Experiment class {name} has no verifier methods. "
                + "It's recommended to have at least one verifier method."
                + "Automatic verification will fail without any verifier methods."
            )

        self.verifier_methods = verifier_methods

    def setup_mcp_servers(self):

        class_info = inspect_class(self.experiment_type)
        name = class_info["name"]

        methods = inspect.getmembers(self.experiment_type, predicate=inspect.ismethod)

        verifier_methods = []
        hypothesis_methods = []
        for name, method in methods:
            if is_verifier(method):
                verifier_methods.append(method)
            if is_hypothesis(method):
                hypothesis_methods.append(method)
        if len(verifier_methods) < 1:
            raise ValueError(
                f"Experiment class {name} must have at least one verifier method."
            )
        if len(hypothesis_methods) > 1:
            filename = os.path.join(self.path, f"{name}_hypotheses.py")
            with open(filename, "w") as f:
                f.write(experiment_to_mcp(class_info, hypothesis_methods))
            self.hypothesis_server_path = filename

        # Not used but generated for future
        # verifier_filename = os.path.join(self.path, f"{name}_verifiers.py")
        # with open(verifier_filename, "w") as f:
        #     f.write(experiment_to_mcp(class_info, verifier_methods))
        # self.verifier_server_path = verifier_filename

    @abstractmethod
    def configure(model: str, backend: str) -> (str, str, str, Dict[str, str]):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def run(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def step(self, agent, task: str):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def chat(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def refine(self, feedback: str):
        raise NotImplementedError("Subclasses must implement this method.")

    def add_std_parser_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Model to use for the orchestrator",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="ollama",
            choices=[
                "ollama",
                "openai",
                "gemini",
                "livai",
                "livchat",
            ],
            help="Backend to use for the orchestrator client",
        )
