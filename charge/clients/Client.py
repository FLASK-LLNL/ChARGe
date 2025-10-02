from typing import Type
from abc import ABC, abstractmethod
from charge.Experiment import Experiment
from charge._tags import is_verifier, is_hypothesis
from charge.inspector import inspect_class
import inspect
import os
from charge._to_mcp import experiment_to_mcp


class Client:
    def __init__(
        self, experiment_type: Experiment, path: str = ".", max_retries: int = 3
    ):
        self.experiment_type = experiment_type
        self.path = path
        self.max_retries = max_retries
        self._setup()

    def _setup(self):
        class_info = inspect_class(self.experiment_type)

        methods = inspect.getmembers(self.experiment_type, predicate=inspect.isfunction)

        verifier_methods = []
        for name, method in methods:
            if is_verifier(method):
                verifier_methods.append(method)
        if len(verifier_methods) < 1:
            raise ValueError(
                f"Experiment class {self.experiment_type.__name__} must have at least one verifier method."
            )
        self.verifier_methods = verifier_methods

    def setup_mcp_servers(self):

        class_info = inspect_class(self.experiment_type)

        methods = inspect.getmembers(self.experiment_type, predicate=inspect.isfunction)

        verifier_methods = []
        hypothesis_methods = []
        for name, method in methods:
            if is_verifier(method):
                verifier_methods.append(method)
            if is_hypothesis(method):
                hypothesis_methods.append(method)
        if len(verifier_methods) < 1:
            raise ValueError(
                f"Experiment class {self.experiment_type.__name__} must have at least one verifier method."
            )
        if len(hypothesis_methods) > 1:
            filename = os.path.join(
                self.path, f"{self.experiment_type.__name__}_hypotheses.py"
            )
            with open(filename, "w") as f:
                f.write(experiment_to_mcp(class_info, hypothesis_methods))
            self.hypothesis_server_path = filename

        verifier_filename = os.path.join(
            self.path, f"{self.experiment_type.__name__}_verifiers.py"
        )
        with open(verifier_filename, "w") as f:
            f.write(experiment_to_mcp(class_info, verifier_methods))
        self.verifier_server_path = verifier_filename

    @abstractmethod
    async def run(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def refine(self, feedback: str):
        raise NotImplementedError("Subclasses must implement this method.")
