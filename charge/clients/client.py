from typing import Type, Dict, Optional
from abc import ABC, abstractmethod
from charge.tasks.task import Task
from charge._tags import is_verifier, is_hypothesis
from charge.inspector import inspect_class
import inspect
import os
from charge._to_mcp import task_to_mcp
import warnings
import argparse
import atexit
import readline


class Client:
    def __init__(self, task: Task, path: str = ".", max_retries: int = 3):
        self.task = task
        self.path = path
        self.max_retries = max_retries
        self.servers = []
        self.messages = []
        self.reasoning_trace = []
        self._setup()

    def reset(self):
        self.messages = []
        self.reasoning_trace = []

    def _setup(self):
        cls_info = inspect_class(self.task)
        methods = inspect.getmembers(self.task, predicate=inspect.ismethod)
        name = cls_info["name"]

        verifier_methods = []
        for name, method in methods:
            if is_verifier(method):
                verifier_methods.append(method)
        if len(verifier_methods) < 1:
            warnings.warn(
                f"Task class {name} has no verifier methods. "
                + "It's recommended to have at least one verifier method."
                + "Automatic verification will fail without any verifier methods."
            )

        self.verifier_methods = verifier_methods

    def setup_mcp_servers(self):

        class_info = inspect_class(self.task)
        name = class_info["name"]

        methods = inspect.getmembers(self.task, predicate=inspect.ismethod)

        verifier_methods = []
        hypothesis_methods = []
        for name, method in methods:
            if is_verifier(method):
                verifier_methods.append(method)
            if is_hypothesis(method):
                hypothesis_methods.append(method)
        if len(verifier_methods) < 1:
            raise ValueError(
                f"Task class {name} must have at least one verifier method."
            )
        if len(hypothesis_methods) > 1:
            filename = os.path.join(self.path, f"{name}_hypotheses.py")
            with open(filename, "w") as f:
                f.write(task_to_mcp(class_info, hypothesis_methods))
            self.hypothesis_server_path = filename

        # Not used but generated for future
        # verifier_filename = os.path.join(self.path, f"{name}_verifiers.py")
        # with open(verifier_filename, "w") as f:
        #     f.write(task_to_mcp(class_info, verifier_methods))
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

    @staticmethod
    def add_std_parser_arguments(
        parser: argparse.ArgumentParser, defaults: Optional[Dict[str, str]] = None
    ):
        defaults = defaults or {}
        parser.add_argument(
            "--model",
            type=str,
            default=defaults.get("model", None),
            help="Model to use for the orchestrator",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default=defaults.get("backend", "ollama"),
            choices=[
                "ollama",
                "openai",
                "gemini",
                "livai",
                "livchat",
                "huggingface",
                "vllm",
            ],
            help="Backend to use for the orchestrator client",
        )
        parser.add_argument(
            "--server-urls", nargs="*", type=str, default=["http://127.0.0.1:8000/sse"]
        )
        parser.add_argument(
            "--history", action="store", type=str, default=".charge-chat-client-history"
        )
