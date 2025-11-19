################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from typing import Type, Dict, Optional
from abc import ABC, abstractmethod
from charge.tasks.Task import Task
from charge._tags import is_verifier, is_hypothesis
from charge.inspector import inspect_class
import inspect
import os
from charge._to_mcp import task_to_mcp
import warnings
import argparse
import atexit
import readline
from charge.agents.Agent import Agent



class Client:
    """Base client class for orchestrating tasks and interacting with MCP servers.

    Subclasses must implement configuration, execution, and interaction methods.
    """
    def __init__(
        self, task: Task, path: str = ".", max_retries: int = 3
    ):
        """Initialize the client with a task instance.

        Args:
            task: The Task object this client will manage.
            path: Directory path for generated files.
            max_retries: Maximum number of retry attempts for server communication.
        """
        self.task = task
        self.path = path
        self.max_retries = max_retries
        self.servers = []
        self.messages = []
        self.reasoning_trace = []
        self._setup()

    def reset(self):
        """Reset internal message and reasoning traces to start a fresh run."""
        self.messages = []
        self.reasoning_trace = []

    def _setup(self):
        """Inspect the task class and collect verifier methods.

        Populates ``self.verifier_methods`` with methods marked as verifiers.
        """
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
        """Generate MCP server files for hypothesis and verifier methods.

        Creates Python files containing MCP representations of task methods.
        """

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
        """Configure the client with model and backend details.

        Returns a tuple of (model, backend, additional_info, config_dict).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def run(self):
        """Execute the full task workflow.

        Subclasses should implement the orchestration logic here.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def step(self, agent: Agent, task: str):
        """Perform a single step of the task using the given agent.

        Args:
            agent: The agent performing the step.
            task: Description of the task step.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def chat(self):
        """Interactively chat with the orchestrator.

        Subclasses should handle chat I/O.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def refine(self, feedback: str):
        """Refine the task based on feedback.

        Args:
            feedback: Feedback string to adjust the task execution.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def add_std_parser_arguments(parser: argparse.ArgumentParser):
        """Utility method to add standard command‑line arguments for the client.

        Populates an ``argparse.ArgumentParser`` with common options.
        """
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
        parser.add_argument(
            "--server-urls", nargs="*", type=str, default=["http://127.0.0.1:8000/sse"]
        )
        parser.add_argument(
            "--history", action="store", type=str, default=".charge-chat-client-history"
        )
