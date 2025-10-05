from typing import Tuple, Type, Dict, Optional
from abc import ABC, abstractmethod
from charge.Experiment import Experiment

import argparse


class Client:
    def __init__(
        self,
        experiment_type: Experiment,
        path: str = ".",
        max_retries: int = 3,
        check_response: bool = False,
    ):
        self.experiment_type = experiment_type
        self.path = path
        self.max_retries = max_retries
        self.servers = []
        self.messages = []
        self.check_response = check_response
        self._setup()

    def _setup(self):
        self.verifier_methods = self.experiment_type.get_verifier_methods()

    @staticmethod
    @abstractmethod
    def configure(
        model: str, backend: str
    ) -> Tuple[str, str, Optional[str], Dict[str, str]]:
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
        parser.add_argument(
            "--server-urls", nargs="*", type=str, default="http://127.0.0.1:8000/sse"
        )
