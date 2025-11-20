from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type


class Experiment(ABC):

    def __init__(
        self,
        system_prompt=None,
        user_prompt=None,
        verification_prompt=None,
        refinement_prompt=None,
        **kwargs,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verification_prompt = verification_prompt
        self.refinement_prompt = refinement_prompt
        for key, value in kwargs.items():
            if hasattr(self, key):
                raise ValueError(f"Attribute {key} already exists in Experiment class.")
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
