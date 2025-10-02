from abc import ABC, abstractmethod


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
        self.extra_args = kwargs
        self.constructor_args = {}

    def get_system_prompt(self) -> str:
        return self.system_prompt or ""

    def get_user_prompt(self) -> str:
        return self.user_prompt or ""

    def register_buffer(self, name: str, value: str):
        self.constructor_args[name] = value
