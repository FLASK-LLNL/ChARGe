from abc import ABC, abstractmethod


class Experiment(ABC):

    def __init__(
        self,
        system_prompt,
        user_prompt,
        verification_prompt=None,
        refinement_prompt=None,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verification_prompt = verification_prompt
        self.refinement_prompt = refinement_prompt

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def get_user_prompt(self) -> str:
        return self.user_prompt
