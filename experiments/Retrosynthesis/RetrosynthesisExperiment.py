import charge
from charge.Experiment import Experiment
import helper_funcs


class RetrosynthesisExperiment(Experiment):
    def __init__(
        self,
        user_prompt,
        system_prompt,
    ):
        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        print("RetrosynthesisExperiment initialized with the provided prompts.")
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    @charge.hypothesis
    def verify_reaction_SMARTS(self, smarts: str) -> bool:
        """
        Verify if a reaction SMARTS string is valid. Returns True if valid, False otherwise.

        Args:
            smarts (str): The input reaction SMARTS string.
        Returns:
            bool: True if the reaction SMARTS is valid, False otherwise.
        raises:
            ValueError: If the SMARTS string is invalid.
        """
        is_valid, message = helper_funcs.verify_reaction_SMARTS(smarts)
        if not is_valid:
            raise ValueError(f"Invalid reaction SMARTS: {message}")
        return True
