import charge
from charge.Experiment import Experiment
import helper_funcs

SYSTEM_PROMPT = """
You are a world-class medicinal chemist with expertise in drug discovery and molecular design. Your task is to propose novel small molecules that are likely to exhibit high binding affinity to a specified biological target, while also being synthetically accessible. 
You will be provided with a lead molecule as a starting point for your designs.
You can generate new molecules in a SMILES format and optimize
for binding affinity and synthetic accessibility.
"""


class LMOExperiment(Experiment):
    def __init__(
        self,
        user_prompt,
        lead_molecule: str,
        system_prompt=SYSTEM_PROMPT,
        verification_prompt=None,
        refinement_prompt=None,
    ):

        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            verification_prompt=verification_prompt,
            refinement_prompt=refinement_prompt,
        )

        print("LMOExperiment initialized with the provided prompts.")
        self.lead_molecule = lead_molecule
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verification_prompt = verification_prompt
        self.refinement_prompt = refinement_prompt
        self.max_synth_score = helper_funcs.get_synthesizability(lead_molecule)
        self.min_density = helper_funcs.get_density(lead_molecule)

    @charge.hypothesis
    def canonicalize_smiles(self, smiles: str) -> str:
        """
        Canonicalize a SMILES string. Returns the canonical SMILES.
        If the SMILES is invalid, returns "Invalid SMILES".

        Args:
            smiles (str): The input SMILES string.
        Returns:
            str: The canonicalized SMILES string.
        """
        return helper_funcs.canonicalize_smiles(smiles)

    @charge.hypothesis
    def verify_smiles(self, smiles: str) -> bool:
        """
        Verify if a SMILES string is valid. Returns True if valid, False otherwise.

        Args:
            smiles (str): The input SMILES string.
        Returns:
            bool: True if the SMILES is valid, False otherwise.
        """
        return helper_funcs.verify_smiles(smiles)

    @charge.hypothesis
    def get_synthesizability(self, smiles: str) -> float:
        """
        Calculate the synthesizability of a molecule given its SMILES string.
        Values range from 1.0 (highly synthesizable) to 10.0 (not synthesizable).

        Args:
            smiles (str): The input SMILES string.
        Returns:
            float: The synthesizability score.
        """

        return helper_funcs.get_synthesizability(smiles)

    @charge.verifier
    def check_proposal(self, smiles: str) -> bool:
        """
        Check if the proposed SMILES string is valid and meets the criteria.
        The criteria are:
        1. The SMILES must be valid.
        2. The synthesizability score must be less than or equal to 5.0.

        Args:
            smiles (str): The proposed SMILES string.
        Returns:
            bool: True if the proposal is valid and meets the criteria, False otherwise.
        """

        # NOTE: This is used both by the LLM and during verification in the final
        # step of the experiment. So it needs to be deterministic and not
        # rely on any LLM calls.
        is_valid = self.verify_smiles(smiles)
        if not is_valid:
            raise ValueError("Invalid SMILES string.")
        synth_score = self.get_synthesizability(smiles)
