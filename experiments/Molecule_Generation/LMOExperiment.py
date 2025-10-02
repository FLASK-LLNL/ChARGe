import charge
from charge.Experiment import Experiment
import helper_funcs
from typing import Optional

SYSTEM_PROMPT = """
You are a world-class medicinal chemist with expertise in drug discovery and molecular design. Your task is to propose novel small molecules that are likely to exhibit high binding affinity to a specified biological target, while also being synthetically accessible. 
You will be provided with a lead molecule as a starting point for your designs.
You can generate new molecules in a SMILES format and optimize
for binding affinity and synthetic accessibility.
"""

USER_PROMPT = """
Given the lead molecule: {0}, generate 3 new SMILES strings for molecules similar to the lead molecule.
For each molecule you suggest, verify the SMILES, check if it is already known, and
calculate its density and synthetic accessibility. Only return molecules with higher density and
the same or lower synthetic accessibility compared to the lead molecule.
If a molecule is known or doesn't fit the criteria, move on and
generate a different one and try again.
Output a list of the unique molecules.

For example, the output format should be:
["CCO", "CCN", "CCC"]
"""


class LMOExperiment(Experiment):
    def __init__(
        self,
        lead_molecule: str,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        verification_prompt: Optional[str] = None,
        refinement_prompt: Optional[str] = None,
    ):

        if user_prompt is None:
            user_prompt = USER_PROMPT.format(lead_molecule)
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

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

    @charge.hypothesis
    def check_proposal(self, smiles: str) -> bool:
        """
        Check if the proposed SMILES string is valid.
        If it is valid, checks if its synthesizability score is less than or equal to the lead molecule
        and if its density is greater than or equal to the lead molecule.
        Args:
            smiles (str): The proposed SMILES string.
        Returns:
            bool: True if the proposal is valid and meets the criteria, False otherwise.
        Raises:
            ValueError: If the SMILES string is invalid or does not meet the criteria.
        """
        # NOTE: This is used both by the LLM and during verification in the final
        # step of the experiment. So it needs to be deterministic and not
        # rely on any LLM calls.
        if not self.verify_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")

        synth_score = self.get_synthesizability(smiles)
        if synth_score > self.max_synth_score:
            raise ValueError(
                f"Synthesizability score too high: {synth_score} > {self.max_synth_score}"
            )

        density = helper_funcs.get_density(smiles)
        if density < self.min_density:
            raise ValueError(f"Density too low: {density} < {self.min_density}")
        return True

    @charge.verifier
    def check_final_proposal(self, smiles_list_as_string: str) -> bool:
        """
        Check if the proposed SMILES strings are valid and meet the criteria.
        The criteria are:
        1. The SMILES must be valid.
        2. The synthesizability score must be less than or equal to the lead molecule.
        3. The density must be greater than or equal to the lead molecule.

        Args:
            smiles (str): The proposed  list of SMILES strings.
        Returns:
            bool: True if the proposal is valid and meets the criteria, False otherwise.

        Raises:
            ValueError: If the output is not a valid list of SMILES strings or if any
                        SMILES string is invalid or does not meet the criteria.
        """

        # NOTE: This is used both by the LLM and during verification in the final
        # step of the experiment. So it needs to be deterministic and not
        # rely on any LLM calls.
        try:
            smiles_list = eval(smiles_list_as_string)
            if not isinstance(smiles_list, list):
                return False
        except Exception as e:
            raise ValueError("Output is not a valid list of SMILES strings.")

        for smiles in smiles_list:
            self.check_proposal(smiles)
        return True
