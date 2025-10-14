import charge
from charge.experiments.Experiment import Experiment
from charge.servers import SMILES_utils
from charge.servers.molecular_property_utils import get_density
import charge.utils.helper_funcs
from typing import Optional, List
from pydantic import BaseModel, field_validator

SYSTEM_PROMPT = """
You are a world-class medicinal chemist with expertise in drug discovery and molecular design. Your task is to propose novel small molecules that are likely to exhibit high binding affinity to a specified biological target, while also being synthetically accessible. 
You will be provided with a lead molecule as a starting point for your designs.
You can generate new molecules in a SMILES format and optimize
for binding affinity and synthetic accessibility.
"""

USER_PROMPT = """
Given the lead molecule: {0}, generate 1 new SMILES strings for molecules similar to the lead molecule.
For each molecule you suggest, verify the SMILES, check if it is already known, and calculate its density and synthetic accessibility. Do the checks in order.
Only return molecules with higher density and the same or lower synthetic accessibility compared to the lead molecule.
If a molecule is known or doesn't fit the criteria, move on and generate a different one and try again.
Return a list of the unique molecules.

"""


class MoleculeOutputSchema(BaseModel):
    """
    Structure output representing a valid list of SMILES strings.
    """

    reasoning_summary: str
    smiles_list: List[str]

    @field_validator("smiles_list")
    @classmethod
    def validate_smiles_list(cls, smiles_list):
        if not isinstance(smiles_list, list):
            raise ValueError("smiles_list must be a list.")
        for smiles in smiles_list:
            if not isinstance(smiles, str):
                raise ValueError("Each SMILES must be a string.")
            if not SMILES_utils.verify_smiles(smiles):
                raise ValueError(f"Invalid SMILES string: {smiles}")
        return smiles_list

    def as_list(self) -> List[str]:
        return self.smiles_list

    def as_dict(self) -> dict:
        return {
            "reasoning_summary": self.reasoning_summary,
            "smiles_list": self.smiles_list,
        }


SCHEMA_PROMPT = f"""
Return your answer as a JSON object matching this schema:
{MoleculeOutputSchema.model_json_schema()}
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
            user_prompt = USER_PROMPT.format(lead_molecule) + SCHEMA_PROMPT
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
        self.max_synth_score = SMILES_utils.get_synthesizability(lead_molecule)
        self.min_density = get_density(lead_molecule)
        self.set_structured_output_schema(MoleculeOutputSchema)

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
        if not SMILES_utils.verify_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")

        synth_score = SMILES_utils.get_synthesizability(smiles)
        if synth_score > self.max_synth_score:
            raise ValueError(
                f"Synthesizability score too high: {synth_score} > {self.max_synth_score}"
            )

        density = get_density(smiles)
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
