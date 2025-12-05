import charge
from charge.tasks.Task import Task
from charge.servers import SMILES_utils
from charge.servers.molecular_property_utils import get_density
import charge.utils.helper_funcs
from typing import Optional, List
from pydantic import BaseModel, field_validator
from charge.servers.log_progress import LOG_PROGRESS_SYSTEM_PROMPT
from charge.utils.mcp_workbench_utils import call_mcp_tool_directly
import asyncio

SYSTEM_PROMPT = (
    "You are a world-class medicinal chemist with expertise in drug"
    + " discovery and molecular design. Your task is to propose novel small"
    + " molecules that are likely to exhibit high binding affinity to a"
    + " specified biological target, while also being synthetically"
    + " accessible.  You will be provided with a lead molecule as a starting"
    + " point for your designs.  You can generate new molecules in a SMILES"
    + " format and optimize for binding affinity and synthetic accessibility."
    + LOG_PROGRESS_SYSTEM_PROMPT
    + "\n\n"
)


USER_PROMPT = (
    "Given the lead molecule: {0}, generate 1 new SMILES strings for"
    + " molecules similar to the lead molecule.  For each molecule you"
    + " suggest, verify the SMILES, check if it is already known, and"
    + " calculate its density and synthetic accessibility. Do the checks in"
    + " order.  Only return molecules with higher density and the same or"
    + " lower synthetic accessibility compared to the lead molecule.  If a"
    + " molecule is known or doesn't fit the criteria, move on and generate a"
    + " different one and try again.  Return a list of the unique molecules."
    + "\n\n"
)


class MoleculeOutputSchema(BaseModel):
    """
    Structure output representing a valid list of SMILES strings.
    """

    reasoning_summary: str
    smiles_list: List[str]
    property_name: str
    property_list: List[float]

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
        return list(zip(self.smiles_list, self.property_list))

    def as_dict(self) -> dict:
        return {
            "reasoning_summary": self.reasoning_summary,
            "smiles_list": self.smiles_list,
            "property_name": self.property_name,
            "property_list": self.property_list,
        }


SCHEMA_PROMPT = f"""
Return your answer as a JSON object matching this schema:
{MoleculeOutputSchema.model_json_schema()}
"""


class LMOTask(Task):
    def __init__(
        self,
        lead_molecule: str,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        verification_prompt: Optional[str] = None,
        refinement_prompt: Optional[str] = None,
        property_tool_name: Optional[str] = None,
        property_name: str = "density",
        optimize_direction: str = "greater",

        **kwargs,
    ):
        """
        Initialize LMOTask with customizable property optimization.

        Args:
            lead_molecule: SMILES string of the lead molecule
            user_prompt: Custom user prompt (optional)
            system_prompt: Custom system prompt (optional)
            verification_prompt: Custom verification prompt (optional)
            refinement_prompt: Custom refinement prompt (optional)
            property_tool_name: Name of MCP tool that takes SMILES string and returns float property value.
                                Defaults to get_density if None.
            property_name: Name of the property being optimized (for logging)
            optimize_direction: "greater" to maximize property, "less" to minimize it
            **kwargs: Additional arguments passed to Task
        """

        if user_prompt is None:
            user_prompt = USER_PROMPT.format(lead_molecule)
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        # call mcp server directly
        # Default to density if no property function provided
        if property_tool_name is None:
            property_tool_name = "get_density"

        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            verification_prompt=verification_prompt,
            refinement_prompt=refinement_prompt,
            **kwargs,
        )

        print("LMOTask initialized with the provided prompts.")
        self.lead_molecule = lead_molecule
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.verification_prompt = verification_prompt
        self.refinement_prompt = refinement_prompt
        self.max_synth_score = SMILES_utils.get_synthesizability(lead_molecule)
        # Change this to be the min property value - add a function to get the right value
        # add a property name as well
        # Store property function and related attributes
        self.property_tool_name = property_tool_name
        self.property_name = property_name
        self.optimize_direction = optimize_direction
        self.set_structured_output_schema(MoleculeOutputSchema)

    async def get_initial_property_value(self) -> bool:
        # Calculate the reference property value from the lead molecule
        property_result_msg = await call_mcp_tool_directly(
            tool_name = self.property_tool_name,
            arguments = {
                "smiles": self.lead_molecule,
                "property": self.property_name,
            },
            urls = self.server_urls or [],
            paths = self.server_files or [],
        )
        results = property_result_msg.result
        if len(results) > 1:
            property_result = property_result_msg.result[1].content
        else:
            property_result = 0.0
            raise ValueError(f"{property_result_msg.result[0].content}")

        self.reference_property_value = property_result

    async def check_proposal(self, smiles: str) -> bool:
        """
        Check if the proposed SMILES string is valid.
        If it is valid, checks if its synthesizability score is less than or equal to the lead molecule
        and if its property value meets the optimization criteria.

        Args:
            smiles (str): The proposed SMILES string.
        Returns:
            bool: True if the proposal is valid and meets the criteria, False otherwise.
        Raises:
            ValueError: If the SMILES string is invalid or does not meet the criteria.
        """
        # NOTE: This is used both by the LLM and during verification in the final
        # step of the task. So it needs to be deterministic and not
        # rely on any LLM calls.
        if not SMILES_utils.verify_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")

        synth_score = SMILES_utils.get_synthesizability(smiles)
        if synth_score > self.max_synth_score:
            raise ValueError(
                f"Synthesizability score too high: {synth_score} > {self.max_synth_score}"
            )

        property_value = await call_mcp_tool_directly(
            tool_name = self.property_tool_name,
            arguments = {
                "smiles": smiles,
                "property": self.property_name,
            },
            urls = self.server_urls,
            paths = self.server_files,
        )

        # Check if property meets optimization criteria
        if self.optimize_direction == "greater":
            if property_value < self.reference_property_value:
                raise ValueError(
                    f"{self.property_name} too low: {property_value} < {self.reference_property_value}"
                )
        else:  # "less"
            if property_value > self.reference_property_value:
                raise ValueError(
                    f"{self.property_name} too high: {property_value} > {self.reference_property_value}"
                )
        return True

    @charge.verifier
    async def check_final_proposal(self, smiles_list_as_string: str) -> bool:
        """
        Check if the proposed SMILES strings are valid and meet the criteria.
        The criteria are:
        1. The SMILES must be valid.
        2. The synthesizability score must be less than or equal to the lead molecule.
        3. The property value must meet the optimization criteria (greater or less than reference).

        Args:
            smiles_list_as_string (str): The proposed list of SMILES strings.
        Returns:
            bool: True if the proposal is valid and meets the criteria, False otherwise.

        Raises:
            ValueError: If the output is not a valid list of SMILES strings or if any
                        SMILES string is invalid or does not meet the criteria.
        """

        # NOTE: This is used both by the LLM and during verification in the final
        # step of the task. So it needs to be deterministic and not
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
