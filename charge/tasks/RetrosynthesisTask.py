from charge.tasks.Task import Task
from charge.servers.log_progress import LOG_PROGRESS_SYSTEM_PROMPT
from typing import List, Optional
from pydantic import BaseModel, field_validator
from charge.servers.SMARTS_reactions_utils import verify_reaction_SMARTS
from charge.servers.SMILES_utils import verify_smiles


class ReactionOutputSchema(BaseModel):
    """
    Structure output representing a valid reaction SMARTS and reactants.
    """

    reasoning_summary: str
    reaction_smarts: str
    reactants_smiles_list: List[str]
    products_smiles_list: List[str]

    @field_validator("reaction_smarts")
    @classmethod
    def validate_reaction_smarts(cls, reaction_smarts):
        if not isinstance(reaction_smarts, str):
            raise ValueError("reaction_smarts must be a string.")
        if len(reaction_smarts) == 0:
            raise ValueError("reaction_smarts cannot be empty.")
        if not verify_reaction_SMARTS(reaction_smarts):
            raise ValueError(f"Invalid reaction SMARTS: {reaction_smarts}")
        return reaction_smarts

    @field_validator("reactants_smiles_list")
    @classmethod
    def validate_reactants(cls, reactants):
        _check_smiles_list(reactants)
        return reactants

    @field_validator("products_smiles_list")
    @classmethod
    def validate_products(cls, products):
        # TODO: Dynamically generate this based on the input molecule
        # to ensure the product is the correct molecule.
        _check_smiles_list(products)
        return products

    def as_dict(self) -> dict:
        return {
            "reasoning_summary": self.reasoning_summary,
            "reaction_smarts": self.reaction_smarts,
            "reactants_smiles_list": self.reactants_smiles_list,
            "products_smiles_list": self.products_smiles_list,
        }


class TemplateFreeReactionOutputSchema(BaseModel):
    """
    Structure output representing a valid list of reactants and products.
    """

    reasoning_summary: str
    reactants_smiles_list: List[str]
    products_smiles_list: List[str]

    @field_validator("reactants_smiles_list")
    @classmethod
    def validate_reactants(cls, reactants):
        _check_smiles_list(reactants)
        return reactants

    @field_validator("products_smiles_list")
    @classmethod
    def validate_products(cls, products):
        _check_smiles_list(products)
        return products

    def as_dict(self) -> dict:
        return {
            "reasoning_summary": self.reasoning_summary,
            "reactants_smiles_list": self.reactants_smiles_list,
            "products_smiles_list": self.products_smiles_list,
        }


TEMPLATE_REACTION_SCHEMA_PROMPT = f"""
Return your answer as a JSON object matching this schema:
{ReactionOutputSchema.model_json_schema()}
"""

TEMPLATE_FREE_REACTION_SCHEMA_PROMPT = f"""
Return your answer as a JSON object matching this schema:
{TemplateFreeReactionOutputSchema.model_json_schema()}
"""

TEMPLATE_SYSTEM_PROMPT = (
    "You are a retrosynthesis expert. Your task is to provide a retrosynthetic "
    + "pathway for the target molecule provided by the user. The pathway should"
    + " be provided as a tuple of reactants as SMILES and Reaction SMARTS."
    + " Put your thinking tokens in <think> </think> tags."
    + " Perform only single step retrosynthesis. Make sure the SMILES strings are"
    + " valid. Make sure the reaction SMARTS is valid. So you will generate a"
    + " reaction SMARTS and reactants for a given molecule. "
    + " For each reaction SMARTS verify it."
    + " If the reaction SMARTS is valid, check if the reactants are valid SMILES."
    + " If they are valid, check if the reaction can be performed"
    + " Use the diagnosis tools to fix any issues that arise."
    + " and return the reaction SMARTS, reactants, and products."
    + " Prefer reactions that are more likely to be performed in a lab "
    + " setting."
    + LOG_PROGRESS_SYSTEM_PROMPT
    + "\n\n"
)


def _check_smiles_list(smiles_list: List[str]) -> None:
    if not isinstance(smiles_list, list):
        raise ValueError("smiles_list must be a list.")
    for smiles in smiles_list:
        if not isinstance(smiles, str):
            raise ValueError("Each SMILES must be a string.")
        if not verify_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")


class RetrosynthesisTask(Task):
    def __init__(
        self,
        user_prompt,
        system_prompt: Optional[str] = None,  # Add optional parameter
    ):
        # Use provided system prompt or fall back to default
        if system_prompt is None:
            system_prompt = TEMPLATE_SYSTEM_PROMPT

        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt + TEMPLATE_REACTION_SCHEMA_PROMPT
        self.set_structured_output_schema(ReactionOutputSchema)
        print(
            "RetrosynthesisTask initialized with the provided prompts:"
            + f"\n{self.system_prompt}"
            + f"\n{self.user_prompt}"
            + f"\n{TEMPLATE_SYSTEM_PROMPT}"
        )


TEMPLATE_FREE_SYSTEM_PROMPT = (
    "You are a retrosynthesis expert. Your task is to provide a retrosynthetic "
    + "pathway for the target molecule provided by the user. The pathway should"
    + " be provided as a tuple of reactants as SMILES and the product as SMILES."
    + " Perform only single step retrosynthesis. Make sure the SMILES strings are"
    + " valid. Use tools to verify the SMILES strings and diagnose any issues that arise."
    + " Use the appropriate return format."
    + LOG_PROGRESS_SYSTEM_PROMPT
    + "\n\n"
)


class TemplateFreeRetrosynthesisTask(Task):
    def __init__(
        self,
        user_prompt,
        system_prompt: Optional[str] = None,  # Add optional parameter
    ):
        # Use provided system prompt or fall back to default
        if system_prompt is None:
            system_prompt = TEMPLATE_FREE_SYSTEM_PROMPT

        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt + TEMPLATE_FREE_REACTION_SCHEMA_PROMPT
        self.set_structured_output_schema(TemplateFreeReactionOutputSchema)
        print(
            "TemplateFreeRetrosynthesisTask initialized with the provided prompts:"
            + f"\n{self.system_prompt}"
            + f"\n{self.user_prompt}"
            + f"\n{TEMPLATE_FREE_SYSTEM_PROMPT}"
        )
