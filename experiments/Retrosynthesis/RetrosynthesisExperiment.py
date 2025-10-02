from charge.Experiment import Experiment


SYSTEM_PROMPT = (
    "You are a retrosynthesis expert. Your task is to provide a retrosynthetic "
    + "pathway for the target molecule provided by the user. The pathway should"
    + " be provided as a tuple of reactants as SMILES and Reaction SMARTS."
    + " Perform only single step retrosynthesis. Make sure the SMILES strings are"
    + " valid. Make sure the reaciont SMARTS is valid. So you will generate a "
    + "reaction SMARTS and reactants for a given molecule. "
    + " For each reaction SMARTS verify it."
    + " If the reaction SMARTS is valid, check if the reactants are valid SMILES."
    + " If they are valid, check if the reaction can be performed"
    + " Use the diagnoise tools to fix any issues that arise."
    + " and return the reaction SMARTS, reactants, and products."
    + " Prefer reactions that are more likely to be performed in a lab "
    + " setting. \n\n"
)


class RetrosynthesisExperiment(Experiment):
    def __init__(
        self,
        user_prompt,
    ):
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        print("RetrosynthesisExperiment initialized with the provided prompts.")
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = user_prompt
