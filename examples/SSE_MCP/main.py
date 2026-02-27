import argparse
import asyncio
from charge.tasks.task import Task
from charge.clients.client import Client
from charge.clients.autogen import AutoGenBackend
from typing import Optional, Union

parser = argparse.ArgumentParser()
parser.add_argument("--lead-molecule", type=str, default="CC(=O)O[C@H](C)CCN")

parser.add_argument(
    "--system-prompt",
    type=str,
    default=None,
    help="Custom system prompt (optional, uses default chemistry prompt if not provided)",
)

parser.add_argument(
    "--user-prompt",
    type=str,
    default=None,
    help="Custom user prompt template (optional, uses default molecule generation prompt if not provided). Use {lead_molecule} as placeholder.",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

# Default prompts
DEFAULT_SYSTEM_PROMPT = (
    "You are a world-class chemist. Your task is to generate unique molecules "
    "based on the lead molecule provided by the user. The generated molecules "
    "should be chemically valid and diverse, exploring different chemical spaces "
    "while maintaining some structural similarity to the lead molecule. "
    "Provide the final answer in a clear and concise manner."
)

# Note that this is a template and `lead_molecule` is replaced in task
DEFAULT_USER_PROMPT_TEMPLATE = (
    "Generate 3 unique molecules based on the lead molecule {lead_molecule}. "
    "Use tools to verify the molecule is valid. "
    "Return only the SMILES strings in a Python list format."
)


class UniqueMoleculeTask(Task):
    def __init__(
        self,
        lead_molecule: str,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        server_urls: Optional[Union[str, list]] = None,
    ):
        # Use provided prompts or fall back to defaults
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        if user_prompt_template is None:
            user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE

        # Format user prompt with lead molecule
        user_prompt = user_prompt_template.format(lead_molecule=lead_molecule)

        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            server_urls=server_urls,
        )
        print("UniqueMoleculeTask initialized with the provided prompts.")
        self.lead_molecule = lead_molecule
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt


if __name__ == "__main__":

    args = parser.parse_args()
    lead_molecule = args.lead_molecule
    server_urls = args.server_urls
    mytask = UniqueMoleculeTask(
        lead_molecule=lead_molecule,
        system_prompt=args.system_prompt,
        user_prompt_template=args.user_prompt,
        server_urls=server_urls,
    )

    agent_backend = AutoGenBackend(model=args.model, backend=args.backend)
    runner = agent_backend.create_agent(task=mytask)

    results = asyncio.run(runner.run())

    print(f"[{args.model} orchestrated] Task completed. Results: {results}")
